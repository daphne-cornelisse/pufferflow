// PufferDrive 3.0 Drive encoder, as CUDA kernels for the 5.0 trainer.
// Matches pufferlib/ocean/torch.py Drive.encode_observations on branch 3.0:
//   ego / partner / road: Linear -> LayerNorm -> Linear
//   partner and road: max over objects (first index wins ties)
//   concat(ego, road, partner) -> GELU -> Linear -> ReLU
// Road's last feature is a class index, one-hot with 7 classes (3.0 num_classes=7).
// Entity width is the 3.0 training input_size (64). Output width is policy hidden_size.
// Included by src/ocean.cu under PUFFER_DRIVE.

static constexpr int DRIVE_ENC_D = 64;
static constexpr int DRIVE_ROAD_CLASSES = 7;
static constexpr int DRIVE_PARTNERS = MAX_AGENTS - 1;
static constexpr int DRIVE_ROADS = MAX_ROAD_SEGMENT_OBSERVATIONS;
static constexpr int DRIVE_ROAD_CONT = ROAD_FEATURES - 1;
static constexpr int DRIVE_ROAD_IN = DRIVE_ROAD_CONT + DRIVE_ROAD_CLASSES;
static constexpr float DRIVE_LN_EPS = 1e-5f;

static_assert(DRIVE_ENC_D % 4 == 0, "encoder width must keep muon params 16-byte aligned");
static_assert(DRIVE_PARTNERS <= DRIVE_ROADS, "scratch is sized to the road set");
static_assert(DRIVE_ROAD_IN * DRIVE_ROADS >= 3 * DRIVE_ENC_D, "feat prefix holds the GELU vector");
static_assert(EGO_FEATURES + PARTNER_FEATURES * DRIVE_PARTNERS + ROAD_FEATURES * DRIVE_ROADS == OBS_SIZE,
    "drive encoder obs layout");

struct DriveEncoderWeights {
    Prec ego_w1, ego_b1, ego_ln_g, ego_ln_b, ego_w2, ego_b2;
    Prec partner_w1, partner_b1, partner_ln_g, partner_ln_b, partner_w2, partner_b2;
    Prec road_w1, road_b1, road_ln_g, road_ln_b, road_w2, road_b2;
    Prec shared_w, shared_b;
    int obs_size, hidden;
};

struct DriveEncoderActivations {
    Prec saved_obs;
    Prec feat, scratch_a, scratch_b;
    Prec ego_h1, partner_h1, road_h1;
    Prec ego_mean, ego_rstd, partner_mean, partner_rstd, road_mean, road_rstd;
    Prec ego_pool, partner_pool, road_pool;
    Int partner_argmax, road_argmax;
    Prec cat, pre_relu, out;
    Prec ego_w1g, ego_b1g, ego_lng, ego_lnb, ego_w2g, ego_b2g;
    Prec partner_w1g, partner_b1g, partner_lng, partner_lnb, partner_w2g, partner_b2g;
    Prec road_w1g, road_b1g, road_lng, road_lnb, road_w2g, road_b2g;
    Prec shared_wg, shared_bg;
    int cap_b, obs_size, hidden;
};

static Prec drive_view(precision_t* data, int64_t rows, int64_t cols) {
    Prec p = {};
    p.data = data;
    p.shape[0] = rows;
    p.shape[1] = cols;
    return p;
}

static void drive_reg_mat(Allocator* alloc, Prec* t, int64_t rows, int64_t cols) {
    *t = {};
    t->shape[0] = rows;
    t->shape[1] = cols;
    alloc_register(alloc, t);
}

static void drive_reg_vec(Allocator* alloc, Prec* t, int64_t n) {
    *t = {};
    t->shape[0] = n;
    alloc_register(alloc, t);
}

__global__ void drive_fill_kernel(precision_t* dst, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(value);
    }
}

__global__ void drive_add_bias_kernel(precision_t* y, const precision_t* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx >= n) {
        return;
    }
    y[idx] = from_float(to_float(y[idx]) + to_float(bias[idx % cols]));
}

__global__ void drive_sum_cols_kernel(precision_t* out, const precision_t* mat, int rows, int cols) {
    int c = blockIdx.x;
    if (c >= cols) {
        return;
    }
    float sum = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        sum += to_float(mat[(int64_t)r * cols + c]);
    }
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[c] = from_float(smem[0]);
    }
}

// One thread per row. D is 64, so the row fits in a few dozen flops.
__global__ void drive_ln_fwd_kernel(
        precision_t* y, precision_t* mean, precision_t* rstd,
        const precision_t* x, const precision_t* gamma, const precision_t* beta,
        int rows, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const precision_t* xr = x + (int64_t)row * d;
    float m = 0.0f;
    for (int i = 0; i < d; i++) {
        m += to_float(xr[i]);
    }
    m /= (float)d;
    float var = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = to_float(xr[i]) - m;
        var += diff * diff;
    }
    var /= (float)d;
    float rs = rsqrtf(var + DRIVE_LN_EPS);
    mean[row] = from_float(m);
    rstd[row] = from_float(rs);
    precision_t* yr = y + (int64_t)row * d;
    for (int i = 0; i < d; i++) {
        float xhat = (to_float(xr[i]) - m) * rs;
        yr[i] = from_float(xhat * to_float(gamma[i]) + to_float(beta[i]));
    }
}

// Writes dx into grad_h1 and dgamma contributions over h1. grad_h2 stays put (dbeta).
__global__ void drive_ln_bwd_kernel(
        precision_t* grad_h1, precision_t* h1, const precision_t* grad_h2,
        const precision_t* mean, const precision_t* rstd, const precision_t* gamma,
        int rows, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float m = to_float(mean[row]);
    float rs = to_float(rstd[row]);
    precision_t* x = h1 + (int64_t)row * d;
    const precision_t* gh = grad_h2 + (int64_t)row * d;
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (int i = 0; i < d; i++) {
        float xhat = (to_float(x[i]) - m) * rs;
        float dy = to_float(gh[i]) * to_float(gamma[i]);
        sum_dy += dy;
        sum_dy_xhat += dy * xhat;
    }
    sum_dy /= (float)d;
    sum_dy_xhat /= (float)d;
    precision_t* dx = grad_h1 + (int64_t)row * d;
    for (int i = 0; i < d; i++) {
        float xhat = (to_float(x[i]) - m) * rs;
        float dy = to_float(gh[i]) * to_float(gamma[i]);
        dx[i] = from_float(rs * (dy - sum_dy - xhat * sum_dy_xhat));
        x[i] = from_float(to_float(gh[i]) * xhat);
    }
}

__global__ void drive_gather_ego_kernel(precision_t* dst, const precision_t* obs, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * EGO_FEATURES;
    if (idx >= n) {
        return;
    }
    int f = idx % EGO_FEATURES;
    int b = idx / EGO_FEATURES;
    dst[idx] = obs[(int64_t)b * obs_size + f];
}

__global__ void drive_gather_partner_kernel(precision_t* dst, const precision_t* obs, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * DRIVE_PARTNERS * PARTNER_FEATURES;
    if (idx >= n) {
        return;
    }
    int f = idx % PARTNER_FEATURES;
    int o = (idx / PARTNER_FEATURES) % DRIVE_PARTNERS;
    int b = idx / (DRIVE_PARTNERS * PARTNER_FEATURES);
    int src = EGO_FEATURES + o * PARTNER_FEATURES + f;
    dst[idx] = obs[(int64_t)b * obs_size + src];
}

__global__ void drive_gather_road_kernel(precision_t* dst, const precision_t* obs, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * DRIVE_ROADS * DRIVE_ROAD_IN;
    if (idx >= n) {
        return;
    }
    int f = idx % DRIVE_ROAD_IN;
    int o = (idx / DRIVE_ROAD_IN) % DRIVE_ROADS;
    int b = idx / (DRIVE_ROADS * DRIVE_ROAD_IN);
    int base = EGO_FEATURES + DRIVE_PARTNERS * PARTNER_FEATURES + o * ROAD_FEATURES;
    const precision_t* src = obs + (int64_t)b * obs_size + base;
    float outv;
    if (f < DRIVE_ROAD_CONT) {
        outv = to_float(src[f]);
    } else {
        int cls = (int)lrintf(to_float(src[ROAD_FEATURES - 1]));
        if (cls < 0) {
            cls = 0;
        }
        if (cls >= DRIVE_ROAD_CLASSES) {
            cls = DRIVE_ROAD_CLASSES - 1;
        }
        outv = (f - DRIVE_ROAD_CONT) == cls ? 1.0f : 0.0f;
    }
    dst[idx] = from_float(outv);
}

__global__ void drive_max_kernel(
        precision_t* pooled, int* argmax, const precision_t* h, int B, int nobj, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * d;
    if (idx >= n) {
        return;
    }
    int c = idx % d;
    int b = idx / d;
    const precision_t* col = h + ((int64_t)b * nobj) * d + c;
    float best = to_float(col[0]);
    int best_i = 0;
    for (int o = 1; o < nobj; o++) {
        float v = to_float(col[(int64_t)o * d]);
        if (v > best) {
            best = v;
            best_i = o;
        }
    }
    pooled[idx] = from_float(best);
    argmax[idx] = best_i;
}

__global__ void drive_unpool_kernel(
        precision_t* grad_h, const precision_t* grad_pool, const int* argmax,
        int B, int nobj, int d, int stride, int off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * nobj * d;
    if (idx >= total) {
        return;
    }
    int c = idx % d;
    int o = (idx / d) % nobj;
    int b = idx / (nobj * d);
    float g = 0.0f;
    if (argmax[(int64_t)b * d + c] == o) {
        g = to_float(grad_pool[(int64_t)b * stride + off + c]);
    }
    grad_h[idx] = from_float(g);
}

__global__ void drive_pack_kernel(
        precision_t* dst, const precision_t* src, int B, int d, int stride, int off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * d) {
        return;
    }
    int c = idx % d;
    int b = idx / d;
    dst[idx] = src[(int64_t)b * stride + off + c];
}

__global__ void drive_concat_kernel(
        precision_t* cat, const precision_t* ego, const precision_t* road,
        const precision_t* partner, int B, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int width = 3 * d;
    if (idx >= B * width) {
        return;
    }
    int f = idx % width;
    int b = idx / width;
    const precision_t* src = ego;
    int sf = f;
    if (f >= 2 * d) {
        src = partner;
        sf = f - 2 * d;
    } else if (f >= d) {
        src = road;
        sf = f - d;
    }
    cat[idx] = src[(int64_t)b * d + sf];
}

__global__ void drive_gelu_kernel(precision_t* dst, const precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float x = to_float(src[idx]);
    // erf GELU, matching nn.GELU() in the 3.0 training module.
    float y = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    dst[idx] = from_float(y);
}

__global__ void drive_gelu_bwd_kernel(precision_t* cat, const precision_t* dgelu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float x = to_float(cat[idx]);
    float cdf = 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    float pdf = expf(-0.5f * x * x) * 0.3989422804014327f;
    cat[idx] = from_float(to_float(dgelu[idx]) * (cdf + x * pdf));
}

__global__ void drive_relu_kernel(precision_t* dst, const precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float x = to_float(src[idx]);
    dst[idx] = from_float(x > 0.0f ? x : 0.0f);
}

__global__ void drive_relu_bwd_kernel(precision_t* pre, const precision_t* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float p = to_float(pre[idx]);
    pre[idx] = from_float(p > 0.0f ? to_float(grad[idx]) : 0.0f);
}

static void drive_fill(precision_t* dst, float value, int n, cudaStream_t stream) {
    drive_fill_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, value, n);
}

static void drive_init_linear(Prec* w, Prec* b, ulong* seed, cudaStream_t stream) {
    puf_kaiming_init(w, sqrtf(2.0f), (*seed)++, stream);
    drive_fill(b->data, 0.0f, (int)numel(b->shape), stream);
}

static void drive_init_ln(Prec* gamma, Prec* beta, cudaStream_t stream) {
    int n = (int)numel(gamma->shape);
    drive_fill(gamma->data, 1.0f, n, stream);
    drive_fill(beta->data, 0.0f, n, stream);
}

static void drive_tower_forward(
        precision_t* x, int rows, int in_dim,
        Prec* w1, Prec* b1, Prec* ln_g, Prec* ln_b, Prec* w2, Prec* b2,
        precision_t* h1, precision_t* mean, precision_t* rstd,
        precision_t* scratch_a, precision_t* scratch_b, cudaStream_t stream) {
    Prec in = drive_view(x, rows, in_dim);
    Prec h1v = drive_view(h1, rows, DRIVE_ENC_D);
    puf_mm(&in, w1, &h1v, stream);
    int nd = rows * DRIVE_ENC_D;
    drive_add_bias_kernel<<<grid_size(nd), BLOCK_SIZE, 0, stream>>>(
        h1, b1->data, rows, DRIVE_ENC_D);
    drive_ln_fwd_kernel<<<grid_size(rows), BLOCK_SIZE, 0, stream>>>(
        scratch_a, mean, rstd, h1, ln_g->data, ln_b->data, rows, DRIVE_ENC_D);
    Prec h2 = drive_view(scratch_a, rows, DRIVE_ENC_D);
    Prec h3 = drive_view(scratch_b, rows, DRIVE_ENC_D);
    puf_mm(&h2, w2, &h3, stream);
    drive_add_bias_kernel<<<grid_size(nd), BLOCK_SIZE, 0, stream>>>(
        scratch_b, b2->data, rows, DRIVE_ENC_D);
}

static void drive_tower_backward(
        precision_t* grad_h3, precision_t* h1, precision_t* mean, precision_t* rstd,
        precision_t* x, int rows, int in_dim,
        Prec* w1, Prec* b1, Prec* ln_g, Prec* ln_b, Prec* w2, Prec* b2,
        Prec* w1g, Prec* b1g, Prec* lng, Prec* lnb, Prec* w2g, Prec* b2g,
        precision_t* scratch_a, cudaStream_t stream) {
    drive_ln_fwd_kernel<<<grid_size(rows), BLOCK_SIZE, 0, stream>>>(
        scratch_a, mean, rstd, h1, ln_g->data, ln_b->data, rows, DRIVE_ENC_D);
    Prec h2 = drive_view(scratch_a, rows, DRIVE_ENC_D);
    Prec g3 = drive_view(grad_h3, rows, DRIVE_ENC_D);
    puf_mm_tn(&g3, &h2, w2g, stream);
    drive_sum_cols_kernel<<<DRIVE_ENC_D, BLOCK_SIZE, 0, stream>>>(
        b2g->data, grad_h3, rows, DRIVE_ENC_D);
    // h2's buffer becomes grad of the layernorm output. g3 stays until this GEMM is queued.
    puf_mm_nn(&g3, w2, &h2, stream);
    drive_ln_bwd_kernel<<<grid_size(rows), BLOCK_SIZE, 0, stream>>>(
        grad_h3, h1, scratch_a, mean, rstd, ln_g->data, rows, DRIVE_ENC_D);
    drive_sum_cols_kernel<<<DRIVE_ENC_D, BLOCK_SIZE, 0, stream>>>(
        lng->data, h1, rows, DRIVE_ENC_D);
    drive_sum_cols_kernel<<<DRIVE_ENC_D, BLOCK_SIZE, 0, stream>>>(
        lnb->data, scratch_a, rows, DRIVE_ENC_D);
    Prec g1 = drive_view(grad_h3, rows, DRIVE_ENC_D);
    Prec xv = drive_view(x, rows, in_dim);
    puf_mm_tn(&g1, &xv, w1g, stream);
    drive_sum_cols_kernel<<<DRIVE_ENC_D, BLOCK_SIZE, 0, stream>>>(
        b1g->data, grad_h3, rows, DRIVE_ENC_D);
}

static long drive_encoder_param_count(int hidden) {
    long d = DRIVE_ENC_D;
    long ego = d * EGO_FEATURES + d + d + d + d * d + d;
    long partner = d * PARTNER_FEATURES + d + d + d + d * d + d;
    long road = d * DRIVE_ROAD_IN + d + d + d + d * d + d;
    long shared = (long)hidden * (3 * d) + hidden;
    return ego + partner + road + shared;
}

static Prec drive_encoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    DriveEncoderWeights* ew = (DriveEncoderWeights*)w;
    DriveEncoderActivations* a = (DriveEncoderActivations*)activations;
    int B = (int)input.shape[0];
    assert(B > 0 && B <= a->cap_b);
    assert(input.shape[1] == a->obs_size);

    if (a->saved_obs.data) {
        Prec saved = drive_view(a->saved_obs.data, B, a->obs_size);
        puf_copy(&saved, &input, stream);
    }

    int ego_rows = B;
    int partner_rows = B * DRIVE_PARTNERS;
    int road_rows = B * DRIVE_ROADS;

    drive_gather_ego_kernel<<<grid_size(ego_rows * EGO_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->feat.data, input.data, B, a->obs_size);
    drive_tower_forward(
        a->feat.data, ego_rows, EGO_FEATURES,
        &ew->ego_w1, &ew->ego_b1, &ew->ego_ln_g, &ew->ego_ln_b, &ew->ego_w2, &ew->ego_b2,
        a->ego_h1.data, a->ego_mean.data, a->ego_rstd.data,
        a->scratch_a.data, a->scratch_b.data, stream);
    drive_pack_kernel<<<grid_size(B * DRIVE_ENC_D), BLOCK_SIZE, 0, stream>>>(
        a->ego_pool.data, a->scratch_b.data, B, DRIVE_ENC_D, DRIVE_ENC_D, 0);

    drive_gather_partner_kernel<<<grid_size(partner_rows * PARTNER_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->feat.data, input.data, B, a->obs_size);
    drive_tower_forward(
        a->feat.data, partner_rows, PARTNER_FEATURES,
        &ew->partner_w1, &ew->partner_b1, &ew->partner_ln_g, &ew->partner_ln_b,
        &ew->partner_w2, &ew->partner_b2,
        a->partner_h1.data, a->partner_mean.data, a->partner_rstd.data,
        a->scratch_a.data, a->scratch_b.data, stream);
    drive_max_kernel<<<grid_size(B * DRIVE_ENC_D), BLOCK_SIZE, 0, stream>>>(
        a->partner_pool.data, a->partner_argmax.data, a->scratch_b.data,
        B, DRIVE_PARTNERS, DRIVE_ENC_D);

    drive_gather_road_kernel<<<grid_size(road_rows * DRIVE_ROAD_IN), BLOCK_SIZE, 0, stream>>>(
        a->feat.data, input.data, B, a->obs_size);
    drive_tower_forward(
        a->feat.data, road_rows, DRIVE_ROAD_IN,
        &ew->road_w1, &ew->road_b1, &ew->road_ln_g, &ew->road_ln_b, &ew->road_w2, &ew->road_b2,
        a->road_h1.data, a->road_mean.data, a->road_rstd.data,
        a->scratch_a.data, a->scratch_b.data, stream);
    drive_max_kernel<<<grid_size(B * DRIVE_ENC_D), BLOCK_SIZE, 0, stream>>>(
        a->road_pool.data, a->road_argmax.data, a->scratch_b.data,
        B, DRIVE_ROADS, DRIVE_ENC_D);

    int cat_n = B * 3 * DRIVE_ENC_D;
    drive_concat_kernel<<<grid_size(cat_n), BLOCK_SIZE, 0, stream>>>(
        a->cat.data, a->ego_pool.data, a->road_pool.data, a->partner_pool.data,
        B, DRIVE_ENC_D);
    // feat is larger than the concat, so the GELU output fits in its prefix.
    drive_gelu_kernel<<<grid_size(cat_n), BLOCK_SIZE, 0, stream>>>(
        a->feat.data, a->cat.data, cat_n);
    Prec gelu = drive_view(a->feat.data, B, 3 * DRIVE_ENC_D);
    Prec pre = drive_view(a->pre_relu.data, B, ew->hidden);
    puf_mm(&gelu, &ew->shared_w, &pre, stream);
    int hid_n = B * ew->hidden;
    drive_add_bias_kernel<<<grid_size(hid_n), BLOCK_SIZE, 0, stream>>>(
        a->pre_relu.data, ew->shared_b.data, B, ew->hidden);
    drive_relu_kernel<<<grid_size(hid_n), BLOCK_SIZE, 0, stream>>>(
        a->out.data, a->pre_relu.data, hid_n);
    return drive_view(a->out.data, B, ew->hidden);
}

static void drive_encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    DriveEncoderWeights* ew = (DriveEncoderWeights*)w;
    DriveEncoderActivations* a = (DriveEncoderActivations*)activations;
    int B = (int)grad.shape[0];
    assert(a->saved_obs.data && B > 0 && B <= a->cap_b);
    assert(grad.shape[1] == ew->hidden);

    int hid_n = B * ew->hidden;
    drive_relu_bwd_kernel<<<grid_size(hid_n), BLOCK_SIZE, 0, stream>>>(
        a->pre_relu.data, grad.data, hid_n);
    int cat_n = B * 3 * DRIVE_ENC_D;
    drive_gelu_kernel<<<grid_size(cat_n), BLOCK_SIZE, 0, stream>>>(
        a->feat.data, a->cat.data, cat_n);
    Prec gpre = drive_view(a->pre_relu.data, B, ew->hidden);
    Prec gelu = drive_view(a->feat.data, B, 3 * DRIVE_ENC_D);
    puf_mm_tn(&gpre, &gelu, &a->shared_wg, stream);
    drive_sum_cols_kernel<<<ew->hidden, BLOCK_SIZE, 0, stream>>>(
        a->shared_bg.data, a->pre_relu.data, B, ew->hidden);
    puf_mm_nn(&gpre, &ew->shared_w, &gelu, stream);
    drive_gelu_bwd_kernel<<<grid_size(cat_n), BLOCK_SIZE, 0, stream>>>(
        a->cat.data, a->feat.data, cat_n);

    int ego_rows = B;
    int partner_rows = B * DRIVE_PARTNERS;
    int road_rows = B * DRIVE_ROADS;
    int width = 3 * DRIVE_ENC_D;

    drive_pack_kernel<<<grid_size(B * DRIVE_ENC_D), BLOCK_SIZE, 0, stream>>>(
        a->scratch_b.data, a->cat.data, B, DRIVE_ENC_D, width, 0);
    drive_gather_ego_kernel<<<grid_size(ego_rows * EGO_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->feat.data, a->saved_obs.data, B, a->obs_size);
    drive_tower_backward(
        a->scratch_b.data, a->ego_h1.data, a->ego_mean.data, a->ego_rstd.data,
        a->feat.data, ego_rows, EGO_FEATURES,
        &ew->ego_w1, &ew->ego_b1, &ew->ego_ln_g, &ew->ego_ln_b, &ew->ego_w2, &ew->ego_b2,
        &a->ego_w1g, &a->ego_b1g, &a->ego_lng, &a->ego_lnb, &a->ego_w2g, &a->ego_b2g,
        a->scratch_a.data, stream);

    drive_unpool_kernel<<<grid_size(road_rows * DRIVE_ENC_D), BLOCK_SIZE, 0, stream>>>(
        a->scratch_b.data, a->cat.data, a->road_argmax.data,
        B, DRIVE_ROADS, DRIVE_ENC_D, width, DRIVE_ENC_D);
    drive_gather_road_kernel<<<grid_size(road_rows * DRIVE_ROAD_IN), BLOCK_SIZE, 0, stream>>>(
        a->feat.data, a->saved_obs.data, B, a->obs_size);
    drive_tower_backward(
        a->scratch_b.data, a->road_h1.data, a->road_mean.data, a->road_rstd.data,
        a->feat.data, road_rows, DRIVE_ROAD_IN,
        &ew->road_w1, &ew->road_b1, &ew->road_ln_g, &ew->road_ln_b, &ew->road_w2, &ew->road_b2,
        &a->road_w1g, &a->road_b1g, &a->road_lng, &a->road_lnb, &a->road_w2g, &a->road_b2g,
        a->scratch_a.data, stream);

    drive_unpool_kernel<<<grid_size(partner_rows * DRIVE_ENC_D), BLOCK_SIZE, 0, stream>>>(
        a->scratch_b.data, a->cat.data, a->partner_argmax.data,
        B, DRIVE_PARTNERS, DRIVE_ENC_D, width, 2 * DRIVE_ENC_D);
    drive_gather_partner_kernel<<<grid_size(partner_rows * PARTNER_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->feat.data, a->saved_obs.data, B, a->obs_size);
    drive_tower_backward(
        a->scratch_b.data, a->partner_h1.data, a->partner_mean.data, a->partner_rstd.data,
        a->feat.data, partner_rows, PARTNER_FEATURES,
        &ew->partner_w1, &ew->partner_b1, &ew->partner_ln_g, &ew->partner_ln_b,
        &ew->partner_w2, &ew->partner_b2,
        &a->partner_w1g, &a->partner_b1g, &a->partner_lng, &a->partner_lnb,
        &a->partner_w2g, &a->partner_b2g,
        a->scratch_a.data, stream);
}

static void drive_encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    DriveEncoderWeights* ew = (DriveEncoderWeights*)w;
    drive_init_linear(&ew->ego_w1, &ew->ego_b1, seed, stream);
    drive_init_ln(&ew->ego_ln_g, &ew->ego_ln_b, stream);
    drive_init_linear(&ew->ego_w2, &ew->ego_b2, seed, stream);
    drive_init_linear(&ew->partner_w1, &ew->partner_b1, seed, stream);
    drive_init_ln(&ew->partner_ln_g, &ew->partner_ln_b, stream);
    drive_init_linear(&ew->partner_w2, &ew->partner_b2, seed, stream);
    drive_init_linear(&ew->road_w1, &ew->road_b1, seed, stream);
    drive_init_ln(&ew->road_ln_g, &ew->road_ln_b, stream);
    drive_init_linear(&ew->road_w2, &ew->road_b2, seed, stream);
    drive_init_linear(&ew->shared_w, &ew->shared_b, seed, stream);
}

static void drive_reg_tower_params(Allocator* alloc,
        Prec* w1, Prec* b1, Prec* ln_g, Prec* ln_b, Prec* w2, Prec* b2,
        int in_dim) {
    drive_reg_mat(alloc, w1, DRIVE_ENC_D, in_dim);
    drive_reg_vec(alloc, b1, DRIVE_ENC_D);
    drive_reg_vec(alloc, ln_g, DRIVE_ENC_D);
    drive_reg_vec(alloc, ln_b, DRIVE_ENC_D);
    drive_reg_mat(alloc, w2, DRIVE_ENC_D, DRIVE_ENC_D);
    drive_reg_vec(alloc, b2, DRIVE_ENC_D);
}

static void drive_reg_tower_grads(Allocator* alloc,
        Prec* w1, Prec* b1, Prec* ln_g, Prec* ln_b, Prec* w2, Prec* b2,
        int in_dim) {
    drive_reg_tower_params(alloc, w1, b1, ln_g, ln_b, w2, b2, in_dim);
}

static void drive_encoder_reg_params(void* w, Allocator* alloc) {
    DriveEncoderWeights* ew = (DriveEncoderWeights*)w;
    drive_reg_tower_params(alloc,
        &ew->ego_w1, &ew->ego_b1, &ew->ego_ln_g, &ew->ego_ln_b, &ew->ego_w2, &ew->ego_b2,
        EGO_FEATURES);
    drive_reg_tower_params(alloc,
        &ew->partner_w1, &ew->partner_b1, &ew->partner_ln_g, &ew->partner_ln_b,
        &ew->partner_w2, &ew->partner_b2, PARTNER_FEATURES);
    drive_reg_tower_params(alloc,
        &ew->road_w1, &ew->road_b1, &ew->road_ln_g, &ew->road_ln_b, &ew->road_w2, &ew->road_b2,
        DRIVE_ROAD_IN);
    drive_reg_mat(alloc, &ew->shared_w, ew->hidden, 3 * DRIVE_ENC_D);
    drive_reg_vec(alloc, &ew->shared_b, ew->hidden);
}

static void drive_reg_acts(DriveEncoderActivations* a, Allocator* alloc, int B, bool train) {
    int partner_rows = B * DRIVE_PARTNERS;
    int road_rows = B * DRIVE_ROADS;
    a->cap_b = B;
    drive_reg_mat(alloc, &a->feat, road_rows, DRIVE_ROAD_IN);
    drive_reg_mat(alloc, &a->scratch_a, road_rows, DRIVE_ENC_D);
    drive_reg_mat(alloc, &a->scratch_b, road_rows, DRIVE_ENC_D);
    drive_reg_mat(alloc, &a->ego_h1, B, DRIVE_ENC_D);
    drive_reg_mat(alloc, &a->partner_h1, partner_rows, DRIVE_ENC_D);
    drive_reg_mat(alloc, &a->road_h1, road_rows, DRIVE_ENC_D);
    drive_reg_vec(alloc, &a->ego_mean, B);
    drive_reg_vec(alloc, &a->ego_rstd, B);
    drive_reg_vec(alloc, &a->partner_mean, partner_rows);
    drive_reg_vec(alloc, &a->partner_rstd, partner_rows);
    drive_reg_vec(alloc, &a->road_mean, road_rows);
    drive_reg_vec(alloc, &a->road_rstd, road_rows);
    drive_reg_mat(alloc, &a->ego_pool, B, DRIVE_ENC_D);
    drive_reg_mat(alloc, &a->partner_pool, B, DRIVE_ENC_D);
    drive_reg_mat(alloc, &a->road_pool, B, DRIVE_ENC_D);
    a->partner_argmax = {};
    a->partner_argmax.shape[0] = B;
    a->partner_argmax.shape[1] = DRIVE_ENC_D;
    alloc_register(alloc, &a->partner_argmax);
    a->road_argmax = {};
    a->road_argmax.shape[0] = B;
    a->road_argmax.shape[1] = DRIVE_ENC_D;
    alloc_register(alloc, &a->road_argmax);
    drive_reg_mat(alloc, &a->cat, B, 3 * DRIVE_ENC_D);
    drive_reg_mat(alloc, &a->pre_relu, B, a->hidden);
    drive_reg_mat(alloc, &a->out, B, a->hidden);
    if (train) {
        drive_reg_mat(alloc, &a->saved_obs, B, a->obs_size);
    }
}

static void drive_encoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    DriveEncoderWeights* ew = (DriveEncoderWeights*)w;
    DriveEncoderActivations* a = (DriveEncoderActivations*)activations;
    *a = {};
    a->obs_size = ew->obs_size;
    a->hidden = ew->hidden;
    drive_reg_acts(a, acts, B_TT, true);
    drive_reg_tower_grads(grads,
        &a->ego_w1g, &a->ego_b1g, &a->ego_lng, &a->ego_lnb, &a->ego_w2g, &a->ego_b2g,
        EGO_FEATURES);
    drive_reg_tower_grads(grads,
        &a->partner_w1g, &a->partner_b1g, &a->partner_lng, &a->partner_lnb,
        &a->partner_w2g, &a->partner_b2g, PARTNER_FEATURES);
    drive_reg_tower_grads(grads,
        &a->road_w1g, &a->road_b1g, &a->road_lng, &a->road_lnb, &a->road_w2g, &a->road_b2g,
        DRIVE_ROAD_IN);
    drive_reg_mat(grads, &a->shared_wg, ew->hidden, 3 * DRIVE_ENC_D);
    drive_reg_vec(grads, &a->shared_bg, ew->hidden);
}

static void drive_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    DriveEncoderWeights* ew = (DriveEncoderWeights*)w;
    DriveEncoderActivations* a = (DriveEncoderActivations*)activations;
    *a = {};
    a->obs_size = ew->obs_size;
    a->hidden = ew->hidden;
    drive_reg_acts(a, alloc, B, false);
}

static void* drive_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    DriveEncoderWeights* ew = (DriveEncoderWeights*)calloc(1, sizeof(DriveEncoderWeights));
    ew->obs_size = e->in_dim;
    ew->hidden = e->out_dim;
    assert(ew->hidden % 4 == 0 && "drive encoder hidden_size must be a multiple of 4");
    assert(ew->obs_size == OBS_SIZE);
    return ew;
}

static void create_drive_encoder(Encoder* enc) {
    long enc_params = drive_encoder_param_count(enc->out_dim);
    fprintf(stderr,
        "drive encoder: Linear-LN-Linear max-pool, road one-hot %d, width %d, hidden %d, encoder params %ld\n",
        DRIVE_ROAD_CLASSES, DRIVE_ENC_D, enc->out_dim, enc_params);
    fflush(stderr);
    *enc = Encoder{
        .forward = drive_encoder_forward,
        .backward = drive_encoder_backward,
        .init_weights = drive_encoder_init_weights,
        .reg_params = drive_encoder_reg_params,
        .reg_train = drive_encoder_reg_train,
        .reg_rollout = drive_encoder_reg_rollout,
        .create_weights = drive_encoder_create_weights,
        .in_dim = enc->in_dim,
        .out_dim = enc->out_dim,
        .activation_size = sizeof(DriveEncoderActivations),
    };
}
