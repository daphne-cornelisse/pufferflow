#include <cublasLt.h>

// Drive entity encoder. Included by src/ocean.cu.
//
// Each ego, partner, and road token goes through its own
// Linear -> LayerNorm -> SiLU -> Linear. SiLU is the pointwise
// nonlinearity; without it the tower is two affine maps around a norm.
// The last road feature is a class in 0..6 and becomes a 7-way one-hot;
// the first 6 features stay continuous.
//
// Those tokens are not pooled. Four learned queries cross-attend
// ego | road | partner. The MLP output is both the key and the value,
// so there is no second projection per token. An all-zero partner or
// road slot is padding and is masked out of the softmax. A zero road
// would otherwise look like a lane at the origin, because class 0 is
// ROAD_LANE. Width 64 uses 2 heads (dim 32). Width 128 and up keeps
// 4 heads, each at least 32 wide. Scale is 1/sqrt(head_dim).
// Residual adds the unprojected queries, then LayerNorm over the embed
// dim, a linear back to hidden, and tanh.
//
// A batch that fits in one chunk leaves qproj, the token embeddings,
// context, and the pre-norm residual in scratch. Backward reuses them.
// A larger batch recomputes each chunk from the saved observation.

// Trainer minibatch is 8192 rows. One chunk keeps each GEMM one cublas call.
#define DRIVE_ENC_CHUNK 8192
#define DRIVE_LN_EPS 1.0e-5f
#define DRIVE_ROAD_CONT (ROAD_FEATURES - 1)
#define DRIVE_ROAD_CLASSES (DRIVEWAY - ROAD_LANE + 1)
#define DRIVE_ROAD_IN (DRIVE_ROAD_CONT + DRIVE_ROAD_CLASSES)
#define DRIVE_LN_ROWS 8
#define DRIVE_ATTN_Q 4
#define DRIVE_ATTN_HEADS_MAX 4
#define DRIVE_ATTN_S (1 + MAX_ROAD_SEGMENT_OBSERVATIONS + MAX_AGENTS)
#define DRIVE_MASK_LOGIT (-1.0e30f)

// 16-wide heads leave half of each warp idle. Hidden 64 uses 2 heads.
static int drive_attn_heads(int H) {
    return H >= 128 ? DRIVE_ATTN_HEADS_MAX : 2;
}

struct DriveEncWeights {
    Prec ego_w1, ego_b1, ego_lng, ego_lnb, ego_w2, ego_b2;
    Prec part_w1, part_b1, part_lng, part_lnb, part_w2, part_b2;
    Prec road_w1, road_b1, road_lng, road_lnb, road_w2, road_b2;
    Prec q_lat, wq, bq;
    Prec wo, bo;
    Prec attn_lng, attn_lnb, fc_w, fc_b;
    int hidden;
};

// acc[] is fp32 so chunk reductions do not round away small bias grads.
// Order per tower: b1, beta, gamma, b2. Attention biases follow.
enum {
    DRIVE_ACC_EGO_B1 = 0,
    DRIVE_ACC_EGO_BE,
    DRIVE_ACC_EGO_G,
    DRIVE_ACC_EGO_B2,
    DRIVE_ACC_PART_B1,
    DRIVE_ACC_PART_BE,
    DRIVE_ACC_PART_G,
    DRIVE_ACC_PART_B2,
    DRIVE_ACC_ROAD_B1,
    DRIVE_ACC_ROAD_BE,
    DRIVE_ACC_ROAD_G,
    DRIVE_ACC_ROAD_B2,
    DRIVE_ACC_BQ,
    DRIVE_ACC_BO,
    DRIVE_ACC_ATTN_BE,
    DRIVE_ACC_ATTN_G,
    DRIVE_ACC_FC_B,
    DRIVE_ACC_N,
};

struct DriveEncActs {
    Prec out, saved_obs;
    Prec ego_w1_g, ego_b1_g, ego_lng_g, ego_lnb_g, ego_w2_g, ego_b2_g;
    Prec part_w1_g, part_b1_g, part_lng_g, part_lnb_g, part_w2_g, part_b2_g;
    Prec road_w1_g, road_b1_g, road_lng_g, road_lnb_g, road_w2_g, road_b2_g;
    Prec q_lat_g, wq_g, bq_g;
    Prec wo_g, bo_g;
    Prec attn_lng_g, attn_lnb_g, fc_w_g, fc_b_g;
    Prec pack, buf0, buf1, buf2, buf3;
    Prec tok;
    Prec qproj, ctx, proj, pre, hid, dq;
    Float mask, dqsum, acc_lat;
    Float acc[DRIVE_ACC_N];
};

__device__ float drive_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

__global__ void drive_fill_kernel(precision_t* dst, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = from_float(v);
    }
}

__global__ void drive_store_kernel(precision_t* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = from_float(src[i]);
    }
}

__global__ void drive_bias_kernel(precision_t* y, const precision_t* b,
        int n, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = from_float(to_float(y[i]) + to_float(b[i % H]));
    }
}

// Many row tiles so a short hidden vector does not run as two warps.
__global__ void drive_colsum_kernel(float* acc, const precision_t* src,
        int rows, int cols, int rows_per) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r0 = blockIdx.y * rows_per;
    if (c >= cols || r0 >= rows) {
        return;
    }
    int r1 = r0 + rows_per;
    if (r1 > rows) {
        r1 = rows;
    }
    float s = 0.0f;
    for (int r = r0; r < r1; r++) {
        s += to_float(src[(long)r * cols + c]);
    }
    atomicAdd(acc + c, s);
}

__global__ void drive_pack_copy_kernel(precision_t* dst,
        const precision_t* obs, int rows, int obs_size,
        int off, int slots, int feat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * slots * feat;
    if (idx >= total) {
        return;
    }
    int f = idx % feat;
    int tmp = idx / feat;
    int n = tmp % slots;
    int r = tmp / slots;
    dst[idx] = obs[(long)r * obs_size + off + (long)n * feat + f];
}

__global__ void drive_pack_road_kernel(precision_t* dst,
        const precision_t* obs, int rows, int obs_size, int off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int slots = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int total = rows * slots;
    if (idx >= total) {
        return;
    }
    int n = idx % slots;
    int r = idx / slots;
    const precision_t* src = obs + (long)r * obs_size + off
        + (long)n * ROAD_FEATURES;
    precision_t* d = dst + (long)idx * DRIVE_ROAD_IN;
    for (int k = 0; k < DRIVE_ROAD_CONT; k++) {
        d[k] = src[k];
    }
    int cls = (int)to_float(src[DRIVE_ROAD_CONT]);
    if (cls < 0) {
        cls = 0;
    }
    if (cls >= DRIVE_ROAD_CLASSES) {
        cls = DRIVE_ROAD_CLASSES - 1;
    }
    for (int k = 0; k < DRIVE_ROAD_CLASSES; k++) {
        d[DRIVE_ROAD_CONT + k] = from_float(k == cls ? 1.0f : 0.0f);
    }
}

// One warp per token. A 256-thread block per token left the GPU idle.
__global__ void drive_ln_fwd_kernel(precision_t* y, const precision_t* x,
        const precision_t* gamma, const precision_t* beta, int H, int rows) {
    int row = blockIdx.x * DRIVE_LN_ROWS + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const precision_t* xr = x + (long)row * H;
    precision_t* yr = y + (long)row * H;
    float sum = 0.0f;
    for (int i = lane; i < H; i += 32) {
        sum += to_float(xr[i]);
    }
    float mean = drive_warp_sum(sum) / (float)H;
    float var = 0.0f;
    for (int i = lane; i < H; i += 32) {
        float d = to_float(xr[i]) - mean;
        var += d * d;
    }
    float inv = rsqrtf(drive_warp_sum(var) / (float)H + DRIVE_LN_EPS);
    for (int i = lane; i < H; i += 32) {
        float xhat = (to_float(xr[i]) - mean) * inv;
        float o = xhat * to_float(gamma[i]) + to_float(beta[i]);
        yr[i] = from_float(o);
    }
}

__global__ void drive_ln_bwd_kernel(precision_t* dx, precision_t* dgamma,
        const precision_t* dy, const precision_t* x,
        const precision_t* gamma, int H, int rows) {
    int row = blockIdx.x * DRIVE_LN_ROWS + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const precision_t* xr = x + (long)row * H;
    const precision_t* dyr = dy + (long)row * H;
    precision_t* dxr = dx + (long)row * H;
    precision_t* dgr = dgamma + (long)row * H;
    float sum = 0.0f;
    for (int i = lane; i < H; i += 32) {
        sum += to_float(xr[i]);
    }
    float mean = drive_warp_sum(sum) / (float)H;
    float var = 0.0f;
    for (int i = lane; i < H; i += 32) {
        float d = to_float(xr[i]) - mean;
        var += d * d;
    }
    float inv = rsqrtf(drive_warp_sum(var) / (float)H + DRIVE_LN_EPS);
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int i = lane; i < H; i += 32) {
        float xhat = (to_float(xr[i]) - mean) * inv;
        float dxhat = to_float(dyr[i]) * to_float(gamma[i]);
        s1 += dxhat;
        s2 += dxhat * xhat;
    }
    float mean1 = drive_warp_sum(s1) / (float)H;
    float mean2 = drive_warp_sum(s2) / (float)H;
    for (int i = lane; i < H; i += 32) {
        float xhat = (to_float(xr[i]) - mean) * inv;
        float dxhat = to_float(dyr[i]) * to_float(gamma[i]);
        float g = inv * (dxhat - mean1 - xhat * mean2);
        dxr[i] = from_float(g);
        dgr[i] = from_float(to_float(dyr[i]) * xhat);
    }
}

// One row's tokens are staged into shared memory as a contiguous
// [S, H] slab. Ego, then roads, then partners. The MLP output is both
// the key and the value, so the slab is loaded once per use.
#define DRIVE_ATTN_THREADS 256

// Token stride is H+2 bf16. H is a multiple of 8, so the step in
// 4-byte banks is odd and a warp of consecutive keys does not collide.
__device__ int drive_slab_stride(int H) {
    return H + 2;
}

__device__ void drive_stage_row(precision_t* slab, const precision_t* ego,
        const precision_t* road, const precision_t* part,
        int r, int H, int tid, int nt) {
    int stride = drive_slab_stride(H);
    const precision_t* e = ego + (long)r * H;
    for (int h = tid; h < H; h += nt) {
        slab[h] = e[h];
    }
    int road_n = MAX_ROAD_SEGMENT_OBSERVATIONS * H;
    const precision_t* rd = road + (long)r * road_n;
    for (int i = tid; i < road_n; i += nt) {
        int n = i / H;
        int h = i - n * H;
        slab[(1 + n) * stride + h] = rd[i];
    }
    int part_n = MAX_AGENTS * H;
    const precision_t* pt = part + (long)r * part_n;
    int base = 1 + MAX_ROAD_SEGMENT_OBSERVATIONS;
    for (int i = tid; i < part_n; i += nt) {
        int n = i / H;
        int h = i - n * H;
        slab[(base + n) * stride + h] = pt[i];
    }
}

__device__ void drive_scatter_row(precision_t* ego, precision_t* road,
        precision_t* part, int r, int s, int h, int H, float v) {
    precision_t bits = from_float(v);
    if (s == 0) {
        ego[(long)r * H + h] = bits;
        return;
    }
    if (s <= MAX_ROAD_SEGMENT_OBSERVATIONS) {
        int n = s - 1;
        road[((long)r * MAX_ROAD_SEGMENT_OBSERVATIONS + n) * H + h] = bits;
        return;
    }
    int n = s - 1 - MAX_ROAD_SEGMENT_OBSERVATIONS;
    part[((long)r * MAX_AGENTS + n) * H + h] = bits;
}

__device__ int drive_score_at(int head, int q, int s) {
    return (head * DRIVE_ATTN_Q + q) * DRIVE_ATTN_S + s;
}

__device__ float drive_dot_d(const precision_t* qv, const precision_t* kv,
        int d) {
    float dot = 0.0f;
    int i = 0;
    for (; i + 3 < d; i += 4) {
        dot += to_float(qv[i]) * to_float(kv[i]);
        dot += to_float(qv[i + 1]) * to_float(kv[i + 1]);
        dot += to_float(qv[i + 2]) * to_float(kv[i + 2]);
        dot += to_float(qv[i + 3]) * to_float(kv[i + 3]);
    }
    for (; i < d; i++) {
        dot += to_float(qv[i]) * to_float(kv[i]);
    }
    return dot;
}

// 256 threads, one block per row. Scores sit after the padded slab.
// Masked keys are a large negative logit before the softmax.
__global__ void drive_xattn_fwd_kernel(precision_t* ctx,
        const precision_t* Q, const precision_t* ego,
        const precision_t* road, const precision_t* part,
        const float* mask, int H, int heads) {
    extern __shared__ __align__(16) char drive_attn_smem[];
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    int d = H / heads;
    int stride = drive_slab_stride(H);
    float scale = rsqrtf((float)d);
    int slab_n = DRIVE_ATTN_S * stride;
    precision_t* slab = (precision_t*)drive_attn_smem;
    float* scores = (float*)(slab + slab_n);
    const float* row_mask = mask + (long)r * DRIVE_ATTN_S;
    drive_stage_row(slab, ego, road, part, r, H, tid, nt);
    __syncthreads();
    int ndot = heads * DRIVE_ATTN_Q * DRIVE_ATTN_S;
    for (int job = tid; job < ndot; job += nt) {
        int s = job % DRIVE_ATTN_S;
        int t = job / DRIVE_ATTN_S;
        int q = t % DRIVE_ATTN_Q;
        int head = t / DRIVE_ATTN_Q;
        int h0 = head * d;
        float dot = DRIVE_MASK_LOGIT;
        if (row_mask[s] != 0.0f) {
            const precision_t* kv = slab + s * stride + h0;
            const precision_t* qv = Q + (long)q * H + h0;
            dot = drive_dot_d(qv, kv, d) * scale;
        }
        scores[drive_score_at(head, q, s)] = dot;
    }
    __syncthreads();
    int nsoft = heads * DRIVE_ATTN_Q;
    if (tid < nsoft) {
        int head = tid / DRIVE_ATTN_Q;
        int q = tid - head * DRIVE_ATTN_Q;
        float* row = scores + (head * DRIVE_ATTN_Q + q) * DRIVE_ATTN_S;
        float m = row[0];
        for (int s = 1; s < DRIVE_ATTN_S; s++) {
            if (row[s] > m) {
                m = row[s];
            }
        }
        float sum = 0.0f;
        for (int s = 0; s < DRIVE_ATTN_S; s++) {
            float e = expf(row[s] - m);
            row[s] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int s = 0; s < DRIVE_ATTN_S; s++) {
            row[s] *= inv;
        }
    }
    __syncthreads();
    int nout = DRIVE_ATTN_Q * H;
    for (int job = tid; job < nout; job += nt) {
        int h = job % H;
        int q = job / H;
        int head = h / d;
        const float* p = scores + (head * DRIVE_ATTN_Q + q) * DRIVE_ATTN_S;
        float acc = 0.0f;
        for (int s = 0; s < DRIVE_ATTN_S; s++) {
            acc += p[s] * to_float(slab[s * stride + h]);
        }
        long o = (((long)r * DRIVE_ATTN_Q) + q) * H + h;
        ctx[o] = from_float(acc);
    }
}

// Writes dK+dV over the token embeddings. dQ is one row of the batch.
// P and dS sit after the slab.
__global__ void drive_xattn_bwd_kernel(precision_t* dQ,
        const precision_t* dctx, const precision_t* Q,
        precision_t* ego, precision_t* road, precision_t* part,
        const float* mask, int H, int heads) {
    extern __shared__ __align__(16) char drive_attn_smem[];
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    int d = H / heads;
    int stride = drive_slab_stride(H);
    float scale = rsqrtf((float)d);
    int slab_n = DRIVE_ATTN_S * stride;
    int score_n = DRIVE_ATTN_HEADS_MAX * DRIVE_ATTN_Q * DRIVE_ATTN_S;
    precision_t* slab = (precision_t*)drive_attn_smem;
    float* scores = (float*)(slab + slab_n);
    float* dSrow = scores + score_n;
    const float* row_mask = mask + (long)r * DRIVE_ATTN_S;
    const precision_t* row_dy = dctx + ((long)r * DRIVE_ATTN_Q) * H;
    drive_stage_row(slab, ego, road, part, r, H, tid, nt);
    __syncthreads();
    int ndot = heads * DRIVE_ATTN_Q * DRIVE_ATTN_S;
    for (int job = tid; job < ndot; job += nt) {
        int s = job % DRIVE_ATTN_S;
        int t = job / DRIVE_ATTN_S;
        int q = t % DRIVE_ATTN_Q;
        int head = t / DRIVE_ATTN_Q;
        int h0 = head * d;
        float dot = DRIVE_MASK_LOGIT;
        if (row_mask[s] != 0.0f) {
            const precision_t* kv = slab + s * stride + h0;
            const precision_t* qv = Q + (long)q * H + h0;
            dot = drive_dot_d(qv, kv, d) * scale;
        }
        scores[drive_score_at(head, q, s)] = dot;
    }
    __syncthreads();
    int nsoft = heads * DRIVE_ATTN_Q;
    if (tid < nsoft) {
        int head = tid / DRIVE_ATTN_Q;
        int q = tid - head * DRIVE_ATTN_Q;
        float* row = scores + (head * DRIVE_ATTN_Q + q) * DRIVE_ATTN_S;
        float m = row[0];
        for (int s = 1; s < DRIVE_ATTN_S; s++) {
            if (row[s] > m) {
                m = row[s];
            }
        }
        float sum = 0.0f;
        for (int s = 0; s < DRIVE_ATTN_S; s++) {
            float e = expf(row[s] - m);
            row[s] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int s = 0; s < DRIVE_ATTN_S; s++) {
            row[s] *= inv;
        }
    }
    __syncthreads();
    for (int job = tid; job < ndot; job += nt) {
        int s = job % DRIVE_ATTN_S;
        int t = job / DRIVE_ATTN_S;
        int q = t % DRIVE_ATTN_Q;
        int head = t / DRIVE_ATTN_Q;
        int h0 = head * d;
        const precision_t* vv = slab + s * stride + h0;
        const precision_t* dy = row_dy + (long)q * H + h0;
        dSrow[drive_score_at(head, q, s)] = drive_dot_d(dy, vv, d);
    }
    __syncthreads();
    if (tid < nsoft) {
        int head = tid / DRIVE_ATTN_Q;
        int q = tid - head * DRIVE_ATTN_Q;
        float* prow = scores + (head * DRIVE_ATTN_Q + q) * DRIVE_ATTN_S;
        float* drow = dSrow + (head * DRIVE_ATTN_Q + q) * DRIVE_ATTN_S;
        float sum = 0.0f;
        for (int s = 0; s < DRIVE_ATTN_S; s++) {
            sum += drow[s] * prow[s];
        }
        for (int s = 0; s < DRIVE_ATTN_S; s++) {
            float p = prow[s];
            drow[s] = p * (drow[s] - sum);
        }
    }
    __syncthreads();
    int nout = DRIVE_ATTN_Q * H;
    for (int job = tid; job < nout; job += nt) {
        int h = job % H;
        int q = job / H;
        int head = h / d;
        const float* ds = dSrow + (head * DRIVE_ATTN_Q + q) * DRIVE_ATTN_S;
        float acc = 0.0f;
        for (int s = 0; s < DRIVE_ATTN_S; s++) {
            acc += ds[s] * scale * to_float(slab[s * stride + h]);
        }
        long o = (((long)r * DRIVE_ATTN_Q) + q) * H + h;
        dQ[o] = from_float(acc);
    }
    int ntok = DRIVE_ATTN_S * H;
    for (int job = tid; job < ntok; job += nt) {
        int s = job / H;
        int h = job - s * H;
        int head = h / d;
        float dv = 0.0f;
        float dk = 0.0f;
        for (int q = 0; q < DRIVE_ATTN_Q; q++) {
            int at = drive_score_at(head, q, s);
            float dy = to_float(row_dy[(long)q * H + h]);
            dv += scores[at] * dy;
            dk += dSrow[at] * scale * to_float(Q[(long)q * H + h]);
        }
        drive_scatter_row(ego, road, part, r, s, h, H, dv + dk);
    }
}

static int drive_attn_bytes(int H, int score_sets) {
    int stride = H + 2;
    int slab = DRIVE_ATTN_S * stride * (int)sizeof(precision_t);
    int scores = DRIVE_ATTN_HEADS_MAX * DRIVE_ATTN_Q * DRIVE_ATTN_S
        * (int)sizeof(float);
    return slab + score_sets * scores;
}

static void drive_attn_allow_smem(const void* fn) {
    cudaError_t st = cudaFuncSetAttribute(fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    assert(st == cudaSuccess);
}


__global__ void drive_residual_kernel(precision_t* pre,
        const precision_t* proj, const precision_t* lat,
        int n, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int h = i % H;
    int q = (i / H) % DRIVE_ATTN_Q;
    float s = to_float(proj[i]) + to_float(lat[q * H + h]);
    pre[i] = from_float(s);
}

__global__ void drive_silu_kernel(precision_t* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = to_float(y[i]);
        float sig = 1.0f / (1.0f + expf(-v));
        y[i] = from_float(v * sig);
    }
}

// dy is the grad of silu(x). x is the pre-activation.
__global__ void drive_silu_bwd_kernel(precision_t* dy, const precision_t* x,
        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = to_float(x[i]);
        float sig = 1.0f / (1.0f + expf(-v));
        float ds = sig * (1.0f + v * (1.0f - sig));
        dy[i] = from_float(to_float(dy[i]) * ds);
    }
}

// Sequence order is ego, roads, partners. A slot whose raw features are
// all zero is padding. Ego is always a real token.
__global__ void drive_mask_kernel(float* mask, const precision_t* obs,
        int rows, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * DRIVE_ATTN_S;
    if (idx >= total) {
        return;
    }
    int s = idx % DRIVE_ATTN_S;
    int r = idx / DRIVE_ATTN_S;
    const precision_t* row = obs + (long)r * obs_size;
    int valid = 1;
    if (s > 0 && s <= MAX_ROAD_SEGMENT_OBSERVATIONS) {
        int n = s - 1;
        int off = EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS
            + n * ROAD_FEATURES;
        valid = 0;
        for (int k = 0; k < ROAD_FEATURES; k++) {
            if (to_float(row[off + k]) != 0.0f) {
                valid = 1;
            }
        }
    } else if (s > MAX_ROAD_SEGMENT_OBSERVATIONS) {
        int n = s - 1 - MAX_ROAD_SEGMENT_OBSERVATIONS;
        int off = EGO_FEATURES + n * PARTNER_FEATURES;
        valid = 0;
        for (int k = 0; k < PARTNER_FEATURES; k++) {
            if (to_float(row[off + k]) != 0.0f) {
                valid = 1;
            }
        }
    }
    mask[idx] = valid ? 1.0f : 0.0f;
}

__global__ void drive_tanh_kernel(precision_t* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = from_float(tanhf(to_float(y[i])));
    }
}

__global__ void drive_tanh_bwd_kernel(precision_t* dy, const precision_t* y,
        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = to_float(y[i]);
        float g = to_float(dy[i]);
        dy[i] = from_float(g * (1.0f - t * t));
    }
}

__global__ void drive_dq_sum_kernel(float* sum, const precision_t* dQ,
        int rows, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float s = 0.0f;
    for (int r = 0; r < rows; r++) {
        s += to_float(dQ[(long)r * n + i]);
    }
    sum[i] = s;
}

// dQsum is the batch-reduced grad of the shared projected queries.
__global__ void drive_dq_apply_kernel(precision_t* dWq, float* dbq,
        float* dlat, const float* dQsum, const precision_t* latent,
        const precision_t* Wq, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int span = H * H;
    if (idx < span) {
        int n = idx / H;
        int k = idx - n * H;
        float s = 0.0f;
        for (int q = 0; q < DRIVE_ATTN_Q; q++) {
            s += dQsum[q * H + n] * to_float(latent[q * H + k]);
        }
        dWq[idx] = from_float(to_float(dWq[idx]) + s);
    }
    if (idx < H) {
        float s = 0.0f;
        for (int q = 0; q < DRIVE_ATTN_Q; q++) {
            s += dQsum[q * H + idx];
        }
        dbq[idx] += s;
    }
    if (idx < DRIVE_ATTN_Q * H) {
        int q = idx / H;
        int k = idx - q * H;
        float s = 0.0f;
        for (int n = 0; n < H; n++) {
            s += dQsum[q * H + n] * to_float(Wq[n * H + k]);
        }
        dlat[idx] += s;
    }
}

static Prec drive_view(precision_t* data, int rows, int cols) {
    Prec t = {.data = data, .shape = {rows, cols}};
    return t;
}

#define DRIVE_DW_BM 64
#define DRIVE_DW_BN 64
#define DRIVE_DW_BK 32
#define DRIVE_DW_TM 4
#define DRIVE_DW_TN 4

// dC[n, k] = sum_r A[r, n] * B[r, k], split over rows. cublas TN on this
// shape ran at well under 1 TFLOP/s. A 32x32 thread block filled the SM
// with one resident block and stalled on each global load. 256 threads
// each own a 4x4 register tile so several blocks stay in flight.
__global__ void drive_dw_partial(float* partial, const precision_t* A,
        const precision_t* B, int rows, int N, int K, int rows_per) {
    int z = blockIdx.z;
    int n0 = blockIdx.y * DRIVE_DW_BM;
    int k0 = blockIdx.x * DRIVE_DW_BN;
    int tid = threadIdx.x;
    int tx = tid & 15;
    int ty = tid >> 4;
    int r_begin = z * rows_per;
    if (r_begin >= rows) {
        return;
    }
    int r_end = r_begin + rows_per;
    if (r_end > rows) {
        r_end = rows;
    }
    float acc[DRIVE_DW_TM][DRIVE_DW_TN];
    #pragma unroll
    for (int a = 0; a < DRIVE_DW_TM; a++) {
        #pragma unroll
        for (int b = 0; b < DRIVE_DW_TN; b++) {
            acc[a][b] = 0.0f;
        }
    }
    __shared__ float As[DRIVE_DW_BK][DRIVE_DW_BM + 1];
    __shared__ float Bs[DRIVE_DW_BK][DRIVE_DW_BN + 1];
    for (int r0 = r_begin; r0 < r_end; r0 += DRIVE_DW_BK) {
        for (int e = tid; e < DRIVE_DW_BK * DRIVE_DW_BM; e += 256) {
            int bk = e / DRIVE_DW_BM;
            int bm = e - bk * DRIVE_DW_BM;
            int r = r0 + bk;
            int n = n0 + bm;
            float v = 0.0f;
            if (r < r_end && n < N) {
                v = to_float(A[(long)r * N + n]);
            }
            As[bk][bm] = v;
        }
        for (int e = tid; e < DRIVE_DW_BK * DRIVE_DW_BN; e += 256) {
            int bk = e / DRIVE_DW_BN;
            int bn = e - bk * DRIVE_DW_BN;
            int r = r0 + bk;
            int k = k0 + bn;
            float v = 0.0f;
            if (r < r_end && k < K) {
                v = to_float(B[(long)r * K + k]);
            }
            Bs[bk][bn] = v;
        }
        __syncthreads();
        int nlim = DRIVE_DW_BK;
        if (r0 + nlim > r_end) {
            nlim = r_end - r0;
        }
        int nrow = ty * DRIVE_DW_TM;
        int ncol = tx * DRIVE_DW_TN;
        for (int i = 0; i < nlim; i++) {
            float av[DRIVE_DW_TM];
            float bv[DRIVE_DW_TN];
            #pragma unroll
            for (int a = 0; a < DRIVE_DW_TM; a++) {
                av[a] = As[i][nrow + a];
            }
            #pragma unroll
            for (int b = 0; b < DRIVE_DW_TN; b++) {
                bv[b] = Bs[i][ncol + b];
            }
            #pragma unroll
            for (int a = 0; a < DRIVE_DW_TM; a++) {
                #pragma unroll
                for (int b = 0; b < DRIVE_DW_TN; b++) {
                    acc[a][b] += av[a] * bv[b];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int a = 0; a < DRIVE_DW_TM; a++) {
        int n = n0 + ty * DRIVE_DW_TM + a;
        if (n < N) {
            #pragma unroll
            for (int b = 0; b < DRIVE_DW_TN; b++) {
                int k = k0 + tx * DRIVE_DW_TN + b;
                if (k < K) {
                    partial[((long)z * N + n) * K + k] = acc[a][b];
                }
            }
        }
    }
}

__global__ void drive_dw_reduce(precision_t* C, const float* partial,
        int splits, int nk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nk) {
        return;
    }
    float s = to_float(C[idx]);
    for (int z = 0; z < splits; z++) {
        s += partial[(long)z * nk + idx];
    }
    C[idx] = from_float(s);
}

static void drive_fill(precision_t* dst, float v, int n, cudaStream_t stream) {
    drive_fill_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, v, n);
}

static void drive_bias(precision_t* y, precision_t* b, int rows, int H,
        cudaStream_t stream) {
    int n = rows * H;
    drive_bias_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        y, b, n, H);
}

static void drive_colsum(float* acc, precision_t* src, int rows, int cols,
        cudaStream_t stream) {
    const int rows_per = 512;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, (rows + rows_per - 1) / rows_per);
    drive_colsum_kernel<<<grid, block, 0, stream>>>(
        acc, src, rows, cols, rows_per);
}

static void drive_ln_fwd(precision_t* y, precision_t* x,
        precision_t* gamma, precision_t* beta, int rows, int H,
        cudaStream_t stream) {
    assert(rows > 0);
    dim3 block(32, DRIVE_LN_ROWS);
    dim3 grid((rows + DRIVE_LN_ROWS - 1) / DRIVE_LN_ROWS);
    drive_ln_fwd_kernel<<<grid, block, 0, stream>>>(
        y, x, gamma, beta, H, rows);
}

static void drive_silu(precision_t* y, int n, cudaStream_t stream) {
    drive_silu_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(y, n);
}

static void drive_silu_bwd(precision_t* dy, const precision_t* x, int n,
        cudaStream_t stream) {
    drive_silu_bwd_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dy, x, n);
}

static void drive_mask(float* mask, const precision_t* obs, int rows,
        cudaStream_t stream) {
    int n = rows * DRIVE_ATTN_S;
    drive_mask_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        mask, obs, rows, OBS_SIZE);
}

// partial is a float view of a scratch buffer. It must not alias A, B, or C.
// Hidden is often a single 64-wide tile, so the row split is what puts
// more than one block on each SM.
static void drive_dw(precision_t* C, const precision_t* A,
        const precision_t* B, float* partial, int rows, int N, int K,
        int partial_floats, cudaStream_t stream) {
    int tiles_n = (N + DRIVE_DW_BM - 1) / DRIVE_DW_BM;
    int tiles_k = (K + DRIVE_DW_BN - 1) / DRIVE_DW_BN;
    int tiles = tiles_n * tiles_k;
    int splits = 1024 / tiles;
    if (splits < 1) {
        splits = 1;
    }
    if (splits > rows) {
        splits = rows;
    }
    long room = (long)partial_floats / ((long)N * K);
    if (room < 1) {
        room = 1;
    }
    if ((long)splits > room) {
        splits = (int)room;
    }
    long need = (long)splits * N * K;
    assert(need <= partial_floats);
    int rows_per = (rows + splits - 1) / splits;
    cudaMemsetAsync(partial, 0, (size_t)need * sizeof(float), stream);
    dim3 block(256);
    dim3 grid((K + DRIVE_DW_BN - 1) / DRIVE_DW_BN,
        (N + DRIVE_DW_BM - 1) / DRIVE_DW_BM, splits);
    drive_dw_partial<<<grid, block, 0, stream>>>(
        partial, A, B, rows, N, K, rows_per);
    int nk = N * K;
    drive_dw_reduce<<<grid_size(nk), BLOCK_SIZE, 0, stream>>>(
        C, partial, splits, nk);
}

static void drive_zero_prec(Prec* t, cudaStream_t stream) {
    cudaMemsetAsync(t->data, 0,
        numel(t->shape) * sizeof(precision_t), stream);
}

static int drive_partial_cap(int H) {
    long bytes = (long)DRIVE_ENC_CHUNK * MAX_ROAD_SEGMENT_OBSERVATIONS
        * H * (long)sizeof(precision_t);
    return (int)(bytes / (long)sizeof(float));
}

static void drive_pack(precision_t* dst, const precision_t* obs,
        int rows, int obs_size, int off, int slots, int raw, int one_hot,
        cudaStream_t stream) {
    if (one_hot) {
        int n = rows * slots;
        drive_pack_road_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            dst, obs, rows, obs_size, off);
    } else {
        int n = rows * slots * raw;
        drive_pack_copy_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            dst, obs, rows, obs_size, off, slots, raw);
    }
}

// Cached cublasLt gemm that adds the bias in the epilogue, so the big
// token projections do not read and write the activation a second time.
struct DriveLtAlgo {
    int M, N, K;
    int used;
    int ok;
    cublasLtMatmulAlgo_t algo;
};

static thread_local cublasLtHandle_t drive_lt_handle = NULL;
static thread_local void* drive_lt_ws = NULL;
static thread_local DriveLtAlgo drive_lt_cache[16];

static int drive_lt_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess) {
        return 0;
    }
    return cap == cudaStreamCaptureStatusActive;
}

static int drive_lt_ensure(void) {
    if (drive_lt_handle && drive_lt_ws) {
        return 1;
    }
    if (!drive_lt_handle) {
        if (cublasLtCreate(&drive_lt_handle) != CUBLAS_STATUS_SUCCESS) {
            drive_lt_handle = NULL;
            return 0;
        }
    }
    if (!drive_lt_ws) {
        if (cudaMalloc(&drive_lt_ws, 32u * 1024u * 1024u) != cudaSuccess) {
            drive_lt_ws = NULL;
            return 0;
        }
    }
    return 1;
}

static DriveLtAlgo* drive_lt_find(int rows, int N, int K, int ok_only) {
    DriveLtAlgo* slot = NULL;
    for (int i = 0; i < 16; i++) {
        DriveLtAlgo* c = &drive_lt_cache[i];
        if (c->used && c->M == rows && c->N == N && c->K == K) {
            if (ok_only && !c->ok) {
                return NULL;
            }
            return c;
        }
        if (!c->used && !slot) {
            slot = c;
        }
    }
    if (ok_only) {
        return NULL;
    }
    if (!slot) {
        slot = &drive_lt_cache[0];
    }
    return slot;
}

static int drive_lt_bias(const precision_t* x, const precision_t* w,
        const precision_t* bias, precision_t* y, int rows, int N, int K,
        cudaStream_t stream) {
    if (rows < 256) {
        return 0;
    }
    if (drive_lt_capturing(stream)) {
        if (!drive_lt_handle || !drive_lt_ws) {
            return 0;
        }
    } else if (!drive_lt_ensure()) {
        return 0;
    }
    cublasLtMatmulDesc_t desc = NULL;
    cublasLtMatrixLayout_t w_lay = NULL;
    cublasLtMatrixLayout_t x_lay = NULL;
    cublasLtMatrixLayout_t y_lay = NULL;
    cublasStatus_t st = cublasLtMatmulDescCreate(
        &desc, CUBLAS_COMPUTE, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return 0;
    }
    cublasOperation_t trans_w = CUBLAS_OP_T;
    cublasOperation_t trans_x = CUBLAS_OP_N;
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cudaDataType_t bias_type = CUBLAS_PRECISION;
    const void* bias_ptr = bias;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
        &trans_w, sizeof(trans_w));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_x, sizeof(trans_x));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epi, sizeof(epi));
    cublasLtMatmulDescSetAttribute(desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr));
    cublasLtMatmulDescSetAttribute(desc,
        CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type));
    st = cublasLtMatrixLayoutCreate(&w_lay, CUBLAS_PRECISION, K, N, K);
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&x_lay, CUBLAS_PRECISION, K, rows, K);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&y_lay, CUBLAS_PRECISION, N, rows, N);
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
        if (w_lay) cublasLtMatrixLayoutDestroy(w_lay);
        if (x_lay) cublasLtMatrixLayoutDestroy(x_lay);
        cublasLtMatmulDescDestroy(desc);
        return 0;
    }
    int capturing = drive_lt_capturing(stream);
    DriveLtAlgo* slot = drive_lt_find(rows, N, K, capturing);
    if (!slot) {
        cublasLtMatrixLayoutDestroy(w_lay);
        cublasLtMatrixLayoutDestroy(x_lay);
        cublasLtMatrixLayoutDestroy(y_lay);
        cublasLtMatmulDescDestroy(desc);
        return 0;
    }
    if (!slot->used || slot->M != rows || slot->N != N || slot->K != K) {
        cublasLtMatmulPreference_t pref = NULL;
        cublasLtMatmulHeuristicResult_t heur = {};
        int found = 0;
        size_t cap = 32u * 1024u * 1024u;
        slot->used = 1;
        slot->M = rows;
        slot->N = N;
        slot->K = K;
        slot->ok = 0;
        if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
            cublasLtMatmulPreferenceSetAttribute(pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cap, sizeof(cap));
            if (cublasLtMatmulAlgoGetHeuristic(drive_lt_handle, desc,
                    w_lay, x_lay, y_lay, y_lay, pref, 1, &heur, &found)
                    == CUBLAS_STATUS_SUCCESS && found > 0) {
                slot->algo = heur.algo;
                slot->ok = 1;
            }
            cublasLtMatmulPreferenceDestroy(pref);
        }
    }
    int ran = 0;
    st = CUBLAS_STATUS_NOT_SUPPORTED;
    if (slot->ok) {
        float alpha = 1.0f;
        float beta = 0.0f;
        st = cublasLtMatmul(drive_lt_handle, desc, &alpha, w, w_lay,
            x, x_lay, &beta, y, y_lay, y, y_lay, &slot->algo, drive_lt_ws,
            32u * 1024u * 1024u, stream);
        ran = (st == CUBLAS_STATUS_SUCCESS);
    }
    cublasLtMatrixLayoutDestroy(w_lay);
    cublasLtMatrixLayoutDestroy(x_lay);
    cublasLtMatrixLayoutDestroy(y_lay);
    cublasLtMatmulDescDestroy(desc);
    return ran;
}

// Rollout and the train step are CUDA graphs. cublasLt allocates on the
// first use of a shape, and capture rejects that. Warm the shapes this
// batch will launch before BeginCapture, on the thread that captures.
static int drive_lt_push(int spec[][3], int n, int rows, int N, int K) {
    if (rows < 256 || n >= 8) {
        return n;
    }
    for (int i = 0; i < n; i++) {
        if (spec[i][0] == rows && spec[i][1] == N && spec[i][2] == K) {
            return n;
        }
    }
    spec[n][0] = rows;
    spec[n][1] = N;
    spec[n][2] = K;
    return n + 1;
}

static void drive_lt_prepare(int rows, int H, cudaStream_t stream) {
    if (rows < 256 || H < 1 || drive_lt_capturing(stream)) {
        return;
    }
    if (!drive_lt_ensure()) {
        return;
    }
    int spec[8][3];
    int n = 0;
    int road = rows * MAX_ROAD_SEGMENT_OBSERVATIONS;
    int part = rows * MAX_AGENTS;
    n = drive_lt_push(spec, n, rows, H, EGO_FEATURES);
    n = drive_lt_push(spec, n, rows, H, H);
    n = drive_lt_push(spec, n, road, H, DRIVE_ROAD_IN);
    n = drive_lt_push(spec, n, road, H, H);
    n = drive_lt_push(spec, n, part, H, PARTNER_FEATURES);
    n = drive_lt_push(spec, n, part, H, H);
    n = drive_lt_push(spec, n, rows * DRIVE_ATTN_Q, H, H);
    n = drive_lt_push(spec, n, rows, H, DRIVE_ATTN_Q * H);
    long nx = 8, nw = 8, ny = 8, nb = 8;
    for (int i = 0; i < n; i++) {
        long r = spec[i][0], N = spec[i][1], K = spec[i][2];
        if (r * K > nx) nx = r * K;
        if (N * K > nw) nw = N * K;
        if (r * N > ny) ny = r * N;
        if (N > nb) nb = N;
    }
    nx = (nx + 7) & ~7L;
    nw = (nw + 7) & ~7L;
    ny = (ny + 7) & ~7L;
    nb = (nb + 7) & ~7L;
    precision_t* buf = NULL;
    size_t count = (size_t)(nx + nw + ny + nb);
    if (cudaMalloc(&buf, count * sizeof(precision_t)) != cudaSuccess) {
        return;
    }
    cudaMemsetAsync(buf, 0, count * sizeof(precision_t), stream);
    precision_t* x = buf;
    precision_t* w = x + nx;
    precision_t* y = w + nw;
    precision_t* bias = y + ny;
    for (int i = 0; i < n; i++) {
        drive_lt_bias(x, w, bias, y, spec[i][0], spec[i][1], spec[i][2],
            stream);
    }
    cudaStreamSynchronize(stream);
    cudaFree(buf);
}

static void drive_project(Prec w, Prec b, precision_t* x, precision_t* y,
        int rows, int kin, int H, cudaStream_t stream) {
    if (drive_lt_bias(x, w.data, b.data, y, rows, H, kin, stream)) {
        return;
    }
    Prec xv = drive_view(x, rows, kin);
    Prec yv = drive_view(y, rows, H);
    puf_mm(&xv, &w, &yv, stream);
    drive_bias(y, b.data, rows, H, stream);
}

// h1 lands in buf0, ln in buf1, h2 in dest. dest must not alias those.
static void drive_mlp_fwd(Prec w1, Prec b1, Prec gamma, Prec beta,
        Prec w2, Prec b2, precision_t* x, precision_t* h1,
        precision_t* ln, precision_t* dest, int tokens, int feat, int H,
        cudaStream_t stream) {
    drive_project(w1, b1, x, h1, tokens, feat, H, stream);
    drive_ln_fwd(ln, h1, gamma.data, beta.data, tokens, H, stream);
    drive_silu(ln, tokens * H, stream);
    drive_project(w2, b2, ln, dest, tokens, H, H, stream);
}

static precision_t* drive_ego_ptr(precision_t* base, int H) {
    (void)H;
    return base;
}

static precision_t* drive_road_ptr(precision_t* base, int H) {
    return base + (long)DRIVE_ENC_CHUNK * H;
}

static precision_t* drive_part_ptr(precision_t* base, int H) {
    return base + (long)DRIVE_ENC_CHUNK
        * (1 + MAX_ROAD_SEGMENT_OBSERVATIONS) * H;
}

// dest is the token embedding. Attention reads it as both key and value.
static void drive_embed_group(Prec w1, Prec b1, Prec gamma, Prec beta,
        Prec w2, Prec b2, const precision_t* obs, int rows, int obs_size,
        int off, int slots, int raw, int one_hot, int H,
        precision_t* pack, precision_t* h1, precision_t* ln,
        precision_t* dest, cudaStream_t stream) {
    int feat = one_hot ? DRIVE_ROAD_IN : raw;
    int tokens = rows * slots;
    drive_pack(pack, obs, rows, obs_size, off, slots, raw, one_hot, stream);
    drive_mlp_fwd(w1, b1, gamma, beta, w2, b2, pack, h1, ln, dest,
        tokens, feat, H, stream);
}

static void drive_fill_kv(DriveEncWeights* ew, DriveEncActs* a,
        const precision_t* row, int rows, cudaStream_t stream) {
    int H = ew->hidden;
    int obs_size = OBS_SIZE;
    drive_project(ew->wq, ew->bq, ew->q_lat.data, a->qproj.data,
        DRIVE_ATTN_Q, H, H, stream);
    drive_mask(a->mask.data, row, rows, stream);
    drive_embed_group(ew->ego_w1, ew->ego_b1, ew->ego_lng, ew->ego_lnb,
        ew->ego_w2, ew->ego_b2, row, rows, obs_size, 0, 1, EGO_FEATURES, 0,
        H, a->pack.data, a->buf0.data, a->buf1.data,
        drive_ego_ptr(a->tok.data, H), stream);
    drive_embed_group(ew->road_w1, ew->road_b1, ew->road_lng, ew->road_lnb,
        ew->road_w2, ew->road_b2, row, rows, obs_size,
        EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS,
        MAX_ROAD_SEGMENT_OBSERVATIONS, ROAD_FEATURES, 1, H,
        a->pack.data, a->buf0.data, a->buf1.data,
        drive_road_ptr(a->tok.data, H), stream);
    drive_embed_group(ew->part_w1, ew->part_b1, ew->part_lng, ew->part_lnb,
        ew->part_w2, ew->part_b2, row, rows, obs_size, EGO_FEATURES,
        MAX_AGENTS, PARTNER_FEATURES, 0, H,
        a->pack.data, a->buf0.data, a->buf1.data,
        drive_part_ptr(a->tok.data, H), stream);
}

static void drive_xattn_launch(precision_t* ctx, DriveEncActs* a, int rows,
        int H, cudaStream_t stream) {
    int heads = drive_attn_heads(H);
    drive_xattn_fwd_kernel<<<rows, DRIVE_ATTN_THREADS,
        drive_attn_bytes(H, 1), stream>>>(ctx, a->qproj.data,
        drive_ego_ptr(a->tok.data, H),
        drive_road_ptr(a->tok.data, H),
        drive_part_ptr(a->tok.data, H),
        a->mask.data, H, heads);
}

// ctx -> out-proj -> residual with the raw queries -> LayerNorm.
// hid is [rows, Q, H], which is also [rows, Q * H].
static void drive_attn_hidden(DriveEncWeights* ew, DriveEncActs* a,
        int rows, cudaStream_t stream) {
    int H = ew->hidden;
    int nq = rows * DRIVE_ATTN_Q;
    drive_xattn_launch(a->ctx.data, a, rows, H, stream);
    drive_project(ew->wo, ew->bo, a->ctx.data, a->proj.data,
        nq, H, H, stream);
    int n = nq * H;
    drive_residual_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        a->pre.data, a->proj.data, ew->q_lat.data, n, H);
    drive_ln_fwd(a->hid.data, a->pre.data, ew->attn_lng.data,
        ew->attn_lnb.data, nq, H, stream);
}

static void drive_chunk_fwd(DriveEncWeights* ew, DriveEncActs* a,
        const precision_t* obs, int b0, int rows, cudaStream_t stream) {
    int H = ew->hidden;
    const precision_t* row = obs + (long)b0 * OBS_SIZE;
    drive_fill_kv(ew, a, row, rows, stream);
    drive_attn_hidden(ew, a, rows, stream);
    precision_t* y = a->out.data + (long)b0 * H;
    drive_project(ew->fc_w, ew->fc_b, a->hid.data, y,
        rows, DRIVE_ATTN_Q * H, H, stream);
    int n = rows * H;
    drive_tanh_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(y, n);
}

// Dense tower backward. dh is the grad of h2 and must outlive the w2 grad.
static void drive_tower_bwd(Prec w1, Prec b1, Prec gamma, Prec beta,
        Prec w2, Prec b2, Prec w1g, Prec w2g,
        Float acc_b1, Float acc_be, Float acc_g, Float acc_b2,
        precision_t* dh, const precision_t* obs, int rows, int obs_size,
        int off, int slots, int raw, int one_hot, int H,
        precision_t* pack, precision_t* buf0, precision_t* buf1,
        precision_t* buf2, precision_t* buf3, cudaStream_t stream) {
    int feat = one_hot ? DRIVE_ROAD_IN : raw;
    int tokens = rows * slots;
    drive_pack(pack, obs, rows, obs_size, off, slots, raw, one_hot, stream);
    drive_project(w1, b1, pack, buf0, tokens, feat, H, stream);
    drive_ln_fwd(buf1, buf0, gamma.data, beta.data, tokens, H, stream);
    // buf2 keeps the pre-activation. w2's input is silu(buf1).
    cudaMemcpyAsync(buf2, buf1, (size_t)tokens * H * sizeof(precision_t),
        cudaMemcpyDeviceToDevice, stream);
    drive_silu(buf1, tokens * H, stream);
    int partial_n = drive_partial_cap(H);
    drive_dw(w2g.data, dh, buf1, (float*)buf3, tokens, H, H,
        partial_n, stream);
    drive_colsum(acc_b2.data, dh, tokens, H, stream);
    Prec dhv = drive_view(dh, tokens, H);
    Prec dact = drive_view(buf1, tokens, H);
    puf_mm_nn(&dhv, &w2, &dact, stream);
    drive_silu_bwd(buf1, buf2, tokens * H, stream);
    drive_colsum(acc_be.data, buf1, tokens, H, stream);
    dim3 ln_block(32, DRIVE_LN_ROWS);
    dim3 ln_grid((tokens + DRIVE_LN_ROWS - 1) / DRIVE_LN_ROWS);
    drive_ln_bwd_kernel<<<ln_grid, ln_block, 0, stream>>>(
        buf2, buf3, buf1, buf0, gamma.data, H, tokens);
    drive_colsum(acc_g.data, buf3, tokens, H, stream);
    drive_dw(w1g.data, buf2, pack, (float*)buf3, tokens, H, feat,
        partial_n, stream);
    drive_colsum(acc_b1.data, buf2, tokens, H, stream);
}

static void drive_chunk_bwd(DriveEncWeights* ew, DriveEncActs* a,
        const precision_t* obs, const precision_t* grad,
        int b0, int rows, int reuse, cudaStream_t stream) {
    int H = ew->hidden;
    int partial_n = drive_partial_cap(H);
    const precision_t* row = obs + (long)b0 * OBS_SIZE;
    precision_t* y = a->out.data + (long)b0 * H;
    precision_t* d_fc = a->buf2.data;
    cudaMemcpyAsync(d_fc, grad + (long)b0 * H,
        (size_t)rows * H * sizeof(precision_t),
        cudaMemcpyDeviceToDevice, stream);
    drive_tanh_bwd_kernel<<<grid_size(rows * H), BLOCK_SIZE, 0, stream>>>(
        d_fc, y, rows * H);
    // reuse: scratch still holds this chunk's forward. Recompute only
    // when an earlier chunk was overwritten by a later one.
    if (!reuse) {
        drive_fill_kv(ew, a, row, rows, stream);
        drive_attn_hidden(ew, a, rows, stream);
    }
    drive_dw(a->fc_w_g.data, d_fc, a->hid.data, (float*)a->buf3.data,
        rows, H, DRIVE_ATTN_Q * H, partial_n, stream);
    drive_colsum(a->acc[DRIVE_ACC_FC_B].data, d_fc, rows, H, stream);
    Prec dfc = drive_view(d_fc, rows, H);
    Prec fcw = ew->fc_w;
    Prec dhid = drive_view(a->hid.data, rows, DRIVE_ATTN_Q * H);
    puf_mm_nn(&dfc, &fcw, &dhid, stream);
    int nq = rows * DRIVE_ATTN_Q;
    drive_colsum(a->acc[DRIVE_ACC_ATTN_BE].data, a->hid.data, nq, H, stream);
    dim3 ln_block(32, DRIVE_LN_ROWS);
    dim3 ln_grid((nq + DRIVE_LN_ROWS - 1) / DRIVE_LN_ROWS);
    drive_ln_bwd_kernel<<<ln_grid, ln_block, 0, stream>>>(
        a->proj.data, a->buf3.data, a->hid.data, a->pre.data,
        ew->attn_lng.data, H, nq);
    drive_colsum(a->acc[DRIVE_ACC_ATTN_G].data, a->buf3.data, nq, H, stream);
    drive_colsum(a->acc_lat.data, a->proj.data, rows, DRIVE_ATTN_Q * H, stream);
    drive_dw(a->wo_g.data, a->proj.data, a->ctx.data, (float*)a->buf3.data,
        nq, H, H, partial_n, stream);
    drive_colsum(a->acc[DRIVE_ACC_BO].data, a->proj.data, nq, H, stream);
    Prec dpre = drive_view(a->proj.data, nq, H);
    Prec dctx = drive_view(a->ctx.data, nq, H);
    puf_mm_nn(&dpre, &ew->wo, &dctx, stream);
    int heads = drive_attn_heads(H);
    // Writes dK+dV over the token embeddings. Tower backward reads that
    // as the gradient of the MLP output.
    drive_xattn_bwd_kernel<<<rows, DRIVE_ATTN_THREADS,
        drive_attn_bytes(H, 2), stream>>>(a->dq.data,
        a->ctx.data, a->qproj.data,
        drive_ego_ptr(a->tok.data, H),
        drive_road_ptr(a->tok.data, H),
        drive_part_ptr(a->tok.data, H),
        a->mask.data, H, heads);
    int nsum = DRIVE_ATTN_Q * H;
    drive_dq_sum_kernel<<<grid_size(nsum), BLOCK_SIZE, 0, stream>>>(
        a->dqsum.data, a->dq.data, rows, nsum);
    drive_dq_apply_kernel<<<grid_size(H * H), BLOCK_SIZE, 0, stream>>>(
        a->wq_g.data, a->acc[DRIVE_ACC_BQ].data, a->acc_lat.data,
        a->dqsum.data, ew->q_lat.data, ew->wq.data, H);
    drive_tower_bwd(ew->ego_w1, ew->ego_b1, ew->ego_lng, ew->ego_lnb,
        ew->ego_w2, ew->ego_b2, a->ego_w1_g, a->ego_w2_g,
        a->acc[DRIVE_ACC_EGO_B1], a->acc[DRIVE_ACC_EGO_BE],
        a->acc[DRIVE_ACC_EGO_G], a->acc[DRIVE_ACC_EGO_B2],
        drive_ego_ptr(a->tok.data, H), row, rows, OBS_SIZE, 0, 1,
        EGO_FEATURES, 0, H, a->pack.data, a->buf0.data, a->buf1.data,
        a->buf2.data, a->buf3.data, stream);
    drive_tower_bwd(ew->part_w1, ew->part_b1, ew->part_lng, ew->part_lnb,
        ew->part_w2, ew->part_b2, a->part_w1_g, a->part_w2_g,
        a->acc[DRIVE_ACC_PART_B1], a->acc[DRIVE_ACC_PART_BE],
        a->acc[DRIVE_ACC_PART_G], a->acc[DRIVE_ACC_PART_B2],
        drive_part_ptr(a->tok.data, H), row, rows, OBS_SIZE, EGO_FEATURES,
        MAX_AGENTS, PARTNER_FEATURES, 0, H, a->pack.data, a->buf0.data,
        a->buf1.data, a->buf2.data, a->buf3.data, stream);
    drive_tower_bwd(ew->road_w1, ew->road_b1, ew->road_lng, ew->road_lnb,
        ew->road_w2, ew->road_b2, a->road_w1_g, a->road_w2_g,
        a->acc[DRIVE_ACC_ROAD_B1], a->acc[DRIVE_ACC_ROAD_BE],
        a->acc[DRIVE_ACC_ROAD_G], a->acc[DRIVE_ACC_ROAD_B2],
        drive_road_ptr(a->tok.data, H), row, rows, OBS_SIZE,
        EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS,
        MAX_ROAD_SEGMENT_OBSERVATIONS, ROAD_FEATURES, 1, H,
        a->pack.data, a->buf0.data, a->buf1.data, a->buf2.data,
        a->buf3.data, stream);
}

static int drive_rows(Prec input) {
    int nd = ndim(input.shape);
    assert(nd >= 2);
    assert(input.shape[nd - 1] == OBS_SIZE);
    int rows = 1;
    for (int i = 0; i < nd - 1; i++) {
        rows *= input.shape[i];
    }
    return rows;
}

static Prec drive_enc_forward(void* w, void* activations, Prec input,
        cudaStream_t stream) {
    DriveEncWeights* ew = (DriveEncWeights*)w;
    DriveEncActs* a = (DriveEncActs*)activations;
    int B = drive_rows(input);
    assert(a->out.shape[0] == B);
    assert(a->out.shape[1] == ew->hidden);
    const precision_t* obs = input.data;
    if (a->saved_obs.data) {
        puf_copy(&a->saved_obs, &input, stream);
        obs = a->saved_obs.data;
    }
    for (int b0 = 0; b0 < B; b0 += DRIVE_ENC_CHUNK) {
        int rows = B - b0;
        if (rows > DRIVE_ENC_CHUNK) {
            rows = DRIVE_ENC_CHUNK;
        }
        drive_chunk_fwd(ew, a, obs, b0, rows, stream);
    }
    return a->out;
}

static void drive_store_acc(Prec* dst, Float* src, cudaStream_t stream) {
    int n = (int)numel(src->shape);
    assert(n == (int)numel(dst->shape));
    drive_store_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        dst->data, src->data, n);
}

static void drive_enc_backward(void* w, void* activations, Prec grad,
        cudaStream_t stream) {
    DriveEncWeights* ew = (DriveEncWeights*)w;
    DriveEncActs* a = (DriveEncActs*)activations;
    assert(ndim(grad.shape) == 2);
    assert(grad.shape[1] == ew->hidden);
    int B = (int)grad.shape[0];
    assert(a->saved_obs.data);
    assert(a->saved_obs.shape[0] == B);
    Prec* grads[] = {
        &a->ego_w1_g, &a->ego_b1_g, &a->ego_lng_g, &a->ego_lnb_g,
        &a->ego_w2_g, &a->ego_b2_g,
        &a->part_w1_g, &a->part_b1_g, &a->part_lng_g, &a->part_lnb_g,
        &a->part_w2_g, &a->part_b2_g,
        &a->road_w1_g, &a->road_b1_g, &a->road_lng_g, &a->road_lnb_g,
        &a->road_w2_g, &a->road_b2_g,
        &a->q_lat_g, &a->wq_g, &a->bq_g,
        &a->wo_g, &a->bo_g,
        &a->attn_lng_g, &a->attn_lnb_g, &a->fc_w_g, &a->fc_b_g,
    };
    int ngrad = (int)(sizeof(grads) / sizeof(grads[0]));
    for (int i = 0; i < ngrad; i++) {
        drive_zero_prec(grads[i], stream);
    }
    for (int i = 0; i < DRIVE_ACC_N; i++) {
        cudaMemsetAsync(a->acc[i].data, 0,
            numel(a->acc[i].shape) * sizeof(float), stream);
    }
    cudaMemsetAsync(a->acc_lat.data, 0,
        numel(a->acc_lat.shape) * sizeof(float), stream);
    int reuse = (B <= DRIVE_ENC_CHUNK);
    for (int b0 = 0; b0 < B; b0 += DRIVE_ENC_CHUNK) {
        int rows = B - b0;
        if (rows > DRIVE_ENC_CHUNK) {
            rows = DRIVE_ENC_CHUNK;
        }
        drive_chunk_bwd(ew, a, a->saved_obs.data, grad.data,
            b0, rows, reuse, stream);
    }
    drive_store_acc(&a->ego_b1_g, &a->acc[DRIVE_ACC_EGO_B1], stream);
    drive_store_acc(&a->ego_lnb_g, &a->acc[DRIVE_ACC_EGO_BE], stream);
    drive_store_acc(&a->ego_lng_g, &a->acc[DRIVE_ACC_EGO_G], stream);
    drive_store_acc(&a->ego_b2_g, &a->acc[DRIVE_ACC_EGO_B2], stream);
    drive_store_acc(&a->part_b1_g, &a->acc[DRIVE_ACC_PART_B1], stream);
    drive_store_acc(&a->part_lnb_g, &a->acc[DRIVE_ACC_PART_BE], stream);
    drive_store_acc(&a->part_lng_g, &a->acc[DRIVE_ACC_PART_G], stream);
    drive_store_acc(&a->part_b2_g, &a->acc[DRIVE_ACC_PART_B2], stream);
    drive_store_acc(&a->road_b1_g, &a->acc[DRIVE_ACC_ROAD_B1], stream);
    drive_store_acc(&a->road_lnb_g, &a->acc[DRIVE_ACC_ROAD_BE], stream);
    drive_store_acc(&a->road_lng_g, &a->acc[DRIVE_ACC_ROAD_G], stream);
    drive_store_acc(&a->road_b2_g, &a->acc[DRIVE_ACC_ROAD_B2], stream);
    drive_store_acc(&a->bq_g, &a->acc[DRIVE_ACC_BQ], stream);
    drive_store_acc(&a->bo_g, &a->acc[DRIVE_ACC_BO], stream);
    drive_store_acc(&a->attn_lnb_g, &a->acc[DRIVE_ACC_ATTN_BE], stream);
    drive_store_acc(&a->attn_lng_g, &a->acc[DRIVE_ACC_ATTN_G], stream);
    drive_store_acc(&a->fc_b_g, &a->acc[DRIVE_ACC_FC_B], stream);
    drive_store_acc(&a->q_lat_g, &a->acc_lat, stream);
}

// Parameter grads this backward just wrote. Same tensors it zeros.
static int drive_enc_grad_spans(void* activations,
        precision_t** data, int* n, int cap) {
    DriveEncActs* a = (DriveEncActs*)activations;
    Prec* grads[] = {
        &a->ego_w1_g, &a->ego_b1_g, &a->ego_lng_g, &a->ego_lnb_g,
        &a->ego_w2_g, &a->ego_b2_g,
        &a->part_w1_g, &a->part_b1_g, &a->part_lng_g, &a->part_lnb_g,
        &a->part_w2_g, &a->part_b2_g,
        &a->road_w1_g, &a->road_b1_g, &a->road_lng_g, &a->road_lnb_g,
        &a->road_w2_g, &a->road_b2_g,
        &a->q_lat_g, &a->wq_g, &a->bq_g,
        &a->wo_g, &a->bo_g,
        &a->attn_lng_g, &a->attn_lnb_g, &a->fc_w_g, &a->fc_b_g,
    };
    int ngrad = (int)(sizeof(grads) / sizeof(grads[0]));
    int c = 0;
    for (int i = 0; i < ngrad && c < cap; i++) {
        data[c] = grads[i]->data;
        n[c] = (int)numel(grads[i]->shape);
        c++;
    }
    return c;
}

static void drive_init_tower(Prec w1, Prec b1, Prec gamma, Prec beta,
        Prec w2, Prec b2, ulong* seed, cudaStream_t stream) {
    puf_kaiming_init(&w1, sqrtf(2.0f), (*seed)++, stream);
    puf_kaiming_init(&w2, 1.0f, (*seed)++, stream);
    int H = (int)w2.shape[0];
    drive_fill(b1.data, 0.0f, H, stream);
    drive_fill(b2.data, 0.0f, H, stream);
    drive_fill(beta.data, 0.0f, H, stream);
    drive_fill(gamma.data, 1.0f, H, stream);
}

static void drive_init_square(Prec w, Prec b, ulong* seed,
        cudaStream_t stream) {
    puf_kaiming_init(&w, 1.0f, (*seed)++, stream);
    drive_fill(b.data, 0.0f, (int)b.shape[0], stream);
}

static void drive_enc_init(void* w, ulong* seed, cudaStream_t stream) {
    DriveEncWeights* ew = (DriveEncWeights*)w;
    int H = ew->hidden;
    drive_init_tower(ew->ego_w1, ew->ego_b1, ew->ego_lng, ew->ego_lnb,
        ew->ego_w2, ew->ego_b2, seed, stream);
    drive_init_tower(ew->part_w1, ew->part_b1, ew->part_lng, ew->part_lnb,
        ew->part_w2, ew->part_b2, seed, stream);
    drive_init_tower(ew->road_w1, ew->road_b1, ew->road_lng, ew->road_lnb,
        ew->road_w2, ew->road_b2, seed, stream);
    puf_kaiming_init(&ew->q_lat, 1.0f, (*seed)++, stream);
    drive_init_square(ew->wq, ew->bq, seed, stream);
    drive_init_square(ew->wo, ew->bo, seed, stream);
    drive_fill(ew->attn_lnb.data, 0.0f, H, stream);
    drive_fill(ew->attn_lng.data, 1.0f, H, stream);
    puf_kaiming_init(&ew->fc_w, 1.0f, (*seed)++, stream);
    drive_fill(ew->fc_b.data, 0.0f, H, stream);
}

static void drive_reg_vec(Allocator* alloc, Prec* t, int n) {
    *t = (Prec){.shape = {n}};
    alloc_register(alloc, t);
}

static void drive_reg_mat(Allocator* alloc, Prec* t, int rows, int cols) {
    *t = (Prec){.shape = {rows, cols}};
    alloc_register(alloc, t);
}

// Parameter order is the checkpoint order. CPU load walks the same list.
// Each tensor's length is a multiple of 8 bf16 values when hidden is,
// which keeps the flat grad buffer lined up with these allocations.
static void drive_reg_tower(Allocator* alloc, int H, int feat,
        Prec* w1, Prec* b1, Prec* gamma, Prec* beta, Prec* w2, Prec* b2) {
    drive_reg_mat(alloc, w1, H, feat);
    drive_reg_vec(alloc, b1, H);
    drive_reg_vec(alloc, gamma, H);
    drive_reg_vec(alloc, beta, H);
    drive_reg_mat(alloc, w2, H, H);
    drive_reg_vec(alloc, b2, H);
}

static void drive_reg_params(void* w, Allocator* alloc) {
    DriveEncWeights* ew = (DriveEncWeights*)w;
    int H = ew->hidden;
    assert(H > 0 && H % 8 == 0);
    assert(drive_attn_heads(H) <= DRIVE_ATTN_HEADS_MAX);
    assert(H % drive_attn_heads(H) == 0);
    assert(H <= 4096);
    assert(DRIVE_ROAD_CLASSES == 7);
    assert(DRIVE_ATTN_S == 1 + MAX_ROAD_SEGMENT_OBSERVATIONS + MAX_AGENTS);
    assert(EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS
        + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS == OBS_SIZE);
    drive_reg_tower(alloc, H, EGO_FEATURES,
        &ew->ego_w1, &ew->ego_b1, &ew->ego_lng, &ew->ego_lnb,
        &ew->ego_w2, &ew->ego_b2);
    drive_reg_tower(alloc, H, PARTNER_FEATURES,
        &ew->part_w1, &ew->part_b1, &ew->part_lng, &ew->part_lnb,
        &ew->part_w2, &ew->part_b2);
    drive_reg_tower(alloc, H, DRIVE_ROAD_IN,
        &ew->road_w1, &ew->road_b1, &ew->road_lng, &ew->road_lnb,
        &ew->road_w2, &ew->road_b2);
    drive_reg_mat(alloc, &ew->q_lat, DRIVE_ATTN_Q, H);
    drive_reg_mat(alloc, &ew->wq, H, H);
    drive_reg_vec(alloc, &ew->bq, H);
    drive_reg_mat(alloc, &ew->wo, H, H);
    drive_reg_vec(alloc, &ew->bo, H);
    drive_reg_vec(alloc, &ew->attn_lng, H);
    drive_reg_vec(alloc, &ew->attn_lnb, H);
    drive_reg_mat(alloc, &ew->fc_w, H, DRIVE_ATTN_Q * H);
    drive_reg_vec(alloc, &ew->fc_b, H);
}

static void drive_reg_scratch(DriveEncActs* a, Allocator* alloc, int H) {
    int road_tok = DRIVE_ENC_CHUNK * MAX_ROAD_SEGMENT_OBSERVATIONS;
    int seq_tok = DRIVE_ENC_CHUNK * DRIVE_ATTN_S;
    drive_reg_mat(alloc, &a->pack, road_tok, DRIVE_ROAD_IN);
    drive_reg_mat(alloc, &a->buf0, road_tok, H);
    drive_reg_mat(alloc, &a->buf1, road_tok, H);
    drive_reg_mat(alloc, &a->buf2, road_tok, H);
    drive_reg_mat(alloc, &a->buf3, road_tok, H);
    drive_reg_mat(alloc, &a->tok, seq_tok, H);
    drive_reg_mat(alloc, &a->qproj, DRIVE_ATTN_Q, H);
    drive_reg_mat(alloc, &a->ctx, DRIVE_ENC_CHUNK * DRIVE_ATTN_Q, H);
    drive_reg_mat(alloc, &a->proj, DRIVE_ENC_CHUNK * DRIVE_ATTN_Q, H);
    drive_reg_mat(alloc, &a->pre, DRIVE_ENC_CHUNK * DRIVE_ATTN_Q, H);
    drive_reg_mat(alloc, &a->hid, DRIVE_ENC_CHUNK * DRIVE_ATTN_Q, H);
    drive_reg_mat(alloc, &a->dq, DRIVE_ENC_CHUNK * DRIVE_ATTN_Q, H);
    a->mask = (Float){.shape = {DRIVE_ENC_CHUNK * DRIVE_ATTN_S}};
    alloc_register(alloc, &a->mask);
}

static void drive_enc_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    DriveEncWeights* ew = (DriveEncWeights*)w;
    DriveEncActs* a = (DriveEncActs*)activations;
    int H = ew->hidden;
    drive_reg_mat(acts, &a->out, B_TT, H);
    drive_reg_mat(acts, &a->saved_obs, B_TT, OBS_SIZE);
    drive_reg_scratch(a, acts, H);
    a->dqsum = (Float){.shape = {DRIVE_ATTN_Q, H}};
    a->acc_lat = (Float){.shape = {DRIVE_ATTN_Q, H}};
    alloc_register(acts, &a->dqsum);
    alloc_register(acts, &a->acc_lat);
    for (int i = 0; i < DRIVE_ACC_N; i++) {
        a->acc[i] = (Float){.shape = {H}};
        alloc_register(acts, &a->acc[i]);
    }
    drive_reg_tower(grads, H, EGO_FEATURES,
        &a->ego_w1_g, &a->ego_b1_g, &a->ego_lng_g, &a->ego_lnb_g,
        &a->ego_w2_g, &a->ego_b2_g);
    drive_reg_tower(grads, H, PARTNER_FEATURES,
        &a->part_w1_g, &a->part_b1_g, &a->part_lng_g, &a->part_lnb_g,
        &a->part_w2_g, &a->part_b2_g);
    drive_reg_tower(grads, H, DRIVE_ROAD_IN,
        &a->road_w1_g, &a->road_b1_g, &a->road_lng_g, &a->road_lnb_g,
        &a->road_w2_g, &a->road_b2_g);
    drive_reg_mat(grads, &a->q_lat_g, DRIVE_ATTN_Q, H);
    drive_reg_mat(grads, &a->wq_g, H, H);
    drive_reg_vec(grads, &a->bq_g, H);
    drive_reg_mat(grads, &a->wo_g, H, H);
    drive_reg_vec(grads, &a->bo_g, H);
    drive_reg_vec(grads, &a->attn_lng_g, H);
    drive_reg_vec(grads, &a->attn_lnb_g, H);
    drive_reg_mat(grads, &a->fc_w_g, H, DRIVE_ATTN_Q * H);
    drive_reg_vec(grads, &a->fc_b_g, H);
}

static void drive_enc_reg_rollout(void* w, void* activations,
        Allocator* alloc, int B) {
    DriveEncWeights* ew = (DriveEncWeights*)w;
    DriveEncActs* a = (DriveEncActs*)activations;
    int H = ew->hidden;
    drive_reg_mat(alloc, &a->out, B, H);
    drive_reg_scratch(a, alloc, H);
}

static void* drive_enc_create(void* self) {
    Encoder* e = (Encoder*)self;
    DriveEncWeights* ew = (DriveEncWeights*)calloc(1, sizeof(DriveEncWeights));
    ew->hidden = e->out_dim;
    return ew;
}

static void create_drive_encoder(Encoder* enc) {
    int in_dim = enc->in_dim;
    int out_dim = enc->out_dim;
    assert(in_dim == OBS_SIZE);
    assert(out_dim > 0 && out_dim % 8 == 0);
    assert(out_dim % drive_attn_heads(out_dim) == 0);
    // H=128 backward needs about 54 KB. The default dynamic cap is 48 KB.
    drive_attn_allow_smem((const void*)drive_xattn_fwd_kernel);
    drive_attn_allow_smem((const void*)drive_xattn_bwd_kernel);
    *enc = (Encoder){
        .forward = drive_enc_forward,
        .backward = drive_enc_backward,
        .init_weights = drive_enc_init,
        .reg_params = drive_reg_params,
        .reg_train = drive_enc_reg_train,
        .reg_rollout = drive_enc_reg_rollout,
        .create_weights = drive_enc_create,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .activation_size = sizeof(DriveEncActs),
    };
}
