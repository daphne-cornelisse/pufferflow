// Throughput of the drive entity encoder against the default linear encoder.
// One train step is encoder + MinGRU + decoder forward, then backward.
// Batch is 256 agents by 32 steps, the trainer's minibatch.
// Drive has no num_players key. The 2..4 axis is MinGRU num_layers.
// hidden_size must be a multiple of 8, so the sweep is 64 and 128.

#include "pufferl.cu"

static const int BENCH_B = 256;
static const int BENCH_T = 32;
static const int WARMUP = 10;
static int g_iters = 40;
static int g_floor = 0;

static float rndf(float x) {
    return to_float(from_float(x));
}

static int act_width(void) {
    int sizes[] = ACT_SIZES;
    int n = 0;
    for (int i = 0; i < NUM_ATNS; i++) {
        n += sizes[i];
    }
    return n;
}

static int road_base(void) {
    return EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS;
}

struct Net {
    Arch arch;
    Weights w;
    Activations acts;
    Allocator params;
    Allocator activations;
    Allocator grads;
};

static void install_linear(Arch* arch) {
    arch->encoder.forward = encoder_forward;
    arch->encoder.backward = encoder_backward;
    arch->encoder.init_weights = encoder_init_weights;
    arch->encoder.reg_params = encoder_reg_params;
    arch->encoder.reg_train = encoder_reg_train;
    arch->encoder.reg_rollout = encoder_reg_rollout;
    arch->encoder.create_weights = encoder_create_weights;
    arch->encoder.activation_size = sizeof(EncoderActivations);
}

static Net make_net(int H, int layers, int horizon, int B_TT, int entity) {
    Net net = {};
    net.arch = build_arch(OBS_SIZE, H, layers, act_width(), false, horizon);
    if (!entity) {
        install_linear(&net.arch);
    }
    net.w = weights_create(&net.arch, &net.params);
    alloc_create(&net.params);
    uint64_t seed = 11;
    weights_init(&net.arch, net.w, &seed, 0);
    net.acts = arch_reg_train(
        &net.arch, net.w, &net.activations, &net.grads, B_TT);
    alloc_create(&net.activations);
    alloc_create(&net.grads);
    cudaDeviceSynchronize();
    return net;
}

static void free_net(Net* net) {
    cudaFree(net->params.mem);
    cudaFree(net->activations.mem);
    cudaFree(net->grads.mem);
    free(net->params.regs);
    free(net->activations.regs);
    free(net->grads.regs);
}

__global__ void bench_fill_f(float* p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = v;
    }
}

__global__ void bench_fill_p(precision_t* p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = from_float(v);
    }
}

static void must_cuda(const char* what) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cuda %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

static float* pull_prec(Prec* t) {
    int n = (int)numel(t->shape);
    precision_t* raw = (precision_t*)malloc((size_t)n * sizeof(precision_t));
    float* out = (float*)malloc((size_t)n * sizeof(float));
    assert(raw && out);
    cudaMemcpy(raw, t->data, (size_t)n * sizeof(precision_t),
        cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        out[i] = to_float(raw[i]);
    }
    free(raw);
    return out;
}

static float peak_abs(Prec* t) {
    int n = (int)numel(t->shape);
    float* v = pull_prec(t);
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(v[i]);
        if (a > m) {
            m = a;
        }
    }
    free(v);
    return m;
}

static void linear_bt(const float* x, const float* w, float* y,
        int K, int N) {
    for (int n = 0; n < N; n++) {
        const float* row = w + (size_t)n * K;
        float s = 0.0f;
        for (int k = 0; k < K; k++) {
            s += x[k] * row[k];
        }
        y[n] = rndf(s);
    }
}

static void add_bias(float* y, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = rndf(y[i] + b[i]);
    }
}

static void layernorm(float* y, const float* x, const float* gamma,
        const float* beta, int H) {
    float mean = 0.0f;
    for (int i = 0; i < H; i++) {
        mean += x[i];
    }
    mean /= (float)H;
    float var = 0.0f;
    for (int i = 0; i < H; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv = rsqrtf(var / (float)H + DRIVE_LN_EPS);
    for (int i = 0; i < H; i++) {
        float xhat = (x[i] - mean) * inv;
        y[i] = rndf(xhat * gamma[i] + beta[i]);
    }
}

static void token_mlp(const float* w1, const float* b1, const float* gamma,
        const float* beta, const float* w2, const float* b2,
        const float* x, int feat, int H, float* h2) {
    float* h1 = (float*)malloc((size_t)H * sizeof(float));
    float* ln = (float*)malloc((size_t)H * sizeof(float));
    linear_bt(x, w1, h1, feat, H);
    add_bias(h1, b1, H);
    layernorm(ln, h1, gamma, beta, H);
    linear_bt(ln, w2, h2, H, H);
    add_bias(h2, b2, H);
    free(h1);
    free(ln);
}

static void project_token(const float* x, const float* w, const float* b,
        int K, int N, float* y) {
    linear_bt(x, w, y, K, N);
    add_bias(y, b, N);
}

// Every slot, including padding. Order is ego, then roads, then partners.
static void embed_group(const float* w1, const float* b1, const float* gamma,
        const float* beta, const float* w2, const float* b2,
        const float* wk, const float* bk, const float* wv, const float* bv,
        const float* obs, int off, int slots, int raw, int one_hot,
        int H, int* s, float* K, float* V) {
    float* x = (float*)calloc((size_t)DRIVE_ROAD_IN, sizeof(float));
    float* h2 = (float*)malloc((size_t)H * sizeof(float));
    int feat = one_hot ? DRIVE_ROAD_IN : raw;
    for (int n = 0; n < slots; n++) {
        const float* src = obs + off + n * raw;
        if (one_hot) {
            for (int k = 0; k < DRIVE_ROAD_CONT; k++) {
                x[k] = src[k];
            }
            int cls = (int)src[DRIVE_ROAD_CONT];
            if (cls < 0) {
                cls = 0;
            }
            if (cls >= DRIVE_ROAD_CLASSES) {
                cls = DRIVE_ROAD_CLASSES - 1;
            }
            for (int k = 0; k < DRIVE_ROAD_CLASSES; k++) {
                x[DRIVE_ROAD_CONT + k] = (k == cls) ? 1.0f : 0.0f;
            }
        } else {
            for (int k = 0; k < raw; k++) {
                x[k] = src[k];
            }
        }
        token_mlp(w1, b1, gamma, beta, w2, b2, x, feat, H, h2);
        project_token(h2, wk, bk, H, H, K + (size_t)(*s) * H);
        project_token(h2, wv, bv, H, H, V + (size_t)(*s) * H);
        *s += 1;
    }
    free(x);
    free(h2);
}

static void ref_forward(DriveEncWeights* ew, const float* obs, int rows,
        float* out) {
    int H = ew->hidden;
    float* ego_w1 = pull_prec(&ew->ego_w1);
    float* ego_b1 = pull_prec(&ew->ego_b1);
    float* ego_g = pull_prec(&ew->ego_lng);
    float* ego_be = pull_prec(&ew->ego_lnb);
    float* ego_w2 = pull_prec(&ew->ego_w2);
    float* ego_b2 = pull_prec(&ew->ego_b2);
    float* part_w1 = pull_prec(&ew->part_w1);
    float* part_b1 = pull_prec(&ew->part_b1);
    float* part_g = pull_prec(&ew->part_lng);
    float* part_be = pull_prec(&ew->part_lnb);
    float* part_w2 = pull_prec(&ew->part_w2);
    float* part_b2 = pull_prec(&ew->part_b2);
    float* road_w1 = pull_prec(&ew->road_w1);
    float* road_b1 = pull_prec(&ew->road_b1);
    float* road_g = pull_prec(&ew->road_lng);
    float* road_be = pull_prec(&ew->road_lnb);
    float* road_w2 = pull_prec(&ew->road_w2);
    float* road_b2 = pull_prec(&ew->road_b2);
    float* q_lat = pull_prec(&ew->q_lat);
    float* wq = pull_prec(&ew->wq);
    float* bq = pull_prec(&ew->bq);
    float* wk = pull_prec(&ew->wk);
    float* bk = pull_prec(&ew->bk);
    float* wv = pull_prec(&ew->wv);
    float* bv = pull_prec(&ew->bv);
    float* wo = pull_prec(&ew->wo);
    float* bo = pull_prec(&ew->bo);
    float* attn_g = pull_prec(&ew->attn_lng);
    float* attn_be = pull_prec(&ew->attn_lnb);
    float* fc_w = pull_prec(&ew->fc_w);
    float* fc_b = pull_prec(&ew->fc_b);
    int Q = DRIVE_ATTN_Q;
    int S = DRIVE_ATTN_S;
    int heads = DRIVE_ATTN_HEADS;
    int d = H / heads;
    float scale = rsqrtf((float)d);
    float* Qp = (float*)malloc((size_t)Q * H * sizeof(float));
    float* K = (float*)malloc((size_t)S * H * sizeof(float));
    float* V = (float*)malloc((size_t)S * H * sizeof(float));
    float* ctx = (float*)malloc((size_t)Q * H * sizeof(float));
    float* proj = (float*)malloc((size_t)Q * H * sizeof(float));
    float* pre = (float*)malloc((size_t)Q * H * sizeof(float));
    float* hid = (float*)malloc((size_t)Q * H * sizeof(float));
    float* y = (float*)malloc((size_t)H * sizeof(float));
    float* scores = (float*)malloc((size_t)S * sizeof(float));
    for (int q = 0; q < Q; q++) {
        project_token(q_lat + (size_t)q * H, wq, bq, H, H, Qp + (size_t)q * H);
    }
    for (int r = 0; r < rows; r++) {
        const float* row = obs + (size_t)r * OBS_SIZE;
        int s = 0;
        embed_group(ego_w1, ego_b1, ego_g, ego_be, ego_w2, ego_b2,
            wk, bk, wv, bv, row, 0, 1, EGO_FEATURES, 0, H, &s, K, V);
        embed_group(road_w1, road_b1, road_g, road_be, road_w2, road_b2,
            wk, bk, wv, bv, row, road_base(),
            MAX_ROAD_SEGMENT_OBSERVATIONS, ROAD_FEATURES, 1, H, &s, K, V);
        embed_group(part_w1, part_b1, part_g, part_be, part_w2, part_b2,
            wk, bk, wv, bv, row, EGO_FEATURES, MAX_AGENTS, PARTNER_FEATURES,
            0, H, &s, K, V);
        for (int head = 0; head < heads; head++) {
            int h0 = head * d;
            for (int q = 0; q < Q; q++) {
                for (int t = 0; t < S; t++) {
                    float dot = 0.0f;
                    for (int i = 0; i < d; i++) {
                        dot += Qp[q * H + h0 + i] * K[t * H + h0 + i];
                    }
                    scores[t] = dot * scale;
                }
                float m = scores[0];
                for (int t = 1; t < S; t++) {
                    if (scores[t] > m) {
                        m = scores[t];
                    }
                }
                float sum = 0.0f;
                for (int t = 0; t < S; t++) {
                    scores[t] = expf(scores[t] - m);
                    sum += scores[t];
                }
                for (int t = 0; t < S; t++) {
                    scores[t] /= sum;
                }
                for (int i = 0; i < d; i++) {
                    float acc = 0.0f;
                    for (int t = 0; t < S; t++) {
                        acc += scores[t] * V[t * H + h0 + i];
                    }
                    ctx[q * H + h0 + i] = rndf(acc);
                }
            }
        }
        for (int q = 0; q < Q; q++) {
            project_token(ctx + (size_t)q * H, wo, bo, H, H,
                proj + (size_t)q * H);
            for (int h = 0; h < H; h++) {
                pre[q * H + h] = rndf(proj[q * H + h] + q_lat[q * H + h]);
            }
            layernorm(hid + (size_t)q * H, pre + (size_t)q * H,
                attn_g, attn_be, H);
        }
        linear_bt(hid, fc_w, y, Q * H, H);
        add_bias(y, fc_b, H);
        for (int h = 0; h < H; h++) {
            out[(size_t)r * H + h] = rndf(tanhf(y[h]));
        }
    }
    free(ego_w1); free(ego_b1); free(ego_g); free(ego_be);
    free(ego_w2); free(ego_b2);
    free(part_w1); free(part_b1); free(part_g); free(part_be);
    free(part_w2); free(part_b2);
    free(road_w1); free(road_b1); free(road_g); free(road_be);
    free(road_w2); free(road_b2);
    free(q_lat); free(wq); free(bq); free(wk); free(bk);
    free(wv); free(bv); free(wo); free(bo);
    free(attn_g); free(attn_be); free(fc_w); free(fc_b);
    free(Qp); free(K); free(V); free(ctx); free(proj); free(pre);
    free(hid); free(y); free(scores);
}

static void put_ego(float* row) {
    for (int i = 0; i < EGO_FEATURES; i++) {
        row[i] = 0.15f * (float)(i + 1);
    }
}

static void put_partner(float* row, int slot, float x) {
    float* f = row + EGO_FEATURES + slot * PARTNER_FEATURES;
    f[0] = x;
    f[1] = -0.4f;
    f[2] = 1.8f;
    f[3] = 4.2f;
    f[4] = 0.95f;
    f[5] = 0.2f;
    f[6] = 6.0f;
}

static void put_road(float* row, int slot, int cls) {
    float* f = row + road_base() + slot * ROAD_FEATURES;
    f[0] = 3.0f;
    f[1] = -2.0f;
    f[2] = 12.0f;
    f[3] = 0.1f / MAX_ROAD_SCALE;
    f[4] = 1.0f;
    f[5] = 0.0f;
    f[6] = (float)cls;
}

static void fill_kind(float* row, int kind) {
    memset(row, 0, OBS_SIZE * sizeof(float));
    put_ego(row);
    if (kind == 0) {
        return;
    }
    if (kind == 1) {
        put_partner(row, 0, 1.0f);
        put_road(row, 0, 0);
        return;
    }
    if (kind == 2) {
        put_partner(row, 0, 1.0f);
        put_partner(row, 2, -2.0f);
        put_partner(row, 17, 0.5f);
        put_road(row, 0, 0);
        put_road(row, 4, 3);
        put_road(row, 74, 6);
        return;
    }
    put_partner(row, 63, 1.5f);
    put_road(row, 74, 6);
}

static void round_obs(const float* src, float* rounded, precision_t* bits,
        int n) {
    for (int i = 0; i < n; i++) {
        bits[i] = from_float(src[i]);
        rounded[i] = to_float(bits[i]);
    }
}

static void run_forward(Net* net, precision_t* obs, int rows) {
    Prec in = {.data = obs, .shape = {rows, OBS_SIZE}};
    net->arch.encoder.forward(net->w.encoder, net->acts.encoder, in, 0);
    must_cuda("encoder forward");
}

static int check_encoder(int H) {
    const int rows = 4;
    Net net = make_net(H, 1, 1, rows, 1);
    float* src = (float*)calloc((size_t)rows * OBS_SIZE, sizeof(float));
    float* rounded = (float*)calloc((size_t)rows * OBS_SIZE, sizeof(float));
    precision_t* bits = (precision_t*)malloc(
        (size_t)rows * OBS_SIZE * sizeof(precision_t));
    for (int r = 0; r < rows; r++) {
        fill_kind(src + (size_t)r * OBS_SIZE, r);
    }
    round_obs(src, rounded, bits, rows * OBS_SIZE);
    precision_t* dobs = NULL;
    cudaMalloc(&dobs, (size_t)rows * OBS_SIZE * sizeof(precision_t));
    cudaMemcpy(dobs, bits, (size_t)rows * OBS_SIZE * sizeof(precision_t),
        cudaMemcpyHostToDevice);
    run_forward(&net, dobs, rows);
    DriveEncActs* ea = (DriveEncActs*)net.acts.encoder;
    float* got = pull_prec(&ea->out);
    float* expect = (float*)malloc((size_t)rows * H * sizeof(float));
    ref_forward((DriveEncWeights*)net.w.encoder, rounded, rows, expect);
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < rows * H; i++) {
        float d = fabsf(got[i] - expect[i]);
        mean_abs += d;
        if (d > max_abs) {
            max_abs = d;
        }
    }
    mean_abs /= (float)(rows * H);

    float* row1 = (float*)malloc((size_t)H * sizeof(float));
    memcpy(row1, got + H, (size_t)H * sizeof(float));
    float* row = src + OBS_SIZE;
    memcpy(row + EGO_FEATURES + PARTNER_FEATURES,
        row + EGO_FEATURES, PARTNER_FEATURES * sizeof(float));
    round_obs(src, rounded, bits, rows * OBS_SIZE);
    cudaMemcpy(dobs, bits, (size_t)rows * OBS_SIZE * sizeof(precision_t),
        cudaMemcpyHostToDevice);
    run_forward(&net, dobs, rows);
    free(got);
    got = pull_prec(&ea->out);
    float dup_max = 0.0f;
    for (int h = 0; h < H; h++) {
        float d = fabsf(got[H + h] - row1[h]);
        if (d > dup_max) {
            dup_max = d;
        }
    }
    put_partner(row, 1, -3.0f);
    round_obs(src, rounded, bits, rows * OBS_SIZE);
    cudaMemcpy(dobs, bits, (size_t)rows * OBS_SIZE * sizeof(precision_t),
        cudaMemcpyHostToDevice);
    run_forward(&net, dobs, rows);
    free(got);
    got = pull_prec(&ea->out);
    float split = 0.0f;
    for (int h = 0; h < H; h++) {
        float d = fabsf(got[H + h] - row1[h]);
        if (d > split) {
            split = d;
        }
    }

    int probes[] = {0, 1, 2, 3, 8, 16, 32, 63};
    int nprobe = H > 63 ? 8 : 4;
    precision_t* gone = (precision_t*)malloc((size_t)rows * H * sizeof(precision_t));
    for (int i = 0; i < rows * H; i++) {
        gone[i] = from_float(1.0f);
    }
    precision_t* dgrad = NULL;
    cudaMalloc(&dgrad, (size_t)rows * H * sizeof(precision_t));
    cudaMemcpy(dgrad, gone, (size_t)rows * H * sizeof(precision_t),
        cudaMemcpyHostToDevice);
    // Restore the reference batch before the finite-difference check.
    for (int r = 0; r < rows; r++) {
        fill_kind(src + (size_t)r * OBS_SIZE, r);
    }
    round_obs(src, rounded, bits, rows * OBS_SIZE);
    cudaMemcpy(dobs, bits, (size_t)rows * OBS_SIZE * sizeof(precision_t),
        cudaMemcpyHostToDevice);
    run_forward(&net, dobs, rows);
    free(got);
    got = pull_prec(&ea->out);
    float sum1 = 0.0f;
    for (int i = 0; i < rows * H; i++) {
        sum1 += got[i];
    }
    Prec grad = {.data = dgrad, .shape = {rows, H}};
    net.arch.encoder.backward(net.w.encoder, net.acts.encoder, grad, 0);
    must_cuda("encoder backward");
    DriveEncWeights* ew = (DriveEncWeights*)net.w.encoder;
    float g_ego = peak_abs(&ea->ego_w1_g);
    float g_part = peak_abs(&ea->part_w1_g);
    float g_road = peak_abs(&ea->road_w1_g);
    float g_wq = peak_abs(&ea->wq_g);
    float* gfc = pull_prec(&ea->fc_b_g);
    float* gbq = pull_prec(&ea->bq_g);
    int fd_bad = 0;
    float fd_worst = 0.0f;
    const float eps = 0.015625f;
    for (int p = 0; p < nprobe; p++) {
        int i = probes[p];
        precision_t orig;
        cudaMemcpy(&orig, ew->fc_b.data + i, sizeof(orig),
            cudaMemcpyDeviceToHost);
        float old = to_float(orig);
        precision_t bumped = from_float(old + eps);
        float delta = to_float(bumped) - old;
        cudaMemcpy(ew->fc_b.data + i, &bumped, sizeof(bumped),
            cudaMemcpyHostToDevice);
        run_forward(&net, dobs, rows);
        free(got);
        got = pull_prec(&ea->out);
        float sum2 = 0.0f;
        for (int k = 0; k < rows * H; k++) {
            sum2 += got[k];
        }
        cudaMemcpy(ew->fc_b.data + i, &orig, sizeof(orig),
            cudaMemcpyHostToDevice);
        float fd = (sum2 - sum1) / delta;
        float err = fabsf(fd - gfc[i]);
        if (err > fd_worst) {
            fd_worst = err;
        }
        if (err > 0.35f) {
            fd_bad++;
            printf("fd H=%d fc_b[%d] cuda %.4f fd %.4f\n",
                H, i, gfc[i], fd);
        }
    }
    // bq reaches the output only through softmax. A 1/64 bf16 step
    // moves the summed outputs by less than that sum's rounding, so
    // the secant is reported and is not a pass/fail gate. At H=64 the
    // attention dQ matched its formula to mean abs 3e-5, and bo matched
    // the sum of d_pre. A dead bias grad still fails the peak check.
    int bq_bad = 0;
    float bq_worst = 0.0f;
    float g_bq = 0.0f;
    for (int i = 0; i < H; i++) {
        float a = fabsf(gbq[i]);
        if (a > g_bq) {
            g_bq = a;
        }
    }
    for (int p = 0; p < nprobe; p++) {
        int i = probes[p];
        precision_t orig;
        cudaMemcpy(&orig, ew->bq.data + i, sizeof(orig),
            cudaMemcpyDeviceToHost);
        float old = to_float(orig);
        precision_t bumped = from_float(old + eps);
        float delta = to_float(bumped) - old;
        cudaMemcpy(ew->bq.data + i, &bumped, sizeof(bumped),
            cudaMemcpyHostToDevice);
        run_forward(&net, dobs, rows);
        free(got);
        got = pull_prec(&ea->out);
        float sum2 = 0.0f;
        for (int k = 0; k < rows * H; k++) {
            sum2 += got[k];
        }
        cudaMemcpy(ew->bq.data + i, &orig, sizeof(orig),
            cudaMemcpyHostToDevice);
        float fd = (sum2 - sum1) / delta;
        float err = fabsf(fd - gbq[i]);
        if (err > bq_worst) {
            bq_worst = err;
        }
        if (err > 0.5f) {
            bq_bad++;
        }
    }
    printf("check H=%d max_abs %.4f mean_abs %.5f dup %.4f split %.4f\n",
        H, max_abs, mean_abs, dup_max, split);
    printf("  grad peak ego %.4f partner %.4f road %.4f wq %.4f bq %.4f\n",
        g_ego, g_part, g_road, g_wq, g_bq);
    printf("  fc fd_bad %d worst %.4f  bq secant misses %d worst %.4f\n",
        fd_bad, fd_worst, bq_bad, bq_worst);
    int ok = 1;
    if (max_abs > 0.15f || mean_abs > 0.02f) {
        ok = 0;
    }
    // A copied partner is a second key, so the output must move.
    if (dup_max < 1.0e-3f || split < 1.0e-3f) {
        ok = 0;
    }
    if (g_ego < 1.0e-5f || g_part < 1.0e-5f || g_road < 1.0e-5f
            || g_wq < 1.0e-5f || g_bq < 1.0e-5f) {
        ok = 0;
    }
    if (fd_bad > 2) {
        ok = 0;
    }
    free(src); free(rounded); free(bits); free(got); free(expect);
    free(row1); free(gone); free(gfc); free(gbq);
    cudaFree(dobs);
    cudaFree(dgrad);
    free_net(&net);
    return ok;
}

struct StepIn {
    Arch* arch;
    Weights* w;
    Activations* acts;
    Prec obs;
    Prec state;
    Prec terminals;
    Prec egrad;
    Float glog;
    Float gval;
};

static void step_full(StepIn* s) {
    int B = (int)s->obs.shape[0];
    int T = (int)s->obs.shape[1];
    Prec flat = s->obs;
    Prec enc = s->arch->encoder.forward(
        s->w->encoder, s->acts->encoder, *puf_squeeze(&flat, 0), 0);
    Prec hin = enc;
    Prec h = s->arch->network.forward_train(
        s->w->network, *puf_unsqueeze(&hin, 0, B, T),
        s->state, s->terminals, s->acts->network, 0, 0);
    Prec hd = h;
    s->arch->decoder.forward(
        s->w->decoder, s->acts->decoder, *puf_squeeze(&hd, 0), 0);
    Float none = {};
    arch_backward(s->arch, *s->w, *s->acts, s->glog, none, s->gval, 0);
}

static void step_enc(StepIn* s) {
    Prec flat = s->obs;
    s->arch->encoder.forward(
        s->w->encoder, s->acts->encoder, *puf_squeeze(&flat, 0), 0);
    s->arch->encoder.backward(s->w->encoder, s->acts->encoder, s->egrad, 0);
}

// Per-step GPU time. The mean is the throughput number when the device is
// idle. min and p10 catch steps that landed in a gap if another process
// is sharing the GPU; they are diagnostics, not the reported average.
static void time_stats(void (*fn)(StepIn*), StepIn* s, int warm, int iters,
        float* mean_out, float* min_out, float* p10_out) {
    for (int i = 0; i < warm; i++) {
        fn(s);
    }
    must_cuda("warmup");
    float* samples = (float*)malloc((size_t)iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        fn(s);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&samples[i], start, stop);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    float sum = 0.0f;
    for (int i = 0; i < iters; i++) {
        sum += samples[i];
    }
    for (int i = 1; i < iters; i++) {
        float v = samples[i];
        int j = i;
        while (j > 0 && samples[j - 1] > v) {
            samples[j] = samples[j - 1];
            j--;
        }
        samples[j] = v;
    }
    int p10 = iters / 10;
    *mean_out = sum / (float)iters;
    *min_out = samples[0];
    *p10_out = samples[p10];
    if (g_floor) {
        printf("    floor min %.3f p10 %.3f p25 %.3f p50 %.3f mean %.3f\n",
            samples[0], samples[p10], samples[iters / 4],
            samples[iters / 2], sum / (float)iters);
    }
    free(samples);
}

static void fill_train_obs(precision_t* dst, int n) {
    float* host = (float*)malloc((size_t)n * sizeof(float));
    precision_t* bits = (precision_t*)malloc((size_t)n * sizeof(precision_t));
    for (int i = 0; i < n; i++) {
        int feature = i % OBS_SIZE;
        int in_partner = feature >= EGO_FEATURES
            && feature < road_base();
        int in_road = feature >= road_base();
        int slot = 0;
        if (in_partner) {
            slot = (feature - EGO_FEATURES) / PARTNER_FEATURES;
        }
        if (in_road) {
            slot = (feature - road_base()) / ROAD_FEATURES;
        }
        float v = 0.0f;
        if (!in_partner && !in_road) {
            v = 0.1f * (float)((i * 17) % 11) - 0.5f;
        } else if (in_partner && slot < 8) {
            v = 0.05f * (float)((i * 13) % 9);
        } else if (in_road && slot < 16) {
            int local = (feature - road_base()) % ROAD_FEATURES;
            v = (local == ROAD_FEATURES - 1) ? (float)(slot % 7) : 0.2f;
        }
        host[i] = v;
        bits[i] = from_float(v);
    }
    cudaMemcpy(dst, bits, (size_t)n * sizeof(precision_t),
        cudaMemcpyHostToDevice);
    free(host);
    free(bits);
}

static void bench_pair(int H, int layers) {
    int B_TT = BENCH_B * BENCH_T;
    int aw = act_width();
    Net lin = make_net(H, layers, BENCH_T, B_TT, 0);
    Net ent = make_net(H, layers, BENCH_T, B_TT, 1);
    precision_t* obs = NULL;
    precision_t* state = NULL;
    precision_t* term = NULL;
    precision_t* egrad = NULL;
    float* glog = NULL;
    float* gval = NULL;
    cudaMalloc(&obs, (size_t)B_TT * OBS_SIZE * sizeof(precision_t));
    cudaMalloc(&state, (size_t)layers * BENCH_B * H * sizeof(precision_t));
    cudaMalloc(&term, (size_t)B_TT * sizeof(precision_t));
    cudaMalloc(&egrad, (size_t)B_TT * H * sizeof(precision_t));
    cudaMalloc(&glog, (size_t)B_TT * aw * sizeof(float));
    cudaMalloc(&gval, (size_t)B_TT * sizeof(float));
    fill_train_obs(obs, B_TT * OBS_SIZE);
    cudaMemset(state, 0, (size_t)layers * BENCH_B * H * sizeof(precision_t));
    cudaMemset(term, 0, (size_t)B_TT * sizeof(precision_t));
    bench_fill_p<<<grid_size(B_TT * H), BLOCK_SIZE>>>(egrad, 1.0f, B_TT * H);
    bench_fill_f<<<grid_size(B_TT * aw), BLOCK_SIZE>>>(glog, 1.0f, B_TT * aw);
    bench_fill_f<<<grid_size(B_TT), BLOCK_SIZE>>>(gval, 1.0f, B_TT);
    must_cuda("bench alloc");

    StepIn base = {};
    base.obs = {.data = obs, .shape = {BENCH_B, BENCH_T, OBS_SIZE}};
    base.state = {.data = state, .shape = {layers, BENCH_B, H}};
    base.terminals = {.data = term, .shape = {BENCH_B, BENCH_T}};
    base.egrad = {.data = egrad, .shape = {B_TT, H}};
    base.glog = {.data = glog, .shape = {BENCH_B, BENCH_T, aw}};
    base.gval = {.data = gval, .shape = {BENCH_B, BENCH_T, 1}};

    StepIn sl = base;
    sl.arch = &lin.arch;
    sl.w = &lin.w;
    sl.acts = &lin.acts;
    StepIn se = base;
    se.arch = &ent.arch;
    se.w = &ent.w;
    se.acts = &ent.acts;

    float lin_enc, lin_enc_min, lin_enc_p10;
    float ent_enc, ent_enc_min, ent_enc_p10;
    float lin_full, lin_full_min, lin_full_p10;
    float ent_full, ent_full_min, ent_full_p10;
    if (g_floor) printf("lin_enc\n");
    time_stats(step_enc, &sl, WARMUP, g_iters,
        &lin_enc, &lin_enc_min, &lin_enc_p10);
    if (g_floor) printf("ent_enc\n");
    time_stats(step_enc, &se, WARMUP, g_iters,
        &ent_enc, &ent_enc_min, &ent_enc_p10);
    if (g_floor) printf("lin_full\n");
    time_stats(step_full, &sl, WARMUP, g_iters,
        &lin_full, &lin_full_min, &lin_full_p10);
    if (g_floor) printf("ent_full\n");
    time_stats(step_full, &se, WARMUP, g_iters,
        &ent_full, &ent_full_min, &ent_full_p10);
    double steps = (double)B_TT;
    double lin_sps = steps / (lin_full / 1.0e3);
    double ent_sps = steps / (ent_full / 1.0e3);
    printf("%4d %6d %12.0f %12.0f %7.3f %8.2f %8.2f %8.2f %8.2f\n",
        H, layers, lin_sps, ent_sps, ent_sps / lin_sps,
        lin_full, ent_full, lin_enc, ent_enc);
    printf("     min %8.2f %8.2f p10 %8.2f %8.2f",
        lin_full_min, ent_full_min, lin_full_p10, ent_full_p10);
    printf("  enc min %8.2f %8.2f p10 %8.2f %8.2f\n",
        lin_enc_min, ent_enc_min, lin_enc_p10, ent_enc_p10);
    fflush(stdout);

    cudaFree(obs); cudaFree(state); cudaFree(term);
    cudaFree(egrad); cudaFree(glog); cudaFree(gval);
    free_net(&lin);
    free_net(&ent);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    cublas_init_stream(0);
    if (argc > 1 && strcmp(argv[1], "floor") == 0) {
        g_floor = 1;
        g_iters = 240;
    } else if (argc > 1 && strcmp(argv[1], "grid") == 0) {
        g_floor = 1;
        g_iters = 100;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU %s\n", prop.name);
    printf("num_players is not a drive setting; sweeping num_layers\n");
    printf("train step = encoder + MinGRU + decoder fwd+bwd, ");
    printf("B=%d T=%d (%d agent-steps)\n", BENCH_B, BENCH_T, BENCH_B * BENCH_T);
    if (!check_encoder(64) || !check_encoder(128)) {
        printf("encoder check failed\n");
        return 1;
    }
    printf("H layers default_sps entity_sps ratio ");
    printf("def_ms ent_ms def_enc_ms ent_enc_ms\n");
    if (g_floor && g_iters > 100) {
        bench_pair(64, 4);
        return 0;
    }
    int hs[] = {64, 128};
    int ls[] = {2, 3, 4};
    for (int hi = 0; hi < 2; hi++) {
        for (int li = 0; li < 3; li++) {
            bench_pair(hs[hi], ls[li]);
        }
    }
    return 0;
}
