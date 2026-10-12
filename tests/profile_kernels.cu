#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <chrono>

#include "pufferl.cu"
#include "ini.h"

const int WARMUP_ITERS = 100;
const int TIMING_ITERS = 1000;

const int BUF = 2;
const int BR = 4096;   // Rollout batch (no T dim)
const int BT = 512;    // Train batch (with T dim)
const int T_ = 64;     // T_ to avoid collision with PrefixScan::T
const int H_ = 128;
const int A_ = 4;

#ifndef ENV_NAME
#error "ENV_NAME must be defined at compile time (e.g. -DENV_NAME=breakout)"
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

typedef void (*kernel_fn)(void*);

void print_usage(const char* prog) {
    printf("Usage: %s <profile>\n", prog);
    printf("\nProfiles:\n");
    printf("  kernels        - All individual kernel microbenchmarks\n");
    printf("  mingrugate     - MinGRU gate kernel only\n");
    printf("  fusedscan      - MinGRU seq scan fwd+bwd\n");
    printf("    --batch N --horizon N --hidden N\n");
    printf("  fusedscan-sweep - Seq scan vs T at fixed B*T=65536\n");
    printf("  samplelogits   - Sample logits kernel only\n");
    printf("  ppoloss        - PPO loss fused fwd+bwd kernel\n");
    printf("  envspeed       - Environment step throughput\n");
    printf("    --buffers N  - Number of buffers (default: %d)\n", BUF);
    printf("    --threads N  - Number of threads (default: 16)\n");
    printf("    --horizon N  - Horizon length (default: %d)\n", T_);
    printf("  all            - Run all available profiles\n");
}


inline void print_timing(const char* name, float ms, int N) {
    printf("  %-28s %8.1f us  %8.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
}

inline void warmup_gpu() {
    float* dummy;
    cudaMalloc(&dummy, 64 * 1024 * 1024);
    for (int i = 0; i < 100; i++) cudaMemset(dummy, 0, 64 * 1024 * 1024);
    cudaDeviceSynchronize();
    cudaFree(dummy);
}

inline float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

inline void float_to_device(precision_t* dst, const float* src, int count) {
    precision_t* tmp = (precision_t*)malloc(count * sizeof(precision_t));
    for (int i = 0; i < count; ++i) tmp[i] = (precision_t)src[i];
    cudaMemcpy(dst, tmp, count * sizeof(precision_t), cudaMemcpyHostToDevice);
    free(tmp);
}

inline float profile_kernel(kernel_fn fn, void* args) {
    for (int i = 0; i < WARMUP_ITERS; ++i) fn(args);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaProfilerStart();
    cudaEventRecord(start);
    for (int i = 0; i < TIMING_ITERS; ++i) fn(args);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();
    return ms / TIMING_ITERS;
}

struct MingruGateProfile {
    Prec state, combined, x_in, out, next_state;
    Allocator alloc;
    int B, H;
};

MingruGateProfile* create_mingrugate(int B, int H) {
    auto* p = (MingruGateProfile*)calloc(1, sizeof(MingruGateProfile));
    p->B = B; p->H = H;
    p->state     = {.shape = {B, H}};
    p->combined  = {.shape = {B, 3*H}};
    p->x_in      = {.shape = {B, H}};
    p->out       = {.shape = {B, H}};
    p->next_state = {.shape = {B, H}};
    p->alloc = {};
    alloc_register(&p->alloc, &p->state);
    alloc_register(&p->alloc, &p->combined);
    alloc_register(&p->alloc, &p->x_in);
    alloc_register(&p->alloc, &p->out);
    alloc_register(&p->alloc, &p->next_state);
    alloc_create(&p->alloc);

    int N = B * H;
    float* buf = (float*)malloc((N + 3*N + N) * sizeof(float));
    for (int i = 0; i < N; ++i) buf[i] = fabsf(rand1()) + 0.1f;
    float_to_device(p->state.data, buf, N);
    for (int i = 0; i < 3*N; ++i) buf[i] = rand1() * 5.0f;
    float_to_device(p->combined.data, buf, 3*N);
    for (int i = 0; i < N; ++i) buf[i] = rand1();
    float_to_device(p->x_in.data, buf, N);
    free(buf);
    return p;
}

void run_mingrugate(MingruGateProfile* p) {
    mingru_gate<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(
        p->out.data, p->next_state.data, p->combined.data,
        p->state.data, p->x_in.data, p->H, p->B);
}

void profile_mingrugate(int B, int H) {
    printf("mingru_gate (B=%d, H=%d)\n", B, H);
    auto* p = create_mingrugate(B, H);
    float ms = profile_kernel((kernel_fn)run_mingrugate, p);
    print_timing("forward", ms, B);
    printf("\n");
    cudaFree(p->alloc.mem);
    free(p);
}

struct FusedScanProfile {
    PrefixScan scan;
    Prec grad_out, grad_next_state;
    Allocator alloc;
    int B, T, H;
};

FusedScanProfile* create_fusedscan(int B, int T, int H) {
    auto* p = (FusedScanProfile*)calloc(1, sizeof(FusedScanProfile));
    p->B = B; p->T = T; p->H = H;

    PrefixScan& s = p->scan;
    s.B = B; s.T = T; s.H = H;

    // Allocator needs Prec/Float, but PrefixScan uses raw ptrs
    // for combined/state/input. Allocate those via tensors then assign.
    Prec combined_t = {.shape = {B, T, 3*H}};
    Prec state_t    = {.shape = {B, H}};
    Prec input_t    = {.shape = {B, T, H}};

    s.out            = {.shape = {B, T, H}};
    s.next_state     = {.shape = {B, H}};
    s.scan_h         = {.shape = {B, T, H}};
    s.grad_combined  = {.shape = {B, T, 3*H}};
    s.grad_state     = {.shape = {B, H}};
    s.grad_input     = {.shape = {B, T, H}};

    p->grad_out        = {.shape = {B, T, H}};
    p->grad_next_state = {.shape = {B, H}};

    p->alloc = {};
    alloc_register(&p->alloc, &combined_t);
    alloc_register(&p->alloc, &state_t);
    alloc_register(&p->alloc, &input_t);
    alloc_register(&p->alloc, &s.out);
    alloc_register(&p->alloc, &s.next_state);
    alloc_register(&p->alloc, &s.scan_h);
    alloc_register(&p->alloc, &s.grad_combined);
    alloc_register(&p->alloc, &s.grad_state);
    alloc_register(&p->alloc, &s.grad_input);
    alloc_register(&p->alloc, &p->grad_out);
    alloc_register(&p->alloc, &p->grad_next_state);
    alloc_create(&p->alloc);

    s.combined_ptr = combined_t.data;
    s.state_ptr    = state_t.data;
    s.input_ptr    = input_t.data;

    int N_combined = B * T * 3 * H;
    int N_state = B * H;
    int N_out = B * T * H;
    float* buf = (float*)malloc(N_combined * sizeof(float));
    for (int i = 0; i < N_combined; ++i) buf[i] = rand1() * 5.0f;
    float_to_device(s.combined_ptr, buf, N_combined);
    for (int i = 0; i < N_state; ++i) buf[i] = fabsf(rand1()) + 0.1f;
    float_to_device(s.state_ptr, buf, N_state);
    for (int i = 0; i < N_out; ++i) buf[i] = rand1();
    float_to_device(s.input_ptr, buf, N_out);
    float_to_device(p->grad_out.data, buf, N_out);
    for (int i = 0; i < N_state; ++i) buf[i] = rand1();
    float_to_device(p->grad_next_state.data, buf, N_state);
    free(buf);
    return p;
}

void run_fusedscan_fwd(FusedScanProfile* p) {
    mingru_scan_forward<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(p->scan);
}
void run_fusedscan_bwd(FusedScanProfile* p) {
    mingru_scan_backward<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(
        p->scan, p->grad_out.data, p->grad_next_state.data);
}

void profile_fusedscan(int B, int T, int H) {
    printf("fused_scan seq (N=%d, B=%d T=%d H=%d)\n", B * T * H, B, T, H);
    auto* p = create_fusedscan(B, T, H);
    run_fusedscan_fwd(p);
    cudaDeviceSynchronize();

    float fwd = profile_kernel((kernel_fn)run_fusedscan_fwd, p);
    print_timing("fwd seq", fwd, B * T);
    run_fusedscan_fwd(p);
    cudaDeviceSynchronize();
    float bwd = profile_kernel((kernel_fn)run_fusedscan_bwd, p);
    print_timing("bwd", bwd, B * T);
    printf("  %-28s %8.1f us\n", "fwd+bwd", (fwd + bwd) * 1000);
    printf("\n");
    cudaFree(p->alloc.mem);
    free(p);
}

// Fixed B*T=65536: seq scan cost vs T.
void profile_fusedscan_sweep(int H) {
    const int mb = 65536;
    const int Ts[] = {32, 48, 64, 96, 128, 192, 256, 384, 512};
    printf("=== seq scan  H=%d  fixed B*T=%d ===\n", H, mb);
    printf("%6s %6s %10s %10s %10s\n", "T", "B", "fwd", "bwd", "tot");
    for (int i = 0; i < (int)(sizeof(Ts) / sizeof(Ts[0])); i++) {
        int T = Ts[i];
        int B = mb / T;
        auto* p = create_fusedscan(B, T, H);
        run_fusedscan_fwd(p);
        cudaDeviceSynchronize();
        float fs = profile_kernel((kernel_fn)run_fusedscan_fwd, p);
        run_fusedscan_fwd(p);
        cudaDeviceSynchronize();
        float bwd = profile_kernel((kernel_fn)run_fusedscan_bwd, p);
        printf("%6d %6d %8.1f us %8.1f us %8.1f us\n",
            T, B, fs * 1000, bwd * 1000, (fs + bwd) * 1000);
        cudaFree(p->alloc.mem);
        free(p);
    }
}

struct PPOProfile {
    PPOKernelArgs ka;
    PPOGraphArgs ga;
    Float losses_acc, ppo_partials;
    Float grad_logits_t, grad_values_t, adv_mean_t, adv_var_t, ent_coef_t;
    Prec logits_t, old_logprobs_t, advantages_t, prio_t, values_t, returns_t;
    Float actions_t;
    Int act_sizes_t;
    Allocator alloc;
    int N, T, A, ppo_grid;
};

PPOProfile* create_ppoloss(int N, int T, int A) {
    auto* p = (PPOProfile*)calloc(1, sizeof(PPOProfile));
    p->N = N; p->T = T; p->A = A;

    int NT = N * T;
    int fused_cols = A + 1;
    int ppo_grid = (NT + PPO_THREADS - 1) / PPO_THREADS;
    p->ppo_grid = ppo_grid;

    p->logits_t       = {.shape = {N, T, fused_cols}};
    p->actions_t      = {.shape = {NT}};
    p->old_logprobs_t = {.shape = {NT}};
    p->advantages_t   = {.shape = {NT}};
    p->prio_t         = {.shape = {N}};
    p->values_t       = {.shape = {NT}};
    p->returns_t      = {.shape = {NT}};
    p->grad_logits_t  = {.shape = {N, T, A}};
    p->grad_values_t  = {.shape = {NT}};
    p->adv_mean_t     = {.shape = {1}};
    p->adv_var_t      = {.shape = {1}};
    p->ent_coef_t     = {.shape = {1}};
    p->losses_acc     = {.shape = {LOSS_N + 1}};
    p->ppo_partials   = {.shape = {ppo_grid, LOSS_N}};
    p->act_sizes_t    = {.shape = {1}};

    p->alloc = {};
    alloc_register(&p->alloc, &p->logits_t);
    alloc_register(&p->alloc, &p->actions_t);
    alloc_register(&p->alloc, &p->old_logprobs_t);
    alloc_register(&p->alloc, &p->advantages_t);
    alloc_register(&p->alloc, &p->prio_t);
    alloc_register(&p->alloc, &p->values_t);
    alloc_register(&p->alloc, &p->returns_t);
    alloc_register(&p->alloc, &p->grad_logits_t);
    alloc_register(&p->alloc, &p->grad_values_t);
    alloc_register(&p->alloc, &p->adv_mean_t);
    alloc_register(&p->alloc, &p->adv_var_t);
    alloc_register(&p->alloc, &p->ent_coef_t);
    alloc_register(&p->alloc, &p->losses_acc);
    alloc_register(&p->alloc, &p->ppo_partials);
    alloc_register(&p->alloc, &p->act_sizes_t);
    alloc_create(&p->alloc);

    cudaMemcpy(p->act_sizes_t.data, &A, sizeof(int), cudaMemcpyHostToDevice);

    float ent_coef_val = 0.01f;
    cudaMemcpy(p->ent_coef_t.data, &ent_coef_val, sizeof(float), cudaMemcpyHostToDevice);

    // Fill with random data
    float* buf = (float*)malloc(NT * fused_cols * sizeof(float));

    // Advantages (precision_t) + compute mean/var
    float adv_sum = 0, adv_sq = 0;
    for (int i = 0; i < NT; ++i) {
        float a = rand1();
        buf[i] = a;
        adv_sum += a;
        adv_sq += a * a;
    }
    float adv_mean = adv_sum / NT;
    float adv_var = adv_sq / NT - adv_mean * adv_mean;
    float_to_device(p->advantages_t.data, buf, NT);
    cudaMemcpy(p->adv_mean_t.data, &adv_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p->adv_var_t.data, &adv_var, sizeof(float), cudaMemcpyHostToDevice);

    // Fill logits (fused: A logit cols + 1 value col per row)
    for (int i = 0; i < NT * fused_cols; ++i) buf[i] = rand1() * 2.0f;
    float_to_device(p->logits_t.data, buf, NT * fused_cols);
    // actions are float32 in the loss kernel
    for (int i = 0; i < NT; ++i) buf[i] = (float)(rand() % A);
    cudaMemcpy(p->actions_t.data, buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    // old_logprobs
    for (int i = 0; i < NT; ++i) buf[i] = rand1() * 2.0f;
    float_to_device(p->old_logprobs_t.data, buf, NT);
    // values + returns
    for (int i = 0; i < NT; ++i) buf[i] = rand1();
    float_to_device(p->values_t.data, buf, NT);
    for (int i = 0; i < NT; ++i) buf[i] = rand1();
    float_to_device(p->returns_t.data, buf, NT);
    // prio
    for (int i = 0; i < N; ++i) buf[i] = (float)rand() / RAND_MAX;
    float_to_device(p->prio_t.data, buf, N);
    free(buf);

    // Wire up kernel args
    p->ka = {
        .grad_logits = p->grad_logits_t.data,
        .grad_logstd = NULL,
        .grad_values_pred = p->grad_values_t.data,
        .logits = p->logits_t.data,
        .logstd = NULL,
        .values_pred = p->logits_t.data + A,  // value is last col in fused layout
        .act_sizes = p->act_sizes_t.data,
        .action_mask = NULL,
        .num_atns = 1,
        .clip_coef = 0.1f, .vf_clip_coef = 0.1f, .vf_coef = 0.5f, .ent_coef = p->ent_coef_t.data,
        .T_seq = T, .A_total = A, .N = N,
        .is_continuous = false,
    };
    p->ga = {
        .imp = NULL,
        .actions = p->actions_t.data,
        .old_logprobs = p->old_logprobs_t.data,
        .advantages = p->advantages_t.data,
        .values = p->values_t.data,
        .returns = p->returns_t.data,
    };

    return p;
}

void run_ppoloss(PPOProfile* p) {
    ppo_loss_compute<<<p->ppo_grid, PPO_THREADS>>>(
        p->ppo_partials.data, p->ka, p->ga);
    ppo_loss_reduce<<<1, LOSS_N>>>(
        p->losses_acc.data, p->ppo_partials.data, p->ppo_grid);
}

void profile_ppoloss(int N, int T, int A) {
    int NT = N * T;
    printf("ppo_loss_fwd_bwd (NT=%d, %dx%d, A=%d)\n", NT, N, T, A);
    auto* p = create_ppoloss(N, T, A);
    float ms = profile_kernel((kernel_fn)run_ppoloss, p);
    print_timing("fwd+bwd", ms, NT);
    printf("\n");
    cudaFree(p->alloc.mem);
    free(p);
}

struct SampleLogitsProfile {
    Prec dec_out, logstd, mask, logprobs_t, value_out_t;
    Int act_sizes;
    Float actions_f, env_actions_f;
    curandStatePhilox4_32_10_t* rng_states;
    Allocator alloc;
    int B, A, mask_stride;
};

SampleLogitsProfile* create_samplelogits(int B, int A) {
    auto* p = (SampleLogitsProfile*)calloc(1, sizeof(SampleLogitsProfile));
    p->B = B; p->A = A;

    int logits_dim = NUM_ATNS * A;
    int fused_cols = logits_dim + 1;
    p->mask_stride = logits_dim;
    p->dec_out     = {.shape = {B, fused_cols}};
    p->logstd      = {.shape = {0}};  // empty for discrete
    p->act_sizes   = {.shape = {NUM_ATNS}};
    p->actions_f   = {.shape = {B, NUM_ATNS}};
    p->env_actions_f = {.shape = {B, NUM_ATNS}};
    p->logprobs_t  = {.shape = {B}};
    p->value_out_t = {.shape = {B}};
    p->mask        = {.shape = {B, logits_dim}};

    p->alloc = {};
    alloc_register(&p->alloc, &p->dec_out);
    alloc_register(&p->alloc, &p->act_sizes);
    alloc_register(&p->alloc, &p->actions_f);
    alloc_register(&p->alloc, &p->env_actions_f);
    alloc_register(&p->alloc, &p->logprobs_t);
    alloc_register(&p->alloc, &p->value_out_t);
    alloc_register(&p->alloc, &p->mask);
    alloc_create(&p->alloc);

    int sizes[8];
    for (int h = 0; h < NUM_ATNS; h++) sizes[h] = A;
    cudaMemcpy(p->act_sizes.data, sizes, NUM_ATNS * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&p->rng_states, B * sizeof(curandStatePhilox4_32_10_t));
    rng_init<<<grid_size(B), BLOCK_SIZE>>>(p->rng_states, 42, B);
    cudaDeviceSynchronize();

    float* buf = (float*)malloc(B * fused_cols * sizeof(float));
    for (int i = 0; i < B * fused_cols; ++i) buf[i] = rand1() * 5.0f;
    float_to_device(p->dec_out.data, buf, B * fused_cols);
    for (int i = 0; i < B * logits_dim; ++i) buf[i] = 1.0f;
    float_to_device(p->mask.data, buf, B * logits_dim);
    free(buf);
    return p;
}

void run_samplelogits(SampleLogitsProfile* p) {
    sample_logits<<<grid_size(p->B), BLOCK_SIZE>>>(
        p->dec_out, p->logstd, p->act_sizes.data,
        p->actions_f.data, p->env_actions_f.data,
        p->logprobs_t.data, p->value_out_t.data,
        p->rng_states, p->mask.data, p->mask_stride);
}

void profile_samplelogits(int B, int A) {
    printf("sample_logits (B=%d, A=%d)\n", B, A);
    auto* p = create_samplelogits(B, A);
    float ms = profile_kernel((kernel_fn)run_samplelogits, p);
    print_timing("forward", ms, B);
    printf("\n");
    cudaFree(p->rng_states);
    cudaFree(p->alloc.mem);
    free(p);
}

#ifdef PUFFERLIB_BUILD_MAIN
static void empty_net_callback(void* ctx, int buf, int t) {
    (void)ctx; (void)buf; (void)t;
}
static void empty_thread_init(void* ctx, int buf) {
    (void)ctx; (void)buf;
}

typedef struct {
    StaticVec* vec;
    int num_envs, num_buffers, num_threads, horizon, obs_size, num_atns;
} EnvSpeedArgs;

static int ini_handler_env(void* user, const char* section,
                           const char* name, const char* value) {
    Dict* env_kwargs = (Dict*)user;
    if (strcmp(section, "env") == 0) dict_set(env_kwargs, strdup(name), atof(value));
    return 1;
}

typedef struct { int total_agents; int num_buffers; } VecDefaults;
static int ini_handler_vec(void* user, const char* section,
                           const char* name, const char* value) {
    VecDefaults* defaults = (VecDefaults*)user;
    if (strcmp(section, "vec") == 0) {
        if (strcmp(name, "total_agents") == 0) defaults->total_agents = atoi(value);
        else if (strcmp(name, "num_buffers") == 0) defaults->num_buffers = atoi(value);
    }
    return 1;
}

EnvSpeedArgs* create_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    char ini_path[512];
    snprintf(ini_path, sizeof(ini_path), "config/%s.ini", TOSTRING(ENV_NAME));

    VecDefaults defaults = {0};
    ini_parse(ini_path, ini_handler_vec, &defaults);
    if (total_agents == 0) total_agents = defaults.total_agents > 0 ? defaults.total_agents : 8192;
    if (num_buffers == 0) num_buffers = defaults.num_buffers > 0 ? defaults.num_buffers : 2;

    Dict* env_kwargs = create_dict(64);
    ini_parse(ini_path, ini_handler_env, env_kwargs);
    Dict* vec_kwargs = create_dict(8);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);

    StaticVec* vec = create_static_vec(total_agents, num_buffers, 1, vec_kwargs, env_kwargs);
    if (!vec) { fprintf(stderr, "Failed to create environments\n"); return NULL; }
    for (int i = 0; i < num_buffers; i++)
        cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);

    printf("Created %d envs (%s) for %d total_agents\n", vec->size, TOSTRING(ENV_NAME), total_agents);
    create_static_threads(vec, num_threads, horizon, NULL, empty_net_callback, empty_thread_init);
    static_vec_reset(vec);
    cudaDeviceSynchronize();

    EnvSpeedArgs* args = (EnvSpeedArgs*)calloc(1, sizeof(EnvSpeedArgs));
    args->vec = vec;
    args->num_envs = vec->size;
    args->num_buffers = num_buffers;
    args->num_threads = num_threads;
    args->horizon = horizon;
    args->obs_size = get_obs_size();
    args->num_atns = get_num_atns();
    return args;
}

void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    printf("env_speed_static (total_agents=%d, buffers=%d, threads=%d, horizon=%d)\n",
           total_agents, num_buffers, num_threads, horizon);
    EnvSpeedArgs* args = create_envspeed(total_agents, num_buffers, num_threads, horizon);
    if (!args) { printf("  Failed to create env - skipping\n\n"); return; }
    printf("  num_envs=%d, obs_size=%d, num_atns=%d\n", args->num_envs, args->obs_size, args->num_atns);

    // Warmup
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        static_vec_omp_step(args->vec);
        cudaDeviceSynchronize();
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed > 3.0f) break;
    }

    // Timed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    t0 = std::chrono::steady_clock::now();
    cudaEventRecord(start);
    float completed = 0;
    for (int i = 0; i < 1000; ++i) {
        static_vec_omp_step(args->vec);
        completed += 1;
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed > 3.0f) break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float rollout_ms = ms / completed;
    int total_steps = total_agents * horizon;
    printf("  rollout time: %.2f ms (%d steps)\n", rollout_ms, total_steps);
    printf("  throughput: %.2f M steps/s\n", total_steps / rollout_ms / 1e3);
    free(args);
    printf("\n");
}
#else
void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    (void)total_agents; (void)num_buffers; (void)num_threads; (void)horizon;
    printf("env_speed_static is unavailable in this profile build; rebuild with PUFFERLIB_BUILD_MAIN.\n\n");
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    const char* profile = argv[1];
    int buffers = BUF, threads = 16, horizon = T_;
    int total_agents = BR * buffers;
    int encoder_batch = BR, encoder_hidden = H_;
    int scan_batch = BT, scan_T = T_, scan_H = H_;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--buffers") == 0) buffers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--horizon") == 0) {
            horizon = atoi(argv[++i]);
            scan_T = horizon;
        } else if (strcmp(argv[i], "--total-agents") == 0) total_agents = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0) {
            encoder_batch = atoi(argv[++i]);
            scan_batch = encoder_batch;
        } else if (strcmp(argv[i], "--hidden") == 0) {
            encoder_hidden = atoi(argv[++i]);
            scan_H = encoder_hidden;
        }
    }

    warmup_gpu();
    bool run_all = strcmp(profile, "all") == 0;

    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "mingrugate") == 0 || run_all)
        profile_mingrugate(BR, H_);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "fusedscan") == 0 || run_all)
        profile_fusedscan(scan_batch, scan_T, scan_H);
    if (strcmp(profile, "fusedscan-sweep") == 0)
        profile_fusedscan_sweep(scan_H);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "samplelogits") == 0 || run_all)
        profile_samplelogits(BR, A_);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "ppoloss") == 0 || run_all)
        profile_ppoloss(BT, T_, A_);
    if (strcmp(profile, "envspeed") == 0 || run_all)
        profile_envspeed(total_agents, buffers, threads, horizon);

    if (!run_all
        && strcmp(profile, "kernels") != 0
        && strcmp(profile, "mingrugate") != 0
        && strcmp(profile, "fusedscan") != 0
        && strcmp(profile, "fusedscan-sweep") != 0
        && strcmp(profile, "samplelogits") != 0
        && strcmp(profile, "ppoloss") != 0
        && strcmp(profile, "envspeed") != 0
    ) {
        printf("Unknown profile: %s\n\n", profile);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
