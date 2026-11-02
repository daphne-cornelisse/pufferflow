// CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <nccl.h>
#include <nvml.h>
#include <nvtx3/nvToolsExt.h>

// C standard
#include <cassert>
#include <cmath>
#include <cstdint>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// POSIX / threading
#include <dirent.h>
#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// Project
#include "ini.h"

#ifdef PRECISION_FLOAT
typedef float precision_t;
constexpr bool USE_BF16 = false;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_32F;
constexpr cublasComputeType_t CUBLAS_COMPUTE = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclFloat
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_16BF;
constexpr cublasComputeType_t CUBLAS_COMPUTE = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclBfloat16
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Compile vs a single env: -DENV_HEADER=ocean/<env>/<env>.h or .cu (--cu)
#include ENV_HEADER

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} Float;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} Byte;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} Long;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} Int;

typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} Prec;

__host__ __device__ int ndim(int64_t* shape) {
    int n = 0;
    while (n < PUF_MAX_DIMS && shape[n] != 0) {
        n++;
    }
    return n;
}

__host__ __device__ int64_t numel(int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) {
        n *= shape[i];
    }
    return n;
}

int64_t batch_size(int64_t* shape) {
    int n = ndim(shape);
    int64_t b = 1;
    for (int i = 0; i < n - 2; i++) {
        b *= shape[i];
    }
    return b;
}

void squeeze_shape(int64_t* shape, int dim) {
    int n = ndim(shape);
    shape[dim] *= shape[dim + 1];
    for (int i = dim + 1; i < n - 1; i++) {
        shape[i] = shape[i + 1];
    }
    shape[n - 1] = 0;
}

Prec* puf_squeeze(Prec* t, int dim) {
    squeeze_shape(t->shape, dim);
    return t;
}

Float* puf_squeeze(Float* t, int dim) {
    squeeze_shape(t->shape, dim);
    return t;
}

Prec* puf_unsqueeze(Prec* t, int dim, int64_t d0, int64_t d1) {
    int n = ndim(t->shape);
    assert(n + 1 <= PUF_MAX_DIMS);
    assert(t->shape[dim] == d0 * d1);
    for (int i = n; i > dim; i--) {
        t->shape[i] = t->shape[i - 1];
    }
    t->shape[dim] = d0;
    t->shape[dim + 1] = d1;
    return t;
}

void puf_copy(Prec* dst, Prec* src, cudaStream_t stream) {
    assert(numel(dst->shape) == numel(src->shape) && "puf_copy: size mismatch");
    cudaMemcpyAsync(dst->data, src->data,
        numel(dst->shape) * sizeof(precision_t),
        cudaMemcpyDeviceToDevice, stream);
}

__global__ void cast(precision_t* dst,
        float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(src[idx]);
    }
}

#ifdef PRECISION_FLOAT
// GPU envs fix obs as bf16; float train needs an explicit widen.
__global__ void cast(precision_t* dst,
        __nv_bfloat16* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}
#else
// Identity overload so obs→rollout cast arm typechecks when obs_t is bf16.
__global__ void cast(precision_t* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void cast(float* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}
#endif

__global__ void cast(precision_t* dst,
        unsigned char* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)src[idx]);
    }
}

__global__ void cast(unsigned char* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}

// Fused rew+term cast
__global__ void cast_rew_term(
        precision_t* __restrict__ rew_dst, const float* __restrict__ rew_src,
        precision_t* __restrict__ term_dst, const float* __restrict__ term_src,
        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        rew_dst[idx] = from_float(rew_src[idx]);
        term_dst[idx] = from_float(term_src[idx]);
    }
}

// Allocates large buffers (params/acts/grads) in contiguous memory.
struct AllocEntry {
    void** data_ptr;
    long* shape;
    int elem_size;
};

struct Allocator {
    AllocEntry* regs;
    int num_regs;
    void* mem;
    long total_elems;
    long total_bytes;
};

void _alloc_register(Allocator* alloc,
        void** data_ptr, long* shape, int elem_size) {
    alloc->regs = (AllocEntry*)realloc(alloc->regs,
        (alloc->num_regs + 1) * sizeof(AllocEntry));
    alloc->regs[alloc->num_regs++] = {data_ptr, shape, elem_size};
    long n = numel(shape);
    alloc->total_elems += n;
    alloc->total_bytes = (alloc->total_bytes + 15) & ~15;
    alloc->total_bytes += n * elem_size;
}
void alloc_register(Allocator* a, Prec* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(precision_t));
}
void alloc_register(Allocator* a, Float* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(float));
}
void alloc_register(Allocator* a, Long* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(long));
}
void alloc_register(Allocator* a, Int* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(int));
}

void alloc_create(Allocator* alloc) {
    assert(cudaMalloc(&alloc->mem, alloc->total_bytes) == cudaSuccess
        && "alloc_create: cudaMalloc failed");
    cudaMemset(alloc->mem, 0, alloc->total_bytes);
    long offset = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        offset = (offset + 15) & ~15;
        *alloc->regs[i].data_ptr = (char*)alloc->mem + offset;
        offset += numel(alloc->regs[i].shape) * alloc->regs[i].elem_size;
    }
}

// Deterministic block tree-reduce. smem layout: smem[c * nthreads + tid].
// Caller fills smem; tid 0 writes nchan results to out[0..nchan).
#ifndef PUF_BLOCK_REDUCE_SUM
#define PUF_BLOCK_REDUCE_SUM
__device__ __forceinline__ void block_reduce_sum(
        float* smem, float* out, int tid, int nthreads, int nchan) {
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < nchan; c++) {
                smem[c * nthreads + tid] += smem[c * nthreads + tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int c = 0; c < nchan; c++) {
            out[c] = smem[c * nthreads];
        }
    }
}
#endif

// Algo + sweeps only depend on basic tensor types and utilities
#include "algo.cu"
#include "protein.cu"

typedef struct {
    int horizon;
    int total_agents;
    int num_buffers;
    int hidden_size;
    int num_layers;
    float lr;
    float min_lr_ratio;
    bool anneal_lr;
    float momentum;
    int minibatch_size;
    float replay_ratio;
    long total_timesteps;
    float max_grad_norm;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    float min_ent_coef_ratio;
    bool anneal_ent_coef;
    float gamma;
    float gae_lambda;
    bool vtrace;
    float vtrace_rho_clip;
    float vtrace_c_clip;
    bool async;
    bool reset_every_horizon;
    bool cudagraphs;
    bool profile;
    int rank;
    int world_size;
    int gpu_id;
    int num_threads;
    int seed;
} Hypers;

// Rank / device context for one process in a multi-GPU train job.
typedef struct {
    int rank;
    int world_size;
    int gpu_id;
    int artifact_owner;
    ncclUniqueId* nccl_id;
} TrainContext;

typedef struct ObsTensor {
    obs_t* data;
    int64_t shape[8];
} ObsTensor;

// Each rollout buffer manages a constant subset of environments
// Simulation + inference overlap across buffers
// Note: experimental async path adds an extra slot dimension.
struct RolloutBuf {
    Prec observations;  // (horizon, agents, input_size)
    Prec initial_states;
    Float actions;      // (horizon, agents, num_atns) float32: large discrete IDs
    Prec values;        // (horizon, agents)
    Prec logprobs;      // ...
    Prec rewards;
    Prec terminals;
    Prec action_mask;   // (horizon, agents, mask_size)
};

// Buffers are initialized as raw structs with only shape information.
// alloc_register stores the shape and data pointer.
// Memory is only allocated after all buffers are registered.
void register_rollout_buffers(RolloutBuf* bufs, Allocator* alloc,
        int T, int B, int input_size, int num_atns, int mask_size) {
    memset(bufs, 0, sizeof(*bufs));
    bufs->observations = {.shape = {T, B, input_size}};
    bufs->actions      = {.shape = {T, B, num_atns}};
    bufs->values       = {.shape = {T, B}};
    bufs->logprobs     = {.shape = {T, B}};
    bufs->rewards      = {.shape = {T, B}};
    bufs->terminals    = {.shape = {T, B}};
    bufs->action_mask  = {.shape = {T, B, mask_size}};
    Prec* prec_fields[] = {
        &bufs->observations, &bufs->values, &bufs->logprobs,
        &bufs->rewards, &bufs->terminals, &bufs->action_mask,
    };
    for (int i = 0; i < (int)(sizeof(prec_fields) / sizeof(prec_fields[0])); i++) {
        alloc_register(alloc, prec_fields[i]);
    }
    alloc_register(alloc, &bufs->actions);
}

// Rank-2 or rank-3 time-major tensor. F==0 means rank-2 (zero-terminated shape);
// stride still multiplies by max(F, 1).
Prec puf_time_view(Prec p, int start_t, int T) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)start_t * B * stride_f,
        .shape = {T, B, F},
    };
}

Float puf_time_view(Float p, int start_t, int T) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)start_t * B * stride_f,
        .shape = {T, B, F},
    };
}

RolloutBuf rollout_time_view(RolloutBuf* base, int start_t, int T) {
    RolloutBuf view = *base;
    view.observations = puf_time_view(base->observations, start_t, T);
    view.actions      = puf_time_view(base->actions,      start_t, T);
    view.values       = puf_time_view(base->values,       start_t, T);
    view.logprobs     = puf_time_view(base->logprobs,     start_t, T);
    view.rewards      = puf_time_view(base->rewards,      start_t, T);
    view.terminals    = puf_time_view(base->terminals,    start_t, T);
    view.action_mask  = puf_time_view(base->action_mask,  start_t, T);
    return view;
}

// Env batch. Device IO is always PuffeRL.env (EnvBuf).
// CPU: host pins + workers. GPU: single-buffer device envs (no host pins).
struct VecEnv {
    Env* envs;  // CPU: host array. GPU: device base (batch).
    int size;
    int total_agents;
    int buffers;
    int agents_per_buf;
    int mask_size;
    int num_policies;
    int* policy_layout;
    // CPU pins / workers (unused on GPU)
    int* env_starts;
    int* env_counts;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int* worker_state;
    int shutdown;
    pthread_t* threads;
    float* accum;
    int num_workers;
    // GPU log reduce scratch (unused on CPU)
    float* log_scratch;
};

struct EnvBuf {
    ObsTensor obs;    // (total_agents, obs_size)
    Float actions;    // (total_agents, num_atns)
    Float rewards;    // (total_agents,)
    Float terminals;  // (total_agents,)
    Byte action_mask; // (total_agents, mask_size); always allocated
};

// Owned runnable policy instance. policies[0] is trainable; policies[i>0] are
// frozen opponents (selfplay/match). Frozen policies are rollout-only (no train
// acts/grads/muon) and may use a different arch via hist_policy_hidden_size /
// hist_policy_num_layers. Primary registers rollout acts into PuffeRL.activ_alloc
// (shared with train); frozen policies use their own activ_alloc.
typedef struct {
    bool frozen;
    Arch arch;
    Weights weights;
    Allocator params_alloc;
    Allocator activ_alloc;
    Prec param;
    Float master_weights;
    Prec* buffer_states;    // [num_buffers]
    Activations* buf_acts;  // [num_buffers]
} Policy;

// PROF_* accum; MODEL_*/H2D_* worker+GPU rollout events; TE_* train_impl events.
enum {
    PROF_ROLLOUT,
    PROF_MODEL,
    PROF_ENV,
    PROF_COPY,
    PROF_TRAIN_MISC,
    PROF_TRAIN_MODEL,
    NUM_PROF,
    MODEL_START = 0,
    MODEL_END,
    COPY_END,
    ENV_END = COPY_END,
    H2D_START,
    H2D_END,
    NUM_EV,
    EV_T = ENV_END + 1,
    TE_S = 0,
    TE_MID,
    TE_E,
    NUM_TE,
};

// Index must match PROF_* order above (first NUM_PROF names).
const char* PROF_NAMES[] = {
    "rollout",
    "eval_model",
    "eval_env",
    "eval_copy",
    "train_misc",
    "train_model",
};

// Frozen BC MLP from an npz (W*, b*, Wa, ba, Ws, bs, mu, sig). Same obs
// width and action heads as this env; depth and hidden width may differ.
#define ANCHOR_MAX_LAYERS 8

struct AnchorLayer {
    precision_t* W;
    precision_t* b;
    int in;
    int out;
};

struct Anchor {
    int on;
    int dist;
    float coef;
    int n_layers;
    int in_dim;
    int width;
    int heads[NUM_ATNS];
    int a_total;
    int max_nt;
    precision_t* mu;
    precision_t* sig;
    AnchorLayer layer[ANCHOR_MAX_LAYERS];
    precision_t* head_w[NUM_ATNS];
    precision_t* head_b[NUM_ATNS];
    precision_t* norm;
    precision_t* hid[2];
    precision_t* scratch;
    precision_t* logits;
    float* partials;
    // Own stream, not the dW stream: backward reuses that stream and its
    // events inside this same captured train graph.
    cudaStream_t stream;
    cublasHandle_t handle;
    void* workspace;
    cudaEvent_t fork_ev;
    cudaEvent_t done_ev;
};

// Index must match LossIdx 0..LOSS_N-1 in algo.cu (not LOSS_N counter).
const char* LOSS_NAMES[] = {
    "loss/policy",
    "loss/value",
    "loss/entropy",
    "loss/total",
    "loss/old_kl",
    "loss/kl",
    "loss/clipfrac",
    "importance",
    "loss/anchor",
};

typedef struct {
    unsigned long long* stamps;  // device [2 * NUM_TE]; globaltimer ns
    cudaEvent_t* rollout_ev;  // GPU [EV_T * horizon]; null on CPU
    float accum[NUM_PROF];
    int skip_rollout_time;
} Profile;

typedef struct PuffeRL {
    Policy* policies;        // [num_policies]; policies[0] trainable, rest frozen
    int num_policies;
    Weights actor_weights; // async rollout snapshot of policies[0]; unused when async=0
    Activations train_activs;
    Allocator weight_alloc;      // async actor weights
    Allocator grads_alloc;
    Allocator activ_alloc;       // train + primary rollout (shared)
    VecEnv* vec;
    Muon muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    Hypers hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;  // Pre-allocated transposed copy for train_impl
    EnvBuf env;
    TrainGraph train_buf;
    Prec train_state;  // (L, A, H) carry in env order; graph reads with dest_off
    cudaGraphExec_t* rollout_graphs;  // CPU: [slots][horizon][num_buffers]
    cudaGraphExec_t gpu_rollout_graph[2];  // GPU: full net+env horizon per slot
    cudaGraphExec_t train_cudagraph[2];  // per async slot; src pointers differ
    cudaStream_t* streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    cudaStream_t train_stream;    // dedicated learner stream (always non-default)
    int* act_sizes;        // device ACT_SIZES (NUM_ATNS ints)
    float* losses;         // device loss accumulator (NUM_LOSSES)
    PPOBufs ppo_bufs; // Pre-allocated buffers for ppo_loss_fwd_bwd
    Anchor anchor;
    Prec actor_param;      // async flat actor params
    Prec grad;
    long* rng_offset;      // device counters (num_buffers+1)
    Profile profile;
    nvmlDevice_t nvml_device;
    long epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    int write_slot;
    int async_ready_slot;
    int async_write_slot;
    int async_num_slots;  // 2 when async (Cleanba), else 1
    bool async_boot;
    ulong seed;
    curandStatePhilox4_32_10_t** rng_states;  // per-buffer persistent RNG states [num_buffers]
    char env_name[64];  // For policy arch rebuild at create.
} PuffeRL;

// Infer path: sample + forward, then vec workers.
static void profile_begin(const char* tag, bool enable) {
    if (enable) {
        nvtxRangePushA(tag);
    }
}

static void profile_end(bool enable) {
    if (enable) {
        nvtxRangePop();
    }
}

__global__ void rng_init(curandStatePhilox4_32_10_t* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Action logits and value share one row: [logits..., value]. logstd empty ⇒ discrete.
// Discrete: always-cache logsumexp + inverse-CDF; mask always present (all-ones if env has none).
// Continuous: ignores mask.
__global__ void sample_logits(
        Prec dec_out,              // (B, logits_dim + 1)
        Prec logstd,               // (1, od) continuous only; .data null if discrete
        int* act_sizes,            // (NUM_ATNS,)
        float* actions,            // (B, num_atns) float32 rollout store
        float* env_actions,        // (B, num_atns) env dispatch
        precision_t* logprobs,     // (B,)
        precision_t* value_out,    // (B,)
        curandStatePhilox4_32_10_t* rng_states,
        precision_t* action_mask,  // (B, A_total); always allocated
        int mask_stride) {
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int num_atns = NUM_ATNS;
    precision_t* logits = dec_out.data;
    bool is_continuous = logstd.data != NULL && numel(logstd.shape) > 0;
    precision_t* logstd_data = logstd.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    int logits_base = idx * fused_cols;
    float total_log_prob = 0.0f;

    if (is_continuous) {
        for (int h = 0; h < num_atns; h++) {
            float mean = safe_continuous_mean(logits, logits_base + h);
            float log_std = safe_continuous_logstd(logstd_data, h);
            float std = expf(log_std);
            float action = finite_or_clamp(
                mean + std * curand_normal(&state), -1.0e6f, 1.0e6f);
            // Preserve reduced-precision continuous semantics for logprob.
            precision_t stored_p = from_float(action);
            float stored = to_float(stored_p);
            float lp, ent;
            ppo_continuous_head(mean, log_std, stored, &lp, &ent);
            total_log_prob += lp;
            int aidx = idx * num_atns + h;
            actions[aidx] = stored;
            env_actions[aidx] = stored;
        }
    } else {
        int logits_offset = 0;
        int mask_base = idx * mask_stride;
        for (int h = 0; h < num_atns; h++) {
            int A = act_sizes[h];
            float cache[PPO_MAX_HEAD_A];
            float logsumexp = ppo_discrete_logsumexp(
                logits, logits_base, logits_offset, A, action_mask, mask_base, cache);
#ifdef PUFFER_NETHACK
            float inv_K = 0.0f;
            float eps = h == 0 ? nethack_verb_eps_load(
                action_mask + mask_base + logits_offset, A, &inv_K) : 0.0f;
#endif
            float rand_val = curand_uniform(&state);
            float cumsum = 0.0f;
            int sampled = A - 1;
            for (int a = 0; a < A; a++) {
#ifdef PUFFER_NETHACK
                if (eps > 0.0f) {
                    cumsum += nethack_verb_eps_mix(expf(cache[a] - logsumexp),
                        action_mask[mask_base + logits_offset + a], eps, inv_K);
                } else
#endif
                {
                    cumsum += expf(cache[a] - logsumexp);
                }
                if (rand_val < cumsum) {
                    sampled = a;
                    break;
                }
            }
            // CDF fall-through (float rounding) lands on A - 1, which may be
            // masked; snap to the last legal action. A legitimate A - 1 pick
            // is always legal, so the snap is an exact no-op for it.
            if (sampled == A - 1) {
                for (int a = A - 1; a >= 0; a--) {
                    if (to_float(action_mask[mask_base + logits_offset + a]) != 0.0f) {
                        sampled = a;
                        break;
                    }
                }
            }
            // Float32 preserves large categorical IDs that BF16 cannot represent.
            int aidx = idx * num_atns + h;
            float action = (float)sampled;
            actions[aidx] = action;
            env_actions[aidx] = action;
#ifdef PUFFER_NETHACK
            int verb = (int)actions[idx * num_atns];
            int used = nethack_head_used(verb, h);
            if (used) {
                if (eps > 0.0f) {
                    total_log_prob += logf((1.0f - eps)
                        * expf(cache[sampled] - logsumexp) + eps * inv_K);
                } else {
                    total_log_prob += cache[sampled] - logsumexp;
                }
            }
#else
            total_log_prob += cache[sampled] - logsumexp;
#endif
            logits_offset += A;
        }
    }

    logprobs[idx] = from_float(total_log_prob);
    value_out[idx] = logits[logits_base + fused_cols - 1];
    rng_states[idx] = state;
}

// Index into (L, agents, H): element-parallel over L*count*H.
// state_row is agent index within the state tensor's agent dim.
static __device__ long state_elem_idx(
        int layer, int agents_stride, int state_row, int h, int H) {
    return ((long)layer * agents_stride + state_row) * H + h;
}

// Copy buffer RNN state into rollout initial_states at t=0 (carry path).
// Trainable policy only; src agents are 0..count, dst at dst_start.
__global__ void snapshot_state(Prec dst, Prec src,
        int dst_start, int count) {
    int L = src.shape[0];
    int H = src.shape[2];
    int total = L * count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    long src_i = state_elem_idx(layer, (int)src.shape[1], rel, h, H);
    long dst_i = state_elem_idx(layer, (int)dst.shape[1], dst_start + rel, h, H);
    dst.data[dst_i] = src.data[src_i];
}

// View slot of full (slots, L, A, H) as (L, A, H).
static Prec init_slot(Prec full, int slot) {
    int L = (int)full.shape[1];
    int A = (int)full.shape[2];
    int H = (int)full.shape[3];
    long stride = (long)L * A * H;
    return {.data = full.data + (long)slot * stride, .shape = {L, A, H}};
}

// Zero RNN state for agents that just terminated. Same grid as snapshot
// (L*count*H); non-terminal threads return after one terminal load.
__global__ void zero_term_state(Prec state, Float terminals,
        int state_start, int terminal_start, int count) {
    int L = state.shape[0];
    int H = state.shape[2];
    int total = L * count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    if (terminals.data[terminal_start + rel] == 0.0f) {
        return;
    }
    long i = state_elem_idx(layer, (int)state.shape[1], state_start + rel, h, H);
    state.data[i] = from_float(0.0f);
}

// Select time t, then agents [start, start+count). Rank-2 has F==0 (zero-term shape);
// stride uses max(F, 1). Out shape {count, F} keeps ndim 1 when F==0.
Prec puf_slice(Prec p, int t, int start, int count) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)(t * B + start) * stride_f,
        .shape = {count, F},
    };
}

Float puf_slice(Float p, int t, int start, int count) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)(t * B + start) * stride_f,
        .shape = {count, F},
    };
}

static void pufferl_forward_step(PuffeRL* pufferl, int buf, int t,
        cudaStream_t stream) {
    Hypers* hypers = &pufferl->hypers;
    int graph_slot = hypers->async ? pufferl->write_slot : 0;
    RolloutBuf rollouts = pufferl->rollouts;
    if (hypers->async) {
        rollouts = rollout_time_view(&pufferl->rollouts,
            pufferl->write_slot * hypers->horizon, hypers->horizon);
    }
    EnvBuf* env = &pufferl->env;
    VecEnv* vec = pufferl->vec;
    int block_size = hypers->total_agents / hypers->num_buffers;
    int start = buf * block_size;
    int* layout = vec->policy_layout;

    // Copy observations, rewards, terminals from GPU env buffers to rollout buffer
    ObsTensor* obs_env = &env->obs;
    int n = block_size * obs_env->shape[1];
    Prec obs_dst = puf_slice(rollouts.observations, t, start, block_size);
    // Env obs -> rollout: D2D if same type, else cast (float/uchar → precision_t).
    if (sizeof(obs_t) == sizeof(precision_t)) {
        cudaMemcpyAsync(obs_dst.data,
            obs_env->data + (long)start * obs_env->shape[1],
            n * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
    } else {
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            obs_dst.data, obs_env->data + (long)start * obs_env->shape[1], n);
    }

    Prec rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    Prec term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    cast_rew_term<<<grid_size(block_size), BLOCK_SIZE, 0, stream>>>(
        rew_dst.data, env->rewards.data + start,
        term_dst.data, env->terminals.data + start, block_size);

    // Mask always allocated (env-written or synthetic all-ones). Continuous ignores it in sample.
    int mask_size = rollouts.action_mask.shape[2];
    int mask_stride = mask_size;
    Prec mask_slice = puf_slice(rollouts.action_mask, t, start, block_size);
    cast<<<grid_size(block_size * mask_size), BLOCK_SIZE, 0, stream>>>(
        mask_slice.data,
        env->action_mask.data + (long)start * mask_size,
        block_size * mask_size);

    // Per-policy forward: layout[b]..layout[b+1) within each buffer chunk.
    long act_cols = env->actions.shape[1];
    for (int b = 0; b < pufferl->num_policies; b++) {
        int off = layout[b];
        int n = layout[b + 1] - off;
        if (n == 0) {
            continue;
        }

        Policy* pol = &pufferl->policies[b];
        Weights* w = (!pol->frozen && hypers->async)
            ? &pufferl->actor_weights : &pol->weights;
        Activations* acts = &pol->buf_acts[buf];
        Prec* st = &pol->buffer_states[buf];

        int sub = start + off;
        Prec obs_b  = puf_slice(rollouts.observations, t, sub, n);
        Float act_b = puf_slice(rollouts.actions,      t, sub, n);
        Prec lp_b   = puf_slice(rollouts.logprobs,     t, sub, n);
        Prec val_b  = puf_slice(rollouts.values,       t, sub, n);
        Prec mask_b = puf_slice(rollouts.action_mask,  t, sub, n);

        // Per-policy state is compact (n agents); local index 0..n-1.
        int state_n = (int)st->shape[0] * n * (int)st->shape[2];
        zero_term_state<<<grid_size(state_n), BLOCK_SIZE, 0, stream>>>(
            *st, env->terminals, 0, sub, n);

        // Carry path: snapshot trainable policy state into per-slot initial_states.
        if (!pol->frozen && t == 0 && rollouts.initial_states.data != NULL) {
            Prec slot_st = init_slot(rollouts.initial_states, graph_slot);
            snapshot_state<<<grid_size(state_n), BLOCK_SIZE, 0, stream>>>(
                slot_st, *st, sub, n);
        }

        Prec dec = arch_forward(&pol->arch, *w, *acts, obs_b, *st, stream);

        Prec p_logstd = {};
        DecoderWeights* dw = (DecoderWeights*)w->decoder;
        if (dw->continuous) {
            p_logstd = dw->logstd;
        }

        // Offset RNG by off so policies don't collide on per-buffer rng slots.
        sample_logits<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            dec, p_logstd, pufferl->act_sizes,
            act_b.data, env->actions.data + (long)sub * act_cols,
            lp_b.data, val_b.data,
            pufferl->rng_states[buf] + off,
            mask_b.data, mask_stride);
    }
}

void pufferl_forward(PuffeRL* pufferl, int buf, int t, cudaStream_t stream) {
    Hypers* hypers = &pufferl->hypers;
    // GPU rollout graphs the whole horizon; CPU still graphs one net step.
    bool step_graph = hypers->cudagraphs && pufferl->rollout_graphs != NULL;
    int graph_slot = hypers->async ? pufferl->write_slot : 0;
    int graph = (graph_slot * hypers->horizon + t) * hypers->num_buffers + buf;
    profile_begin("fused_rollout", hypers->profile);

    if (step_graph && pufferl->rollout_graphs[graph] != NULL) {
        assert(cudaGraphLaunch(
            pufferl->rollout_graphs[graph], stream) == cudaSuccess
            && "cudaGraphLaunch failed");
        profile_end(hypers->profile);
        return;
    }

    if (step_graph) {
        int block = hypers->total_agents / hypers->num_buffers;
        int H = pufferl->policies[0].arch.encoder.out_dim;
        drive_lt_prepare(block, H, stream);
        assert(cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess
            && "cudaStreamBeginCapture failed");
    }
    pufferl_forward_step(pufferl, buf, t, stream);
    if (step_graph) {
        cudaGraph_t _graph;
        assert(cudaStreamEndCapture(stream, &_graph) == cudaSuccess
                && "cudaStreamEndCapture failed");
        assert(cudaGraphInstantiate(&pufferl->rollout_graphs[graph], _graph, 0)
                == cudaSuccess && "cudaGraphInstantiate failed");
        cudaGraphDestroy(_graph);
        assert(cudaGraphLaunch(pufferl->rollout_graphs[graph], stream) == cudaSuccess
                && "cudaGraphLaunch failed");
    }
    profile_end(hypers->profile);
}

double wall_clock() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Log is a flat float struct (host + device).
constexpr int LOG_NF = (int)(sizeof(Log) / sizeof(float));

// Shared host/device: fold Log into float acc; optionally clear the source.
__host__ __device__ static inline void log_accum(float* acc, Log* log, int clear) {
    if (log->n != 0.0f) {
        const float* el = (const float*)log;
        for (int j = 0; j < LOG_NF; j++) {
            acc[j] += el[j];
        }
    }
    if (clear) {
        float* el = (float*)log;
        for (int j = 0; j < LOG_NF; j++) {
            el[j] = 0.0f;
        }
    }
}

__global__ void log_reduce(Env* envs,
        float* out, int total_agents, int clear) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    float local[LOG_NF] = {};
    for (int i = tid; i < total_agents / envs->num_agents; i += blockDim.x) {
        log_accum(local, &envs[i].log, clear);
    }
    for (int j = 0; j < LOG_NF; j++) {
        sh[j * blockDim.x + tid] = local[j];
    }
    block_reduce_sum(sh, out, tid, blockDim.x, LOG_NF);
}

static void env_log_sum(VecEnv* vec, Log* out, int clear) {
    if (PUF_BACKEND == PUF_GPU) {
        log_reduce<<<1, 256, LOG_NF * 256 * sizeof(float)>>>(
            vec->envs, vec->log_scratch, vec->total_agents, clear);
        cudaMemcpy(out, vec->log_scratch, sizeof(Log), cudaMemcpyDeviceToHost);
        return;
    }
    float* acc = (float*)out;
    for (int i = 0; i < vec->size; i++) {
        log_accum(acc, &vec->envs[i].log, clear);
    }
}

#if PUF_BACKEND != PUF_GPU
static void puf_bind_stream(cudaStream_t) {}
static Env* puf_vec_create(int, Dict*, obs_t*, float*, float*, float*) {
    return NULL;
}
#endif


// Map index for env_i of nenvs copies. The first (nenvs % num_maps)
// maps get one extra copy. Returns -1 when every copy uses map_id.
static int env_map_slot(int env_i, int nenvs, int num_maps) {
    if (num_maps <= 1 || nenvs <= 1) {
        return -1;
    }
    if (env_i < 0) {
        env_i = 0;
    }
    if (env_i >= nenvs) {
        env_i = nenvs - 1;
    }
    int extra = nenvs % num_maps;
    int big = nenvs / num_maps + 1;
    if (extra == 0) {
        return env_i / (nenvs / num_maps);
    }
    if (env_i < extra * big) {
        return env_i / big;
    }
    int small = nenvs / num_maps;
    if (small < 1) {
        return env_i;
    }
    return extra + (env_i - extra * big) / small;
}

static void env_setup(PuffeRL* p, VecEnv* vec, Dict* vk, Dict* ek) {
    if (PUF_BACKEND == PUF_GPU) {
        assert(vec->buffers == 1 && "GPU env: num_buffers must be 1");
        vec->envs = puf_vec_create(vec->total_agents, ek,
            p->env.obs.data, p->env.actions.data,
            p->env.rewards.data, p->env.terminals.data);
        cudaMalloc((void**)&vec->log_scratch, sizeof(Log));
        cudaMemset(vec->log_scratch, 0, sizeof(Log));
        vec->policy_layout[0] = 0;
        vec->policy_layout[1] = vec->agents_per_buf;
        return;
    }
    int total_agents = vec->total_agents;
    int num_buffers = vec->buffers;
    int apb = vec->agents_per_buf;
    vec->num_workers = dict_get(vk, "num_threads") / num_buffers;
    if (vec->num_workers < 1) {
        vec->num_workers = 1;
    }
    vec->env_starts = (int*)calloc(1, num_buffers * sizeof(int));
    vec->env_counts = (int*)calloc(1, num_buffers * sizeof(int));
    int num_envs = 0;
#ifdef MY_VEC_INIT
    vec->envs = my_vec_init(&num_envs, vec->env_starts,
        vec->env_counts, vk, ek);
#else
    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    int agents_created = 0;
    int num_maps = 1;
    DictItem* maps_item = dict_find(ek, "num_maps");
    if (maps_item && maps_item->value > 1.0) {
        num_maps = (int)maps_item->value;
    }
    int agents_per = 1;
    DictItem* agents_item = dict_find(ek, "num_agents");
    if (agents_item && agents_item->value >= 1.0) {
        agents_per = (int)agents_item->value;
    }
    int nenvs_est = (total_agents + agents_per - 1) / agents_per;
    if (nenvs_est < 1) {
        nenvs_est = 1;
    }
    while (agents_created < total_agents) {
        envs[num_envs].rng = num_envs;
        Dict* ek_i = ek;
        int slot = env_map_slot(num_envs, nenvs_est, num_maps);
        if (slot >= 0) {
            ek_i = (Dict*)calloc(1, sizeof(Dict));
            assert(ek_i);
            dict_copy(ek_i, ek);
            // Files are map_000.bin .. map_{num_maps-1}.bin. Each copy
            // keeps this dict so a later env cannot change its map_id.
            dict_set(ek_i, "map_id", (double)slot);
        }
        puf_init(&envs[num_envs], ek_i);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }
    envs = (Env*)realloc(envs, num_envs * sizeof(Env));
    int buf = 0, buf_agents = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        vec->env_counts[buf]++;
        if (buf_agents >= apb && buf < num_buffers - 1) {
            buf++;
            vec->env_starts[buf] = i + 1;
            buf_agents = 0;
        }
    }
    vec->envs = envs;
#endif
    vec->size = num_envs;

    size_t obs_bytes = total_agents * OBS_SIZE * sizeof(obs_t);
    size_t mask_bytes = total_agents * vec->mask_size * sizeof(unsigned char);
    cudaHostAlloc((void**)&vec->observations, obs_bytes, cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->action_mask, mask_bytes, cudaHostAllocPortable);
    memset(vec->action_mask, 1, mask_bytes);

    float frozen_pct = dict_get(vk, "hist_policy_percent");
    for (int buf = 0; buf < num_buffers; buf++) {
        int buf_start = buf * apb;
        int env_start = vec->env_starts[buf];
        int env_count = vec->env_counts[buf];
        int frozen_start = env_count;
        if (vec->num_policies > 1 && frozen_pct > 0.0f) {
            frozen_start = env_count - (int)(frozen_pct * env_count);
        }
        int* counts = (int*)calloc(vec->num_policies, sizeof(int));
        for (int e = 0; e < env_count; e++) {
            Env* eptr = &vec->envs[env_start + e];
            for (int s = 0; s < eptr->num_agents; s++) {
                int policy = e < frozen_start ? 0 : eptr->agents[s].policy;
                assert(policy >= 0 && policy < vec->num_policies);
                counts[policy]++;
            }
        }
        int offset = 0;
        for (int b = 0; b <= vec->num_policies; b++) {
            if (buf == 0) {
                vec->policy_layout[b] = offset;
            } else {
                assert(vec->policy_layout[b] == offset);
            }
            if (b < vec->num_policies) {
                offset += counts[b];
            }
        }
        assert(offset == apb);
        int* cursors = (int*)calloc(vec->num_policies, sizeof(int));
        for (int b = 0; b < vec->num_policies; b++) {
            cursors[b] = buf_start + vec->policy_layout[b];
        }
        for (int e = 0; e < env_count; e++) {
            Env* eptr = &vec->envs[env_start + e];
            int tag = 0;
            for (int s = 0; s < eptr->num_agents; s++) {
                int policy = e < frozen_start ? 0 : eptr->agents[s].policy;
                if (policy > tag) {
                    tag = policy;
                }
                int phys = cursors[policy]++;
                Agent* a = &eptr->agents[s];
                a->observations = vec->observations + (size_t)phys * OBS_SIZE;
                a->actions = vec->actions + (size_t)phys * NUM_ATNS;
                a->rewards = vec->rewards + phys;
                a->terminals = vec->terminals + phys;
                a->action_mask = vec->action_mask + (size_t)phys * vec->mask_size;
            }
            eptr->tag = tag;
            eptr->boundary_reached = 0;
        }
        free(cursors);
        free(counts);
    }
}

static void cpu_upload(PuffeRL* p, int start, int n, cudaStream_t stream) {
    VecEnv* v = p->vec;
    EnvBuf* e = &p->env;
    long mask = v->mask_size;
    cudaMemcpyAsync(e->obs.data + (size_t)start * OBS_SIZE,
        v->observations + (size_t)start * OBS_SIZE,
        n * OBS_SIZE * sizeof(obs_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->rewards.data + start, v->rewards + start,
        n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->terminals.data + start, v->terminals + start,
        n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->action_mask.data + (size_t)start * mask,
        v->action_mask + (size_t)start * mask,
        n * mask * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
}

// CPU worker handshake. Atomic on worker_state[]; calloc leaves BUF_STARTING.
enum {
    BUF_STARTING,
    BUF_WAITING,
    BUF_RUNNING,
};

typedef struct {
    PuffeRL* pufferl;
    int buf;
} VecThreadArg;

static void* vec_thread_main(void* arg) {
    VecThreadArg* a = (VecThreadArg*)arg;
    PuffeRL* pufferl = a->pufferl;
    VecEnv* vec = pufferl->vec;
    int buf = a->buf;
    int* state = &vec->worker_state[buf];
    int horizon = pufferl->hypers.horizon;
    cudaSetDevice(pufferl->hypers.gpu_id);
    int apb = vec->agents_per_buf;
    int agent_start = buf * apb;
    int env_start = vec->env_starts[buf];
    int env_count = vec->env_counts[buf];
    Env* envs = vec->envs;
    cudaStream_t stream = pufferl->streams[buf];
    cublas_init_stream(stream);
    cudaEvent_t ev[NUM_EV];
    for (int i = 0; i < NUM_EV; i++) {
        cudaEventCreate(&ev[i]);
    }
    __atomic_store_n(state, BUF_WAITING, __ATOMIC_SEQ_CST);
    float* my_accum = &vec->accum[buf * NUM_PROF];
    struct timespec t0, t1;
    float ms = 0.0f;
    while (true) {
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_RUNNING) {
            if (__atomic_load_n(&vec->shutdown, __ATOMIC_SEQ_CST)) {
                for (int i = 0; i < NUM_EV; i++) {
                    cudaEventDestroy(ev[i]);
                }
                return NULL;
            }
        }
        if (__atomic_load_n(&vec->shutdown, __ATOMIC_SEQ_CST)) {
            for (int i = 0; i < NUM_EV; i++) {
                cudaEventDestroy(ev[i]);
            }
            return NULL;
        }
        int h2d_pending = 0;
        for (int t = 0; t < horizon; t++) {
            cudaEventRecord(ev[MODEL_START], stream);
            pufferl_forward(pufferl, buf, t, stream);
            cudaEventRecord(ev[MODEL_END], stream);
            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &pufferl->env.actions.data[agent_start * NUM_ATNS],
                apb * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ev[COPY_END], stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&ms, ev[MODEL_START], ev[MODEL_END]);
            my_accum[PROF_MODEL] += ms;
            cudaEventElapsedTime(&ms, ev[MODEL_END], ev[COPY_END]);
            my_accum[PROF_COPY] += ms;
            if (h2d_pending) {
                cudaEventElapsedTime(&ms, ev[H2D_START], ev[H2D_END]);
                my_accum[PROF_COPY] += ms;
                h2d_pending = 0;
            }
            memset(&vec->rewards[agent_start], 0, apb * sizeof(float));
            memset(&vec->terminals[agent_start], 0, apb * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &t0);
            #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
            for (int i = env_start; i < env_start + env_count; i++) {
                puf_step(&envs[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[PROF_ENV] += (t1.tv_sec - t0.tv_sec) * 1000.0f
                + (t1.tv_nsec - t0.tv_nsec) / 1e6f;
            cudaEventRecord(ev[H2D_START], stream);
            cpu_upload(pufferl, agent_start, apb, stream);
            cudaEventRecord(ev[H2D_END], stream);
            h2d_pending = 1;
        }
        cudaStreamSynchronize(stream);
        if (h2d_pending) {
            cudaEventElapsedTime(&ms, ev[H2D_START], ev[H2D_END]);
            my_accum[PROF_COPY] += ms;
        }
        __atomic_store_n(state, BUF_WAITING, __ATOMIC_SEQ_CST);
    }
}

// Fresh episodes + zero RNN carry. Workers sit in BUF_WAITING between rollouts.
static void env_restart(PuffeRL* p) {
    VecEnv* vec = p->vec;
    if (PUF_BACKEND == PUF_GPU) {
        puf_reset(vec->envs);
    } else {
        #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
        for (int i = 0; i < vec->size; i++) {
            puf_reset(&vec->envs[i]);
        }
        cpu_upload(p, 0, vec->total_agents, p->default_stream);
    }
    for (int b = 0; b < p->num_policies; b++) {
        for (int i = 0; i < vec->buffers; i++) {
            Prec* st = &p->policies[b].buffer_states[i];
            cudaMemset(st->data, 0, numel(st->shape) * sizeof(precision_t));
        }
    }
    cudaDeviceSynchronize();
}

static void env_start(PuffeRL* p) {
    VecEnv* vec = p->vec;
    if (PUF_BACKEND == PUF_GPU) {
        env_restart(p);
        return;
    }
    vec->worker_state = (int*)calloc(1, vec->buffers * sizeof(int));
    vec->threads = (pthread_t*)calloc(1, vec->buffers * sizeof(pthread_t));
    VecThreadArg* args = (VecThreadArg*)calloc(1,
        vec->buffers * sizeof(VecThreadArg));
    vec->accum = (float*)calloc(1, vec->buffers * NUM_PROF * sizeof(float));
    env_restart(p);
    for (int i = 0; i < vec->buffers; i++) {
        args[i].pufferl = p;
        args[i].buf = i;
        pthread_create(&vec->threads[i], NULL, vec_thread_main, &args[i]);
    }
    for (int i = 0; i < vec->buffers; i++) {
        int* state = &vec->worker_state[i];
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_WAITING) {}
    }
}

static void rollout_start(PuffeRL* p) {
    if (PUF_BACKEND == PUF_GPU) {
        cudaStream_t stream = p->streams[0];
        puf_bind_stream(stream);
        cudaEvent_t* ev = p->profile.rollout_ev;
        int slot = p->write_slot;
        bool first = p->hypers.cudagraphs && p->gpu_rollout_graph[slot] == NULL;
        p->profile.skip_rollout_time = first;
        if (p->hypers.cudagraphs && !first) {
            cudaEventRecord(ev[0], stream);
            cudaGraphLaunch(p->gpu_rollout_graph[slot], stream);
            cudaEventRecord(ev[1], stream);
            return;
        }
        double t_cap = 0;
        if (first) {
            t_cap = wall_clock();
            int block = p->hypers.total_agents / p->hypers.num_buffers;
            int H = p->policies[0].arch.encoder.out_dim;
            drive_lt_prepare(block, H, stream);
            assert(cudaStreamBeginCapture(
                stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess
                && "cudaStreamBeginCapture failed");
        }
        int H = p->hypers.horizon;
        for (int t = 0; t < H; t++) {
            int base = t * EV_T;
            cudaEventRecord(ev[base + MODEL_START], stream);
            pufferl_forward_step(p, 0, t, stream);
            cudaEventRecord(ev[base + MODEL_END], stream);
            puf_step(p->vec->envs);
            cudaEventRecord(ev[base + ENV_END], stream);
        }
        if (first) {
            cudaGraph_t graph;
            assert(cudaStreamEndCapture(stream, &graph) == cudaSuccess
                && "cudaStreamEndCapture failed");
            assert(cudaGraphInstantiate(
                &p->gpu_rollout_graph[slot], graph, 0)
                == cudaSuccess && "cudaGraphInstantiate failed");
            cudaGraphDestroy(graph);
            double dt = wall_clock() - t_cap;
            p->start_time += dt;
            p->last_log_time += dt;
            cudaGraphLaunch(p->gpu_rollout_graph[slot], stream);
        }
        return;
    }
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        int* state = &p->vec->worker_state[buf];
        __atomic_store_n(state, BUF_RUNNING, __ATOMIC_SEQ_CST);
    }
}

void rollout_finish(PuffeRL* p, double t0) {
    if (PUF_BACKEND == PUF_GPU) {
        cudaStreamSynchronize(p->streams[0]);
        float model_ms = 0.0f, env_ms = 0.0f, ms;
        cudaEvent_t* ev = p->profile.rollout_ev;
        if (p->profile.skip_rollout_time) {
            p->profile.skip_rollout_time = 0;
            return;
        }
        if (p->hypers.cudagraphs && p->gpu_rollout_graph[p->write_slot] != NULL) {
            cudaEventElapsedTime(&ms, ev[0], ev[1]);
            p->profile.accum[PROF_ROLLOUT] += ms;
            return;
        }
        int H = p->hypers.horizon;
        for (int t = 0; t < H; t++) {
            int base = t * EV_T;
            cudaEventElapsedTime(&ms, ev[base + MODEL_START], ev[base + MODEL_END]);
            model_ms += ms;
            cudaEventElapsedTime(&ms, ev[base + MODEL_END], ev[base + ENV_END]);
            env_ms += ms;
        }
        p->profile.accum[PROF_MODEL] += model_ms;
        p->profile.accum[PROF_ENV] += env_ms;
        p->profile.accum[PROF_ROLLOUT] += model_ms + env_ms;
        return;
    }
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        int* state = &p->vec->worker_state[buf];
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_WAITING) {}
    }
    float sec = (float)(wall_clock() - t0);
    p->profile.accum[PROF_ROLLOUT] += sec * 1000.0f;
    float eval_prof[NUM_PROF] = {0};
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        float* src = &p->vec->accum[buf * NUM_PROF];
        for (int i = 0; i < NUM_PROF; i++) {
            eval_prof[i] += src[i];
        }
        memset(src, 0, NUM_PROF * sizeof(float));
    }
    p->profile.accum[PROF_MODEL] += eval_prof[PROF_MODEL] / p->vec->buffers;
    p->profile.accum[PROF_ENV] += eval_prof[PROF_ENV] / p->vec->buffers;
    p->profile.accum[PROF_COPY] += eval_prof[PROF_COPY] / p->vec->buffers;
}

static void env_close(VecEnv* vec) {
    if (PUF_BACKEND == PUF_GPU) {
        puf_close(vec->envs);
        cudaFree(vec->log_scratch);
        return;
    }
    __atomic_store_n(&vec->shutdown, 1, __ATOMIC_SEQ_CST);
    for (int i = 0; i < vec->buffers; i++) {
        pthread_join(vec->threads[i], NULL);
    }
    for (int i = 0; i < vec->size; i++) {
        puf_close(&vec->envs[i]);
    }
#ifdef MY_VEC_CLOSE
    my_vec_close(vec->envs);
#endif
}

void vec_log(VecEnv* vec, Dict* out, int clear) {
    // Mean score of finished episodes on each town, before the sum clears logs.
    float town_score[4] = {0};
    float town_n[4] = {0};
    int town_seen[4] = {0};
    if (PUF_BACKEND != PUF_GPU) {
        for (int i = 0; i < vec->size; i++) {
            int id = vec->envs[i].map_id;
            if (id < 0 || id > 3 || !drive_score_town_key(id)) {
                continue;
            }
            town_seen[id] = 1;
            town_score[id] += vec->envs[i].log.score;
            town_n[id] += vec->envs[i].log.n;
        }
    }
    Log aggregate = {0};
    float* acc = (float*)&aggregate;
    env_log_sum(vec, &aggregate, clear);

    float n = aggregate.n;
    Dict env_out = {0};
    if (n > 0.0f) {
        for (int j = 0; j < LOG_NF; j++) {
            acc[j] /= n;
        }
        puf_log(&aggregate, &env_out);
        for (int id = 0; id < 4; id++) {
            if (!town_seen[id]) {
                continue;
            }
            float mean = 0.0f;
            if (town_n[id] > 0.0f) {
                mean = town_score[id] / town_n[id];
            }
            dict_set(&env_out, drive_score_town_key(id), mean);
        }
    }
    dict_set(&env_out, "n", n);
    for (int i = 0; i < env_out.size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out.items[i].key);
        dict_set(out, key, env_out.items[i].value);
    }
    dict_clear(&env_out);
}

static Prec slice_rows(Prec p, int off, int n) {
    int row = 1;
    for (int i = 1; i < PUF_MAX_DIMS && p.shape[i]; i++) {
        row *= (int)p.shape[i];
    }
    Prec out = p;
    out.data = p.data + (int64_t)off * row;
    out.shape[0] = n;
    return out;
}

static Float slice_rows(Float p, int off, int n) {
    int row = 1;
    for (int i = 1; i < PUF_MAX_DIMS && p.shape[i]; i++) {
        row *= (int)p.shape[i];
    }
    Float out = p;
    out.data = p.data + (int64_t)off * row;
    out.shape[0] = n;
    return out;
}

// Transpose (A, B, C) → (B, A, C). Sequential, coalesced on dest rows.
// Two types: actions are float32 (large discrete IDs); everything else is Prec.
__global__ void transpose_102(precision_t* dst, const precision_t* src,
        int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}

#if !defined(PRECISION_FLOAT)
__global__ void transpose_102(float* dst, const float* src, int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}
#endif

// Cosine decay base → min over t in [0, T). Double for t/T (float loses
// precision past 2^24). Caller passes epoch and total train epochs.
float cosine_annealing(float base, float min_v, long t, long T) {
    double u = (double)t / (double)T;
    return min_v + 0.5f * (base - min_v) * (1.0f + (float)cos(M_PI * u));
}

__global__ void clamp_precision_kernel(precision_t* dst, float lo, float hi, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        float v = to_float(dst[idx]);
        dst[idx] = from_float(fminf(fmaxf(v, lo), hi));
    }
}

// 1-thread graph node. cudaEventElapsedTime does not timestamp in-graph events.
__global__ void puf_stamp(unsigned long long* dst) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *dst = t;
}

// (obs - mu) / sig. BC stores sig < 1e-6 as 1; keep that floor here too.
__global__ void anchor_norm(precision_t* dst, const precision_t* obs,
        const precision_t* mu, const precision_t* sig, int nt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * n) {
        return;
    }
    int col = idx % n;
    float s = to_float(sig[col]);
    if (s < 1e-6f) {
        s = 1.0f;
    }
    float v = (to_float(obs[idx]) - to_float(mu[col])) / s;
    dst[idx] = from_float(v);
}

__global__ void anchor_bias_lrelu(precision_t* x, const precision_t* b,
        int nt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * n) {
        return;
    }
    float v = to_float(x[idx]) + to_float(b[idx % n]);
    if (v < 0.0f) {
        v *= 0.01f;
    }
    x[idx] = from_float(v);
}

// GEMM writes a packed (nt, A) head. Place it at the action offset.
__global__ void anchor_pack(precision_t* logits, const precision_t* scratch,
        const precision_t* bias, int nt, int A, int A_total, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * A) {
        return;
    }
    int row = idx / A;
    int col = idx % A;
    float v = to_float(scratch[idx]) + to_float(bias[col]);
    logits[row * A_total + offset + col] = from_float(v);
}

static void anchor_gemm(precision_t* x, int nt, int k,
        precision_t* w, int n, precision_t* y, cudaStream_t stream,
        cublasHandle_t handle) {
    Prec a = {.data = x, .shape = {nt, k}};
    Prec b = {.data = w, .shape = {k, n}};
    Prec c = {.data = y, .shape = {nt, n}};
    puf_mm_nn(&a, &b, &c, stream, 1.0f, 0.0f, handle);
}

// obs is flat (nt, in_dim), same order as the squeezed train minibatch.
static void anchor_forward(Anchor* a, precision_t* obs, int nt,
        cudaStream_t stream, cublasHandle_t handle) {
    assert(nt > 0 && nt <= a->max_nt);
    anchor_norm<<<grid_size(nt * a->in_dim), BLOCK_SIZE, 0, stream>>>(
        a->norm, obs, a->mu, a->sig, nt, a->in_dim);
    precision_t* src = a->norm;
    int src_n = a->in_dim;
    for (int i = 0; i < a->n_layers; i++) {
        AnchorLayer* layer = &a->layer[i];
        precision_t* dst = a->hid[i & 1];
        anchor_gemm(src, nt, src_n, layer->W, layer->out, dst,
            stream, handle);
        int n = layer->out;
        anchor_bias_lrelu<<<grid_size(nt * n), BLOCK_SIZE, 0, stream>>>(
            dst, layer->b, nt, n);
        src = dst;
        src_n = n;
    }
    int offset = 0;
    for (int h = 0; h < NUM_ATNS; h++) {
        int A = a->heads[h];
        anchor_gemm(src, nt, src_n, a->head_w[h], A, a->scratch,
            stream, handle);
        anchor_pack<<<grid_size(nt * A), BLOCK_SIZE, 0, stream>>>(
            a->logits, a->scratch, a->head_b[h],
            nt, A, a->a_total, offset);
        offset += A;
    }
}

struct AnchorFork {
    Anchor* anchor;
    precision_t* obs;
    int nt;
};

// Recorded on the train stream just after the encoder GEMM, so minGRU and
// PPO overlap this MLP. Joined before the distance kernel. Not the dW
// stream: backward reuses that stream and its events in this same graph.
static void anchor_fork(void* ctx, cudaStream_t main) {
    AnchorFork* f = (AnchorFork*)ctx;
    Anchor* a = f->anchor;
    cudaEventRecord(a->fork_ev, main);
    cudaStreamWaitEvent(a->stream, a->fork_ev, 0);
    anchor_forward(a, f->obs, f->nt, a->stream, a->handle);
    cudaEventRecord(a->done_ev, a->stream);
}

static void anchor_join(Anchor* a, cudaStream_t main) {
    cudaStreamWaitEvent(main, a->done_ev, 0);
}

static void anchor_open_stream(Anchor* a) {
    cublas_init_one(&a->handle, &a->workspace);
    assert(cudaStreamCreateWithFlags(
        &a->stream, cudaStreamNonBlocking) == cudaSuccess);
    cublasSetStream(a->handle, a->stream);
    assert(cudaEventCreateWithFlags(
        &a->fork_ev, cudaEventDisableTiming) == cudaSuccess);
    assert(cudaEventCreateWithFlags(
        &a->done_ev, cudaEventDisableTiming) == cudaSuccess);
}

static void train_epoch_gpu(PuffeRL* pufferl, RolloutBuf src, int slot,
        cudaStream_t stream) {
    Hypers* hypers = &pufferl->hypers;
    RolloutBuf* rollouts = &pufferl->train_rollouts;
    unsigned long long* st = pufferl->profile.stamps + slot * NUM_TE;
    puf_stamp<<<1, 1, 0, stream>>>(st + TE_S);

    int T = src.observations.shape[0];
    int B = src.observations.shape[1];
    int obs_size = (int)src.observations.shape[2];
    int num_atns = (int)src.actions.shape[2];
    int mask_c = src.action_mask.shape[2];
    transpose_102<<<grid_size(T * B * obs_size), BLOCK_SIZE, 0, stream>>>(
        rollouts->observations.data, src.observations.data, T, B, obs_size);
    transpose_102<<<grid_size(T * B * num_atns), BLOCK_SIZE, 0, stream>>>(
        rollouts->actions.data, src.actions.data, T, B, num_atns);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, stream>>>(
        rollouts->logprobs.data, src.logprobs.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, stream>>>(
        rollouts->rewards.data, src.rewards.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, stream>>>(
        rollouts->terminals.data, src.terminals.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, stream>>>(
        rollouts->values.data, src.values.data, T, B, 1);
    transpose_102<<<grid_size(T * B * mask_c), BLOCK_SIZE, 0, stream>>>(
        rollouts->action_mask.data, src.action_mask.data, T, B, mask_c);

    clamp_precision_kernel<<<grid_size(
        numel(rollouts->rewards.shape)), BLOCK_SIZE, 0, stream>>>(
        rollouts->rewards.data, -1.0f, 1.0f, numel(rollouts->rewards.shape));

    if (hypers->reset_every_horizon || src.initial_states.data == NULL) {
        cudaMemsetAsync(pufferl->train_state.data, 0,
            numel(pufferl->train_state.shape) * sizeof(precision_t), stream);
    } else {
        Prec slot_st = init_slot(src.initial_states, slot);
        cudaMemcpyAsync(pufferl->train_state.data, slot_st.data,
            numel(pufferl->train_state.shape) * sizeof(precision_t),
            cudaMemcpyDeviceToDevice, stream);
    }
    puf_stamp<<<1, 1, 0, stream>>>(st + TE_MID);

    int batch_size = hypers->total_agents * hypers->horizon;
    int mb_segs = hypers->minibatch_size / hypers->horizon;
    int total_minibatches = hypers->replay_ratio * batch_size / hypers->minibatch_size;
    int n_rows = (int)rollouts->observations.shape[0];
    int Nmb = (int)pufferl->train_buf.mb_advantages.shape[0];
    int Tmb = (int)pufferl->train_buf.mb_advantages.shape[1];
    constexpr int ADV_THREADS = 64;
    int adv_grid = (Nmb + ADV_THREADS - 1) / ADV_THREADS;
    Policy* primary = &pufferl->policies[0];
    for (int mb = 0; mb < total_minibatches; ++mb) {
        int dest_off = (mb * mb_segs) % n_rows;
        TrainGraph graph = pufferl->train_buf;
        graph.mb_obs = slice_rows(rollouts->observations, dest_off, Nmb);
        graph.mb_actions = slice_rows(rollouts->actions, dest_off, Nmb);
        graph.mb_logprobs = slice_rows(rollouts->logprobs, dest_off, Nmb);
        graph.mb_terminals = slice_rows(rollouts->terminals, dest_off, Nmb);
        graph.mb_rewards = slice_rows(rollouts->rewards, dest_off, Nmb);
        graph.mb_values = slice_rows(rollouts->values, dest_off, Nmb);
        graph.mb_action_mask = slice_rows(rollouts->action_mask, dest_off, Nmb);
        graph.mb_state = pufferl->train_state;
        DecoderWeights* dw_train = (DecoderWeights*)primary->weights.decoder;
        Prec p_logstd = {};
        if (dw_train->continuous) {
            p_logstd = dw_train->logstd;
        }
        ArchHook hook = NULL;
        AnchorFork fork = {};
        if (pufferl->anchor.on) {
            assert((int)graph.mb_obs.shape[2] == pufferl->anchor.in_dim);
            fork.anchor = &pufferl->anchor;
            fork.obs = graph.mb_obs.data;
            fork.nt = Nmb * Tmb;
            hook = anchor_fork;
        }
        Prec dec = arch_forward_train(&primary->arch, primary->weights,
            pufferl->train_activs, graph.mb_obs, graph.mb_state,
            graph.mb_terminals, dest_off, graph, p_logstd,
            pufferl->act_sizes, pufferl->ppo_bufs.grad_logits.data,
            pufferl->ppo_bufs.grad_values.data, stream, hook, &fork);
        puff_advantage<<<adv_grid, ADV_THREADS, 0, stream>>>(
            graph.mb_gae_v.data, graph.mb_rewards.data,
            graph.mb_terminals.data,
            hypers->vtrace ? graph.mb_imp.data : NULL,
            graph.mb_advantages.data, graph.mb_gae_v.data,
            hypers->gamma, hypers->gae_lambda,
            hypers->vtrace_rho_clip, hypers->vtrace_c_clip, Nmb, Tmb);
        graph.mb_returns = graph.mb_gae_v;

        ppo_loss_fwd_bwd(dec, p_logstd, graph,
            pufferl->act_sizes, pufferl->losses,
            hypers->clip_coef, hypers->vf_clip_coef, hypers->vf_coef,
            pufferl->ppo_bufs.ent_coef,
            pufferl->ppo_bufs, pufferl->is_continuous, stream);

        // Logit grads are PPO's. Wait for the anchor logits, then add
        // the frozen term before backward reads those grads.
        if (pufferl->anchor.on) {
            int nt = Nmb * Tmb;
            int A_total = (int)dec.shape[2] - 1;
            assert(A_total == pufferl->anchor.a_total);
            anchor_join(&pufferl->anchor, stream);
            anchor_distance(dec.data, pufferl->anchor.logits,
                pufferl->ppo_bufs.grad_logits.data, pufferl->act_sizes,
                pufferl->losses, pufferl->anchor.partials,
                A_total, nt, pufferl->anchor.dist,
                pufferl->anchor.coef, stream);
        }

        Float grad_logits = pufferl->ppo_bufs.grad_logits;
        Float grad_logstd = pufferl->is_continuous
            ? pufferl->ppo_bufs.grad_logstd : Float();
        Float grad_values = pufferl->ppo_bufs.grad_values;
        arch_backward(&primary->arch, primary->weights, pufferl->train_activs,
            grad_logits, grad_logstd, grad_values, stream);

        if (pufferl->nccl_comm != NULL && hypers->world_size > 1) {
            ncclAllReduce(pufferl->grad.data, pufferl->grad.data,
                numel(pufferl->grad.shape), NCCL_PRECISION, ncclAvg,
                pufferl->nccl_comm, stream);
        }
        muon_step(&pufferl->muon, primary->master_weights,
            pufferl->grad, hypers->max_grad_norm, stream);
        if (USE_BF16) {
            int n = numel(primary->param.shape);
            cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                primary->param.data, primary->master_weights.data, n);
        }
    }
    puf_stamp<<<1, 1, 0, stream>>>(st + TE_E);
}

void train_impl(PuffeRL* pufferl, RolloutBuf* src_arg) {
    Hypers* hypers = &pufferl->hypers;
    RolloutBuf src = src_arg ? *src_arg : pufferl->rollouts;
    cudaStream_t train_stream = pufferl->train_stream;

    int batch_size = hypers->total_agents * hypers->horizon;
    bool anneal_lr = hypers->anneal_lr;
    int current_epoch = pufferl->epoch;
    Muon* muon = &pufferl->muon;

    // Schedule over this rank's train epochs (same as outer loop), not global
    // total_timesteps/batch — multi-GPU would otherwise only traverse 1/W of the
    // cosine and never reach min_lr / min_ent.
    int total_epochs = hypers->total_timesteps / hypers->world_size / batch_size;
    if (anneal_lr) {
        float lr_min = hypers->min_lr_ratio * hypers->lr;
        float lr = cosine_annealing(hypers->lr, lr_min, current_epoch, total_epochs);
        cudaMemcpy(muon->lr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Annealed entropy coefficient — same cosine shape as lr. With PG signal
    // alive, the entropy bonus that kept early-training exploratory becomes
    // load-bearing dead weight late in training; cosine-decay frees the policy
    // to commit harder on what it has already learned.
    float current_ent_coef = hypers->ent_coef;
    if (hypers->anneal_ent_coef) {
        float ent_min = hypers->min_ent_coef_ratio * hypers->ent_coef;
        current_ent_coef = cosine_annealing(
            hypers->ent_coef, ent_min, current_epoch, total_epochs);
    }
    // Host write + H2D stay outside the graph; device ptr is what kernels read.
    cudaMemcpyAsync(pufferl->ppo_bufs.ent_coef, &current_ent_coef,
        sizeof(float), cudaMemcpyHostToDevice, train_stream);

    int slot = hypers->async ? pufferl->async_ready_slot : 0;
    int total_minibatches = hypers->replay_ratio * batch_size / hypers->minibatch_size;
    bool first = hypers->cudagraphs && pufferl->train_cudagraph[slot] == NULL;
    profile_begin("train_forward_backward", hypers->profile);
    if (first) {
        double t_cap = wall_clock();
        int H = pufferl->policies[0].arch.encoder.out_dim;
        drive_lt_prepare(hypers->minibatch_size, H, train_stream);
        assert(cudaStreamBeginCapture(
            train_stream, cudaStreamCaptureModeThreadLocal)
            == cudaSuccess && "cudaStreamBeginCapture failed");
        train_epoch_gpu(pufferl, src, slot, train_stream);
        cudaGraph_t graph;
        assert(cudaStreamEndCapture(train_stream, &graph)
            == cudaSuccess && "cudaStreamEndCapture failed");
        assert(cudaGraphInstantiate(
            &pufferl->train_cudagraph[slot], graph, 0)
            == cudaSuccess && "cudaGraphInstantiate failed");
        cudaGraphDestroy(graph);
        double dt = wall_clock() - t_cap;
        pufferl->start_time += dt;
        pufferl->last_log_time += dt;
    }
    if (hypers->cudagraphs) {
        cudaGraphLaunch(pufferl->train_cudagraph[slot], train_stream);
    } else {
        train_epoch_gpu(pufferl, src, slot, train_stream);
    }
    profile_end(hypers->profile);

    cudaStreamSynchronize(train_stream);

    if (total_minibatches > 0) {
        unsigned long long h[NUM_TE];
        cudaMemcpy(h, pufferl->profile.stamps + slot * NUM_TE,
            sizeof(h), cudaMemcpyDeviceToHost);
        pufferl->profile.accum[PROF_TRAIN_MISC] +=
            (float)(h[TE_MID] - h[TE_S]) * 1e-6f;
        pufferl->profile.accum[PROF_TRAIN_MODEL] +=
            (float)(h[TE_E] - h[TE_MID]) * 1e-6f;
    }
    pufferl->epoch += 1;
}

// Load policy weights (flat fp32 checkpoint). Safe between rollouts —
// graphs hold the pointer, not a copy of the data.
// Checkpoint I/O (load/save weights).
void mkdir_p(const char* path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            assert((mkdir(tmp, 0777) == 0 || errno == EEXIST)
                && "failed to create directory");
            *p = '/';
        }
    }
    assert((mkdir(tmp, 0777) == 0 || errno == EEXIST)
        && "failed to create directory");
}

void puf_find_latest_checkpoint(const char* dir,
        char* out, size_t out_size, time_t* best_time) {
    DIR* dp = opendir(dir);
    if (!dp) {
        return;
    }

    struct dirent* ent = NULL;
    while ((ent = readdir(dp))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            puf_find_latest_checkpoint(path, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode)) {
            size_t plen = strlen(path);
            if (plen < 4 || strcmp(path + plen - 4, ".bin") != 0) continue;
            if (st.st_ctime < *best_time) continue;
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", path);
        }
    }

    closedir(dp);
}

const char* puf_checkpoint_path_key(Ini* ini, const char* key,
        char* out, size_t out_size) {
    const char* load_path = puf_ini_get_str(ini, "base", key);
    if (!load_path || strcmp(load_path, "None") == 0) {
        return NULL;
    }

    if (strcmp(load_path, "latest") != 0) {
        return load_path;
    }

    char root[2048];
    snprintf(root, sizeof(root), "%s/%s",
        puf_ini_get_str(ini, "base", "checkpoint_dir"),
        puf_ini_get_str(ini, "base", "env_name"));

    out[0] = 0;
    time_t best_time = 0;
    puf_find_latest_checkpoint(root, out, out_size, &best_time);
    assert(out[0] && "no .bin checkpoints found");
    return out;
}

void puf_save_weights(PuffeRL* p, const char* path) {
    Float mw = p->policies[0].master_weights;
    int64_t nbytes = numel(mw.shape) * sizeof(float);
    char* buf = (char*)malloc(nbytes);
    cudaMemcpy(buf, mw.data, nbytes, cudaMemcpyDeviceToHost);
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE* fp = fopen(tmp, "wb");
    assert(fp && "failed to open weights for writing");
    assert(fwrite(buf, 1, nbytes, fp) == (size_t)nbytes
        && "failed to write weights");
    fclose(fp);
    free(buf);
    assert(rename(tmp, path) == 0 && "failed to publish weights");
}

void puf_load_weights_into(Float dst, Prec params,
        cudaStream_t stream, const char* path) {
    int64_t nbytes = numel(dst.shape) * sizeof(float);
    FILE* fp = fopen(path, "rb");
    assert(fp && "failed to open weights for reading");
    char* buf = (char*)malloc(nbytes);
    size_t nread = fread(buf, 1, nbytes, fp);
    fclose(fp);
    assert((int64_t)nread == nbytes && "failed to read weights");
    cudaMemcpy(dst.data, buf, nbytes, cudaMemcpyHostToDevice);
    free(buf);
    if (USE_BF16) {
        int n = numel(params.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(params.data, dst.data, n);
    }
}

// Load weights into policies[i] (full index; 0 = trainable, i>0 = frozen).
void pufferl_load_policy(PuffeRL* pufferl, int i, const char* path) {
    assert(i >= 0 && i < pufferl->num_policies);
    Policy* pol = &pufferl->policies[i];
    puf_load_weights_into(pol->master_weights, pol->param,
        pufferl->default_stream, path);
    cudaDeviceSynchronize();
}

// fp32 master weights: alias param buffer in float mode; separate fp32 copy in bf16.
// cast_now: copy param→master now (primary after init). Frozen policies load later.
static void master_weights_setup(Float* mw, Prec* param,
        bool cast_now, cudaStream_t stream) {
    long n = numel(param->shape);
    if (USE_BF16) {
        *mw = (Float){.shape = {n}};
        cudaMalloc((void**)&mw->data, n * sizeof(float));
        if (cast_now) {
            cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(mw->data, param->data, n);
        }
    } else {
        *mw = (Float){.data = (float*)param->data, .shape = {n}};
    }
}

#define ANCHOR_ZIP_ENTS 32
#define ANCHOR_NAME 64

struct AnchorZipEnt {
    char name[ANCHOR_NAME];
    long data;
    int nbytes;
};

struct AnchorZip {
    unsigned char* buf;
    long n;
    AnchorZipEnt ent[ANCHOR_ZIP_ENTS];
    int nent;
};

struct AnchorNpy {
    const unsigned char* data;
    int rows;
    int cols;
    int n;
    int fortran;
    char kind;
};

static uint16_t anchor_u16(const unsigned char* p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

static uint32_t anchor_u32(const unsigned char* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static uint64_t anchor_u64(const unsigned char* p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static void anchor_zip_open(AnchorZip* z, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("anchor npz %s\n", path);
        assert(0 && "anchor npz open");
    }
    assert(fseek(f, 0, SEEK_END) == 0);
    long n = ftell(f);
    assert(n > 0 && n < 256L * 1024 * 1024);
    rewind(f);
    z->buf = (unsigned char*)malloc((size_t)n);
    assert(z->buf);
    assert(fread(z->buf, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    z->n = n;
    z->nent = 0;
    long off = 0;
    while (off + 30 <= n) {
        if (anchor_u32(z->buf + off) != 0x04034b50u) {
            break;
        }
        int flags = anchor_u16(z->buf + off + 6);
        int method = anchor_u16(z->buf + off + 8);
        uint32_t csize32 = anchor_u32(z->buf + off + 18);
        uint32_t usize32 = anchor_u32(z->buf + off + 22);
        int namelen = anchor_u16(z->buf + off + 26);
        int extralen = anchor_u16(z->buf + off + 28);
        assert(method == 0 && "anchor npz compressed");
        assert((flags & 8) == 0 && "anchor npz descriptor");
        assert(z->nent < ANCHOR_ZIP_ENTS);
        assert(namelen > 0 && namelen < ANCHOR_NAME);
        uint64_t csize = csize32;
        if (csize32 == 0xffffffffu || usize32 == 0xffffffffu) {
            const unsigned char* extra = z->buf + off + 30 + namelen;
            int left = extralen;
            int found = 0;
            while (left >= 4) {
                int hid = anchor_u16(extra);
                int hsz = anchor_u16(extra + 2);
                assert(4 + hsz <= left);
                if (hid == 1) {
                    int q = 0;
                    if (usize32 == 0xffffffffu) {
                        q += 8;
                    }
                    if (csize32 == 0xffffffffu) {
                        csize = anchor_u64(extra + 4 + q);
                    }
                    found = 1;
                    break;
                }
                extra += 4 + hsz;
                left -= 4 + hsz;
            }
            assert(found && "anchor zip64 extra");
        }
        assert(csize < 256ull * 1024 * 1024);
        AnchorZipEnt* e = &z->ent[z->nent++];
        memcpy(e->name, z->buf + off + 30, namelen);
        e->name[namelen] = 0;
        e->data = off + 30 + namelen + extralen;
        e->nbytes = (int)csize;
        assert(e->data + e->nbytes <= n);
        off = e->data + e->nbytes;
    }
    assert(z->nent > 0 && "anchor npz empty");
}

static int anchor_zip_index(AnchorZip* z, const char* key) {
    char want[ANCHOR_NAME];
    snprintf(want, sizeof(want), "%s.npy", key);
    for (int i = 0; i < z->nent; i++) {
        if (strcmp(z->ent[i].name, want) == 0) {
            return i;
        }
    }
    return -1;
}

static void anchor_npz_missing(AnchorZip* z, const char* key) {
    printf("anchor npz missing %s (%d entries)\n", key, z->nent);
    for (int k = 0; k < z->nent; k++) {
        printf("  %s\n", z->ent[k].name);
    }
    assert(0 && "anchor npz key");
}

static AnchorNpy anchor_npy(const unsigned char* p, int nbytes) {
    assert(nbytes >= 10);
    assert(p[0] == 0x93 && memcmp(p + 1, "NUMPY", 5) == 0);
    int major = p[6];
    int hlen = 0;
    int hoff = 10;
    if (major == 1) {
        hlen = anchor_u16(p + 8);
    } else if (major == 2) {
        hlen = (int)anchor_u32(p + 8);
        hoff = 12;
    } else {
        assert(0 && "anchor npy version");
    }
    assert(hlen > 0 && hlen < 1024);
    assert(hoff + hlen <= nbytes);
    char hdr[1024];
    memcpy(hdr, p + hoff, hlen);
    hdr[hlen] = 0;
    const char* d = strstr(hdr, "'descr': '<");
    assert(d && "anchor npy descr");
    d += 11;
    assert((d[0] == 'f' || d[0] == 'i') && d[1] == '4');
    const char* s = strstr(hdr, "'shape': (");
    assert(s);
    s += 10;
    int dims[2] = {0, 0};
    int nd = 0;
    while (*s && *s != ')') {
        while (*s == ' ' || *s == ',') {
            s++;
        }
        if (*s == ')') {
            break;
        }
        char* end = NULL;
        long v = strtol(s, &end, 10);
        assert(end != s && nd < 2);
        dims[nd++] = (int)v;
        s = end;
    }
    AnchorNpy o;
    memset(&o, 0, sizeof(o));
    o.kind = d[0];
    o.fortran = strstr(hdr, "fortran_order': True") != NULL;
    if (nd == 0) {
        o.rows = 1;
    } else if (nd == 1) {
        o.rows = dims[0];
    } else {
        o.rows = dims[0];
        o.cols = dims[1];
    }
    o.n = o.cols ? o.rows * o.cols : o.rows;
    assert(o.n > 0);
    o.data = p + hoff + hlen;
    assert(hoff + hlen + o.n * 4 <= nbytes);
    return o;
}

static float* anchor_load_f32(AnchorZip* z, const char* key,
        int* rows, int* cols) {
    int i = anchor_zip_index(z, key);
    if (i < 0) {
        anchor_npz_missing(z, key);
    }
    AnchorNpy np = anchor_npy(z->buf + z->ent[i].data, z->ent[i].nbytes);
    assert(np.kind == 'f');
    float* src = (float*)malloc((size_t)np.n * sizeof(float));
    assert(src);
    memcpy(src, np.data, (size_t)np.n * sizeof(float));
    *rows = np.rows;
    *cols = np.cols;
    if (!(np.fortran && np.cols)) {
        return src;
    }
    float* dst = (float*)malloc((size_t)np.n * sizeof(float));
    assert(dst);
    for (int col = 0; col < np.cols; col++) {
        for (int row = 0; row < np.rows; row++) {
            size_t f = (size_t)row + (size_t)col * np.rows;
            dst[(size_t)row * np.cols + col] = src[f];
        }
    }
    free(src);
    return dst;
}

static int anchor_load_i32(AnchorZip* z, const char* key) {
    int i = anchor_zip_index(z, key);
    if (i < 0) {
        anchor_npz_missing(z, key);
    }
    AnchorNpy np = anchor_npy(z->buf + z->ent[i].data, z->ent[i].nbytes);
    assert(np.kind == 'i' && np.n == 1);
    int v = 0;
    memcpy(&v, np.data, 4);
    return v;
}

static void anchor_h2d_into(precision_t* dst, const float* src, int n,
        cudaStream_t stream) {
    float* stage = NULL;
    assert(cudaMalloc(&stage, (size_t)n * sizeof(float)) == cudaSuccess);
    cudaMemcpyAsync(stage, src, (size_t)n * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, stage, n);
    cudaStreamSynchronize(stream);
    cudaFree(stage);
}

static void anchor_h2d(precision_t** dst, const float* src, int n,
        cudaStream_t stream) {
    assert(cudaMalloc(dst, (size_t)n * sizeof(precision_t)) == cudaSuccess);
    anchor_h2d_into(*dst, src, n, stream);
}

static float anchor_max_abs(const float* x, const float* y, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(x[i] - y[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static void anchor_d2h(float* dst, precision_t* src, int n,
        cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("anchor sync %s\n", cudaGetErrorString(err));
        assert(0 && "anchor sync");
    }
    precision_t* tmp = (precision_t*)malloc((size_t)n * sizeof(precision_t));
    assert(tmp);
    cudaMemcpy(tmp, src, (size_t)n * sizeof(precision_t),
        cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        dst[i] = to_float(tmp[i]);
    }
    free(tmp);
}

static void anchor_cpu_row(float* dst, const float* obs,
        float** W, float** b, int* lin, int* lout, int n_layers,
        float** hw, float** hb, int* heads,
        const float* mu, const float* sig, int in_dim, int cap) {
    float* cur = (float*)malloc((size_t)cap * sizeof(float));
    float* nxt = (float*)malloc((size_t)cap * sizeof(float));
    assert(cur && nxt);
    for (int j = 0; j < in_dim; j++) {
        float s = sig[j];
        if (s < 1e-6f) {
            s = 1.0f;
        }
        cur[j] = (obs[j] - mu[j]) / s;
    }
    int n = in_dim;
    for (int i = 0; i < n_layers; i++) {
        int out = lout[i];
        for (int j = 0; j < out; j++) {
            float v = b[i][j];
            for (int k = 0; k < lin[i]; k++) {
                v += cur[k] * W[i][(size_t)k * out + j];
            }
            if (v < 0.0f) {
                v *= 0.01f;
            }
            nxt[j] = v;
        }
        float* swap = cur;
        cur = nxt;
        nxt = swap;
        n = out;
    }
    int off = 0;
    for (int h = 0; h < NUM_ATNS; h++) {
        int A = heads[h];
        for (int j = 0; j < A; j++) {
            float v = hb[h][j];
            for (int k = 0; k < n; k++) {
                v += cur[k] * hw[h][(size_t)k * A + j];
            }
            dst[off + j] = v;
        }
        off += A;
    }
    free(cur);
    free(nxt);
}

static int anchor_dist_id(const char* s) {
    if (strcmp(s, "kl") == 0) {
        return ANCHOR_KL;
    }
    if (strcmp(s, "reverse_kl") == 0) {
        return ANCHOR_REVERSE_KL;
    }
    if (strcmp(s, "wasserstein") == 0) {
        return ANCHOR_WASSERSTEIN;
    }
    printf("anchor_dist %s\n", s);
    assert(0 && "anchor_dist");
    return 0;
}

// One new observation input. The inserted row is zero, so the old
// forward is unchanged. mu at that slot is 0 and sig is 1.
static float* anchor_insert_row(float* W, int rows, int cols, int at) {
    float* grown = (float*)calloc((size_t)(rows + 1) * (size_t)cols,
        sizeof(float));
    assert(grown);
    assert(at >= 0 && at < rows);
    memcpy(grown, W, (size_t)at * (size_t)cols * sizeof(float));
    memcpy(grown + (size_t)(at + 1) * (size_t)cols,
        W + (size_t)at * (size_t)cols,
        (size_t)(rows - at) * (size_t)cols * sizeof(float));
    free(W);
    return grown;
}

static float* anchor_insert_stat(float* x, int n, int at, float fill) {
    float* grown = (float*)malloc((size_t)(n + 1) * sizeof(float));
    assert(grown);
    assert(at >= 0 && at < n);
    memcpy(grown, x, (size_t)at * sizeof(float));
    grown[at] = fill;
    memcpy(grown + at + 1, x + at, (size_t)(n - at) * sizeof(float));
    free(x);
    return grown;
}

static void anchor_init(PuffeRL* p, Ini* ini, int* act_sizes,
        bool is_continuous) {
    Anchor* a = &p->anchor;
    const char* dist_s = puf_ini_get_str(ini, "train", "anchor_dist");
    const char* path = puf_ini_get_str(ini, "train", "anchor");
    a->dist = anchor_dist_id(dist_s);
    a->coef = (float)puf_ini_get(ini, "train", "anchor_coef");
    a->on = 0;
    if (path[0] == 0) {
        return;
    }
    assert(!is_continuous && "anchor is discrete only");
    assert(NUM_ATNS == 2 && "anchor heads are Wa and Ws");
    size_t stack = 0;
    cudaDeviceGetLimit(&stack, cudaLimitStackSize);
    if (stack < 8192) {
        cudaDeviceSetLimit(cudaLimitStackSize, 8192);
    }
    anchor_distance_selfcheck();

    AnchorZip zip;
    memset(&zip, 0, sizeof(zip));
    anchor_zip_open(&zip, path);

    float* host_w[ANCHOR_MAX_LAYERS];
    float* host_b[ANCHOR_MAX_LAYERS];
    int lin[ANCHOR_MAX_LAYERS];
    int lout[ANCHOR_MAX_LAYERS];
    int n_layers = 0;
    int grew_obs = 0;
    while (n_layers < ANCHOR_MAX_LAYERS) {
        char wk[16];
        char bk[16];
        snprintf(wk, sizeof(wk), "W%d", n_layers);
        snprintf(bk, sizeof(bk), "b%d", n_layers);
        if (anchor_zip_index(&zip, wk) < 0) {
            break;
        }
        int wr = 0;
        int wc = 0;
        int br = 0;
        int bc = 0;
        host_w[n_layers] = anchor_load_f32(&zip, wk, &wr, &wc);
        host_b[n_layers] = anchor_load_f32(&zip, bk, &br, &bc);
        assert(wc > 0 && bc == 0 && br == wc);
        if (n_layers == 0 && wr + 1 == OBS_SIZE) {
            int at = EGO_TARGET_SPEED;
            host_w[n_layers] = anchor_insert_row(host_w[n_layers], wr, wc, at);
            wr = OBS_SIZE;
            grew_obs = 1;
        }
        if (n_layers == 0) {
            assert(wr == OBS_SIZE);
        } else {
            assert(wr == lout[n_layers - 1]);
        }
        lin[n_layers] = wr;
        lout[n_layers] = wc;
        n_layers++;
    }
    assert(n_layers > 0 && "anchor npz has no hidden layers");
    if (n_layers == ANCHOR_MAX_LAYERS) {
        char wk[16];
        snprintf(wk, sizeof(wk), "W%d", n_layers);
        assert(anchor_zip_index(&zip, wk) < 0 && "anchor too many layers");
    }

    int n_accel = anchor_load_i32(&zip, "n_accel");
    int n_steer = anchor_load_i32(&zip, "n_steer");
    assert(n_accel == act_sizes[0] && n_steer == act_sizes[1]);
    int last = lout[n_layers - 1];
    int ar = 0;
    int ac = 0;
    int sr = 0;
    int sc = 0;
    int bar = 0;
    int bac = 0;
    int bsr = 0;
    int bsc = 0;
    float* Wa = anchor_load_f32(&zip, "Wa", &ar, &ac);
    float* ba = anchor_load_f32(&zip, "ba", &bar, &bac);
    float* Ws = anchor_load_f32(&zip, "Ws", &sr, &sc);
    float* bs = anchor_load_f32(&zip, "bs", &bsr, &bsc);
    assert(ar == last && ac == n_accel && bac == 0 && bar == n_accel);
    assert(sr == last && sc == n_steer && bsc == 0 && bsr == n_steer);
    int mr = 0;
    int mc = 0;
    int sgr = 0;
    int sgc = 0;
    float* mu = anchor_load_f32(&zip, "mu", &mr, &mc);
    float* sig = anchor_load_f32(&zip, "sig", &sgr, &sgc);
    if (grew_obs) {
        int at = EGO_TARGET_SPEED;
        assert(mr + 1 == OBS_SIZE && sgr + 1 == OBS_SIZE);
        mu = anchor_insert_stat(mu, mr, at, 0.0f);
        sig = anchor_insert_stat(sig, sgr, at, 1.0f);
        mr = OBS_SIZE;
        sgr = OBS_SIZE;
        printf("anchor inserted a zero-weight target speed at index %d\n", at);
    }
    assert(mr == OBS_SIZE && mc == 0);
    assert(sgr == OBS_SIZE && sgc == 0);

    int max_w = 0;
    for (int i = 0; i < n_layers; i++) {
        if (lout[i] > max_w) {
            max_w = lout[i];
        }
    }
    int heads[2] = {n_accel, n_steer};
    int max_head = n_accel > n_steer ? n_accel : n_steer;
    int a_total = n_accel + n_steer;
    int max_nt = p->hypers.minibatch_size;
    assert(max_nt >= 2 && max_w > 0);

    a->n_layers = n_layers;
    a->in_dim = OBS_SIZE;
    a->width = max_w;
    a->heads[0] = n_accel;
    a->heads[1] = n_steer;
    a->a_total = a_total;
    a->max_nt = max_nt;

    cudaStream_t stream = p->train_stream;
    for (int i = 0; i < n_layers; i++) {
        anchor_h2d(&a->layer[i].W, host_w[i], lin[i] * lout[i], stream);
        anchor_h2d(&a->layer[i].b, host_b[i], lout[i], stream);
        a->layer[i].in = lin[i];
        a->layer[i].out = lout[i];
    }
    anchor_h2d(&a->head_w[0], Wa, last * n_accel, stream);
    anchor_h2d(&a->head_b[0], ba, n_accel, stream);
    anchor_h2d(&a->head_w[1], Ws, last * n_steer, stream);
    anchor_h2d(&a->head_b[1], bs, n_steer, stream);
    anchor_h2d(&a->mu, mu, OBS_SIZE, stream);
    anchor_h2d(&a->sig, sig, OBS_SIZE, stream);
    assert(cudaMalloc(&a->norm,
        (size_t)max_nt * OBS_SIZE * sizeof(precision_t)) == cudaSuccess);
    assert(cudaMalloc(&a->hid[0],
        (size_t)max_nt * max_w * sizeof(precision_t)) == cudaSuccess);
    assert(cudaMalloc(&a->hid[1],
        (size_t)max_nt * max_w * sizeof(precision_t)) == cudaSuccess);
    assert(cudaMalloc(&a->scratch,
        (size_t)max_nt * max_head * sizeof(precision_t)) == cudaSuccess);
    assert(cudaMalloc(&a->logits,
        (size_t)max_nt * a_total * sizeof(precision_t)) == cudaSuccess);
    int blocks = (max_nt + PPO_THREADS - 1) / PPO_THREADS;
    assert(cudaMalloc(&a->partials,
        (size_t)blocks * 2 * sizeof(float)) == cudaSuccess);

    int cap = OBS_SIZE > max_w ? OBS_SIZE : max_w;
    if (max_head > cap) {
        cap = max_head;
    }
    // obs = mu is an exact 0 after the norm and does not touch W0.
    // A constant 0.25 does. mu + 0.25*sig is not that constant in bf16:
    // tiny sigs cancel against mu, so the quarter row uses mu=0, sig=1.
    float* raw = (float*)malloc((size_t)OBS_SIZE * sizeof(float));
    float* cpu = (float*)malloc((size_t)a_total * sizeof(float));
    float* gpu = (float*)malloc((size_t)a_total * sizeof(float));
    float* zmu = (float*)calloc((size_t)OBS_SIZE, sizeof(float));
    float* zsig = (float*)malloc((size_t)OBS_SIZE * sizeof(float));
    assert(raw && cpu && gpu && zmu && zsig);
    float* hw[2] = {Wa, Ws};
    float* hb[2] = {ba, bs};
    for (int j = 0; j < OBS_SIZE; j++) {
        raw[j] = mu[j];
    }
    anchor_cpu_row(cpu, raw, host_w, host_b, lin, lout, n_layers,
        hw, hb, heads, mu, sig, OBS_SIZE, cap);
    precision_t* draw = NULL;
    anchor_h2d(&draw, raw, OBS_SIZE, stream);
    // Same handle the train graph will capture. First gemm is here, not
    // inside the capture (cublas lazy init allocates).
    anchor_open_stream(a);
    anchor_forward(a, draw, 1, a->stream, a->handle);
    anchor_d2h(gpu, a->logits, a_total, a->stream);
    float max0 = anchor_max_abs(gpu, cpu, a_total);

    for (int j = 0; j < OBS_SIZE; j++) {
        raw[j] = 0.25f;
        zsig[j] = 1.0f;
    }
    anchor_cpu_row(cpu, raw, host_w, host_b, lin, lout, n_layers,
        hw, hb, heads, zmu, zsig, OBS_SIZE, cap);
    anchor_h2d_into(a->mu, zmu, OBS_SIZE, stream);
    anchor_h2d_into(a->sig, zsig, OBS_SIZE, stream);
    anchor_h2d_into(draw, raw, OBS_SIZE, stream);
    anchor_forward(a, draw, 1, a->stream, a->handle);
    anchor_d2h(gpu, a->logits, a_total, a->stream);
    float max1 = anchor_max_abs(gpu, cpu, a_total);
    anchor_h2d_into(a->mu, mu, OBS_SIZE, stream);
    anchor_h2d_into(a->sig, sig, OBS_SIZE, stream);
    cudaFree(draw);

    // bf16 rounds the stored weights and each layer. The 704x3 anchor
    // with 257-wide heads differs from this fp32 row by about 0.18 on
    // the zero input and about 1.2 on the quarter input.
    float tol0 = USE_BF16 ? 0.5f : 1e-3f;
    float tol1 = USE_BF16 ? 2.0f : 1e-3f;
    if (max0 > tol0 || max1 > tol1) {
        printf("anchor forward max diff zeros %g quarter %g\n", max0, max1);
        assert(0 && "anchor forward check");
    }
    // Train captures this handle at minibatch_size. cublas allocates on
    // the first call of that shape, which a graph capture is not allowed to do.
    anchor_forward(a, a->norm, a->max_nt, a->stream, a->handle);
    assert(cudaStreamSynchronize(a->stream) == cudaSuccess);
    a->on = 1;
    printf("anchor %s  %s  coef %g  %d x %d\n",
        path, dist_s, a->coef, n_layers, max_w);

    for (int i = 0; i < n_layers; i++) {
        free(host_w[i]);
        free(host_b[i]);
    }
    free(Wa);
    free(ba);
    free(Ws);
    free(bs);
    free(mu);
    free(sig);
    free(raw);
    free(cpu);
    free(gpu);
    free(zmu);
    free(zsig);
    free(zip.buf);
}

PuffeRL* create_pufferl(Ini* ini, TrainContext* ctx) {
    Hypers hypers = {
        .horizon = puf_ini_get(ini, "train", "horizon"),
        .total_agents = puf_ini_get(ini, "vec", "total_agents"),
        .num_buffers = puf_ini_get(ini, "vec", "num_buffers"),
        .hidden_size = puf_ini_get(ini, "policy", "hidden_size"),
        .num_layers = puf_ini_get(ini, "policy", "num_layers"),
        .lr = puf_ini_get(ini, "train", "learning_rate"),
        .min_lr_ratio = puf_ini_get(ini, "train", "min_lr_ratio"),
        .anneal_lr = puf_ini_get(ini, "train", "anneal_lr") != 0,
        .momentum = puf_ini_get(ini, "train", "momentum"),
        .minibatch_size = puf_ini_get(ini, "train", "minibatch_size"),
        .replay_ratio = puf_ini_get(ini, "train", "replay_ratio"),
        .total_timesteps = puf_ini_get(ini, "train", "total_timesteps"),
        .max_grad_norm = puf_ini_get(ini, "train", "max_grad_norm"),
        .clip_coef = puf_ini_get(ini, "train", "clip_coef"),
        .vf_clip_coef = puf_ini_get(ini, "train", "vf_clip_coef"),
        .vf_coef = puf_ini_get(ini, "train", "vf_coef"),
        .ent_coef = puf_ini_get(ini, "train", "ent_coef"),
        .min_ent_coef_ratio = puf_ini_get(ini, "train", "min_ent_coef_ratio"),
        .anneal_ent_coef = puf_ini_get(ini, "train", "anneal_ent_coef") != 0,
        .gamma = puf_ini_get(ini, "train", "gamma"),
        .gae_lambda = puf_ini_get(ini, "train", "gae_lambda"),
        .vtrace = puf_ini_get(ini, "train", "vtrace") != 0,
        .vtrace_rho_clip = puf_ini_get(ini, "train", "vtrace_rho_clip"),
        .vtrace_c_clip = puf_ini_get(ini, "train", "vtrace_c_clip"),
        .async = puf_ini_get(ini, "base", "async") != 0,
        .reset_every_horizon = puf_ini_get(ini, "base", "reset_every_horizon") != 0,
        .cudagraphs = puf_ini_get(ini, "base", "cudagraphs") >= 0,
        .profile = puf_ini_get(ini, "base", "profile") != 0,
        .rank = ctx->rank,
        .world_size = ctx->world_size,
        .gpu_id = ctx->gpu_id,
        .num_threads = puf_ini_get(ini, "vec", "num_threads"),
        .seed = puf_ini_get(ini, "base", "seed"),
    };
    Dict vec_kwargs = {0};
    dict_copy(&vec_kwargs, puf_ini_section(ini, "vec", 0));
    Dict* env_kwargs = puf_ini_section(ini, "env", 0);
    ncclUniqueId* nccl_id = ctx->nccl_id;

    PuffeRL* pufferl = (PuffeRL*)calloc(1, sizeof(PuffeRL));
    pufferl->hypers = hypers;
    snprintf(pufferl->env_name, sizeof(pufferl->env_name), "%s", PUFFER_ENV_NAME);

    cudaSetDevice(hypers.gpu_id);
    cublas_init_handle();

    if (hypers.world_size > 1) {
        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, *nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    pufferl->seed = (ulong)hypers.seed + hypers.rank;

    // GPU tensors allocated into pufferl->env; vec aliases them.
    int total_agents = dict_get(&vec_kwargs, "total_agents");
    int num_buffers = dict_get(&vec_kwargs, "num_buffers");
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buf = total_agents / num_buffers;
    // Total policies including trainable policies[0]. Default 1 = train only.
    int num_policies = dict_get(&vec_kwargs, "num_policies");
    if (num_policies < 1) {
        num_policies = 1;
    }
    // GPU envs have no per-env tags / frozen layout — selfplay and match stay CPU-only.
    assert(!(PUF_BACKEND == PUF_GPU
            && (num_policies > 1 || puf_ini_get(ini, "selfplay", "enabled")))
        && "GPU env backend does not support selfplay or multi-policy (match)");

    // Discrete action layout. Continuous dims are size 1. Mask width is act_n.
    int num_action_heads = NUM_ATNS;
    int act_sizes[] = ACT_SIZES;
    int act_n = 0;
    int n_cont = 0;
    int n_disc = 0;
    for (int i = 0; i < num_action_heads; i++) {
        if (act_sizes[i] == 1) {
            n_cont++;
        } else {
            n_disc++;
        }
        act_n += act_sizes[i];
    }
    assert(!(n_cont > 0 && n_disc > 0)
        && "mixed continuous/discrete action spaces not supported");
    bool is_continuous = n_cont > 0;
    pufferl->is_continuous = is_continuous;
    vec->num_policies = num_policies;
    vec->policy_layout = (int*)calloc(1, (vec->num_policies + 1) * sizeof(int));
    vec->mask_size = act_n;

    // Device env IO (EnvBuf).
    pufferl->env = {
        .obs =         {.shape = {total_agents, OBS_SIZE}},
        .actions =     {.shape = {total_agents, NUM_ATNS}},
        .rewards =     {.shape = {total_agents}},
        .terminals =   {.shape = {total_agents}},
        .action_mask = {.shape = {total_agents, act_n}},
    };
    EnvBuf* env = &pufferl->env;
    size_t mask_bytes = total_agents * act_n * sizeof(unsigned char);
    cudaMalloc((void**)&env->obs.data, total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMalloc((void**)&env->actions.data, total_agents * NUM_ATNS * sizeof(float));
    cudaMalloc((void**)&env->rewards.data, total_agents * sizeof(float));
    cudaMalloc((void**)&env->terminals.data, total_agents * sizeof(float));
    cudaMalloc((void**)&env->action_mask.data, mask_bytes);
    cudaMemset(env->obs.data, 0, total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMemset(env->actions.data, 0, total_agents * NUM_ATNS * sizeof(float));
    cudaMemset(env->rewards.data, 0, total_agents * sizeof(float));
    cudaMemset(env->terminals.data, 0, total_agents * sizeof(float));
    cudaMemset(env->action_mask.data, 1, mask_bytes);

    env_setup(pufferl, vec, &vec_kwargs, env_kwargs);
    pufferl->vec = vec;

    cudaMalloc((void**)&pufferl->profile.stamps,
        2 * NUM_TE * sizeof(unsigned long long));
    if (PUF_BACKEND == PUF_GPU) {
        int H = hypers.horizon;
        pufferl->profile.rollout_ev = (cudaEvent_t*)calloc(
            1, EV_T * H * sizeof(cudaEvent_t));
        for (int i = 0; i < EV_T * H; i++) {
            assert(cudaEventCreate(&pufferl->profile.rollout_ev[i]) == cudaSuccess);
        }
    }
    nvmlInit();
    nvmlDeviceGetHandleByIndex(hypers.gpu_id, &pufferl->nvml_device);

    int input_size = OBS_SIZE;
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;
    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int B_TT = minibatch_segments * hypers.horizon;
    int horizon = hypers.horizon;
    int agents_per_buf = total_agents / num_buffers;

    // Dedicated learner stream (always non-default; nonblocking when async).
    if (hypers.async) {
        assert(cudaStreamCreateWithFlags(
            &pufferl->train_stream, cudaStreamNonBlocking) == cudaSuccess);
    } else {
        assert(cudaStreamCreate(&pufferl->train_stream) == cudaSuccess);
    }
    cublas_init_stream(pufferl->train_stream);

    Allocator* acts = &pufferl->activ_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // All policies: policies[0] trainable, rest historical/frozen (rollout-only).
    int hist_hidden = dict_get(&vec_kwargs, "hist_policy_hidden_size");
    int hist_layers = dict_get(&vec_kwargs, "hist_policy_num_layers");
    pufferl->num_policies = vec->num_policies;
    assert(!(pufferl->num_policies > 1 && (hist_hidden <= 0 || hist_layers <= 0))
        && "num_policies > 1 requires hist_policy_hidden_size and hist_policy_num_layers > 0");
    pufferl->policies = (Policy*)calloc(1, pufferl->num_policies * sizeof(Policy));

    for (int b = 0; b < pufferl->num_policies; b++) {
        Policy* pol = &pufferl->policies[b];
        pol->frozen = (b > 0);
        int h = pol->frozen ? hist_hidden : hidden_size;
        int L = pol->frozen ? hist_layers : num_layers;
        int slice = vec->policy_layout[b + 1] - vec->policy_layout[b];
        assert(slice > 0 && "policy has no agents");

        pol->arch = build_arch(input_size, h, L,
            decoder_output_size, is_continuous, hypers.horizon);
        pol->weights = weights_create(&pol->arch, &pol->params_alloc);
        Allocator* aalloc = pol->frozen ? &pol->activ_alloc : acts;
        pol->buf_acts = (Activations*)calloc(
            1, num_buffers * sizeof(Activations));
        pol->buffer_states = (Prec*)calloc(1, num_buffers * sizeof(Prec));
        for (int i = 0; i < num_buffers; i++) {
            pol->buf_acts[i] = arch_reg_rollout(
                &pol->arch, pol->weights, aalloc, slice);
            pol->buffer_states[i] = {.shape = {L, slice, h}};
            alloc_register(aalloc, &pol->buffer_states[i]);
        }
    }

    // Train-only extras on policies[0] (trainable).
    Policy* primary = &pufferl->policies[0];
    if (hypers.async) {
        pufferl->actor_weights = weights_create(
            &primary->arch, &pufferl->weight_alloc);
    }
    pufferl->train_activs = arch_reg_train(
        &primary->arch, primary->weights, acts, grads, B_TT);

    // Async is Cleanba 2-slot only (lag 1). Sync uses a single slot.
    int async_slots = hypers.async ? 2 : 1;
    pufferl->async_num_slots = async_slots;
    int rollout_horizon = async_slots * horizon;
    register_rollout_buffers(&pufferl->rollouts,
        acts, rollout_horizon, total_agents, input_size, num_action_heads, act_n);
    // Carry path: per-slot initial RNN states. reset_every_horizon zeros train_state.
    if (!hypers.reset_every_horizon) {
        pufferl->rollouts.initial_states = {
            .shape = {async_slots, num_layers, total_agents, hidden_size}};
        alloc_register(acts, &pufferl->rollouts.initial_states);
    }
    register_train_buffers(pufferl->train_buf, acts, minibatch_segments, horizon);
    register_rollout_buffers(&pufferl->train_rollouts,
        acts, total_agents, horizon, input_size, num_action_heads, act_n);
    register_ppo_buffers(pufferl->ppo_bufs, acts, minibatch_segments,
        hypers.horizon, decoder_output_size, is_continuous);
    pufferl->train_state = {.shape = {num_layers, total_agents, hidden_size}};
    alloc_register(acts, &pufferl->train_state);

    cudaMalloc((void**)&pufferl->rng_offset, (num_buffers + 1) * sizeof(long));
    cudaMemset(pufferl->rng_offset, 0, (num_buffers + 1) * sizeof(long));
    cudaMalloc((void**)&pufferl->act_sizes, num_action_heads * sizeof(int));
    cudaMalloc((void**)&pufferl->losses, NUM_LOSSES * sizeof(float));

    muon_init(&pufferl->muon, &primary->params_alloc, hypers.momentum, acts);

    // Allocate all policy param/activ pools, then train grads + shared acts.
    for (int b = 0; b < pufferl->num_policies; b++) {
        Policy* pol = &pufferl->policies[b];
        alloc_create(&pol->params_alloc);
        if (pol->frozen) {
            alloc_create(&pol->activ_alloc);
        }
        pol->param = {
            .data = (precision_t*)pol->params_alloc.mem,
            .shape = {pol->params_alloc.total_elems},
        };
    }
    if (hypers.async) {
        alloc_create(&pufferl->weight_alloc);
    }
    alloc_create(grads);
    alloc_create(acts);

    pufferl->grad = {.data = (precision_t*)grads->mem, .shape = {grads->total_elems}};
    if (hypers.async) {
        pufferl->actor_param = {
            .data = (precision_t*)pufferl->weight_alloc.mem,
            .shape = {pufferl->weight_alloc.total_elems},
        };
    }

    ulong init_seed = hypers.seed;
    weights_init(&primary->arch,
        primary->weights, &init_seed, pufferl->default_stream);
    // Primary: cast param→master now. Frozen: load later via pufferl_load_policy.
    for (int b = 0; b < pufferl->num_policies; b++) {
        Policy* pol = &pufferl->policies[b];
        master_weights_setup(&pol->master_weights, &pol->param,
            !pol->frozen, pufferl->default_stream);
    }
    if (hypers.async) {
        puf_copy(&pufferl->actor_param, &primary->param, pufferl->default_stream);
        cudaStreamSynchronize(pufferl->default_stream);
    }

    // Per-buffer persistent RNG states
    pufferl->rng_states = (curandStatePhilox4_32_10_t**)calloc(
        1, num_buffers * sizeof(curandStatePhilox4_32_10_t*));
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc((void**)&pufferl->rng_states[i],
            agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        cudaMemset(pufferl->rng_states[i], 0,
            agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
            pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
    }

    // Post-create initialization
    cudaMemcpy(pufferl->act_sizes, act_sizes,
        num_action_heads*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses, 0, NUM_LOSSES * sizeof(float));
    anchor_init(pufferl, ini, act_sizes, is_continuous);
    cudaMemcpy(pufferl->muon.lr, &hypers.lr, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->muon.mb.data, 0, numel(pufferl->muon.mb.shape) * sizeof(float));

#ifdef PUFFER_NETHACK
    nethack_policy_init(ini);
#endif

    // CPU: per-step net graphs. GPU: full-horizon. Both captured on first use.
    if (hypers.cudagraphs && PUF_BACKEND != PUF_GPU) {
        int rollout_graph_slots = pufferl->async_num_slots;
        pufferl->rollout_graphs = (cudaGraphExec_t*)calloc(1,
            rollout_graph_slots*horizon*num_buffers*sizeof(cudaGraphExec_t));
    }
    pufferl->streams = (cudaStream_t*)calloc(1, num_buffers * sizeof(cudaStream_t));
    for (int i = 0; i < num_buffers; i++) {
        if (hypers.async) {
            assert(cudaStreamCreateWithFlags(
                &pufferl->streams[i], cudaStreamNonBlocking) == cudaSuccess);
        } else {
            assert(cudaStreamCreate(&pufferl->streams[i]) == cudaSuccess);
        }
    }
    // CPU rollout workers register their streams in vec_thread_main.
    if (PUF_BACKEND == PUF_GPU) {
        cublas_init_stream(pufferl->streams[0]);
    }

    env_start(pufferl);

    if (hypers.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
    }

    double now = wall_clock();
    pufferl->start_time = now;
    pufferl->last_log_time = now;
    pufferl->last_log_step = 0;

    dict_clear(&vec_kwargs);
    return pufferl;
}

// OS reclaims memory/CUDA context on exit. This is the intended design.
// All memory is allocated up front and static across training.
// Skip cudaDeviceSynchronize/ncclCommDestroy: captured NCCL graphs deadlock DP.
void close_pufferl(PuffeRL* p) {
    if (p->hypers.profile) {
        cudaProfilerStop();
    }
    nvmlShutdown();
    env_close(p->vec);
}

// Dashboard
static int puf_dashboard_tty = 0;
static int puf_dashboard_last_rows = 0;
static int puf_dashboard_last_cols = 0;
static int puf_dashboard_frame = 0;

#define PUF_DASH_W 80
#define PUF_DASH_BASE_ROWS 13
#define PUF_DASH_MAX_USER_ROWS 15

// Colors no-op when not a TTY.
#define PUF_A  (puf_dashboard_tty ? "\033[96m" : "")
#define PUF_W  (puf_dashboard_tty ? "\033[97m" : "")
#define PUF_G  (puf_dashboard_tty ? "\033[90m" : "")
#define PUF_R  (puf_dashboard_tty ? "\033[0m" : "")

static void dash_eol(void) {
    if (puf_dashboard_tty) {
        printf("\033[K");
    }
    putchar('\n');
}

static void dash_end(void) {
    if (puf_dashboard_tty) {
        printf("\033[J\033[?2026l");
    }
    fflush(stdout);
}

// Right-align into w cols (truncate if long). Gray unit letters
static void dash_cell(const char* s, int w) {
    int n = (int)strlen(s);
    if (n > w) {
        n = w;
    }
    int numeric = 0;
    for (int i = 0; i < n; i++) {
        numeric |= (s[i] >= '0' && s[i] <= '9');
    }
    printf("%s%*s", PUF_W, w - n, "");
    for (int i = 0; i < n; i++) {
        int unit = numeric && strchr("%KMBTGdhms", s[i]);
        printf("%s%c", unit ? PUF_G : PUF_W, s[i]);
    }
    printf("%s", PUF_R);
}

static void dash_rule(const char* left, const char* right) {
    printf("%s%s", PUF_W, left);
    for (int i = 0; i < PUF_DASH_W - 2; i++) printf("─");
    printf("%s%s", right, PUF_R);
    dash_eol();
}

static void dash_blank(void) {
    printf("%s│%*s│%s", PUF_W, PUF_DASH_W - 2, "", PUF_R);
    dash_eol();
}

static void dash_abbrev(char* out, size_t n, double val) {
    const char* suf[] = {"", "K", "M", "B", "T"};
    int i = 0;
    while (val >= 1000.0 && i < 4) {
        val /= 1000.0;
        i++;
    }
    snprintf(out, n, "%.1f%s", val, suf[i]);
}

// <1s = ms; sub-hour keeps ms; longer → d/h/m/s.
static void dash_duration(char* out, size_t n, double sec) {
    if (sec < 0) {
        sec = 0;
    }
    long ms = (long)(sec * 1000.0 + 0.5);
    if (ms < 1000) {
        snprintf(out, n, "%ldms", ms);
    } else if (ms < 60000) {
        snprintf(out, n, "%lds %03ldms", ms / 1000, ms % 1000);
    } else if (ms < 3600000) {
        snprintf(out, n, "%ldm %02lds %03ldms",
            ms / 60000, (ms / 1000) % 60, ms % 1000);
    } else {
        long s = ms / 1000;
        snprintf(out, n, "%ldd %ldh %ldm %lds",
            s / 86400, (s / 3600) % 24, (s / 60) % 60, s % 60);
    }
}

// Missing keys keep last train values when merged into last_log.
static double dash_num(Dict* log, const char* key, double fallback) {
    DictItem* it = dict_find(log, key);
    return it ? it->value : fallback;
}

// Last finished episode. A log with only env/n leaves these in place.
#define DASH_ENV_KEEP (2 * PUF_DASH_MAX_USER_ROWS)
static char dash_env_key[DASH_ENV_KEEP][32];
static double dash_env_val[DASH_ENV_KEEP];
static int dash_env_n = 0;

void puf_dashboard_print(Ini* ini, PuffeRL* p, Dict* log, int epoch) {
    puf_dashboard_tty = isatty(STDOUT_FILENO);
    int term_rows = 1000, term_cols = PUF_DASH_W;
    if (puf_dashboard_tty) {
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
            if (ws.ws_row > 0) term_rows = ws.ws_row;
            if (ws.ws_col > 0) term_cols = ws.ws_col;
        }
    }

    int env_n = 0;
    for (int i = 0; i < log->size && env_n < DASH_ENV_KEEP; i++) {
        const char* k = log->items[i].key;
        if (strncmp(k, "env/", 4) != 0 || strcmp(k, "env/n") == 0) {
            continue;
        }
        snprintf(dash_env_key[env_n], sizeof(dash_env_key[env_n]),
            "%.31s", k + 4);
        dash_env_val[env_n] = log->items[i].value;
        env_n++;
    }
    if (env_n > 0) {
        dash_env_n = env_n;
    }

    const char* env_name = puf_ini_get_str(ini, "base", "env_name");
    double steps = dash_num(log, "agent_steps",
        (double)p->global_step * p->hypers.world_size);
    double sps = dash_num(log, "SPS", 0);
    long configured = puf_ini_get(ini, "train", "total_timesteps");
    long local_batch = (long)p->hypers.total_agents * p->hypers.horizon;
    long local_steps = configured / p->hypers.world_size;
    double target = (double)((local_steps / local_batch) * local_batch * p->hypers.world_size);
    double remain_sec = sps > 0 && target > steps ? (target - steps) / sps : 0;
    double rollout = dash_num(log, "perf/rollout", 0);
    double train_time = dash_num(log, "perf/train", 0);
    double perf_total = rollout + train_time;

    char params[32], steps_s[32], sps_s[32], uptime[64], remaining[64], epoch_s[32];
    dash_abbrev(params, sizeof(params), (double)numel(p->policies[0].master_weights.shape));
    dash_abbrev(steps_s, sizeof(steps_s), steps);
    dash_abbrev(sps_s, sizeof(sps_s), sps);
    double up = dash_num(log, "uptime", wall_clock() - p->start_time);
    dash_duration(uptime, sizeof(uptime), up);
    dash_duration(remaining, sizeof(remaining), remain_sec);
    snprintf(epoch_s, sizeof(epoch_s), "%d", epoch);

    if (puf_dashboard_tty) {
        printf("\033[?2026h");
        if (term_rows != puf_dashboard_last_rows || term_cols != puf_dashboard_last_cols) {
            printf("\033[H\033[J");
        } else {
            printf("\033[H");
        }
        puf_dashboard_last_rows = term_rows;
        puf_dashboard_last_cols = term_cols;
    }
    // Tiny terminal: one-line compact summary.
    if (puf_dashboard_tty && (term_cols < PUF_DASH_W || term_rows <= PUF_DASH_BASE_ROWS)) {
        double score = 0;
        for (int i = 0; i < dash_env_n; i++) {
            if (strcmp(dash_env_key[i], "score") == 0) {
                score = dash_env_val[i];
                break;
            }
        }
        char compact[512];
        snprintf(compact, sizeof(compact),
            "PufferLib 5.0  env=%s  steps=%s  SPS=%s  score=%.3f  epoch=%s  to_go=%s",
            env_name, steps_s, sps_s, score, epoch_s, remaining);
        printf("%.*s", term_cols > 1 ? term_cols - 1 : term_cols, compact);
        if (term_rows > 1) {
            dash_eol();
        } else if (puf_dashboard_tty) {
            printf("\033[K");
        }
        dash_end();
        return;
    }

    char gpu[16], vram[32], ram[16];
    snprintf(gpu, sizeof(gpu), "%3.0f%%", dict_get(log, "util/gpu_percent"));
    snprintf(vram, sizeof(vram), "%.1f/%.0fG",
        dict_get(log, "util/vram_used_gb"),
        dict_get(log, "util/vram_total_gb"));
    snprintf(ram, sizeof(ram), "%.1fG", dict_get(log, "util/cpu_mem_gb"));

    int fish_span = 18;
    int fish_pos = (fish_span - 3) - (puf_dashboard_frame++ % (fish_span - 2));

    dash_rule("╭", "╮");
    printf("%s│%s %sPufferLib %s5.0%s%*s%s🐡%s%*s%sGPU%s:%s ",
        PUF_W, PUF_R, PUF_A, PUF_W, PUF_R, fish_pos, "", PUF_A, PUF_R,
        fish_span - 2 - fish_pos, "", PUF_A, PUF_G, PUF_R);
    dash_cell(gpu, 4);
    printf("   %sVRAM%s:%s", PUF_A, PUF_G, PUF_R);
    dash_cell(vram, 10);
    printf("    %sRAM%s:%s", PUF_A, PUF_G, PUF_R);
    dash_cell(ram, 6);
    printf("     %s│%s", PUF_W, PUF_R);
    dash_eol();
    dash_blank();

    // Stats | Perf | Losses — fixed widths sum to 80 with borders/spacing.
    struct {
        const char* a;
        const char* av;
        const char* b;
        const char* perf_key;  // null → use perf_sec
        double perf_sec;
        const char* c;
        const char* loss_key;
    } rows[] = {
        {"Env", env_name, "Evaluate", NULL, rollout, "Losses", "loss/total"},
        {"Params", params, "  Model", "perf/eval_model", 0, "policy", "loss/policy"},
        {"Steps", steps_s, "  Env", "perf/eval_env", 0, "value", "loss/value"},
        {"SPS", sps_s, "  Copy", "perf/eval_copy", 0, "entropy", "loss/entropy"},
        {"Epoch", epoch_s, "Train", NULL, train_time, "old_kl", "loss/old_kl"},
        {"Uptime", uptime, "  Model", "perf/train_model", 0, "kl", "loss/kl"},
        {"To go", remaining, "  Misc", "perf/train_misc", 0, "clipfrac", "loss/clipfrac"},
        {"", "", "", NULL, 0, "anchor", "loss/anchor"},
    };
    for (int i = 0; i < (int)(sizeof(rows) / sizeof(rows[0])); i++) {
        char bt[64], bp[16], cv[32];
        double sec = rows[i].perf_key
            ? dash_num(log, rows[i].perf_key, 0) : rows[i].perf_sec;
        dash_duration(bt, sizeof(bt), sec);
        snprintf(bp, sizeof(bp), "%d%%",
            perf_total > 0 ? (int)(100.0 * sec / perf_total) : 0);
        DictItem* lit = dict_find(log, rows[i].loss_key);
        if (lit) {
            snprintf(cv, sizeof(cv), "%.3f", lit->value);
        } else {
            cv[0] = 0;
        }
        printf("%s│%s %s%-9.9s%s ", PUF_W, PUF_R, PUF_A, rows[i].a, PUF_R);
        dash_cell(rows[i].av, 13);
        printf("    %s%-12.12s%s ", PUF_A, rows[i].b, PUF_R);
        dash_cell(bt, 6);
        putchar(' ');
        dash_cell(bp, 4);
        printf("    %s%-10.10s%s ", PUF_A, rows[i].c, PUF_R);
        dash_cell(cv, 7);
        printf("    %s│%s", PUF_W, PUF_R);
        dash_eol();
    }
    dash_blank();

    int user_rows = PUF_DASH_MAX_USER_ROWS;
    if (puf_dashboard_tty) {
        user_rows = term_rows - PUF_DASH_BASE_ROWS - 1;
        if (user_rows < 0) user_rows = 0;
        if (user_rows > PUF_DASH_MAX_USER_ROWS) user_rows = PUF_DASH_MAX_USER_ROWS;
    }

    int shown = dash_env_n;
    int max_items = 2 * user_rows;
    if (shown > max_items) {
        shown = max_items;
    }
    int pairs = shown / 2;
    for (int r = 0; r < pairs; r++) {
        int i = r * 2;
        char ls[32], rs[32];
        snprintf(ls, sizeof(ls), "%.3f", dash_env_val[i]);
        snprintf(rs, sizeof(rs), "%.3f", dash_env_val[i + 1]);
        printf("%s│%s %s%-25.25s%s %s%9.9s%s   %s%-25.25s%s %s%9.9s%s    %s│%s",
            PUF_W, PUF_R, PUF_A, dash_env_key[i], PUF_R, PUF_W, ls, PUF_R,
            PUF_A, dash_env_key[i + 1], PUF_R, PUF_W, rs, PUF_R, PUF_W, PUF_R);
        dash_eol();
    }
    if (shown % 2) {
        char ls[32];
        snprintf(ls, sizeof(ls), "%.3f", dash_env_val[shown - 1]);
        printf("%s│%s %s%-25.25s%s %s%9.9s%s   %-25.25s %9.9s    %s│%s",
            PUF_W, PUF_R, PUF_A, dash_env_key[shown - 1],
            PUF_R, PUF_W, ls, PUF_R, "", "", PUF_W, PUF_R);
        dash_eol();
    }
    dash_rule("╰", "╯");
    dash_end();
}

void log_util(PuffeRL* p, Dict* out) {
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(p->nvml_device, &util);
    dict_set(out, "util/gpu_percent", (double)util.gpu);

    size_t cuda_free;
    size_t cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    dict_set(out, "util/vram_used_gb",
        (double)(cuda_total - cuda_free) / (1024.0 * 1024.0 * 1024.0));
    dict_set(out, "util/vram_total_gb",
        (double)cuda_total / (1024.0 * 1024.0 * 1024.0));

    long rss_kb = 0;
    FILE* status = fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (fgets(line, sizeof(line), status)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) {
                break;
            }
        }
        fclose(status);
    }
    dict_set(out, "util/cpu_mem_gb", (double)rss_kb / (1024.0 * 1024.0));
}

void trainer_eval_log(PuffeRL* p, Dict* out) {
    p->last_log_time = wall_clock();
    p->last_log_step = p->global_step;
    log_util(p, out);
    vec_log(p->vec, out, 0);
}

// One dict copy per train log (+ optional final snapshot). Capacity fixed at
// train start: ≤ train_epochs + 1 final append.
typedef struct {
    Dict* items;
    int size;
    int capacity;
} PufLogHistory;

static void puf_log_history_init(PufLogHistory* h, int capacity) {
    h->size = 0;
    h->capacity = capacity;
    h->items = (Dict*)calloc(capacity, sizeof(Dict));
}

static void puf_log_history_add(PufLogHistory* h, Dict* log) {
    assert(h->size < h->capacity);
    dict_copy(&h->items[h->size], log);
    h->size++;
}

// Bin-mean of history[key] over agent_steps into out[0..points-1]. Dense keys.
// points==1 → last value only. Last bin forced to final sample.
static void log_history_bin_mean(PufLogHistory* h, const char* key,
        int points, double* out) {
    assert(h->size > 0 && points >= 1);
    if (points == 1) {
        out[0] = dict_get(&h->items[h->size - 1], key);
        return;
    }
    double final_steps = dict_get(&h->items[h->size - 1], "agent_steps");
    int out_idx = 0;
    int bin_n = 0;
    double bin_sum = 0;
    double fallback = dict_get(&h->items[0], key);
    double next_bin = final_steps / (points - 1);
    for (int i = 0; i < h->size; i++) {
        Dict* log = &h->items[i];
        bin_sum += dict_get(log, key);
        bin_n++;
        double steps = dict_get(log, "agent_steps");
        if (steps < next_bin || out_idx >= points - 1) {
            continue;
        }
        fallback = bin_n ? bin_sum / bin_n : fallback;
        out[out_idx++] = fallback;
        bin_n = 0;
        bin_sum = 0;
        next_bin += final_steps / (points - 1);
    }
    out[points - 1] = dict_get(&h->items[h->size - 1], key);
    while (out_idx < points - 1) {
        out[out_idx++] = fallback;
    }
}

double rollout_start(PuffeRL* p, int slot) {
#ifdef PUFFER_NETHACK
    nethack_policy_on_rollout(p->global_step, p->hypers.total_timesteps);
#endif
    p->write_slot = slot;
    if (p->hypers.async) {
        Prec* param = &p->policies[0].param;
        int64_t n = numel(param->shape);
        cudaMemcpyAsync(p->actor_param.data, param->data,
            n * sizeof(precision_t), cudaMemcpyDeviceToDevice, p->default_stream);
        cudaStreamSynchronize(p->default_stream);
    }
    if (p->hypers.reset_every_horizon) {
        for (int b = 0; b < p->num_policies; b++) {
            for (int i = 0; i < p->hypers.num_buffers; i++) {
                Prec* st = &p->policies[b].buffer_states[i];
                cudaMemsetAsync(st->data, 0, numel(st->shape) * sizeof(precision_t),
                    p->default_stream);
            }
        }
        cudaStreamSynchronize(p->default_stream);
    }

    double t0 = wall_clock();
    rollout_start(p);
    return t0;
}
void rollouts(PuffeRL* p) {
    double t0 = rollout_start(p, 0);
    rollout_finish(p, t0);
    p->global_step += p->hypers.horizon * p->hypers.total_agents;
}

typedef struct {
    float score;
    float perf;
    float draw;
    int games;
} EvalResult;

#define TRAIN_RESULT_MAX_POINTS 64
typedef struct {
    float score;
    float cost;
    float steps;
    int points;
    float scores[TRAIN_RESULT_MAX_POINTS];
    float costs[TRAIN_RESULT_MAX_POINTS];
    float step_points[TRAIN_RESULT_MAX_POINTS];
} TrainResult;

#define EVAL_RENDER 0
#define EVAL_SCORE 1
#define EVAL_MATCH 2

#define SELFPLAY_MAX_HIST 8
#define SELFPLAY_MAX_LADDER 16
#define SELFPLAY_PATH_MAX 4096
// One historical opponent ↔ policies[policy_idx] (env tag == policy_idx).
typedef struct {
    int policy_idx;
    long opp_started_step;
} SelfplayHist;

typedef struct {
    int num_hist;
    int max_size;
    long opp_timeout_steps;
    unsigned int rng;
    char (*pool)[SELFPLAY_PATH_MAX];
    int pool_size;
    SelfplayHist hist[SELFPLAY_MAX_HIST];
} Selfplay;

void selfplay_add_checkpoint(Selfplay* sp, const char* path) {
    for (int i = 0; i < sp->pool_size; i++) {
        if (strcmp(sp->pool[i], path) == 0) return;
    }
    if (sp->pool_size == sp->max_size) {
        memmove(sp->pool, sp->pool + 1, (sp->max_size - 1) * sizeof(*sp->pool));
        sp->pool_size--;
    }
    snprintf(sp->pool[sp->pool_size++], sizeof(sp->pool[0]), "%s", path);
}

const char* selfplay_sample(Selfplay* sp) {
    int idx = (int)(rand_r(&sp->rng) % (unsigned int)sp->pool_size);
    return sp->pool[idx];
}

typedef struct {
    char section[64];
    char key[64];
} SweepParam;

extern char** environ;

typedef struct {
    int run;
    int random;
    int gp_obs;
    int pareto;
    int fd;
    pid_t pid;
    float* sample;
    TrainResult result;
} SweepJob;

void run_sweep(Ini* ini, const char* exe_path) {
    // Build SweepSpace + param map from [sweep.<section>.<key>] sections.
    const char* goal = puf_ini_get_str(ini, "sweep", "goal");
    assert((strcmp(goal, "maximize") == 0 || strcmp(goal, "minimize") == 0)
        && "sweep.goal must be maximize or minimize");
    int direction = strcmp(goal, "minimize") == 0 ? -1 : 1;

    SweepParam* params = (SweepParam*)calloc(ini->num_sections, sizeof(SweepParam));
    SweepSpace* space = (SweepSpace*)calloc(1, sizeof(SweepSpace));
    space->spaces = (Space*)calloc((size_t)ini->num_sections, sizeof(Space));
    space->cost_idx = -1;
    space->optimize_direction = direction;
    int n_params = 0;
    for (int i = 0; i < ini->num_sections; i++) {
        Dict* dict = &ini->sections[i];
        if (strncmp(dict->name, "sweep.", 6) != 0) {
            continue;
        }
        const char* path = dict->name + 6;
        const char* dot = strrchr(path, '.');
        assert(dot && dot != path && dot[1]
            && "expected section [sweep.<section>.<key>]");
        snprintf(params[n_params].section, sizeof(params[n_params].section),
            "%.*s", (int)(dot - path), path);
        snprintf(params[n_params].key, sizeof(params[n_params].key), "%s", dot + 1);

        const char* dist = dict_get_str(dict, "distribution");
        SpaceType type = SPACE_LINEAR;
        int is_integer = 0;
        if (strcmp(dist, "uniform") == 0) {
            type = SPACE_LINEAR;
        } else if (strcmp(dist, "int_uniform") == 0) {
            type = SPACE_LINEAR;
            is_integer = 1;
        } else if (strcmp(dist, "uniform_pow2") == 0) {
            type = SPACE_POW2;
            is_integer = 1;
        } else if (strcmp(dist, "log_normal") == 0) {
            type = SPACE_LOG;
        } else if (strcmp(dist, "logit_normal") == 0) {
            type = SPACE_LOGIT;
        } else {
            assert(0 && "invalid sweep distribution (use uniform/int_uniform/"
                "uniform_pow2/log_normal/logit_normal)");
        }

        float min_v = dict_get(dict, "min");
        float max_v = dict_get(dict, "max");
        const char* scale_s = dict_get_str(dict, "scale");
        float scale;
        if (strcmp(scale_s, "auto") == 0) {
            scale = 0.5f;
        } else if (strcmp(scale_s, "time") == 0) {
            assert(min_v > 0 && max_v > 0
                && "scale=time requires positive min/max");
            scale = 1.0f / (log2f(max_v) - log2f(min_v));
        } else {
            scale = dict_get(dict, "scale");
        }

        space->spaces[n_params] = (Space){
            .type = type, .min = min_v, .max = max_v, .scale = scale, .is_integer = is_integer};
        if (strcmp(dict->name, "sweep.train.total_timesteps") == 0) {
            space->cost_idx = n_params;
        }
        n_params++;
    }
    space->num = n_params;

    int max_runs = puf_ini_get(ini, "sweep", "max_runs");
    int downsample = puf_ini_get(ini, "sweep", "downsample");
    int prune_pareto = puf_ini_get(ini, "sweep", "prune_pareto");
    const char* metric_dist = puf_ini_get_str(ini, "sweep", "metric_distribution");
    assert((strcmp(metric_dist, "linear") == 0 || strcmp(metric_dist, "logit") == 0)
        && "sweep.metric_distribution must be linear or logit");
    int use_logit = strcmp(metric_dist, "logit") == 0;
    float max_cost = puf_ini_get(ini, "sweep", "max_suggestion_cost");
    float early_stop_quantile = puf_ini_get(ini, "sweep", "early_stop_quantile");
    const char* sweep_init = puf_ini_get_str(ini, "sweep", "init");
    assert((strcmp(sweep_init, "random") == 0 || strcmp(sweep_init, "current") == 0)
        && "sweep.init must be random or current");
    int random_init = strcmp(sweep_init, "random") == 0;
    const char* sweep_focus = puf_ini_get_str(ini, "sweep", "focus");
    assert((strcmp(sweep_focus, "pareto") == 0 || strcmp(sweep_focus, "score") == 0)
        && "sweep.focus must be pareto or score");
    int score_focus = strcmp(sweep_focus, "score") == 0;
    assert(max_runs >= 1 && "sweep.max_runs must be >= 1");
    assert(downsample >= 1 && downsample <= TRAIN_RESULT_MAX_POINTS
        && "sweep.downsample must be in [1, TRAIN_RESULT_MAX_POINTS]");
    int success_cap = (int)fmaxf((float)(max_runs * downsample * 2), 8192.0f);

    // GPU packing: each trial is a full train (launch_train) that may itself
    // use train.gpus for NCCL DP. Concurrent trials = sweep_gpus / train_gpus
    // on disjoint blocks [0,W), [W,2W), ...  e.g. 8 GPUs, train.gpus=2 → 4 trials.
    int total_gpus = 0;
    assert(cudaGetDeviceCount(&total_gpus) == cudaSuccess && total_gpus >= 1);
    int train_gpus = puf_ini_get(ini, "train", "gpus");
    int sweep_gpus = puf_ini_get(ini, "sweep", "gpus");
    if (sweep_gpus == -1) {
        sweep_gpus = total_gpus;
    }
    assert(train_gpus >= 1 && (float)sweep_gpus == fminf(
        fmaxf((float)sweep_gpus, (float)train_gpus), (float)total_gpus));

    ProteinSweep* protein = protein_sweep_create((ProteinSweep){
        .space = space,
        .num_random_samples = 10,
        .random_init = random_init,
        .score_focus = score_focus,
        .suggestions_per_pareto = 256,
        .gp_training_iter = 50,
        .gp_learning_rate = 0.001f,
        .optimizer_reset_frequency = 50,
        .gp_max_obs = 750,
        .infer_batch_size = 4096,
        .use_success_prob = downsample == 1,
        .prune_pareto = prune_pareto,
        .use_logit = use_logit,
        .global_search_scale = 1.0f,
        .max_suggestion_cost = max_cost,
        .expansion_rate = 0.1f,
        .cost_random_suggestion = -0.8f,
        .early_stop_quantile = early_stop_quantile,
        .success_cap = success_cap,
        .failure_cap = 1024,
        .top_k = 5,
        .rng_seed = 73ULL,
    });

    int parallel = sweep_gpus / train_gpus;
    // Row 0 = staging for suggest; rows 1..parallel = per-slot copies for observe.
    float* samples = (float*)calloc((parallel + 1) * space->num, sizeof(float));
    SweepJob* jobs = (SweepJob*)calloc(parallel, sizeof(SweepJob));
    int failed_workers = 0;
    int next_run_id = 0;
    int completed = 0;
    int active = 0;
    for (int j = 0; j < space->num; j++) {
        float val = puf_ini_get(ini, params[j].section, params[j].key);
        float norm = space_normalize(&space->spaces[j], val);
        if (!(isfinite(norm) && norm >= -1.0f && norm <= 1.0f)) {
            fprintf(stderr,
                "default sweep value outside its sweep range: %s.%s\n",
                params[j].section, params[j].key);
            exit(1);
        }
        protein->origin[j] = norm;
    }
    // Free-list: slot i owns GPUs [i*train_gpus, (i+1)*train_gpus). Refill as
    // soon as any trial exits (waitpid -1), so finished GPUs are never idle
    // while more runs remain.
    while (completed < max_runs || active > 0) {
        for (int i = 0; i < parallel; i++) {
            if (jobs[i].pid || completed + active >= max_runs) {
                continue;
            }

            ProteinSweepInfo info = {0};
            if (!next_run_id) {
                memcpy(samples, protein->origin, space->num * sizeof(float));
            } else {
                // Invalid jobs die and produce "fail" obs
                info = protein_sweep_suggest(protein, samples, NAN);
            }

            SweepJob job = {
                .run = next_run_id++,
                .random = info.is_random,
                .gp_obs = info.n_gp_obs,
                .pareto = info.n_pareto,
                .sample = samples + (size_t)(i + 1) * space->num,
            };
            memcpy(job.sample, samples, space->num * sizeof(float));

            for (int p = 0; p < space->num; p++) {
                float val = space_unnormalize(&space->spaces[p], samples[p]);
                char buf[64];
                snprintf(buf, sizeof(buf), "%.9g", val);
                char key[256];
                snprintf(key, sizeof(key), "%s.%s",
                    params[p].section, params[p].key);
                puf_ini_put(ini, key, buf);
            }
            char run_id[128];
            snprintf(run_id, sizeof(run_id), "sweep_%ld_%04d",
                (long)(1000.0 * wall_clock()), job.run);
            puf_ini_put(ini, "base.run_id", run_id);

            // Spawn train: result pipe + full ini as section.key=value argv.
            // Child may NCCL-fork again if train.gpus > 1.
            int pipefd[2];
            assert(pipe(pipefd) == 0);
            char buf[64];
            snprintf(buf, sizeof(buf), "%d", i * train_gpus);
            puf_ini_put(ini, "base.gpu_offset", buf);
            snprintf(buf, sizeof(buf), "%d", pipefd[1]);
            puf_ini_put(ini, "base.result_fd", buf);

            int nkeys = 0;
            for (int s = 0; s < ini->num_sections; s++) {
                nkeys += ini->sections[s].size;
            }
            char** argv = (char**)calloc(nkeys + 3, sizeof(char*));
            argv[0] = (char*)exe_path;
            argv[1] = (char*)"train";
            int argc = 2;
            char full_key[PUF_DICT_MAX_KEY * 2];
            char val[128];
            for (int s = 0; s < ini->num_sections; s++) {
                Dict* dict = &ini->sections[s];
                for (int k = 0; k < dict->size; k++) {
                    DictItem* item = &dict->items[k];
                    const char* src = item->str;
                    if (!src) {
                        snprintf(val, sizeof(val), "%.17g", item->value);
                        src = val;
                    }
                    snprintf(full_key, sizeof(full_key), "%s.%s",
                        dict->name, item->key);
                    size_t n = strlen(full_key) + strlen(src) + 4;
                    argv[argc] = (char*)malloc(n);
                    snprintf(argv[argc], n, "--%s=%s", full_key, src);
                    argc++;
                }
            }
            argv[argc] = NULL;

            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_addclose(&actions, pipefd[0]);
            posix_spawn_file_actions_addopen(
                &actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
            assert(posix_spawnp(&job.pid, exe_path, &actions, NULL, argv, environ) == 0
                && "posix_spawn train failed");
            posix_spawn_file_actions_destroy(&actions);
            for (int a = 2; a < argc; a++) {
                free(argv[a]);
            }
            free(argv);
            close(pipefd[1]);
            job.fd = pipefd[0];
            jobs[i] = job;
            active++;
        }
        if (!active) {
            break;
        }

        // Reap one finished worker and feed protein (curve points or a fail obs).
        int status = 0;
        pid_t done = waitpid(-1, &status, 0);
        assert(done > 0 && "sweep waitpid failed");
        SweepJob* job = NULL;
        for (int j = 0; j < parallel; j++) {
            if (jobs[j].pid == done) {
                job = &jobs[j];
                break;
            }
        }
        assert(job && "waitpid reaped unknown child");
        int nread = (int)read(job->fd, &job->result, sizeof(job->result));
        close(job->fd);
        job->pid = 0;
        active--;

        if (nread != (int)sizeof(job->result)
                || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "sweep worker run=%d failed; marking sample bad\n",
                job->run);
            protein_sweep_observe(protein, job->sample, NAN, max_cost, 1);
            assert(++failed_workers <= 1000 && "too many failed sweep workers");
            continue;
        }
        // points[]: learning-curve downsample, or 1 final score (e.g. selfplay).
        for (int pi = 0; pi < job->result.points; pi++) {
            protein_sweep_observe(protein, job->sample,
                job->result.scores[pi], job->result.costs[pi], 0);
        }
        printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n",
            job->run, job->result.score, job->result.cost, job->result.steps,
            job->random, job->gp_obs, job->pareto);
        completed++;
    }
}

// Sweep objective: bare names → env/<name>; keys with '/' used as-is.
static void sweep_metric_key(Ini* ini, char* buf, size_t n) {
    const char* metric = puf_ini_get_str(ini, "sweep", "metric");
    snprintf(buf, n, "%s%s", strchr(metric, '/') ? "" : "env/", metric);
}

// board!=NULL: merge env/* into train last_log (uptime + util/* stay frozen).
static EvalResult eval_loop(Ini* ini, PuffeRL* p, int mode, int verbose,
        int render, long eval_episodes, Dict* board, int epoch) {
    int match = mode == EVAL_MATCH;
    char metric_key[128];
    sweep_metric_key(ini, metric_key, sizeof(metric_key));
    EvalResult result = {0};
    if (!render) {
        Dict wipe = {0};
        vec_log(p->vec, &wipe, 1);
        dict_clear(&wipe);
    }
    double last_dash = 0;
    while (true) {
        if (render) {
            puf_render(p->vec->envs);
            if (WindowShouldClose()) {
                return result;
            }
        }
        rollouts(p);
        Dict el = {0};
        trainer_eval_log(p, &el);
        if (board) {
            for (int i = 0; i < el.size; i++) {
                const char* k = el.items[i].key;
                if (strncmp(k, "util/", 5) == 0) {
                    continue;
                }
                dict_set(board, k, el.items[i].value);
            }
        }
        Dict* show = board ? board : &el;
        double now = wall_clock();
        if (!render && verbose && now - last_dash >= 0.6) {
            puf_dashboard_print(ini, p, show, board ? epoch : 0);
            last_dash = now;
        }
        if (render) {
            dict_clear(&el);
            continue;
        }
        long n = dict_get(&el, "env/n");
        if (n < eval_episodes) {
            dict_clear(&el);
            continue;
        }
        if (verbose) {
            puf_dashboard_print(ini, p, show, board ? epoch : 0);
        }
        DictItem* m = dict_find(&el, metric_key);
        result.score = match ? dict_get(&el, "env/policy_0_score")
            : (m ? m->value : dict_get(&el, "env/score"));
        result.perf = dict_get(&el, "env/perf");
        if (match) {
            result.draw = dict_get(&el, "env/draw_rate");
        }
        result.games = (int)n;
        dict_clear(&el);
        return result;
    }
}

static PuffeRL* eval_make(Ini* ini, TrainContext* ctx, int mode, int render) {
    int match = mode == EVAL_MATCH;
    long eval_agents = puf_ini_get(ini, "base", "eval_agents");
    if (render) {
        // One env on screen. Some envs (drone num_drones, boids num_boids) do
        // not use env.num_agents; default.ini's 1 then under-sizes the buffer
        // and env_setup's offset==apb assert fires.
        Env probe = {0};
        puf_init(&probe, puf_ini_section(ini, "env", 0));
        int n = probe.num_agents;
        puf_close(&probe);
        assert(n > 0 && "render eval: env reported 0 agents");
        char nb[32];
        snprintf(nb, sizeof(nb), "%d", n);
        puf_ini_put(ini, "vec.total_agents", nb);
        puf_ini_put(ini, "vec.num_buffers", "1");
        puf_ini_put(ini, "vec.num_threads", "1");
        puf_ini_put(ini, "train.verb_eps", "1");
    } else if (eval_agents != -1) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%ld", eval_agents);
        puf_ini_put(ini, "vec.total_agents", buf);
        puf_ini_put(ini, "train.verb_eps", "0");
    }
    if (match) {
        int h = puf_ini_get(ini, "policy", "hidden_size");
        int L = puf_ini_get(ini, "policy", "num_layers");
        char hb[32], lb[32];
        snprintf(hb, sizeof(hb), "%d", h);
        snprintf(lb, sizeof(lb), "%d", L);
        puf_ini_put(ini, "vec.num_policies", "2");
        puf_ini_put(ini, "vec.hist_policy_percent", "1");
        puf_ini_put(ini, "vec.hist_policy_hidden_size", hb);
        puf_ini_put(ini, "vec.hist_policy_num_layers", lb);
        puf_ini_put(ini, "selfplay.enabled", "0");
    }
    puf_ini_put(ini, "base.reset_every_horizon", "0");
    if (render) {
        puf_ini_put(ini, "train.horizon", "1");
    }
    PuffeRL* p = create_pufferl(ini, ctx);
    int loaded = 0;
    if (match) {
        char a_buf[4096], b_buf[4096];
        const char* a = puf_checkpoint_path_key(ini, "load_model_path", a_buf, sizeof(a_buf));
        const char* b = puf_checkpoint_path_key(ini,
            "load_enemy_model_path", b_buf, sizeof(b_buf));
        assert(a && b && "match requires load_model_path and load_enemy_model_path");
        pufferl_load_policy(p, 0, a);
        pufferl_load_policy(p, 1, b);
        loaded = 1;
    } else {
        char buf[4096];
        const char* path = puf_checkpoint_path_key(ini, "load_model_path", buf, sizeof(buf));
        if (path) {
            pufferl_load_policy(p, 0, path);
            loaded = 1;
        }
    }
    // The HUD treats 0 as a rejected file. Eval loaded the weights itself.
#ifdef PUF_HAS_SIM_CONTROL
    if (loaded) {
        p->vec->envs[0].checkpoint_ok = 1;
    }
#endif
    return p;
}

EvalResult run_eval(Ini* ini, TrainContext* ctx, int mode, int verbose,
        int render) {
    long n = puf_ini_get(ini, "base", "eval_episodes");
    assert((render || n > 0) && "eval requires positive base.eval_episodes");
    PuffeRL* p = eval_make(ini, ctx, mode, render);
    EvalResult r = eval_loop(ini, p, mode, verbose, render, n, NULL, 0);
    if (!render) {
        long nparams = 0;
        if (p && p->policies) {
            nparams = numel(p->policies[0].master_weights.shape);
        }
        printf("CUDA_EVAL env=%s score=%.6f perf=%.6f games=%d params=%ld\n",
            puf_ini_get_str(ini, "base", "env_name"),
            r.score, r.perf, r.games, nparams);
        fflush(stdout);
    }
    close_pufferl(p);
    return r;
}

// entropy, kl, and the other loss/* stats stay off the ini.
// policy, value, and anchor are kept. importance is not a loss/ key.
static int puf_log_metric(const char* key) {
    if (strncmp(key, "loss/", 5) != 0) {
        return 1;
    }
    if (strcmp(key, "loss/policy") == 0) {
        return 1;
    }
    if (strcmp(key, "loss/value") == 0) {
        return 1;
    }
    if (strcmp(key, "loss/anchor") == 0) {
        return 1;
    }
    return 0;
}

// Config plus the binned train curve. last_log is included as the final
// sample and then removed, so a later eval can append its own snapshot.
// Called before post-train eval: that eval reopens every map, and an abort
// there used to discard a finished run because this file was written after.
static void write_run_log(Ini* ini, const char* log_path, PufLogHistory* hist,
        Dict* last_log, const char* target_key, int points) {
    assert(hist->size == 0 || dict_find(last_log, target_key));
    FILE* fp = fopen(log_path, "w");
    assert(fp && "failed to open log for writing");
    fprintf(fp, "# PufferLib log v1\n");
    puf_ini_write(fp, ini);
    fprintf(fp, "\n[metrics]\n");

    int pushed = 0;
    if (last_log->size > 0) {
        puf_log_history_add(hist, last_log);
        pushed = 1;
    }
    if (hist->size > 0) {
        double* out = (double*)calloc((size_t)points, sizeof(double));
        Dict* key_src = &hist->items[hist->size - 1];
        for (int k = 0; k < key_src->size; k++) {
            const char* key = key_src->items[k].key;
            if (!puf_log_metric(key)) {
                continue;
            }
            log_history_bin_mean(hist, key, points, out);
            fprintf(fp, "%s = ", key);
            for (int i = 0; i < points; i++) {
                fprintf(fp, "%s%.17g", i ? "," : "", out[i]);
            }
            fputc('\n', fp);
        }
        free(out);
    }
    if (pushed) {
        dict_clear(&hist->items[hist->size - 1]);
        hist->size--;
    }
    fclose(fp);
}

// CPU player records a split BEV / agent view from this checkpoint.
// Failure leaves the train result intact. The caller is still inside the
// train process, so the next checkpoint waits until the clip is written.
static void record_rollout_video(Ini* ini, const char* checkpoint_dir,
        const char* checkpoint) {
    if (!checkpoint || !checkpoint[0] || !checkpoint_dir || !checkpoint_dir[0]) {
        return;
    }
    size_t n = strlen(checkpoint);
    if (n < 5 || strcmp(checkpoint + n - 4, ".bin") != 0) {
        fprintf(stderr, "record: checkpoint path is not a .bin, skipping\n");
        return;
    }
    char ini_path[4096];
    char out_path[4096];
    snprintf(ini_path, sizeof(ini_path), "%s/rollout.ini", checkpoint_dir);
    snprintf(out_path, sizeof(out_path), "%.*s.mp4", (int)(n - 4), checkpoint);
    if (strchr(ini_path, '\'') || strchr(checkpoint, '\'') || strchr(out_path, '\'')) {
        fprintf(stderr, "record: path has a quote, skipping video\n");
        return;
    }
    FILE* fp = fopen(ini_path, "w");
    if (!fp) {
        fprintf(stderr, "record: could not write %s\n", ini_path);
        return;
    }
    puf_ini_write(fp, ini);
    fclose(fp);
    char cmd[8192];
    int wrote = snprintf(cmd, sizeof(cmd),
        "env -u DISPLAY python3 scripts/record_policy.py --ini '%s' --checkpoint '%s' --out '%s'",
        ini_path, checkpoint, out_path);
    if (wrote < 0 || wrote >= (int)sizeof(cmd)) {
        fprintf(stderr, "record: command too long, skipping video\n");
        return;
    }
    fprintf(stderr, "record: %s\n", cmd);
    fflush(stderr);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "record: video failed (%d); training result kept\n", rc);
        fflush(stderr);
    }
}

// record_rollout_every: 0 never. Otherwise every Nth checkpoint saved
// during training (1 records each save), including the final epoch.
// The untrained initial policy is not recorded. Only the artifact owner
// calls this.
static void maybe_record_rollout(Ini* ini, int every, int* saved,
        const char* checkpoint_dir, const char* checkpoint) {
    if (!checkpoint || !checkpoint[0]) {
        return;
    }
    int index = (*saved)++;
    if (every > 0 && index % every == 0) {
        record_rollout_video(ini, checkpoint_dir, checkpoint);
    }
}

TrainResult run_train(Ini* ini, TrainContext* ctx) {
    int use_selfplay = puf_ini_get(ini, "selfplay", "enabled");
    if (!use_selfplay) {
        puf_ini_put(ini, "vec.num_policies", "1");
        puf_ini_put(ini, "vec.hist_policy_percent", "0");
    } else {
        int npol = puf_ini_get(ini, "vec", "num_policies");
        assert(npol >= 2 && npol <= SELFPLAY_MAX_HIST + 1
            && "selfplay requires vec.num_policies in 2..SELFPLAY_MAX_HIST+1");
        assert(puf_ini_get(ini, "vec", "hist_policy_percent") > 0
            && "selfplay requires vec.hist_policy_percent > 0");
        // Pool opps are this run's checkpoints; hist arch must match policy.
        char hb[32], lb[32];
        snprintf(hb, sizeof(hb), "%d",
            (int)puf_ini_get(ini, "policy", "hidden_size"));
        snprintf(lb, sizeof(lb), "%d",
            (int)puf_ini_get(ini, "policy", "num_layers"));
        puf_ini_put(ini, "vec.hist_policy_hidden_size", hb);
        puf_ini_put(ini, "vec.hist_policy_num_layers", lb);
        double ladder[SELFPLAY_MAX_LADDER];
        assert((puf_ini_get(ini, "selfplay", "eval_bot_games") <= 0
            || puf_ini_get_list(ini, "selfplay", "eval_bots", ladder,
                SELFPLAY_MAX_LADDER) > 0)
            && "selfplay.eval_bot_games requires selfplay.eval_bots");
    }

    char run_id[64];
    snprintf(run_id, sizeof(run_id), "%s", puf_ini_get_str(ini, "base", "run_id"));

    char checkpoint_dir[2048];
    char log_dir[2048];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/%s/%s",
        puf_ini_get_str(ini, "base", "checkpoint_dir"),
        puf_ini_get_str(ini, "base", "env_name"), run_id);
    snprintf(log_dir, sizeof(log_dir), "%s/%s",
        puf_ini_get_str(ini, "base", "log_dir"),
        puf_ini_get_str(ini, "base", "env_name"));
    if (ctx->artifact_owner) {
        mkdir_p(checkpoint_dir);
        mkdir_p(log_dir);
    }

    PuffeRL* pufferl = create_pufferl(ini, ctx);
    int record_every = (int)puf_ini_get(ini, "base", "record_rollout_every");
    int checkpoints_saved = 0;
    Selfplay selfplay = {0};
    if (use_selfplay) {
        char initial_checkpoint[4096];
        snprintf(initial_checkpoint, sizeof(initial_checkpoint),
            "%s/%016ld.bin", checkpoint_dir, pufferl->global_step);
        puf_save_weights(pufferl, initial_checkpoint);
        selfplay.num_hist = pufferl->num_policies - 1;
        assert(selfplay.num_hist > 0 && selfplay.num_hist <= SELFPLAY_MAX_HIST
            && "selfplay requires num_policies in 2..SELFPLAY_MAX_HIST+1");
        selfplay.max_size = puf_ini_get(ini, "selfplay", "max_size");
        assert(selfplay.max_size > 0 && "selfplay.max_size must be positive");
        selfplay.pool = (char (*)[SELFPLAY_PATH_MAX])calloc(
            selfplay.max_size, sizeof(*selfplay.pool));
        selfplay.opp_timeout_steps = puf_ini_get(ini, "selfplay", "opp_timeout_steps");
        selfplay.rng = puf_ini_get(ini, "selfplay", "seed") + pufferl->hypers.rank;
        long current_step = pufferl->global_step * pufferl->hypers.world_size;

        selfplay_add_checkpoint(&selfplay, initial_checkpoint);
        for (int s = 0; s < selfplay.num_hist; s++) {
            SelfplayHist* hist = &selfplay.hist[s];
            hist->policy_idx = s + 1;
            pufferl_load_policy(pufferl, hist->policy_idx, selfplay_sample(&selfplay));
            hist->opp_started_step = current_step;
        }
    }

    long total_timesteps = puf_ini_get(ini, "train", "total_timesteps");
    long batch_size = puf_ini_get(ini, "vec", "total_agents") *
        puf_ini_get(ini, "train", "horizon");
    long local_timesteps = total_timesteps / ctx->world_size;
    long train_epochs = local_timesteps / batch_size;
    long checkpoint_interval = puf_ini_get(ini, "base", "checkpoint_interval");
    char target_key[128];
    sweep_metric_key(ini, target_key, sizeof(target_key));
    Dict last_log = {0};
    // At most one history entry per train epoch (+1 final snapshot for log dump).
    PufLogHistory log_history;
    puf_log_history_init(&log_history, (int)train_epochs + 1);
    TrainResult result = {0};
    char final_checkpoint[4096] = {0};

    for (long epoch = 0; epoch < train_epochs; epoch++) {
        if (pufferl->hypers.async) {
            // Cleanba 2-slot: warmup fills slot 0; then collect into write
            // while training the other slot (exactly one epoch old).
            int prefetch_next = epoch + 1 < train_epochs;
            if (!pufferl->async_boot) {
                double t0 = rollout_start(pufferl, 0);
                rollout_finish(pufferl, t0);
                pufferl->async_ready_slot = 0;
                pufferl->async_write_slot = 1;
                pufferl->async_boot = true;
            }

            int ready_slot = pufferl->async_ready_slot;
            int write_slot = pufferl->async_write_slot;
            double t0 = 0.0;
            if (prefetch_next) {
                t0 = rollout_start(pufferl, write_slot);
            }

            pufferl->global_step += pufferl->hypers.horizon
                * pufferl->hypers.total_agents;
            RolloutBuf train_src = rollout_time_view(&pufferl->rollouts,
                ready_slot * pufferl->hypers.horizon, pufferl->hypers.horizon);
            train_impl(pufferl, &train_src);

            if (prefetch_next) {
                rollout_finish(pufferl, t0);
                pufferl->async_ready_slot = 1 - ready_slot;
                pufferl->async_write_slot = 1 - write_slot;
            }
        } else {
            rollouts(pufferl);
            train_impl(pufferl, NULL);
        }

        char saved_checkpoint[4096] = {0};
        if (epoch == train_epochs - 1 || (checkpoint_interval > 0
                && (epoch + 1) % checkpoint_interval == 0)) {
            snprintf(saved_checkpoint, sizeof(saved_checkpoint),
                "%s/%016ld.bin", checkpoint_dir, pufferl->global_step);
            if (ctx->artifact_owner || use_selfplay) {
                puf_save_weights(pufferl, saved_checkpoint);
            }
            if (ctx->artifact_owner) {
                snprintf(final_checkpoint, sizeof(final_checkpoint),
                    "%s", saved_checkpoint);
                maybe_record_rollout(ini, record_every, &checkpoints_saved,
                    checkpoint_dir, saved_checkpoint);
            }
        }
        if (use_selfplay && saved_checkpoint[0]) {
            selfplay_add_checkpoint(&selfplay, saved_checkpoint);
        }

        // Opponent swap mid-episode: treat as truncate + reset (not boundary wait).
        if (use_selfplay && selfplay.opp_timeout_steps > 0) {
            long step = pufferl->global_step * pufferl->hypers.world_size;
            for (int s = 0; s < selfplay.num_hist; s++) {
                SelfplayHist* hist = &selfplay.hist[s];
                if (step - hist->opp_started_step >= selfplay.opp_timeout_steps) {
                    pufferl_load_policy(pufferl, hist->policy_idx,
                        selfplay_sample(&selfplay));
                    hist->opp_started_step = step;
                }
            }
        }

        if (last_log.size && wall_clock()
                < pufferl->last_log_time + 0.6 && epoch < train_epochs - 1) {
            continue;
        }

        Dict new_log = {0};
        long global_step = pufferl->global_step;
        double now = wall_clock();
        double dt = now - pufferl->last_log_time;
        double sps = dt > 0 ? (double)(global_step -
            pufferl->last_log_step) / dt * pufferl->hypers.world_size : 0;
        pufferl->last_log_time = now;
        pufferl->last_log_step = global_step;

        dict_set(&new_log, "SPS", sps);
        dict_set(&new_log, "agent_steps", (double)global_step * pufferl->hypers.world_size);
        dict_set(&new_log, "uptime", now - pufferl->start_time);
        dict_set(&new_log, "epoch", (double)pufferl->epoch);

        vec_log(pufferl->vec, &new_log, 1);

        float losses_host[NUM_LOSSES];
        cudaMemcpy(losses_host, pufferl->losses, sizeof(losses_host),
            cudaMemcpyDeviceToHost);
        float inv_n = losses_host[LOSS_N] > 0 ? 1.0f / losses_host[LOSS_N] : 0.0f;
        for (int i = 0; i < LOSS_N; i++) {
            dict_set(&new_log, LOSS_NAMES[i], losses_host[i] * inv_n);
        }
        cudaMemset(pufferl->losses, 0, NUM_LOSSES * sizeof(float));

        log_util(pufferl, &new_log);

        float train_total = 0;
        for (int i = 0; i < NUM_PROF; i++) {
            float sec = pufferl->profile.accum[i] / 1000.0f;
            char key[256];
            snprintf(key, sizeof(key), "perf/%s", PROF_NAMES[i]);
            dict_set(&new_log, key, sec);
            if (i >= PROF_TRAIN_MISC) {
                train_total += sec;
            }
        }
        dict_set(&new_log, "perf/train", train_total);
        memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));

        if (use_selfplay) {
            dict_set(&new_log, "pool/size", selfplay.pool_size);
            dict_set(&new_log, "pool/num_hist", selfplay.num_hist);
            dict_set(&new_log, "pool/num_policies", pufferl->num_policies);
        }
        // n=0 logs omit env/*; keep last complete-episode snapshot.
        int episodes = dict_get(&new_log, "env/n") > 0;
        if (ctx->artifact_owner) {
            puf_dashboard_print(ini, pufferl, &new_log, (int)pufferl->epoch);
        }
        result.cost = dict_get(&new_log, "uptime");
        result.steps = dict_get(&new_log, "agent_steps");
        if (episodes || !last_log.size) {
            dict_clear(&last_log);
            dict_copy(&last_log, &new_log);
        } else {
            dict_set(&last_log, "uptime", result.cost);
            dict_set(&last_log, "agent_steps", result.steps);
        }
        // Wait until the objective appears; do not treat negative values as missing.
        if (episodes && dict_find(&new_log, target_key)) {
            puf_log_history_add(&log_history, &last_log);
        }
        dict_clear(&new_log);
    }

    // TrainResult curve: bin-mean over log_history (same as artifact metrics).
    DictItem* target = dict_find(&last_log, target_key);
    result.score = target ? (float)target->value : 0;

    int points = use_selfplay ? 1 : puf_ini_get(ini, "sweep", "downsample");
    assert(points >= 1 && points <= TRAIN_RESULT_MAX_POINTS
        && "sweep.downsample must be in [1, TRAIN_RESULT_MAX_POINTS]");
    result.points = points;

    if (log_history.size == 0 || points == 1) {
        result.scores[0] = result.score;
        result.costs[0] = result.cost;
        result.step_points[0] = result.steps;
    } else {
        double tmp[TRAIN_RESULT_MAX_POINTS];
        log_history_bin_mean(&log_history, target_key, points, tmp);
        for (int p = 0; p < points; p++) {
            result.scores[p] = (float)tmp[p];
        }
        log_history_bin_mean(&log_history, "uptime", points, tmp);
        for (int p = 0; p < points; p++) {
            result.costs[p] = (float)tmp[p];
        }
        log_history_bin_mean(&log_history, "agent_steps", points, tmp);
        for (int p = 0; p < points; p++) {
            result.step_points[p] = (float)tmp[p];
        }
        result.scores[points - 1] = result.score;
        result.costs[points - 1] = result.cost;
        result.step_points[points - 1] = result.steps;
    }

    long bot_games = use_selfplay ? puf_ini_get(ini, "selfplay", "eval_bot_games") : 0;
    int max_opp = use_selfplay ? puf_ini_get(ini, "selfplay", "eval_pool_size") : 0;
    long pool_games = use_selfplay ? puf_ini_get(ini, "selfplay", "eval_games") : 0;
    int bot_ladder = use_selfplay && bot_games > 0 && final_checkpoint[0];
    int pool_eval = !bot_ladder && use_selfplay && max_opp > 0
        && pool_games > 0 && final_checkpoint[0];
    long eval_episodes = puf_ini_get(ini, "base", "eval_episodes");
    int eval_epoch = (int)pufferl->epoch;
    char eval_ckpt[4096] = {0};
    if (final_checkpoint[0]) {
        snprintf(eval_ckpt, sizeof(eval_ckpt), "%s", final_checkpoint);
        puf_ini_put(ini, "base.load_model_path", eval_ckpt);
    }
    // Eval reopens the maps. A missing file used to abort here, after the
    // checkpoint was saved and before this log existed.
    char log_path[4096];
    snprintf(log_path, sizeof(log_path), "%s/%s.ini", log_dir, run_id);
    if (ctx->artifact_owner) {
        write_run_log(ini, log_path, &log_history, &last_log, target_key, points);
    }
    // Close every rank before rank-0 eval. Eval on the live DP trainer leaves
    // child vec workers in BUF_RUNNING/BUF_WAITING while rank 0 waitpid()s
    // (and previously poisoned rank-0 CUDA graphs if children exited early).
    close_pufferl(pufferl);
    if (ctx->artifact_owner && !bot_ladder && !pool_eval && eval_episodes > 0
            && eval_ckpt[0]) {
        puf_ini_put(ini, "base.load_model_path", eval_ckpt);
        TrainContext eval_ctx = {
            .rank = 0,
            .world_size = 1,
            .gpu_id = ctx->gpu_id,
            .artifact_owner = 1,
            .nccl_id = NULL,
        };
        PuffeRL* ep = eval_make(ini, &eval_ctx, EVAL_SCORE, 0);
        EvalResult r = eval_loop(ini, ep, EVAL_SCORE, 1, 0, eval_episodes,
            &last_log, eval_epoch);
        result.score = result.scores[result.points - 1] = r.score;
        close_pufferl(ep);
    }

    if (ctx->artifact_owner) {
        FILE* fp = fopen(log_path, "w");
        assert(fp && "failed to open log for writing");
        fprintf(fp, "# PufferLib log v1\n");
        puf_ini_write(fp, ini);
        fclose(fp);
    }

    // Final Protein point (points=1): bot ladder > pool match > train curve.
    if (bot_ladder && ctx->artifact_owner) {
        puf_ini_put(ini, "base.load_model_path", final_checkpoint);
        puf_ini_put(ini, "env.num_agents", "1");
        puf_ini_put(ini, "env.num_bots", "1");
        puf_ini_put(ini, "selfplay.enabled", "0");
        puf_ini_put(ini, "vec.num_policies", "1");
        puf_ini_put(ini, "vec.hist_policy_percent", "0");
        // [bot_eval] keys are [env] names (dr, not env.dr). Dotted keys break
        // sweep argv, which flattens as section.key and splits on the last dot.
        Dict* bot_eval = puf_ini_section(ini, "bot_eval", 1);
        for (int i = 0; i < bot_eval->size; i++) {
            DictItem* over = &bot_eval->items[i];
            char ek[PUF_DICT_MAX_KEY + 8];
            snprintf(ek, sizeof(ek), "env.%s", over->key);
            puf_ini_put(ini, ek, over->str);
        }
        // Fixed parallelism; ignore swept train total_agents. Each env plays
        // bot_games / ladder_envs games, so bots face a warmed-up kNN rather
        // than being re-measured cold once per env.
        char nbuf[32];
        long envs = puf_ini_get(ini, "selfplay", "eval_bot_envs");
        snprintf(nbuf, sizeof(nbuf), "%ld", envs);
        puf_ini_put(ini, "vec.total_agents", nbuf);
        long threads = puf_ini_get(ini, "selfplay", "eval_bot_threads");
        if (threads > 0) {
            snprintf(nbuf, sizeof(nbuf), "%ld", threads);
            puf_ini_put(ini, "vec.num_threads", nbuf);
        }
        double ladder[SELFPLAY_MAX_LADDER];
        int rungs = puf_ini_get_list(ini, "selfplay", "eval_bots", ladder,
            SELFPLAY_MAX_LADDER);
        // One PuffeRL for the whole ladder. close_pufferl frees nothing, so a
        // trainer per rung is a leak. Swap bot_policy and restart instead.
        PuffeRL* ep = eval_make(ini, ctx, EVAL_SCORE, 0);
        float sum = 0;
        for (int i = 0; i < rungs; i++) {
            if (PUF_BACKEND != PUF_GPU) {
                for (int e = 0; e < ep->vec->size; e++) {
                    puf_set_bot_policy(&ep->vec->envs[e], (int)ladder[i]);
                }
            }
            env_restart(ep);
            EvalResult r = eval_loop(ini, ep, EVAL_SCORE, 0, 0, bot_games, NULL, 0);
            sum += r.perf;
            printf("bot_eval policy=%d games=%d perf=%.4f\n",
                (int)ladder[i], r.games, r.perf);
        }
        close_pufferl(ep);
        result.score = result.scores[0] = sum / rungs;
        result.points = 1;
        result.costs[0] = result.cost;
        result.step_points[0] = result.steps;
        dict_set(&last_log, "selfplay/bot_ladder_perf", result.score);
        printf("bot_eval mean_perf=%.4f\n", result.score);
    } else if (pool_eval && ctx->artifact_owner) {
        puf_ini_put(ini, "base.load_model_path", final_checkpoint);
        TrainContext eval_ctx = {.world_size = 1, .artifact_owner = 1};
        int n_opp = 0;
        float sum = 0;
        for (int i = 0; i < selfplay.pool_size && n_opp < max_opp; i++) {
            if (strcmp(selfplay.pool[i], final_checkpoint) == 0) {
                continue;
            }
            puf_ini_put(ini, "base.load_enemy_model_path", selfplay.pool[i]);
            PuffeRL* ep = eval_make(ini, &eval_ctx, EVAL_MATCH, 0);
            EvalResult r = eval_loop(ini, ep, EVAL_MATCH, 0, 0, pool_games, NULL, 0);
            close_pufferl(ep);
            sum += r.score;
            n_opp++;
            printf("selfplay_eval vs %s games=%d score=%.4f draw=%.4f\n",
                selfplay.pool[i], r.games, r.score, r.draw);
        }
        if (n_opp) {
            result.score = result.scores[0] = sum / n_opp;
            result.points = 1;
            result.costs[0] = result.cost;
            result.step_points[0] = result.steps;
            dict_set(&last_log, "selfplay/pool_score", result.score);
            printf("selfplay_eval mean_score=%.4f n=%d\n", result.score, n_opp);
        }
    }

    if (ctx->artifact_owner) {
        assert(log_history.size == 0 || dict_find(&last_log, target_key));
        puf_log_history_add(&log_history, &last_log);
        FILE* fp = fopen(log_path, "a");
        assert(fp && "failed to open log for writing");
        fprintf(fp, "\n[metrics]\n");

        // Last snapshot keys (train + selfplay eval overlays).
        if (log_history.size > 0) {
            int metric_points = points;
            double* out = (double*)calloc(metric_points, sizeof(double));
            Dict* key_src = &log_history.items[log_history.size - 1];
            for (int k = 0; k < key_src->size; k++) {
                const char* key = key_src->items[k].key;
                if (!puf_log_metric(key)) {
                    continue;
                }
                log_history_bin_mean(&log_history, key, metric_points, out);
                fprintf(fp, "%s = ", key);
                for (int i = 0; i < metric_points; i++) {
                    fprintf(fp, "%s%.17g", i ? "," : "", out[i]);
                }
                fputc('\n', fp);
            }
            free(out);
        }
        fclose(fp);
    }
    for (int i = 0; i < log_history.size; i++) {
        dict_clear(&log_history.items[i]);
    }
    free(log_history.items);
    free(selfplay.pool);
    return result;
}

// Human-replay score from config/benchmark.ini. Empty sweep_metric skips
// the launch. after_train runs it after ./puffer train. A sweep worker
// (result_fd > 0) does not: its meta-objective is [sweep] metric.
static int bench_cfg_metric(char* metric, int n, int* after_train) {
    Ini bench = {0};
    const char* name;
    if (access("config/benchmark.ini", R_OK) != 0) {
        return 0;
    }
    puf_ini_load_file(&bench, "config/benchmark.ini");
    name = puf_ini_get_str(&bench, "benchmark", "sweep_metric");
    snprintf(metric, (size_t)n, "%s", name ? name : "");
    *after_train = (int)puf_ini_get(&bench, "benchmark", "after_train");
    puf_ini_free(&bench);
    return metric[0] != 0;
}

static float bench_report_value(const char* path, const char* metric) {
    FILE* fp = fopen(path, "r");
    char line[256];
    float value = 0.0f;
    int found = 0;
    if (!fp) {
        fprintf(stderr, "replay objective: missing %s\n", path);
        exit(1);
    }
    while (fgets(line, (int)sizeof(line), fp)) {
        char key[64];
        float v = 0.0f;
        if (sscanf(line, "sweep %63s %f", key, &v) == 2
                && strcmp(key, metric) == 0) {
            value = v;
            found = 1;
        }
    }
    fclose(fp);
    if (!found) {
        fprintf(stderr, "replay objective: %s not in %s\n", metric, path);
        exit(1);
    }
    return value;
}

static void apply_replay_objective(Ini* ini, TrainResult* result) {
    char metric[64];
    char out_dir[512];
    char report[576];
    char bench_arg[1024];
    char out_arg[576];
    char hidden_arg[64];
    char layers_arg[64];
    char* argv[6];
    const char* ckpt;
    const char* run_id;
    int after_train = 0;
    int result_fd;
    int status = 0;
    pid_t pid = 0;
    float score;
    if (!bench_cfg_metric(metric, (int)sizeof(metric), &after_train)) {
        return;
    }
    result_fd = (int)puf_ini_get(ini, "base", "result_fd");
    // Sweep workers report env/<sweep.metric> and return. The benchmark
    // is a separate ./drive --benchmark launch, or after_train on a
    // normal train.
    if (result_fd > 0 || !after_train) {
        return;
    }
    ckpt = puf_ini_get_str(ini, "base", "load_model_path");
    if (!ckpt || !ckpt[0] || strcmp(ckpt, "None") == 0
            || strcmp(ckpt, "'None'") == 0) {
        fprintf(stderr, "replay objective: no checkpoint\n");
        exit(1);
    }
    run_id = puf_ini_get_str(ini, "base", "run_id");
    snprintf(out_dir, sizeof(out_dir), "analysis/benchmark/%s", run_id);
    snprintf(bench_arg, sizeof(bench_arg), "--benchmark=%s", ckpt);
    snprintf(out_arg, sizeof(out_arg), "--benchmark-out=%s", out_dir);
    snprintf(hidden_arg, sizeof(hidden_arg), "--policy.hidden_size=%d",
        (int)puf_ini_get(ini, "policy", "hidden_size"));
    snprintf(layers_arg, sizeof(layers_arg), "--policy.num_layers=%d",
        (int)puf_ini_get(ini, "policy", "num_layers"));
    argv[0] = (char*)"./drive";
    argv[1] = bench_arg;
    argv[2] = out_arg;
    argv[3] = hidden_arg;
    argv[4] = layers_arg;
    argv[5] = NULL;
    fprintf(stderr, "replay objective %s %s\n", metric, ckpt);
    if (posix_spawn(&pid, "./drive", NULL, NULL, argv, environ) != 0) {
        perror("replay objective");
        exit(1);
    }
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)
            || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "replay objective benchmark failed\n");
        exit(1);
    }
    snprintf(report, sizeof(report), "%s/report.txt", out_dir);
    // "score" still logs replay and horizon, and protein keeps env/score.
    if (strcmp(metric, "score") == 0) {
        if (access(report, R_OK) != 0) {
            fprintf(stderr, "replay objective: missing %s\n", report);
            exit(1);
        }
        fprintf(stderr, "replay objective kept training score\n");
        return;
    }
    score = bench_report_value(report, metric);
    // One protein point. The training curve stays in the run ini.
    result->score = score;
    result->scores[0] = score;
    result->points = 1;
    result->costs[0] = result->cost;
    result->step_points[0] = result->steps;
    fprintf(stderr, "replay objective %s=%.6f\n", metric, score);
}

// Fork DP workers before any CUDA/NCCL call (those start runtime threads;
// fork after that SIGSEGVs children and rank 0 hangs in ncclCommInitRank).
// Rank 0 generates the NCCL id after fork and pipes it. Sweep trials occupy
// contiguous GPU blocks; rank 0 owns the last GPU and writes TrainResult.
TrainResult launch_train(Ini* ini) {
    int mb = puf_ini_get(ini, "train", "minibatch_size");
    int horizon = puf_ini_get(ini, "train", "horizon");
    int agents = puf_ini_get(ini, "vec", "total_agents");
    int world_size = puf_ini_get(ini, "train", "gpus");
    int gpu_offset = puf_ini_get(ini, "base", "gpu_offset");
    assert(world_size >= 1 && "train.gpus must be >= 1");
    assert(horizon > 0 && mb % horizon == 0
        && "train.minibatch_size must be divisible by train.horizon");
    assert((long)mb <= (long)horizon * agents
        && "train.minibatch_size must be <= train.horizon * vec.total_agents");
    assert(agents % (mb / horizon) == 0
        && "vec.total_agents must be divisible by minibatch rows");
    assert(horizon % ADV_VEC_WIDTH == 0
        && "train.horizon must be a multiple of ADV_VEC_WIDTH (4 float / 8 bf16)");

    const char* run_id = puf_ini_get_str(ini, "base", "run_id");
    if (!run_id[0] || strcmp(run_id, "None") == 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%ld", (long)(1000.0 * wall_clock()));
        puf_ini_put(ini, "base.run_id", buf);
    }

    int nccl_pipe[2] = {-1, -1};
    if (world_size > 1) {
        assert(pipe(nccl_pipe) == 0 && "pipe failed");
    }

    int n_workers = world_size - 1;
    pid_t* pids = (pid_t*)calloc(n_workers, sizeof(pid_t));
    for (int rank = world_size - 1; rank >= 1; rank--) {
        pid_t pid = fork();
        assert(pid >= 0 && "fork failed");
        if (pid == 0) {
            close(nccl_pipe[1]);
            ncclUniqueId nccl_id;
            assert(read(nccl_pipe[0], &nccl_id, sizeof(nccl_id)) == (ssize_t)sizeof(nccl_id)
                && "failed to read ncclUniqueId");
            assert(freopen("/dev/null", "w", stdout) == stdout);
            TrainContext child = {
                .rank = rank,
                .world_size = world_size,
                .gpu_id = gpu_offset + rank - 1,
                .artifact_owner = 0,
                .nccl_id = &nccl_id,
            };
            run_train(ini, &child);
            // Rank 0 may still be in post-train eval. Stay alive until it
            // closes the pipe; exiting earlier aborts NCCL and poisons rank 0
            // CUDA graphs (cudaGraphLaunch fails in pufferl_forward).
            char b;
            ssize_t n = read(nccl_pipe[0], &b, 1);
            assert(n >= 0);
            close(nccl_pipe[0]);
            puf_ini_free(ini);
            exit(0);
        }
        pids[rank - 1] = pid;
    }

    ncclUniqueId nccl_id;
    ncclUniqueId* nccl_ptr = NULL;
    if (world_size > 1) {
        close(nccl_pipe[0]);
        ncclGetUniqueId(&nccl_id);
        for (int i = 0; i < n_workers; i++) {
            assert(write(nccl_pipe[1], &nccl_id, sizeof(nccl_id)) == (ssize_t)sizeof(nccl_id)
                && "failed to write ncclUniqueId");
        }
        nccl_ptr = &nccl_id;
    }

    TrainContext host = {
        .rank = 0,
        .world_size = world_size,
        .gpu_id = gpu_offset + world_size - 1,
        .artifact_owner = 1,
        .nccl_id = nccl_ptr,
    };
    TrainResult result = run_train(ini, &host);
    apply_replay_objective(ini, &result);
    if (world_size > 1) {
        close(nccl_pipe[1]);
    }
    for (int i = 0; i < n_workers; i++) {
        int status = 0;
        waitpid(pids[i], &status, 0);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0
            && "train rank worker failed");
    }
    free(pids);

    // Sweep parent reads this over the pipe; CLI train ignores (result_fd=0).
    int result_fd = puf_ini_get(ini, "base", "result_fd");
    if (result_fd > 0) {
        assert(write(result_fd, &result, sizeof(result)) == sizeof(result));
        close(result_fd);
    }
    return result;
}

#ifdef PUFFERLIB_BUILD_MAIN
int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s train|eval|match|sweep [latest|MODEL.bin] [--headless] [--section.key=value ...]\n",
            argv[0]);
        exit(1);
    }
    const char* mode = argv[1];
    // Train forks DP workers; CUDA before that fork SIGSEGVs the children.
    if (strcmp(mode, "train") != 0) {
        int total_gpus = 0;
        assert(cudaGetDeviceCount(&total_gpus) == cudaSuccess && total_gpus >= 1
            && "no CUDA devices available");
    }
    int argi = 2;
    const char* model = NULL;
    if (strcmp(mode, "eval") == 0 && argi < argc && argv[argi][0] &&
            argv[argi][0] != '-' && strchr(argv[argi], '=') == NULL) {
        model = argv[argi++];
    }
    int headless = 0;
    char* ini_argv[argc > 0 ? (size_t)argc : 1];
    int ini_argc = 0;
    for (int i = argi; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) {
            headless = 1;
            continue;
        }
        ini_argv[ini_argc++] = argv[i];
    }
    Ini ini = {0};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, ini_argc, ini_argv);
    if (model) {
        puf_ini_put(&ini, "base.load_model_path", model);
    }
    TrainContext ctx = {.world_size = 1, .artifact_owner = 1};
    int render = !headless;

    if (strcmp(mode, "train") == 0) {
        launch_train(&ini);
    } else if (strcmp(mode, "sweep") == 0) {
        run_sweep(&ini, argv[0]);
    } else if (strcmp(mode, "eval") == 0) {
        run_eval(&ini, &ctx, EVAL_SCORE, 1, render);
    } else if (strcmp(mode, "match") == 0) {
        run_eval(&ini, &ctx, EVAL_MATCH, 1, render);
    } else {
        assert(0 && "unknown mode (train|eval|match|sweep)");
    }

    puf_ini_free(&ini);
    return 0;
}
#endif