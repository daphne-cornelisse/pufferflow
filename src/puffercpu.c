#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

const char* __asan_default_options(void) { return "detect_leaks=0"; }

// File format: flat fp32 tensors.
typedef struct Weights {
    float* data;
    int size;
    int idx;
} Weights;

Weights* load_weights(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    size_t num_weights = file_size / sizeof(float);
    // +7 pad so 8-float-aligned bf16 reads never go past the buffer.
    Weights* weights = calloc(1, sizeof(Weights) + (num_weights + 7)*sizeof(float));
    weights->data = (float*)(weights + 1);
    size_t read_size = fread(weights->data, sizeof(float), num_weights, file);
    fclose(file);
    if (read_size != num_weights) {
        perror("Error reading file");
    }
    weights->size = num_weights + 7;
    weights->idx = 0;
    return weights;
}

float* get_weights(Weights* weights, int num_weights) {
    float* data = &weights->data[weights->idx];
    weights->idx += num_weights;
    assert(weights->idx <= weights->size);
    return data;
}

// Advances index to next 8-float (16-byte) boundary after reading, matching
// the native backend's Allocator which aligns to 16 bytes with bf16 params.
float* get_weights_aligned(Weights* weights, int num_weights) {
    float* data = &weights->data[weights->idx];
    weights->idx += num_weights;
    weights->idx = (weights->idx + 7) & ~7;
    assert(weights->idx <= weights->size);
    return data;
}

// PufferNet implementation of PyTorch functions
// These are tested against the PyTorch implementation
void _relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

float _sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void _linear(float* input, float* weights, float* bias, float* output,
        int batch_size, int input_dim, int output_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++)
                sum += input[b*input_dim + i] * weights[o*input_dim + i];
            output[b*output_dim + o] = sum + bias[o];
        }
    }
}

void _conv2d(float* input, float* weights,
        float* output, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride) {
    int h_out = (in_height - kernel_size)/stride + 1;
    int w_out = (in_width - kernel_size)/stride + 1;
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int h = 0; h < h_out; h++) {
                for (int w = 0; w < w_out; w++) {
                    int out_adr = (
                        b*out_channels*h_out*w_out
                        + oc*h_out*w_out+ 
                        + h*w_out
                        + w
                    );
                    output[out_adr] = 0.0f;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int in_adr = (
                                    b*in_channels*in_height*in_width
                                    + ic*in_height*in_width
                                    + (h*stride + kh)*in_width
                                    + (w*stride + kw)
                                );
                                int weight_adr = (
                                    oc*in_channels*kernel_size*kernel_size
                                    + ic*kernel_size*kernel_size
                                    + kh*kernel_size
                                    + kw
                                );
                                output[out_adr] += input[in_adr]*weights[weight_adr];
                            }
                        }
                    }
               }
            }
        }
    }
}

void _embedding(int* input, float* weights, float* output, int batch_size, int num_embeddings, int embedding_dim) {
    for (int b = 0; b < batch_size; b++) {
        memcpy(output + b*embedding_dim, weights + input[b]*embedding_dim, embedding_dim*sizeof(float));
    }
}

void _one_hot(int* input, int* output, int batch_size, int input_size, int num_classes) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < input_size; i++) {
            int in_adr = b*input_size + i;
            int out_adr = (
                b*input_size*num_classes
                + i*num_classes
                + input[in_adr]
            );
            output[out_adr] = 1;
        }
    }
}

double _randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

void _gaussian_sample(float* input, float* log_std, float* output, int batch_size, int num_actions) {
    for (int b = 0; b < batch_size; b++) {
        // +1 skips the value head fused into the decoder output
        int in_adr = b * (num_actions + 1);
        for (int a = 0; a < num_actions; a++) {
            int out_adr = b * num_actions + a;
            float mean = input[in_adr + a];
            float std = expf(log_std[a]);
            output[out_adr] = _randn(mean, std);
        }
    }
}

// hard=1: argmax; hard=0: softmax sample. +1 on in_adr skips fused value head.
// mask is optional (batch * atn_sum); NULL means every logit is legal.
void _multidiscrete(float* input, float* output, int batch_size, int logit_sizes[],
        int num_actions, int hard, const unsigned char* mask) {
    int atn_sum = 0;
    for (int a = 0; a < num_actions; a++) {
        atn_sum += logit_sizes[a];
    }
    for (int b = 0; b < batch_size; b++) {
        int in_adr = b * (atn_sum + 1);
        int mask_adr = b * atn_sum;
        for (int a = 0; a < num_actions; a++) {
            int out_adr = b * num_actions + a;
            int n = logit_sizes[a];
            const unsigned char* head_mask = mask ? mask + mask_adr : NULL;
            output[out_adr] = 0.0f;
            int first = -1;
            for (int i = 0; i < n; i++) {
                if (!head_mask || head_mask[i]) {
                    first = i;
                    break;
                }
            }
            if (first < 0) {
                in_adr += n;
                mask_adr += n;
                continue;
            }
            if (hard) {
                float max_logit = input[in_adr + first];
                output[out_adr] = first;
                for (int i = first + 1; i < n; i++) {
                    if (head_mask && !head_mask[i]) {
                        continue;
                    }
                    float out = input[in_adr + i];
                    if (out > max_logit) {
                        max_logit = out;
                        output[out_adr] = i;
                    }
                }
            } else {
                float max_logit = input[in_adr + first];
                for (int i = first + 1; i < n; i++) {
                    if ((!head_mask || head_mask[i]) &&
                            input[in_adr + i] > max_logit) {
                        max_logit = input[in_adr + i];
                    }
                }
                float logit_exp_sum = 0.0f;
                for (int i = 0; i < n; i++) {
                    if (!head_mask || head_mask[i]) {
                        logit_exp_sum += expf(input[in_adr + i] - max_logit);
                    }
                }
                float prob = rand() / (float)RAND_MAX;
                float logit_prob = 0.0f;
                output[out_adr] = first;
                for (int i = 0; i < n; i++) {
                    if (head_mask && !head_mask[i]) {
                        continue;
                    }
                    logit_prob += expf(input[in_adr + i] - max_logit) / logit_exp_sum;
                    if (prob < logit_prob) {
                        output[out_adr] = i;
                        break;
                    }
                }
            }
            in_adr += n;
            mask_adr += n;
        }
    }
}

void _max_dim1(float* input, float* output, int batch_size, int seq_len, int feature_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < feature_dim; f++) {
            float max_val = input[b*seq_len*feature_dim + f];
            for (int s = 1; s < seq_len; s++) {
                float val = input[b*seq_len*feature_dim + s*feature_dim + f];
                if (val > max_val) {
                    max_val = val;
                }
            }
            output[b*feature_dim + f] = max_val;
        }
    }
}

// User API. Provided to help organize layers
typedef struct Linear {
    float* output;
    float* weights;
    int batch_size;
    int input_dim;
    int output_dim;
} Linear;

Linear* make_linear(Weights* weights, int batch_size, int input_dim, int output_dim) {
    size_t buffer_size = batch_size*output_dim*sizeof(float);
    Linear* layer = calloc(1, sizeof(Linear) + buffer_size);
    *layer = (Linear){
        .output = (float*)(layer + 1),
        .weights = get_weights_aligned(weights, output_dim*input_dim),
        .batch_size = batch_size,
        .input_dim = input_dim,
        .output_dim = output_dim,
    };
    return layer;
}

void linear(Linear* layer, float* input) {
    int B = layer->batch_size;
    int Id = layer->input_dim;
    int O = layer->output_dim;
    // One car is a small matmul. A fleet of 8 or more covers the fork.
    int parallel = B >= 8;
#ifdef _OPENMP
#pragma omp parallel for if (parallel) schedule(static)
#endif
    for (int b = 0; b < B; b++) {
        const float* in = input + b * Id;
        float* out = layer->output + b * O;
        for (int o = 0; o < O; o++) {
            const float* w = layer->weights + (size_t)o * Id;
            float sum = 0.0f;
            for (int i = 0; i < Id; i++) {
                sum += in[i] * w[i];
            }
            out[o] = sum;
        }
    }
}

typedef struct ReLU {
    float* output;
    int batch_size;
    int input_dim;
} ReLU;

ReLU* make_relu(int batch_size, int input_dim) {
    size_t buffer_size = batch_size*input_dim*sizeof(float);
    ReLU* layer = calloc(1, sizeof(ReLU) + buffer_size);
    *layer = (ReLU){
        .output = (float*)(layer + 1),
        .batch_size = batch_size,
        .input_dim = input_dim,
    };
    return layer;
}

void relu(ReLU* layer, float* input) {
    _relu(input, layer->output, layer->batch_size*layer->input_dim);
}

typedef struct MaxDim1 {
    float* output;
    int batch_size;
    int seq_len;
    int feature_dim;
} MaxDim1;

MaxDim1* make_max_dim1(int batch_size, int seq_len, int feature_dim) {
    size_t buffer_size = batch_size*feature_dim*sizeof(float);
    MaxDim1* layer = calloc(1, sizeof(MaxDim1) + buffer_size);
    *layer = (MaxDim1){
        .output = (float*)(layer + 1),
        .batch_size = batch_size,
        .seq_len = seq_len,
        .feature_dim = feature_dim,
    };
    return layer;
}

void max_dim1(MaxDim1* layer, float* input) {
    _max_dim1(input, layer->output, layer->batch_size, layer->seq_len, layer->feature_dim);
}

typedef struct Conv2D {
    float* output;
    float* weights;
    int batch_size;
    int in_width;
    int in_height;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
} Conv2D;

Conv2D* make_conv2d(Weights* weights, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride) {
    size_t buffer_size = batch_size*out_channels*in_height*in_width*sizeof(float);
    int num_weights = out_channels*in_channels*kernel_size*kernel_size;
    Conv2D* layer = calloc(1, sizeof(Conv2D) + buffer_size);
    *layer = (Conv2D){
        .output = (float*)(layer + 1),
        .weights = get_weights(weights, num_weights),
        .batch_size = batch_size,
        .in_width = in_width,
        .in_height = in_height,
        .in_channels = in_channels,
        .out_channels = out_channels,
        .kernel_size = kernel_size,
        .stride = stride,
    };
    return layer;
}

void conv2d(Conv2D* layer, float* input) {
    _conv2d(input, layer->weights, layer->output,
        layer->batch_size, layer->in_width, layer->in_height,
        layer->in_channels, layer->out_channels, layer->kernel_size, layer->stride);
}

typedef struct Embedding {
    float* output;
    float* weights;
    int batch_size;
    int num_embeddings;
    int embedding_dim;
} Embedding;

Embedding* make_embedding(Weights* weights, int batch_size, int num_embeddings, int embedding_dim) {
    size_t output_size = batch_size*embedding_dim*sizeof(float);
    Embedding* layer = calloc(1, sizeof(Embedding) + batch_size + output_size);
    *layer = (Embedding){
        .output = (float*)(layer + 1),
        .weights = get_weights(weights, num_embeddings*embedding_dim),
        .batch_size = batch_size,
        .num_embeddings = num_embeddings,
        .embedding_dim = embedding_dim,
    };
    return layer;
}

void embedding(Embedding* layer, int* input) {
    _embedding(input, layer->weights, layer->output, layer->batch_size, layer->num_embeddings, layer->embedding_dim);
}

typedef struct OneHot {
    int* output;
    int batch_size;
    int input_size;
    int num_classes;
} OneHot;

OneHot* make_one_hot(int batch_size, int input_size, int num_classes) {
    size_t buffer_size = batch_size*input_size*num_classes*sizeof(int);
    OneHot* layer = calloc(1, sizeof(OneHot) + buffer_size);
    *layer = (OneHot){
        .output = (int*)(layer + 1),
        .batch_size = batch_size,
        .input_size = input_size,
        .num_classes = num_classes,
    };
    return layer;
}

void one_hot(OneHot* layer, int* input) {
    _one_hot(input, layer->output, layer->batch_size, layer->input_size, layer->num_classes);
}

typedef struct Multidiscrete {
    int batch_size;
    int logit_sizes[32];
    int num_actions;
} Multidiscrete;

Multidiscrete* make_multidiscrete(int batch_size, int logit_sizes[], int num_actions) {
    Multidiscrete* layer = calloc(1, sizeof(Multidiscrete));
    layer->batch_size = batch_size;
    layer->num_actions = num_actions;
    memcpy(layer->logit_sizes, logit_sizes, num_actions*sizeof(int));
    return layer;
}

void multidiscrete(Multidiscrete* layer, float* input, float* output, int hard,
        const unsigned char* mask) {
    _multidiscrete(input, output, layer->batch_size, layer->logit_sizes,
        layer->num_actions, hard, mask);
}

// MinGRU: inference-only single-step recurrent layer.
// Matches the fused gate + highway connection in models.cu mingru_gate kernel.
// Each layer has a bias-free projection (hidden -> 3*hidden).
// State layout: (num_layers, batch_size, hidden_size).
typedef struct MinGRU {
    float* state;    // (num_layers, batch_size, hidden_size) - persists across steps
    float* output;   // (batch_size, hidden_size)
    Linear** proj;   // [num_layers], each projects hidden -> 3*hidden
    int batch_size;
    int hidden_size;
    int num_layers;
} MinGRU;

MinGRU* make_mingru(Weights* weights, int batch_size, int hidden_size, int num_layers) {
    MinGRU* layer = calloc(1, sizeof(MinGRU));
    layer->state = calloc(num_layers * batch_size * hidden_size, sizeof(float));
    layer->output = calloc(batch_size * hidden_size, sizeof(float));
    layer->proj = calloc(num_layers, sizeof(Linear*));
    layer->batch_size  = batch_size;
    layer->hidden_size = hidden_size;
    layer->num_layers  = num_layers;
    for (int l = 0; l < num_layers; l++) {
        layer->proj[l] = make_linear(weights, batch_size, hidden_size, 3 * hidden_size);
    }
    return layer;
}

void mingru(MinGRU* layer, float* input) {
    int B = layer->batch_size;
    int H = layer->hidden_size;
    float* x = input;
    for (int l = 0; l < layer->num_layers; l++) {
        float* state_l = layer->state + l * B * H;
        linear(layer->proj[l], x);
        float* combined = layer->proj[l]->output;
#ifdef _OPENMP
#pragma omp parallel for if (B >= 8) schedule(static)
#endif
        for (int b = 0; b < B; b++) {
            float* cb = combined + b * 3 * H;
            float* sb = state_l + b * H;
            float* xb = x + b * H;
            float* ob = layer->output + b * H;
            for (int h = 0; h < H; h++) {
                float hidden     = cb[h];
                float gate       = cb[H + h];
                float hw         = cb[2*H + h];
                float s          = sb[h];
                float gate_s     = _sigmoid(gate);
                float h_tilde    = (hidden >= 0.0f) ? hidden + 0.5f : _sigmoid(hidden);
                float mingru_out = s + gate_s * (h_tilde - s);
                float hw_s       = _sigmoid(hw);
                ob[h] = hw_s * mingru_out + (1.0f - hw_s) * xb[h];
                sb[h] = mingru_out;
            }
        }
        x = layer->output;
    }
}

void free_mingru(MinGRU* layer) {
    for (int l = 0; l < layer->num_layers; l++) free(layer->proj[l]);
    free(layer->state);
    free(layer->output);
    free(layer->proj);
    free(layer);
}

// Trainer zero_term_state: clear carry before encoding a post-terminal obs.
void mingru_zero_term(MinGRU* layer, const float* terminals) {
    int B = layer->batch_size;
    int H = layer->hidden_size;
    for (int b = 0; b < B; b++) {
        if (terminals[b] <= 0.5f) {
            continue;
        }
        for (int l = 0; l < layer->num_layers; l++) {
            memset(layer->state + (l * B + b) * H, 0, H * sizeof(float));
        }
    }
}

// PufferNet: default policy matching the native backend Arch in algo.cu.
// Architecture: Linear encoder -> N x MinGRU -> Linear decoder (fused value).
// Weight file order (matches weights_create reg_params call order):
//   encoder weight (hidden_dim x input_dim)
//   decoder weight ((atn_sum+1) x hidden_dim, last output is value)
//   decoder logstd (1 x num_actions) IF continuous
//   mingru weights[0..num_layers-1] (3*hidden_dim x hidden_dim each)
// Drive replaces the linear encoder. Its parameter order is the tower
// list in ocean/drive/drive_encoder.cu.
#ifdef PUFFER_DRIVE
typedef struct DriveEnc DriveEnc;
#endif
typedef struct PufferNet {
    int num_agents;
    float* obs;
    Linear* encoder;
    MinGRU* mingru;
    Linear* decoder;   // output_dim = atn_sum+1; last element is value
    float* log_std;
    int is_continuous;
    int num_actions;
    Multidiscrete* multidiscrete;
#ifdef PUFFER_DRIVE
    DriveEnc* drive;
#endif
} PufferNet;

#ifdef PUFFER_DRIVE
#include "constants.h"

#define DRIVE_ROAD_CONT (ROAD_FEATURES - 1)
#define DRIVE_ROAD_CLASSES (DRIVEWAY - ROAD_LANE + 1)
#define DRIVE_ROAD_IN (DRIVE_ROAD_CONT + DRIVE_ROAD_CLASSES)

#define DRIVE_ATTN_Q 4
#define DRIVE_ATTN_HEADS 4

struct DriveEnc {
    int hidden;
    int batch;
    float* ego_w1; float* ego_b1; float* ego_g; float* ego_be;
    float* ego_w2; float* ego_b2;
    float* part_w1; float* part_b1; float* part_g; float* part_be;
    float* part_w2; float* part_b2;
    float* road_w1; float* road_b1; float* road_g; float* road_be;
    float* road_w2; float* road_b2;
    float* q_lat; float* wq; float* bq; float* wk; float* bk;
    float* wv; float* bv; float* wo; float* bo;
    float* attn_g; float* attn_be; float* fc_w; float* fc_b;
    float* qproj;
    float* scratch;
};

static int drive_align8(int n) {
    return (n + 7) & ~7;
}

static int drive_tower_nfloats(int n, int H, int feat) {
    n = drive_align8(n + H * feat);
    n = drive_align8(n + H);
    n = drive_align8(n + H);
    n = drive_align8(n + H);
    n = drive_align8(n + H * H);
    n = drive_align8(n + H);
    return n;
}

static int drive_encoder_nfloats(int H) {
    int n = 0;
    int Q = DRIVE_ATTN_Q;
    n = drive_tower_nfloats(n, H, EGO_FEATURES);
    n = drive_tower_nfloats(n, H, PARTNER_FEATURES);
    n = drive_tower_nfloats(n, H, DRIVE_ROAD_IN);
    n = drive_align8(n + Q * H);
    n = drive_align8(n + H * H);
    n = drive_align8(n + H);
    n = drive_align8(n + H * H);
    n = drive_align8(n + H);
    n = drive_align8(n + H * H);
    n = drive_align8(n + H);
    n = drive_align8(n + H * H);
    n = drive_align8(n + H);
    n = drive_align8(n + H);
    n = drive_align8(n + H);
    n = drive_align8(n + H * Q * H);
    n = drive_align8(n + H);
    return n;
}

static float* drive_take(Weights* w, int n) {
    return get_weights_aligned(w, n);
}

static void drive_load_tower(Weights* w, int H, int feat,
        float** w1, float** b1, float** g, float** be,
        float** w2, float** b2) {
    *w1 = drive_take(w, H * feat);
    *b1 = drive_take(w, H);
    *g = drive_take(w, H);
    *be = drive_take(w, H);
    *w2 = drive_take(w, H * H);
    *b2 = drive_take(w, H);
}

static DriveEnc* drive_enc_load(Weights* w, int batch, int H) {
    DriveEnc* d = calloc(1, sizeof(DriveEnc));
    d->hidden = H;
    d->batch = batch;
    drive_load_tower(w, H, EGO_FEATURES,
        &d->ego_w1, &d->ego_b1, &d->ego_g, &d->ego_be,
        &d->ego_w2, &d->ego_b2);
    drive_load_tower(w, H, PARTNER_FEATURES,
        &d->part_w1, &d->part_b1, &d->part_g, &d->part_be,
        &d->part_w2, &d->part_b2);
    drive_load_tower(w, H, DRIVE_ROAD_IN,
        &d->road_w1, &d->road_b1, &d->road_g, &d->road_be,
        &d->road_w2, &d->road_b2);
    d->q_lat = drive_take(w, DRIVE_ATTN_Q * H);
    d->wq = drive_take(w, H * H);
    d->bq = drive_take(w, H);
    d->wk = drive_take(w, H * H);
    d->bk = drive_take(w, H);
    d->wv = drive_take(w, H * H);
    d->bv = drive_take(w, H);
    d->wo = drive_take(w, H * H);
    d->bo = drive_take(w, H);
    d->attn_g = drive_take(w, H);
    d->attn_be = drive_take(w, H);
    d->fc_w = drive_take(w, H * DRIVE_ATTN_Q * H);
    d->fc_b = drive_take(w, H);
    d->qproj = calloc((size_t)DRIVE_ATTN_Q * H, sizeof(float));
    int S = 1 + MAX_ROAD_SEGMENT_OBSERVATIONS + MAX_AGENTS;
    int stride = (3 + 2 * S + 2 * DRIVE_ATTN_Q) * H + S;
    d->scratch = calloc((size_t)batch * (size_t)stride, sizeof(float));
    return d;
}

static void drive_token(float* h2, const float* x, int F,
        const float* w1, const float* b1, const float* g, const float* be,
        const float* w2, const float* b2, float* h1, float* ln, int H) {
    for (int o = 0; o < H; o++) {
        const float* wr = w1 + (size_t)o * F;
        float sum = b1[o];
        for (int i = 0; i < F; i++) {
            sum += x[i] * wr[i];
        }
        h1[o] = sum;
    }
    float mean = 0.0f;
    for (int i = 0; i < H; i++) {
        mean += h1[i];
    }
    mean /= (float)H;
    float var = 0.0f;
    for (int i = 0; i < H; i++) {
        float d = h1[i] - mean;
        var += d * d;
    }
    var /= (float)H;
    float inv = 1.0f / sqrtf(var + 1.0e-5f);
    for (int i = 0; i < H; i++) {
        ln[i] = (h1[i] - mean) * inv * g[i] + be[i];
    }
    for (int o = 0; o < H; o++) {
        const float* wr = w2 + (size_t)o * H;
        float sum = b2[o];
        for (int i = 0; i < H; i++) {
            sum += ln[i] * wr[i];
        }
        h2[o] = sum;
    }
}

static void drive_project(float* y, const float* x, const float* w,
        const float* b, int K, int N) {
    for (int n = 0; n < N; n++) {
        const float* wr = w + (size_t)n * K;
        float sum = b[n];
        for (int k = 0; k < K; k++) {
            sum += x[k] * wr[k];
        }
        y[n] = sum;
    }
}

static void drive_ln_vec(float* y, const float* x, const float* g,
        const float* be, int H) {
    float mean = 0.0f;
    int i;
    for (i = 0; i < H; i++) {
        mean += x[i];
    }
    mean /= (float)H;
    float var = 0.0f;
    for (i = 0; i < H; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)H;
    float inv = 1.0f / sqrtf(var + 1.0e-5f);
    for (i = 0; i < H; i++) {
        y[i] = (x[i] - mean) * inv * g[i] + be[i];
    }
}

// One token into K[s], V[s]. Padding is a real key, same as the CUDA path.
static void drive_kv_token(DriveEnc* d, float* K, float* V, int s,
        const float* x, int F, const float* w1, const float* b1,
        const float* g, const float* be, const float* w2, const float* b2,
        float* h1, float* ln, float* h2) {
    int H = d->hidden;
    drive_token(h2, x, F, w1, b1, g, be, w2, b2, h1, ln, H);
    drive_project(K + (size_t)s * H, h2, d->wk, d->bk, H, H);
    drive_project(V + (size_t)s * H, h2, d->wv, d->bv, H, H);
}

static void drive_kv_slots(DriveEnc* d, float* K, float* V, int* s,
        const float* row, int off, int slots, int raw, int one_hot,
        const float* w1, const float* b1, const float* g, const float* be,
        const float* w2, const float* b2, float* h1, float* ln, float* h2) {
    int n;
    for (n = 0; n < slots; n++) {
        const float* rawf = row + off + n * raw;
        float feat[16];
        const float* x = rawf;
        int F = raw;
        if (one_hot) {
            int cls;
            int k;
            for (k = 0; k < DRIVE_ROAD_CONT; k++) {
                feat[k] = rawf[k];
            }
            cls = (int)rawf[DRIVE_ROAD_CONT];
            if (cls < 0) {
                cls = 0;
            }
            if (cls >= DRIVE_ROAD_CLASSES) {
                cls = DRIVE_ROAD_CLASSES - 1;
            }
            for (k = 0; k < DRIVE_ROAD_CLASSES; k++) {
                feat[DRIVE_ROAD_CONT + k] = (k == cls) ? 1.0f : 0.0f;
            }
            x = feat;
            F = DRIVE_ROAD_IN;
        }
        drive_kv_token(d, K, V, *s, x, F, w1, b1, g, be, w2, b2, h1, ln, h2);
        *s += 1;
    }
}

// One car. y is the encoder output (H). sc is that car's scratch row.
// qproj is already filled. Does not touch another car's state.
static void drive_encode_row(DriveEnc* d, const float* row, float* y, float* sc) {
    int H = d->hidden;
    int Q = DRIVE_ATTN_Q;
    int S = 1 + MAX_ROAD_SEGMENT_OBSERVATIONS + MAX_AGENTS;
    int part_off = EGO_FEATURES;
    int road_off = EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS;
    int head_dim = H / DRIVE_ATTN_HEADS;
    float scale = 1.0f / sqrtf((float)head_dim);
    float* h1 = sc;
    float* ln = sc + H;
    float* h2 = sc + 2 * H;
    float* K = sc + 3 * H;
    float* V = K + (size_t)S * H;
    float* ctx = V + (size_t)S * H;
    float* pre = ctx + (size_t)Q * H;
    float* scores = pre + (size_t)Q * H;
    int s = 0;
    int h;
    int head;
    drive_kv_slots(d, K, V, &s, row, 0, 1, EGO_FEATURES, 0,
        d->ego_w1, d->ego_b1, d->ego_g, d->ego_be,
        d->ego_w2, d->ego_b2, h1, ln, h2);
    drive_kv_slots(d, K, V, &s, row, road_off,
        MAX_ROAD_SEGMENT_OBSERVATIONS, ROAD_FEATURES, 1,
        d->road_w1, d->road_b1, d->road_g, d->road_be,
        d->road_w2, d->road_b2, h1, ln, h2);
    drive_kv_slots(d, K, V, &s, row, part_off, MAX_AGENTS,
        PARTNER_FEATURES, 0,
        d->part_w1, d->part_b1, d->part_g, d->part_be,
        d->part_w2, d->part_b2, h1, ln, h2);
    for (h = 0; h < Q * H; h++) {
        ctx[h] = 0.0f;
    }
    for (head = 0; head < DRIVE_ATTN_HEADS; head++) {
        int h0 = head * head_dim;
        for (int q = 0; q < Q; q++) {
            float m;
            float sum = 0.0f;
            int t;
            int i;
            for (t = 0; t < S; t++) {
                float dot = 0.0f;
                for (i = 0; i < head_dim; i++) {
                    dot += d->qproj[q * H + h0 + i] * K[t * H + h0 + i];
                }
                scores[t] = dot * scale;
            }
            m = scores[0];
            for (t = 1; t < S; t++) {
                if (scores[t] > m) {
                    m = scores[t];
                }
            }
            for (t = 0; t < S; t++) {
                scores[t] = expf(scores[t] - m);
                sum += scores[t];
            }
            for (t = 0; t < S; t++) {
                scores[t] /= sum;
            }
            for (i = 0; i < head_dim; i++) {
                float acc = 0.0f;
                for (t = 0; t < S; t++) {
                    acc += scores[t] * V[t * H + h0 + i];
                }
                ctx[q * H + h0 + i] = acc;
            }
        }
    }
    for (int q = 0; q < Q; q++) {
        drive_project(pre + (size_t)q * H, ctx + (size_t)q * H,
            d->wo, d->bo, H, H);
        for (h = 0; h < H; h++) {
            pre[q * H + h] += d->q_lat[q * H + h];
        }
        drive_ln_vec(ctx + (size_t)q * H, pre + (size_t)q * H,
            d->attn_g, d->attn_be, H);
    }
    drive_project(y, ctx, d->fc_w, d->fc_b, Q * H, H);
    for (h = 0; h < H; h++) {
        y[h] = tanhf(y[h]);
    }
}

static void drive_encode(PufferNet* net, float* obs) {
    DriveEnc* d = net->drive;
    int B = d->batch;
    int H = d->hidden;
    int Q = DRIVE_ATTN_Q;
    int S = 1 + MAX_ROAD_SEGMENT_OBSERVATIONS + MAX_AGENTS;
    int stride = (3 + 2 * S + 2 * Q) * H + S;
    int q;
    for (q = 0; q < Q; q++) {
        drive_project(d->qproj + (size_t)q * H, d->q_lat + (size_t)q * H,
            d->wq, d->bq, H, H);
    }
#ifdef _OPENMP
#pragma omp parallel for if (B >= 8) schedule(static)
#endif
    for (int b = 0; b < B; b++) {
        drive_encode_row(d, obs + (size_t)b * OBS_SIZE,
            net->encoder->output + (size_t)b * H,
            d->scratch + (size_t)b * stride);
    }
}
#endif

static void puf_encode(PufferNet* net, float* obs) {
#ifdef PUFFER_DRIVE
    drive_encode(net, obs);
#else
    linear(net->encoder, obs);
#endif
}

PufferNet* make_puffernet(Weights* weights, int num_agents, int input_dim,
        int hidden_dim, int num_layers, int logit_sizes[], int num_actions) {
    PufferNet* net = calloc(1, sizeof(PufferNet));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents * input_dim, sizeof(float));
    int atn_sum = 0;
    int is_continuous = 1;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += logit_sizes[i];
        if (logit_sizes[i] != 1) is_continuous = 0;
    }
    net->is_continuous = is_continuous;
    net->num_actions = num_actions;

#ifdef PUFFER_DRIVE
    assert(input_dim == OBS_SIZE);
    assert(hidden_dim % 8 == 0);
    net->encoder = calloc(1, sizeof(Linear)
        + (size_t)num_agents * hidden_dim * sizeof(float));
    net->encoder->output = (float*)(net->encoder + 1);
    net->encoder->batch_size = num_agents;
    net->encoder->input_dim = input_dim;
    net->encoder->output_dim = hidden_dim;
    net->drive = drive_enc_load(weights, num_agents, hidden_dim);
#else
    net->encoder = make_linear(weights, num_agents, input_dim, hidden_dim);
#endif
    net->decoder = make_linear(weights, num_agents, hidden_dim, atn_sum + 1);
    if (net->is_continuous) {
        net->log_std = get_weights_aligned(weights, num_actions);
    }
    net->mingru  = make_mingru(weights, num_agents, hidden_dim, num_layers);
    if (!net->is_continuous) {
        net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, num_actions);
    }
    return net;
}

void _gaussian_mean(float* input, float* output, int batch_size, int num_actions) {
    for (int b = 0; b < batch_size; b++) {
        // +1 skips the value head fused into the decoder output
        int in_adr = b * (num_actions + 1);
        for (int a = 0; a < num_actions; a++)
            output[b * num_actions + a] = input[in_adr + a];
    }
}

void forward_puffernet(PufferNet* net, float* observations, float* actions,
        const unsigned char* mask, const float* terminals) {
    mingru_zero_term(net->mingru, terminals);
    puf_encode(net, observations);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    if (net->is_continuous) {
        _gaussian_mean(net->decoder->output, actions, net->num_agents, net->num_actions);
    } else {
        multidiscrete(net->multidiscrete, net->decoder->output, actions, 0, mask);
    }
}

#ifdef PUFFER_DRIVE
// Critic head for one car. Reads the MinGRU state and does not write it,
// so the action forward still consumes this observation once.
static float puffernet_value(PufferNet* net, const float* obs, int agent) {
    DriveEnc* d = net->drive;
    MinGRU* g = net->mingru;
    int H = d->hidden;
    int Q = DRIVE_ATTN_Q;
    int S = 1 + MAX_ROAD_SEGMENT_OBSERVATIONS + MAX_AGENTS;
    int stride = (3 + 2 * S + 2 * Q) * H + S;
    static float* scratch = NULL;
    static int scratch_n = 0;
    static float* hid = NULL;
    static float* nxt = NULL;
    static float* combined = NULL;
    static int hid_n = 0;
    int q;
    int l;
    assert(agent >= 0 && agent < d->batch);
    assert(H == g->hidden_size);
    if (scratch_n < stride) {
        scratch = (float*)realloc(scratch, (size_t)stride * sizeof(float));
        assert(scratch);
        scratch_n = stride;
    }
    if (hid_n < 3 * H) {
        hid = (float*)realloc(hid, (size_t)H * sizeof(float));
        nxt = (float*)realloc(nxt, (size_t)H * sizeof(float));
        combined = (float*)realloc(combined, (size_t)3 * H * sizeof(float));
        assert(hid && nxt && combined);
        hid_n = 3 * H;
    }
    for (q = 0; q < Q; q++) {
        drive_project(d->qproj + (size_t)q * H, d->q_lat + (size_t)q * H,
            d->wq, d->bq, H, H);
    }
    drive_encode_row(d, obs, hid, scratch);
    for (l = 0; l < g->num_layers; l++) {
        Linear* proj = g->proj[l];
        int Id = proj->input_dim;
        int O = proj->output_dim;
        const float* sb = g->state + ((size_t)l * g->batch_size + agent) * H;
        int o;
        int h;
        assert(Id == H && O == 3 * H);
        for (o = 0; o < O; o++) {
            const float* w = proj->weights + (size_t)o * Id;
            float sum = 0.0f;
            int i;
            for (i = 0; i < Id; i++) {
                sum += hid[i] * w[i];
            }
            combined[o] = sum;
        }
        for (h = 0; h < H; h++) {
            float hidden = combined[h];
            float gate = combined[H + h];
            float hw = combined[2 * H + h];
            float s = sb[h];
            float gate_s = _sigmoid(gate);
            float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : _sigmoid(hidden);
            float mingru_out = s + gate_s * (h_tilde - s);
            float hw_s = _sigmoid(hw);
            nxt[h] = hw_s * mingru_out + (1.0f - hw_s) * hid[h];
        }
        memcpy(hid, nxt, (size_t)H * sizeof(float));
    }
    {
        Linear* dec = net->decoder;
        int Id = dec->input_dim;
        int row = dec->output_dim - 1;
        const float* w = dec->weights + (size_t)row * Id;
        float sum = 0.0f;
        int i;
        assert(row >= 0 && Id == H);
        for (i = 0; i < Id; i++) {
            sum += hid[i] * w[i];
        }
        return sum;
    }
}
#endif

void free_puffernet(PufferNet* net) {
    free(net->obs);
#ifdef PUFFER_DRIVE
    if (net->drive) {
        free(net->drive->scratch);
        free(net->drive->qproj);
        free(net->drive);
    }
#endif
    free(net->encoder);
    free(net->decoder);
    free_mingru(net->mingru);
    free(net->multidiscrete);
    free(net);
}

#ifdef PUFFERCPU_EVAL_MAIN

#ifndef ENV_HEADER
#error "ENV_HEADER required for PUFFERCPU_EVAL_MAIN"
#endif

#include ENV_HEADER

#ifdef PUFFER_DRIVE
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "cJSON.c"
#include "offline.h"
#pragma clang diagnostic pop
#include "anchor_interactive.h"
#include "eval.h"
#endif

#if !defined(PUF_NMMO3_NET) && !defined(PUF_ASTEROIDS_NET) && !defined(PUF_MINIMAL_NET) && !defined(PUF_CRAFTAX_NET)
static int puf_align8(int n) {
    return (n + 7) & ~7;
}

static int puffernet_weight_count(int input_dim, int hidden, int layers,
        const int act_sizes[], int num_actions) {
    int atn_sum = 0;
    int continuous = 1;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += act_sizes[i];
        if (act_sizes[i] != 1) {
            continuous = 0;
        }
    }
    int n = 0;
#ifdef PUFFER_DRIVE
    assert(input_dim == OBS_SIZE);
    n = drive_encoder_nfloats(hidden);
#else
    n = puf_align8(n + hidden * input_dim);
#endif
    n = puf_align8(n + (atn_sum + 1) * hidden);
    if (continuous) {
        n = puf_align8(n + num_actions);
    }
    for (int l = 0; l < layers; l++) {
        n = puf_align8(n + 3 * hidden * hidden);
    }
    return n;
}

#ifdef MAX_AGENTS
// An old checkpoint has 7 ego features and 63 partners (973 inputs).
// Extra ego features and the extra partner slot are zero, so that
// policy's actions stay put.
static Weights* puf_widen_partner_ckpt(Weights* weights, int hidden,
        int layers, const int act_sizes[], int num_actions,
        int file_floats) {
    const int legacy_ego = 7;
    const int legacy_partners = 63;
    if (EGO_FEATURES < legacy_ego) return weights;
    if (MAX_AGENTS < legacy_partners) return weights;
    int legacy_in = legacy_ego
        + PARTNER_FEATURES * legacy_partners
        + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    int ego_insert = EGO_FEATURES - legacy_ego;
    int partner_insert =
        (MAX_AGENTS - legacy_partners) * PARTNER_FEATURES;
    if (OBS_SIZE != legacy_in + ego_insert + partner_insert) {
        return weights;
    }
    int legacy_need = puffernet_weight_count(legacy_in, hidden, layers,
        act_sizes, num_actions);
    if (file_floats != legacy_need) return weights;

    int road_at = legacy_ego + PARTNER_FEATURES * legacy_partners;
    int old_enc = puf_align8(hidden * legacy_in);
    int new_enc = puf_align8(hidden * OBS_SIZE);
    int tail = file_floats - old_enc;
    if (tail < 0) return weights;
    int new_floats = new_enc + tail;
    int need = puffernet_weight_count(OBS_SIZE, hidden, layers,
        act_sizes, num_actions);
    if (new_floats != need) return weights;

    Weights* grown = calloc(1, sizeof(Weights)
        + (size_t)(new_floats + 7) * sizeof(float));
    if (!grown) {
        perror("calloc");
        exit(1);
    }
    grown->data = (float*)(grown + 1);
    grown->size = new_floats + 7;
    int partner_n = road_at - legacy_ego;
    int new_road = legacy_ego + ego_insert + partner_n + partner_insert;
    for (int o = 0; o < hidden; o++) {
        float* src = weights->data + (size_t)o * (size_t)legacy_in;
        float* dst = grown->data + (size_t)o * (size_t)OBS_SIZE;
        memcpy(dst, src, (size_t)legacy_ego * sizeof(float));
        memcpy(dst + legacy_ego + ego_insert, src + legacy_ego,
            (size_t)partner_n * sizeof(float));
        memcpy(dst + new_road, src + road_at,
            (size_t)(legacy_in - road_at) * sizeof(float));
    }
    memcpy(grown->data + new_enc, weights->data + old_enc,
        (size_t)tail * sizeof(float));
    free(weights);
    fprintf(stderr,
        "loaded a %d-input checkpoint; added ego and partner inputs are zero\n",
        legacy_in);
    return grown;
}

// The previous ego block has no target speed. That input is one column
// at the end of the ego features, and its weight is zero.
static Weights* puf_widen_target_ckpt(Weights* weights, int hidden,
        int layers, const int act_sizes[], int num_actions,
        int file_floats) {
    int old_ego = EGO_TARGET_SPEED;
    int old_in = old_ego
        + PARTNER_FEATURES * MAX_AGENTS
        + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    if (OBS_SIZE != old_in + 1) {
        return weights;
    }
    int old_need = puffernet_weight_count(old_in, hidden, layers,
        act_sizes, num_actions);
    if (file_floats != old_need) {
        return weights;
    }
    int old_enc = puf_align8(hidden * old_in);
    int new_enc = puf_align8(hidden * OBS_SIZE);
    int tail = file_floats - old_enc;
    if (tail < 0) {
        return weights;
    }
    int new_floats = new_enc + tail;
    int need = puffernet_weight_count(OBS_SIZE, hidden, layers,
        act_sizes, num_actions);
    if (new_floats != need) {
        return weights;
    }

    Weights* grown = calloc(1, sizeof(Weights)
        + (size_t)(new_floats + 7) * sizeof(float));
    if (!grown) {
        perror("calloc");
        exit(1);
    }
    grown->data = (float*)(grown + 1);
    grown->size = new_floats + 7;
    for (int o = 0; o < hidden; o++) {
        float* src = weights->data + (size_t)o * (size_t)old_in;
        float* dst = grown->data + (size_t)o * (size_t)OBS_SIZE;
        memcpy(dst, src, (size_t)old_ego * sizeof(float));
        memcpy(dst + old_ego + 1, src + old_ego,
            (size_t)(old_in - old_ego) * sizeof(float));
    }
    memcpy(grown->data + new_enc, weights->data + old_enc,
        (size_t)tail * sizeof(float));
    free(weights);
    fprintf(stderr,
        "loaded a %d-input checkpoint; target-speed input has zero weight\n",
        old_in);
    return grown;
}
#endif
#endif

static void puf_find_latest_checkpoint(const char* dir,
        char* out, size_t out_size, time_t* best_time) {
    DIR* dp = opendir(dir);
    if (!dp) {
        return;
    }

    struct dirent* ent;
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

        size_t pn = strlen(path);
        if (S_ISDIR(st.st_mode)) {
            puf_find_latest_checkpoint(path, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode) && pn >= 4 &&
                strcmp(path + pn - 4, ".bin") == 0 &&
                st.st_ctime >= *best_time) {
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", path);
        }
    }

    closedir(dp);
}

static const char* puf_model_path(const char* env_name, const char* cli_path,
        int latest, char* out, size_t out_size) {
    if (cli_path) {
        return cli_path;
    }
    if (latest) {
        char root[2048];
        snprintf(root, sizeof(root), "checkpoints/%s", env_name);
        out[0] = 0;
        time_t best_time = 0;
        puf_find_latest_checkpoint(root, out, out_size, &best_time);
        return out[0] ? out : NULL;
    }
#ifdef PLATFORM_WEB
    snprintf(out, out_size, "resources/%s/%s_weights.bin", env_name, env_name);
    return out;
#else
    return NULL;
#endif
}

#ifdef PUFFER_DRIVE
// Value of the observation the sim panel is about to draw.
static void drive_note_value(PufferNet* net, Drive* env) {
    int n = env->active_agent_count;
    int slot = env->selected_agent_idx;
    float* obs;
    if (!net || n <= 0) {
        env->predicted_ready = 0;
        return;
    }
    if (slot < 0 || slot >= n) {
        slot = 0;
    }
    obs = env->agents[slot].observations;
    if (!obs) {
        env->predicted_ready = 0;
        return;
    }
    env->predicted_value = puffernet_value(net, obs, slot);
    env->predicted_ready = 1;
}
#endif

int main(int argc, char** argv) {
    const char* env_name = PUFFER_ENV_NAME;
    int argi = 1;
    if (argc >= 2 && argv[1][0] && argv[1][0] != '-' &&
            strchr(argv[1], '=') == NULL && strchr(argv[1], '/') == NULL &&
            strstr(argv[1], ".bin") == NULL && strcmp(argv[1], "latest") != 0) {
        if (strcmp(argv[1], "eval") != 0 && strcmp(argv[1], "train") != 0
                && strcmp(argv[1], "match") != 0 && strcmp(argv[1], "sweep") != 0) {
            env_name = argv[1];
        }
        argi = 2;
    }
    Ini ini = {0};
    int headless = 0;
    int record = 0;
    int record_frames = 91;
    const char* record_path = NULL;
    const char* cli_path = NULL;
    int cli_latest = 0;
    const char* anchor_path = NULL;
    int do_eval = 0;
    int saw_data_dir = 0;
    int saw_map_id = 0;
    int saw_episode = 0;

    char* ini_argv[argc > 0 ? argc : 1];
    int ini_argc = 0;
    for (int i = argi; i < argc; i++) {
        if (strstr(argv[i], "data_dir") != NULL) {
            saw_data_dir = 1;
        }
        if (strstr(argv[i], "map_id") != NULL) {
            saw_map_id = 1;
        }
        if (strstr(argv[i], "episode_length") != NULL) {
            saw_episode = 1;
        }
        if (strcmp(argv[i], "--anchor") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--anchor needs a checkpoint path\n");
                return 1;
            }
            anchor_path = argv[++i];
            continue;
        }
        if (strncmp(argv[i], "--anchor=", 9) == 0) {
            anchor_path = argv[i] + 9;
            continue;
        }
        if (strcmp(argv[i], "--eval") == 0) {
            do_eval = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cli_path = argv[++i];
            }
            continue;
        }
        if (strncmp(argv[i], "--eval=", 7) == 0) {
            do_eval = 1;
            cli_path = argv[i] + 7;
            continue;
        }
        if (strcmp(argv[i], "--headless") == 0) {
            headless = 1;
            continue;
        }
        if (strcmp(argv[i], "--record") == 0) {
            record = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                record_path = argv[++i];
            }
            continue;
        }
        if (strcmp(argv[i], "--record-frames") == 0 && i + 1 < argc) {
            record_frames = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "latest") == 0) {
            cli_latest = 1;
            continue;
        }
        if (strchr(argv[i], '=') == NULL &&
                (strstr(argv[i], ".bin") != NULL || strchr(argv[i], '/'))) {
            cli_path = argv[i];
            continue;
        }
        ini_argv[ini_argc++] = argv[i];
    }
#ifndef PUFFER_DRIVE
    if (anchor_path) {
        fprintf(stderr, "--anchor is the drive player\n");
        return 1;
    }
    if (do_eval) {
        fprintf(stderr, "--eval is the drive player\n");
        return 1;
    }
#else
    // The anchor demo loads the BC anchor and the regularized Town01
    // policy together. A positional .bin replaces that policy.
    if (do_eval && anchor_path) {
        fprintf(stderr, "--eval cannot be combined with --anchor\n");
        return 1;
    }
    if (anchor_path && !cli_path && !cli_latest) {
        cli_path = "checkpoints/drive/town01_bc1000/0000000578813952.bin";
    }
#endif
    puf_ini_load_env(&ini, env_name, ini_argc, ini_argv);
#ifdef PUFFER_DRIVE
    if (do_eval) {
        puf_ini_load_file(&ini, "config/benchmark.ini");
    }
#endif
#ifdef PUFFER_DRIVE
    EvalPick pick = {0};
    if (do_eval && !cli_path && !cli_latest) {
        eval_save_best(&ini, &pick);
        cli_path = pick.path;
    }
#endif
    if (record) {
        char default_path[256];
        snprintf(default_path, sizeof(default_path), "logs/%s.mp4", env_name);
        puf_record_request(record_path ? record_path : default_path, record_frames, 10);
    }
    // Trainer default.ini eval_episodes is for ./puffer eval, not CPU play.
    // Honor it only when the CLI actually passed --eval_episodes=N.
    int eval_episodes = 0;
    for (int i = 0; i < ini_argc; i++) {
        if (strstr(ini_argv[i], "eval_episodes") != NULL) {
            eval_episodes = puf_ini_get(&ini, "base", "eval_episodes");
            break;
        }
    }

    int act_sizes[] = ACT_SIZES;
    int num_actions = sizeof(act_sizes) / sizeof(act_sizes[0]);
    char path_buf[1024];
    const char* path = puf_model_path(env_name, cli_path, cli_latest,
        path_buf, sizeof(path_buf));
    Weights* weights = path ? load_weights(path) : NULL;
    int hidden_size = puf_ini_get(&ini, "policy", "hidden_size");
    int num_layers = puf_ini_get(&ini, "policy", "num_layers");
    int file_floats = weights ? (weights->size - 7) : 0;
#ifdef PUF_NMMO3_NET
    int need = nmmo3_weight_count(hidden_size, num_layers);
#elif defined(PUF_ASTEROIDS_NET)
    int need = asteroids_weight_count(hidden_size, num_layers);
#elif defined(PUF_MINIMAL_NET)
    int need = minimal_weight_count(hidden_size, num_layers);
#elif defined(PUF_CRAFTAX_NET)
    int need = craftax_weight_count(hidden_size, num_layers);
#else
    int need = puffernet_weight_count(OBS_SIZE, hidden_size, num_layers,
        act_sizes, num_actions);
#ifdef MAX_AGENTS
    if (weights && file_floats + 7 < need) {
        weights = puf_widen_partner_ckpt(weights, hidden_size, num_layers,
            act_sizes, num_actions, file_floats);
        file_floats = weights->size - 7;
    }
    if (weights && file_floats + 7 < need) {
        weights = puf_widen_target_ckpt(weights, hidden_size, num_layers,
            act_sizes, num_actions, file_floats);
        file_floats = weights->size - 7;
    }
#endif
#endif
    // A missing file or the wrong network used to abort, or play action 0
    // with no message. Keep going and say so.
    int opened = weights != NULL;
    int raw_floats = file_floats;
    int fits = !weights || (need - file_floats <= 7 && file_floats <= need);
#if !defined(PUF_NMMO3_NET) && !defined(PUF_ASTEROIDS_NET) \
        && !defined(PUF_MINIMAL_NET) && !defined(PUF_CRAFTAX_NET)
    // A sweep trial's width and depth are not in config/drive.ini.
    // One power-of-two width and layer count matches the file.
    if (do_eval && weights && !fits) {
        int match_h = 0;
        int match_l = 0;
        int hits = 0;
        for (int h = 64; h <= 512; h *= 2) {
            for (int layers = 1; layers <= 4; layers++) {
                int alt = puffernet_weight_count(OBS_SIZE, h, layers,
                    act_sizes, num_actions);
                if (file_floats <= alt && alt - file_floats <= 7) {
                    match_h = h;
                    match_l = layers;
                    hits++;
                }
            }
        }
        if (hits == 1) {
            hidden_size = match_h;
            num_layers = match_l;
            need = puffernet_weight_count(OBS_SIZE, hidden_size,
                num_layers, act_sizes, num_actions);
            fits = 1;
            fprintf(stderr,
                "eval weights match hidden %d layers %d\n",
                hidden_size, num_layers);
        }
    }
#endif
    if (weights && !fits) {
        free(weights);
        weights = NULL;
        file_floats = 0;
    }
    int have_net = weights != NULL;
    int shown_floats = have_net ? file_floats : raw_floats;
    if (headless && eval_episodes > 0) {
        printf("CPU_META env=%s path=%s file_floats=%d need=%d untrained=%d hidden=%d layers=%d\n",
            env_name, path ? path : "-", shown_floats, need, !have_net,
            hidden_size, num_layers);
        fflush(stdout);
    }

    Dict* env_sec = puf_ini_section(&ini, "env", 0);
    int frameskip = 1;
    DictItem* fs_item = dict_find(env_sec, "frameskip");
    if (fs_item && fs_item->value > 1.0) {
        frameskip = fs_item->value;
        dict_set(env_sec, "frameskip", 1);
    }
#ifdef PUF_EVAL_SHOULD_FORWARD
    frameskip = 1;
#endif

#ifdef PUFFER_DRIVE
    if (do_eval) {
        if (!have_net) {
            fprintf(stderr, "eval checkpoint did not load (%s)\n",
                path ? path : "-");
            return 1;
        }
        if (!pick.path[0]) {
            snprintf(pick.path, sizeof(pick.path), "%s", path);
        }
        eval_run(&ini, env_sec, weights, hidden_size, num_layers,
            act_sizes, num_actions, &pick);
        free(weights);
        puf_ini_free(&ini);
        return 0;
    }
    if (anchor_path) {
        if (!saw_data_dir) {
            dict_set_str(env_sec, "data_dir", "data/womd");
        }
        if (!saw_episode) {
            dict_set(env_sec, "episode_length", 100000);
        }
        if (headless || record) {
            fprintf(stderr, "--anchor opens a window\n");
            return 1;
        }
        Env env = {0};
        env.checkpoint_ok = 1;
        env.checkpoint_error[0] = 0;
        env.sim_follow = 1;
        env.sim_lock_actions = 1;
        env.sim_user_size = 1;
        int start = 0;
        if (saw_map_id) {
            start = ademo_scene_index((int)dict_get(env_sec, "map_id"));
        }
        PufferNet* policy = NULL;
        if (have_net) {
            policy = make_puffernet(weights, 1, OBS_SIZE,
                hidden_size, num_layers, act_sizes, num_actions);
            printf("policy %s  hidden %d  layers %d\n",
                path, hidden_size, num_layers);
            fflush(stdout);
        } else if (path) {
            if (!opened) {
                fprintf(stderr, "policy did not load (%s: file not found)\n",
                    path);
            } else {
                fprintf(stderr,
                    "policy did not load (%s: %d floats, network wants %d)\n",
                    path, raw_floats, need);
            }
        }
        ademo_open(anchor_path);
        ademo_link(policy);
        ademo_boot(&env, env_sec, start);
        env.hud_extra = ademo_hud;
        env.ui_hot = ademo_hot;
        env.sim_reset = ademo_on_reset;
        env.sim_scene = ademo_on_scene;
        ademo_run(&env);
        ademo_close();
        if (policy) {
            free_puffernet(policy);
        }
        free(weights);
        ademo_shutdown();
        puf_close(&env);
        free(env.map_name);
        env.map_name = NULL;
        puf_ini_free(&ini);
        return 0;
    }
#endif
    Env env = {0};
    puf_init(&env, env_sec);
#ifdef PUF_HAS_SIM_CONTROL
#ifdef PUFFER_DRIVE
    if (anchor_path) {
        env.checkpoint_ok = 1;
        env.checkpoint_error[0] = 0;
        env.sim_follow = 1;
        env.sim_lock_actions = 1;
        env.sim_user_size = 1;
    } else
#endif
    {
    env.checkpoint_ok = have_net;
    if (!have_net) {
        if (!path) {
            snprintf(env.checkpoint_error, sizeof(env.checkpoint_error),
                "Checkpoint cannot be loaded, showing random behavior");
        } else if (!opened) {
            snprintf(env.checkpoint_error, sizeof(env.checkpoint_error),
                "Checkpoint cannot be loaded, showing random behavior (%s: file not found)",
                path);
        } else {
            snprintf(env.checkpoint_error, sizeof(env.checkpoint_error),
                "Checkpoint cannot be loaded, showing random behavior (%s: %d floats, network wants %d)",
                path, raw_floats, need);
        }
        sim_checkpoint_rejected(&env);
    }
    }
#endif

    // Heap, not VLAs: snake=256 / obs=968 overflows the 512KB WASM stack.
    size_t n_obs = (size_t)env.num_agents * (size_t)OBS_SIZE;
    size_t n_atn = (size_t)env.num_agents * (size_t)NUM_ATNS;
    size_t n_agt = env.num_agents;
    obs_t* observations = calloc(n_obs, sizeof(obs_t));
#if !defined(PUF_NMMO3_NET) && !defined(PUF_ASTEROIDS_NET) && !defined(PUF_MINIMAL_NET) && !defined(PUF_CRAFTAX_NET)
    float* obs_f = calloc(n_obs, sizeof(float));
#endif
    float* actions = calloc(n_atn, sizeof(float));
    float* rewards = calloc(n_agt, sizeof(float));
    float* terminals = calloc(n_agt, sizeof(float));
    int act_n = 0;
    for (int i = 0; i < num_actions; i++) {
        act_n += act_sizes[i];
    }
    unsigned char* masks = calloc(
        (size_t)env.num_agents * (size_t)act_n, 1);
    memset(masks, 1, (size_t)env.num_agents * (size_t)act_n);
    for (int i = 0; i < env.num_agents; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i * NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
        env.agents[i].action_mask = masks + i * act_n;
        env.agents[i].policy = 0;
    }
    puf_reset(&env);



#ifdef PUF_NMMO3_NET
    MMONet* net = NULL;
    if (have_net) {
        net = init_mmonet_arch(weights, env.num_agents,
            hidden_size, num_layers);
    }
#elif defined(PUF_ASTEROIDS_NET)
    AsteroidsNet* net = NULL;
    if (have_net) {
        net = init_asteroids_net(weights, env.num_agents,
            hidden_size, num_layers);
    }
#elif defined(PUF_MINIMAL_NET)
    MinimalNet* net = NULL;
    if (have_net) {
        net = init_minimal_net(weights, env.num_agents,
            hidden_size, num_layers);
    }
#elif defined(PUF_CRAFTAX_NET)
    CraftaxNet* net = NULL;
    if (have_net) {
        net = init_craftax_net(weights, env.num_agents,
            hidden_size, num_layers);
    }
#else
    PufferNet* net = NULL;
    if (have_net) {
        net = make_puffernet(weights, env.num_agents, OBS_SIZE,
            hidden_size, num_layers, act_sizes, num_actions);
    }
#endif

#ifndef PLATFORM_WEB
    SetConfigFlags(FLAG_MSAA_4X_HINT);
#endif

#ifndef PUF_STEPS_PER_SEC
#define PUF_STEPS_PER_SEC 60
#endif
    // Interactive: one displayed frame is one step, paced by SetTargetFPS.
    // A catch-up burst draws several steps in one frame, so the view jumps.
#ifndef PUF_HAS_SIM_CONTROL
    const double sim_dt = 1.0 / PUF_STEPS_PER_SEC;
    double sim_accum = 0.0;
    double sim_prev = -1.0;
    int sim_tick_cap = 5;
#endif
    int hold = 0;
    int steps = 0;
    int recording = puf_record_enabled();
    if (recording && path && !have_net) {
        fprintf(stderr, "record: checkpoint did not load (%s)\n", path);
        exit(1);
    }
    if (recording && !have_net) {
        // Coast at the logged speed. Action 0 is full brake and full steer.
        for (int i = 0; i < env.num_agents; i++) {
            actions[i * NUM_ATNS + 0] = 3;
            actions[i * NUM_ATNS + 1] = 6;
        }
    }
#ifndef PLATFORM_WEB
    if (!headless && !recording) {
        SetTargetFPS(60);
    }
#endif
    if (!headless && !recording) {
#ifdef PUFFER_DRIVE
        drive_note_value(net, &env);
#endif
        puf_render(&env);
    }
    // Web: Raylib 5.5 WindowShouldClose() always emscripten_sleep(16).
    // Pace frames with puf_web_vsync. Native checks WindowShouldClose after render.
    while (recording
            ? !puf_record_done()
            : (headless
            ? (eval_episodes > 0 ? (env.log.n < eval_episodes)
                                 : (steps < 1024))
            : IsWindowReady())) {
        int ticks = 1;
        if (!headless && !recording) {
            puf_sim_poll(&env);
#ifdef PUF_HAS_SIM_CONTROL
            ticks = 0;
            if (!puf_sim_paused(&env)) {
                ticks = 1;
            }
#else
            double now = GetTime();
            ticks = 0;
            if (puf_sim_paused(&env)) {
                // A held accumulator would burst several steps on resume.
                sim_accum = 0.0;
            } else if (sim_prev >= 0.0) {
                double dt = now - sim_prev;
                if (dt < 0.0) dt = 0.0;
                if (dt > 0.25) dt = 0.25;
                sim_accum += dt;
                if (sim_accum > sim_dt * sim_tick_cap) {
                    sim_accum = sim_dt * sim_tick_cap;
                }
                while (sim_accum >= sim_dt && ticks < sim_tick_cap) {
                    sim_accum -= sim_dt;
                    ticks++;
                }
            } else {
                ticks = 1;
            }
            sim_prev = now;
#endif
        }
        for (int t = 0; t < ticks; t++) {
#ifdef PUF_EVAL_SHOULD_FORWARD
            int eval_fwd = env.tick_frames_left <= 0;
#else
            int eval_fwd = 1;
#endif
            int eval_human = !headless && IsWindowReady() &&
                IsKeyDown(KEY_LEFT_SHIFT);
            // Bin 0 is full brake and full lock. With no weights, coast,
            // and let Shift overwrite the selected car inside puf_step.
            if (!have_net) {
                for (int i = 0; i < env.num_agents; i++) {
                    actions[i * NUM_ATNS + 0] = 3;
                    actions[i * NUM_ATNS + 1] = 6;
                }
            }
            if (have_net && eval_fwd && hold == 0 && !eval_human) {
#ifdef PUF_NMMO3_NET
                forward(net, observations, terminals, actions);
#elif defined(PUF_ASTEROIDS_NET)
                forward_asteroids(net, (float*)observations, terminals, actions);
#elif defined(PUF_MINIMAL_NET)
                forward_minimal(net, (float*)observations, terminals, actions);
#elif defined(PUF_CRAFTAX_NET)
                forward_craftax(net, (float*)observations, terminals, actions, masks);
                env.predicted_value = craftax_value(net, 0);
#else
                float* fwd = (float*)observations;
                if (sizeof(obs_t) != sizeof(float)) {
                    for (size_t i = 0; i < n_obs; i++) {
                        obs_f[i] = observations[i];
                    }
                    fwd = obs_f;
                }
                forward_puffernet(net, fwd, actions, masks, terminals);
#endif
            }
            puf_step(&env);
            if (headless) {
                steps++;
            }
            if (eval_fwd) {
                int reset_hold = 0;
                for (int a = 0; a < env.num_agents; a++) {
                    if (rewards[a] != 0.0f || terminals[a] > 0.5f) {
                        reset_hold = 1;
                    }
                }
                hold = reset_hold ? 0 : (hold + 1) % frameskip;
            }
        }
        if (!headless || recording) {
#ifdef PUFFER_DRIVE
            if (!recording) {
                drive_note_value(net, &env);
            }
#endif
            puf_render(&env);
            if (recording && puf_record_done()) {
                break;
            }
#ifndef PLATFORM_WEB
            if (!recording && WindowShouldClose()) {
                break;
            }
#endif
        }
    }
    if (recording) {
        puf_record_finish();
    }

    if (headless && eval_episodes > 0) {
        float n = env.log.n;
        float perf = 0.0f;
        float score = 0.0f;
        if (n > 0.0f) {
            Log avg = env.log;
            float* acc = (float*)&avg;
            int nf = sizeof(Log) / sizeof(float);
            for (int i = 0; i < nf; i++) {
                acc[i] /= n;
            }
            Dict out = {0};
            puf_log(&avg, &out);
            dict_set(&out, "n", n);
            DictItem* pi = dict_find(&out, "perf");
            DictItem* si = dict_find(&out, "score");
            if (pi) perf = pi->value;
            if (si) score = si->value;
            dict_clear(&out);
        }
        printf("CPU_EVAL env=%s file_floats=%d need=%d match=%d untrained=%d n=%.0f perf=%.6f score=%.6f steps=%d agents=%d\n",
            env_name, shown_floats, need,
            (need - file_floats <= 7) ? 1 : 0,
            !have_net, n, perf, score, steps, env.num_agents);
        fflush(stdout);
    }

    puf_close(&env);
    puf_ini_free(&ini);
    return 0;
}

#endif
