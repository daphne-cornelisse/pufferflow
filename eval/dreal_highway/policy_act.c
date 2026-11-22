/* Greedy or sampled bins from a drive .bin checkpoint.
 * The forward is the CPU player's PufferNet, including MinGRU carry.
 * One process stays up. Records are little-endian on stdin and stdout.
 *
 *   int32 op        1 = step, 0 = quit
 *   int32 n         must equal the batch given on the command line
 *   float obs[n * OBS_SIZE]
 *   float terminals[n]   >0.5 clears that car's carry before the step
 *
 * The reply is float actions[n * 2], accel bin then steer bin.
 *
 *   policy_act CKPT HIDDEN LAYERS BATCH [SEED]
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "constants.h"

typedef struct Weights Weights;
typedef struct PufferNet PufferNet;

Weights* load_weights(const char* filename);
int weights_count(Weights* weights);
int puffernet_weight_count(int input_dim, int hidden, int layers,
    const int logit_sizes[], int num_actions);
PufferNet* make_puffernet(Weights* weights, int num_agents, int input_dim,
    int hidden_dim, int num_layers, int logit_sizes[], int num_actions);
void forward_puffernet(PufferNet* net, float* observations, float* actions,
    const unsigned char* mask, const float* terminals);

static int read_full(void* dst, size_t n) {
    return fread(dst, 1, n, stdin) == n;
}

int main(int argc, char** argv) {
    const char* path;
    int hidden;
    int layers;
    int batch;
    unsigned int seed = 1;
    int acts[2] = ACT_SIZES;
    Weights* weights;
    PufferNet* net;
    float* obs;
    float* terms;
    float* actions;
    if (argc == 2 && strcmp(argv[1], "--obs") == 0) {
        printf("%d\n", OBS_SIZE);
        return 0;
    }
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: policy_act CKPT HIDDEN LAYERS BATCH [SEED]\n");
        return 2;
    }
    path = argv[1];
    hidden = atoi(argv[2]);
    layers = atoi(argv[3]);
    batch = atoi(argv[4]);
    if (argc == 6) {
        seed = (unsigned int)strtoul(argv[5], NULL, 10);
    }
    if (hidden <= 0 || layers <= 0 || batch <= 0) {
        fprintf(stderr, "policy_act bad shape\n");
        return 2;
    }
    weights = load_weights(path);
    if (!weights) {
        fprintf(stderr, "policy_act cannot read %s\n", path);
        return 1;
    }
    int actual = weights_count(weights);
    int expected = puffernet_weight_count(
        OBS_SIZE, hidden, layers, acts, NUM_ATNS);
    if (actual > expected || expected - actual > 7) {
        fprintf(stderr,
            "policy_act checkpoint shape mismatch: %s has %d floats, "
            "but hidden=%d layers=%d needs %d\n",
            path, actual, hidden, layers, expected);
        return 1;
    }
    net = make_puffernet(weights, batch, OBS_SIZE, hidden, layers, acts, NUM_ATNS);
    obs = (float*)calloc((size_t)batch * OBS_SIZE, sizeof(float));
    terms = (float*)calloc((size_t)batch, sizeof(float));
    actions = (float*)calloc((size_t)batch * NUM_ATNS, sizeof(float));
    if (!obs || !terms || !actions || !net) {
        fprintf(stderr, "policy_act alloc failed\n");
        return 1;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    srand(seed);
    for (;;) {
        int32_t op = 0;
        int32_t n = 0;
        if (!read_full(&op, sizeof(op))) {
            break;
        }
        if (op == 0) {
            break;
        }
        if (op != 1) {
            fprintf(stderr, "policy_act bad op %d\n", op);
            return 3;
        }
        if (!read_full(&n, sizeof(n)) || n != batch) {
            fprintf(stderr, "policy_act batch %d got %d\n", batch, n);
            return 4;
        }
        if (!read_full(obs, (size_t)n * OBS_SIZE * sizeof(float))) {
            return 5;
        }
        if (!read_full(terms, (size_t)n * sizeof(float))) {
            return 6;
        }
        forward_puffernet(net, obs, actions, NULL, terms);
        if (fwrite(actions, sizeof(float), (size_t)n * NUM_ATNS, stdout)
                != (size_t)n * NUM_ATNS) {
            return 7;
        }
    }
    return 0;
}
