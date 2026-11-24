// Custom ocean env CUDA. Included by algo.cu.
// Per-env nets live under ocean/<env>/<env>.cu

// Normal(0, std). Used by custom ocean encoders for embeddings.
void puf_normal_init(Prec* dst, float std, ulong seed, cudaStream_t stream) {
    long n = numel(dst->shape);
    assert(n > 0);
    long rand_count = (n % 2 == 0) ? n : n + 1;
    float* buf;
    cudaMalloc(&buf, rand_count * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, buf, rand_count, 0.0f, std);
    curandDestroyGenerator(gen);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst->data, buf, n);
    cudaFree(buf);
}

#ifdef PUFFER_DRIVE
#include "../ocean/drive/drive_encoder.cu"
#endif

// Override encoder vtable when this env has a custom net. No-op otherwise.
static void create_custom_encoder(Encoder* enc) {
#ifdef PUFFER_DRIVE
    create_drive_encoder(enc);
#else
    (void)enc;
#endif
}

static void create_custom_decoder(Decoder* dec) {
    (void)dec;
}
