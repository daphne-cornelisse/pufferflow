// Custom ocean env CUDA. Included by algo.cu.
// Per-env nets live under ocean/<env>/ and install themselves here.

#ifdef PUFFER_DRIVE
#include "../ocean/drive/drive_encoder.cu"
#endif

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
