// One car and a frozen behavioral-cloning anchor. Included from the drive
// player after drive.h. The anchor demo also loads the regularized RL
// policy. A positional .bin replaces it. While that policy drives, the
// panel reports KL(anchor || policy), KL(policy || anchor), and
// 1-Wasserstein, the same three distances as the trainer. While a person
// or the anchor drives, it reports only the entropy of the anchor at o_t.

#ifndef ANCHOR_INTERACTIVE_H
#define ANCHOR_INTERACTIVE_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ADEMO_LAYERS 8
#define ADEMO_ZENTS 32
#define ADEMO_NAME 64
#define ADEMO_HIST 20
#define ADEMO_YOU 0
#define ADEMO_ANCHOR 1
#define ADEMO_POLICY 2

typedef struct {
    char name[ADEMO_NAME];
    int data;
    int nbytes;
} AdemoZipEnt;

typedef struct {
    unsigned char* buf;
    long n;
    int nent;
    AdemoZipEnt ent[ADEMO_ZENTS];
} AdemoZip;

typedef struct {
    int kind;
    int fortran;
    int rows;
    int cols;
    int n;
    const unsigned char* data;
} AdemoNpy;

typedef struct {
    Rectangle card;
    Rectangle button;
    Rectangle policy;
    Rectangle reset;
    Rectangle accel;
    Rectangle steer;
    float text_x;
    float text_w;
    float title_y;
    float hint_y;
    float cmd_y;
    float accel_y;
    float steer_y;
    float legend_y;
    float dist_y;
    float series_y;
    Rectangle series;
    float dist_size;
    float dist_step;
    float head_size;
} AdemoLayout;

static Font g_math_font;
static int g_math_ready;

typedef struct {
    int who;
    int n_layers;
    int in_dim;
    int lin[ADEMO_LAYERS];
    int lout[ADEMO_LAYERS];
    float* W[ADEMO_LAYERS];
    float* b[ADEMO_LAYERS];
    float* Wa;
    float* ba;
    float* Ws;
    float* bs;
    float* mu;
    float* sig;
    float* cur;
    float* nxt;
    int cap;
    float pa[NUM_ACCEL_BINS];
    float ps[NUM_STEER_BINS];
    float log_pa[NUM_ACCEL_BINS];
    float log_ps[NUM_STEER_BINS];
    float rla[NUM_ACCEL_BINS];
    float rls[NUM_STEER_BINS];
    float log_rla[NUM_ACCEL_BINS];
    float log_rls[NUM_STEER_BINS];
    int act_a;
    int act_s;
    float kl;
    float rkl;
    float w1;
    float ent_a;
    float ent_s;
    float hist_kl[ADEMO_HIST];
    float hist_rkl[ADEMO_HIST];
    float hist_w1[ADEMO_HIST];
    int hist_n;
    int hist_at;
    int hist_step;
    PufferNet* net;
    float* gru_save;
    int gru_n;
    int commit;
} Ademo;

static Ademo g_ademo;

static int ademo_policy_on(void) {
    return g_ademo.net != NULL && g_ademo.who == ADEMO_POLICY;
}

// Sorted WOMD index, same order as data/womd/binaries/map_%03d.bin.
// The SDC starts outside the 5 m goal disk, on a busy junction.
// A NULL name is CARLA Town01 (data/mixed/binaries/map_000.bin). That
// map has no trajectory, so its goals come from pick_forward_goal.
#define ADEMO_LOOP 90

typedef struct {
    int map_id;
    const char* name;
} AdemoScene;

static const AdemoScene g_scenes[] = {
    {395, "tfrecord-00040-of-01000_215.json"},
    {565, "tfrecord-00060-of-01000_141.json"},
    {457, "tfrecord-00047-of-01000_197.json"},
    {714, "tfrecord-00076-of-01000_422.json"},
    {405, "tfrecord-00041-of-01000_176.json"},
    {712, "tfrecord-00076-of-01000_338.json"},
    {851, "tfrecord-00089-of-01000_43.json"},
    {601, "tfrecord-00063-of-01000_238.json"},
    {500, "tfrecord-00052-of-01000_184.json"},
    {241, "tfrecord-00024-of-01000_121.json"},
    {0, NULL},
};

#define ADEMO_NSCENE ((int)(sizeof(g_scenes) / sizeof(g_scenes[0])))

static OfflineAgent* g_logs;
static int g_log_n;
static int g_sdc;
static int g_scene_i;
// Waymo logs stay on this directory. Town01 always loads data/mixed.
static char g_log_dir[512];
// Fleet captured at boot, before a Waymo scene overwrites num_agents.
static int g_fleet;
static float* g_obs;
static float* g_act;
static float* g_rew;
static float* g_term;

static uint16_t ademo_u16(const unsigned char* p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

static uint32_t ademo_u32(const unsigned char* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static uint64_t ademo_u64(const unsigned char* p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static void ademo_zip_open(AdemoZip* z, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "anchor npz %s\n", path);
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
        if (ademo_u32(z->buf + off) != 0x04034b50u) {
            break;
        }
        int flags = ademo_u16(z->buf + off + 6);
        int method = ademo_u16(z->buf + off + 8);
        uint32_t csize32 = ademo_u32(z->buf + off + 18);
        uint32_t usize32 = ademo_u32(z->buf + off + 22);
        int namelen = ademo_u16(z->buf + off + 26);
        int extralen = ademo_u16(z->buf + off + 28);
        assert(method == 0 && "anchor npz compressed");
        assert((flags & 8) == 0 && "anchor npz descriptor");
        assert(z->nent < ADEMO_ZENTS);
        assert(namelen > 0 && namelen < ADEMO_NAME);
        uint64_t csize = csize32;
        if (csize32 == 0xffffffffu || usize32 == 0xffffffffu) {
            const unsigned char* extra = z->buf + off + 30 + namelen;
            int left = extralen;
            int found = 0;
            while (left >= 4) {
                int hid = ademo_u16(extra);
                int hsz = ademo_u16(extra + 2);
                assert(4 + hsz <= left);
                if (hid == 1) {
                    int q = 0;
                    if (usize32 == 0xffffffffu) {
                        q += 8;
                    }
                    if (csize32 == 0xffffffffu) {
                        csize = ademo_u64(extra + 4 + q);
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
        AdemoZipEnt* e = &z->ent[z->nent++];
        memcpy(e->name, z->buf + off + 30, (size_t)namelen);
        e->name[namelen] = 0;
        e->data = (int)(off + 30 + namelen + extralen);
        e->nbytes = (int)csize;
        assert(e->data >= 0 && e->nbytes >= 0);
        assert((long)e->data + e->nbytes <= n);
        off = (long)e->data + e->nbytes;
    }
    assert(z->nent > 0 && "anchor npz empty");
}

static int ademo_zip_index(AdemoZip* z, const char* key) {
    char want[ADEMO_NAME];
    snprintf(want, sizeof(want), "%s.npy", key);
    for (int i = 0; i < z->nent; i++) {
        if (strcmp(z->ent[i].name, want) == 0) {
            return i;
        }
    }
    return -1;
}

static void ademo_npz_missing(AdemoZip* z, const char* key) {
    fprintf(stderr, "anchor npz missing %s (%d entries)\n", key, z->nent);
    for (int k = 0; k < z->nent; k++) {
        fprintf(stderr, "  %s\n", z->ent[k].name);
    }
    assert(0 && "anchor npz key");
}

static AdemoNpy ademo_npy(const unsigned char* p, int nbytes) {
    assert(nbytes >= 10);
    assert(p[0] == 0x93 && memcmp(p + 1, "NUMPY", 5) == 0);
    int major = p[6];
    int hlen = 0;
    int hoff = 10;
    if (major == 1) {
        hlen = ademo_u16(p + 8);
    } else if (major == 2) {
        hlen = (int)ademo_u32(p + 8);
        hoff = 12;
    } else {
        assert(0 && "anchor npy version");
    }
    assert(hlen > 0 && hlen < 1024);
    assert(hoff + hlen <= nbytes);
    char hdr[1024];
    memcpy(hdr, p + hoff, (size_t)hlen);
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
    AdemoNpy o;
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

static float* ademo_load_f32(AdemoZip* z, const char* key, int* rows,
        int* cols) {
    int i = ademo_zip_index(z, key);
    if (i < 0) {
        ademo_npz_missing(z, key);
    }
    AdemoNpy np = ademo_npy(z->buf + z->ent[i].data, z->ent[i].nbytes);
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
            size_t f = (size_t)row + (size_t)col * (size_t)np.rows;
            dst[(size_t)row * (size_t)np.cols + (size_t)col] = src[f];
        }
    }
    free(src);
    return dst;
}

static int ademo_load_i32(AdemoZip* z, const char* key) {
    int i = ademo_zip_index(z, key);
    if (i < 0) {
        ademo_npz_missing(z, key);
    }
    AdemoNpy np = ademo_npy(z->buf + z->ent[i].data, z->ent[i].nbytes);
    assert(np.kind == 'i' && np.n == 1);
    int v = 0;
    memcpy(&v, np.data, 4);
    return v;
}

static void ademo_finite(const float* x, int n, const char* key) {
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) {
            fprintf(stderr, "anchor %s not finite\n", key);
            assert(0 && "anchor weight");
        }
    }
}

static void ademo_linear(const float* x, const float* W, const float* b,
        float* y, int in, int out, int relu) {
    for (int j = 0; j < out; j++) {
        float v = b[j];
        for (int i = 0; i < in; i++) {
            v += x[i] * W[(size_t)i * (size_t)out + (size_t)j];
        }
        if (relu && v < 0.0f) {
            v *= 0.01f;
        }
        y[j] = v;
    }
}

// log_p is the log probability in log space, so a bin that underflows
// to probability 0 still has a finite log. Pass NULL to skip it.
static void ademo_softmax(const float* logits, float* p, float* log_p, int n) {
    float m = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > m) {
            m = logits[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        p[i] = expf(logits[i] - m);
        sum += p[i];
    }
    assert(sum > 0.0f);
    float log_sum = logf(sum);
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        p[i] *= inv;
        if (log_p) {
            log_p[i] = logits[i] - m - log_sum;
        }
    }
}

static int ademo_argmax(const float* p, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (p[i] > p[best]) {
            best = i;
        }
    }
    return best;
}

// cur follows the latest activation. The two scratch buffers swap.
static void ademo_forward_obs(const float* obs) {
    Ademo* a = &g_ademo;
    float* cur = a->cur;
    float* nxt = a->nxt;
    for (int j = 0; j < a->in_dim; j++) {
        float s = a->sig[j];
        if (s < 1.0e-6f) {
            s = 1.0f;
        }
        cur[j] = (obs[j] - a->mu[j]) / s;
    }
    int n = a->in_dim;
    for (int layer = 0; layer < a->n_layers; layer++) {
        int out = a->lout[layer];
        ademo_linear(cur, a->W[layer], a->b[layer], nxt, n, out, 1);
        float* swap = cur;
        cur = nxt;
        nxt = swap;
        n = out;
    }
    float la[NUM_ACCEL_BINS];
    float ls[NUM_STEER_BINS];
    ademo_linear(cur, a->Wa, a->ba, la, n, NUM_ACCEL_BINS, 0);
    ademo_linear(cur, a->Ws, a->bs, ls, n, NUM_STEER_BINS, 0);
    ademo_softmax(la, a->pa, a->log_pa, NUM_ACCEL_BINS);
    ademo_softmax(ls, a->ps, a->log_ps, NUM_STEER_BINS);
}

static float ademo_kl(const float* q, const float* p, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        if (q[i] <= 0.0f) {
            continue;
        }
        if (p[i] <= 0.0f) {
            return INFINITY;
        }
        s += q[i] * (logf(q[i]) - logf(p[i]));
    }
    return s;
}

// KL(q || p) with log probabilities kept in log space. Matches the
// trainer anchor term: a zero probability is skipped, and the log of
// the other side stays finite.
static float ademo_kl_log(const float* log_q, const float* q,
        const float* log_p, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        if (q[i] > 0.0f) {
            s += q[i] * (log_q[i] - log_p[i]);
        }
    }
    return s;
}

// Natural log. A zero bin adds nothing.
static float ademo_entropy(const float* p, int n) {
    float h = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = p[i];
        if (x <= 0.0f) {
            continue;
        }
        h -= x * logf(x);
    }
    return h;
}

static float ademo_w1(const float* a, const float* b, int n) {
    float ca = 0.0f;
    float cb = 0.0f;
    float s = 0.0f;
    for (int k = 0; k < n - 1; k++) {
        ca += a[k];
        cb += b[k];
        float d = ca - cb;
        if (d < 0.0f) {
            d = -d;
        }
        s += d;
    }
    return s;
}

static void ademo_selfcheck(void) {
    float obs[OBS_SIZE];
    memset(obs, 0, sizeof(obs));
    ademo_forward_obs(obs);
    float sa = 0.0f;
    float ss = 0.0f;
    for (int i = 0; i < NUM_ACCEL_BINS; i++) {
        assert(isfinite(g_ademo.pa[i]));
        sa += g_ademo.pa[i];
    }
    for (int i = 0; i < NUM_STEER_BINS; i++) {
        assert(isfinite(g_ademo.ps[i]));
        ss += g_ademo.ps[i];
    }
    assert(fabsf(sa - 1.0f) < 1.0e-4f);
    assert(fabsf(ss - 1.0f) < 1.0e-4f);
    float same = ademo_kl(g_ademo.pa, g_ademo.pa, NUM_ACCEL_BINS)
        + ademo_kl(g_ademo.ps, g_ademo.ps, NUM_STEER_BINS);
    float w0 = ademo_w1(g_ademo.pa, g_ademo.pa, NUM_ACCEL_BINS)
        + ademo_w1(g_ademo.ps, g_ademo.ps, NUM_STEER_BINS);
    assert(fabsf(same) < 1.0e-4f);
    assert(fabsf(w0) < 1.0e-4f);
    float ha = ademo_entropy(g_ademo.pa, NUM_ACCEL_BINS);
    float hs = ademo_entropy(g_ademo.ps, NUM_STEER_BINS);
    assert(ha >= -1.0e-4f);
    assert(hs >= -1.0e-4f);
    assert(ha <= logf((float)NUM_ACCEL_BINS) + 1.0e-3f);
    assert(hs <= logf((float)NUM_STEER_BINS) + 1.0e-3f);
    // Same fixture as anchor_distance_selfcheck: logits [1,2,3] vs
    // [0,0,0] and [0,1,0] vs [2,0,1]. q is the anchor, p is the policy.
    float pol0[3] = {1.0f, 2.0f, 3.0f};
    float pol1[3] = {0.0f, 1.0f, 0.0f};
    float anc0[3] = {0.0f, 0.0f, 0.0f};
    float anc1[3] = {2.0f, 0.0f, 1.0f};
    float p0[3], p1[3], q0[3], q1[3];
    float lp0[3], lp1[3], lq0[3], lq1[3];
    ademo_softmax(pol0, p0, lp0, 3);
    ademo_softmax(pol1, p1, lp1, 3);
    ademo_softmax(anc0, q0, lq0, 3);
    ademo_softmax(anc1, q1, lq1, 3);
    float kl = ademo_kl_log(lq0, q0, lp0, 3) + ademo_kl_log(lq1, q1, lp1, 3);
    float rkl = ademo_kl_log(lp0, p0, lq0, 3) + ademo_kl_log(lp1, p1, lq1, 3);
    float w = ademo_w1(q0, p0, 3) + ademo_w1(q1, p1, 3);
    if (fabsf(kl - 0.9380122346980022f) > 1.0e-4f
            || fabsf(rkl - 1.0626701692550728f) > 1.0e-4f
            || fabsf(w - 1.06129669419989f) > 1.0e-4f) {
        printf("anchor distance fixture kl %g rkl %g w1 %g\n", kl, rkl, w);
        assert(0 && "anchor distance fixture");
    }
}

// One new observation input. The inserted row is zero, so the old
// forward is unchanged. mu at that slot is 0 and sig is 1.
static float* ademo_insert_row(float* W, int rows, int cols, int at) {
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

static float* ademo_insert_stat(float* x, int n, int at, float fill) {
    float* grown = (float*)malloc((size_t)(n + 1) * sizeof(float));
    assert(grown);
    assert(at >= 0 && at < n);
    memcpy(grown, x, (size_t)at * sizeof(float));
    grown[at] = fill;
    memcpy(grown + at + 1, x + at, (size_t)(n - at) * sizeof(float));
    free(x);
    return grown;
}

static void ademo_open(const char* path) {
    Ademo* a = &g_ademo;
    memset(a, 0, sizeof(*a));
    a->act_a = ACCEL_COAST_BIN;
    a->act_s = STEER_COAST_BIN;
    AdemoZip zip;
    memset(&zip, 0, sizeof(zip));
    ademo_zip_open(&zip, path);

    int n_layers = 0;
    int grew_obs = 0;
    while (n_layers < ADEMO_LAYERS) {
        char wk[16];
        char bk[16];
        snprintf(wk, sizeof(wk), "W%d", n_layers);
        snprintf(bk, sizeof(bk), "b%d", n_layers);
        if (ademo_zip_index(&zip, wk) < 0) {
            break;
        }
        int wr = 0;
        int wc = 0;
        int br = 0;
        int bc = 0;
        a->W[n_layers] = ademo_load_f32(&zip, wk, &wr, &wc);
        a->b[n_layers] = ademo_load_f32(&zip, bk, &br, &bc);
        assert(wc > 0 && bc == 0 && br == wc);
        if (n_layers == 0 && wr + 1 == OBS_SIZE) {
            int at = EGO_TARGET_SPEED;
            a->W[n_layers] = ademo_insert_row(a->W[n_layers], wr, wc, at);
            wr = OBS_SIZE;
            grew_obs = 1;
        }
        if (n_layers == 0) {
            assert(wr == OBS_SIZE);
        } else {
            assert(wr == a->lout[n_layers - 1]);
        }
        a->lin[n_layers] = wr;
        a->lout[n_layers] = wc;
        ademo_finite(a->W[n_layers], wr * wc, wk);
        ademo_finite(a->b[n_layers], br, bk);
        n_layers++;
    }
    assert(n_layers > 0 && "anchor npz has no hidden layers");
    if (n_layers == ADEMO_LAYERS) {
        char wk[16];
        snprintf(wk, sizeof(wk), "W%d", n_layers);
        assert(ademo_zip_index(&zip, wk) < 0 && "anchor too many layers");
    }
    int n_accel = ademo_load_i32(&zip, "n_accel");
    int n_steer = ademo_load_i32(&zip, "n_steer");
    assert(n_accel == NUM_ACCEL_BINS && n_steer == NUM_STEER_BINS);
    int last = a->lout[n_layers - 1];
    int ar = 0;
    int ac = 0;
    int sr = 0;
    int sc = 0;
    int bar = 0;
    int bac = 0;
    int bsr = 0;
    int bsc = 0;
    a->Wa = ademo_load_f32(&zip, "Wa", &ar, &ac);
    a->ba = ademo_load_f32(&zip, "ba", &bar, &bac);
    a->Ws = ademo_load_f32(&zip, "Ws", &sr, &sc);
    a->bs = ademo_load_f32(&zip, "bs", &bsr, &bsc);
    assert(ar == last && ac == n_accel && bac == 0 && bar == n_accel);
    assert(sr == last && sc == n_steer && bsc == 0 && bsr == n_steer);
    int mr = 0;
    int mc = 0;
    int sgr = 0;
    int sgc = 0;
    a->mu = ademo_load_f32(&zip, "mu", &mr, &mc);
    a->sig = ademo_load_f32(&zip, "sig", &sgr, &sgc);
    if (grew_obs) {
        int at = EGO_TARGET_SPEED;
        assert(mr + 1 == OBS_SIZE && sgr + 1 == OBS_SIZE);
        a->mu = ademo_insert_stat(a->mu, mr, at, 0.0f);
        a->sig = ademo_insert_stat(a->sig, sgr, at, 1.0f);
        mr = OBS_SIZE;
        sgr = OBS_SIZE;
        printf("anchor inserted a zero-weight target speed at index %d\n", at);
    }
    assert(mr == OBS_SIZE && mc == 0);
    assert(sgr == OBS_SIZE && sgc == 0);
    ademo_finite(a->Wa, ar * ac, "Wa");
    ademo_finite(a->ba, bar, "ba");
    ademo_finite(a->Ws, sr * sc, "Ws");
    ademo_finite(a->bs, bsr, "bs");
    ademo_finite(a->mu, mr, "mu");
    ademo_finite(a->sig, sgr, "sig");
    free(zip.buf);
    zip.buf = NULL;

    int cap = OBS_SIZE;
    for (int i = 0; i < n_layers; i++) {
        if (a->lout[i] > cap) {
            cap = a->lout[i];
        }
    }
    a->n_layers = n_layers;
    a->in_dim = OBS_SIZE;
    a->cap = cap;
    a->cur = (float*)malloc((size_t)cap * sizeof(float));
    a->nxt = (float*)malloc((size_t)cap * sizeof(float));
    assert(a->cur && a->nxt);
    ademo_selfcheck();

    int same = 1;
    for (int i = 1; i < n_layers; i++) {
        if (a->lout[i] != a->lout[0]) {
            same = 0;
        }
    }
    if (same) {
        printf("anchor %s  %d x %d\n", path, n_layers, a->lout[0]);
    } else {
        printf("anchor %s  widths", path);
        for (int i = 0; i < n_layers; i++) {
            printf(" %d", a->lout[i]);
        }
        printf("\n");
    }
    fflush(stdout);
}

// The caller owns net and frees it after ademo_close. Batch size is 1.
static void ademo_link(PufferNet* net) {
    g_ademo.net = net;
    if (!net) {
        return;
    }
    assert(!net->is_continuous);
    assert(net->num_agents == 1);
    assert(net->num_actions == 2);
    assert(net->multidiscrete);
    assert(net->multidiscrete->logit_sizes[0] == NUM_ACCEL_BINS);
    assert(net->multidiscrete->logit_sizes[1] == NUM_STEER_BINS);
    assert(net->decoder->output_dim == NUM_ACCEL_BINS + NUM_STEER_BINS + 1);
    int n = net->mingru->num_layers * net->mingru->batch_size
        * net->mingru->hidden_size;
    assert(n > 0);
    g_ademo.gru_n = n;
    g_ademo.gru_save = (float*)calloc((size_t)n, sizeof(float));
    assert(g_ademo.gru_save);
}

static void ademo_close(void) {
    Ademo* a = &g_ademo;
    for (int i = 0; i < a->n_layers; i++) {
        free(a->W[i]);
        free(a->b[i]);
        a->W[i] = NULL;
        a->b[i] = NULL;
    }
    free(a->Wa);
    free(a->ba);
    free(a->Ws);
    free(a->bs);
    free(a->mu);
    free(a->sig);
    free(a->cur);
    free(a->nxt);
    free(a->gru_save);
    memset(a, 0, sizeof(*a));
    if (g_math_ready && IsWindowReady()) {
        UnloadFont(g_math_font);
        g_math_ready = 0;
    }
}

static int ademo_town_scene(int which) {
    return which >= 0 && which < ADEMO_NSCENE
        && g_scenes[which].name == NULL;
}

static int ademo_town(void) {
    return ademo_town_scene(g_scene_i);
}

static void ademo_forward(Drive* env) {
    if (g_logs || ademo_town()) {
        env->selected_agent_idx = g_sdc;
    }
    int idx = env->selected_agent_idx;
    if (idx < 0 || idx >= env->active_agent_count) {
        idx = 0;
    }
    float* obs = env->agents[idx].observations;
    assert(obs);
    ademo_forward_obs(obs);
}

static void ademo_policy_zero(void) {
    PufferNet* net = g_ademo.net;
    if (!net || !net->mingru || !net->mingru->state || g_ademo.gru_n <= 0) {
        return;
    }
    memset(net->mingru->state, 0, (size_t)g_ademo.gru_n * sizeof(float));
}

// commit writes the MinGRU state. A redraw while paused restores it,
// so the same observation is not applied twice.
static void ademo_policy_dist(const float* obs, int commit) {
    PufferNet* net = g_ademo.net;
    assert(net && obs);
    float* state = net->mingru->state;
    size_t nbytes = (size_t)g_ademo.gru_n * sizeof(float);
    if (!commit) {
        memcpy(g_ademo.gru_save, state, nbytes);
    }
    float term = 0.0f;
    mingru_zero_term(net->mingru, &term);
    puf_encode(net, (float*)obs);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    float* logits = net->decoder->output;
    ademo_softmax(logits, g_ademo.rla, g_ademo.log_rla, NUM_ACCEL_BINS);
    ademo_softmax(logits + NUM_ACCEL_BINS, g_ademo.rls, g_ademo.log_rls,
        NUM_STEER_BINS);
    if (!commit) {
        memcpy(state, g_ademo.gru_save, nbytes);
    }
}

static void ademo_actions(Drive* env) {
    int n = env->active_agent_count;
    for (int i = 0; i < n; i++) {
        float* atn = env->agents[i].actions;
        if (!atn) {
            continue;
        }
        atn[0] = ACCEL_COAST_BIN;
        atn[1] = STEER_COAST_BIN;
    }
    if (g_logs || ademo_town()) {
        env->selected_agent_idx = g_sdc;
    }
    int idx = env->selected_agent_idx;
    if (!g_logs && !ademo_town() && (idx < 0 || idx >= n)) {
        idx = 0;
        env->selected_agent_idx = 0;
    }
    float* atn = env->agents[idx].actions;
    assert(atn);
    if (g_ademo.who == ADEMO_ANCHOR) {
        atn[0] = (float)ademo_argmax(g_ademo.pa, NUM_ACCEL_BINS);
        atn[1] = (float)ademo_argmax(g_ademo.ps, NUM_STEER_BINS);
    } else if (ademo_policy_on()) {
        float* obs = env->agents[idx].observations;
        assert(obs);
        ademo_policy_dist(obs, g_ademo.commit);
        atn[0] = (float)ademo_argmax(g_ademo.rla, NUM_ACCEL_BINS);
        atn[1] = (float)ademo_argmax(g_ademo.rls, NUM_STEER_BINS);
    } else if (IsWindowReady()) {
        int accel_kick = (NUM_ACCEL_BINS - 1) / 3;
        int steer_kick = (NUM_STEER_BINS - 1) / 3;
        int accel = ACCEL_COAST_BIN;
        int steer = STEER_COAST_BIN;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
            accel += accel_kick;
        }
        if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
            accel -= accel_kick;
        }
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            steer += steer_kick;
        }
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            steer -= steer_kick;
        }
        if (accel < 0) {
            accel = 0;
        }
        if (accel > NUM_ACCEL_BINS - 1) {
            accel = NUM_ACCEL_BINS - 1;
        }
        if (steer < 0) {
            steer = 0;
        }
        if (steer > NUM_STEER_BINS - 1) {
            steer = NUM_STEER_BINS - 1;
        }
        atn[0] = (float)accel;
        atn[1] = (float)steer;
        if (!g_logs && !ademo_town() && IsKeyPressed(KEY_TAB) && n > 0) {
            env->selected_agent_idx = (idx + 1) % n;
        }
    }
    g_ademo.act_a = (int)atn[0];
    g_ademo.act_s = (int)atn[1];
}

static void ademo_distances(void) {
    g_ademo.ent_a = ademo_entropy(g_ademo.pa, NUM_ACCEL_BINS);
    g_ademo.ent_s = ademo_entropy(g_ademo.ps, NUM_STEER_BINS);
    if (!ademo_policy_on()) {
        return;
    }
    // q is the anchor, p is the policy. Same reduction as the trainer:
    // KL(anchor || policy), KL(policy || anchor), W1, summed over heads.
    g_ademo.kl = ademo_kl_log(g_ademo.log_pa, g_ademo.pa, g_ademo.log_rla,
            NUM_ACCEL_BINS)
        + ademo_kl_log(g_ademo.log_ps, g_ademo.ps, g_ademo.log_rls,
            NUM_STEER_BINS);
    g_ademo.rkl = ademo_kl_log(g_ademo.log_rla, g_ademo.rla, g_ademo.log_pa,
            NUM_ACCEL_BINS)
        + ademo_kl_log(g_ademo.log_rls, g_ademo.rls, g_ademo.log_ps,
            NUM_STEER_BINS);
    g_ademo.w1 = ademo_w1(g_ademo.pa, g_ademo.rla, NUM_ACCEL_BINS)
        + ademo_w1(g_ademo.ps, g_ademo.rls, NUM_STEER_BINS);
}

static void ademo_math_open(void) {
    if (g_math_ready || !IsWindowReady()) {
        return;
    }
    int cps[160];
    int n = 0;
    for (int c = 32; c < 127; c++) {
        cps[n++] = c;
    }
    cps[n++] = 0x03C0;
    cps[n++] = 0x03C4;
    cps[n++] = 0x00B7;
    cps[n++] = 0x2081;
    cps[n++] = 0x2016;
    const char* paths[] = {
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    };
    for (int i = 0; i < 2; i++) {
        if (!FileExists(paths[i])) {
            continue;
        }
        Font font = LoadFontEx(paths[i], 64, cps, n);
        if (font.glyphCount <= 0) {
            continue;
        }
        SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
        g_math_font = font;
        g_math_ready = 1;
        return;
    }
}

static float ademo_adv(int cp, float size) {
    int i = GetGlyphIndex(g_math_font, cp);
    float adv = (float)g_math_font.glyphs[i].advanceX;
    float base = (float)g_math_font.baseSize;
    if (base < 1.0f) {
        base = 1.0f;
    }
    if (adv <= 0.0f) {
        adv = g_math_font.recs[i].width;
    }
    return adv * (size / base);
}

static float ademo_one(int cp, float x, float y, float size, Color color) {
    DrawTextCodepoint(g_math_font, cp, (Vector2){x, y}, size, color);
    return x + ademo_adv(cp, size);
}

static float ademo_runes(const int* cps, int n, float x, float y,
        float size, Color color) {
    for (int i = 0; i < n; i++) {
        x = ademo_one(cps[i], x, y, size, color);
    }
    return x;
}

// o_t). Liberation Sans has no subscript t, so the t is drawn
// smaller, lowered, and given a gap so it does not sit in the paren.
static float ademo_ot(float x, float y, float size, Color color) {
    x = ademo_one('o', x, y, size, color);
    float sub = 12.0f;
    if (size * 0.75f > sub) {
        sub = size * 0.75f;
    }
    x += 1.0f;
    ademo_one('t', x, y + size * 0.32f, sub, color);
    x += ademo_adv('t', sub) + 2.0f;
    return ademo_one(')', x, y, size, color);
}

// τ(· | o_t)
static float ademo_tau_ot(float x, float y, float size, Color color) {
    const int head[] = {
        0x03C4, '(', ' ', 0x00B7, ' ', '|', ' '
    };
    x = ademo_runes(head, 7, x, y, size, color);
    return ademo_ot(x, y, size, color);
}

// Left column. Width stays put when the window grows.
// Bar height uses the leftover room, with a cap.
static void ademo_layout(Drive* env, AdemoLayout* L) {
    float height = 880.0f;
    if (env->client) {
        height = env->client->height;
    }
    float x = 8.0f;
    float y = 8.0f;
    float col_w = 308.0f;
    float bottom = height - 8.0f;
    if (bottom < y + 420.0f) {
        bottom = y + 420.0f;
    }
    float pad = 10.0f;
    float dist = 16.0f;
    float step = 22.0f;
    float head = 14.0f;
    float axis = 48.0f;
    float btn_h = 26.0f;
    // Room under each histogram for the action-space ticks.
    float xrow = 20.0f;
    float inner_x = x + pad;
    float inner_w = col_w - pad * 2.0f;
    float plot_w = inner_w - axis;
    float yb = y + pad;
    L->title_y = yb;
    yb += dist + 4.0f;
    L->hint_y = yb;
    yb += head + 2.0f;
    L->cmd_y = yb;
    yb += head + 2.0f + head + 8.0f;
    L->button = (Rectangle){inner_x, yb, inner_w, btn_h};
    yb += btn_h + 6.0f;
    if (g_ademo.net) {
        L->policy = (Rectangle){inner_x, yb, inner_w, btn_h};
        yb += btn_h + 6.0f;
    } else {
        L->policy = (Rectangle){0.0f, 0.0f, 0.0f, 0.0f};
    }
    L->reset = (Rectangle){inner_x, yb, inner_w, btn_h};
    yb += btn_h + 10.0f;
    L->accel_y = yb;
    yb += head + 4.0f;
    int dist_rows = 1;
    float series_h = 0.0f;
    float series_extra = 0.0f;
    if (ademo_policy_on()) {
        dist_rows = 3;
        series_h = 88.0f;
        series_extra = 8.0f + head + 4.0f + series_h + 4.0f + head;
    }
    float fixed = xrow + head + 4.0f + xrow + head + 8.0f
        + step * (float)dist_rows + series_extra + pad;
    float room = bottom - yb - fixed;
    float bar_h = room * 0.5f;
    if (bar_h < 72.0f) {
        bar_h = 72.0f;
    }
    if (bar_h > 120.0f) {
        bar_h = 120.0f;
    }
    L->accel = (Rectangle){inner_x + axis, yb, plot_w, bar_h};
    yb += bar_h + xrow;
    L->steer_y = yb;
    yb += head + 4.0f;
    L->steer = (Rectangle){inner_x + axis, yb, plot_w, bar_h};
    yb += bar_h + xrow;
    L->legend_y = yb;
    yb += head + 8.0f;
    L->dist_y = yb;
    yb += step * (float)dist_rows;
    if (series_h > 0.0f) {
        yb += 8.0f;
        L->series_y = yb;
        yb += head + 4.0f;
        L->series = (Rectangle){
            inner_x + 40.0f, yb, inner_w - 40.0f, series_h
        };
        yb += series_h + 4.0f + head;
    }
    yb += pad;
    L->card = (Rectangle){x, y, col_w, yb - y};
    L->text_x = inner_x;
    L->text_w = inner_w;
    L->dist_size = dist;
    L->dist_step = step;
    L->head_size = head;
}

static void ademo_hist_clear(void);

// 1 when Reset was clicked. The caller resets after puf_sim_poll,
// which clears sim_hold_step at the start of the frame.
static int ademo_poll(Drive* env) {
    if (!IsWindowReady() || !IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        return 0;
    }
    AdemoLayout L;
    ademo_layout(env, &L);
    Vector2 mouse = GetMousePosition();
    if (CheckCollisionPointRec(mouse, L.button)) {
        if (g_ademo.who == ADEMO_ANCHOR) {
            g_ademo.who = ADEMO_YOU;
        } else {
            g_ademo.who = ADEMO_ANCHOR;
        }
        ademo_hist_clear();
        return 0;
    }
    if (g_ademo.net && CheckCollisionPointRec(mouse, L.policy)) {
        if (g_ademo.who == ADEMO_POLICY) {
            g_ademo.who = ADEMO_YOU;
        } else {
            g_ademo.who = ADEMO_POLICY;
            ademo_policy_zero();
        }
        ademo_hist_clear();
        return 0;
    }
    if (CheckCollisionPointRec(mouse, L.reset)) {
        return 1;
    }
    return 0;
}

// Height is the probability. p = 1 fills the axis. The mid line is 0.5.
// This plot is narrower than the 257-bin table. A fractional bar width
// does not rasterize, and a 1px floor runs past the frame. Each pixel
// shows the tallest bin whose center falls in it.
static void ademo_bars(Rectangle plot, const float* p, int n, int taken) {
    DrawRectangleRec(plot, WELL);
    float top = plot.y + 1.0f;
    float span = plot.height - 2.0f;
    float mid = top + span * 0.5f;
    DrawLineEx(
        (Vector2){plot.x + 1.0f, mid},
        (Vector2){plot.x + plot.width - 1.0f, mid},
        1.0f, GRID);
    int cols = (int)plot.width;
    if (cols < 1) {
        cols = 1;
    }
    if (cols > 512) {
        cols = 512;
    }
    float peak[512];
    for (int c = 0; c < cols; c++) {
        peak[c] = 0.0f;
    }
    for (int i = 0; i < n; i++) {
        int c = (int)(((float)i + 0.5f) * (float)cols / (float)n);
        if (c < 0) {
            c = 0;
        }
        if (c >= cols) {
            c = cols - 1;
        }
        if (p[i] > peak[c]) {
            peak[c] = p[i];
        }
    }
    for (int c = 0; c < cols; c++) {
        float h = peak[c] * span;
        if (h > span) {
            h = span;
        }
        if (h < 1.0f) {
            continue;
        }
        DrawRectangleRec((Rectangle){
            plot.x + (float)c, top + span - h, 1.0f, h
        }, SAGE);
    }
    if (taken >= 0 && taken < n) {
        int c = (int)(((float)taken + 0.5f) * (float)cols / (float)n);
        if (c < 0) {
            c = 0;
        }
        if (c >= cols) {
            c = cols - 1;
        }
        float bx = plot.x + (float)c - 1.0f;
        float right = plot.x + plot.width - 3.0f;
        if (bx < plot.x) {
            bx = plot.x;
        }
        if (bx > right) {
            bx = right;
        }
        DrawRectangleLinesEx(
            (Rectangle){bx, top, 3.0f, span}, 1.0f, CORAL);
    }
    DrawRectangleLinesEx(plot, 1.0f, LINE);
}

static float ademo_tw(Drive* env, const char* text, float size) {
    Vector2 v = MeasureTextEx(sim_font(env), text, size, 0.4f);
    return v.x;
}

static void ademo_yaxis(Drive* env, Rectangle plot) {
    float top = plot.y + 1.0f;
    float span = plot.height - 2.0f;
    float bot = top + span;
    float axis = plot.x - 1.0f;
    DrawLineEx((Vector2){axis, top}, (Vector2){axis, bot}, 1.0f, INK);
    const char* lab[] = {"1", "0.5", "0"};
    float frac[] = {1.0f, 0.5f, 0.0f};
    float size = 11.0f;
    for (int i = 0; i < 3; i++) {
        float y = bot - frac[i] * span;
        DrawLineEx(
            (Vector2){axis - 3.0f, y}, (Vector2){axis, y}, 1.0f, INK);
        float tw = ademo_tw(env, lab[i], size);
        float ty = y - size * 0.5f;
        if (i == 0) {
            ty = top;
        }
        if (i == 2) {
            ty = bot - size;
        }
        sim_sans(env, lab[i], axis - 5.0f - tw, ty, size, MUTED);
    }
    Font font = sim_font(env);
    const char* word = "probability";
    float wsz = 11.0f;
    Vector2 m = MeasureTextEx(font, word, wsz, 0.4f);
    if (m.x > span) {
        return;
    }
    float cx = plot.x - 36.0f;
    float cy = top + span * 0.5f;
    Vector2 at = {cx, cy};
    Vector2 origin = {m.x * 0.5f, m.y * 0.5f};
    DrawTextPro(font, word, at, origin, -90.0f, wsz, 0.4f, MUTED);
}

// Min, zero, and max command. Counts are odd and the bins are uniform,
// so the marks are the first bin, the coast bin, and the last. A whole
// command prints as an integer. ±1.5 would round to ±2 with %+.0f.
static void ademo_xaxis(Drive* env, Rectangle plot, int n,
        const float* values) {
    float slot = plot.width / (float)n;
    float size = 11.0f;
    float base = plot.y + plot.height;
    int marks[3];
    marks[0] = 0;
    marks[1] = (n - 1) / 2;
    marks[2] = n - 1;
    for (int k = 0; k < 3; k++) {
        int i = marks[k];
        float cx = plot.x + ((float)i + 0.5f) * slot;
        char buf[16];
        float v = values[i];
        float mag = fabsf(v);
        float rounded = floorf(mag + 0.5f);
        if (mag < 0.05f) {
            snprintf(buf, sizeof(buf), "0");
        } else if (fabsf(mag - rounded) < 0.05f) {
            snprintf(buf, sizeof(buf), "%+.0f", v);
        } else {
            snprintf(buf, sizeof(buf), "%+.1f", v);
        }
        float tw = ademo_tw(env, buf, size);
        float tx = cx - tw * 0.5f;
        if (k == 0 && tx < plot.x) {
            tx = plot.x;
        }
        if (k == 2 && tx + tw > plot.x + plot.width) {
            tx = plot.x + plot.width - tw;
        }
        DrawLineEx(
            (Vector2){cx, base},
            (Vector2){cx, base + 3.0f},
            1.0f, INK);
        sim_sans(env, buf, tx, base + 4.0f, size, MUTED);
    }
}

static void ademo_num(char* dst, int n, float v) {
    if (!isfinite(v)) {
        snprintf(dst, (size_t)n, "inf");
        return;
    }
    snprintf(dst, (size_t)n, "%.3f", v);
}

static void ademo_value(Drive* env, AdemoLayout* L, int row,
        const char* value, int bad) {
    float y = L->dist_y + (float)row * L->dist_step;
    float size = L->dist_size;
    Color col = INK;
    if (bad) {
        col = CRIMSON;
    }
    float tw = ademo_tw(env, value, size);
    sim_sans(env, value, L->text_x + L->text_w - tw, y, size, col);
}

static void ademo_button(Drive* env, Rectangle r, const char* label,
        int on, int hot) {
    Color fill = WELL;
    Color edge = LINE;
    if (on) {
        fill = SAGE;
        edge = SAGE;
    } else if (hot) {
        fill = HOVER;
    }
    if (hot) {
        edge = INK;
    }
    DrawRectangleRec(r, fill);
    DrawRectangleLinesEx(r, 1.5f, edge);
    float size = 15.0f;
    float tw = ademo_tw(env, label, size);
    if (tw > r.width - 8.0f) {
        size = 13.0f;
        tw = ademo_tw(env, label, size);
    }
    float tx = r.x + (r.width - tw) * 0.5f;
    float ty = r.y + (r.height - size) * 0.5f - 1.0f;
    sim_sans(env, label, tx, ty, size, INK);
}

static void ademo_legend(Drive* env, AdemoLayout* L) {
    float y = L->legend_y;
    float x = L->text_x;
    float s = L->head_size;
    DrawRectangleRec((Rectangle){x, y + 2.0f, 11.0f, 9.0f}, SAGE);
    sim_sans(env, "anchor probability", x + 16.0f, y, s, MUTED);
    float tw = ademo_tw(env, "anchor probability", s);
    float x2 = x + 16.0f + tw + 12.0f;
    DrawRectangleLinesEx(
        (Rectangle){x2, y + 1.0f, 9.0f, 11.0f}, 1.0f, CORAL);
    sim_sans(env, "selected bin", x2 + 14.0f, y, s, MUTED);
}

// KL(τ ‖ π) or KL(π ‖ τ). π is the linked policy.
static void ademo_kl_row(AdemoLayout* L, int row, int reverse, Color col) {
    float y = L->dist_y + (float)row * L->dist_step;
    float size = L->dist_size;
    float x = L->text_x;
    const int kl[] = {'K', 'L', '('};
    const int mid[] = {' ', 0x2016, ' '};
    x = ademo_runes(kl, 3, x, y, size, col);
    if (reverse) {
        x = ademo_one(0x03C0, x, y, size, col);
        x = ademo_runes(mid, 3, x, y, size, col);
        x = ademo_one(0x03C4, x, y, size, col);
    } else {
        x = ademo_one(0x03C4, x, y, size, col);
        x = ademo_runes(mid, 3, x, y, size, col);
        x = ademo_one(0x03C0, x, y, size, col);
    }
    ademo_one(')', x, y, size, col);
}

// H(τ). Sum of the two heads, same reduction as the KL rows.
static void ademo_entropy_label(AdemoLayout* L, Color col) {
    float y = L->dist_y;
    float size = L->dist_size;
    float x = ademo_one('H', L->text_x, y, size, col);
    const int tail[] = {'(', 0x03C4, ')'};
    ademo_runes(tail, 3, x, y, size, col);
}

static void ademo_w1_label(AdemoLayout* L, int row, Color col) {
    float y = L->dist_y + (float)row * L->dist_step;
    float size = L->dist_size;
    float x = ademo_one('W', L->text_x, y, size, col);
    ademo_one(0x2081, x, y, size, col);
}

// One sample per sim step. Pause does not fill the trace with copies.
static void ademo_hist_clear(void) {
    g_ademo.hist_n = 0;
    g_ademo.hist_at = 0;
    g_ademo.hist_step = -1;
}

// hist_at is the slot of the current step. A new step advances it.
// The same step is overwritten so the trace matches the readout.
static void ademo_hist_note(Drive* env) {
    if (!ademo_policy_on()) {
        return;
    }
    int fresh = env->timestep != g_ademo.hist_step;
    if (fresh) {
        if (g_ademo.hist_n > 0) {
            g_ademo.hist_at++;
            if (g_ademo.hist_at >= ADEMO_HIST) {
                g_ademo.hist_at = 0;
            }
        }
        if (g_ademo.hist_n < ADEMO_HIST) {
            g_ademo.hist_n++;
        }
        g_ademo.hist_step = env->timestep;
    }
    g_ademo.hist_kl[g_ademo.hist_at] = g_ademo.kl;
    g_ademo.hist_rkl[g_ademo.hist_at] = g_ademo.rkl;
    g_ademo.hist_w1[g_ademo.hist_at] = g_ademo.w1;
}

static float ademo_hist_get(const float* h, int age) {
    int start = g_ademo.hist_at - (g_ademo.hist_n - 1);
    if (start < 0) {
        start += ADEMO_HIST;
    }
    int i = start + age;
    if (i >= ADEMO_HIST) {
        i -= ADEMO_HIST;
    }
    return h[i];
}

static void ademo_trace(Rectangle plot, const float* h, float maxv,
        Color col) {
    float top = plot.y + 3.0f;
    float bot = plot.y + plot.height - 3.0f;
    float span = bot - top;
    if (span < 1.0f) {
        span = 1.0f;
    }
    float x0 = plot.x + 2.0f;
    float dx = (plot.width - 4.0f) / (float)(ADEMO_HIST - 1);
    int n = g_ademo.hist_n;
    int slot0 = ADEMO_HIST - n;
    int have = 0;
    Vector2 prev = {0.0f, 0.0f};
    for (int age = 0; age < n; age++) {
        float v = ademo_hist_get(h, age);
        float x = x0 + (float)(slot0 + age) * dx;
        float y = bot;
        int ok = isfinite(v);
        if (ok) {
            if (v < 0.0f) {
                v = 0.0f;
            }
            y = bot - (v / maxv) * span;
        }
        Vector2 cur = {x, y};
        if (have && ok) {
            DrawLineEx(prev, cur, 2.0f, col);
        }
        have = ok;
        prev = cur;
        if (ok && age == n - 1) {
            DrawCircleV(cur, 2.5f, col);
        }
    }
}

static void ademo_series_key(Drive* env, float* x, float y,
        Color col, const char* name) {
    DrawRectangleRec((Rectangle){*x, y + 4.0f, 14.0f, 3.0f}, col);
    *x += 18.0f;
    sim_sans(env, name, *x, y, 13.0f, col);
    *x += ademo_tw(env, name, 13.0f) + 12.0f;
}

static void ademo_series(Drive* env, AdemoLayout* L) {
    sim_sans(env, "last 20 steps", L->text_x, L->series_y,
        L->head_size, MUTED);
    Rectangle plot = L->series;
    DrawRectangleRec(plot, WELL);
    float top = plot.y + 3.0f;
    float bot = plot.y + plot.height - 3.0f;
    float mid = (top + bot) * 0.5f;
    DrawLineEx(
        (Vector2){plot.x + 1.0f, mid},
        (Vector2){plot.x + plot.width - 1.0f, mid},
        1.0f, GRID);
    DrawRectangleLinesEx(plot, 1.0f, LINE);
    float maxv = 0.0f;
    for (int age = 0; age < g_ademo.hist_n; age++) {
        float vs[3];
        vs[0] = ademo_hist_get(g_ademo.hist_kl, age);
        vs[1] = ademo_hist_get(g_ademo.hist_rkl, age);
        vs[2] = ademo_hist_get(g_ademo.hist_w1, age);
        for (int k = 0; k < 3; k++) {
            if (!isfinite(vs[k])) {
                continue;
            }
            if (vs[k] > maxv) {
                maxv = vs[k];
            }
        }
    }
    if (maxv < 1.0e-3f) {
        maxv = 1.0f;
    }
    char hi[16];
    snprintf(hi, sizeof(hi), "%.1f", maxv);
    float size = 11.0f;
    float tw = ademo_tw(env, hi, size);
    sim_sans(env, hi, plot.x - 4.0f - tw, top - 1.0f, size, MUTED);
    float ztw = ademo_tw(env, "0", size);
    sim_sans(env, "0", plot.x - 4.0f - ztw, bot - size, size, MUTED);
    ademo_trace(plot, g_ademo.hist_kl, maxv, STEEL);
    ademo_trace(plot, g_ademo.hist_rkl, maxv, CRIMSON);
    ademo_trace(plot, g_ademo.hist_w1, maxv, SAGE_INK);
    float x = L->text_x;
    float y = plot.y + plot.height + 4.0f;
    ademo_series_key(env, &x, y, STEEL, "KL");
    ademo_series_key(env, &x, y, CRIMSON, "reverse KL");
    ademo_series_key(env, &x, y, SAGE_INK, "W1");
}

// Command in the air: the bin the driver or the anchor is holding.
static void ademo_cmd(Drive* env, AdemoLayout* L) {
    int ia = g_ademo.act_a;
    int is_ = g_ademo.act_s;
    if (ia < 0) {
        ia = 0;
    }
    if (ia >= NUM_ACCEL_BINS) {
        ia = NUM_ACCEL_BINS - 1;
    }
    if (is_ < 0) {
        is_ = 0;
    }
    if (is_ >= NUM_STEER_BINS) {
        is_ = NUM_STEER_BINS - 1;
    }
    char buf[40];
    float size = L->head_size;
    float y = L->cmd_y;
    snprintf(buf, sizeof(buf), "accel  %+.1f m/s^2",
        ACCELERATION_VALUES[ia]);
    sim_sans(env, buf, L->text_x, y, size, INK);
    snprintf(buf, sizeof(buf), "steer  %+.3f rad",
        STEERING_VALUES[is_]);
    sim_sans(env, buf, L->text_x, y + size + 2.0f, size, INK);
}

static void ademo_nbins(Drive* env, AdemoLayout* L, float y, int n) {
    char buf[16];
    snprintf(buf, sizeof(buf), "n = %d", n);
    float size = L->head_size;
    float tw = ademo_tw(env, buf, size);
    sim_sans(env, buf, L->text_x + L->text_w - tw, y, size, MUTED);
}

static void ademo_hud(Drive* env) {
    ademo_math_open();
    AdemoLayout L;
    ademo_layout(env, &L);
    DrawRectangleRec(L.card, CARD);
    DrawRectangleLinesEx(L.card, 1.0f, LINE);
    if (g_math_ready) {
        ademo_tau_ot(L.text_x, L.title_y, L.dist_size, INK);
    } else {
        sim_sans(env, "tau(. | o_t)", L.text_x, L.title_y, L.dist_size, INK);
    }
    const char* hint = "You drive the SDC    WASD or arrows";
    const char* button = "Let anchor take control";
    if (g_ademo.who == ADEMO_ANCHOR) {
        hint = "Anchor drives the SDC    greedy bins";
        button = "You drive";
    } else if (g_ademo.who == ADEMO_POLICY) {
        hint = "Policy drives the SDC    greedy bins";
    }
    sim_sans(env, hint, L.text_x, L.hint_y, L.head_size, MUTED);
    ademo_cmd(env, &L);
    Vector2 mouse = GetMousePosition();
    int hot = CheckCollisionPointRec(mouse, L.button);
    int hot_reset = CheckCollisionPointRec(mouse, L.reset);
    ademo_button(env, L.button, button, g_ademo.who == ADEMO_ANCHOR, hot);
    if (g_ademo.net) {
        const char* plab = "Let policy take control";
        if (g_ademo.who == ADEMO_POLICY) {
            plab = "You drive";
        }
        int hot_policy = CheckCollisionPointRec(mouse, L.policy);
        ademo_button(env, L.policy, plab, g_ademo.who == ADEMO_POLICY,
            hot_policy);
    }
    ademo_button(env, L.reset, "Reset", 0, hot_reset);
    sim_sans(env, "Acceleration (m/s^2)", L.text_x, L.accel_y,
        L.head_size, MUTED);
    ademo_nbins(env, &L, L.accel_y, NUM_ACCEL_BINS);
    sim_sans(env, "Steering (rad)", L.text_x, L.steer_y,
        L.head_size, MUTED);
    ademo_nbins(env, &L, L.steer_y, NUM_STEER_BINS);
    ademo_bars(L.accel, g_ademo.pa, NUM_ACCEL_BINS, g_ademo.act_a);
    ademo_yaxis(env, L.accel);
    ademo_xaxis(env, L.accel, NUM_ACCEL_BINS, ACCELERATION_VALUES);
    ademo_bars(L.steer, g_ademo.ps, NUM_STEER_BINS, g_ademo.act_s);
    ademo_yaxis(env, L.steer);
    ademo_xaxis(env, L.steer, NUM_STEER_BINS, STEERING_VALUES);
    ademo_legend(env, &L);
    DrawLineEx(
        (Vector2){L.text_x, L.dist_y - 8.0f},
        (Vector2){L.text_x + L.text_w, L.dist_y - 8.0f},
        1.0f, LINE);
    char buf[64];
    if (!ademo_policy_on()) {
        float ent = g_ademo.ent_a + g_ademo.ent_s;
        snprintf(buf, sizeof(buf), "%.3f  a %.3f  s %.3f",
            ent, g_ademo.ent_a, g_ademo.ent_s);
        ademo_value(env, &L, 0, buf, !isfinite(ent));
        if (!g_math_ready) {
            sim_sans(env, "H(tau)", L.text_x, L.dist_y, L.dist_size,
                isfinite(ent) ? INK : CRIMSON);
        } else {
            ademo_entropy_label(&L, isfinite(ent) ? INK : CRIMSON);
        }
    } else {
        ademo_num(buf, 64, g_ademo.kl);
        ademo_value(env, &L, 0, buf, !isfinite(g_ademo.kl));
        ademo_num(buf, 64, g_ademo.rkl);
        ademo_value(env, &L, 1, buf, !isfinite(g_ademo.rkl));
        ademo_num(buf, 64, g_ademo.w1);
        ademo_value(env, &L, 2, buf, !isfinite(g_ademo.w1));
        if (!g_math_ready) {
            const char* names[] = {
                "KL(tau || pi)",
                "KL(pi || tau)",
                "W1",
            };
            float vals[] = {g_ademo.kl, g_ademo.rkl, g_ademo.w1};
            for (int i = 0; i < 3; i++) {
                float y = L.dist_y + (float)i * L.dist_step;
                Color col = INK;
                if (!isfinite(vals[i])) {
                    col = CRIMSON;
                }
                sim_sans(env, names[i], L.text_x, y, L.dist_size, col);
            }
        } else {
            ademo_kl_row(&L, 0, 0, isfinite(g_ademo.kl) ? INK : CRIMSON);
            ademo_kl_row(&L, 1, 1, isfinite(g_ademo.rkl) ? INK : CRIMSON);
            ademo_w1_label(&L, 2, isfinite(g_ademo.w1) ? INK : CRIMSON);
        }
        ademo_series(env, &L);
    }
}

static void ademo_sync(Drive* env) {
    if (!env->client || !IsWindowReady()) {
        return;
    }
    int sw = GetScreenWidth();
    int sh = GetScreenHeight();
    if (sw < 100 || sh < 100) {
        return;
    }
    env->client->width = (float)sw;
    env->client->height = (float)sh;
}

static int ademo_hot(Drive* env) {
    AdemoLayout L;
    ademo_layout(env, &L);
    return CheckCollisionPointRec(GetMousePosition(), L.card);
}

static void ademo_title(void) {
    char buf[64];
    if (ademo_town()) {
        if (g_ademo.net) {
            snprintf(buf, sizeof(buf), "Anchor  Town01  policy");
        } else {
            snprintf(buf, sizeof(buf), "Anchor  Town01");
        }
    } else {
        int map_id = 0;
        if (g_scene_i >= 0 && g_scene_i < ADEMO_NSCENE) {
            map_id = g_scenes[g_scene_i].map_id;
        }
        if (g_ademo.net) {
            snprintf(buf, sizeof(buf), "Anchor  map %d  policy", map_id);
        } else {
            snprintf(buf, sizeof(buf), "Anchor  map %d", map_id);
        }
    }
    if (IsWindowReady()) {
        SetWindowTitle(buf);
    }
}

static void ademo_release(Drive* env) {
    free(g_logs);
    g_logs = NULL;
    g_log_n = 0;
    if (!env->map_name) {
        return;
    }
    puf_close(env);
    free(env->map_name);
    env->map_name = NULL;
}

static void ademo_bind(Drive* env) {
    if (!g_obs) {
        g_obs = (float*)calloc(OBS_SIZE, sizeof(float));
        g_act = (float*)calloc(
            (size_t)DRIVE_AGENT_MAX * NUM_ATNS, sizeof(float));
        g_rew = (float*)calloc(DRIVE_AGENT_MAX, sizeof(float));
        g_term = (float*)calloc(DRIVE_AGENT_MAX, sizeof(float));
    }
    assert(g_obs && g_act && g_rew && g_term);
    for (int i = 0; i < DRIVE_AGENT_MAX; i++) {
        env->agents[i].observations = NULL;
        env->agents[i].actions = NULL;
        env->agents[i].rewards = NULL;
        env->agents[i].terminals = NULL;
    }
    int n = env->num_agents;
    if (n > DRIVE_AGENT_MAX) {
        n = DRIVE_AGENT_MAX;
    }
    for (int i = 0; i < n; i++) {
        env->agents[i].actions = g_act + (size_t)i * NUM_ATNS;
        env->agents[i].rewards = g_rew + i;
        env->agents[i].terminals = g_term + i;
        env->agents[i].policy = 0;
        env->agents[i].actions[0] = ACCEL_COAST_BIN;
        env->agents[i].actions[1] = STEER_COAST_BIN;
    }
    if (g_sdc >= 0 && g_sdc < n) {
        env->agents[g_sdc].observations = g_obs;
    }
}

// Partners stay on the logged pose.
static void ademo_pin(Drive* env, int frame) {
    for (int i = 0; i < g_log_n; i++) {
        if (i == g_sdc) {
            continue;
        }
        offline_write_agent(&env->entities[i], &g_logs[i], frame);
        env->entities[i].goal_position_x = INVALID_POSITION;
        env->entities[i].goal_position_y = INVALID_POSITION;
    }
    env->bins_ready = 0;
}

static void ademo_place(Drive* env, int frame) {
    if (frame < 0) {
        frame = 0;
    }
    if (frame >= OFFLINE_TRAJ) {
        frame = OFFLINE_TRAJ - 1;
    }
    for (int i = 0; i < g_log_n; i++) {
        offline_write_agent(&env->entities[i], &g_logs[i], frame);
    }
    ademo_pin(env, frame);
    env->selected_agent_idx = g_sdc;
    env->timestep = frame;
    env->bins_ready = 0;
    compute_observations(env);
}

static void ademo_replay_step(Drive* env) {
    int frame = env->timestep;
    int next = frame + 1;
    if (next >= OFFLINE_TRAJ) {
        next = OFFLINE_TRAJ - 1;
    }
    Entity* ego = &env->entities[g_sdc];
    if (ego->x != INVALID_POSITION && ego->y != INVALID_POSITION) {
        move_dynamics(env, g_sdc);
    }
    ademo_pin(env, next);
    collision_check(env, g_sdc);
    env->timestep = next;
    compute_observations(env);
}

// Town01 uses the trainer fleet. Goals are the forward-lane sample
// from initialize_env, then again on arrival.
static void ademo_load_town(Drive* env, Dict* cfg, int which) {
    dict_set_str(cfg, "data_dir", "data/mixed");
    dict_set(cfg, "map_id", g_scenes[which].map_id);
    int n = g_fleet;
    if (n < 1) {
        n = 1;
    }
    if (n > DRIVE_AGENT_MAX) {
        n = DRIVE_AGENT_MAX;
    }
    dict_set(cfg, "num_agents", n);
    g_logs = NULL;
    g_log_n = 0;
    g_sdc = 0;
    puf_init(env, cfg);
    ademo_bind(env);
    puf_reset(env);
    env->selected_agent_idx = g_sdc;
    g_scene_i = which;
    printf("scenario map %d  town01  vehicles %d  sdc %d\n",
        g_scenes[which].map_id, env->active_agent_count, g_sdc);
    fflush(stdout);
    ademo_title();
}

static void ademo_load(Drive* env, Dict* cfg, int which) {
    if (which < 0 || which >= ADEMO_NSCENE) {
        which = 0;
    }
    ademo_release(env);
    if (ademo_town_scene(which)) {
        ademo_load_town(env, cfg, which);
        return;
    }
    dict_set_str(cfg, "data_dir", g_log_dir[0] ? g_log_dir : "data/womd");
    const char* dir = dict_get_str(cfg, "data_dir");
    char path[1200];
    snprintf(path, sizeof(path), "%s/json/%s",
        dir, g_scenes[which].name);
    Offline parsed = {0};
    offline_parse(&parsed, path);
    g_logs = parsed.logs;
    g_log_n = parsed.log_n;
    g_sdc = parsed.sdc;
    assert(g_log_n > 0);
    assert(g_sdc >= 0 && g_sdc < g_log_n);
    dict_set(cfg, "map_id", g_scenes[which].map_id);
    dict_set(cfg, "num_agents", g_log_n);
    puf_init(env, cfg);
    ademo_bind(env);
    Offline shift = {0};
    shift.logs = g_logs;
    shift.log_n = g_log_n;
    shift.mean_x = env->world_mean_x;
    shift.mean_y = env->world_mean_y;
    offline_shift_logs(&shift);
    ademo_place(env, 0);
    g_scene_i = which;
    printf("scenario map %d  %s  vehicles %d  sdc %d\n",
        g_scenes[which].map_id, g_scenes[which].name,
        g_log_n, g_sdc);
    fflush(stdout);
    ademo_title();
}

static int ademo_scene_index(int map_id) {
    for (int i = 0; i < ADEMO_NSCENE; i++) {
        if (g_scenes[i].map_id == map_id) {
            return i;
        }
    }
    return 0;
}

static void ademo_boot(Drive* env, Dict* cfg, int start) {
    if (env->rng == 0) {
        env->rng = (unsigned int)time(NULL);
        if (env->rng == 0) {
            env->rng = 1;
        }
    }
    const char* dir = dict_get_str(cfg, "data_dir");
    snprintf(g_log_dir, sizeof(g_log_dir), "%s", dir);
    g_fleet = (int)dict_get(cfg, "num_agents");
    ademo_load(env, cfg, start);
}

static void ademo_on_reset(Drive* env) {
    if (ademo_town()) {
        puf_reset(env);
        env->selected_agent_idx = g_sdc;
        ademo_hist_clear();
        ademo_policy_zero();
        return;
    }
    if (!g_logs) {
        return;
    }
    ademo_place(env, 0);
    ademo_hist_clear();
    ademo_policy_zero();
}

static void ademo_on_scene(Drive* env) {
    int n = ADEMO_NSCENE;
    int pick = g_scene_i;
    if (n > 1) {
        unsigned draw = rand_r(&env->rng) % (unsigned)(n - 1);
        pick = (int)draw;
        if (pick >= g_scene_i) {
            pick += 1;
        }
    }
    ademo_load(env, env->cfg, pick);
    ademo_hist_clear();
    ademo_policy_zero();
}

static void ademo_shutdown(void) {
    free(g_logs);
    g_logs = NULL;
    g_log_n = 0;
    free(g_obs);
    g_obs = NULL;
    free(g_act);
    g_act = NULL;
    free(g_rew);
    g_rew = NULL;
    free(g_term);
    g_term = NULL;
}

// The policy forward just ran on the observation this frame draws.
static void ademo_note_value(Drive* env) {
    PufferNet* net = g_ademo.net;
    if (!ademo_policy_on() || !net || !net->decoder
            || net->decoder->output_dim < 1) {
        env->predicted_ready = 0;
        return;
    }
    env->predicted_value = net->decoder->output[net->decoder->output_dim - 1];
    env->predicted_ready = 1;
}

static void ademo_frame(Drive* env, int step) {
    g_ademo.commit = step && !puf_sim_paused(env);
    ademo_forward(env);
    ademo_actions(env);
    ademo_distances();
    ademo_hist_note(env);
    ademo_note_value(env);
    puf_render(env);
    if (step && !puf_sim_paused(env)) {
        if (g_logs && env->timestep >= ADEMO_LOOP) {
            ademo_place(env, 0);
            ademo_hist_clear();
            ademo_policy_zero();
        } else if (g_logs) {
            ademo_replay_step(env);
        } else {
            puf_step(env);
        }
    }
}

static void ademo_run(Drive* env) {
    ademo_hist_clear();
    ademo_frame(env, 0);
    ademo_title();
    while (IsWindowReady()) {
        ademo_sync(env);
        int reset = ademo_poll(env);
        puf_sim_poll(env);
        if (reset) {
            ademo_on_reset(env);
            env->sim_hold_step = 1;
        }
        ademo_frame(env, 1);
        if (WindowShouldClose()) {
            break;
        }
    }
}

#endif
