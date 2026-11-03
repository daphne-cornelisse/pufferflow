// Post-training generalization. ./drive --eval [checkpoint]
// Density changes cars per town. Horizon changes episode length.
// Writes logs/<env>/eval.json and logs/<env>/eval-panel.png.

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

#define EVAL_BIN 64
#define EVAL_TOWN_MAX 4
#define EVAL_SCORE_TIE 1e-8

static const char* EVAL_TOWNS[EVAL_TOWN_MAX] = {
    "Town01", "Town02", "Town06", "Town10HD",
};

typedef struct {
    char path[1024];
    char source[1024];
    char run_id[256];
    double train_score;
    int have_score;
} EvalPick;

typedef struct {
    long stamp;
    int run;
    char name[256];
} EvalHist;

typedef struct {
    Drive* env;
    int agents;
    int map_id;
    int rollout;
    int* was_failed;
    int* first_failures;
    int* infractions;
    // Vehicle collisions in each bin. Taken from Log.collision_count
    // so a goal respawn cannot hide the step.
    int* collisions;
    int nbins;
} EvalSlot;

static int eval_metric_last(DictItem* item, double* out) {
    if (item->len > 0) {
        *out = item->values[item->len - 1];
        return 1;
    }
    if (item->str && puf_ini_parse_val(item->str, out)) {
        return 1;
    }
    return 0;
}

static Dict* eval_section(Ini* ini, const char* name) {
    for (int i = 0; i < ini->num_sections; i++) {
        if (strcmp(ini->sections[i].name, name) == 0) {
            return &ini->sections[i];
        }
    }
    return NULL;
}

static int eval_parse_name(const char* name, long* stamp, int* run) {
    const char* us;
    const char* dot;
    const char* p;
    if (strncmp(name, "sweep_", 6) != 0) {
        return 0;
    }
    us = strrchr(name, '_');
    dot = strrchr(name, '.');
    if (!us || !dot || strcmp(dot, ".ini") != 0 || us <= name + 6) {
        return 0;
    }
    for (p = name + 6; p < us; p++) {
        if (*p < '0' || *p > '9') {
            return 0;
        }
    }
    for (p = us + 1; p < dot; p++) {
        if (*p < '0' || *p > '9') {
            return 0;
        }
    }
    *stamp = strtol(name + 6, NULL, 10);
    *run = atoi(us + 1);
    return 1;
}

static int eval_hist_cmp(const void* a, const void* b) {
    const EvalHist* ha = a;
    const EvalHist* hb = b;
    if (ha->stamp < hb->stamp) {
        return -1;
    }
    if (ha->stamp > hb->stamp) {
        return 1;
    }
    return ha->run - hb->run;
}

static void eval_stem(const char* name, char* out, int n) {
    const char* dot = strrchr(name, '.');
    int len = dot ? (int)(dot - name) : (int)strlen(name);
    assert(len > 0 && len < n);
    memcpy(out, name, (size_t)len);
    out[len] = 0;
}

static void eval_mkdir_p(const char* path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p != '/') {
            continue;
        }
        *p = 0;
        assert((mkdir(tmp, 0777) == 0 || errno == EEXIST)
            && "failed to create directory");
        *p = '/';
    }
    assert((mkdir(tmp, 0777) == 0 || errno == EEXIST)
        && "failed to create directory");
}

static void eval_copy_file(const char* src, const char* dst) {
    FILE* in = fopen(src, "rb");
    FILE* out = fopen(dst, "wb");
    char buf[1 << 16];
    size_t nread;
    assert(in && out);
    while ((nread = fread(buf, 1, sizeof(buf), in)) > 0) {
        assert(fwrite(buf, 1, nread, out) == nread);
    }
    assert(!ferror(in));
    fclose(in);
    fclose(out);
}

// Highest numeric step stem. ctime would follow a rewrite of an older step.
static int eval_latest_bin(const char* dir, char* out, int n) {
    DIR* dp = opendir(dir);
    struct dirent* ent;
    long long best = -1;
    int have = 0;
    if (!dp) {
        return 0;
    }
    while ((ent = readdir(dp))) {
        long long step;
        char path[1024];
        int wrote;
        if (ent->d_name[0] == '.') {
            continue;
        }
        step = -1;
        {
            size_t len = strlen(ent->d_name);
            int digits = 1;
            size_t i;
            if (len < 5 || strcmp(ent->d_name + len - 4, ".bin") != 0) {
                continue;
            }
            for (i = 0; i < len - 4; i++) {
                if (ent->d_name[i] < '0' || ent->d_name[i] > '9') {
                    digits = 0;
                }
            }
            if (!digits) {
                continue;
            }
            step = strtoll(ent->d_name, NULL, 10);
        }
        if (have && step <= best) {
            continue;
        }
        wrote = snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        assert(wrote > 0 && wrote < (int)sizeof(path));
        best = step;
        have = 1;
        snprintf(out, (size_t)n, "%s", path);
    }
    closedir(dp);
    return have;
}

static int eval_better(int minimize, int have, double score, double best) {
    if (!have || fabs(score - best) <= EVAL_SCORE_TIE) {
        return 1;
    }
    if (minimize) {
        return score < best;
    }
    return score > best;
}

static void eval_log_dir(Ini* ini, char* out, int n) {
    int wrote = snprintf(out, (size_t)n, "%s/%s",
        puf_ini_get_str(ini, "base", "log_dir"),
        puf_ini_get_str(ini, "base", "env_name"));
    assert(wrote > 0 && wrote < n);
}

// Highest finished sweep metric. A tie keeps the later log stamp.
// The checkpoint is copied to checkpoints/<env>/best.bin.
static void eval_save_best(Ini* ini, EvalPick* pick) {
    const char* metric = puf_ini_get_str(ini, "sweep", "metric");
    const char* goal = puf_ini_get_str(ini, "sweep", "goal");
    int minimize = strcmp(goal, "minimize") == 0;
    char metric_key[128];
    char log_dir[1024];
    DIR* dp;
    EvalHist* logs = NULL;
    struct dirent* ent;
    int cap = 0;
    int n = 0;
    int have = 0;
    double best_score = 0;
    char best_src[1024];
    char best_run[256];
    char saved_dir[1024];
    assert(minimize || strcmp(goal, "maximize") == 0);
    snprintf(metric_key, sizeof(metric_key), "%s%s",
        strchr(metric, '/') ? "" : "env/", metric);
    eval_log_dir(ini, log_dir, (int)sizeof(log_dir));
    dp = opendir(log_dir);
    assert(dp && "no sweep logs");
    while ((ent = readdir(dp))) {
        long stamp = 0;
        int run = 0;
        int wrote;
        if (!eval_parse_name(ent->d_name, &stamp, &run)) {
            continue;
        }
        if (n == cap) {
            cap = cap ? cap * 2 : 16;
            logs = (EvalHist*)realloc(logs, (size_t)cap * sizeof(EvalHist));
            assert(logs);
        }
        logs[n].stamp = stamp;
        logs[n].run = run;
        wrote = snprintf(logs[n].name, sizeof(logs[n].name), "%s", ent->d_name);
        assert(wrote > 0 && wrote < (int)sizeof(logs[n].name));
        n++;
    }
    closedir(dp);
    assert(n > 0 && "no sweep logs");
    qsort(logs, (size_t)n, sizeof(EvalHist), eval_hist_cmp);
    best_src[0] = 0;
    best_run[0] = 0;
    for (int i = 0; i < n; i++) {
        char path[1200];
        char run_id[256];
        char ckpt_dir[1200];
        char bin[1024];
        Ini hist = {0};
        Dict* base;
        Dict* metrics;
        DictItem* item;
        const char* ckpt;
        const char* envn;
        const char* rid;
        double score = 0;
        int wrote;
        wrote = snprintf(path, sizeof(path), "%s/%s", log_dir, logs[i].name);
        assert(wrote > 0 && wrote < (int)sizeof(path));
        puf_ini_load_file(&hist, path);
        metrics = eval_section(&hist, "metrics");
        item = metrics ? dict_find(metrics, metric_key) : NULL;
        if (!item || !eval_metric_last(item, &score) || !isfinite(score)) {
            puf_ini_free(&hist);
            continue;
        }
        if (!eval_better(minimize, have, score, best_score)) {
            puf_ini_free(&hist);
            continue;
        }
        base = eval_section(&hist, "base");
        ckpt = NULL;
        envn = NULL;
        rid = NULL;
        if (base) {
            item = dict_find(base, "checkpoint_dir");
            if (item && item->str) {
                ckpt = item->str;
            }
            item = dict_find(base, "env_name");
            if (item && item->str) {
                envn = item->str;
            }
            item = dict_find(base, "run_id");
            if (item && item->str) {
                rid = item->str;
            }
        }
        if (!ckpt) {
            ckpt = puf_ini_get_str(ini, "base", "checkpoint_dir");
        }
        if (!envn) {
            envn = puf_ini_get_str(ini, "base", "env_name");
        }
        if (rid && rid[0]) {
            snprintf(run_id, sizeof(run_id), "%s", rid);
        } else {
            eval_stem(logs[i].name, run_id, (int)sizeof(run_id));
        }
        wrote = snprintf(ckpt_dir, sizeof(ckpt_dir), "%s/%s/%s",
            ckpt, envn, run_id);
        assert(wrote > 0 && wrote < (int)sizeof(ckpt_dir));
        if (!eval_latest_bin(ckpt_dir, bin, (int)sizeof(bin))) {
            puf_ini_free(&hist);
            continue;
        }
        have = 1;
        best_score = score;
        snprintf(best_src, sizeof(best_src), "%s", bin);
        snprintf(best_run, sizeof(best_run), "%s", run_id);
        puf_ini_free(&hist);
    }
    free(logs);
    assert(have && "no sweep checkpoint with a metric");
    {
        int wrote = snprintf(saved_dir, sizeof(saved_dir), "%s/%s",
            puf_ini_get_str(ini, "base", "checkpoint_dir"),
            puf_ini_get_str(ini, "base", "env_name"));
        assert(wrote > 0 && wrote < (int)sizeof(saved_dir));
    }
    eval_mkdir_p(saved_dir);
    snprintf(pick->path, sizeof(pick->path), "%s/best.bin", saved_dir);
    snprintf(pick->source, sizeof(pick->source), "%s", best_src);
    snprintf(pick->run_id, sizeof(pick->run_id), "%s", best_run);
    pick->train_score = best_score;
    pick->have_score = 1;
    assert(strcmp(pick->source, pick->path) != 0);
    eval_copy_file(pick->source, pick->path);
    printf("eval best %s score=%.4f\n", pick->run_id, pick->train_score);
    printf("eval source %s\n", pick->source);
    printf("eval saved %s\n", pick->path);
    fflush(stdout);
}

static void eval_path_run(const char* path, char* out, int n) {
    const char* slash = strrchr(path, '/');
    const char* start;
    int len;
    out[0] = 0;
    if (!slash || slash == path) {
        return;
    }
    start = slash;
    while (start > path && start[-1] != '/') {
        start--;
    }
    len = (int)(slash - start);
    if (len <= 0 || len >= n) {
        return;
    }
    memcpy(out, start, (size_t)len);
    out[len] = 0;
}

static void eval_lookup(Ini* ini, EvalPick* pick) {
    char log_dir[1024];
    char metric_key[128];
    const char* metric;
    DIR* dp;
    struct dirent* ent;
    if (!pick->source[0]) {
        snprintf(pick->source, sizeof(pick->source), "%s", pick->path);
    }
    if (!pick->run_id[0]) {
        eval_path_run(pick->path, pick->run_id, (int)sizeof(pick->run_id));
    }
    if (!pick->run_id[0]) {
        return;
    }
    metric = puf_ini_get_str(ini, "sweep", "metric");
    snprintf(metric_key, sizeof(metric_key), "%s%s",
        strchr(metric, '/') ? "" : "env/", metric);
    eval_log_dir(ini, log_dir, (int)sizeof(log_dir));
    dp = opendir(log_dir);
    if (!dp) {
        return;
    }
    while ((ent = readdir(dp))) {
        long stamp = 0;
        int run = 0;
        char stem[256];
        char path[1200];
        Ini hist = {0};
        Dict* metrics;
        DictItem* item;
        double score = 0;
        if (!eval_parse_name(ent->d_name, &stamp, &run)) {
            continue;
        }
        eval_stem(ent->d_name, stem, (int)sizeof(stem));
        if (strcmp(stem, pick->run_id) != 0) {
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", log_dir, ent->d_name);
        puf_ini_load_file(&hist, path);
        metrics = eval_section(&hist, "metrics");
        item = metrics ? dict_find(metrics, metric_key) : NULL;
        if (item && eval_metric_last(item, &score) && isfinite(score)) {
            pick->train_score = score;
            pick->have_score = 1;
        }
        puf_ini_free(&hist);
        break;
    }
    closedir(dp);
}

static void eval_greedy(PufferNet* net, const float* obs, int n,
        float* actions, const float* terminals) {
    assert(net->num_agents == n);
    assert(!net->is_continuous);
    mingru_zero_term(net->mingru, terminals);
    puf_encode(net, (float*)obs);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    multidiscrete(net->multidiscrete, net->decoder->output, actions, 1, NULL);
}

static void eval_bind(Drive* env, float* obs, float* act,
        float* rew, float* term) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = obs + (size_t)i * OBS_SIZE;
        env->agents[i].actions = act + (size_t)i * NUM_ATNS;
        env->agents[i].rewards = rew + i;
        env->agents[i].terminals = term + i;
        env->agents[i].policy = 0;
    }
}

// vec_log divides every sum by n. n stays the episode count.
static void eval_mean_log(Log* log) {
    float* acc = (float*)log;
    int nfields = (int)(sizeof(Log) / sizeof(float));
    float n = log->n;
    assert(n > 0.0f);
    for (int j = 0; j < nfields; j++) {
        if (acc + j == &log->n) {
            continue;
        }
        acc[j] /= n;
    }
}

static void eval_push(cJSON* arr, Drive* env, int map_id, int rollout,
        int agents, int length, int* fails, int* infr, int* cols,
        int nbins) {
    cJSON* obj = cJSON_CreateObject();
    Dict out = {0};
    assert(map_id >= 0 && map_id < EVAL_TOWN_MAX);
    cJSON_AddNumberToObject(obj, "agents", agents);
    cJSON_AddNumberToObject(obj, "map_id", map_id);
    cJSON_AddStringToObject(obj, "town", EVAL_TOWNS[map_id]);
    cJSON_AddNumberToObject(obj, "rollout", rollout);
    // Configured horizon. The log mean is one step shorter: the last
    // tick only aggregates, so it must not replace this field.
    cJSON_AddNumberToObject(obj, "episode_length", length);
    puf_log(&env->log, &out);
    cJSON_AddNumberToObject(obj, "total_infractions",
        env->log.total_infractions);
    cJSON_AddNumberToObject(obj, "total_distance_travelled",
        env->log.total_distance_travelled);
    for (int i = 0; i < out.size; i++) {
        if (strcmp(out.items[i].key, "episode_length") == 0) {
            continue;
        }
        cJSON_AddNumberToObject(obj, out.items[i].key, out.items[i].value);
    }
    dict_clear(&out);
    if (nbins > 0) {
        cJSON* bins = cJSON_CreateArray();
        assert(fails && infr && cols);
        for (int b = 0; b < nbins; b++) {
            cJSON* one = cJSON_CreateObject();
            cJSON_AddNumberToObject(one, "start", b * EVAL_BIN);
            cJSON_AddNumberToObject(one, "first_failures", fails[b]);
            cJSON_AddNumberToObject(one, "infractions", infr[b]);
            cJSON_AddNumberToObject(one, "collisions", cols[b]);
            cJSON_AddItemToArray(bins, one);
        }
        cJSON_AddItemToObject(obj, "bins", bins);
    }
    cJSON_AddItemToArray(arr, obj);
}

// One greedy net over every town copy. puf_reset does not clear env->log.
static void eval_episode(EvalSlot* slots, int nslot, PufferNet* net,
        float* obs, float* act, float* term, int batch) {
    int agents = slots[0].agents;
    int length = slots[0].env->episode_length;
    int tick = 0;
    assert(length >= EVAL_BIN && length % EVAL_BIN == 0);
#pragma omp parallel for schedule(static)
    for (int s = 0; s < nslot; s++) {
        Drive* env = slots[s].env;
        memset(&env->log, 0, sizeof(env->log));
        env->rng = 1000u + (unsigned)slots[s].rollout * 4u
            + (unsigned)slots[s].map_id;
        if (slots[s].first_failures) {
            memset(slots[s].first_failures, 0,
                (size_t)slots[s].nbins * sizeof(int));
            memset(slots[s].infractions, 0,
                (size_t)slots[s].nbins * sizeof(int));
            memset(slots[s].collisions, 0,
                (size_t)slots[s].nbins * sizeof(int));
        }
        puf_reset(env);
    }
    fprintf(stderr, "eval placed agents=%d length=%d\n", agents, length);
    fflush(stderr);
    while (slots[0].env->log.n < (float)agents) {
        tick++;
#pragma omp parallel for schedule(static)
        for (int s = 0; s < nslot; s++) {
            Drive* env = slots[s].env;
            int n = env->num_agents;
            float* term_s = term + (size_t)s * n;
            float mark = env->timestep == 0 ? 1.0f : 0.0f;
            for (int i = 0; i < n; i++) {
                term_s[i] = mark;
                slots[s].was_failed[i] = env->entities[i].failed;
            }
        }
        eval_greedy(net, obs, batch, act, term);
#pragma omp parallel for schedule(static)
        for (int s = 0; s < nslot; s++) {
            Drive* env = slots[s].env;
            int n = env->num_agents;
            int step_i;
            int bin;
            float prev_col[DRIVE_AGENT_MAX];
            assert(n <= DRIVE_AGENT_MAX);
            for (int i = 0; i < n; i++) {
                prev_col[i] = env->logs[i].collision_count;
            }
            puf_step(env);
            // The step that reaches episode_length only aggregates.
            // It also resets the env, so it is not part of the episode.
            if (env->timestep == 0 || !slots[s].first_failures) {
                continue;
            }
            step_i = env->timestep - 1;
            bin = step_i / EVAL_BIN;
            assert(bin >= 0 && bin < slots[s].nbins);
            for (int i = 0; i < n; i++) {
                float delta = env->logs[i].collision_count - prev_col[i];
                int hit = env->entities[i].collision_state > NO_COLLISION;
                if (delta > 0.0f) {
                    slots[s].collisions[bin] += (int)(delta + 0.5f);
                }
                if (!slots[s].was_failed[i] && env->entities[i].failed) {
                    slots[s].first_failures[bin] += 1;
                }
                if (hit) {
                    slots[s].infractions[bin] += 1;
                }
            }
        }
        if (tick == 1 || tick % EVAL_BIN == 0) {
            fprintf(stderr, "eval agents=%d length=%d step %d/%d\n",
                agents, length, tick, length);
            fflush(stderr);
        }
    }
    for (int s = 0; s < nslot; s++) {
        assert(slots[s].env->log.n == (float)slots[s].agents);
    }
}

// Cars need goal_distance_min ahead, then GOAL_FOLLOW_GAP before the next.
static int eval_lane_cap(Drive* env, float* lane_m_out) {
    float lane_m = 0.0f;
    float gap = env->goal_distance_min + GOAL_FOLLOW_GAP;
    int cap;
    assert(gap > 1.0f);
    for (int i = 0; i < env->lane_n; i++) {
        int nxt = env->lane_next[i];
        float dx;
        float dy;
        if (nxt < 0) {
            continue;
        }
        dx = env->lane_x[nxt] - env->lane_x[i];
        dy = env->lane_y[nxt] - env->lane_y[i];
        lane_m += sqrtf(dx * dx + dy * dy);
    }
    cap = (int)(lane_m / gap);
    *lane_m_out = lane_m;
    return cap;
}

static void eval_skip(cJSON* arr, int map, int agents, float lane_m, int cap) {
    cJSON* obj = cJSON_CreateObject();
    assert(map >= 0 && map < EVAL_TOWN_MAX);
    cJSON_AddNumberToObject(obj, "agents", agents);
    cJSON_AddNumberToObject(obj, "map_id", map);
    cJSON_AddStringToObject(obj, "town", EVAL_TOWNS[map]);
    cJSON_AddNumberToObject(obj, "lane_m", lane_m);
    cJSON_AddNumberToObject(obj, "capacity", cap);
    cJSON_AddItemToArray(arr, obj);
    fprintf(stderr, "eval skip %s agents=%d lane_m=%.0f capacity=%d\n",
        EVAL_TOWNS[map], agents, lane_m, cap);
    fflush(stderr);
}

static int eval_copies(int cap, int agents, int towns) {
    int copies = cap / (agents * towns);
    // At least three rollouts. More while the training fleet can hold them.
    if (copies < 3) {
        copies = 3;
    }
    return copies;
}

static void eval_fleet(Dict* cfg, Weights* weights, int hidden, int layers,
        int act_sizes[], int num_actions, int agents, int towns,
        int copies, const int* lengths, int nlen, int train_length,
        cJSON* density, cJSON* horizon, cJSON* unplaced) {
    int max_bins = 0;
    int nmaps = 0;
    int nslot;
    int batch;
    int maps[EVAL_TOWN_MAX];
    EvalSlot* slots;
    float* obs;
    float* act;
    float* rew;
    float* term;
    int* was;
    int* fails = NULL;
    int* infr = NULL;
    int* hits = NULL;
    PufferNet* net;
    fprintf(stderr, "eval init agents=%d copies=%d\n", agents, copies);
    fflush(stderr);
    for (int i = 0; i < nlen; i++) {
        int bins = lengths[i] / EVAL_BIN;
        assert(lengths[i] >= EVAL_BIN && lengths[i] % EVAL_BIN == 0);
        if (bins > max_bins) {
            max_bins = bins;
        }
    }
    dict_set(cfg, "num_agents", agents);
    for (int map = 0; map < towns; map++) {
        float lane_m = 0.0f;
        int cap;
        Drive* env = (Drive*)calloc(1, sizeof(Drive));
        assert(env);
        dict_set(cfg, "map_id", map);
        fprintf(stderr, "eval init agents=%d %s\n", agents, EVAL_TOWNS[map]);
        fflush(stderr);
        puf_init(env, cfg);
        assert(env->num_agents == agents);
        cap = eval_lane_cap(env, &lane_m);
        if (agents > cap) {
            eval_skip(unplaced, map, agents, lane_m, cap);
            free(env->map_name);
            env->map_name = NULL;
            puf_close(env);
            free(env);
            continue;
        }
        maps[nmaps++] = map;
        free(env->map_name);
        env->map_name = NULL;
        puf_close(env);
        free(env);
    }
    assert(nmaps > 0);
    nslot = nmaps * copies;
    batch = nslot * agents;
    slots = (EvalSlot*)calloc((size_t)nslot, sizeof(EvalSlot));
    obs = (float*)calloc((size_t)batch * OBS_SIZE, sizeof(float));
    act = (float*)calloc((size_t)batch * NUM_ATNS, sizeof(float));
    rew = (float*)calloc((size_t)batch, sizeof(float));
    term = (float*)calloc((size_t)batch, sizeof(float));
    was = (int*)calloc((size_t)batch, sizeof(int));
    assert(slots && obs && act && rew && term && was);
    if (horizon) {
        fails = (int*)calloc((size_t)nslot * max_bins, sizeof(int));
        infr = (int*)calloc((size_t)nslot * max_bins, sizeof(int));
        hits = (int*)calloc((size_t)nslot * max_bins, sizeof(int));
        assert(fails && infr && hits);
    }
    for (int mi = 0; mi < nmaps; mi++) {
        int map = maps[mi];
        dict_set(cfg, "map_id", map);
        for (int r = 0; r < copies; r++) {
            int s = mi * copies + r;
            Drive* env = (Drive*)calloc(1, sizeof(Drive));
            assert(env);
            puf_init(env, cfg);
            assert(env->num_agents == agents);
            slots[s].env = env;
            slots[s].agents = agents;
            slots[s].map_id = map;
            slots[s].rollout = r;
            slots[s].was_failed = was + (size_t)s * agents;
            if (horizon) {
                slots[s].first_failures = fails + (size_t)s * max_bins;
                slots[s].infractions = infr + (size_t)s * max_bins;
                slots[s].collisions = hits + (size_t)s * max_bins;
            }
            eval_bind(env,
                obs + (size_t)s * agents * OBS_SIZE,
                act + (size_t)s * agents * NUM_ATNS,
                rew + (size_t)s * agents,
                term + (size_t)s * agents);
        }
    }
    weights->idx = 0;
    net = make_puffernet(weights, batch, OBS_SIZE, hidden, layers,
        act_sizes, num_actions);
    assert(net);
    for (int li = 0; li < nlen; li++) {
        int length = lengths[li];
        int nbins = length / EVAL_BIN;
        for (int s = 0; s < nslot; s++) {
            slots[s].env->episode_length = length;
            slots[s].nbins = horizon ? nbins : 0;
        }
        fprintf(stderr, "eval roll agents=%d length=%d\n", agents, length);
        fflush(stderr);
        eval_episode(slots, nslot, net, obs, act, term, batch);
        for (int s = 0; s < nslot; s++) {
            Drive* env = slots[s].env;
            int fails_n = 0;
            eval_mean_log(&env->log);
            if (horizon) {
                for (int b = 0; b < nbins; b++) {
                    fails_n += slots[s].first_failures[b];
                }
                fprintf(stderr,
                    "eval agents=%d length=%d %s rollout=%d "
                    "score=%.4f collisions=%.3f infractions=%.3f "
                    "failures=%d\n",
                    agents, length, EVAL_TOWNS[slots[s].map_id],
                    slots[s].rollout, env->log.score,
                    env->log.collision_count,
                    env->log.total_infractions, fails_n);
            } else {
                fprintf(stderr,
                    "eval agents=%d length=%d %s rollout=%d "
                    "score=%.4f collisions=%.3f infractions=%.3f\n",
                    agents, length, EVAL_TOWNS[slots[s].map_id],
                    slots[s].rollout, env->log.score,
                    env->log.collision_count,
                    env->log.total_infractions);
            }
            fflush(stderr);
            if (density && length == train_length) {
                eval_push(density, env, slots[s].map_id, slots[s].rollout,
                    agents, length, NULL, NULL, NULL, 0);
            }
            if (horizon) {
                eval_push(horizon, env, slots[s].map_id, slots[s].rollout,
                    agents, length, slots[s].first_failures,
                    slots[s].infractions, slots[s].collisions, nbins);
            }
        }
    }
    free_puffernet(net);
    for (int s = 0; s < nslot; s++) {
        free(slots[s].env->map_name);
        slots[s].env->map_name = NULL;
        puf_close(slots[s].env);
        free(slots[s].env);
    }
    free(slots);
    free(obs);
    free(act);
    free(rew);
    free(term);
    free(was);
    free(fails);
    free(infr);
    free(hits);
}

static void eval_write(cJSON* root, const char* path) {
    char* text = cJSON_Print(root);
    FILE* fp;
    assert(text);
    fp = fopen(path, "w");
    assert(fp);
    fputs(text, fp);
    fputc('\n', fp);
    assert(!ferror(fp));
    fclose(fp);
    cJSON_free(text);
}

static void eval_plot(const char* script, const char* json_path,
        const char* png_path) {
    const char* py = "/home/daphne/pufferflow/.venv/bin/python";
    char cmd[4096];
    int n;
    int status;
    if (access(py, X_OK) != 0) {
        py = "python3";
    }
    n = snprintf(cmd, sizeof(cmd),
        "%s %s '%s' '%s'", py, script, json_path, png_path);
    assert(n > 0 && n < (int)sizeof(cmd));
    fprintf(stderr, "eval plot %s\n", png_path);
    fflush(stderr);
    status = system(cmd);
    assert(status == 0);
}

// One Town06 episode per car count. Horizon bins are windows inside
// that episode. The env resets only when the episode ends.
static void eval_benchmark(Ini* ini, Dict* cfg, Weights* weights, int hidden,
        int layers, int act_sizes[], int num_actions, EvalPick* pick) {
    int map = (int)dict_get(cfg, "map_id");
    int length = (int)dict_get(cfg, "episode_length");
    int amin = (int)puf_ini_get(ini, "benchmark", "agents_min");
    int amax = (int)puf_ini_get(ini, "benchmark", "agents_max");
    int astep = (int)puf_ini_get(ini, "benchmark", "agents_step");
    int bin = (int)puf_ini_get(ini, "benchmark", "failure_bin");
    int agents;
    char log_dir[1024];
    char json_path[1200];
    char png_path[1200];
    cJSON* root;
    cJSON* runs;
    int wrote;
    float* obs;
    float* act;
    float* rew;
    float* term;
    assert(map >= 0 && map < EVAL_TOWN_MAX);
    assert(amin > 0 && astep > 0 && amax >= amin);
    assert((amax - amin) % astep == 0);
    assert(amax <= DRIVE_AGENT_MAX);
    assert(bin > 0 && length > bin && length % bin == 0);
    if (!pick->have_score) {
        eval_lookup(ini, pick);
    }
    if (!pick->source[0]) {
        snprintf(pick->source, sizeof(pick->source), "%s", pick->path);
    }
    printf("eval benchmark %s %s agents %d..%d step %d length %d\n",
        EVAL_TOWNS[map], pick->path, amin, amax, astep, length);
    fflush(stdout);
    obs = (float*)calloc((size_t)amax * OBS_SIZE, sizeof(float));
    act = (float*)calloc((size_t)amax * NUM_ATNS, sizeof(float));
    rew = (float*)calloc((size_t)amax, sizeof(float));
    term = (float*)calloc((size_t)amax, sizeof(float));
    assert(obs && act && rew && term);
    root = cJSON_CreateObject();
    runs = cJSON_CreateArray();
    cJSON_AddStringToObject(root, "checkpoint", pick->path);
    cJSON_AddStringToObject(root, "source", pick->source);
    cJSON_AddStringToObject(root, "town", EVAL_TOWNS[map]);
    cJSON_AddNumberToObject(root, "map_id", map);
    cJSON_AddNumberToObject(root, "episode_length", length);
    cJSON_AddNumberToObject(root, "failure_bin", bin);
    cJSON_AddItemToObject(root, "runs", runs);
    dict_set(cfg, "map_id", map);
    dict_set(cfg, "episode_length", length);
    for (agents = amin; agents <= amax; agents += astep) {
        Drive* env = (Drive*)calloc(1, sizeof(Drive));
        PufferNet* net;
        int nbins = length / bin;
        int* fail_at = (int*)malloc((size_t)agents * sizeof(int));
        int* was = (int*)calloc((size_t)agents, sizeof(int));
        int* cols = (int*)calloc((size_t)nbins, sizeof(int));
        float* prev_col = (float*)malloc((size_t)agents * sizeof(float));
        int tick = 0;
        int i;
        cJSON* run;
        cJSON* steps;
        cJSON* collisions;
        assert(env && fail_at && was && cols && prev_col);
        dict_set(cfg, "num_agents", agents);
        puf_init(env, cfg);
        assert(env->num_agents == agents);
        env->episode_length = length;
        eval_bind(env, obs, act, rew, term);
        weights->idx = 0;
        net = make_puffernet(weights, agents, OBS_SIZE, hidden, layers,
            act_sizes, num_actions);
        assert(net);
        for (i = 0; i < agents; i++) {
            fail_at[i] = -1;
        }
        memset(&env->log, 0, sizeof(env->log));
        env->rng = 1000u + (unsigned)agents;
        puf_reset(env);
        fprintf(stderr, "eval placed agents=%d length=%d %s\n",
            agents, length, EVAL_TOWNS[map]);
        fflush(stderr);
        while (env->log.n < (float)agents) {
            float mark = env->timestep == 0 ? 1.0f : 0.0f;
            tick++;
            for (i = 0; i < agents; i++) {
                term[i] = mark;
                was[i] = env->entities[i].failed;
                prev_col[i] = env->logs[i].collision_count;
            }
            eval_greedy(net, obs, agents, act, term);
            puf_step(env);
            // Reaching episode_length only aggregates and resets.
            if (env->timestep == 0) {
                continue;
            }
            {
                int step_i = env->timestep - 1;
                int b = step_i / bin;
                assert(b >= 0 && b < nbins);
                for (i = 0; i < agents; i++) {
                    float delta = env->logs[i].collision_count - prev_col[i];
                    if (!was[i] && env->entities[i].failed
                            && fail_at[i] < 0) {
                        fail_at[i] = step_i;
                    }
                    if (delta > 0.0f) {
                        cols[b] += (int)(delta + 0.5f);
                    }
                }
            }
            if (tick == 1 || tick % bin == 0) {
                fprintf(stderr, "eval agents=%d length=%d step %d/%d\n",
                    agents, length, tick, length);
                fflush(stderr);
            }
        }
        assert(env->log.n == (float)agents);
        eval_mean_log(&env->log);
        run = cJSON_CreateObject();
        steps = cJSON_CreateArray();
        cJSON_AddNumberToObject(run, "agents", agents);
        cJSON_AddNumberToObject(run, "episode_length", length);
        cJSON_AddNumberToObject(run, "score", env->log.score);
        for (i = 0; i < agents; i++) {
            if (fail_at[i] < 0) {
                continue;
            }
            assert(fail_at[i] < length);
            cJSON_AddItemToArray(steps, cJSON_CreateNumber(fail_at[i]));
        }
        cJSON_AddItemToObject(run, "failure_step", steps);
        collisions = cJSON_CreateArray();
        for (i = 0; i < nbins; i++) {
            cJSON_AddItemToArray(collisions, cJSON_CreateNumber(cols[i]));
        }
        cJSON_AddItemToObject(run, "collisions", collisions);
        cJSON_AddItemToArray(runs, run);
        fprintf(stderr, "eval agents=%d score=%.4f failures=%d\n",
            agents, env->log.score, cJSON_GetArraySize(steps));
        fflush(stderr);
        free_puffernet(net);
        free(env->map_name);
        env->map_name = NULL;
        puf_close(env);
        free(env);
        free(fail_at);
        free(was);
        free(cols);
        free(prev_col);
    }
    eval_log_dir(ini, log_dir, (int)sizeof(log_dir));
    eval_mkdir_p(log_dir);
    wrote = snprintf(json_path, sizeof(json_path), "%s/benchmark.json", log_dir);
    assert(wrote > 0 && wrote < (int)sizeof(json_path));
    wrote = snprintf(png_path, sizeof(png_path),
        "%s/benchmark-summary.png", log_dir);
    assert(wrote > 0 && wrote < (int)sizeof(png_path));
    eval_write(root, json_path);
    cJSON_Delete(root);
    free(obs);
    free(act);
    free(rew);
    free(term);
    printf("eval json %s\n", json_path);
    fflush(stdout);
    eval_plot("scripts/plot_benchmark.py", json_path, png_path);
    printf("eval panel %s\n", png_path);
    fflush(stdout);
}

static void eval_run(Ini* ini, Dict* cfg, Weights* weights, int hidden,
        int layers, int act_sizes[], int num_actions, EvalPick* pick) {
    if (puf_ini_section(ini, "benchmark", 0)) {
        eval_benchmark(ini, cfg, weights, hidden, layers,
            act_sizes, num_actions, pick);
        return;
    }
    int towns = (int)dict_get(cfg, "num_maps");
    int train_agents = (int)dict_get(cfg, "num_agents");
    int train_length = (int)dict_get(cfg, "episode_length");
    int cap = (int)puf_ini_get(ini, "vec", "total_agents");
    int counts[3] = {128, 256, 512};
    int horizons[3] = {256, 512, 1024};
    char log_dir[1024];
    char json_path[1200];
    char png_path[1200];
    cJSON* root;
    cJSON* density;
    cJSON* horizon;
    cJSON* unplaced;
    int wrote;
    assert(DRIVE_AGENT_MAX >= 512);
    assert(towns == EVAL_TOWN_MAX);
    assert(train_agents == counts[0]);
    assert(train_length == horizons[0]);
    assert(cap >= train_agents * towns);
    if (!pick->have_score) {
        eval_lookup(ini, pick);
    }
    if (!pick->source[0]) {
        snprintf(pick->source, sizeof(pick->source), "%s", pick->path);
    }
    printf("eval %s hidden %d layers %d\n", pick->path, hidden, layers);
    fflush(stdout);
    root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "checkpoint", pick->path);
    cJSON_AddStringToObject(root, "source", pick->source);
    if (pick->run_id[0]) {
        cJSON_AddStringToObject(root, "run", pick->run_id);
    }
    if (pick->have_score) {
        cJSON_AddNumberToObject(root, "train_score", pick->train_score);
    }
    cJSON_AddNumberToObject(root, "hidden", hidden);
    cJSON_AddNumberToObject(root, "layers", layers);
    cJSON_AddNumberToObject(root, "failure_bin", EVAL_BIN);
    density = cJSON_CreateArray();
    horizon = cJSON_CreateArray();
    unplaced = cJSON_CreateArray();
    cJSON_AddItemToObject(root, "density", density);
    cJSON_AddItemToObject(root, "horizon", horizon);
    cJSON_AddItemToObject(root, "unplaced", unplaced);
    // 512 first: a map that cannot hold the fleet fails before the long runs.
    for (int i = 2; i >= 1; i--) {
        int copies = eval_copies(cap, counts[i], towns);
        int one = train_length;
        eval_fleet(cfg, weights, hidden, layers, act_sizes, num_actions,
            counts[i], towns, copies, &one, 1, train_length,
            density, NULL, unplaced);
    }
    {
        int copies = eval_copies(cap, counts[0], towns);
        eval_fleet(cfg, weights, hidden, layers, act_sizes, num_actions,
            counts[0], towns, copies, horizons, 3, train_length,
            density, horizon, unplaced);
    }
    eval_log_dir(ini, log_dir, (int)sizeof(log_dir));
    eval_mkdir_p(log_dir);
    wrote = snprintf(json_path, sizeof(json_path), "%s/eval.json", log_dir);
    assert(wrote > 0 && wrote < (int)sizeof(json_path));
    wrote = snprintf(png_path, sizeof(png_path), "%s/eval-panel.png", log_dir);
    assert(wrote > 0 && wrote < (int)sizeof(png_path));
    eval_write(root, json_path);
    cJSON_Delete(root);
    printf("eval json %s\n", json_path);
    fflush(stdout);
    eval_plot("scripts/plot_eval.py", json_path, png_path);
    printf("eval panel %s\n", png_path);
    fflush(stdout);
}
