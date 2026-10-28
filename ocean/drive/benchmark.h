// Human-replay and long-horizon evals. Included after drive.h and offline.h.
// The caller supplies greedy actions. Replay controls the logged SDC and
// pins every other vehicle to the log. Horizon is the Town01 trainer env.

#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#define BENCH_HORIZON_MAX 8
#define BENCH_NAME 256

typedef struct BenchPolicy {
    void* ctx;
    int batch;
    // Replay scenes run this many ways. Horizon stays on one net.
    int threads;
    void (*act)(void* ctx, const float* obs, int n,
        float* actions, const float* terminals);
} BenchPolicy;

typedef struct BenchCfg {
    int scenes;
    int map_start;
    char replay_dir[512];
    int map_id;
    char horizon_dir[512];
    int agents;
    int horizons[BENCH_HORIZON_MAX];
    int n_horizons;
    int rollouts;
    char sweep_metric[64];
    char out_dir[512];
    int after_train;
} BenchCfg;

static void bench_copy(char* dst, int n, const char* src) {
    snprintf(dst, (size_t)n, "%s", src ? src : "");
}

static void bench_cfg_load(BenchCfg* cfg) {
    Ini ini = {0};
    double horizons[BENCH_HORIZON_MAX];
    int n;
    int i;
    memset(cfg, 0, sizeof(*cfg));
    puf_ini_load_file(&ini, "config/benchmark.ini");
    cfg->scenes = (int)puf_ini_get(&ini, "replay", "scenes");
    cfg->map_start = (int)puf_ini_get(&ini, "replay", "map_start");
    bench_copy(cfg->replay_dir, (int)sizeof(cfg->replay_dir),
        puf_ini_get_str(&ini, "replay", "data_dir"));
    cfg->map_id = (int)puf_ini_get(&ini, "horizon", "map_id");
    bench_copy(cfg->horizon_dir, (int)sizeof(cfg->horizon_dir),
        puf_ini_get_str(&ini, "horizon", "data_dir"));
    cfg->agents = (int)puf_ini_get(&ini, "horizon", "agents");
    cfg->rollouts = (int)puf_ini_get(&ini, "horizon", "rollouts");
    n = puf_ini_get_list(&ini, "horizon", "horizons",
        horizons, BENCH_HORIZON_MAX);
    cfg->n_horizons = n;
    for (i = 0; i < n; i++) {
        cfg->horizons[i] = (int)horizons[i];
    }
    bench_copy(cfg->sweep_metric, (int)sizeof(cfg->sweep_metric),
        puf_ini_get_str(&ini, "benchmark", "sweep_metric"));
    bench_copy(cfg->out_dir, (int)sizeof(cfg->out_dir),
        puf_ini_get_str(&ini, "benchmark", "out_dir"));
    cfg->after_train = (int)puf_ini_get(&ini, "benchmark", "after_train");
    puf_ini_free(&ini);
    assert(cfg->scenes >= 0);
    assert(cfg->map_start >= 0);
    assert(cfg->agents >= 1 && cfg->agents <= DRIVE_AGENT_MAX);
    assert(cfg->rollouts >= 1);
    assert(cfg->n_horizons >= 1);
}

static int bench_json_only(const struct dirent* ent) {
    size_t n = strlen(ent->d_name);
    if (n < 6) {
        return 0;
    }
    return strcmp(ent->d_name + n - 5, ".json") == 0;
}

static void bench_mkdir(const char* path) {
    if (mkdir(path, 0755) == 0 || errno == EEXIST) {
        return;
    }
    perror(path);
    assert(0 && "benchmark out dir");
}

static void bench_prepare_dir(const char* path) {
    char tmp[512];
    char* slash;
    bench_copy(tmp, (int)sizeof(tmp), path);
    slash = strrchr(tmp, '/');
    if (slash && slash != tmp) {
        *slash = 0;
        bench_mkdir(tmp);
    }
    bench_mkdir(path);
}

static float bench_metric(const BenchCfg* cfg, float collision, float offroad,
        float completion, float score, float ade_lat, float ade_lon,
        float horizon_collision, float horizon_score) {
    const char* name = cfg->sweep_metric;
    if (strcmp(name, "replay_collision") == 0) {
        return collision;
    }
    if (strcmp(name, "replay_offroad") == 0) {
        return offroad;
    }
    if (strcmp(name, "replay_completion") == 0) {
        return completion;
    }
    if (strcmp(name, "replay_score") == 0) {
        return score;
    }
    if (strcmp(name, "replay_ade_lat") == 0) {
        return ade_lat;
    }
    if (strcmp(name, "replay_ade_lon") == 0) {
        return ade_lon;
    }
    if (strcmp(name, "horizon_collision") == 0) {
        return horizon_collision;
    }
    if (strcmp(name, "horizon_score") == 0) {
        return horizon_score;
    }
    return 0.0f;
}

// "score" is the training metric. The report still lists replay and
// horizon numbers, and it omits the sweep line so protein is not told
// that the benchmark score is zero.
static int bench_metric_known(const char* name) {
    if (strcmp(name, "replay_collision") == 0) {
        return 1;
    }
    if (strcmp(name, "replay_offroad") == 0) {
        return 1;
    }
    if (strcmp(name, "replay_completion") == 0) {
        return 1;
    }
    if (strcmp(name, "replay_score") == 0) {
        return 1;
    }
    if (strcmp(name, "replay_ade_lat") == 0) {
        return 1;
    }
    if (strcmp(name, "replay_ade_lon") == 0) {
        return 1;
    }
    if (strcmp(name, "horizon_collision") == 0) {
        return 1;
    }
    if (strcmp(name, "horizon_score") == 0) {
        return 1;
    }
    return 0;
}

static void bench_bind_one(Drive* env, int agent, float* obs, float* act) {
    int i;
    for (i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = NULL;
        env->agents[i].actions = NULL;
        env->agents[i].rewards = NULL;
        env->agents[i].terminals = NULL;
    }
    env->agents[agent].observations = obs;
    env->agents[agent].actions = act;
}

static void bench_place(Drive* env, OfflineAgent* logs, int n, int sdc, int frame) {
    int i;
    for (i = 0; i < n; i++) {
        offline_write_agent(&env->entities[i], &logs[i], frame);
        env->entities[i].active_agent = i == sdc;
        if (i != sdc) {
            env->entities[i].goal_position_x = INVALID_POSITION;
            env->entities[i].goal_position_y = INVALID_POSITION;
        }
    }
    env->entities[sdc].goals_sampled = 1;
    env->selected_agent_idx = sdc;
    env->timestep = frame;
    env->bins_ready = 0;
}

static void bench_pin(Drive* env, OfflineAgent* logs, int n, int sdc, int frame) {
    int i;
    for (i = 0; i < n; i++) {
        if (i == sdc) {
            continue;
        }
        offline_write_agent(&env->entities[i], &logs[i], frame);
        env->entities[i].active_agent = 0;
        env->entities[i].goal_position_x = INVALID_POSITION;
        env->entities[i].goal_position_y = INVALID_POSITION;
    }
    env->entities[sdc].active_agent = 1;
    env->bins_ready = 0;
}

// Last valid sample, already shifted into the road frame. The JSON
// goalPosition is that point; the eval does not keep a second copy.
static void bench_set_log_goal(Entity* ego, const OfflineAgent* human) {
    int last = -1;
    int t;
    for (t = 0; t < OFFLINE_TRAJ; t++) {
        if (human->valid[t]) {
            last = t;
        }
    }
    assert(last >= 0);
    ego->goal_position_x = human->x[last];
    ego->goal_position_y = human->y[last];
    ego->goal_position_z = 0.0f;
}

// Displacement of the policy car from the log, in the log heading frame.
static void bench_ade(Entity* ego, OfflineAgent* human, int frame,
        float* lat, float* lon) {
    float dx = ego->x - human->x[frame];
    float dy = ego->y - human->y[frame];
    float ch = cosf(human->heading[frame]);
    float sh = sinf(human->heading[frame]);
    *lon = dx * ch + dy * sh;
    *lat = -dx * sh + dy * ch;
}

static int bench_replay_scene(Dict* cfg, int map_id, const char* json_path,
        BenchPolicy* pol, float* collision, float* offroad,
        float* completion, float* score, float* ade_lat, float* ade_lon) {
    Offline parsed = {0};
    Drive env = {0};
    Offline shifted = {0};
    float obs[OBS_SIZE];
    float act[NUM_ATNS];
    float lat_sum = 0.0f;
    float lon_sum = 0.0f;
    int ade_n = 0;
    int hit = 0;
    int road = 0;
    int reached = 0;
    int frame;
    int sdc;
    int n;
    offline_parse(&parsed, json_path);
    n = parsed.log_n;
    sdc = parsed.sdc;
    assert(n >= 1 && n <= DRIVE_AGENT_MAX);
    assert(sdc >= 0 && sdc < n);
    dict_set(cfg, "map_id", map_id);
    dict_set(cfg, "num_agents", n);
    puf_init(&env, cfg);
    shifted.logs = parsed.logs;
    shifted.log_n = n;
    shifted.mean_x = env.world_mean_x;
    shifted.mean_y = env.world_mean_y;
    offline_shift_logs(&shifted);
    bench_bind_one(&env, sdc, obs, act);
    bench_place(&env, parsed.logs, n, sdc, 0);
    bench_set_log_goal(&env.entities[sdc], &parsed.logs[sdc]);
    for (frame = 0; frame + 1 < OFFLINE_TRAJ; frame++) {
        float term = 0.0f;
        Entity* ego;
        if (!parsed.logs[sdc].valid[frame]
                || !parsed.logs[sdc].valid[frame + 1]) {
            break;
        }
        if (env.entities[sdc].x == INVALID_POSITION) {
            break;
        }
        compute_observations(&env);
        term = ade_n == 0 ? 1.0f : 0.0f;
        pol->act(pol->ctx, obs, 1, act, &term);
        move_dynamics(&env, sdc);
        bench_pin(&env, parsed.logs, n, sdc, frame + 1);
        collision_check(&env, sdc);
        ego = &env.entities[sdc];
        if (ego->collision_state == AGENT_COLLISION) {
            hit = 1;
        } else if (ego->collision_state == OFFROAD) {
            road = 1;
        }
        if (parsed.logs[sdc].valid[frame + 1]) {
            float lat = 0.0f;
            float lon = 0.0f;
            bench_ade(ego, &parsed.logs[sdc], frame + 1, &lat, &lon);
            lat_sum += fabsf(lat);
            lon_sum += fabsf(lon);
            ade_n++;
        }
        // Stop on the step that enters the goal. Later log frames are
        // not part of the episode, so they do not add collisions or ADE.
        if (relative_distance_2d(ego->x, ego->y,
                ego->goal_position_x, ego->goal_position_y)
                < MIN_DISTANCE_TO_GOAL) {
            reached = 1;
            break;
        }
    }
    puf_close(&env);
    free(env.map_name);
    free(parsed.logs);
    if (ade_n < 1) {
        return 0;
    }
    *collision = hit ? 1.0f : 0.0f;
    *offroad = road ? 1.0f : 0.0f;
    *completion = reached ? 1.0f : 0.0f;
    // Same rule as logs[i].score: first goal, and no hit before it.
    // Off-road counts. The collision rate above stays vehicle-only.
    *score = (reached && !hit && !road) ? 1.0f : 0.0f;
    *ade_lat = lat_sum / (float)ade_n;
    *ade_lon = lon_sum / (float)ade_n;
    return 1;
}

static void bench_replay(Dict* cfg, const BenchCfg* spec, BenchPolicy* pol,
        float* collision, float* offroad, float* completion, float* score,
        float* ade_lat, float* ade_lon, int* used, int* skipped) {
    struct dirent** names = NULL;
    char json_dir[512];
    int listed;
    int begin;
    int count;
    int i;
    float c_sum = 0.0f;
    float o_sum = 0.0f;
    float g_sum = 0.0f;
    float s_sum = 0.0f;
    float lat_sum = 0.0f;
    float lon_sum = 0.0f;
    int got = 0;
    int skip = 0;
    assert(pol->batch == 1);
    snprintf(json_dir, sizeof(json_dir), "%s/json", spec->replay_dir);
    dict_set_str(cfg, "data_dir", spec->replay_dir);
    listed = scandir(json_dir, &names, bench_json_only, alphasort);
    assert(listed >= 0 && "replay json dir");
    begin = spec->map_start;
    if (begin > listed) {
        begin = listed;
    }
    count = spec->scenes;
    if (begin + count > listed) {
        count = listed - begin;
    }
    fprintf(stderr, "replay %d scenes from %s  maps %d .. %d  threads %d\n",
        count, json_dir, begin, begin + count - 1, pol->threads);
    fflush(stderr);
    // puf_init writes the shared accel table on first use.
    init_action_space();
    // Each scene writes map_id into the dict, so the threads do not share it.
#ifdef _OPENMP
#pragma omp parallel num_threads(pol->threads)
#endif
    {
        Dict local = {0};
        dict_copy(&local, cfg);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (i = 0; i < count; i++) {
            char path[1024];
            float c = 0.0f;
            float o = 0.0f;
            float g = 0.0f;
            float s = 0.0f;
            float lat = 0.0f;
            float lon = 0.0f;
            int map_id = begin + i;
            int ok;
            int done;
            snprintf(path, sizeof(path), "%s/%s",
                json_dir, names[map_id]->d_name);
            ok = bench_replay_scene(&local, map_id, path, pol,
                &c, &o, &g, &s, &lat, &lon);
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (!ok) {
                    skip++;
                } else {
                    c_sum += c;
                    o_sum += o;
                    g_sum += g;
                    s_sum += s;
                    lat_sum += lat;
                    lon_sum += lon;
                    got++;
                }
                done = got + skip;
                if (done % 32 == 0 || done == count) {
                    float inv = got > 0 ? 1.0f / (float)got : 0.0f;
                    fprintf(stderr,
                        "replay %d/%d  collision %.3f  offroad %.3f  "
                        "score %.3f  completion %.3f  "
                        "ade lat %.3f  lon %.3f\n",
                        done, count, c_sum * inv, o_sum * inv,
                        s_sum * inv, g_sum * inv,
                        lat_sum * inv, lon_sum * inv);
                    fflush(stderr);
                }
            }
        }
        dict_clear(&local);
    }
    for (i = 0; i < listed; i++) {
        free(names[i]);
    }
    free(names);
    *used = got;
    *skipped = skip;
    if (got > 0) {
        float inv = 1.0f / (float)got;
        *collision = c_sum * inv;
        *offroad = o_sum * inv;
        *completion = g_sum * inv;
        *score = s_sum * inv;
        *ade_lat = lat_sum * inv;
        *ade_lon = lon_sum * inv;
    }
}

static void bench_bind_fleet(Drive* env, float* obs, float* act,
        float* rew, float* term) {
    int i;
    for (i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = obs + (size_t)i * OBS_SIZE;
        env->agents[i].actions = act + (size_t)i * NUM_ATNS;
        env->agents[i].rewards = rew + i;
        env->agents[i].terminals = term + i;
        env->agents[i].policy = 0;
    }
}

static void bench_one_rollout(Drive* env, BenchPolicy* pol, int rollout,
        float* obs, float* act, float* rew, float* term,
        float* collision_count, float* score) {
    int agents = env->num_agents;
    int steps_before[DRIVE_AGENT_MAX];
    int collided[DRIVE_AGENT_MAX];
    assert(agents >= 1 && agents <= DRIVE_AGENT_MAX);
    int i;
    int n;
    float sum = 0.0f;
    env->rng = 1000u + (unsigned)rollout;
    memset(&env->log, 0, sizeof(env->log));
    memset(steps_before, 0, sizeof(steps_before));
    memset(collided, 0, sizeof(collided));
    puf_reset(env);
    // The tick that reaches episode_length ends the episode without a
    // dynamics step, so a clean car records episode_length - 1 steps.
    while (env->log.n < (float)agents) {
        for (i = 0; i < agents; i++) {
            term[i] = env->timestep == 0 ? 1.0f : 0.0f;
        }
        pol->act(pol->ctx, obs, agents, act, term);
        puf_step(env);
        if (env->timestep == 0) {
            break;
        }
        for (i = 0; i < agents; i++) {
            if (collided[i]) {
                continue;
            }
            if (env->entities[i].collision_state > NO_COLLISION) {
                collided[i] = 1;
                continue;
            }
            steps_before[i] += 1;
        }
    }
    assert(env->log.n > 0.0f);
    n = (int)env->log.n;
    assert(n <= agents && n <= DRIVE_AGENT_MAX);
    for (i = 0; i < n; i++) {
        sum += (float)steps_before[i];
    }
    *collision_count = env->log.collision_count / env->log.n;
    *score = sum / (float)n;
}

static void bench_horizon(Dict* cfg, const BenchCfg* spec, BenchPolicy* pol,
        FILE* csv, float* collision_mean, float* score_mean) {
    float* obs;
    float* act;
    float* rew;
    float* term;
    float total = 0.0f;
    float score_total = 0.0f;
    int h;
    assert(pol->batch == spec->agents);
    obs = (float*)calloc((size_t)spec->agents * OBS_SIZE, sizeof(float));
    act = (float*)calloc((size_t)spec->agents * NUM_ATNS, sizeof(float));
    rew = (float*)calloc((size_t)spec->agents, sizeof(float));
    term = (float*)calloc((size_t)spec->agents, sizeof(float));
    assert(obs && act && rew && term);
    dict_set_str(cfg, "data_dir", spec->horizon_dir);
    dict_set(cfg, "map_id", spec->map_id);
    dict_set(cfg, "num_agents", spec->agents);
    fprintf(csv, "horizon,rollout,collision_count,steps_before\n");
    // The map does not depend on the horizon. Only the reset length does.
    dict_set(cfg, "episode_length", spec->horizons[0]);
    {
        Drive env = {0};
        puf_init(&env, cfg);
        bench_bind_fleet(&env, obs, act, rew, term);
        for (h = 0; h < spec->n_horizons; h++) {
            float sum = 0.0f;
            float score_sum = 0.0f;
            int r;
            int length = spec->horizons[h];
            env.episode_length = length;
            for (r = 0; r < spec->rollouts; r++) {
                float count = 0.0f;
                float score = 0.0f;
                bench_one_rollout(&env, pol, r, obs, act, rew, term,
                    &count, &score);
                sum += count;
                score_sum += score;
                fprintf(csv, "%d,%d,%.6f,%.6f\n", length, r, count, score);
                fprintf(stderr,
                    "horizon %d rollout %d  collision_count %.3f  "
                    "steps_before %.3f\n",
                    length, r, count, score);
                fflush(stderr);
            }
            total += sum / (float)spec->rollouts;
            score_total += score_sum / (float)spec->rollouts;
        }
        puf_close(&env);
        free(env.map_name);
    }
    free(obs);
    free(act);
    free(rew);
    free(term);
    *collision_mean = total / (float)spec->n_horizons;
    *score_mean = score_total / (float)spec->n_horizons;
}

// checkpoints/drive/<run>/<step>.bin -> logs/drive/<run>.benchmark.txt
static void bench_sidecar(const char* ckpt, const char* src_path) {
    char run[BENCH_NAME];
    char dst[512];
    const char* base;
    const char* slash;
    const char* mark;
    FILE* in;
    FILE* out;
    char buf[4096];
    size_t nread;
    if (!ckpt || !ckpt[0]) {
        return;
    }
    mark = strstr(ckpt, "checkpoints/drive/");
    if (!mark) {
        return;
    }
    base = mark + strlen("checkpoints/drive/");
    slash = strchr(base, '/');
    if (!slash || slash == base) {
        return;
    }
    {
        int nrun = (int)(slash - base);
        if (nrun <= 0 || nrun >= BENCH_NAME) {
            return;
        }
        memcpy(run, base, (size_t)nrun);
        run[nrun] = 0;
    }
    snprintf(dst, sizeof(dst), "logs/drive/%s.benchmark.txt", run);
    in = fopen(src_path, "rb");
    out = fopen(dst, "wb");
    if (!in || !out) {
        if (in) {
            fclose(in);
        }
        if (out) {
            fclose(out);
        }
        return;
    }
    while ((nread = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, nread, out) != nread) {
            break;
        }
    }
    fclose(in);
    fclose(out);
    fprintf(stderr, "wrote %s\n", dst);
}

static void bench_plot(const char* csv_path, const char* png_path, int map_id) {
    const char* py = "/home/daphne/pufferflow/.venv/bin/python";
    char cmd[2048];
    char title[64];
    int wrote;
    if (access(py, X_OK) != 0) {
        py = "python3";
    }
    if (map_id == 1) {
        snprintf(title, sizeof(title), "Town01");
    } else {
        snprintf(title, sizeof(title), "map %d", map_id);
    }
    wrote = snprintf(cmd, sizeof(cmd),
        "%s tests/plot_benchmark.py '%s' '%s' '%s'",
        py, csv_path, png_path, title);
    if (wrote < 0 || wrote >= (int)sizeof(cmd)) {
        fprintf(stderr, "benchmark plot command too long\n");
        return;
    }
    if (system(cmd) != 0) {
        fprintf(stderr, "benchmark plot failed\n");
    }
}

static void bench_run(Dict* cfg, const BenchCfg* spec,
        BenchPolicy* replay, BenchPolicy* horizon_pol, const char* ckpt) {
    char data_dir[512];
    char report_path[512];
    char csv_path[512];
    char png_path[512];
    FILE* report;
    FILE* csv;
    float collision = 0.0f;
    float offroad = 0.0f;
    float completion = 0.0f;
    float replay_score = 0.0f;
    float ade_lat = 0.0f;
    float ade_lon = 0.0f;
    float horizon_mean = 0.0f;
    float horizon_score = 0.0f;
    float sweep;
    int map_id = (int)dict_get(cfg, "map_id");
    int agents = (int)dict_get(cfg, "num_agents");
    int episode = (int)dict_get(cfg, "episode_length");
    int used = 0;
    int skipped = 0;
    bench_copy(data_dir, (int)sizeof(data_dir), dict_get_str(cfg, "data_dir"));
    bench_prepare_dir(spec->out_dir);
    snprintf(report_path, sizeof(report_path), "%s/report.txt", spec->out_dir);
    snprintf(csv_path, sizeof(csv_path), "%s/horizon.csv", spec->out_dir);
    snprintf(png_path, sizeof(png_path), "%s/eval.png", spec->out_dir);
    report = fopen(report_path, "w");
    csv = fopen(csv_path, "w");
    assert(report && csv);
    bench_replay(cfg, spec, replay, &collision, &offroad, &completion,
        &replay_score, &ade_lat, &ade_lon, &used, &skipped);
    bench_horizon(cfg, spec, horizon_pol, csv, &horizon_mean, &horizon_score);
    fclose(csv);
    sweep = bench_metric(spec, collision, offroad, completion, replay_score,
        ade_lat, ade_lon, horizon_mean, horizon_score);
    fprintf(report,
        "replay scenes %d skipped %d\n"
        "replay_collision %.6f\n"
        "replay_offroad %.6f\n"
        "replay_completion %.6f\n"
        "replay_score %.6f\n"
        "replay_ade_lat %.6f\n"
        "replay_ade_lon %.6f\n"
        "horizon_collision %.6f\n"
        "horizon_score %.6f\n",
        used, skipped, collision, offroad, completion, replay_score,
        ade_lat, ade_lon, horizon_mean, horizon_score);
    if (bench_metric_known(spec->sweep_metric)) {
        fprintf(report, "sweep %s %.6f\n", spec->sweep_metric, sweep);
    }
    fclose(report);
    bench_sidecar(ckpt, report_path);
    bench_plot(csv_path, png_path, spec->map_id);
    printf(
        "BENCHMARK replay_collision=%.6f replay_offroad=%.6f "
        "replay_completion=%.6f replay_score=%.6f "
        "replay_ade_lat=%.6f replay_ade_lon=%.6f "
        "horizon_collision=%.6f horizon_score=%.6f\n",
        collision, offroad, completion, replay_score,
        ade_lat, ade_lon, horizon_mean, horizon_score);
    fflush(stdout);
    fprintf(stderr, "wrote %s\n", report_path);
    dict_set_str(cfg, "data_dir", data_dir);
    dict_set(cfg, "map_id", map_id);
    dict_set(cfg, "num_agents", agents);
    dict_set(cfg, "episode_length", episode);
}
