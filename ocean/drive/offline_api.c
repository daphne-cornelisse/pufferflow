// clang -O2 -fPIC -shared -I. -Isrc -Iocean/drive -Ivendor \
//   -Iraylib-5.5_linux_amd64/include -DPLATFORM_DESKTOP -D_GNU_SOURCE \
//   -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable \
//   ocean/drive/offline_api.c vendor/cJSON.c \
//   raylib-5.5_linux_amd64/lib/libraylib.a \
//   -lGL -lm -lpthread -ldl -lrt -lX11 \
//   -o build/libdrive_offline.so

#include "drive.h"
#include "offline.h"

#include <math.h>

enum { PF_SIM = 0, PF_SCENARIO = 1 };

typedef struct PfHandle {
    int kind;
    Offline off;
    Dict cfg;
    char data_dir[1024];
    float* obs;
    float* act;
    float* rew;
    float* term;
} PfHandle;

static void pf_fill(PfHandle* h, const char* data_dir, int map_id, int num_agents,
        int episode_length, float goal_min, float goal_max) {
    memset(&h->cfg, 0, sizeof(h->cfg));
    h->cfg.name = "drive";
    snprintf(h->data_dir, sizeof(h->data_dir), "%s", data_dir);
    dict_set(&h->cfg, "selected_agent_idx", 0);
    dict_set(&h->cfg, "reward_vehicle_collision", -1.0);
    dict_set(&h->cfg, "reward_offroad_collision", -1.0);
    dict_set(&h->cfg, "reward_goal", 1.0);
    dict_set(&h->cfg, "reward_jerk", -0.0003);
    // Waymo offline reward stays the logged task.
    dict_set(&h->cfg, "target_speed", OFFLINE_TARGET_SPEED_KMH);
    dict_set(&h->cfg, "speed_band", SPEED_BAND_KMH);
    dict_set_str(&h->cfg, "data_dir", h->data_dir);
    dict_set(&h->cfg, "map_id", map_id);
    dict_set(&h->cfg, "num_agents", num_agents);
    dict_set(&h->cfg, "goal_distance_min", goal_min);
    dict_set(&h->cfg, "goal_distance_max", goal_max);
    dict_set(&h->cfg, "episode_length", episode_length);
}

static void pf_bind(PfHandle* h) {
    int n = h->off.drive.num_agents;
    assert(n >= 1 && n <= DRIVE_AGENT_MAX);
    h->obs = (float*)calloc((size_t)n * OBS_SIZE, sizeof(float));
    h->act = (float*)calloc((size_t)n * NUM_ATNS, sizeof(float));
    h->rew = (float*)calloc((size_t)n, sizeof(float));
    h->term = (float*)calloc((size_t)n, sizeof(float));
    assert(h->obs && h->act && h->rew && h->term);
    for (int i = 0; i < n; i++) {
        h->off.drive.agents[i].observations = h->obs + (size_t)i * OBS_SIZE;
        h->off.drive.agents[i].actions = h->act + (size_t)i * NUM_ATNS;
        h->off.drive.agents[i].rewards = h->rew + i;
        h->off.drive.agents[i].terminals = h->term + i;
    }
}

PfHandle* pf_create_sim(const char* data_dir, int map_id, int num_agents,
        int episode_length, float goal_min, float goal_max) {
    PfHandle* h = (PfHandle*)calloc(1, sizeof(PfHandle));
    assert(h);
    h->kind = PF_SIM;
    pf_fill(h, data_dir, map_id, num_agents, episode_length, goal_min, goal_max);
    puf_init(&h->off.drive, &h->cfg);
    pf_bind(h);
    return h;
}

PfHandle* pf_create_scenario(const char* data_dir, int map_id,
        const char* json_path) {
    PfHandle* h = (PfHandle*)calloc(1, sizeof(PfHandle));
    assert(h);
    h->kind = PF_SCENARIO;
    offline_parse(&h->off, json_path);
    pf_fill(h, data_dir, map_id, h->off.log_n, 100000, 5.0f, 30.0f);
    // The labeler places every track, including ones invalid at frame 0.
    dict_set(&h->cfg, "replay_invalid", 0);
    puf_init(&h->off.drive, &h->cfg);
    pf_bind(h);
    h->off.mean_x = h->off.drive.world_mean_x;
    h->off.mean_y = h->off.drive.world_mean_y;
    offline_shift_logs(&h->off);
    for (int i = 0; i < h->off.log_n; i++) {
        offline_label_agent(&h->off.logs[i]);
    }
    offline_mark_offroad(&h->off);
    offline_clear_episode(&h->off);
    offline_place(&h->off, 0);
    compute_observations(&h->off.drive);
    return h;
}

void pf_close(PfHandle* h) {
    if (!h) {
        return;
    }
    free(h->off.logs);
    h->off.logs = NULL;
    free(h->off.drive.map_name);
    h->off.drive.map_name = NULL;
    puf_close(&h->off.drive);
    free(h->obs);
    free(h->act);
    free(h->rew);
    free(h->term);
    if (h->cfg.items) {
        for (int i = 0; i < h->cfg.size; i++) {
            free(h->cfg.items[i].str);
            free(h->cfg.items[i].values);
        }
        free(h->cfg.items);
    }
    free(h);
}

void pf_reset(PfHandle* h, unsigned int seed) {
    if (h->kind == PF_SIM) {
        offline_reset_sim(&h->off.drive, seed);
        return;
    }
    (void)seed;
    offline_reset_scenario(&h->off);
}

void pf_step(PfHandle* h) {
    assert(h->kind == PF_SIM);
    offline_step_sim(&h->off.drive);
}

void pf_integrate(PfHandle* h, int stop_at_goal) {
    offline_integrate(&h->off.drive, stop_at_goal);
}

void pf_recompute(PfHandle* h) {
    h->off.drive.bins_ready = 0;
    compute_observations(&h->off.drive);
}

void pf_place(PfHandle* h, int frame) {
    assert(h->kind == PF_SCENARIO);
    offline_place(&h->off, frame);
    h->off.drive.timestep = frame;
}

void pf_replay(PfHandle* h, int controlled) {
    assert(h->kind == PF_SCENARIO);
    offline_replay_step(&h->off, controlled);
}

void pf_spawn_due(PfHandle* h) {
    assert(h->kind == PF_SCENARIO);
    offline_spawn_due(&h->off);
}

void pf_set_timestep(PfHandle* h, int timestep) {
    h->off.drive.timestep = timestep;
}

void pf_collect(PfHandle* h, int teleport, float* obs, int* acts,
        unsigned char* valid, float* ade) {
    assert(h->kind == PF_SCENARIO);
    offline_collect(&h->off, teleport, obs, acts, valid, ade);
}

int pf_n(PfHandle* h) {
    return h->off.drive.num_agents;
}

int pf_obs_size(PfHandle* h) {
    (void)h;
    return OBS_SIZE;
}

int pf_traj(void) {
    return OFFLINE_TRAJ;
}

int pf_log_traj(PfHandle* h) {
    return h->off.traj;
}

float* pf_obs(PfHandle* h) {
    return h->obs;
}

void pf_set_action(PfHandle* h, int i, float accel_bin, float steer_bin) {
    assert(i >= 0 && i < h->off.drive.num_agents);
    h->off.drive.agents[i].actions[0] = accel_bin;
    h->off.drive.agents[i].actions[1] = steer_bin;
}

void pf_set_actions(PfHandle* h, const float* actions) {
    for (int i = 0; i < h->off.drive.num_agents; i++) {
        h->off.drive.agents[i].actions[0] = actions[(size_t)i * NUM_ATNS];
        h->off.drive.agents[i].actions[1] = actions[(size_t)i * NUM_ATNS + 1];
    }
}

// Re-bin logged commands. Does not change which agents are fit.
void pf_relabel_grid(PfHandle* h, float amin, float amax, int na,
        float smin, float smax, int ns) {
    assert(h->kind == PF_SCENARIO);
    offline_relabel_grid(&h->off, amin, amax, na, smin, smax, ns);
}

// Float command. Later integrates of this handle use the value path.
void pf_set_action_value(PfHandle* h, int i, float accel, float steer) {
    assert(i >= 0 && i < h->off.drive.num_agents);
    h->off.drive.use_action_values = 1;
    h->off.drive.action_accel[i] = accel;
    h->off.drive.action_steer[i] = steer;
}

float pf_reward(PfHandle* h, int i) {
    return h->off.drive.agents[i].rewards[0];
}

int pf_timestep(PfHandle* h) {
    return h->off.drive.timestep;
}

// Pose only. Type, size, and goal stay as placed.
void pf_set_pose(PfHandle* h, int i, float x, float y, float heading) {
    Entity* e = &h->off.drive.entities[i];
    assert(i >= 0 && i < h->off.drive.num_agents);
    h->off.drive.bins_ready = 0;
    if (x == INVALID_POSITION || y == INVALID_POSITION) {
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
        e->valid = 0;
        e->collision_state = NO_COLLISION;
        e->offroad_state = 0;
        return;
    }
    e->x = x;
    e->y = y;
    e->heading = heading;
    e->heading_x = cosf(heading);
    e->heading_y = sinf(heading);
    e->valid = 1;
}

void pf_contact(PfHandle* h, int i, int* collision, int* offroad) {
    Entity* e = &h->off.drive.entities[i];
    assert(i >= 0 && i < h->off.drive.num_agents);
    *collision = e->collision_state;
    *offroad = e->offroad_state;
}

void pf_check_contacts(PfHandle* h) {
    Drive* env = &h->off.drive;
    env->bins_ready = 0;
    for (int i = 0; i < env->active_agent_count; i++) {
        env->entities[i].collision_state = NO_COLLISION;
        collision_check(env, i);
    }
}

void pf_agent(PfHandle* h, int i, float* out) {
    Entity* e = &h->off.drive.entities[i];
    out[0] = e->x;
    out[1] = e->y;
    out[2] = e->heading;
    out[3] = e->vx;
    out[4] = e->vy;
    out[5] = e->goal_position_x;
    out[6] = e->goal_position_y;
    out[7] = e->width;
    out[8] = e->length;
    out[9] = (float)e->collision_state;
    out[10] = (e->x == INVALID_POSITION || e->y == INVALID_POSITION) ? 0.0f : 1.0f;
}

void pf_agents(PfHandle* h, float* out) {
    for (int i = 0; i < h->off.drive.num_agents; i++) {
        pf_agent(h, i, out + (size_t)i * 11);
    }
}

void pf_censor_invalid(PfHandle* h, int frame) {
    assert(h->kind == PF_SCENARIO);
    assert(frame >= 0 && frame < h->off.traj);
    for (int i = 0; i < h->off.log_n; i++) {
        if (h->off.logs[i].valid[frame]) {
            continue;
        }
        Entity* e = &h->off.drive.entities[i];
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
        e->valid = 0;
        e->vx = 0.0f;
        e->vy = 0.0f;
        e->collision_state = NO_COLLISION;
        e->offroad_state = 0;
    }
    h->off.drive.bins_ready = 0;
}

void pf_set_agent(PfHandle* h, int i, float x, float y, float heading,
        float vx, float vy, float width, float length, float gx, float gy) {
    Entity* e = &h->off.drive.entities[i];
    e->type = VEHICLE;
    e->height = CAR_HEIGHT;
    e->lane_node = -1;
    e->width = width;
    e->length = length;
    e->goal_position_x = gx;
    e->goal_position_y = gy;
    e->goal_position_z = 0.0f;
    e->collision_state = NO_COLLISION;
    e->reached_goal = 0;
    e->collided_before_goal = 0;
    e->collided_this_goal = 0;
    h->off.drive.bins_ready = 0;
    if (x == INVALID_POSITION || y == INVALID_POSITION) {
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
        e->valid = 0;
        e->vx = 0.0f;
        e->vy = 0.0f;
        return;
    }
    e->x = x;
    e->y = y;
    e->z = 0.0f;
    e->heading = heading;
    e->heading_x = cosf(heading);
    e->heading_y = sinf(heading);
    e->vx = vx;
    e->vy = vy;
    e->vz = 0.0f;
    e->valid = 1;
}

void pf_set_goal(PfHandle* h, int i, float gx, float gy) {
    h->off.drive.entities[i].goal_position_x = gx;
    h->off.drive.entities[i].goal_position_y = gy;
}

void pf_mean(PfHandle* h, float* x, float* y) {
    *x = h->off.drive.world_mean_x;
    *y = h->off.drive.world_mean_y;
}

int pf_log_n(PfHandle* h) {
    return h->off.log_n;
}

int pf_sdc(PfHandle* h) {
    return h->off.sdc;
}

int pf_base_unfit(PfHandle* h, int i) {
    return h->off.logs[i].base_unfit;
}

// Full-scene macro evaluation performs its eligibility filtering before the
// scenario is loaded.  It deliberately does not drop snowy-rut trajectories
// merely because their boxes touch a recovered calibration edge.
void pf_allow_all_logs(PfHandle* h) {
    assert(h->kind == PF_SCENARIO);
    for (int i = 0; i < h->off.log_n; i++) {
        h->off.logs[i].base_unfit = 0;
    }
}

int pf_log_valid(PfHandle* h, int i, int t) {
    return h->off.logs[i].valid[t];
}

void pf_log_xy(PfHandle* h, int i, int t, float* x, float* y) {
    *x = h->off.logs[i].x[t];
    *y = h->off.logs[i].y[t];
}

void pf_log_bins(PfHandle* h, int i, int t, int* accel_bin, int* steer_bin) {
    *accel_bin = h->off.logs[i].accel_bin[t];
    *steer_bin = h->off.logs[i].steer_bin[t];
}

void pf_outcome(PfHandle* h, int i, float* ret, float* ade, float* goal_dist,
        int* collided, int* offroad, int* reached, int* collided_before) {
    OfflineAgent* a = &h->off.logs[i];
    Entity* e = &h->off.drive.entities[i];
    *ret = a->ret;
    *ade = a->ade_n > 0 ? a->ade_sum / (float)a->ade_n : 0.0f;
    *goal_dist = a->goal_dist;
    *collided = a->collided;
    *offroad = a->offroad;
    *reached = a->reached || e->reached_goal;
    *collided_before = e->collided_before_goal;
}

int pf_n_accel(void) {
    init_action_space();
    return NUM_ACCEL_BINS;
}

int pf_n_steer(void) {
    init_action_space();
    return NUM_STEER_BINS;
}

int pf_coast_accel(void) {
    return ACCEL_COAST_BIN;
}

int pf_coast_steer(void) {
    return STEER_COAST_BIN;
}

float pf_invalid(void) {
    return INVALID_POSITION;
}

int pf_accel_bin(float v) {
    init_action_space();
    return offline_quantize_bin(v, ACCEL_MIN, ACCEL_MAX, NUM_ACCEL_BINS);
}

int pf_steer_bin(float v) {
    init_action_space();
    return offline_quantize_bin(v, STEER_MIN, STEER_MAX, NUM_STEER_BINS);
}

float pf_accel_value(int bin) {
    init_action_space();
    assert(bin >= 0 && bin < NUM_ACCEL_BINS);
    return ACCELERATION_VALUES[bin];
}

float pf_steer_value(int bin) {
    init_action_space();
    assert(bin >= 0 && bin < NUM_STEER_BINS);
    return STEERING_VALUES[bin];
}

void pf_road_counts(PfHandle* h, int* nroads, int* npts) {
    Drive* d = &h->off.drive;
    int n = 0;
    int p = 0;
    for (int i = d->active_agent_count; i < d->num_entities; i++) {
        Entity* e = &d->entities[i];
        if (e->type < ROAD_LANE || !e->pts_x) {
            continue;
        }
        n++;
        p += e->array_size;
    }
    *nroads = n;
    *npts = p;
}

void pf_roads(PfHandle* h, int* offsets, int* types, float* xs, float* ys) {
    Drive* d = &h->off.drive;
    int r = 0;
    int p = 0;
    offsets[0] = 0;
    for (int i = d->active_agent_count; i < d->num_entities; i++) {
        Entity* e = &d->entities[i];
        if (e->type < ROAD_LANE || !e->pts_x) {
            continue;
        }
        types[r] = e->type;
        for (int j = 0; j < e->array_size; j++) {
            xs[p] = e->pts_x[j];
            ys[p] = e->pts_y[j];
            p++;
        }
        r++;
        offsets[r] = p;
    }
}

// Agent-major, span frames. A null xs skips the store.
static void trace_store(float* xs, float* ys, unsigned char* on,
        int i, int t, int span, const Entity* e) {
    if (!xs) {
        return;
    }
    int k = i * span + t;
    if (e->x == INVALID_POSITION) {
        xs[k] = 0.0f;
        ys[k] = 0.0f;
        on[k] = 0;
        return;
    }
    xs[k] = e->x;
    ys[k] = e->y;
    on[k] = 1;
}

static void trace_frame(float* xs, float* ys, unsigned char* on,
        int n, int t, int span, const Drive* drive) {
    if (!xs) {
        return;
    }
    for (int i = 0; i < n; i++) {
        trace_store(xs, ys, on, i, t, span, &drive->entities[i]);
    }
}

// Frame 0 matches the log by placement and is left out of the ADE.
static void quantized_replay(PfHandle* h, float amin, float amax, int na,
        float smin, float smax, int ns, float* ade_mean, int* n_agents,
        float* clip_accel, float* clip_steer,
        float* trace_x, float* trace_y, unsigned char* trace_on) {
    assert(h && h->kind == PF_SCENARIO);
    assert((na == 0 || na >= 2) && (ns == 0 || ns >= 2));
    assert(amax > amin && smax > smin);
    Offline* env = &h->off;
    Drive* drive = &env->drive;
    int n = env->log_n;
    assert(n <= DRIVE_AGENT_MAX);
    offline_reset_scenario(env);
    double agent_err[DRIVE_AGENT_MAX];
    int agent_n[DRIVE_AGENT_MAX];
    memset(agent_err, 0, sizeof(agent_err));
    memset(agent_n, 0, sizeof(agent_n));
    int commands = 0;
    int clipped_a = 0;
    int clipped_s = 0;
    int span = env->traj;
    drive->use_action_values = 1;
    for (int t = 0; t < span - 1; t++) {
        drive->timestep = t;
        offline_spawn_due(env);
        trace_frame(trace_x, trace_y, trace_on, n, t, span, drive);
        for (int i = 0; i < n; i++) {
            Entity* e = &drive->entities[i];
            OfflineAgent* a = &env->logs[i];
            float accel = 0.0f;
            float steer = 0.0f;
            int labeled = !a->base_unfit && a->valid[t] && a->valid[t + 1]
                && e->x != INVALID_POSITION;
            if (labeled) {
                float raw_a = 0.0f;
                float raw_s = 0.0f;
                offline_inverse_raw(a->x[t], a->y[t], a->heading[t],
                    a->vx[t], a->vy[t], a->x[t + 1], a->y[t + 1],
                    a->heading[t + 1], a->vx[t + 1], a->vy[t + 1],
                    a->length, &raw_a, &raw_s);
                commands++;
                if (raw_a < amin || raw_a > amax) {
                    clipped_a++;
                }
                if (raw_s < smin || raw_s > smax) {
                    clipped_s++;
                }
                if (na >= 2) {
                    accel = offline_quantize(raw_a, amin, amax, na);
                } else {
                    accel = raw_a;
                    if (accel < amin) {
                        accel = amin;
                    }
                    if (accel > amax) {
                        accel = amax;
                    }
                }
                if (ns >= 2) {
                    steer = offline_quantize(raw_s, smin, smax, ns);
                } else {
                    steer = raw_s;
                    if (steer < smin) {
                        steer = smin;
                    }
                    if (steer > smax) {
                        steer = smax;
                    }
                }
            }
            drive->action_accel[i] = accel;
            drive->action_steer[i] = steer;
            e->reached_goal = 0;
        }
        offline_integrate(drive, 0);
        for (int i = 0; i < n; i++) {
            OfflineAgent* a = &env->logs[i];
            Entity* e = &drive->entities[i];
            if (a->base_unfit || !a->valid[t + 1] || e->x == INVALID_POSITION) {
                continue;
            }
            float dx = e->x - a->x[t + 1];
            float dy = e->y - a->y[t + 1];
            agent_err[i] += sqrtf(dx * dx + dy * dy);
            agent_n[i] += 1;
        }
    }
    trace_frame(trace_x, trace_y, trace_on, n, span - 1, span, drive);
    drive->use_action_values = 0;
    int used = 0;
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        if (agent_n[i] <= 0) {
            continue;
        }
        mean += agent_err[i] / (double)agent_n[i];
        used++;
    }
    *ade_mean = used > 0 ? (float)(mean / (double)used) : 0.0f;
    *n_agents = used;
    *clip_accel = commands > 0 ? (float)clipped_a / (float)commands : 0.0f;
    *clip_steer = commands > 0 ? (float)clipped_s / (float)commands : 0.0f;
}

void pf_quantized_ade(PfHandle* h, float amin, float amax, int na,
        float smin, float smax, int ns, float* ade_mean, int* n_agents,
        float* clip_accel, float* clip_steer) {
    quantized_replay(h, amin, amax, na, smin, smax, ns,
        ade_mean, n_agents, clip_accel, clip_steer, NULL, NULL, NULL);
}

void pf_quantized_trace(PfHandle* h, float amin, float amax, int na,
        float smin, float smax, int ns,
        float* xs, float* ys, unsigned char* on) {
    float ade = 0.0f;
    int n_agents = 0;
    float clip_a = 0.0f;
    float clip_s = 0.0f;
    assert(xs && ys && on);
    quantized_replay(h, amin, amax, na, smin, smax, ns,
        &ade, &n_agents, &clip_a, &clip_s, xs, ys, on);
}

// n_dx == 0 keeps the clipped float. Otherwise each component is snapped.
static void delta_replay(PfHandle* h, int n_dx, int n_dy, int n_yaw,
        float* ade_mean, int* n_agents,
        float* clip_dx, float* clip_dy, float* clip_yaw,
        float* trace_x, float* trace_y, unsigned char* trace_on) {
    assert(h && h->kind == PF_SCENARIO);
    assert(n_dx == 0 || (n_dx >= 2 && n_dy >= 2 && n_yaw >= 2));
    Offline* env = &h->off;
    Drive* drive = &env->drive;
    int n = env->log_n;
    assert(n <= DRIVE_AGENT_MAX);
    offline_reset_scenario(env);
    double agent_err[DRIVE_AGENT_MAX];
    int agent_n[DRIVE_AGENT_MAX];
    float prev_dx[DRIVE_AGENT_MAX];
    int ready[DRIVE_AGENT_MAX];
    memset(agent_err, 0, sizeof(agent_err));
    memset(agent_n, 0, sizeof(agent_n));
    memset(prev_dx, 0, sizeof(prev_dx));
    memset(ready, 0, sizeof(ready));
    int commands = 0;
    int clipped_x = 0;
    int clipped_y = 0;
    int clipped_yaw = 0;
    int span = env->traj;
    for (int t = 0; t < span - 1; t++) {
        drive->timestep = t;
        offline_spawn_due(env);
        trace_frame(trace_x, trace_y, trace_on, n, t, span, drive);
        for (int i = 0; i < n; i++) {
            Entity* e = &drive->entities[i];
            OfflineAgent* a = &env->logs[i];
            if (a->base_unfit || e->x == INVALID_POSITION) {
                continue;
            }
            if (!ready[i]) {
                float speed_init = e->vx * e->heading_x + e->vy * e->heading_y;
                prev_dx[i] = speed_init * SIM_DT;
                ready[i] = 1;
            }
            float dx = 0.0f;
            float dy = 0.0f;
            float dyaw = 0.0f;
            int labeled = a->valid[t] && a->valid[t + 1];
            if (labeled) {
                offline_delta_inverse(a->x[t], a->y[t], a->heading[t],
                    a->x[t + 1], a->y[t + 1], a->heading[t + 1],
                    &dx, &dy, &dyaw);
                commands++;
                if (dx < -DELTA_MAX_DX || dx > DELTA_MAX_DX) {
                    clipped_x++;
                }
                if (dy < -DELTA_MAX_DY || dy > DELTA_MAX_DY) {
                    clipped_y++;
                }
                if (dyaw < -DELTA_MAX_DYAW || dyaw > DELTA_MAX_DYAW) {
                    clipped_yaw++;
                }
                if (n_dx > 0) {
                    dx = offline_quantize(dx, -DELTA_MAX_DX, DELTA_MAX_DX, n_dx);
                    dy = offline_quantize(dy, -DELTA_MAX_DY, DELTA_MAX_DY, n_dy);
                    dyaw = offline_quantize(dyaw, -DELTA_MAX_DYAW, DELTA_MAX_DYAW, n_yaw);
                }
            }
            offline_delta_step(e, dx, dy, dyaw, &prev_dx[i]);
            e->reached_goal = 0;
        }
        for (int i = 0; i < n; i++) {
            OfflineAgent* a = &env->logs[i];
            Entity* e = &drive->entities[i];
            if (a->base_unfit || !a->valid[t + 1] || e->x == INVALID_POSITION) {
                continue;
            }
            float ex = e->x - a->x[t + 1];
            float ey = e->y - a->y[t + 1];
            agent_err[i] += sqrtf(ex * ex + ey * ey);
            agent_n[i] += 1;
        }
    }
    trace_frame(trace_x, trace_y, trace_on, n, span - 1, span, drive);
    int used = 0;
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        if (agent_n[i] <= 0) {
            continue;
        }
        mean += agent_err[i] / (double)agent_n[i];
        used++;
    }
    *ade_mean = used > 0 ? (float)(mean / (double)used) : 0.0f;
    *n_agents = used;
    *clip_dx = commands > 0 ? (float)clipped_x / (float)commands : 0.0f;
    *clip_dy = commands > 0 ? (float)clipped_y / (float)commands : 0.0f;
    *clip_yaw = commands > 0 ? (float)clipped_yaw / (float)commands : 0.0f;
}

void pf_delta_ade(PfHandle* h, int n_dx, int n_dy, int n_yaw,
        float* ade_mean, int* n_agents,
        float* clip_dx, float* clip_dy, float* clip_yaw) {
    delta_replay(h, n_dx, n_dy, n_yaw, ade_mean, n_agents,
        clip_dx, clip_dy, clip_yaw, NULL, NULL, NULL);
}

void pf_delta_trace(PfHandle* h, int n_dx, int n_dy, int n_yaw,
        float* xs, float* ys, unsigned char* on) {
    float ade = 0.0f;
    int n_agents = 0;
    float clip_x = 0.0f;
    float clip_y = 0.0f;
    float clip_yaw = 0.0f;
    assert(xs && ys && on);
    delta_replay(h, n_dx, n_dy, n_yaw, &ade, &n_agents,
        &clip_x, &clip_y, &clip_yaw, xs, ys, on);
}
