// Shared library for the offline Waymo env and the drive.h sim.
// Build from the repo root:
//   clang -O2 -fPIC -shared -I. -Isrc -Iocean/drive -Ivendor \
//     -Iraylib-5.5_linux_amd64/include -DPLATFORM_DESKTOP -D_GNU_SOURCE \
//     -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable \
//     ocean/drive/offline_api.c vendor/cJSON.c \
//     raylib-5.5_linux_amd64/lib/libraylib.a \
//     -lGL -lm -lpthread -ldl -lrt -lX11 \
//     -o build/libdrive_offline.so
//
// Sim handles call puf_init / puf_reset / puf_step. Scenario handles load one
// WOMD json on top of that map and never call puf_reset.

#include "drive.h"
#include "offline.h"

#include <math.h>

enum { PF_SIM = 0, PF_SCENARIO = 1 };

typedef struct PfHandle {
    int kind;
    Offline off;
    Dict cfg;
    char data_dir[1024];
    char physics[32];
    float* obs;
    float* act;
    float* rew;
    float* term;
} PfHandle;

static void pf_fill(PfHandle* h, const char* data_dir, int map_id, int num_agents,
        int episode_length, float goal_min, float goal_max, const char* physics) {
    memset(&h->cfg, 0, sizeof(h->cfg));
    h->cfg.name = "drive";
    snprintf(h->data_dir, sizeof(h->data_dir), "%s", data_dir);
    snprintf(h->physics, sizeof(h->physics), "%s", physics);
    dict_set(&h->cfg, "selected_agent_idx", 0);
    dict_set_str(&h->cfg, "collision_physics", h->physics);
    dict_set(&h->cfg, "reward_vehicle_collision", -1.0);
    dict_set(&h->cfg, "reward_offroad_collision", -1.0);
    dict_set(&h->cfg, "reward_goal", 1.0);
    dict_set(&h->cfg, "reward_jerk", -0.0003);
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
        int episode_length, float goal_min, float goal_max, const char* physics) {
    PfHandle* h = (PfHandle*)calloc(1, sizeof(PfHandle));
    assert(h);
    h->kind = PF_SIM;
    pf_fill(h, data_dir, map_id, num_agents, episode_length, goal_min, goal_max, physics);
    puf_init(&h->off.drive, &h->cfg);
    pf_bind(h);
    return h;
}

PfHandle* pf_create_scenario(const char* data_dir, int map_id,
        const char* json_path, const char* physics) {
    PfHandle* h = (PfHandle*)calloc(1, sizeof(PfHandle));
    assert(h);
    h->kind = PF_SCENARIO;
    offline_parse(&h->off, json_path);
    pf_fill(h, data_dir, map_id, h->off.log_n, 100000, 5.0f, 30.0f, physics);
    puf_init(&h->off.drive, &h->cfg);
    pf_bind(h);
    h->off.mean_x = h->off.drive.world_mean_x;
    h->off.mean_y = h->off.drive.world_mean_y;
    offline_shift_logs(&h->off);
    for (int i = 0; i < h->off.log_n; i++) {
        offline_label_agent(&h->off.logs[i]);
    }
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

float* pf_obs(PfHandle* h) {
    return h->obs;
}

void pf_set_action(PfHandle* h, int i, float accel_bin, float steer_bin) {
    assert(i >= 0 && i < h->off.drive.num_agents);
    h->off.drive.agents[i].actions[0] = accel_bin;
    h->off.drive.agents[i].actions[1] = steer_bin;
}

float pf_reward(PfHandle* h, int i) {
    return h->off.drive.agents[i].rewards[0];
}

int pf_timestep(PfHandle* h) {
    return h->off.drive.timestep;
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

void pf_set_agent(PfHandle* h, int i, float x, float y, float heading,
        float vx, float vy, float width, float length, float gx, float gy) {
    Entity* e = &h->off.drive.entities[i];
    e->type = VEHICLE;
    e->active_agent = 1;
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
    return offline_nearest(v, ACCELERATION_VALUES, NUM_ACCEL_BINS);
}

int pf_steer_bin(float v) {
    init_action_space();
    return offline_nearest(v, STEERING_VALUES, NUM_STEER_BINS);
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
