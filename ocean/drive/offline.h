// Waymo logs on the drive.h observation. Included after drive.h.

#pragma once

#include <math.h>
#include "cJSON.h"

#define OFFLINE_TRAJ 91
#define OFFLINE_ADE_MAX 2.0f
#define OFFLINE_MIN_VALID 10
// Parked logs jitter by centimeters. A track inside this radius never moved.
#define OFFLINE_STILL_M 1.0f

typedef struct OfflineAgent {
    int id;
    int is_sdc;
    int base_unfit;
    int valid[OFFLINE_TRAJ];
    float x[OFFLINE_TRAJ];
    float y[OFFLINE_TRAJ];
    float heading[OFFLINE_TRAJ];
    float vx[OFFLINE_TRAJ];
    float vy[OFFLINE_TRAJ];
    float accel[OFFLINE_TRAJ];
    float steer[OFFLINE_TRAJ];
    int accel_bin[OFFLINE_TRAJ];
    int steer_bin[OFFLINE_TRAJ];
    float width;
    float length;
    float goal_x;
    float goal_y;
    float ret;
    float ade_sum;
    int ade_n;
    int collided;
    int offroad;
    int reached;
    float goal_dist;
} OfflineAgent;

typedef struct Offline {
    Drive drive;
    OfflineAgent* logs;
    int log_n;
    int sdc;
    float mean_x;
    float mean_y;
} Offline;

// Signed-speed inverse of bicycle_integrate. Steer is atan2(L Δθ, v1 Δt).
// A near-zero end speed forces steer 0. Displacement heading is not a command.
static void offline_inverse_raw(float x0, float y0, float h0, float vx0, float vy0,
        float x1, float y1, float h1, float vx1, float vy1, float length,
        float* accel, float* steer) {
    float v0 = vx0 * cosf(h0) + vy0 * sinf(h0);
    float v1 = vx1 * cosf(h1) + vy1 * sinf(h1);
    float dth = normalize_heading(h1 - h0);
    (void)x0;
    (void)y0;
    (void)x1;
    (void)y1;
    *accel = (v1 - v0) / SIM_DT;
    if (fabsf(v1) < 1.0e-3f) {
        *steer = 0.0f;
    } else {
        *steer = atan2f(dth * bicycle_wheelbase(length), v1 * SIM_DT);
    }
}

static void offline_inverse(float x0, float y0, float h0, float vx0, float vy0,
        float x1, float y1, float h1, float vx1, float vy1, float length,
        float* accel, float* steer) {
    offline_inverse_raw(x0, y0, h0, vx0, vy0, x1, y1, h1, vx1, vy1, length,
        accel, steer);
    if (*accel > ACCEL_MAX) {
        *accel = ACCEL_MAX;
    }
    if (*accel < ACCEL_MIN) {
        *accel = ACCEL_MIN;
    }
    if (*steer > STEER_MAX) {
        *steer = STEER_MAX;
    }
    if (*steer < STEER_MIN) {
        *steer = STEER_MIN;
    }
}

// Nearest bin. Half steps go toward +inf.
static int offline_quantize_bin(float v, float lo, float hi, int n) {
    if (v < lo) {
        v = lo;
    }
    if (v > hi) {
        v = hi;
    }
    float step = (hi - lo) / (float)(n - 1);
    int bin = (int)floorf((v - lo) / step + 0.5f);
    if (bin < 0) {
        bin = 0;
    }
    if (bin >= n) {
        bin = n - 1;
    }
    return bin;
}

// Odd counts on a symmetric range have a center bin. That command is 0.
static float offline_bin_center(float lo, float hi, int n, int bin) {
    if ((n % 2) == 1 && bin == n / 2 && lo == -hi) {
        return 0.0f;
    }
    float step = (hi - lo) / (float)(n - 1);
    return lo + (float)bin * step;
}

static float offline_quantize(float v, float lo, float hi, int n) {
    return offline_bin_center(lo, hi, n, offline_quantize_bin(v, lo, hi, n));
}

// Local-frame step. Longitudinal accel is clipped to ±8 m/s², then
// |dy| ≤ |dx| tan(0.7).
#define DELTA_MAX_DX 3.5f
#define DELTA_MAX_DY 0.1f
#define DELTA_MAX_DYAW ((float)M_PI / 6.0f)
#define DELTA_A_LONG_MAX 8.0f
#define DELTA_MAX_STEER 0.7f
#define DELTA_BINS_DX 51
#define DELTA_BINS_DY 51
#define DELTA_BINS_YAW 127

static float offline_wrap_pi(float heading) {
    if (heading > (float)M_PI) {
        heading -= 2.0f * (float)M_PI;
    }
    if (heading < -(float)M_PI) {
        heading += 2.0f * (float)M_PI;
    }
    return heading;
}

static void offline_delta_inverse(float x0, float y0, float h0,
        float x1, float y1, float h1, float* dx, float* dy, float* dyaw) {
    float c = cosf(h0);
    float s = sinf(h0);
    float wx = x1 - x0;
    float wy = y1 - y0;
    *dx = c * wx + s * wy;
    *dy = -s * wx + c * wy;
    *dyaw = offline_wrap_pi(h1 - h0);
}

// prev_dx is the last longitudinal step, seeded from the logged speed.
static void offline_delta_step(Entity* e, float action_dx, float action_dy,
        float action_dyaw, float* prev_dx) {
    if (action_dx > DELTA_MAX_DX) {
        action_dx = DELTA_MAX_DX;
    }
    if (action_dx < -DELTA_MAX_DX) {
        action_dx = -DELTA_MAX_DX;
    }
    if (action_dy > DELTA_MAX_DY) {
        action_dy = DELTA_MAX_DY;
    }
    if (action_dy < -DELTA_MAX_DY) {
        action_dy = -DELTA_MAX_DY;
    }
    action_dyaw = offline_wrap_pi(action_dyaw);

    float prev_speed = (*prev_dx) / SIM_DT;
    float speed_lo = prev_speed - DELTA_A_LONG_MAX * SIM_DT;
    float speed_hi = prev_speed + DELTA_A_LONG_MAX * SIM_DT;
    float new_speed = action_dx / SIM_DT;
    if (new_speed < speed_lo) {
        new_speed = speed_lo;
    }
    if (new_speed > speed_hi) {
        new_speed = speed_hi;
    }
    action_dx = new_speed * SIM_DT;

    float max_dy = fabsf(action_dx) * tanf(DELTA_MAX_STEER);
    if (action_dy > max_dy) {
        action_dy = max_dy;
    }
    if (action_dy < -max_dy) {
        action_dy = -max_dy;
    }

    *prev_dx = action_dx;
    float cos_h = cosf(e->heading);
    float sin_h = sinf(e->heading);
    float global_dx = cos_h * action_dx - sin_h * action_dy;
    float global_dy = sin_h * action_dx + cos_h * action_dy;
    e->x += global_dx;
    e->y += global_dy;
    e->heading = offline_wrap_pi(e->heading + action_dyaw);
    e->heading_x = cosf(e->heading);
    e->heading_y = sinf(e->heading);
    e->vx = global_dx / SIM_DT;
    e->vy = global_dy / SIM_DT;
}

static void offline_label_agent(OfflineAgent* a) {
    int count = 0;
    float x0 = 0.0f;
    float y0 = 0.0f;
    float reach = 0.0f;
    for (int t = 0; t < OFFLINE_TRAJ; t++) {
        a->accel[t] = -1.0f;
        a->steer[t] = -1.0f;
        a->accel_bin[t] = -1;
        a->steer_bin[t] = -1;
        if (!a->valid[t]) {
            continue;
        }
        if (count == 0) {
            x0 = a->x[t];
            y0 = a->y[t];
        } else {
            float dx = a->x[t] - x0;
            float dy = a->y[t] - y0;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist > reach) {
                reach = dist;
            }
        }
        count++;
        if (t + 1 >= OFFLINE_TRAJ || !a->valid[t + 1]) {
            continue;
        }
        float accel = 0.0f;
        float steer = 0.0f;
        int ab;
        int sb;
        offline_inverse(a->x[t], a->y[t], a->heading[t], a->vx[t], a->vy[t],
            a->x[t + 1], a->y[t + 1], a->heading[t + 1], a->vx[t + 1], a->vy[t + 1],
            a->length, &accel, &steer);
        ab = offline_quantize_bin(accel, ACCEL_MIN, ACCEL_MAX, NUM_ACCEL_BINS);
        sb = offline_quantize_bin(steer, STEER_MIN, STEER_MAX, NUM_STEER_BINS);
        a->accel[t] = offline_bin_center(ACCEL_MIN, ACCEL_MAX, NUM_ACCEL_BINS, ab);
        a->steer[t] = offline_bin_center(STEER_MIN, STEER_MAX, NUM_STEER_BINS, sb);
        a->accel_bin[t] = ab;
        a->steer_bin[t] = sb;
    }
    int still = reach < OFFLINE_STILL_M;
    a->base_unfit = (count < OFFLINE_MIN_VALID) || still;
}

// Re-bin the raw inverse. Leaves base_unfit alone. Stored commands are centers.
static void offline_relabel_grid(Offline* env, float amin, float amax, int na,
        float smin, float smax, int ns) {
    assert(na >= 2 && ns >= 2);
    assert(amax > amin && smax > smin);
    assert(na % 2 == 1 && ns % 2 == 1);
    for (int i = 0; i < env->log_n; i++) {
        OfflineAgent* a = &env->logs[i];
        for (int t = 0; t < OFFLINE_TRAJ; t++) {
            float raw_a = 0.0f;
            float raw_s = 0.0f;
            int ab;
            int sb;
            if (t + 1 >= OFFLINE_TRAJ || !a->valid[t] || !a->valid[t + 1]) {
                continue;
            }
            offline_inverse_raw(a->x[t], a->y[t], a->heading[t], a->vx[t], a->vy[t],
                a->x[t + 1], a->y[t + 1], a->heading[t + 1], a->vx[t + 1], a->vy[t + 1],
                a->length, &raw_a, &raw_s);
            ab = offline_quantize_bin(raw_a, amin, amax, na);
            sb = offline_quantize_bin(raw_s, smin, smax, ns);
            a->accel[t] = offline_bin_center(amin, amax, na, ab);
            a->steer[t] = offline_bin_center(smin, smax, ns, sb);
            a->accel_bin[t] = ab;
            a->steer_bin[t] = sb;
        }
    }
}

static int offline_jbool(cJSON* v) {
    if (!v) {
        return 0;
    }
    if (cJSON_IsBool(v)) {
        return cJSON_IsTrue(v);
    }
    if (cJSON_IsNumber(v)) {
        return v->valuedouble != 0.0;
    }
    return 0;
}

static double offline_jnum(cJSON* obj, const char* key) {
    cJSON* v = cJSON_GetObjectItemCaseSensitive(obj, key);
    assert(v && cJSON_IsNumber(v));
    return v->valuedouble;
}

static void offline_read_series(cJSON* arr, float* dst, int n, const char* key) {
    for (int t = 0; t < n; t++) {
        dst[t] = 0.0f;
    }
    if (!cJSON_IsArray(arr)) {
        return;
    }
    int t = 0;
    cJSON* item = NULL;
    cJSON_ArrayForEach(item, arr) {
        if (t >= n) {
            break;
        }
        if (key) {
            cJSON* f = cJSON_GetObjectItemCaseSensitive(item, key);
            dst[t] = f && cJSON_IsNumber(f) ? (float)f->valuedouble : 0.0f;
        } else {
            dst[t] = cJSON_IsNumber(item) ? (float)item->valuedouble : 0.0f;
        }
        t++;
    }
}

static void offline_shift_logs(Offline* env) {
    float mx = env->mean_x;
    float my = env->mean_y;
    for (int i = 0; i < env->log_n; i++) {
        OfflineAgent* a = &env->logs[i];
        a->goal_x -= mx;
        a->goal_y -= my;
        for (int t = 0; t < OFFLINE_TRAJ; t++) {
            a->x[t] -= mx;
            a->y[t] -= my;
        }
    }
}

// Vehicles only, in json order. sdc_track_index is an index into objects,
// not into the vehicle list. Centered after the drive world exists.
static void offline_parse(Offline* env, const char* json_path) {
    FILE* file = fopen(json_path, "rb");
    assert(file);
    assert(fseek(file, 0, SEEK_END) == 0);
    long nbytes = ftell(file);
    assert(nbytes > 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    char* text = (char*)malloc((size_t)nbytes + 1);
    assert(text);
    assert(fread(text, 1, (size_t)nbytes, file) == (size_t)nbytes);
    text[nbytes] = 0;
    fclose(file);
    cJSON* root = cJSON_Parse(text);
    free(text);
    assert(root);

    int sdc_track = 0;
    cJSON* meta = cJSON_GetObjectItemCaseSensitive(root, "metadata");
    if (meta) {
        cJSON* sdc = cJSON_GetObjectItemCaseSensitive(meta, "sdc_track_index");
        if (sdc && cJSON_IsNumber(sdc)) {
            sdc_track = sdc->valueint;
        }
    }

    cJSON* objects = cJSON_GetObjectItemCaseSensitive(root, "objects");
    assert(cJSON_IsArray(objects));
    int nobj = cJSON_GetArraySize(objects);
    int nveh = 0;
    for (int i = 0; i < nobj; i++) {
        cJSON* obj = cJSON_GetArrayItem(objects, i);
        cJSON* type = cJSON_GetObjectItemCaseSensitive(obj, "type");
        if (type && cJSON_IsString(type) && strcmp(type->valuestring, "vehicle") == 0) {
            nveh++;
        }
    }
    assert(nveh > 0);
    int keep = nveh;
    if (keep > DRIVE_AGENT_MAX) {
        keep = DRIVE_AGENT_MAX;
    }
    env->logs = (OfflineAgent*)calloc((size_t)keep, sizeof(OfflineAgent));
    assert(env->logs);
    env->log_n = 0;
    env->sdc = 0;
    int seen = 0;
    for (int i = 0; i < nobj; i++) {
        cJSON* obj = cJSON_GetArrayItem(objects, i);
        cJSON* type = cJSON_GetObjectItemCaseSensitive(obj, "type");
        if (!type || !cJSON_IsString(type) || strcmp(type->valuestring, "vehicle") != 0) {
            continue;
        }
        int is_sdc = i == sdc_track;
        // Past the cap, still take the sdc by replacing the last non-sdc.
        if (seen >= keep && !is_sdc) {
            seen++;
            continue;
        }
        int slot = env->log_n;
        if (seen >= keep && is_sdc) {
            slot = keep - 1;
            for (int k = 0; k < keep; k++) {
                if (!env->logs[k].is_sdc) {
                    slot = k;
                    break;
                }
            }
        } else {
            env->log_n++;
        }
        seen++;
        OfflineAgent* a = &env->logs[slot];
        memset(a, 0, sizeof(*a));
        cJSON* id = cJSON_GetObjectItemCaseSensitive(obj, "id");
        a->id = id && cJSON_IsNumber(id) ? id->valueint : i;
        a->is_sdc = is_sdc;
        if (is_sdc) {
            env->sdc = slot;
        }
        a->width = (float)offline_jnum(obj, "width");
        a->length = (float)offline_jnum(obj, "length");
        if (a->length < 0.5f) {
            a->length = 4.5f;
        }
        if (a->width < 0.2f) {
            a->width = 1.8f;
        }
        cJSON* goal = cJSON_GetObjectItemCaseSensitive(obj, "goalPosition");
        assert(goal);
        a->goal_x = (float)offline_jnum(goal, "x");
        a->goal_y = (float)offline_jnum(goal, "y");
        offline_read_series(cJSON_GetObjectItemCaseSensitive(obj, "position"),
            a->x, OFFLINE_TRAJ, "x");
        float* ys = (float*)malloc(OFFLINE_TRAJ * sizeof(float));
        assert(ys);
        offline_read_series(cJSON_GetObjectItemCaseSensitive(obj, "position"),
            ys, OFFLINE_TRAJ, "y");
        memcpy(a->y, ys, OFFLINE_TRAJ * sizeof(float));
        free(ys);
        offline_read_series(cJSON_GetObjectItemCaseSensitive(obj, "heading"),
            a->heading, OFFLINE_TRAJ, NULL);
        offline_read_series(cJSON_GetObjectItemCaseSensitive(obj, "velocity"),
            a->vx, OFFLINE_TRAJ, "x");
        float* vys = (float*)malloc(OFFLINE_TRAJ * sizeof(float));
        assert(vys);
        offline_read_series(cJSON_GetObjectItemCaseSensitive(obj, "velocity"),
            vys, OFFLINE_TRAJ, "y");
        memcpy(a->vy, vys, OFFLINE_TRAJ * sizeof(float));
        free(vys);
        cJSON* valids = cJSON_GetObjectItemCaseSensitive(obj, "valid");
        assert(cJSON_IsArray(valids));
        int t = 0;
        cJSON* flag = NULL;
        cJSON_ArrayForEach(flag, valids) {
            if (t >= OFFLINE_TRAJ) {
                break;
            }
            a->valid[t] = offline_jbool(flag);
            t++;
        }
    }
    cJSON_Delete(root);
    assert(env->log_n > 0);
}

static void offline_write_agent(Entity* e, OfflineAgent* a, int frame) {
    e->type = VEHICLE;
    e->width = a->width;
    e->length = a->length;
    e->height = CAR_HEIGHT;
    e->active_agent = 1;
    e->goal_position_x = a->goal_x;
    e->goal_position_y = a->goal_y;
    e->goal_position_z = 0.0f;
    e->lane_node = -1;
    if (!a->valid[frame]) {
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
        e->valid = 0;
        e->vx = 0.0f;
        e->vy = 0.0f;
        return;
    }
    e->x = a->x[frame];
    e->y = a->y[frame];
    e->z = 0.0f;
    e->heading = a->heading[frame];
    e->heading_x = cosf(e->heading);
    e->heading_y = sinf(e->heading);
    e->vx = a->vx[frame];
    e->vy = a->vy[frame];
    e->vz = 0.0f;
    e->valid = 1;
    e->collision_state = NO_COLLISION;
    e->reached_goal = 0;
    e->dyn_ready = 0;
    e->dyn_speed = 0.0f;
    e->dyn_heading = 0.0f;
    if (frame <= 0 || !a->valid[frame - 1]) {
        return;
    }
    float h0 = a->heading[frame - 1];
    e->dyn_speed = a->vx[frame - 1] * cosf(h0) + a->vy[frame - 1] * sinf(h0);
    e->dyn_heading = h0;
    e->dyn_ready = 1;
}

// Other vehicles stay off the map, so a hit here is the road edge only.
static void offline_mark_offroad(Offline* env) {
    Drive* drive = &env->drive;
    for (int i = 0; i < env->log_n; i++) {
        OfflineAgent* a = &env->logs[i];
        if (a->base_unfit) {
            continue;
        }
        for (int t = 0; t < OFFLINE_TRAJ; t++) {
            if (!a->valid[t]) {
                continue;
            }
            offline_write_agent(&drive->entities[i], a, t);
            drive->bins_ready = 0;
            collision_check(drive, i);
            if (drive->entities[i].collision_state == OFFROAD) {
                a->base_unfit = 1;
                break;
            }
        }
        drive->entities[i].x = INVALID_POSITION;
        drive->entities[i].y = INVALID_POSITION;
        drive->entities[i].valid = 0;
    }
    drive->bins_ready = 0;
}

static void offline_place(Offline* env, int frame) {
    Drive* drive = &env->drive;
    assert(frame >= 0 && frame < OFFLINE_TRAJ);
    for (int i = 0; i < env->log_n; i++) {
        offline_write_agent(&drive->entities[i], &env->logs[i], frame);
    }
    drive->bins_ready = 0;
}

// A parked car whose log is valid enters on that pose. Moving cars stay put.
static void offline_spawn_due(Offline* env) {
    int frame = env->drive.timestep;
    if (frame < 0 || frame >= OFFLINE_TRAJ) {
        return;
    }
    for (int i = 0; i < env->log_n; i++) {
        Entity* e = &env->drive.entities[i];
        OfflineAgent* a = &env->logs[i];
        if (a->base_unfit || e->x != INVALID_POSITION || !a->valid[frame]) {
            continue;
        }
        offline_write_agent(e, a, frame);
    }
    env->drive.bins_ready = 0;
}

// Integrate without reset or respawn. stop_at_goal holds a car in the arrival disk.
static void offline_integrate(Drive* env, int stop_at_goal) {
    if (!env->entities || env->active_agent_count <= 0) {
        return;
    }
    for (int i = 0; i < env->active_agent_count; i++) {
        if (!env->agents[i].rewards) {
            continue;
        }
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
    }
    float step_jerk[DRIVE_AGENT_MAX];
    memset(step_jerk, 0, sizeof(step_jerk));
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* agent = &env->entities[i];
        if (agent->x == INVALID_POSITION) {
            continue;
        }
        if (stop_at_goal && agent->reached_goal) {
            continue;
        }
        float prev_vx = agent->vx;
        float prev_vy = agent->vy;
        move_dynamics(env, i);
        float dvx = agent->vx - prev_vx;
        float dvy = agent->vy - prev_vy;
        step_jerk[i] = env->reward_jerk * sqrtf(dvx * dvx + dvy * dvy) / SIM_DT;
    }
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* agent = &env->entities[i];
        if (agent->x == INVALID_POSITION) {
            continue;
        }
        if (stop_at_goal && agent->reached_goal) {
            continue;
        }
        agent->collision_state = NO_COLLISION;
        collision_check(env, i);
        int collision_state = agent->collision_state;
        if (collision_state == AGENT_COLLISION) {
            env->agents[i].rewards[0] = env->reward_vehicle_collision;
        } else if (collision_state == OFFROAD) {
            env->agents[i].rewards[0] = env->reward_offroad_collision;
        }
        if (collision_state > NO_COLLISION) {
            agent->collided_this_goal = 1;
            if (agent->goals_completed == 0) {
                agent->collided_before_goal = 1;
            }
        }
        float distance_to_goal = relative_distance_2d(
            agent->x, agent->y, agent->goal_position_x, agent->goal_position_y);
        int arrived = distance_to_goal < MIN_DISTANCE_TO_GOAL && !agent->reached_goal;
        if (arrived) {
            env->agents[i].rewards[0] += env->reward_goal;
            agent->reached_goal = 1;
            agent->goals_completed++;
            if (!agent->collided_this_goal) {
                agent->goals_clean++;
            }
        }
        env->agents[i].rewards[0] += step_jerk[i];
        env->agents[i].rewards[0] += speed_bonus(env, agent);
        if (arrived && stop_at_goal) {
            agent->vx = 0.0f;
            agent->vy = 0.0f;
        }
    }
    bump_agents(env);
    compute_observations(env);
}

static void offline_clear_episode(Offline* env) {
    for (int i = 0; i < env->log_n; i++) {
        OfflineAgent* a = &env->logs[i];
        a->ret = 0.0f;
        a->ade_sum = 0.0f;
        a->ade_n = 0;
        a->collided = 0;
        a->offroad = 0;
        a->reached = 0;
        a->goal_dist = 0.0f;
        env->drive.entities[i].reached_goal = 0;
        env->drive.entities[i].collided_before_goal = 0;
        env->drive.entities[i].collided_this_goal = 0;
        env->drive.entities[i].goals_completed = 0;
        env->drive.entities[i].goals_clean = 0;
    }
}

static void offline_init_sim(Drive* env, Dict* cfg) {
    memset(env, 0, sizeof(*env));
    puf_init(env, cfg);
}

static void offline_reset_sim(Drive* env, unsigned int seed) {
    env->rng = seed;
    puf_reset(env);
}

static void offline_step_sim(Drive* env) {
    puf_step(env);
}

static void offline_prepare_scenario(Offline* env, const char* json_path) {
    offline_parse(env, json_path);
    env->mean_x = env->drive.world_mean_x;
    env->mean_y = env->drive.world_mean_y;
    offline_shift_logs(env);
    for (int i = 0; i < env->log_n; i++) {
        offline_label_agent(&env->logs[i]);
    }
    offline_mark_offroad(env);
    env->drive.timestep = 0;
    offline_clear_episode(env);
    offline_place(env, 0);
    compute_observations(&env->drive);
}

static void offline_reset_scenario(Offline* env) {
    env->drive.timestep = 0;
    offline_clear_episode(env);
    offline_place(env, 0);
    for (int i = 0; i < env->log_n; i++) {
        Entity* e = &env->drive.entities[i];
        if (e->x == INVALID_POSITION) {
            env->logs[i].goal_dist = -1.0f;
            continue;
        }
        env->logs[i].goal_dist = relative_distance_2d(
            e->x, e->y, e->goal_position_x, e->goal_position_y);
    }
    compute_observations(&env->drive);
}

// The controlled car integrates. Partners are placed on the log.
static void offline_replay_step(Offline* env, int controlled) {
    Drive* drive = &env->drive;
    int frame = drive->timestep;
    assert(controlled >= 0 && controlled < env->log_n);
    for (int i = 0; i < env->log_n; i++) {
        if (i == controlled) {
            continue;
        }
        drive->agents[i].actions[0] = ACCEL_COAST_BIN;
        drive->agents[i].actions[1] = STEER_COAST_BIN;
    }
    // Partners are parked so move_dynamics steps only the controlled car.
    int active[DRIVE_AGENT_MAX];
    int n = drive->active_agent_count;
    for (int i = 0; i < n; i++) {
        active[i] = drive->entities[i].active_agent;
        if (i != controlled) {
            drive->entities[i].active_agent = 0;
            drive->entities[i].x = INVALID_POSITION;
            drive->entities[i].y = INVALID_POSITION;
        }
    }
    drive->bins_ready = 0;
    offline_integrate(drive, 1);
    drive->timestep = frame + 1;
    int next = frame + 1;
    if (next >= OFFLINE_TRAJ) {
        next = OFFLINE_TRAJ - 1;
    }
    for (int i = 0; i < n; i++) {
        drive->entities[i].active_agent = active[i];
        if (i == controlled) {
            continue;
        }
        if (!env->logs[i].valid[next]) {
            drive->entities[i].x = INVALID_POSITION;
            drive->entities[i].y = INVALID_POSITION;
            continue;
        }
        OfflineAgent* a = &env->logs[i];
        Entity* e = &drive->entities[i];
        e->x = a->x[next];
        e->y = a->y[next];
        e->heading = a->heading[next];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->vx = a->vx[next];
        e->vy = a->vy[next];
        e->valid = 1;
        e->collision_state = NO_COLLISION;
    }
    drive->bins_ready = 0;
    Entity* ego = &drive->entities[controlled];
    OfflineAgent* log = &env->logs[controlled];
    int was_reached = log->reached;
    int before = ego->collision_state;
    collision_check(drive, controlled);
    // Road was scored with partners parked. Re-score overlap on the full scene.
    if (drive->agents[controlled].rewards) {
        float* rw = &drive->agents[controlled].rewards[0];
        if (before == AGENT_COLLISION) {
            *rw -= drive->reward_vehicle_collision;
        } else if (before == OFFROAD) {
            *rw -= drive->reward_offroad_collision;
        }
        if (ego->collision_state == AGENT_COLLISION) {
            *rw += drive->reward_vehicle_collision;
        } else if (ego->collision_state == OFFROAD) {
            *rw += drive->reward_offroad_collision;
        }
    }
    if (ego->collision_state > NO_COLLISION) {
        ego->collided_this_goal = 1;
        // Arrival this step already counted. A partner hit still counts as before.
        if (ego->goals_completed == 0 || !was_reached) {
            ego->collided_before_goal = 1;
        }
    }
    if (ego->collision_state == AGENT_COLLISION) {
        log->collided = 1;
    } else if (ego->collision_state == OFFROAD) {
        log->offroad = 1;
    }
    if (ego->x != INVALID_POSITION) {
        log->goal_dist = relative_distance_2d(
            ego->x, ego->y, ego->goal_position_x, ego->goal_position_y);
        if (log->goal_dist < MIN_DISTANCE_TO_GOAL) {
            log->reached = 1;
        }
        if (drive->timestep < OFFLINE_TRAJ
                && env->logs[controlled].valid[drive->timestep]) {
            float dx = ego->x - log->x[drive->timestep];
            float dy = ego->y - log->y[drive->timestep];
            log->ade_sum += sqrtf(dx * dx + dy * dy);
            log->ade_n += 1;
        }
    }
    if (drive->agents[controlled].rewards) {
        log->ret += drive->agents[controlled].rewards[0];
    }
    compute_observations(drive);
}

static void offline_collect(Offline* env, int teleport, float* obs_out,
        int* act_out, unsigned char* valid_out, float* ade_out) {
    Drive* drive = &env->drive;
    int n = env->log_n;
    int obs_n = OBS_SIZE;
    float* sim_x = (float*)malloc((size_t)n * OFFLINE_TRAJ * sizeof(float));
    float* sim_y = (float*)malloc((size_t)n * OFFLINE_TRAJ * sizeof(float));
    assert(sim_x && sim_y);
    offline_reset_scenario(env);
    for (int t = 0; t < OFFLINE_TRAJ; t++) {
        if (teleport || t == 0) {
            offline_place(env, t);
            drive->timestep = t;
            compute_observations(drive);
        }
        for (int i = 0; i < n; i++) {
            float* src = drive->agents[i].observations;
            assert(src);
            memcpy(obs_out + ((size_t)t * n + i) * obs_n, src,
                (size_t)obs_n * sizeof(float));
        }
        for (int i = 0; i < n; i++) {
            sim_x[(size_t)i * OFFLINE_TRAJ + t] = drive->entities[i].x;
            sim_y[(size_t)i * OFFLINE_TRAJ + t] = drive->entities[i].y;
            int label_ok = env->logs[i].accel_bin[t] >= 0;
            int off = (t * n + i) * 2;
            if (label_ok) {
                act_out[off] = env->logs[i].accel_bin[t];
                act_out[off + 1] = env->logs[i].steer_bin[t];
                valid_out[t * n + i] = 1;
            } else {
                act_out[off] = -1;
                act_out[off + 1] = -1;
                valid_out[t * n + i] = 0;
            }
        }
        if (t + 1 >= OFFLINE_TRAJ || teleport) {
            continue;
        }
        for (int i = 0; i < n; i++) {
            float* atn = drive->agents[i].actions;
            if (env->logs[i].accel_bin[t] >= 0) {
                atn[0] = (float)env->logs[i].accel_bin[t];
                atn[1] = (float)env->logs[i].steer_bin[t];
            } else {
                atn[0] = ACCEL_COAST_BIN;
                atn[1] = STEER_COAST_BIN;
            }
            drive->entities[i].reached_goal = 0;
        }
        offline_integrate(drive, 0);
    }
    for (int i = 0; i < n; i++) {
        OfflineAgent* a = &env->logs[i];
        float err = 0.0f;
        int count = 0;
        if (!teleport) {
            for (int t = 0; t < OFFLINE_TRAJ; t++) {
                if (!a->valid[t]) {
                    continue;
                }
                if (sim_x[(size_t)i * OFFLINE_TRAJ + t] == INVALID_POSITION) {
                    continue;
                }
                float dx = sim_x[(size_t)i * OFFLINE_TRAJ + t] - a->x[t];
                float dy = sim_y[(size_t)i * OFFLINE_TRAJ + t] - a->y[t];
                err += sqrtf(dx * dx + dy * dy);
                count++;
            }
        }
        float ade = count > 0 ? err / (float)count : 0.0f;
        ade_out[i] = teleport ? 0.0f : ade;
        int unfit = a->base_unfit;
        if (!teleport && (ade > OFFLINE_ADE_MAX || count < OFFLINE_MIN_VALID)) {
            unfit = 1;
        }
        if (unfit) {
            for (int t = 0; t < OFFLINE_TRAJ; t++) {
                valid_out[t * n + i] = 0;
                int off = (t * n + i) * 2;
                act_out[off] = -1;
                act_out[off + 1] = -1;
            }
        }
    }
    free(sim_x);
    free(sim_y);
    offline_reset_scenario(env);
}
