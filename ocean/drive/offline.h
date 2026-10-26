// Offline Waymo dataset. Included after drive.h, in one translation unit.
//
// Simulation mode steps the same Drive world as drive.h (puf_reset / puf_step),
// so a behavioral clone trained here sees the observation layout used online.
//
// Scenario mode reads one WOMD json. Roads stay the centered binary from
// dataset.py. Agent poses are the log, shifted by Drive.world_mean.
// collect_expert writes observation-action pairs:
//   teleport=1  observations come from the logged pose at every frame
//   teleport=0  the logged discrete action is integrated with move_dynamics,
//               then the track is dropped when its ADE exceeds ADE_THRESHOLD
// Actions are the drive.h accel and steer bins nearest to the inverse of
// move_dynamics. A static track, or one with fewer than OFFLINE_MIN_VALID
// logged frames, is dropped in both modes.

#pragma once

#include <math.h>
#include "cJSON.h"

#define OFFLINE_TRAJ 91
#define OFFLINE_ADE_MAX 2.0f
#define OFFLINE_MIN_VALID 10

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

static int offline_nearest(float v, const float* bins, int n) {
    int best = 0;
    float best_d = fabsf(v - bins[0]);
    for (int j = 1; j < n; j++) {
        float d = fabsf(v - bins[j]);
        if (d < best_d) {
            best_d = d;
            best = j;
        }
    }
    return best;
}

// Inverse of move_dynamics for one step that starts at the logged velocity
// and ends at the logged displacement. Steering outside the tanh range is
// clamped; the discrete bins then drop whatever is left.
static void offline_inverse(float x0, float y0, float h0, float vx0, float vy0,
        float x1, float y1, float* accel, float* steer) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float speed0 = sqrtf(vx0 * vx0 + vy0 * vy0);
    float dist = sqrtf(dx * dx + dy * dy);
    float speed1 = dist / SIM_DT;
    float a = (speed1 - speed0) / (0.5f * SIM_DT);
    // ACCELERATION_VALUES spans [-4, 4]. A larger inverse cannot be labeled.
    if (a > 4.0f) {
        a = 4.0f;
    }
    if (a < -4.0f) {
        a = -4.0f;
    }
    float s = 0.0f;
    if (dist > 1.0e-3f) {
        float motion = atan2f(dy, dx);
        float beta = motion - h0;
        if (beta > M_PI) {
            beta -= 2.0f * M_PI;
        }
        if (beta < -M_PI) {
            beta += 2.0f * M_PI;
        }
        if (beta > 0.999f) {
            beta = 0.999f;
        }
        if (beta < -0.999f) {
            beta = -0.999f;
        }
        s = atanf(2.0f * atanhf(beta));
        if (s > 1.0f) {
            s = 1.0f;
        }
        if (s < -1.0f) {
            s = -1.0f;
        }
    }
    *accel = a;
    *steer = s;
}

static void offline_label_agent(OfflineAgent* a) {
    int count = 0;
    int moved = 0;
    float x0 = 0.0f;
    float y0 = 0.0f;
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
        } else if (a->x[t] != x0 || a->y[t] != y0) {
            moved = 1;
        }
        count++;
        if (t + 1 >= OFFLINE_TRAJ || !a->valid[t + 1]) {
            continue;
        }
        float accel = 0.0f;
        float steer = 0.0f;
        offline_inverse(a->x[t], a->y[t], a->heading[t], a->vx[t], a->vy[t],
            a->x[t + 1], a->y[t + 1], &accel, &steer);
        a->accel[t] = accel;
        a->steer[t] = steer;
        a->accel_bin[t] = offline_nearest(accel, ACCELERATION_VALUES, NUM_ACCEL_BINS);
        a->steer_bin[t] = offline_nearest(steer, STEERING_VALUES, NUM_STEER_BINS);
    }
    a->base_unfit = (count < OFFLINE_MIN_VALID) || !moved;
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
}

static void offline_place(Offline* env, int frame) {
    Drive* drive = &env->drive;
    assert(frame >= 0 && frame < OFFLINE_TRAJ);
    for (int i = 0; i < env->log_n; i++) {
        offline_write_agent(&drive->entities[i], &env->logs[i], frame);
    }
    drive->bins_ready = 0;
}

// An agent that is still parked, and whose log is valid at this frame,
// enters on the logged pose. Agents already moving are left alone, and
// static or short tracks stay parked.
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

// Same body as puf_step's integrate, without the episode reset and without
// respawn. stop_at_goal holds a car that has entered the arrival disk so a
// scenario rollout stays on that destination. Drive's puf_step respawns
// instead; the parity test leaves stop_at_goal at 0 and puts the goal far
// enough that neither path takes the arrival branch.
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

// Controlled agent integrates the action already written into Drive.
// Every other vehicle is moved onto its logged pose at the new frame so the
// partner half of the observation stays the human scene.
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
    // Integrate only the controlled car. Partners sit at INVALID so
    // move_dynamics skips them, then they are written back from the log.
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
    // Integrate scored the road while partners were parked. Put the
    // partner overlap onto the same reward the flag below reports.
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
        // Same step as arrival: goals_completed is already 1, but the
        // partner was not in the integrate check. Still counts as before.
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
