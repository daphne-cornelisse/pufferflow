#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "constants.h"
#include "pufferenv.h"
#include "rlgl.h"
#include "puf_record.h"
#include <time.h>

typedef Env Drive;

typedef struct Client Client;

// One logged mover at init_steps. Waymo binaries only.
typedef struct LogSpawn {
    float x;
    float y;
    float heading;
    float vx;
    float vy;
    float gx;
    float gy;
    float length;
    float width;
    int id;
    int type;
} LogSpawn;

struct Log {
    float episode_return;
    float episode_length;
    float perf;
    float score;
    float jerk;
    float offroad_rate;
    float collision_rate;
    float collision_count;
    float steps_before_first_failure;
    float total_distance_travelled;
    float total_infractions;
    float distance_before_first_failure;
    float completion_rate;
    // Sum of speed in m/s, and how many steps that sum covers.
    float speed_sum;
    float speed_steps;
    float n;
    // Human-replay. 1 if this car caused the contact. Route progress
    // is arc length along the log. The two errors are means, in meters.
    float at_fault_collision_rate;
    float route_progress;
    float lateral_error;
    float longitudinal_error;
};

typedef struct Entity {
    int type;
    int array_size;
    float* pts_x;
    float* pts_y;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    int collision_state;
    int offroad_state;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    int valid;
    int reached_goal;
    int collided_before_goal;
    int goals_sampled;
    int goals_completed;
    int goals_clean;
    int collided_this_goal;
    int failed;
    int lane_node;
    // Signed speed and heading at the previous observation. dyn_ready
    // is 0 until that pair exists, so a fresh spawn reports no lag.
    float dyn_speed;
    float dyn_heading;
    int dyn_ready;
    // Human-replay sums along the logged path. path_n is the step count.
    float path_lat;
    float path_lon;
    int path_n;
} Entity;

float relative_distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

float normalize_heading(float heading) {
    if (heading > M_PI) {
        heading -= 2 * M_PI;
    }
    if (heading < -M_PI) {
        heading += 2 * M_PI;
    }
    return heading;
}

// The logged body is longer than the wheelbase.
static float bicycle_wheelbase(float length) {
    float wheelbase = 0.6f * length;
    if (wheelbase < 1.0f) {
        wheelbase = 1.0f;
    }
    return wheelbase;
}

// Ackermann bicycle. The rear axle has no sideslip and yaws at
// v tan(δ) / L. The body center is half a wheelbase ahead of it.
static void bicycle_integrate(float* x, float* y, float* heading,
        float* vx, float* vy, float length, float acceleration, float steering) {
    float c = cosf(*heading);
    float s = sinf(*heading);
    float v = (*vx) * c + (*vy) * s + acceleration * SIM_DT;
    float wheelbase = bicycle_wheelbase(length);
    float half = 0.5f * wheelbase;
    float rx = *x - half * c;
    float ry = *y - half * s;
    float dth;
    float dx_b;
    float dy_b;
    if (v > DYN_SPEED_MAX) {
        v = DYN_SPEED_MAX;
    }
    if (v < -DYN_SPEED_MAX) {
        v = -DYN_SPEED_MAX;
    }
    if (steering > STEER_MAX) {
        steering = STEER_MAX;
    } else if (steering < STEER_MIN) {
        steering = STEER_MIN;
    }
    float tsteer = tanf(steering);
    dth = v * tsteer / wheelbase * SIM_DT;
    if (fabsf(dth) < 1.0e-4f) {
        dx_b = v * SIM_DT;
        dy_b = 0.0f;
    } else {
        float radius = v * SIM_DT / dth;
        dx_b = radius * sinf(dth);
        dy_b = radius * (1.0f - cosf(dth));
    }
    rx += dx_b * c - dy_b * s;
    ry += dx_b * s + dy_b * c;
    *heading = normalize_heading(*heading + dth);
    c = cosf(*heading);
    s = sinf(*heading);
    *x = rx + half * c;
    *y = ry + half * s;
    // Center velocity: axle speed along the heading, plus the yaw swing.
    float v_lat = v * tsteer * 0.5f;
    *vx = v * c - v_lat * s;
    *vy = v * s + v_lat * c;
}

struct Env {
    Client* client;
    Agent agents[DRIVE_AGENT_MAX];
    Log log;
    Log* logs;
    int num_agents;
    int tag;
    int boundary_reached;
    int max_agents;
    int active_agent_count;
    int selected_agent_idx;
    Entity* entities;
    int num_entities;
    int num_objects;
    int num_roads;
    int timestep;
    int episode_length;
    float* map_corners;
    int* grid_cells;
    int grid_cols;
    int grid_rows;
    int vision_range;
    int* neighbor_offsets;
    int* neighbor_cache_entities;
    int* neighbor_cache_indices;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    char* map_name;
    // Index of map_%03d.bin. 0 Town01, 1 Town02, 2 Town06, 3 Town10HD.
    int map_id;
    // Env steps between map batches. 0 keeps the batch chosen at startup.
    // The trainer applies it between rollouts, not inside puf_step.
    int resample_frequency;
    // Finished episodes per copy between map batches. 0 uses the step count.
    int resample_episodes;
    // Finished episodes since the trainer last read this copy.
    int episode_ends;
    float world_mean_x;
    float world_mean_y;
    float reward_goal;
    float reward_jerk;
    // Target and the band around it, in m/s.
    float target_speed;
    float speed_band;
    unsigned int rng;
    int checkpoint_ok;
    char checkpoint_error[256];
    // Euclidean goal band, meters. The point is on the forward centerline.
    float goal_distance_min;
    float goal_distance_max;
    // Lane points in polyline order. lane_x owns the block; the other
    // lane pointers sit inside it. lane_next is the next point on this
    // polyline. A join is decided while walking, not stored.
    int lane_n;
    float* lane_x;
    float* lane_y;
    float* lane_hx;
    float* lane_hy;
    int* lane_next;
    int lane_polys;
    int* lane_poly_start;
    float* lane_dist;
    float* lane_block;
    int* lane_seen;
    // Set from the window. Headless eval and recording never read these.
    int sim_paused;
    // Overhead camera stays on selected_agent_idx. Off restores the map view.
    int sim_follow;
    // One frame after Reset, so that click draws timestep 0.
    int sim_hold_step;
    // Anchor demo owns the selected car. Shift does not overwrite it.
    int sim_lock_actions;
    // Anchor demo. Do not grow to the monitor. The user can resize.
    int sim_user_size;
    // Critic head for the observation on screen. ready is 0 with no policy.
    float predicted_value;
    int predicted_ready;
    void (*hud_extra)(struct Env* env);
    int (*ui_hot)(struct Env* env);
    // Anchor demo. Log rewind instead of a random spawn.
    void (*sim_reset)(struct Env* env);
    // Anchor demo. Top-right button loads another scenario.
    void (*sim_scene)(struct Env* env);
    int near_cols;
    int near_rows;
    int bins_ready;
    // When set, move_dynamics reads these floats instead of the bin tables.
    // Both paths go through bicycle_integrate. Default off.
    int use_action_values;
    float action_accel[DRIVE_AGENT_MAX];
    float action_steer[DRIVE_AGENT_MAX];
    int* bin_head;
    int* bin_next;
    Dict* cfg;
    // Waymo log spawn, centered with the roads. Empty on CARLA.
    // control_mode picks who is spawned. max_controlled caps mixed play.
    LogSpawn* spawns;
    int spawn_n;
    // Invalid at init_steps. Not controlled. They follow the log.
    LogSpawn* ghosts;
    int ghost_n;
    // 0 skips that replay. Offline labeling places those tracks.
    int replay_invalid;
    int log_spawn;
    // State count of this log. 0 on a road map. Waymo is 91.
    int log_traj;
    int control_mode;
    int max_controlled;
    // 0 uses frame 0. WOSAC eval starts at frame 10.
    int init_steps;
    // 0 keeps MIN_DISTANCE_TO_GOAL. WOSAC eval uses 2 m.
    float goal_radius;
    float goal_speed;
    // Hold the logged goal. Training removes the car and can end early.
    int goal_stop;
    // 1 samples another goal on arrival. 0 removes the car, including CARLA.
    int goal_resample;
    int wosac_on;
    int wosac_fill;
    int wosac_rolls;
    char wosac_dir[512];
    float* wosac_x;
    float* wosac_y;
    float* wosac_h;
    // Policy cars plus log-replay bodies. Equal to active_agent_count
    // except when partners sit past the policy slots.
    int body_count;
    // Logged poses, centered with the roads. Index is
    // series * replay_len + frame. Human-replay stores every spawn,
    // including the self-driving car. Training stores ghosts only.
    int replay_len;
    float* replay_x;
    float* replay_y;
    float* replay_h;
    float* replay_vx;
    float* replay_vy;
    unsigned char* replay_v;
    // Logged pose of each policy car, same centering as the roads.
    // Index is agent * gt_len + frame. Drawing only. The step never
    // reads it.
    int gt_n;
    int gt_len;
    float* gt_x;
    float* gt_y;
    float* gt_h;
    unsigned char* gt_v;
};

static int puf_sim_paused(Drive* env) {
    if (puf_record_enabled()) {
        return 0;
    }
    return env->sim_paused || env->sim_hold_step;
}

static void read_road(FILE* file, Entity* e, int type, int size) {
    assert(type >= ROAD_LANE && type <= DRIVEWAY);
    assert(size >= 1);
    e->type = type;
    e->array_size = size;
    e->pts_x = (float*)malloc((size_t)size * sizeof(float));
    e->pts_y = (float*)malloc((size_t)size * sizeof(float));
    float* z = (float*)malloc((size_t)size * sizeof(float));
    assert(e->pts_x && e->pts_y && z);
    assert(fread(e->pts_x, sizeof(float), size, file) == (size_t)size);
    assert(fread(e->pts_y, sizeof(float), size, file) == (size_t)size);
    assert(fread(z, sizeof(float), size, file) == (size_t)size);
    free(z);
    int mark = 0;
    assert(fread(&e->width, sizeof(float), 1, file) == 1);
    assert(fread(&e->length, sizeof(float), 1, file) == 1);
    assert(fread(&e->height, sizeof(float), 1, file) == 1);
    assert(fread(&e->goal_position_x, sizeof(float), 1, file) == 1);
    assert(fread(&e->goal_position_y, sizeof(float), 1, file) == 1);
    assert(fread(&e->goal_position_z, sizeof(float), 1, file) == 1);
    assert(fread(&mark, sizeof(int), 1, file) == 1);
}

static void skip_bytes(FILE* file, long n) {
    assert(n >= 0);
    assert(fseek(file, n, SEEK_CUR) == 0);
}

#define CONTROL_VEHICLES 0
#define CONTROL_AGENTS 1
#define CONTROL_WOSAC 2
#define CONTROL_SDC_ONLY 3
#define CONTROL_MIXED_PLAY 4
// PufferDrive keeps this many actors in a human-replay scene.
#define REPLAY_ACTORS 32
// Objects read from one log before the agent cap. One Waymo scene has 515.
#define LOG_CAND_MAX 1024

static int drive_control_mode(const char* name) {
    if (strcmp(name, "control_vehicles") == 0) {
        return CONTROL_VEHICLES;
    }
    if (strcmp(name, "control_wosac") == 0) {
        return CONTROL_WOSAC;
    }
    if (strcmp(name, "control_sdc_only") == 0) {
        return CONTROL_SDC_ONLY;
    }
    if (strcmp(name, "control_mixed_play") == 0) {
        return CONTROL_MIXED_PLAY;
    }
    if (strcmp(name, "control_agents") == 0) {
        return CONTROL_AGENTS;
    }
    assert(0 && "unknown control_mode");
    return -1;
}

static int control_type_ok(int mode, int type) {
    if (type == VEHICLE || type == TRUCK) {
        return 1;
    }
    if (mode == CONTROL_VEHICLES) {
        return 0;
    }
    return type == PEDESTRIAN || type == CYCLIST;
}

static void free_replay(Drive* env) {
    free(env->replay_x);
    free(env->replay_y);
    free(env->replay_h);
    free(env->replay_vx);
    free(env->replay_vy);
    free(env->replay_v);
    env->replay_x = NULL;
    env->replay_y = NULL;
    env->replay_h = NULL;
    env->replay_vx = NULL;
    env->replay_vy = NULL;
    env->replay_v = NULL;
    env->replay_len = 0;
    free(env->gt_x);
    free(env->gt_y);
    free(env->gt_h);
    free(env->gt_v);
    env->gt_x = NULL;
    env->gt_y = NULL;
    env->gt_h = NULL;
    env->gt_v = NULL;
    env->gt_n = 0;
    env->gt_len = 0;
}

// map_id-th int in <map dir>/sdc.bin. -1 if that file is absent.
static int read_map_sdc(const char* map_file, int map_id) {
    const char* slash = strrchr(map_file, '/');
    assert(slash && map_id >= 0);
    char path[512];
    int dir_n = (int)(slash - map_file);
    assert(dir_n > 0 && dir_n + 9 < (int)sizeof(path));
    memcpy(path, map_file, (size_t)dir_n);
    memcpy(path + dir_n, "/sdc.bin", 9);
    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    int count = 0;
    int sdc = 0;
    assert(fread(&count, sizeof(int), 1, f) == 1);
    assert(map_id < count);
    assert(fseek(f, (long)map_id * (long)sizeof(int), SEEK_CUR) == 0);
    assert(fread(&sdc, sizeof(int), 1, f) == 1);
    fclose(f);
    return sdc;
}

// One actor's compacted pose row. vx and vy are omitted when those
// pointers are null. Heading is channel 4 of the stored series.
static void copy_log_tracks(float* x, float* y, float* h,
        float* vx, float* vy, unsigned char* valid,
        float* series, int* vseries, LogSpawn* src, int n, int traj) {
    for (int j = 0; j < n; j++) {
        size_t row = (size_t)src[j].id * 5 * (size_t)traj;
        size_t dst = (size_t)j * (size_t)traj;
        size_t bytes = (size_t)traj * sizeof(float);
        int* flags = vseries + (size_t)src[j].id * (size_t)traj;
        memcpy(x + dst, series + row, bytes);
        memcpy(y + dst, series + row + (size_t)traj, bytes);
        if (vx) {
            memcpy(vx + dst, series + row + 2 * (size_t)traj, bytes);
            memcpy(vy + dst, series + row + 3 * (size_t)traj, bytes);
        }
        memcpy(h + dst, series + row + 4 * (size_t)traj, bytes);
        for (int t = 0; t < traj; t++) {
            valid[dst + (size_t)t] = flags[t] ? 1 : 0;
        }
    }
}

// CARLA files are roads only: int count, then road records.
// Log files are int objects, roads, first agent type, trajectory
// length, then each object (series, size, goal, mark, next type,
// next length). The length is that scenario's state count, from 2
// through DRIVE_LOG_MAX. The last object's next type and length are
// the first road header. Roads then use the CARLA record.
Entity* load_map_binary(const char* filename, Drive* env) {
    FILE* file = fopen(filename, "rb");
    assert(file);
    free(env->spawns);
    env->spawns = NULL;
    env->spawn_n = 0;
    free(env->ghosts);
    env->ghosts = NULL;
    env->ghost_n = 0;
    env->log_spawn = 0;
    env->log_traj = 0;
    free_replay(env);
    int head[4];
    assert(fread(head, sizeof(int), 4, file) == 4);
    int log_map = head[3] >= 2 && head[3] <= DRIVE_LOG_MAX
        && head[0] >= 0 && head[0] <= LOG_CAND_MAX
        && head[1] > 0 && head[2] >= VEHICLE && head[2] <= TRUCK;
    if (!log_map) {
        assert(fseek(file, 0, SEEK_SET) == 0);
        assert(fread(&env->num_roads, sizeof(int), 1, file) == 1);
        assert(env->num_roads >= 0);
        // Road maps from before truck took id 4 stored lane as 4.
        // A file that still has a 4 is that encoding. Logs are unchanged.
        long roads_at = ftell(file);
        int legacy = 0;
        for (int i = 0; i < env->num_roads; i++) {
            int type = 0;
            int size = 0;
            assert(fread(&type, sizeof(int), 1, file) == 1);
            assert(fread(&size, sizeof(int), 1, file) == 1);
            if (type == 4) {
                legacy = 1;
            }
            assert(size >= 1);
            long skip = (long)size * 3L * (long)sizeof(float)
                + 6L * (long)sizeof(float) + (long)sizeof(int);
            assert(fseek(file, skip, SEEK_CUR) == 0);
        }
        assert(fseek(file, roads_at, SEEK_SET) == 0);
        env->num_objects = 0;
        env->log_spawn = 0;
        env->num_entities = env->num_roads;
        Entity* entities = (Entity*)calloc((size_t)env->num_roads, sizeof(Entity));
        assert(entities || env->num_roads == 0);
        for (int i = 0; i < env->num_roads; i++) {
            int type = 0;
            int size = 0;
            assert(fread(&type, sizeof(int), 1, file) == 1);
            assert(fread(&size, sizeof(int), 1, file) == 1);
            if (legacy) {
                type += 1;
            }
            read_road(file, &entities[i], type, size);
        }
        unsigned char extra = 0;
        assert(fread(&extra, 1, 1, file) == 0);
        fclose(file);
        return entities;
    }

    int nobj = head[0];
    int nroads = head[1];
    int type = head[2];
    int traj = head[3];
    assert(traj >= 2 && traj <= DRIVE_LOG_MAX);
    int replay = env->control_mode == CONTROL_SDC_ONLY;
    // Tracks invalid at init stay on the log. WOSAC and human-replay
    // do not. Room is the body slots left after the policy cars.
    int ghost_mode = env->replay_invalid
        && env->control_mode != CONTROL_SDC_ONLY
        && env->control_mode != CONTROL_WOSAC;
    int room = 0;
    if (ghost_mode) {
        room = DRIVE_AGENT_MAX - env->max_agents;
        if (room > 512) {
            room = 512;
        }
        if (room < 0) {
            room = 0;
        }
    }
    // Five pose series per object. Replay keeps the rows it needs.
    float* series = NULL;
    int* vseries = NULL;
    if (nobj > 0) {
        series = (float*)malloc(
            (size_t)5 * (size_t)nobj * (size_t)traj * sizeof(float));
        vseries = (int*)malloc((size_t)nobj * (size_t)traj * sizeof(int));
        assert(series && vseries);
    }
    LogSpawn cand[LOG_CAND_MAX];
    int cand_expert[LOG_CAND_MAX];
    float dist2[LOG_CAND_MAX];
    int nc = 0;
    LogSpawn gkeep[512];
    int gk = 0;
    int road_type = 0;
    int road_size = 0;
    for (int i = 0; i < nobj; i++) {
        int step = env->init_steps;
        assert(step >= 0 && step < traj);
        float x = 0.0f;
        float y = 0.0f;
        float vx = 0.0f;
        float vy = 0.0f;
        float heading = 0.0f;
        int valid = 0;
        float* row = series + (size_t)i * 5 * (size_t)traj;
        // File order is x, y, z, vx, vy, one unused channel, heading.
        // z and the unused channel are not stored.
        int kept[7] = {0, 1, -1, 2, 3, -1, 4};
        for (int ch = 0; ch < 7; ch++) {
            if (kept[ch] < 0) {
                skip_bytes(file, (long)traj * (long)sizeof(float));
                continue;
            }
            float* dest = row + kept[ch] * traj;
            assert(fread(dest, sizeof(float), (size_t)traj, file)
                == (size_t)traj);
        }
        int* vv = vseries + (size_t)i * (size_t)traj;
        assert(fread(vv, sizeof(int), (size_t)traj, file) == (size_t)traj);
        x = row[step];
        y = row[traj + step];
        vx = row[2 * traj + step];
        vy = row[3 * traj + step];
        heading = row[4 * traj + step];
        valid = vv[step];
        float width = 0.0f;
        float length = 0.0f;
        float height = 0.0f;
        float gx = 0.0f;
        float gy = 0.0f;
        float gz = 0.0f;
        int mark = 0;
        assert(fread(&width, sizeof(float), 1, file) == 1);
        assert(fread(&length, sizeof(float), 1, file) == 1);
        assert(fread(&height, sizeof(float), 1, file) == 1);
        assert(fread(&gx, sizeof(float), 1, file) == 1);
        assert(fread(&gy, sizeof(float), 1, file) == 1);
        assert(fread(&gz, sizeof(float), 1, file) == 1);
        assert(fread(&mark, sizeof(int), 1, file) == 1);
        assert(fread(&road_type, sizeof(int), 1, file) == 1);
        assert(fread(&road_size, sizeof(int), 1, file) == 1);
        int moving = type == VEHICLE || type == TRUCK
            || type == PEDESTRIAN || type == CYCLIST;
        if (moving && valid) {
            assert(nc < LOG_CAND_MAX);
            float dx = gx - x;
            float dy = gy - y;
            cand[nc] = (LogSpawn){
                .x = x, .y = y, .heading = heading,
                .vx = vx, .vy = vy, .gx = gx, .gy = gy,
                .length = length, .width = width,
                .id = i, .type = type,
            };
            cand_expert[nc] = mark != 0;
            dist2[nc] = dx * dx + dy * dy;
            nc++;
        } else if (gk < room && moving
                && control_type_ok(env->control_mode, type)) {
            int later = 0;
            for (int t = step; t < traj; t++) {
                if (vv[t]) {
                    later = 1;
                    break;
                }
            }
            if (later) {
                gkeep[gk] = (LogSpawn){
                    .x = INVALID_POSITION,
                    .y = INVALID_POSITION,
                    .length = length,
                    .width = width,
                    .id = i,
                    .type = type,
                };
                gk++;
            }
        }
        type = road_type;
    }
    // File order, self-driving car first. WOSAC keeps every valid mover.
    // Human-replay keeps the SDC plus other valid movers, capped at
    // REPLAY_ACTORS, and does not drop experts or near goals.
    // The other modes skip expert tracks and goals closer than 5 m.
    // A mixed pool sets max_agents to that controllable count, so the
    // cap keeps every selected car. A fixed fleet still drops the rest.
    // A mover that is invalid here and valid later is log-replayed.
    int mode = env->control_mode;
    int sdc = read_map_sdc(filename, env->map_id);
    int sdc_at = -1;
    if (sdc >= 0) {
        for (int i = 0; i < nc; i++) {
            if (cand[i].id == sdc) {
                sdc_at = i;
                break;
            }
        }
    }
    int cap = env->max_agents;
    if (mode == CONTROL_MIXED_PLAY && env->max_controlled > 0
            && env->max_controlled < cap) {
        cap = env->max_controlled;
    }
    float min2 = MIN_DISTANCE_TO_GOAL * MIN_DISTANCE_TO_GOAL;
    int limit = cap;
    if (mode == CONTROL_SDC_ONLY) {
        limit = REPLAY_ACTORS;
        assert(sdc_at >= 0);
    }
    LogSpawn keep[512];
    int nk = 0;
    // Self-driving car first. Human-replay keeps it even when the
    // control filter would drop that type. Everyone else is file order.
    if (sdc_at >= 0 && (mode == CONTROL_SDC_ONLY
            || control_type_ok(mode, cand[sdc_at].type))) {
        keep[nk++] = cand[sdc_at];
    }
    for (int i = 0; i < nc && nk < limit; i++) {
        if (i == sdc_at || !control_type_ok(mode, cand[i].type)) {
            continue;
        }
        if (mode != CONTROL_SDC_ONLY && mode != CONTROL_WOSAC
                && (cand_expert[i] || dist2[i] < min2)) {
            continue;
        }
        keep[nk++] = cand[i];
    }
    if (nk > 0) {
        env->spawns = (LogSpawn*)malloc((size_t)nk * sizeof(LogSpawn));
        assert(env->spawns);
        memcpy(env->spawns, keep, (size_t)nk * sizeof(LogSpawn));
    }
    // Human-replay stores every kept actor, SDC first, so route
    // progress reads series 0. Training stores the ghost tracks only.
    LogSpawn* tracked = NULL;
    int nreplay = 0;
    if (replay && nk > 0) {
        tracked = keep;
        nreplay = nk;
    } else if (gk > 0) {
        tracked = gkeep;
        nreplay = gk;
    }
    if (nreplay > 0) {
        size_t m = (size_t)nreplay * (size_t)traj;
        env->replay_len = traj;
        env->replay_x = (float*)malloc(m * sizeof(float));
        env->replay_y = (float*)malloc(m * sizeof(float));
        env->replay_h = (float*)malloc(m * sizeof(float));
        env->replay_vx = (float*)malloc(m * sizeof(float));
        env->replay_vy = (float*)malloc(m * sizeof(float));
        env->replay_v = (unsigned char*)malloc(m);
        assert(env->replay_x && env->replay_y && env->replay_h
            && env->replay_vx && env->replay_vy && env->replay_v);
        copy_log_tracks(env->replay_x, env->replay_y, env->replay_h,
            env->replay_vx, env->replay_vy, env->replay_v,
            series, vseries, tracked, nreplay, traj);
    }
    // Policy cars. Kept beside the ghost replay so a rollout can draw
    // the logged pose without moving the controlled body.
    if (nk > 0) {
        size_t m = (size_t)nk * (size_t)traj;
        env->gt_len = traj;
        env->gt_n = nk;
        env->gt_x = (float*)malloc(m * sizeof(float));
        env->gt_y = (float*)malloc(m * sizeof(float));
        env->gt_h = (float*)malloc(m * sizeof(float));
        env->gt_v = (unsigned char*)malloc(m);
        assert(env->gt_x && env->gt_y && env->gt_h && env->gt_v);
        copy_log_tracks(env->gt_x, env->gt_y, env->gt_h,
            NULL, NULL, env->gt_v,
            series, vseries, keep, nk, traj);
    }
    if (gk > 0) {
        env->ghosts = (LogSpawn*)malloc((size_t)gk * sizeof(LogSpawn));
        assert(env->ghosts);
        memcpy(env->ghosts, gkeep, (size_t)gk * sizeof(LogSpawn));
        env->ghost_n = gk;
    }
    free(series);
    free(vseries);
    env->spawn_n = nk;
    env->log_spawn = 1;
    env->log_traj = traj;
    env->num_roads = nroads;
    env->num_objects = 0;
    env->num_entities = nroads;
    Entity* entities = (Entity*)calloc((size_t)nroads, sizeof(Entity));
    assert(entities);
    for (int i = 0; i < nroads; i++) {
        int rtype = road_type;
        int rsize = road_size;
        if (i > 0) {
            assert(fread(&rtype, sizeof(int), 1, file) == 1);
            assert(fread(&rsize, sizeof(int), 1, file) == 1);
        }
        read_road(file, &entities[i], rtype, rsize);
    }
    unsigned char extra = 0;
    assert(fread(&extra, 1, 1, file) == 0);
    fclose(file);
    return entities;
}

// Roads keep their first point. Cars are placed on lane points at reset.
void set_start_position(Drive* env) {
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type < ROAD_LANE || !e->pts_x || e->array_size < 1) {
            continue;
        }
        e->x = e->pts_x[0];
        e->y = e->pts_y[0];
        e->z = 0.0f;
    }
}

// Grid map
int getGridIndex(Drive* env, float x1, float y1) {
    if (env->map_corners[0] >= env->map_corners[2] ||
        env->map_corners[1] >= env->map_corners[3]) {
        return -1;
    }
    float relativeX = x1 - env->map_corners[0];
    float relativeY = y1 - env->map_corners[1];
    int gridX = (int)(relativeX / GRID_CELL_SIZE);
    int gridY = (int)(relativeY / GRID_CELL_SIZE);
    if (gridX < 0 || gridX >= env->grid_cols
            || gridY < 0 || gridY >= env->grid_rows) {
        return -1;
    }
    return (gridY * env->grid_cols) + gridX;
}

void init_grid_map(Drive* env) {
    free(env->map_corners);
    free(env->grid_cells);
    env->map_corners = NULL;
    env->grid_cells = NULL;
    env->grid_cols = 0;
    env->grid_rows = 0;
    float top_left_x = 0.0f;
    float top_left_y = 0.0f;
    float bottom_right_x = 0.0f;
    float bottom_right_y = 0.0f;
    int first_valid_point = 0;

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type < ROAD_LANE
                || env->entities[i].type > ROAD_EDGE) {
            continue;
        }
        Entity* e = &env->entities[i];
        for (int j = 0; j < e->array_size; j++) {
            if (e->pts_x[j] == INVALID_POSITION
                    || e->pts_y[j] == INVALID_POSITION) {
                continue;
            }
            if (!first_valid_point) {
                top_left_x = bottom_right_x = e->pts_x[j];
                top_left_y = bottom_right_y = e->pts_y[j];
                first_valid_point = 1;
                continue;
            }
            if (e->pts_x[j] < top_left_x) {
                top_left_x = e->pts_x[j];
            }
            if (e->pts_x[j] > bottom_right_x) {
                bottom_right_x = e->pts_x[j];
            }
            if (e->pts_y[j] < top_left_y) {
                top_left_y = e->pts_y[j];
            }
            if (e->pts_y[j] > bottom_right_y) {
                bottom_right_y = e->pts_y[j];
            }
        }
    }

    env->map_corners = (float*)calloc(4, sizeof(float));
    if (!first_valid_point) {
        env->grid_cols = 0;
        env->grid_rows = 0;
        env->grid_cells = NULL;
        return;
    }
    env->map_corners[0] = top_left_x;
    env->map_corners[1] = top_left_y;
    env->map_corners[2] = bottom_right_x;
    env->map_corners[3] = bottom_right_y;

    float grid_width = bottom_right_x - top_left_x;
    float grid_height = bottom_right_y - top_left_y;
    env->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_cols * env->grid_rows;
    env->grid_cells = (int*)calloc(grid_cell_count * SLOTS_PER_CELL, sizeof(int));

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type < ROAD_LANE
                || env->entities[i].type > ROAD_EDGE) {
            continue;
        }
        Entity* e = &env->entities[i];
        for (int j = 0; j < e->array_size - 1; j++) {
            float x_center = (e->pts_x[j] + e->pts_x[j + 1]) / 2;
            float y_center = (e->pts_y[j] + e->pts_y[j + 1]) / 2;
            int grid_index = getGridIndex(env, x_center, y_center);
            if (grid_index == -1) {
                continue;
            }
            int base_index = grid_index * SLOTS_PER_CELL;
            int count = env->grid_cells[base_index];
            if (count >= MAX_ENTITIES_PER_CELL) {
                continue;
            }
            env->grid_cells[base_index + count * 2 + 1] = i;
            env->grid_cells[base_index + count * 2 + 2] = j;
            env->grid_cells[base_index] = count + 1;
        }
    }
}

void init_neighbor_offsets(Drive* env) {
    int vr = env->vision_range;
    free(env->neighbor_offsets);
    env->neighbor_offsets = (int*)calloc(vr * vr * 2, sizeof(int));

    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0, y = 0, dir = 0;
    int steps_to_take = 1, steps_taken = 0, segments_completed = 0;
    int total = 0, max_offsets = vr * vr;
    int curr_idx = 0;

    env->neighbor_offsets[curr_idx++] = 0;
    env->neighbor_offsets[curr_idx++] = 0;
    total++;

    while (total < max_offsets) {
        x += dx[dir];
        y += dy[dir];
        if (abs(x) <= vr / 2 && abs(y) <= vr / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        if (steps_taken != steps_to_take) {
            continue;
        }
        steps_taken = 0;
        dir = (dir + 1) % 4;
        segments_completed++;
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

void cache_neighbor_offsets(Drive* env) {
    int vr = env->vision_range;
    int count = 0;
    free(env->neighbor_cache_entities);
    env->neighbor_cache_entities = NULL;
    int cell_count = env->grid_cols * env->grid_rows;

    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_cols;
        int cell_y = i / env->grid_cols;
        env->neighbor_cache_indices[i] = count;
        for (int j = 0; j < vr * vr; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= env->grid_cols
                    || y < 0 || y >= env->grid_rows) {
                continue;
            }
            int grid_index = env->grid_cols * y + x;
            count += env->grid_cells[grid_index * SLOTS_PER_CELL] * 2;
        }
    }
    env->neighbor_cache_indices[cell_count] = count;
    env->neighbor_cache_entities = (int*)calloc(count, sizeof(int));

    for (int i = 0; i < cell_count; i++) {
        int neighbor_cache_base_index = 0;
        int cell_x = i % env->grid_cols;
        int cell_y = i / env->grid_cols;
        for (int j = 0; j < vr * vr; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= env->grid_cols
                    || y < 0 || y >= env->grid_rows) {
                continue;
            }
            int grid_index = env->grid_cols * y + x;
            int grid_count = env->grid_cells[grid_index * SLOTS_PER_CELL];
            int base_index = env->neighbor_cache_indices[i];
            int src_idx = grid_index * SLOTS_PER_CELL + 1;
            int dst_idx = base_index + neighbor_cache_base_index;
            memcpy(&env->neighbor_cache_entities[dst_idx],
                &env->grid_cells[src_idx],
                grid_count * 2 * sizeof(int));
            neighbor_cache_base_index += grid_count * 2;
        }
    }
}

int get_neighbor_cache_entities(Drive* env, int cell_idx,
        int* entities, int max_entities) {
    if (cell_idx < 0 || cell_idx >= (env->grid_cols * env->grid_rows)) {
        return 0;
    }
    int base_index = env->neighbor_cache_indices[cell_idx];
    int end_index = env->neighbor_cache_indices[cell_idx + 1];
    int count = end_index - base_index;
    int pairs = count / 2;
    if (pairs > max_entities) {
        pairs = max_entities;
        count = pairs * 2;
    }
    memcpy(entities, env->neighbor_cache_entities + base_index,
        count * sizeof(int));
    return pairs;
}

void set_means(Drive* env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type < ROAD_LANE || !e->pts_x) {
            continue;
        }
        for (int j = 0; j < e->array_size; j++) {
            if (e->pts_x[j] == INVALID_POSITION
                    || e->pts_y[j] == INVALID_POSITION) {
                continue;
            }
            point_count++;
            mean_x += (e->pts_x[j] - mean_x) / point_count;
            mean_y += (e->pts_y[j] - mean_y) / point_count;
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;

    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type < ROAD_LANE || !e->pts_x) {
            continue;
        }
        for (int j = 0; j < e->array_size; j++) {
            if (e->pts_x[j] == INVALID_POSITION) {
                continue;
            }
            e->pts_x[j] -= mean_x;
            e->pts_y[j] -= mean_y;
        }
    }
}

int check_line_intersection(float p1[2], float p2[2],
        float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0])
            || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0])
            || fmax(p1[1], p2[1]) < fmin(q1[1], q2[1])
            || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1])) {
        return 0;
    }

    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];
    float cross = dx1 * dy2 - dy1 * dx2;
    if (cross == 0) {
        return 0;
    }

    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;
    return s >= 0 && s <= 1 && t >= 0 && t <= 1;
}

int check_neighbors(Drive* env, float x, float y, int* entity_list,
        int max_size, const int (*local_offsets)[2], int offset_size) {
    int index = getGridIndex(env, x, y);
    if (index == -1) {
        return 0;
    }
    int cellsX = env->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;

    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        if (nx < 0 || nx >= env->grid_cols
                || ny < 0 || ny >= env->grid_rows) {
            continue;
        }
        int neighborIndex = (ny * env->grid_cols + nx) * SLOTS_PER_CELL;
        int count = env->grid_cells[neighborIndex];
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            entity_list[entity_list_count] =
                env->grid_cells[neighborIndex + 1 + j * 2];
            entity_list[entity_list_count + 1] =
                env->grid_cells[neighborIndex + 2 + j * 2];
            entity_list_count += 2;
        }
    }
    return entity_list_count;
}

int check_aabb_collision(Entity* car1, Entity* car2) {
    float cos1 = car1->heading_x, sin1 = car1->heading_y;
    float cos2 = car2->heading_x, sin2 = car2->heading_y;
    float hl1 = car1->length * 0.5f, hw1 = car1->width * 0.5f;
    float hl2 = car2->length * 0.5f, hw2 = car2->width * 0.5f;

    float car1_corners[4][2] = {
        {car1->x + (hl1 * cos1 - hw1 * sin1), car1->y + (hl1 * sin1 + hw1 * cos1)},
        {car1->x + (hl1 * cos1 + hw1 * sin1), car1->y + (hl1 * sin1 - hw1 * cos1)},
        {car1->x + (-hl1 * cos1 - hw1 * sin1), car1->y + (-hl1 * sin1 + hw1 * cos1)},
        {car1->x + (-hl1 * cos1 + hw1 * sin1), car1->y + (-hl1 * sin1 - hw1 * cos1)}
    };

    float car2_corners[4][2] = {
        {car2->x + (hl2 * cos2 - hw2 * sin2), car2->y + (hl2 * sin2 + hw2 * cos2)},
        {car2->x + (hl2 * cos2 + hw2 * sin2), car2->y + (hl2 * sin2 - hw2 * cos2)},
        {car2->x + (-hl2 * cos2 - hw2 * sin2), car2->y + (-hl2 * sin2 + hw2 * cos2)},
        {car2->x + (-hl2 * cos2 + hw2 * sin2), car2->y + (-hl2 * sin2 - hw2 * cos2)}
    };

    float axes[4][2] = {
        {cos1, sin1}, {-sin1, cos1},
        {cos2, sin2}, {-sin2, cos2}
    };

    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;
        for (int j = 0; j < 4; j++) {
            float proj1 = car1_corners[j][0] * axes[i][0]
                + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj1);
            max1 = fmaxf(max1, proj1);
            float proj2 = car2_corners[j][0] * axes[i][0]
                + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj2);
            max2 = fmaxf(max2, proj2);
        }
        if (max1 < min2 || min1 > max2) {
            return 0;
        }
    }
    return 1;
}

static int agent_cell(Drive* env, float x, float y) {
    int cx = (int)((x - env->map_corners[0]) / COLLISION_NEAR);
    int cy = (int)((y - env->map_corners[1]) / COLLISION_NEAR);
    if (cx < 0) {
        cx = 0;
    } else if (cx >= env->near_cols) {
        cx = env->near_cols - 1;
    }
    if (cy < 0) {
        cy = 0;
    } else if (cy >= env->near_rows) {
        cy = env->near_rows - 1;
    }
    return cy * env->near_cols + cx;
}

// Partner index for the last agent-agent contact, or -1. The other
// body's own check may not run when it is a log car.
static int drive_hit_with[DRIVE_AGENT_MAX];

int collision_check(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    if (agent->x == INVALID_POSITION) {
        agent->offroad_state = 0;
        return -1;
    }

    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        float ox = offsets[i][0] * half_length;
        float oy = offsets[i][1] * half_width;
        corners[i][0] = agent->x + ox * cos_heading - oy * sin_heading;
        corners[i][1] = agent->y + ox * sin_heading + oy * cos_heading;
    }

    int collided = NO_COLLISION;
    int offroad = 0;
    int car_collided_with_index = -1;

    // Check road edge collisions via grid
    int entity_list[MAX_ENTITIES_PER_CELL * 2 * 25];
    int list_size = check_neighbors(env, agent->x, agent->y, entity_list,
        MAX_ENTITIES_PER_CELL * 2 * 25, collision_offsets, 25);
    for (int i = 0; i < list_size; i += 2) {
        if (entity_list[i] == -1 || entity_list[i] == agent_idx) {
            continue;
        }
        Entity* entity = &env->entities[entity_list[i]];
        if (entity->type != ROAD_EDGE) {
            continue;
        }
        int geometry_idx = entity_list[i + 1];
        float start[2] = {
            entity->pts_x[geometry_idx], entity->pts_y[geometry_idx]
        };
        float end[2] = {
            entity->pts_x[geometry_idx + 1],
            entity->pts_y[geometry_idx + 1]
        };
        for (int k = 0; k < 4; k++) {
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], start, end)) {
                offroad = 1;
                collided = OFFROAD;
                break;
            }
        }
        if (offroad) {
            break;
        }
    }

    // Cells are COLLISION_NEAR meters wide. Center distance is tested
    // first, and the box test runs only inside that boundary.
    if (!env->bins_ready) {
        int cells = env->near_cols * env->near_rows;
        memset(env->bin_head, 0xff, (size_t)cells * sizeof(int));
        for (int i = 0; i < env->body_count; i++) {
            Entity* e = &env->entities[i];
            if (e->valid == 0
                    || e->x == INVALID_POSITION
                    || e->y == INVALID_POSITION) {
                continue;
            }
            int cell = agent_cell(env, e->x, e->y);
            env->bin_next[i] = env->bin_head[cell];
            env->bin_head[cell] = i;
        }
        env->bins_ready = 1;
    }
    int home = agent_cell(env, agent->x, agent->y);
    int gx = home % env->near_cols;
    int gy = home / env->near_cols;
    int x0 = gx > 0 ? gx - 1 : gx;
    int y0 = gy > 0 ? gy - 1 : gy;
    int x1 = gx + 1 < env->near_cols ? gx + 1 : gx;
    int y1 = gy + 1 < env->near_rows ? gy + 1 : gy;
    for (int cy = y0; cy <= y1 && collided != AGENT_COLLISION; cy++) {
        int row = cy * env->near_cols;
        for (int cx = x0; cx <= x1 && collided != AGENT_COLLISION; cx++) {
            int other = env->bin_head[row + cx];
            while (other >= 0 && collided != AGENT_COLLISION) {
                int next = env->bin_next[other];
                Entity* b = &env->entities[other];
                float dx = b->x - agent->x;
                float dy = b->y - agent->y;
                if (other != agent_idx
                    && dx * dx + dy * dy <= COLLISION_NEAR_SQ
                    && check_aabb_collision(agent, b)) {
                    collided = AGENT_COLLISION;
                    car_collided_with_index = other;
                }
                other = next;
            }
        }
    }

    agent->offroad_state = offroad;
    agent->collision_state = collided;
    assert(agent_idx >= 0 && agent_idx < DRIVE_AGENT_MAX);
    drive_hit_with[agent_idx] = car_collided_with_index;
    return car_collided_with_index;
}

static void build_lanes(Drive* env) {
    int n = 0;
    int polys = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type != ROAD_LANE || e->array_size < 2) {
            continue;
        }
        n += e->array_size;
        polys++;
    }
    assert(n > 1);
    env->lane_n = n;
    env->lane_polys = polys;
    size_t bytes = (size_t)n * (6 * sizeof(float) + 2 * sizeof(int));
    bytes += (size_t)polys * sizeof(int);
    char* buf = (char*)calloc(1, bytes);
    assert(buf);
    float* f = (float*)buf;
    env->lane_x = f;
    env->lane_y = f + n;
    env->lane_hx = f + 2 * n;
    env->lane_hy = f + 3 * n;
    env->lane_dist = f + 4 * n;
    env->lane_block = f + 5 * n;
    int* ip = (int*)(f + 6 * n);
    env->lane_next = ip;
    env->lane_seen = ip + n;
    env->lane_poly_start = ip + 2 * n;

    int cursor = 0;
    int pi = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type != ROAD_LANE || e->array_size < 2) {
            continue;
        }
        env->lane_poly_start[pi++] = cursor;
        for (int k = 0; k < e->array_size; k++) {
            int a = k;
            int b = k + 1;
            if (b == e->array_size) {
                a = k - 1;
                b = k;
            }
            float dx = e->pts_x[b] - e->pts_x[a];
            float dy = e->pts_y[b] - e->pts_y[a];
            float hyp = sqrtf(dx * dx + dy * dy);
            if (hyp > 0.0f) {
                dx /= hyp;
                dy /= hyp;
            }
            env->lane_x[cursor] = e->pts_x[k];
            env->lane_y[cursor] = e->pts_y[k];
            env->lane_hx[cursor] = dx;
            env->lane_hy[cursor] = dy;
            env->lane_next[cursor] = -1;
            if (k + 1 < e->array_size) {
                env->lane_next[cursor] = cursor + 1;
            }
            cursor++;
        }
    }
    assert(cursor == n && pi == polys);
}

static int lane_succ(Drive* env, int node, int* out) {
    int nxt = env->lane_next[node];
    if (nxt >= 0) {
        out[0] = nxt;
        return 1;
    }
    int nout = 0;
    float x = env->lane_x[node];
    float y = env->lane_y[node];
    float hx = env->lane_hx[node];
    float hy = env->lane_hy[node];
    float stitch2 = LANE_STITCH * LANE_STITCH;
    for (int p = 0; p < env->lane_polys; p++) {
        int s = env->lane_poly_start[p];
        float dx = env->lane_x[s] - x;
        float dy = env->lane_y[s] - y;
        float d2 = dx * dx + dy * dy;
        if (s == node || d2 > stitch2) {
            continue;
        }
        float hdot = hx * env->lane_hx[s] + hy * env->lane_hy[s];
        if (hdot <= LANE_HEADING_DOT) {
            continue;
        }
        if (d2 > 1.0e-6f) {
            float dist = sqrtf(d2);
            float sdot = (dx / dist) * hx + (dy / dist) * hy;
            if (sdot <= LANE_HEADING_DOT) {
                continue;
            }
        }
        assert(nout < LANE_SUCC_MAX);
        out[nout++] = s;
    }
    return nout;
}

// Initialization / Cleanup
static void release_world(Drive* env) {
    if (env->entities) {
        for (int i = 0; i < env->num_entities; i++) {
            free(env->entities[i].pts_x);
            free(env->entities[i].pts_y);
        }
        free(env->entities);
        env->entities = NULL;
    }
    env->num_entities = 0;
    env->num_objects = 0;
    env->num_roads = 0;
    env->active_agent_count = 0;
    free(env->logs);
    env->logs = NULL;
    free(env->map_corners);
    env->map_corners = NULL;
    free(env->grid_cells);
    env->grid_cells = NULL;
    env->grid_cols = 0;
    env->grid_rows = 0;
    free(env->neighbor_offsets);
    env->neighbor_offsets = NULL;
    free(env->neighbor_cache_entities);
    env->neighbor_cache_entities = NULL;
    free(env->neighbor_cache_indices);
    env->neighbor_cache_indices = NULL;
    free(env->lane_x);
    env->lane_x = NULL;
    free(env->bin_head);
    env->bin_head = NULL;
    free(env->bin_next);
    env->bin_next = NULL;
    env->bins_ready = 0;
    free(env->spawns);
    env->spawns = NULL;
    env->spawn_n = 0;
    free(env->ghosts);
    env->ghosts = NULL;
    env->ghost_n = 0;
    env->log_spawn = 0;
    env->log_traj = 0;
    free_replay(env);
    free(env->wosac_x);
    free(env->wosac_y);
    free(env->wosac_h);
    env->wosac_x = NULL;
    env->wosac_y = NULL;
    env->wosac_h = NULL;
}

static void load_world(Drive* env) {
    env->entities = load_map_binary(env->map_name, env);
    assert(env->entities);
    int n = env->max_agents;
    assert(n >= 1 && n <= DRIVE_AGENT_MAX);
    int extra = 0;
    if (env->control_mode == CONTROL_SDC_ONLY && env->spawn_n > 1) {
        extra = env->spawn_n - 1;
    } else if (env->ghost_n > 0) {
        extra = env->ghost_n;
    }
    int bodies = n + extra;
    assert(bodies <= DRIVE_AGENT_MAX);
    int nroads = env->num_roads;
    Entity* neu = (Entity*)calloc(
        (size_t)bodies + (size_t)nroads, sizeof(Entity));
    assert(neu);
    for (int i = 0; i < bodies; i++) {
        Entity* e = &neu[i];
        e->type = VEHICLE;
        e->width = CAR_WIDTH * COLLISION_BOX_SCALE;
        e->length = CAR_LENGTH * COLLISION_BOX_SCALE;
        e->height = CAR_HEIGHT;
        e->valid = 1;
        e->collision_state = NO_COLLISION;
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
        e->lane_node = -1;
    }
    if (nroads > 0) {
        memcpy(neu + bodies, env->entities, (size_t)nroads * sizeof(Entity));
    }
    free(env->entities);
    env->entities = neu;
    env->num_objects = bodies;
    env->num_entities = bodies + nroads;
    env->active_agent_count = n;
    env->body_count = bodies;
    env->num_agents = n;
    set_means(env);
    for (int i = 0; i < env->spawn_n; i++) {
        env->spawns[i].x -= env->world_mean_x;
        env->spawns[i].y -= env->world_mean_y;
        env->spawns[i].gx -= env->world_mean_x;
        env->spawns[i].gy -= env->world_mean_y;
    }
    if (env->replay_x) {
        int len = env->replay_len;
        int nseries = env->spawn_n;
        if (env->control_mode != CONTROL_SDC_ONLY) {
            nseries = env->ghost_n;
        }
        for (int s = 0; s < nseries; s++) {
            for (int t = 0; t < len; t++) {
                int k = s * len + t;
                if (!env->replay_v[k]) {
                    continue;
                }
                env->replay_x[k] -= env->world_mean_x;
                env->replay_y[k] -= env->world_mean_y;
            }
        }
    }
    if (env->gt_x) {
        int len = env->gt_len;
        for (int s = 0; s < env->gt_n; s++) {
            for (int t = 0; t < len; t++) {
                int k = s * len + t;
                if (!env->gt_v[k]) {
                    continue;
                }
                env->gt_x[k] -= env->world_mean_x;
                env->gt_y[k] -= env->world_mean_y;
            }
        }
    }
    build_lanes(env);
    init_grid_map(env);
    env->vision_range = VISION_RANGE;
    init_neighbor_offsets(env);
    free(env->neighbor_cache_indices);
    env->neighbor_cache_indices = (int*)calloc(
        (size_t)(env->grid_cols * env->grid_rows) + 1, sizeof(int));
    cache_neighbor_offsets(env);
    set_start_position(env);
    free(env->logs);
    env->logs = (Log*)calloc((size_t)n, sizeof(Log));
    assert(env->logs);
    float span_x = env->map_corners[2] - env->map_corners[0];
    float span_y = env->map_corners[3] - env->map_corners[1];
    int cols = (int)ceilf(span_x / COLLISION_NEAR);
    int rows = (int)ceilf(span_y / COLLISION_NEAR);
    assert(cols >= 1 && rows >= 1);
    env->near_cols = cols;
    env->near_rows = rows;
    free(env->bin_head);
    free(env->bin_next);
    env->bin_head = (int*)malloc((size_t)cols * rows * sizeof(int));
    env->bin_next = (int*)malloc((size_t)DRIVE_AGENT_MAX * sizeof(int));
    assert(env->bin_head && env->bin_next);
    env->bins_ready = 0;
}

void puf_close(Drive* env) {
    release_world(env);
}

// Dynamics
void move_dynamics(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    float acceleration;
    float steering;
    if (env->use_action_values) {
        acceleration = env->action_accel[agent_idx];
        steering = env->action_steer[agent_idx];
    } else {
        float* atn = env->agents[agent_idx].actions;
        int acceleration_index = (int)atn[0];
        int steering_index = (int)atn[1];
        acceleration = ACCELERATION_VALUES[acceleration_index];
        steering = STEERING_VALUES[steering_index];
    }

    float x = agent->x;
    float y = agent->y;
    float heading = agent->heading;
    float vx = agent->vx;
    float vy = agent->vy;
    bicycle_integrate(&x, &y, &heading, &vx, &vy, agent->length,
        acceleration, steering);
    agent->x = x;
    agent->y = y;
    agent->heading = heading;
    agent->heading_x = cosf(heading);
    agent->heading_y = sinf(heading);
    agent->vx = vx;
    agent->vy = vy;
    env->bins_ready = 0;
}

// Observations
void compute_observations(Drive* env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        float* obs = env->agents[i].observations;
        if (!obs) {
            continue;
        }
        memset(obs, 0, OBS_SIZE * sizeof(float));
        Entity* ego = &env->entities[i];
        if (ego->type != VEHICLE && ego->type != TRUCK) {
            continue;
        }
        if (ego->x == INVALID_POSITION || ego->y == INVALID_POSITION) {
            continue;
        }

        float cos_h = ego->heading_x;
        float sin_h = ego->heading_y;
        float ego_speed = sqrtf(ego->vx * ego->vx + ego->vy * ego->vy);

        // Goal in ego frame
        float goal_x = ego->goal_position_x - ego->x;
        float goal_y = ego->goal_position_y - ego->y;
        float rel_goal_x = goal_x * cos_h + goal_y * sin_h;
        float rel_goal_y = -goal_x * sin_h + goal_y * cos_h;

        // Ego features
        obs[0] = rel_goal_x * OBS_GOAL_SCALE;
        obs[1] = rel_goal_y * OBS_GOAL_SCALE;
        obs[2] = ego_speed * OBS_SPEED_SCALE;
        obs[3] = ego->width / MAX_VEH_WIDTH;
        obs[4] = ego->length / MAX_VEH_LEN;
        obs[5] = (ego->collision_state > NO_COLLISION) ? 1 : 0;
        float signed_speed = ego->vx * cos_h + ego->vy * sin_h;
        float accel_lag = 0.0f;
        float yaw_lag = 0.0f;
        if (ego->dyn_ready) {
            accel_lag = (signed_speed - ego->dyn_speed) / SIM_DT;
            if (accel_lag > ACCEL_MAX) {
                accel_lag = ACCEL_MAX;
            }
            if (accel_lag < ACCEL_MIN) {
                accel_lag = ACCEL_MIN;
            }
            yaw_lag = normalize_heading(ego->heading - ego->dyn_heading) / SIM_DT;
            if (yaw_lag > OBS_YAW_ABS) {
                yaw_lag = OBS_YAW_ABS;
            }
            if (yaw_lag < -OBS_YAW_ABS) {
                yaw_lag = -OBS_YAW_ABS;
            }
        }
        obs[6] = accel_lag * OBS_ACCEL_SCALE;
        obs[7] = yaw_lag * OBS_YAW_SCALE;
        obs[EGO_TARGET_SPEED] = env->target_speed * OBS_SPEED_SCALE;
        ego->dyn_speed = signed_speed;
        ego->dyn_heading = ego->heading;
        ego->dyn_ready = 1;

        // First MAX_AGENTS partners in index order, inside range.
        int obs_idx = EGO_FEATURES;
        int cars_seen = 0;
        int nscan = env->body_count;
        if (nscan > DRIVE_AGENT_MAX) {
            nscan = DRIVE_AGENT_MAX;
        }
        for (int index = 0; index < nscan && cars_seen < MAX_AGENTS; index++) {
            if (index == i) {
                continue;
            }

            Entity* other = &env->entities[index];
            if (other->valid == 0
                    || other->x == INVALID_POSITION
                    || other->y == INVALID_POSITION) {
                continue;
            }

            float dx = other->x - ego->x;
            float dy = other->y - ego->y;
            if ((dx * dx + dy * dy) > OBS_DIST_SQ) {
                continue;
            }

            float rel_x = dx * cos_h + dy * sin_h;
            float rel_y = -dx * sin_h + dy * cos_h;

            obs[obs_idx + 0] = rel_x * OBS_POSITION_SCALE;
            obs[obs_idx + 1] = rel_y * OBS_POSITION_SCALE;
            obs[obs_idx + 2] = other->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other->length / MAX_VEH_LEN;
            obs[obs_idx + 4] = other->heading_x * ego->heading_x
                + other->heading_y * ego->heading_y;
            obs[obs_idx + 5] = other->heading_y * ego->heading_x
                - other->heading_x * ego->heading_y;
            float other_speed = sqrtf(other->vx * other->vx + other->vy * other->vy);
            obs[obs_idx + 6] = other_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += PARTNER_FEATURES;
        }
        // The buffer was cleared. Empty partner slots stay zero.
        obs_idx += (MAX_AGENTS - cars_seen) * PARTNER_FEATURES;

        // Road observations
        int entity_list[MAX_ROAD_SEGMENT_OBSERVATIONS * 2];
        int grid_idx = getGridIndex(env, ego->x, ego->y);
        int list_size = get_neighbor_cache_entities(
            env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k * 2];
            int geometry_idx = entity_list[k * 2 + 1];
            Entity* entity = &env->entities[entity_idx];

            float start_x = entity->pts_x[geometry_idx];
            float start_y = entity->pts_y[geometry_idx];
            float end_x = entity->pts_x[geometry_idx + 1];
            float end_y = entity->pts_y[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego->x;
            float rel_y = mid_y - ego->y;
            float x_obs = rel_x * cos_h + rel_y * sin_h;
            float y_obs = -rel_x * sin_h + rel_y * cos_h;
            float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);

            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float hypot = sqrtf(dx * dx + dy * dy);
            float dx_norm = dx;
            float dy_norm = dy;
            if (hypot > 0) {
                dx_norm /= hypot;
                dy_norm /= hypot;
            }

            float cos_angle = dx_norm * cos_h + dy_norm * sin_h;
            float sin_angle = -dx_norm * sin_h + dy_norm * cos_h;

            obs[obs_idx + 0] = x_obs * OBS_POSITION_SCALE;
            obs[obs_idx + 1] = y_obs * OBS_POSITION_SCALE;
            obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 3] = 0.1f / MAX_ROAD_SCALE;
            obs[obs_idx + 4] = cos_angle;
            obs[obs_idx + 5] = sin_angle;
            obs[obs_idx + 6] = entity->type - (float)ROAD_LANE;
            obs_idx += ROAD_FEATURES;
        }
    }
}

static int spot_taken(Drive* env, int self) {
    Entity* a = &env->entities[self];
    int n = env->active_agent_count;
    for (int j = 0; j < n; j++) {
        if (j == self) {
            continue;
        }
        Entity* o = &env->entities[j];
        if (o->x == INVALID_POSITION || o->y == INVALID_POSITION) {
            continue;
        }
        if (check_aabb_collision(a, o)) {
            return 1;
        }
    }
    return 0;
}

static void write_goal(Entity* e, float gx, float gy) {
    e->goal_position_x = gx;
    e->goal_position_y = gy;
    e->goal_position_z = 0.0f;
    e->reached_goal = 0;
}

static void sit_on_lane(Drive* env, Entity* e, int k) {
    e->x = env->lane_x[k];
    e->y = env->lane_y[k];
    e->z = 0.0f;
    e->heading_x = env->lane_hx[k];
    e->heading_y = env->lane_hy[k];
    e->heading = normalize_heading(atan2f(e->heading_y, e->heading_x));
    e->vx = 0.0f;
    e->vy = 0.0f;
    e->vz = 0.0f;
    e->collision_state = NO_COLLISION;
    e->dyn_ready = 0;
    env->bins_ready = 0;
}

// Shortest forward path. lane_dist stays negative off the walked road.
// The walk stops past the goal band: a car farther than that cannot
// sit between the car and a point still inside the band.
static void lane_reach(Drive* env, int src) {
    int n = env->lane_n;
    float path_cap = env->goal_distance_max * 3.0f + GOAL_FOLLOW_GAP;
    for (int i = 0; i < n; i++) {
        env->lane_dist[i] = -1.0f;
        env->lane_seen[i] = 0;
    }
    assert(src >= 0 && src < n);
    env->lane_dist[src] = 0.0f;
    for (;;) {
        int u = -1;
        float best = path_cap;
        for (int i = 0; i < n; i++) {
            if (env->lane_seen[i] || env->lane_dist[i] < 0.0f) {
                continue;
            }
            if (env->lane_dist[i] <= best) {
                best = env->lane_dist[i];
                u = i;
            }
        }
        if (u < 0) {
            break;
        }
        env->lane_seen[u] = 1;
        int succ[LANE_SUCC_MAX];
        int fn = lane_succ(env, u, succ);
        for (int k = 0; k < fn; k++) {
            int v = succ[k];
            if (env->lane_seen[v]) {
                continue;
            }
            float dx = env->lane_x[v] - env->lane_x[u];
            float dy = env->lane_y[v] - env->lane_y[u];
            float nd = env->lane_dist[u] + sqrtf(dx * dx + dy * dy);
            if (nd > path_cap) {
                continue;
            }
            if (env->lane_dist[v] < 0.0f || nd < env->lane_dist[v]) {
                env->lane_dist[v] = nd;
            }
        }
    }
}

// lane_block[i] is the path distance of the nearest car downstream of
// i, or 1e30 when that branch is empty. A car on a different fork does
// not close this one. Lanes are about 3.2 m apart, so a car farther
// than LANE_CAR_RADIUS is next door. Call after lane_reach.
static void lane_blocks(Drive* env, int self) {
    int n = env->lane_n;
    int left = 0;
    float r2 = LANE_CAR_RADIUS * LANE_CAR_RADIUS;
    for (int i = 0; i < n; i++) {
        if (env->lane_dist[i] < 0.0f) {
            env->lane_block[i] = -2.0f;
            continue;
        }
        env->lane_block[i] = -1.0f;
        left++;
    }
    while (left > 0) {
        int u = -1;
        float best = -1.0f;
        for (int i = 0; i < n; i++) {
            if (env->lane_block[i] != -1.0f) {
                continue;
            }
            if (env->lane_dist[i] >= best) {
                best = env->lane_dist[i];
                u = i;
            }
        }
        if (u < 0) {
            break;
        }
        float block = 1.0e30f;
        if (env->lane_dist[u] > 0.0f) {
            for (int j = 0; j < env->active_agent_count; j++) {
                if (j == self) {
                    continue;
                }
                Entity* o = &env->entities[j];
                if (o->x == INVALID_POSITION || o->y == INVALID_POSITION) {
                    continue;
                }
                float dx = o->x - env->lane_x[u];
                float dy = o->y - env->lane_y[u];
                if (dx * dx + dy * dy > r2) {
                    continue;
                }
                float dot = o->heading_x * env->lane_hx[u]
                    + o->heading_y * env->lane_hy[u];
                if (dot > LANE_HEADING_DOT) {
                    block = env->lane_dist[u];
                    break;
                }
            }
        }
        int succ[LANE_SUCC_MAX];
        int fn = lane_succ(env, u, succ);
        for (int k = 0; k < fn; k++) {
            int v = succ[k];
            if (env->lane_block[v] < 0.0f) {
                continue;
            }
            if (env->lane_dist[v] + 1.0e-3f < env->lane_dist[u]) {
                continue;
            }
            if (env->lane_block[v] < block) {
                block = env->lane_block[v];
            }
        }
        env->lane_block[u] = block;
        left--;
    }
}

static float open_path_limit(float blocker) {
    if (blocker >= 1.0e29f) {
        return 1.0e30f;
    }
    return blocker - GOAL_FOLLOW_GAP;
}

// Reservoir sample of forward nodes in the 5–50 m band whose path ends
// GOAL_FOLLOW_GAP meters before the next car. Returns 0 when that
// window is empty.
static int pick_forward_goal(Drive* env, Entity* e, int self) {
    int node = e->lane_node;
    if (node < 0 || node >= env->lane_n) {
        return 0;
    }
    lane_reach(env, node);
    lane_blocks(env, self);
    float x = e->x;
    float y = e->y;
    float hx = e->heading_x;
    float hy = e->heading_y;
    float lo2 = env->goal_distance_min * env->goal_distance_min;
    float hi2 = env->goal_distance_max * env->goal_distance_max;
    int seen = 0;
    float pick_x = 0.0f;
    float pick_y = 0.0f;
    for (int i = 0; i < env->lane_n; i++) {
        float path = env->lane_dist[i];
        float max_path = open_path_limit(env->lane_block[i]);
        if (path <= 0.0f || path > max_path) {
            continue;
        }
        float dx = env->lane_x[i] - x;
        float dy = env->lane_y[i] - y;
        float d2 = dx * dx + dy * dy;
        if (d2 < lo2 || d2 > hi2) {
            continue;
        }
        if (dx * hx + dy * hy <= 0.0f) {
            continue;
        }
        seen++;
        if ((int)(rand_r(&env->rng) % (unsigned)seen) == 0) {
            pick_x = env->lane_x[i];
            pick_y = env->lane_y[i];
        }
    }
    if (seen == 0) {
        return 0;
    }
    write_goal(e, pick_x, pick_y);
    return 1;
}

// Free lane point, not on a curb. Does not clear collided_before_goal.
static int place_car(Drive* env, int agent_idx) {
    Entity* e = &env->entities[agent_idx];
    int npts = env->lane_n;
    int tries = npts * 4;
    if (tries < 256) {
        tries = 256;
    }
    for (int attempt = 0; attempt < tries; attempt++) {
        int k = (int)(rand_r(&env->rng) % (unsigned)npts);
        if (env->lane_hx[k] == 0.0f && env->lane_hy[k] == 0.0f) {
            continue;
        }
        sit_on_lane(env, e, k);
        e->reached_goal = 0;
        e->valid = 1;
        e->lane_node = k;
        if (spot_taken(env, agent_idx)) {
            e->x = INVALID_POSITION;
            e->y = INVALID_POSITION;
            continue;
        }
        collision_check(env, agent_idx);
        if (e->collision_state == NO_COLLISION) {
            return 1;
        }
        e->collision_state = NO_COLLISION;
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
    }
    e->x = INVALID_POSITION;
    e->y = INVALID_POSITION;
    return 0;
}

// Cars first, then a goal on each forward centerline. A goal chosen
// before the later cars exist can sit past a car placed afterward.
// Does not clear collided_before_goal. The caller does that here.
void initialize_env(Drive* env) {
    int npts = env->lane_n;
    assert(npts > 1);
    int n = env->active_agent_count;
    float lo2 = env->goal_distance_min * env->goal_distance_min;
    float hi2 = env->goal_distance_max * env->goal_distance_max;
    for (int i = 0; i < n; i++) {
        env->entities[i].x = INVALID_POSITION;
        env->entities[i].y = INVALID_POSITION;
        env->entities[i].collided_before_goal = 0;
        env->entities[i].reached_goal = 0;
        env->entities[i].goals_sampled = 0;
        env->entities[i].goals_completed = 0;
        env->entities[i].goals_clean = 0;
        env->entities[i].collided_this_goal = 0;
        env->entities[i].failed = 0;
    }
    for (int i = 0; i < n; i++) {
        assert(place_car(env, i) && "no free lane point");
    }
    int rounds = npts * 4;
    if (rounds < 64) {
        rounds = 64;
    }
    int assigned = 0;
    for (int round = 0; round < rounds && !assigned; round++) {
        int fail = -1;
        for (int i = 0; i < n; i++) {
            Entity* e = &env->entities[i];
            if (!pick_forward_goal(env, e, i)) {
                fail = i;
                break;
            }
        }
        if (fail < 0) {
            assigned = 1;
            break;
        }
        env->entities[fail].x = INVALID_POSITION;
        env->entities[fail].y = INVALID_POSITION;
        assert(place_car(env, fail) && "no free lane point");
    }
    assert(assigned && "no forward goal short of the next car");
    for (int i = 0; i < n; i++) {
        collision_check(env, i);
        assert(env->entities[i].collision_state == NO_COLLISION);
        Entity* e = &env->entities[i];
        float dx = e->goal_position_x - e->x;
        float dy = e->goal_position_y - e->y;
        float d2 = dx * dx + dy * dy;
        assert(d2 >= lo2 && d2 <= hi2);
        assert(dx * e->heading_x + dy * e->heading_y > 0.0f);
        float speed = ((float)rand_r(&env->rng) / (float)RAND_MAX)
            * SPAWN_SPEED_MAX_MPS;
        e->vx = speed * e->heading_x;
        e->vy = speed * e->heading_y;
        e->vz = 0.0f;
        assert(speed >= 0.0f && speed <= SPAWN_SPEED_MAX_MPS + 1.0e-4f);
        assert(e->vx * e->heading_x + e->vy * e->heading_y >= -1.0e-4f);
        lane_reach(env, e->lane_node);
        lane_blocks(env, i);
        int open = 0;
        for (int k = 0; k < npts; k++) {
            if (env->lane_x[k] != e->goal_position_x
                    || env->lane_y[k] != e->goal_position_y) {
                continue;
            }
            float path = env->lane_dist[k];
            if (path > 0.0f && path <= open_path_limit(env->lane_block[k])) {
                open = 1;
                break;
            }
        }
        assert(open);
        for (int j = i + 1; j < n; j++) {
            assert(!check_aabb_collision(e, &env->entities[j]));
        }
        e->goals_sampled = 1;
    }
}

// Logged pose and logged goal. scored cars count in the return.
// A null spawn is a pad: off the map and out of the score.
static void seat_logged(Entity* e, LogSpawn* s, int scored) {
    e->collided_before_goal = 0;
    e->reached_goal = 0;
    e->goals_sampled = 0;
    e->goals_completed = 0;
    e->goals_clean = 0;
    e->collided_this_goal = 0;
    e->failed = 0;
    e->dyn_ready = 0;
    e->lane_node = -1;
    e->offroad_state = 0;
    e->path_lat = 0.0f;
    e->path_lon = 0.0f;
    e->path_n = 0;
    e->valid = 0;
    e->vx = 0.0f;
    e->vy = 0.0f;
    e->vz = 0.0f;
    e->collision_state = NO_COLLISION;
    if (!s) {
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
        return;
    }
    e->x = s->x;
    e->y = s->y;
    e->z = 0.0f;
    e->heading = s->heading;
    e->heading_x = cosf(e->heading);
    e->heading_y = sinf(e->heading);
    e->vx = s->vx;
    e->vy = s->vy;
    float length = s->length * COLLISION_BOX_SCALE;
    float width = s->width * COLLISION_BOX_SCALE;
    if ((s->type == VEHICLE || s->type == TRUCK) && length < 1.0f) {
        length = CAR_LENGTH * COLLISION_BOX_SCALE;
    }
    if ((s->type == VEHICLE || s->type == TRUCK) && width < 0.5f) {
        width = CAR_WIDTH * COLLISION_BOX_SCALE;
    }
    e->type = s->type;
    e->length = length;
    e->width = width;
    e->valid = 1;
    write_goal(e, s->gx, s->gy);
    e->goals_sampled = scored;
}

// Policy slot 0 is the self-driving car. Partners and training ghosts
// sit past active_agent_count and are not scored. A ghost stays off
// the map until its log frame is valid.
static void place_logged(Drive* env) {
    int n = env->active_agent_count;
    int sdc = env->control_mode == CONTROL_SDC_ONLY;
    for (int i = 0; i < env->body_count; i++) {
        LogSpawn* s = NULL;
        int scored = 0;
        if (sdc && i == 0 && env->spawn_n > 0) {
            s = &env->spawns[0];
            scored = 1;
        } else if (sdc && i >= n && i - n + 1 < env->spawn_n) {
            s = &env->spawns[i - n + 1];
        } else if (!sdc && i < env->spawn_n) {
            s = &env->spawns[i];
            scored = 1;
        } else if (!sdc && i >= n && i - n < env->ghost_n) {
            s = &env->ghosts[i - n];
        }
        seat_logged(&env->entities[i], s, scored);
    }
    env->bins_ready = 0;
}

// Log frame is init_steps plus the sim clock. An invalid flag, or a
// frame past the log, parks the body. The policy car is series 0 in
// a human replay and is not moved. Training ghosts start at series 0.
static void replay_partners(Drive* env) {
    if (!env->replay_x) {
        return;
    }
    int sdc = env->control_mode == CONTROL_SDC_ONLY;
    int frame = env->init_steps + env->timestep;
    int len = env->replay_len;
    int count = env->ghost_n;
    int series = 0;
    if (sdc) {
        count = env->spawn_n - 1;
        series = 1;
    }
    if (count < 0) {
        count = 0;
    }
    int live = frame >= 0 && frame < len;
    for (int i = 0; i < count; i++) {
        Entity* e = &env->entities[env->active_agent_count + i];
        int k = (series + i) * len + frame;
        if (!live || !env->replay_v[k]) {
            e->x = INVALID_POSITION;
            e->y = INVALID_POSITION;
            e->vx = 0.0f;
            e->vy = 0.0f;
            e->valid = 0;
            continue;
        }
        e->x = env->replay_x[k];
        e->y = env->replay_y[k];
        e->heading = env->replay_h[k];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->vx = env->replay_vx[k];
        e->vy = env->replay_vy[k];
        e->valid = 1;
    }
    env->bins_ready = 0;
}

void puf_reset(Drive* env) {
    env->timestep = 0;
    set_start_position(env);
    if (env->log_spawn) {
        place_logged(env);
        replay_partners(env);
    } else {
        initialize_env(env);
    }
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i] = (Log){0};
        env->entities[i].dyn_ready = 0;
    }
    env->wosac_fill = 0;
    compute_observations(env);
}

// Centered poses, one record per finished horizon. Python scores them.
static void wosac_record(Drive* env) {
    if (!env->wosac_on || env->wosac_fill >= env->episode_length) {
        return;
    }
    int steps = env->episode_length;
    int t = env->wosac_fill++;
    for (int i = 0; i < env->spawn_n; i++) {
        int k = i * steps + t;
        env->wosac_x[k] = env->entities[i].x;
        env->wosac_y[k] = env->entities[i].y;
        env->wosac_h[k] = env->entities[i].heading;
    }
}

static void wosac_dump(Drive* env) {
    if (!env->wosac_on || env->wosac_fill != env->episode_length) {
        return;
    }
    char path[576];
    int nw = snprintf(path, sizeof(path), "%s/map_%04d.bin",
        env->wosac_dir, env->map_id);
    assert(nw > 0 && nw < (int)sizeof(path));
    FILE* f = fopen(path, "ab");
    assert(f);
    int n = env->spawn_n;
    int steps = env->episode_length;
    int roll = env->wosac_rolls;
    assert(fwrite(&roll, sizeof(int), 1, f) == 1);
    assert(fwrite(&n, sizeof(int), 1, f) == 1);
    assert(fwrite(&steps, sizeof(int), 1, f) == 1);
    int ids[DRIVE_AGENT_MAX];
    int types[DRIVE_AGENT_MAX];
    float lens[DRIVE_AGENT_MAX];
    float widths[DRIVE_AGENT_MAX];
    assert(n <= DRIVE_AGENT_MAX);
    for (int i = 0; i < n; i++) {
        ids[i] = env->spawns[i].id;
        types[i] = env->spawns[i].type;
        lens[i] = env->spawns[i].length;
        widths[i] = env->spawns[i].width;
    }
    if (n > 0) {
        assert(fwrite(ids, sizeof(int), n, f) == (size_t)n);
        assert(fwrite(types, sizeof(int), n, f) == (size_t)n);
        assert(fwrite(lens, sizeof(float), n, f) == (size_t)n);
        assert(fwrite(widths, sizeof(float), n, f) == (size_t)n);
        size_t m = (size_t)n * (size_t)steps;
        assert(fwrite(env->wosac_x, sizeof(float), m, f) == m);
        assert(fwrite(env->wosac_y, sizeof(float), m, f) == m);
        assert(fwrite(env->wosac_h, sizeof(float), m, f) == m);
    }
    assert(fclose(f) == 0);
    env->wosac_rolls++;
}

static void wosac_meta(Drive* env) {
    char path[576];
    int nw = snprintf(path, sizeof(path), "%s/map_%04d.meta",
        env->wosac_dir, env->map_id);
    assert(nw > 0 && nw < (int)sizeof(path));
    FILE* f = fopen(path, "wb");
    assert(f);
    assert(fwrite(&env->world_mean_x, sizeof(float), 1, f) == 1);
    assert(fwrite(&env->world_mean_y, sizeof(float), 1, f) == 1);
    int nedges = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type == ROAD_EDGE) {
            nedges++;
        }
    }
    assert(fwrite(&nedges, sizeof(int), 1, f) == 1);
    for (int i = env->num_objects; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type != ROAD_EDGE) {
            continue;
        }
        assert(fwrite(&e->array_size, sizeof(int), 1, f) == 1);
        assert(fwrite(e->pts_x, sizeof(float), e->array_size, f)
            == (size_t)e->array_size);
        assert(fwrite(e->pts_y, sizeof(float), e->array_size, f)
            == (size_t)e->array_size);
    }
    assert(fclose(f) == 0);
}

static void human_controls(Drive* env) {
    if (env->sim_lock_actions) {
        return;
    }
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (env->active_agent_count <= 0) {
        return;
    }
    if (env->selected_agent_idx < 0
            || env->selected_agent_idx >= env->active_agent_count) {
        env->selected_agent_idx = 0;
    }
    int idx = env->selected_agent_idx;
    float* atn = env->agents[idx].actions;
    int accel_kick = (NUM_ACCEL_BINS - 1) / 3;
    int steer_kick = (NUM_STEER_BINS - 1) / 3;
    atn[0] = ACCEL_COAST_BIN;
    atn[1] = STEER_COAST_BIN;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        atn[0] += accel_kick;
    }
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        atn[0] -= accel_kick;
    }
    if (atn[0] < 0) {
        atn[0] = 0;
    }
    if (atn[0] > NUM_ACCEL_BINS - 1) {
        atn[0] = NUM_ACCEL_BINS - 1;
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        atn[1] += steer_kick;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        atn[1] -= steer_kick;
    }
    if (atn[1] < 0) {
        atn[1] = 0;
    }
    if (atn[1] > NUM_STEER_BINS - 1) {
        atn[1] = NUM_STEER_BINS - 1;
    }
    if (IsKeyPressed(KEY_TAB)) {
        env->selected_agent_idx = (env->selected_agent_idx + 1) % env->active_agent_count;
    }
}

static int try_respawn_pose(Drive* env, int agent_idx, int want_new_goal,
        float gx, float gy) {
    Entity* e = &env->entities[agent_idx];
    int npts = env->lane_n;
    int tries = npts * 2;
    if (tries < 64) {
        tries = 64;
    }
    float keep2 = MIN_DISTANCE_TO_GOAL * MIN_DISTANCE_TO_GOAL;
    for (int attempt = 0; attempt < tries; attempt++) {
        int k = (int)(rand_r(&env->rng) % (unsigned)npts);
        if (env->lane_hx[k] == 0.0f && env->lane_hy[k] == 0.0f) {
            continue;
        }
        sit_on_lane(env, e, k);
        if (spot_taken(env, agent_idx)) {
            continue;
        }
        collision_check(env, agent_idx);
        if (e->collision_state != NO_COLLISION) {
            e->collision_state = NO_COLLISION;
            continue;
        }
        e->lane_node = k;
        if (want_new_goal) {
            if (!pick_forward_goal(env, e, agent_idx)) {
                continue;
            }
        } else {
            float dx = e->x - gx;
            float dy = e->y - gy;
            if (dx * dx + dy * dy < keep2) {
                continue;
            }
            write_goal(e, gx, gy);
        }
        e->goals_sampled++;
        e->collided_this_goal = 0;
        return 1;
    }
    return 0;
}

static void respawn_agent(Drive* env, int agent_idx) {
    Entity* e = &env->entities[agent_idx];
    float ox = e->x;
    float oy = e->y;
    float oh = e->heading;
    float ohx = e->heading_x;
    float ohy = e->heading_y;
    float ovx = e->vx;
    float ovy = e->vy;
    float gx = e->goal_position_x;
    float gy = e->goal_position_y;
    int onode = e->lane_node;
    int node = -1;
    float best_d = LANE_NODE_SNAP * LANE_NODE_SNAP;
    for (int i = 0; i < env->lane_n; i++) {
        float dot = ohx * env->lane_hx[i] + ohy * env->lane_hy[i];
        if (dot <= LANE_HEADING_DOT) {
            continue;
        }
        float dx = env->lane_x[i] - ox;
        float dy = env->lane_y[i] - oy;
        float d2 = dx * dx + dy * dy;
        if (d2 <= best_d) {
            best_d = d2;
            node = i;
        }
    }
    if (node >= 0) {
        e->lane_node = node;
        if (pick_forward_goal(env, e, agent_idx)) {
            e->goals_sampled++;
            e->collided_this_goal = 0;
            return;
        }
    }
    if (try_respawn_pose(env, agent_idx, 1, gx, gy)
            || try_respawn_pose(env, agent_idx, 0, gx, gy)) {
        return;
    }
    e->x = ox;
    e->y = oy;
    e->heading = oh;
    e->heading_x = ohx;
    e->heading_y = ohy;
    e->vx = ovx;
    e->vy = ovy;
    e->lane_node = onode;
    e->goal_position_x = gx;
    e->goal_position_y = gy;
    e->reached_goal = 1;
}

// Gigaflow's velocity term, with the speed floor replaced by the band.
// SPEED_ALPHA * dt * max(lane alignment, 0). Speeds are m/s.
float speed_bonus(Drive* env, Entity* agent) {
    if (env->target_speed <= 0.0f) {
        return 0.0f;
    }
    float signed_speed = agent->vx * agent->heading_x
        + agent->vy * agent->heading_y;
    float err = signed_speed - env->target_speed;
    if (err < 0.0f) {
        err = -err;
    }
    if (err > env->speed_band) {
        return 0.0f;
    }
    int list[MAX_ENTITIES_PER_CELL * 2 * 25];
    int n = check_neighbors(env, agent->x, agent->y, list,
        MAX_ENTITIES_PER_CELL * 2 * 25, collision_offsets, 25);
    float best = -1.0f;
    float lane_x = 0.0f;
    float lane_y = 0.0f;
    for (int i = 0; i < n; i += 2) {
        Entity* road = &env->entities[list[i]];
        int seg = list[i + 1];
        if (road->type != ROAD_LANE) {
            continue;
        }
        float ax = road->pts_x[seg];
        float ay = road->pts_y[seg];
        float bx = road->pts_x[seg + 1];
        float by = road->pts_y[seg + 1];
        float dx = bx - ax;
        float dy = by - ay;
        float len2 = dx * dx + dy * dy;
        if (len2 <= 0.0f) {
            continue;
        }
        float t = ((agent->x - ax) * dx + (agent->y - ay) * dy) / len2;
        if (t < 0.0f) {
            t = 0.0f;
        } else if (t > 1.0f) {
            t = 1.0f;
        }
        float ox = agent->x - (ax + t * dx);
        float oy = agent->y - (ay + t * dy);
        float d2 = ox * ox + oy * oy;
        if (best >= 0.0f && d2 >= best) {
            continue;
        }
        best = d2;
        lane_x = dx;
        lane_y = dy;
    }
    if (best < 0.0f) {
        return 0.0f;
    }
    float inv = 1.0f / sqrtf(lane_x * lane_x + lane_y * lane_y);
    float align = agent->heading_x * lane_x * inv
        + agent->heading_y * lane_y * inv;
    if (align < 0.0f) {
        align = 0.0f;
    }
    return SPEED_ALPHA * SIM_DT * align;
}

// Arc length along the logged SDC track. lateral is meters to the
// closest logged point. longitudinal is signed meters versus the
// log at this frame, positive when the car is ahead of the log.
static float route_progress(Drive* env, float px, float py, int frame,
        float* lateral, float* longitudinal) {
    int len = env->replay_len;
    float* xs = env->replay_x;
    float* ys = env->replay_y;
    unsigned char* vs = env->replay_v;
    float cumulative = 0.0f;
    float d_p = 0.0f;
    float d_q = 0.0f;
    float d_xt = 0.0f;
    float d_now = 0.0f;
    float best = 1.0e30f;
    float prev_x = xs[0];
    float prev_y = ys[0];
    int t_lookup = frame;
    int init = env->init_steps;
    if (t_lookup < 0) {
        t_lookup = 0;
    }
    if (t_lookup >= len) {
        t_lookup = len - 1;
    }
    if (init < 0) {
        init = 0;
    }
    if (init >= len) {
        init = len - 1;
    }
    for (int t = 0; t < len; t++) {
        if (!vs[t]) {
            prev_x = xs[t];
            prev_y = ys[t];
            continue;
        }
        if (t > 0 && vs[t - 1]) {
            float dx = xs[t] - prev_x;
            float dy = ys[t] - prev_y;
            cumulative += sqrtf(dx * dx + dy * dy);
        }
        if (t == init) {
            d_p = cumulative;
        }
        if (t == t_lookup) {
            d_now = cumulative;
        }
        d_q = cumulative;
        float dx = px - xs[t];
        float dy = py - ys[t];
        float dist = dx * dx + dy * dy;
        if (dist < best) {
            best = dist;
            d_xt = cumulative;
        }
        prev_x = xs[t];
        prev_y = ys[t];
    }
    *lateral = sqrtf(best);
    *longitudinal = d_xt - d_now;
    float denom = d_q - d_p;
    if (denom < 1.0e-3f) {
        return 0.0f;
    }
    float ratio = (d_xt - d_p) / denom;
    if (ratio < 0.0f) {
        return 0.0f;
    }
    return ratio;
}

// Pads (goals_sampled == 0) are not cars. A logged scene ends when
// every spawned car has arrived, or at episode_length, whichever
// comes first. That length is this log's state count, unless the
// ini is shorter. goal_stop runs the configured horizon. CARLA
// respawns while goal_resample is on, and ends the same way as a
// log when it is off.
static void end_episode(Drive* env) {
    env->episode_ends++;
    if (env->wosac_on) {
        assert(env->wosac_fill == env->episode_length);
        wosac_dump(env);
        env->log.n += 1.0f;
        puf_reset(env);
        return;
    }
    if (env->control_mode == CONTROL_SDC_ONLY && env->replay_x
            && env->entities[0].goals_sampled > 0) {
        Entity* sdc = &env->entities[0];
        Log* lg = &env->logs[0];
        if (sdc->path_n > 0) {
            lg->lateral_error = sdc->path_lat / (float)sdc->path_n;
            lg->longitudinal_error = sdc->path_lon / (float)sdc->path_n;
        }
        if (!sdc->reached_goal && sdc->x != INVALID_POSITION) {
            float lat = 0.0f;
            float lon = 0.0f;
            int frame = env->init_steps + env->timestep - 1;
            lg->route_progress = route_progress(
                env, sdc->x, sdc->y, frame, &lat, &lon);
        }
    }
    for (int i = 0; i < env->active_agent_count; i++) {
        if (env->entities[i].goals_sampled <= 0) {
            continue;
        }
        float sampled = (float)env->entities[i].goals_sampled;
        float completion = (float)env->entities[i].goals_completed / sampled;
        env->logs[i].completion_rate = completion;
        env->log.completion_rate += completion;
        env->log.offroad_rate += env->logs[i].offroad_rate;
        env->log.collision_rate += env->logs[i].collision_rate;
        env->log.collision_count += env->logs[i].collision_count;
        env->log.steps_before_first_failure += env->logs[i].steps_before_first_failure;
        env->log.total_distance_travelled += env->logs[i].total_distance_travelled;
        env->log.total_infractions += env->logs[i].total_infractions;
        env->log.distance_before_first_failure += env->logs[i].distance_before_first_failure;
        int reached = env->entities[i].goals_completed;
        float safe = 0.0f;
        if (reached > 0) {
            safe = (float)env->entities[i].goals_clean / (float)reached;
        }
        env->logs[i].score = safe;
        env->log.score += env->logs[i].score;
        env->log.jerk += env->logs[i].jerk;
        env->log.perf += env->logs[i].perf;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.speed_sum += env->logs[i].speed_sum;
        env->log.speed_steps += env->logs[i].speed_steps;
        env->log.at_fault_collision_rate +=
            env->logs[i].at_fault_collision_rate;
        env->log.route_progress += env->logs[i].route_progress;
        env->log.lateral_error += env->logs[i].lateral_error;
        env->log.longitudinal_error += env->logs[i].longitudinal_error;
        env->log.n += 1;
    }
    puf_reset(env);
}

static int logs_finished(Drive* env) {
    int any = 0;
    for (int i = 0; i < env->active_agent_count; i++) {
        if (env->entities[i].goals_sampled <= 0) {
            continue;
        }
        any = 1;
        if (!env->entities[i].reached_goal) {
            return 0;
        }
    }
    return any;
}

void puf_step(Drive* env) {
    human_controls(env);
    if (!env->entities || env->active_agent_count <= 0) {
        return;
    }
    wosac_record(env);
    for (int i = 0; i < env->active_agent_count; i++) {
        if (!env->agents[i].rewards) {
            continue;
        }
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
        int pad = env->entities[i].goals_sampled <= 0;
        int done = env->entities[i].reached_goal && !env->goal_stop;
        int hold = env->log_spawn || !env->goal_resample;
        if (hold && (pad || done)) {
            env->agents[i].terminals[0] = 1;
        }
    }
    env->timestep++;

    if (env->timestep == env->episode_length) {
        end_episode(env);
        return;
    }
    replay_partners(env);

    float step_jerk[DRIVE_AGENT_MAX] = {0};
    float step_distance[DRIVE_AGENT_MAX] = {0};
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* agent = &env->entities[i];
        if (agent->x == INVALID_POSITION) {
            continue;
        }
        if (env->goal_stop && agent->reached_goal) {
            agent->vx = 0.0f;
            agent->vy = 0.0f;
            continue;
        }
        env->logs[i].episode_length += 1;
        agent->collision_state = NO_COLLISION;
        // |Δv| / dt, same term as PufferDrive classic dynamics. The
        // coefficient is reward_jerk (that code uses -0.0002).
        float prev_vx = agent->vx;
        float prev_vy = agent->vy;
        move_dynamics(env, i);
        // Path length for meters per infraction. Includes the contact step.
        float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
        step_distance[i] = speed * SIM_DT;
        env->logs[i].total_distance_travelled += step_distance[i];
        env->logs[i].speed_sum += speed;
        env->logs[i].speed_steps += 1.0f;
        float dvx = agent->vx - prev_vx;
        float dvy = agent->vy - prev_vy;
        step_jerk[i] = env->reward_jerk
            * sqrtf(dvx * dvx + dvy * dvy) / SIM_DT;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* agent = &env->entities[i];
        if (agent->x == INVALID_POSITION) {
            continue;
        }
        agent->collision_state = NO_COLLISION;
        int other = collision_check(env, i);
        int collision_state = agent->collision_state;
        if (collision_state == AGENT_COLLISION) {
            env->agents[i].rewards[0] = env->reward_vehicle_collision;
            env->logs[i].episode_return += env->reward_vehicle_collision;
            env->logs[i].collision_rate = 1.0f;
            env->logs[i].collision_count += 1.0f;
            // NAVSIM at-fault: the other body is ahead and this car
            // is moving toward it. A replay car that arrives from
            // behind does not set the flag.
            if (other >= 0) {
                Entity* b = &env->entities[other];
                float dx = b->x - agent->x;
                float dy = b->y - agent->y;
                float forward = dx * agent->heading_x + dy * agent->heading_y;
                float approach = agent->vx * dx + agent->vy * dy;
                if (forward > 0.0f && approach > 0.0f) {
                    env->logs[i].at_fault_collision_rate = 1.0f;
                }
            }
        } else if (collision_state == OFFROAD) {
            env->agents[i].rewards[0] = env->reward_offroad_collision;
            env->logs[i].offroad_rate = 1.0f;
            env->logs[i].episode_return += env->reward_offroad_collision;
        }
        if (agent->offroad_state) {
            env->logs[i].offroad_rate = 1.0f;
        }
        if (collision_state > NO_COLLISION) {
            env->logs[i].total_infractions += 1.0f;
            agent->collided_this_goal = 1;
            if (agent->goals_completed == 0) {
                agent->collided_before_goal = 1;
            }
            agent->failed = 1;
        } else if (!agent->failed) {
            env->logs[i].steps_before_first_failure += 1.0f;
            env->logs[i].distance_before_first_failure += step_distance[i];
        }

        float distance_to_goal = relative_distance_2d(
            agent->x, agent->y,
            agent->goal_position_x, agent->goal_position_y);

        // Inside the goal radius, once per goal. Respawn samples the next one.
        // Staying on the same point does not pay again. goal_stop also
        // requires speed at or under goal_speed, matching PufferDrive.
        float radius = MIN_DISTANCE_TO_GOAL;
        if (env->goal_radius > 0.0f) {
            radius = env->goal_radius;
        }
        int arrived = distance_to_goal < radius && !agent->reached_goal;
        if (arrived && env->goal_stop) {
            float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
            if (speed > env->goal_speed) {
                arrived = 0;
            }
        }
        if (env->control_mode == CONTROL_SDC_ONLY && env->replay_x
                && i == 0 && agent->goals_sampled > 0
                && agent->x != INVALID_POSITION) {
            float lat = 0.0f;
            float lon = 0.0f;
            int frame = env->init_steps + env->timestep;
            route_progress(env, agent->x, agent->y, frame, &lat, &lon);
            agent->path_lat += lat;
            agent->path_lon += fabsf(lon);
            agent->path_n += 1;
        }

        if (arrived) {
            env->agents[i].rewards[0] += env->reward_goal;
            env->logs[i].episode_return += env->reward_goal;
            agent->reached_goal = 1;
            if (env->control_mode == CONTROL_SDC_ONLY && env->replay_x
                    && i == 0) {
                env->logs[i].route_progress = 1.0f;
            }
            agent->goals_completed++;
            if (!agent->collided_this_goal) {
                agent->goals_clean++;
            }
            // First clean arrival. Score is the full-episode safe share.
            if (agent->goals_completed == 1 && !agent->collided_before_goal) {
                env->logs[i].perf = 1.0f;
            }
        }

        float jerk_penalty = step_jerk[i];
        env->agents[i].rewards[0] += jerk_penalty;
        env->logs[i].episode_return += jerk_penalty;
        env->logs[i].jerk += jerk_penalty;
        float bonus = speed_bonus(env, agent);
        env->agents[i].rewards[0] += bonus;
        env->logs[i].episode_return += bonus;
        int remove = env->log_spawn || !env->goal_resample;
        if (arrived && env->log_spawn && env->goal_stop) {
            agent->vx = 0.0f;
            agent->vy = 0.0f;
        } else if (arrived && remove) {
            env->agents[i].terminals[0] = 1;
            agent->vx = 0.0f;
            agent->vy = 0.0f;
            agent->x = INVALID_POSITION;
            agent->y = INVALID_POSITION;
            env->bins_ready = 0;
        } else if (arrived) {
            respawn_agent(env, i);
        }
    }

    // Human-replay runs to episode_length. Stopping at the goal would
    // reset inside a trainer horizon and count another episode.
    // Training ends once every scored car has arrived. A recording
    // leaves that reset to the player: each arrival still parks that
    // car, and the player stops when every followed car has arrived.
    // goal_resample off uses that same early end on a road map.
    if ((env->log_spawn || !env->goal_resample) && !env->goal_stop
            && env->control_mode != CONTROL_SDC_ONLY
            && !puf_record_enabled()
            && logs_finished(env)) {
        end_episode(env);
        return;
    }

    compute_observations(env);
}

void puf_init(Env* env, Dict* kwargs) {
    init_action_space();
    env->selected_agent_idx = dict_get(kwargs, "selected_agent_idx");
    env->reward_vehicle_collision = dict_get(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = dict_get(kwargs, "reward_offroad_collision");
    env->reward_goal = dict_get(kwargs, "reward_goal");
    env->reward_jerk = dict_get(kwargs, "reward_jerk");
    // Ini values are km/h. A target of 0 pays nothing.
    env->target_speed = dict_get(kwargs, "target_speed") * KMH_TO_MPS;
    env->speed_band = dict_get(kwargs, "speed_band") * KMH_TO_MPS;
    env->cfg = kwargs;
    env->use_action_values = 0;
    env->replay_invalid = 1;
    DictItem* replay_off = dict_find(kwargs, "replay_invalid");
    if (replay_off && replay_off->value == 0.0) {
        env->replay_invalid = 0;
    }
    DictItem* control = dict_find(kwargs, "control_mode");
    env->control_mode = drive_control_mode(
        (control && control->str && control->str[0])
            ? control->str : "control_agents");
    DictItem* cap = dict_find(kwargs, "max_controlled_agents");
    env->max_controlled = 32;
    if (cap && cap->value > 0.0) {
        env->max_controlled = (int)cap->value;
    }
    int map_id = dict_get(kwargs, "map_id");
    env->map_id = map_id;
    env->resample_frequency = 0;
    env->resample_episodes = 0;
    env->episode_ends = 0;
    DictItem* every = dict_find(kwargs, "resample_frequency");
    if (every && every->value > 0.0) {
        env->resample_frequency = (int)every->value;
    }
    DictItem* every_ep = dict_find(kwargs, "resample_episodes");
    if (every_ep && every_ep->value > 0.0) {
        env->resample_episodes = (int)every_ep->value;
    }
    char bin_dir[480];
    char map_file[512];
    snprintf(bin_dir, sizeof(bin_dir), "%s/binaries", dict_get_str(kwargs, "data_dir"));
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", bin_dir, map_id);
    env->map_name = strdup(map_file);
    env->max_agents = (int)dict_get(kwargs, "num_agents");
    env->goal_distance_min = (float)dict_get(kwargs, "goal_distance_min");
    env->goal_distance_max = (float)dict_get(kwargs, "goal_distance_max");
    env->timestep = 0;
    env->episode_length = (int)dict_get(kwargs, "episode_length");
    env->init_steps = 0;
    env->goal_radius = 0.0f;
    env->goal_speed = 100.0f;
    env->goal_stop = 0;
    env->wosac_on = 0;
    env->wosac_fill = 0;
    env->wosac_rolls = 0;
    env->wosac_dir[0] = 0;
    DictItem* init_at = dict_find(kwargs, "init_steps");
    if (init_at && init_at->value > 0.0) {
        env->init_steps = (int)init_at->value;
    }
    DictItem* radius = dict_find(kwargs, "goal_radius");
    if (radius && radius->value > 0.0) {
        env->goal_radius = (float)radius->value;
    }
    DictItem* gspeed = dict_find(kwargs, "goal_speed");
    if (gspeed && gspeed->value > 0.0) {
        env->goal_speed = (float)gspeed->value;
    }
    DictItem* gstop = dict_find(kwargs, "goal_stop");
    if (gstop && gstop->value != 0.0) {
        env->goal_stop = 1;
    }
    env->goal_resample = 1;
    DictItem* gres = dict_find(kwargs, "goal_resample");
    if (gres && gres->value == 0.0) {
        env->goal_resample = 0;
    }
    DictItem* wdir = dict_find(kwargs, "wosac_dir");
    if (wdir && wdir->str && wdir->str[0]
            && strcmp(wdir->str, "none") != 0) {
        assert(strlen(wdir->str) < sizeof(env->wosac_dir));
        memcpy(env->wosac_dir, wdir->str, strlen(wdir->str) + 1);
        env->wosac_on = 1;
    }
    load_world(env);
    // Waymo training stops at the scenario length (91). The ini value
    // is the longest log, so a shorter scene is not held open past its
    // last frame. Human-replay and goal_stop keep the ini horizon.
    if (env->log_spawn && !env->wosac_on
            && env->control_mode != CONTROL_SDC_ONLY
            && !env->goal_stop && env->log_traj > 1
            && env->episode_length > env->log_traj) {
        env->episode_length = env->log_traj;
    }
    if (env->wosac_on) {
        int steps = env->episode_length;
        assert(steps > 0 && steps <= 256);
        size_t m = (size_t)env->num_agents * (size_t)steps;
        env->wosac_x = (float*)calloc(m, sizeof(float));
        env->wosac_y = (float*)calloc(m, sizeof(float));
        env->wosac_h = (float*)calloc(m, sizeof(float));
        assert(env->wosac_x && env->wosac_y && env->wosac_h);
        wosac_meta(env);
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

// Swap the logged scene. Agent buffer pointers, policy ids, and masks stay.
// puf_init clears those, so this reloads the world only.
void puf_reload_map(Drive* env, int map_id) {
    if (map_id == env->map_id && env->entities) {
        puf_reset(env);
        return;
    }
    char bin_dir[480];
    char map_file[512];
    const char* data_dir = dict_get_str(env->cfg, "data_dir");
    snprintf(bin_dir, sizeof(bin_dir), "%s/binaries", data_dir);
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", bin_dir, map_id);
    release_world(env);
    free(env->map_name);
    env->map_name = strdup(map_file);
    env->map_id = map_id;
    // A short log clamps episode_length. The next file may be longer.
    env->episode_length = (int)dict_get(env->cfg, "episode_length");
    load_world(env);
    if (env->log_spawn && !env->wosac_on
            && env->control_mode != CONTROL_SDC_ONLY
            && !env->goal_stop && env->log_traj > 1
            && env->episode_length > env->log_traj) {
        env->episode_length = env->log_traj;
    }
    if (env->wosac_on) {
        int steps = env->episode_length;
        assert(steps > 0 && steps <= 256);
        size_t m = (size_t)env->num_agents * (size_t)steps;
        env->wosac_x = (float*)calloc(m, sizeof(float));
        env->wosac_y = (float*)calloc(m, sizeof(float));
        env->wosac_h = (float*)calloc(m, sizeof(float));
        assert(env->wosac_x && env->wosac_y && env->wosac_h);
        wosac_meta(env);
    }
    puf_reset(env);
}

// Logged as env/score_town_01 and so on. Town10HD is score_town_10.
static const char* drive_score_town_key(int map_id) {
    switch (map_id) {
    case 0: return "score_town_01";
    case 1: return "score_town_02";
    case 2: return "score_town_06";
    case 3: return "score_town_10";
    default: return NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "jerk", log->jerk);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "offroad_rate", log->offroad_rate);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "collision_count", log->collision_count);
    dict_set(out, "steps_before_first_failure", log->steps_before_first_failure);
    dict_set(out, "distance_before_first_failure", log->distance_before_first_failure);
    float avg_distance_per_infraction = log->total_distance_travelled;
    if (log->total_infractions > 0.0f) {
        avg_distance_per_infraction = log->total_distance_travelled / log->total_infractions;
    }
    dict_set(out, "avg_distance_per_infraction", avg_distance_per_infraction);
    dict_set(out, "n", log->n);
    dict_set(out, "completion_rate", log->completion_rate);
    dict_set(out, "at_fault_collision_rate", log->at_fault_collision_rate);
    dict_set(out, "route_progress", log->route_progress);
    dict_set(out, "lateral_error", log->lateral_error);
    dict_set(out, "longitudinal_error", log->longitudinal_error);
    // km/h. Mean over every active car-step in the finished episodes.
    float avg_speed = 0.0f;
    if (log->speed_steps > 0.0f) {
        avg_speed = (log->speed_sum / log->speed_steps) / KMH_TO_MPS;
    }
    dict_set(out, "avg_speed", avg_speed);
}

// Window, camera, and HUD.
#include "render.h"

