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
// Window playback. pufferenv.h supplies a no-op for other envs.
#define PUF_HAS_SIM_CONTROL
#define PUF_HAS_AGENT_SLIDER
#include "pufferenv.h"
#define float3 rl_float3
#define double3 rl_double3
#include "raymath.h"
#undef float3
#undef double3
#include "rlgl.h"
#include "puf_record.h"
#include <time.h>

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

#define INVALID_POSITION -10000.0f

// Simulation constants
#define TRAJECTORY_LENGTH 91 // Discretized Waymo scenarios
#define SIM_DT 0.1f
// Interactive playback only. 20 steps/s is 2x real time so the window is
// easier to follow. Headless eval still runs as fast as it can.
#define PUF_STEPS_PER_SEC 20

// Velocity diagram for structured flows. One heatmap. Each segment is one
// lane, listed on y. A cell is the mean speed of the cars on that lane.
// Time is on x. Colors match CIRCLES Fig. 7, 10 mph through 80 mph.
// The panel beside it marks a new crash in red and an off-road event in yellow.
#define FLOW_MPS_TO_MPH 2.2369363f
#define FLOW_MPH_LO 10.0f
#define FLOW_MPH_HI 80.0f
#define FLOW_CMAP_N 16
#define FLOW_LANE_MAX 8
#define FLOW_PANEL_HEAD 40.0f
#define FLOW_PANEL_FOOT 36.0f
#define FLOW_PANEL_ROW 40.0f
#define FLOW_PANEL_W 480.0f
#define FLOW_PANEL_GAP 8.0f
#define FLOW_HIT_MAX 1024
// Half the 3.2 m lane spacing. A vehicle is counted on the closest centerline.
#define FLOW_LATERAL 1.6f
// A collision just off the paint still belongs to the nearest lane.
#define FLOW_HIT_REACH 8.0f
// Split a new lane when centerline y jumps by more than this.
#define FLOW_LANE_GAP 1.2f
// |dy| below this is a horizontal lane. The ramp is steeper.
#define FLOW_FLAT_DY 1.0f

// Agent limits
#ifndef MAX_AGENTS
#define MAX_AGENTS 64
#endif
// Cars one map copy may spawn. The viewer slider runs from 1 to this.
#define DRIVE_AGENT_MAX 128
// Partners listed in one observation. Fixed even when more cars exist.
#define MAX_OBS_AGENTS 64

// Dynamics models
#define CLASSIC 0

// Where the initial vehicle poses come from.
// 0 replays the logged trajectories in the map binary.
// 1 places vehicles from the SUMO-style flows in the ini (ocean/drive/flow_init.h).
#define AGENT_INIT_LOGS 0
#define AGENT_INIT_STRUCTURED 1

// Collision State
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Grid Map
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 10
#define SLOTS_PER_CELL (MAX_ENTITIES_PER_CELL * 2 + 1)
#define VISION_RANGE 21

// Observation Space
#define MAX_ROAD_SEGMENT_OBSERVATIONS 75

#define PARTNER_FEATURES 7
// Index 6 stays 0 so an old checkpoint keeps that weight. Index 7
// is this episode's max speed divided by MAX_SPEED.
#define EGO_FEATURES 8
#define ROAD_FEATURES 7

#define OBS_SIZE (EGO_FEATURES + PARTNER_FEATURES * MAX_OBS_AGENTS \
    + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS)
#define ACT_SIZES {7, 13}
#define NUM_ATNS 2
#define MY_VEC_INIT

typedef Env Drive;

// Observation normalization
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f

// Observation scaling factors
#define OBS_GOAL_SCALE 0.005f
#define OBS_SPEED_SCALE 0.01f
#define OBS_POSITION_SCALE 0.02f

// Distance thresholds
#define MIN_DISTANCE_TO_GOAL 5.0f
// Steps of collision immunity after a respawn, so the car can overlap
// traffic already sitting on its spawn. Off-road is still penalized.
#define RESPAWN_GRACE 10
// Infinite: farther than the trip by this much also ends the lap.
#define RESPAWN_OVERRUN 40.0f
#define COLLISION_DIST_SQ 225.0f
// Closing-speed bounce. 0 stops at the contact, 1 sends it back at full speed.
#define BUMP_BOUNCE 0.4f
// Extra gap so the next step does not start already overlapping.
#define BUMP_SLOP 0.05f
// Cap on one shove, in meters. A bad normal cannot throw a car across the map.
#define BUMP_MAX 4.0f
#define OBS_DIST_SQ 2500.0f
#define COLLISION_BOX_SCALE 0.7f

// Action space
#define NUM_ACCEL_BINS 7
#define NUM_STEER_BINS 13

static const float ACCELERATION_VALUES[NUM_ACCEL_BINS] = {
    -4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f
};

static const float STEERING_VALUES[NUM_STEER_BINS] = {
    -1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f,
     0.167f,  0.333f,  0.500f,  0.667f,  0.833f,  1.000f
};

// Geometry helpers
static const float offsets[4][2] = {
    {-1,  1},  // top-left
    { 1,  1},  // top-right
    { 1, -1},  // bottom-right
    {-1, -1}   // bottom-left
};

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
    {-2,  0}, {-1,  0}, {0,  0}, {1,  0}, {2,  0},
    {-2,  1}, {-1,  1}, {0,  1}, {1,  1}, {2,  1},
    {-2,  2}, {-1,  2}, {0,  2}, {1,  2}, {2,  2}
};

// Rendering Colors
const Color STONE_GRAY       = (Color){80, 80, 80, 255};
const Color PUFF_RED         = (Color){187, 0, 0, 255};
const Color PUFF_CYAN        = (Color){0, 187, 187, 255};
const Color PUFF_WHITE       = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND  = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color ROAD_COLOR       = (Color){35, 35, 37, 255};

// Forward declarations
typedef struct Client Client;
typedef struct Log Log;
typedef struct Entity Entity;

struct Log {
    float episode_return;
    float episode_length;
    float perf;
    float score;
    float offroad_rate;
    float collision_rate;
    float clean_collision_rate;
    float completion_rate;
    float dnf_rate;
    float n;
};

struct Entity {
    int type;
    int array_size;
    float* traj_x;
    float* traj_y;
    float* traj_z;
    float* traj_vx;
    float* traj_vy;
    float* traj_vz;
    float* traj_heading;
    int* traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    int mark_as_expert;
    int collision_state;
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
    int respawn_timestep;
    int collided_before_goal;
    int reached_goal_this_episode;
    int active_agent;
    float prev_accel;
    float prev_steer;
    // Personal max speed in m/s. 0 leaves the bicycle clip alone.
    float max_speed;
};

void free_entity(Entity* entity) {
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
}

// Utility
float relative_distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

float clipSpeed(float speed) {
    if (speed > MAX_SPEED) return MAX_SPEED;
    if (speed < -MAX_SPEED) return -MAX_SPEED;
    return speed;
}

float normalize_heading(float heading) {
    if (heading > M_PI) heading -= 2 * M_PI;
    if (heading < -M_PI) heading += 2 * M_PI;
    return heading;
}

// Centerline piece for the velocity diagram. Endpoints are centered map
// coordinates. pos is the network-x midpoint, shared by every lane strip.
typedef struct FlowSeg {
    float ax, ay, bx, by;
    float pos;
    int lane;
} FlowSeg;

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
    int* active_agent_indices;
    int human_agent_idx;
    Entity* entities;
    int num_entities;
    int num_actors;  // Total agents (active + static)
    int num_objects;
    int num_roads;
    int static_agent_count;
    int* static_agent_indices;
    int expert_static_agent_count;
    int* expert_static_agent_indices;
    int timestep;
    int episode_length;
    int dynamics_model;
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
    float world_mean_x;
    float world_mean_y;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float reward_jerk;
    float reward_speed;
    unsigned int rng;
    int agent_initialization;
    // 1 after ./drive loads a checkpoint that matches this network.
    int checkpoint_ok;
    char checkpoint_error[256];
    // Copied routes and each car's spawn arc. Owned by flow_init.h.
    void* lane_state;
    // Mean speed (m/s) of every vehicle on a lane, per step. -1 is empty.
    // Index is time * flow_nlanes + lane. The lane is the road area.
    FlowSeg* flow_segs;
    int flow_nlanes;
    int flow_lane_beg[FLOW_LANE_MAX];
    int flow_lane_n[FLOW_LANE_MAX];
    float flow_lane_lo[FLOW_LANE_MAX];
    float flow_lane_hi[FLOW_LANE_MAX];
    float flow_pos_lo;
    float flow_pos_hi;
    char flow_lane_name[FLOW_LANE_MAX][16];
    float* flow_speed;
    int* flow_bin_count;
    int flow_nseg;
    int flow_rows;
    // Onset of each collision: absolute step, lane, and kind. A ring.
    int flow_hit_t[FLOW_HIT_MAX];
    int flow_hit_lane[FLOW_HIT_MAX];
    unsigned char flow_hit_kind[FLOW_HIT_MAX];
    int flow_hit_n;
    int flow_hit_i;
    unsigned char flow_hit_prev[DRIVE_AGENT_MAX];
    // Set from the window. Headless eval and recording never read these.
    int sim_paused;
    int sim_infinite;
    // Overhead camera stays on human_agent_idx. Off restores the map view.
    int sim_follow;
    // One frame after Reset, so that click draws timestep 0.
    int sim_hold_step;
    // Agents per map. The slider writes req. Play copies it to on.
    int sim_agents_req;
    int sim_agents_on;
    int sim_agents_apply;
    int sim_agents_drag;
    // Uniform target speed, m/s. req is staged; Play copies it to on.
    // sim_speed_max is the vtype cap recorded on the first load.
    float sim_speed_req;
    float sim_speed_on;
    float sim_speed_max;
    float sim_hi_req;
    float sim_hi_on;
    int sim_speed_apply;
    int sim_speed_drag;
    int sim_hi_apply;
    int sim_hi_drag;
    Dict* cfg;
};

static inline int puf_sim_paused(Drive* env) {
    if (puf_record_enabled()) {
        return 0;
    }
    return env->sim_paused || env->sim_hold_step;
}

void add_log(Drive* env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* e = &env->entities[env->active_agent_indices[i]];
        if (e->reached_goal_this_episode) {
            env->log.completion_rate += 1.0f;
        }
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += env->logs[i].offroad_rate;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        int clean_collided = env->logs[i].clean_collision_rate;
        env->log.clean_collision_rate += clean_collided;
        if (e->reached_goal_this_episode && !e->collided_before_goal) {
            env->log.score += 1.0f;
            env->log.perf += 1.0f;
        }
        if (!offroad && !collided && !e->reached_goal_this_episode) {
            env->log.dnf_rate += 1.0f;
        }
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.n += 1;
    }
}

// Map loading
Entity* load_map_binary(const char* filename, Drive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;

    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;

    Entity* entities = (Entity*)malloc(env->num_entities * sizeof(Entity));
    for (int i = 0; i < env->num_entities; i++) {
        // Read base entity data
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float*)malloc(size * sizeof(float));
        entities[i].traj_y = (float*)malloc(size * sizeof(float));
        entities[i].traj_z = (float*)malloc(size * sizeof(float));

        int is_actor = (entities[i].type == VEHICLE ||
                        entities[i].type == PEDESTRIAN ||
                        entities[i].type == CYCLIST);
        if (is_actor) {
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float*)malloc(size * sizeof(float));
            entities[i].traj_vy = (float*)malloc(size * sizeof(float));
            entities[i].traj_vz = (float*)malloc(size * sizeof(float));
            entities[i].traj_heading = (float*)malloc(size * sizeof(float));
            entities[i].traj_valid = (int*)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (is_actor) {
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
        }
        // Read remaining scalar fields
        fread(&entities[i].width, sizeof(float), 1, file);
        fread(&entities[i].length, sizeof(float), 1, file);
        fread(&entities[i].height, sizeof(float), 1, file);
        fread(&entities[i].goal_position_x, sizeof(float), 1, file);
        fread(&entities[i].goal_position_y, sizeof(float), 1, file);
        fread(&entities[i].goal_position_z, sizeof(float), 1, file);
        fread(&entities[i].mark_as_expert, sizeof(int), 1, file);
        entities[i].max_speed = 0;
    }
    fclose(file);
    return entities;
}

// Position initialization
void set_start_position(Drive* env) {
    for (int i = 0; i < env->num_entities; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Entity* e = &env->entities[i];
        e->x = e->traj_x[0];
        e->y = e->traj_y[0];
        e->z = e->traj_z[0];

        if (e->type > CYCLIST || e->type == NONE) {
            continue;
        }
        if (!is_active) {
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->reached_goal = 0;
            e->collided_before_goal = 0;
        } else {
            e->vx = e->traj_vx[0];
            e->vy = e->traj_vy[0];
            e->vz = e->traj_vz[0];
        }
        e->heading = e->traj_heading[0];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->prev_accel = 0.0f;
        e->prev_steer = 0.0f;
        e->valid = e->traj_valid[0];
        e->collision_state = NO_COLLISION;
        e->respawn_timestep = -1;
    }
}

// Grid Map
int getGridIndex(Drive* env, float x1, float y1) {
    if (env->map_corners[0] >= env->map_corners[2] ||
        env->map_corners[1] >= env->map_corners[3]) {
        return -1;
    }
    float relativeX = x1 - env->map_corners[0];
    float relativeY = y1 - env->map_corners[1];
    int gridX = (int)(relativeX / GRID_CELL_SIZE);
    int gridY = (int)(relativeY / GRID_CELL_SIZE);
    if (gridX < 0 || gridX >= env->grid_cols || gridY < 0 || gridY >= env->grid_rows) {
        return -1;
    }
    return (gridY * env->grid_cols) + gridX;
}

void add_entity_to_grid(Drive* env, int grid_index, int entity_idx, int geometry_idx) {
    if (grid_index == -1) return;

    int base_index = grid_index * SLOTS_PER_CELL;
    int count = env->grid_cells[base_index];
    if (count >= MAX_ENTITIES_PER_CELL) return;

    env->grid_cells[base_index + count * 2 + 1] = entity_idx;
    env->grid_cells[base_index + count * 2 + 2] = geometry_idx;
    env->grid_cells[base_index] = count + 1;
}

void init_grid_map(Drive* env) {
    free(env->map_corners);
    free(env->grid_cells);
    env->map_corners = NULL;
    env->grid_cells = NULL;
    env->grid_cols = 0;
    env->grid_rows = 0;
    float top_left_x = 0, top_left_y = 0, bottom_right_x = 0, bottom_right_y = 0;
    int first_valid_point = 0;

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type >= ROAD_LANE && env->entities[i].type <= ROAD_EDGE) {
            Entity* e = &env->entities[i];
            for (int j = 0; j < e->array_size; j++) {
                if (e->traj_x[j] == INVALID_POSITION) continue;
                if (e->traj_y[j] == INVALID_POSITION) continue;
                if (!first_valid_point) {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = 1;
                    continue;
                }
                if (e->traj_x[j] < top_left_x) top_left_x = e->traj_x[j];
                if (e->traj_x[j] > bottom_right_x) bottom_right_x = e->traj_x[j];
                if (e->traj_y[j] < top_left_y) top_left_y = e->traj_y[j];
                if (e->traj_y[j] > bottom_right_y) bottom_right_y = e->traj_y[j];
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
        if (env->entities[i].type >= ROAD_LANE && env->entities[i].type <= ROAD_EDGE) {
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j);
            }
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
        if (steps_taken != steps_to_take) continue;
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
            if (x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows) continue;
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
            if (x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows) continue;
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

int get_neighbor_cache_entities(Drive* env, int cell_idx, int* entities, int max_entities) {
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
    memcpy(entities, env->neighbor_cache_entities + base_index, count * sizeof(int));
    return pairs;
}

// Mean Centering
void set_means(Drive* env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if (env->entities[i].traj_valid[j]) {
                    point_count++;
                    mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                    mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
                }
            }
        } else if (env->entities[i].type >= ROAD_LANE) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                point_count++;
                mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
            }
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type >= ROAD_LANE) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if (env->entities[i].traj_x[j] == INVALID_POSITION) continue;
                env->entities[i].traj_x[j] -= mean_x;
                env->entities[i].traj_y[j] -= mean_y;
            }
            env->entities[i].goal_position_x -= mean_x;
            env->entities[i].goal_position_y -= mean_y;
        }
    }
}

// Expert Movement
void move_expert(Drive* env, float* actions, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->array_size) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        return;
    }
    agent->x = agent->traj_x[t];
    agent->y = agent->traj_y[t];
    agent->z = agent->traj_z[t];
    agent->heading = agent->traj_heading[t];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
}

// Collision Detection
bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
        fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
        return false;

    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];
    float cross = dx1 * dy2 - dy1 * dx2;
    if (cross == 0) return false;

    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int checkNeighbors(Drive* env, float x, float y, int* entity_list, int max_size,
                   const int (*local_offsets)[2], int offset_size) {
    int index = getGridIndex(env, x, y);
    if (index == -1) return 0;
    int cellsX = env->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;

    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        if (nx < 0 || nx >= env->grid_cols || ny < 0 || ny >= env->grid_rows) continue;
        int neighborIndex = (ny * env->grid_cols + nx) * SLOTS_PER_CELL;
        int count = env->grid_cells[neighborIndex];
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            entity_list[entity_list_count] = env->grid_cells[neighborIndex + 1 + j * 2];
            entity_list[entity_list_count + 1] = env->grid_cells[neighborIndex + 2 + j * 2];
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
            float proj1 = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj1);
            max1 = fmaxf(max1, proj1);
            float proj2 = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj2);
            max2 = fmaxf(max2, proj2);
        }
        if (max1 < min2 || min1 > max2) return 0;
    }
    return 1;
}

// Half-extent of the car's box along a unit axis.
static float box_radius(const Entity* e, float nx, float ny) {
    float hl = e->length * 0.5f;
    float hw = e->width * 0.5f;
    float along = fabsf(hl * e->heading_x * nx + hl * e->heading_y * ny);
    float side = fabsf(hw * (-e->heading_y) * nx + hw * e->heading_x * ny);
    return along + side;
}

static void box_span(const Entity* e, float ax, float ay, float* lo, float* hi) {
    float c = e->x * ax + e->y * ay;
    float r = box_radius(e, ax, ay);
    *lo = c - r;
    *hi = c + r;
}

// Smallest translation that pulls a out of b. Normal points from b toward a.
static int box_mtv(const Entity* a, const Entity* b,
        float* nx, float* ny, float* depth) {
    float axes[4][2] = {
        {a->heading_x, a->heading_y},
        {-a->heading_y, a->heading_x},
        {b->heading_x, b->heading_y},
        {-b->heading_y, b->heading_x}
    };
    float best = 1.0e9f;
    float bx = 1.0f;
    float by = 0.0f;
    for (int i = 0; i < 4; i++) {
        float ax = axes[i][0];
        float ay = axes[i][1];
        float a0, a1, b0, b1;
        box_span(a, ax, ay, &a0, &a1);
        box_span(b, ax, ay, &b0, &b1);
        float overlap = fminf(a1, b1) - fmaxf(a0, b0);
        if (overlap <= 0.0f) {
            return 0;
        }
        if (overlap < best) {
            best = overlap;
            bx = ax;
            by = ay;
        }
    }
    if ((a->x - b->x) * bx + (a->y - b->y) * by < 0.0f) {
        bx = -bx;
        by = -by;
    }
    *nx = bx;
    *ny = by;
    *depth = best;
    return 1;
}

// Centers came within both boxes during the step and finished past each
// other. The sample at the end of the step would miss that pass-through.
static int swept_bump(const Entity* a, const Entity* b,
        float pax, float pay, float pbx, float pby,
        float* nx, float* ny, float* depth) {
    float rx = pax - pbx;
    float ry = pay - pby;
    float dx = (a->x - b->x) - rx;
    float dy = (a->y - b->y) - ry;
    float dd = dx * dx + dy * dy;
    float t = 0.0f;
    if (dd > 1.0e-8f) {
        t = -(rx * dx + ry * dy) / dd;
        if (t < 0.0f) {
            t = 0.0f;
        }
        if (t > 1.0f) {
            t = 1.0f;
        }
    }
    float cx = rx + t * dx;
    float cy = ry + t * dy;
    float dist = sqrtf(cx * cx + cy * cy);
    float nxx;
    float nyy;
    if (dist > 1.0e-3f) {
        nxx = cx / dist;
        nyy = cy / dist;
    } else {
        float rvx = a->vx - b->vx;
        float rvy = a->vy - b->vy;
        float rl = sqrtf(rvx * rvx + rvy * rvy);
        if (rl < 1.0e-3f) {
            return 0;
        }
        nxx = -rvx / rl;
        nyy = -rvy / rl;
    }
    float sep = box_radius(a, nxx, nyy) + box_radius(b, nxx, nyy);
    if (dist > sep) {
        return 0;
    }
    float cur = (a->x - b->x) * nxx + (a->y - b->y) * nyy;
    float dep = sep - cur;
    if (dep <= 0.0f) {
        return 0;
    }
    *nx = nxx;
    *ny = nyy;
    *depth = dep;
    return 1;
}

static int bump_grace(const Drive* env, const Entity* e) {
    return e->respawn_timestep >= 0
        && env->timestep - e->respawn_timestep < RESPAWN_GRACE;
}

static void bump_pair(Drive* env, Entity* a, Entity* b,
        float pax, float pay, float pbx, float pby, int sweep) {
    if (a->x == INVALID_POSITION || b->x == INVALID_POSITION) {
        return;
    }
    if (bump_grace(env, a) || bump_grace(env, b)) {
        return;
    }
    int a_moves = a->active_agent == 1;
    int b_moves = b->active_agent == 1;
    if (!a_moves && !b_moves) {
        return;
    }
    float dx = a->x - b->x;
    float dy = a->y - b->y;
    if (dx * dx + dy * dy > 40.0f * 40.0f) {
        return;
    }
    float nx, ny, depth;
    if (!box_mtv(a, b, &nx, &ny, &depth)) {
        if (!sweep || !a_moves || !b_moves
                || !swept_bump(a, b, pax, pay, pbx, pby, &nx, &ny, &depth)) {
            return;
        }
    }
    float share_a = a_moves ? (b_moves ? 0.5f : 1.0f) : 0.0f;
    float share_b = b_moves ? (a_moves ? 0.5f : 1.0f) : 0.0f;
    float push = depth + BUMP_SLOP;
    float ma = push * share_a;
    float mb = push * share_b;
    if (ma > BUMP_MAX) {
        ma = BUMP_MAX;
    }
    if (mb > BUMP_MAX) {
        mb = BUMP_MAX;
    }
    a->x += nx * ma;
    a->y += ny * ma;
    b->x -= nx * mb;
    b->y -= ny * mb;

    float inv = (a_moves ? 1.0f : 0.0f) + (b_moves ? 1.0f : 0.0f);
    float rv = (a->vx - b->vx) * nx + (a->vy - b->vy) * ny;
    if (inv <= 0.0f || rv >= 0.0f) {
        return;
    }
    float j = -(1.0f + BUMP_BOUNCE) * rv / inv;
    if (a_moves) {
        a->vx += j * nx;
        a->vy += j * ny;
    }
    if (b_moves) {
        b->vx -= j * nx;
        b->vy -= j * ny;
    }
}

static void car_corners(const Entity* e, float c[4][2]) {
    float hl = e->length * 0.5f;
    float hw = e->width * 0.5f;
    float co = e->heading_x;
    float si = e->heading_y;
    for (int i = 0; i < 4; i++) {
        c[i][0] = e->x + (offsets[i][0] * hl * co - offsets[i][1] * hw * si);
        c[i][1] = e->y + (offsets[i][0] * hl * si + offsets[i][1] * hw * co);
    }
}

// Push the car back onto the side its center already occupies. If the
// center crossed this segment during the step, push it back to the side
// it left. Depth is how far the box still reaches across the curb.
static int curb_push(const Entity* car, float px, float py,
        float ax, float ay, float bx, float by,
        float* nx, float* ny, float* depth) {
    float dx = bx - ax;
    float dy = by - ay;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 1.0e-4f) {
        return 0;
    }
    float lx = -dy / len;
    float ly = dx / len;
    float sd_c = (car->x - ax) * lx + (car->y - ay) * ly;
    float sd_p = (px - ax) * lx + (py - ay) * ly;
    float p0[2] = {px, py};
    float p1[2] = {car->x, car->y};
    float q0[2] = {ax, ay};
    float q1[2] = {bx, by};
    int crossed = check_line_intersection(p0, p1, q0, q1);
    float keep = crossed ? sd_p : sd_c;
    float sign = (keep >= 0.0f) ? 1.0f : -1.0f;
    float ox = lx * sign;
    float oy = ly * sign;
    float corners[4][2];
    car_corners(car, corners);
    float pen = 0.0f;
    for (int i = 0; i < 4; i++) {
        float sd = (corners[i][0] - ax) * ox + (corners[i][1] - ay) * oy;
        if (-sd > pen) {
            pen = -sd;
        }
    }
    if (pen <= 0.0f) {
        return 0;
    }
    *nx = ox;
    *ny = oy;
    *depth = pen;
    return 1;
}

static int curb_hit(const Entity* car, float px, float py,
        float ax, float ay, float bx, float by) {
    float corners[4][2];
    car_corners(car, corners);
    float q0[2] = {ax, ay};
    float q1[2] = {bx, by};
    for (int k = 0; k < 4; k++) {
        if (check_line_intersection(corners[k], corners[(k + 1) % 4],
                q0, q1)) {
            return 1;
        }
    }
    float p0[2] = {px, py};
    float p1[2] = {car->x, car->y};
    return check_line_intersection(p0, p1, q0, q1);
}

static void bump_curb(Drive* env, Entity* car, float px, float py) {
    if (car->x == INVALID_POSITION || car->active_agent != 1) {
        return;
    }
    // A bend is several short segments. Each push can land on the next
    // one, so repeat. Bounce once, on the first contact.
    int bounced = 0;
    for (int iter = 0; iter < 4; iter++) {
        int list[MAX_ENTITIES_PER_CELL * 2 * 25];
        int n = checkNeighbors(env, car->x, car->y, list,
            MAX_ENTITIES_PER_CELL * 2 * 25, collision_offsets, 25);
        float best = 0.0f;
        float bnx = 0.0f;
        float bny = 0.0f;
        for (int i = 0; i < n; i += 2) {
            int idx = list[i];
            if (idx < 0) {
                continue;
            }
            Entity* edge = &env->entities[idx];
            if (edge->type != ROAD_EDGE) {
                continue;
            }
            int g = list[i + 1];
            if (g < 0 || g + 1 >= edge->array_size) {
                continue;
            }
            float ax = edge->traj_x[g];
            float ay = edge->traj_y[g];
            float bx = edge->traj_x[g + 1];
            float by = edge->traj_y[g + 1];
            if (!curb_hit(car, px, py, ax, ay, bx, by)) {
                continue;
            }
            float ox, oy, dep;
            if (!curb_push(car, px, py, ax, ay, bx, by, &ox, &oy, &dep)) {
                continue;
            }
            if (dep > best) {
                best = dep;
                bnx = ox;
                bny = oy;
            }
        }
        if (best <= 0.0f) {
            return;
        }
        float push = best + BUMP_SLOP;
        if (push > BUMP_MAX) {
            push = BUMP_MAX;
        }
        car->x += bnx * push;
        car->y += bny * push;
        if (bounced) {
            continue;
        }
        float vn = car->vx * bnx + car->vy * bny;
        if (vn < 0.0f) {
            float j = -(1.0f + BUMP_BOUNCE) * vn;
            car->vx += j * bnx;
            car->vy += j * bny;
            bounced = 1;
        }
    }
}

// The bicycle step has already moved every car. Separate overlaps and
// bounce the closing speed. Two passes, because one shove creates the
// next contact. The sweep follows the step-start segment, so only the
// first pass uses it. Respawn grace still lets a fresh car overlap.
static void bump_agents(Drive* env) {
    int nact = env->active_agent_count;
    int n = nact + env->static_agent_count;
    if (n > DRIVE_AGENT_MAX) {
        n = DRIVE_AGENT_MAX;
    }
    float prev_x[DRIVE_AGENT_MAX];
    float prev_y[DRIVE_AGENT_MAX];
    for (int i = 0; i < n; i++) {
        int ia = (i < nact)
            ? env->active_agent_indices[i]
            : env->static_agent_indices[i - nact];
        if (ia < 0) {
            prev_x[i] = 0.0f;
            prev_y[i] = 0.0f;
            continue;
        }
        Entity* e = &env->entities[ia];
        // Respawn this step teleported. vx is the spawn speed, not the
        // displacement. A static car snaps the same way.
        int stepped = i < nact && e->respawn_timestep != env->timestep;
        if (stepped) {
            prev_x[i] = e->x - e->vx * SIM_DT;
            prev_y[i] = e->y - e->vy * SIM_DT;
        } else {
            prev_x[i] = e->x;
            prev_y[i] = e->y;
        }
    }
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < n; i++) {
            int ia = (i < nact)
                ? env->active_agent_indices[i]
                : env->static_agent_indices[i - nact];
            if (ia < 0) {
                continue;
            }
            for (int j = i + 1; j < n; j++) {
                int ib = (j < nact)
                    ? env->active_agent_indices[j]
                    : env->static_agent_indices[j - nact];
                if (ib < 0) {
                    continue;
                }
                bump_pair(env,
                    &env->entities[ia], &env->entities[ib],
                    prev_x[i], prev_y[i], prev_x[j], prev_y[j],
                    pass == 0);
            }
        }
        for (int i = 0; i < nact; i++) {
            int ia = env->active_agent_indices[i];
            if (ia < 0) {
                continue;
            }
            bump_curb(env, &env->entities[ia], prev_x[i], prev_y[i]);
        }
    }
}

int collision_check(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    if (agent->x == INVALID_POSITION) return -1;

    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] = agent->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

    int collided = NO_COLLISION;
    int car_collided_with_index = -1;

    // Check road edge collisions via grid
    int entity_list[MAX_ENTITIES_PER_CELL * 2 * 25];
    int list_size = checkNeighbors(env, agent->x, agent->y, entity_list,
                                   MAX_ENTITIES_PER_CELL * 2 * 25, collision_offsets, 25);
    for (int i = 0; i < list_size; i += 2) {
        if (entity_list[i] == -1 || entity_list[i] == agent_idx) continue;
        Entity* entity = &env->entities[entity_list[i]];
        if (entity->type != ROAD_EDGE) continue;
        int geometry_idx = entity_list[i + 1];
        float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
        float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
        for (int k = 0; k < 4; k++) {
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], start, end)) {
                collided = OFFROAD;
                break;
            }
        }
        if (collided == OFFROAD) break;
    }

    // Check vehicle-vehicle collisions
    int nscan = env->active_agent_count + env->static_agent_count;
    if (nscan > DRIVE_AGENT_MAX) nscan = DRIVE_AGENT_MAX;
    for (int i = 0; i < nscan; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (i - env->active_agent_count < env->static_agent_count) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1 || index == agent_idx) continue;
        Entity* entity = &env->entities[index];
        float dx = entity->x - agent->x;
        float dy = entity->y - agent->y;
        if ((dx * dx + dy * dy) > COLLISION_DIST_SQ) continue;
        if (check_aabb_collision(agent, entity)) {
            collided = VEHICLE_COLLISION;
            car_collided_with_index = index;
            break;
        }
    }

    agent->collision_state = collided;

    // Spawn immunity: agent just respawned
    if (collided == VEHICLE_COLLISION && agent->active_agent == 1 &&
        agent->respawn_timestep != -1) {
        agent->collision_state = NO_COLLISION;
    }

    if (collided == OFFROAD) return -1;
    if (car_collided_with_index == -1) return -1;

    // Spawn immunity: collided-with agent just respawned
    if (env->entities[car_collided_with_index].respawn_timestep != -1) {
        agent->collision_state = NO_COLLISION;
    }

    return car_collided_with_index;
}

// Agent Selection
int valid_active_agent(Drive* env, int agent_idx) {
    if (agent_idx < 0 || agent_idx >= env->num_entities) return 0;
    Entity* e = &env->entities[agent_idx];
    if (e->type != VEHICLE || e->traj_valid[0] != 1) return 0;
    float cos_heading = cosf(e->traj_heading[0]);
    float sin_heading = sinf(e->traj_heading[0]);
    float goal_x = e->goal_position_x - e->traj_x[0];
    float goal_y = e->goal_position_y - e->traj_y[0];
    float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
    float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, rel_goal_x, rel_goal_y);

    e->width *= COLLISION_BOX_SCALE;
    e->length *= COLLISION_BOX_SCALE;

    if (distance_to_goal >= MIN_DISTANCE_TO_GOAL &&
        e->mark_as_expert == 0 &&
        env->active_agent_count < env->max_agents) {
        return distance_to_goal;
    }
    return 0;
}

void set_active_agents(Drive* env) {
    free(env->active_agent_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    env->active_agent_indices = NULL;
    env->static_agent_indices = NULL;
    env->expert_static_agent_indices = NULL;
    env->active_agent_count = 0;
    env->static_agent_count = 0;
    env->num_actors = 1;
    env->expert_static_agent_count = 0;

    int active_agent_indices[DRIVE_AGENT_MAX];
    int static_agent_indices[DRIVE_AGENT_MAX];
    int expert_static_agent_indices[DRIVE_AGENT_MAX];

    if (env->max_agents == 0) {
        env->max_agents = MAX_AGENTS;
    }

    // First agent: last object (SDC equivalent)
    int first_agent_id = env->num_objects - 1;
    float distance_to_goal = valid_active_agent(env, first_agent_id);
    if (distance_to_goal) {
        env->active_agent_count = 1;
        active_agent_indices[0] = first_agent_id;
        env->entities[first_agent_id].active_agent = 1;
        env->num_actors = 1;
    } else {
        env->active_agent_count = 0;
        env->num_actors = 0;
    }

    for (int i = 0; i < env->num_objects - 1 && env->num_actors < DRIVE_AGENT_MAX; i++) {
        if (env->entities[i].type != VEHICLE) continue;
        if (env->entities[i].traj_valid[0] != 1) continue;
        env->num_actors++;

        float dist = valid_active_agent(env, i);
        if (dist > 0) {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
        } else {
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->entities[i].active_agent = 0;
            if (env->entities[i].mark_as_expert == 1 ||
                (dist >= MIN_DISTANCE_TO_GOAL && env->active_agent_count == env->max_agents)) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }

    env->active_agent_indices = (int*)malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int*)malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int*)malloc(env->expert_static_agent_count * sizeof(int));
    memcpy(env->active_agent_indices, active_agent_indices, env->active_agent_count * sizeof(int));
    memcpy(env->static_agent_indices, static_agent_indices, env->static_agent_count * sizeof(int));
    memcpy(env->expert_static_agent_indices, expert_static_agent_indices, env->expert_static_agent_count * sizeof(int));
}

// Trajectory Validation
void remove_bad_trajectories(Drive* env) {
    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));

    for (int t = 0; t < TRAJECTORY_LENGTH; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            move_expert(env, NULL, env->active_agent_indices[i]);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->entities[expert_idx].x == INVALID_POSITION) continue;
            move_expert(env, NULL, expert_idx);
        }
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            env->entities[agent_idx].collision_state = NO_COLLISION;
            int collided_with = collision_check(env, agent_idx);
            if (env->entities[agent_idx].collision_state > NO_COLLISION && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        if (collided_with_indices[i] == -1) continue;
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_idx = env->static_agent_indices[j];
            if (static_idx != collided_with_indices[i]) continue;
            env->entities[static_idx].traj_x[0] = INVALID_POSITION;
            env->entities[static_idx].traj_y[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

// Lowest target speed the viewer slider can stage, in m/s.
#define SIM_SPEED_MIN 1.0f
#include "flow_init.h"

static void flow_diagram_clear(Drive* env) {
    env->flow_hit_n = 0;
    env->flow_hit_i = 0;
    memset(env->flow_hit_prev, 0, sizeof(env->flow_hit_prev));
    if (!env->flow_speed) {
        return;
    }
    int n = env->flow_rows * env->flow_nlanes;
    for (int i = 0; i < n; i++) {
        env->flow_speed[i] = -1.0f;
    }
}

static int flow_seg_flat(const FlowSeg* s) {
    float dy = s->by - s->ay;
    if (dy < 0.0f) {
        dy = -dy;
    }
    return dy < FLOW_FLAT_DY;
}

static float flow_seg_ymid(const FlowSeg* s) {
    return 0.5f * (s->ay + s->by);
}

// Larger network x first, so texture row 0 of a lane is downstream.
static void flow_sort_pos(FlowSeg* s, int lo, int hi) {
    for (int i = lo + 1; i < hi; i++) {
        FlowSeg key = s[i];
        int j = i - 1;
        while (j >= lo && s[j].pos < key.pos) {
            s[j + 1] = s[j];
            j--;
        }
        s[j + 1] = key;
    }
}

static void flow_sort_ymid(FlowSeg* s, int n) {
    for (int i = 1; i < n; i++) {
        FlowSeg key = s[i];
        float ky = flow_seg_ymid(&key);
        int j = i - 1;
        while (j >= 0 && flow_seg_ymid(&s[j]) < ky) {
            s[j + 1] = s[j];
            j--;
        }
        s[j + 1] = key;
    }
}

// Horizontal centerlines move to the front. Returns how many.
static int flow_partition_flat(FlowSeg* s, int n) {
    int h = 0;
    int d = n - 1;
    while (h <= d) {
        if (flow_seg_flat(&s[h])) {
            h++;
        } else {
            FlowSeg tmp = s[h];
            s[h] = s[d];
            s[d] = tmp;
            d--;
        }
    }
    return h;
}

static int flow_take_lane(Drive* env, int lane, int beg, int n) {
    if (lane < 0 || lane >= FLOW_LANE_MAX || n <= 0) {
        return -1;
    }
    flow_sort_pos(env->flow_segs, beg, beg + n);
    env->flow_lane_beg[lane] = beg;
    env->flow_lane_n[lane] = n;
    float lo = 1.0e30f;
    float hi = -1.0e30f;
    for (int i = 0; i < n; i++) {
        FlowSeg* g = &env->flow_segs[beg + i];
        g->lane = lane;
        float a = g->ax + env->world_mean_x;
        float b = g->bx + env->world_mean_x;
        if (a < lo) lo = a;
        if (b < lo) lo = b;
        if (a > hi) hi = a;
        if (b > hi) hi = b;
    }
    env->flow_lane_lo[lane] = lo;
    env->flow_lane_hi[lane] = hi;
    env->flow_lane_name[lane][0] = '\0';
    return 0;
}

// Display order is already top to bottom. Labels are 1..n in that order.
static void flow_number_lanes(Drive* env) {
    for (int i = 0; i < env->flow_nlanes; i++) {
        snprintf(env->flow_lane_name[i], 16, "%d", i + 1);
    }
}

// Group ROAD_LANE pieces into lanes. Do not stitch by endpoint: the ramp
// meets the accel lane, and a heading test joins them.
static void flow_diagram_alloc(Drive* env) {
    free(env->flow_speed);
    free(env->flow_bin_count);
    free(env->flow_segs);
    env->flow_speed = NULL;
    env->flow_bin_count = NULL;
    env->flow_segs = NULL;
    env->flow_nseg = 0;
    env->flow_rows = 0;
    env->flow_nlanes = 0;
    env->flow_pos_lo = 0.0f;
    env->flow_pos_hi = 0.0f;
    if (env->agent_initialization != AGENT_INIT_STRUCTURED) {
        return;
    }
    if (!env->entities || env->episode_length < 1) {
        return;
    }
    int nseg = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type != ROAD_LANE || e->array_size < 2) {
            continue;
        }
        nseg += e->array_size - 1;
    }
    if (nseg <= 0) {
        return;
    }
    FlowSeg* segs = (FlowSeg*)malloc((size_t)nseg * sizeof(FlowSeg));
    if (!segs) {
        fprintf(stderr, "velocity diagram alloc failed\n");
        exit(1);
    }
    int w = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type != ROAD_LANE || e->array_size < 2) {
            continue;
        }
        for (int j = 0; j < e->array_size - 1; j++) {
            float ax = e->traj_x[j];
            float ay = e->traj_y[j];
            float bx = e->traj_x[j + 1];
            float by = e->traj_y[j + 1];
            if (ax == INVALID_POSITION || bx == INVALID_POSITION) {
                continue;
            }
            segs[w].ax = ax;
            segs[w].ay = ay;
            segs[w].bx = bx;
            segs[w].by = by;
            segs[w].pos = 0.5f * (ax + bx) + env->world_mean_x;
            segs[w].lane = -1;
            w++;
        }
    }
    nseg = w;
    env->flow_segs = segs;
    env->flow_nseg = nseg;
    if (nseg <= 0) {
        free(segs);
        env->flow_segs = NULL;
        return;
    }
    int nflat_seg = flow_partition_flat(segs, nseg);
    if (nflat_seg > 1) {
        flow_sort_ymid(segs, nflat_seg);
    }
    int lane = 0;
    int start = 0;
    for (int i = 1; i <= nflat_seg; i++) {
        int split = (i == nflat_seg);
        if (!split) {
            float prev = flow_seg_ymid(&segs[i - 1]);
            float cur = flow_seg_ymid(&segs[i]);
            if (prev - cur > FLOW_LANE_GAP) {
                split = 1;
            }
        }
        if (!split) {
            continue;
        }
        if (flow_take_lane(env, lane, start, i - start) < 0) {
            fprintf(stderr, "velocity diagram has more than %d lanes\n",
                FLOW_LANE_MAX);
            exit(1);
        }
        lane++;
        start = i;
    }
    if (nseg > nflat_seg) {
        if (flow_take_lane(env, lane, nflat_seg, nseg - nflat_seg) < 0) {
            fprintf(stderr, "velocity diagram has more than %d lanes\n",
                FLOW_LANE_MAX);
            exit(1);
        }
        lane++;
    }
    if (lane <= 0) {
        return;
    }
    env->flow_nlanes = lane;
    flow_number_lanes(env);
    float plo = env->flow_lane_lo[0];
    float phi = env->flow_lane_hi[0];
    for (int i = 1; i < lane; i++) {
        if (env->flow_lane_lo[i] < plo) {
            plo = env->flow_lane_lo[i];
        }
        if (env->flow_lane_hi[i] > phi) {
            phi = env->flow_lane_hi[i];
        }
    }
    env->flow_pos_lo = plo;
    env->flow_pos_hi = phi;
    int rows = env->episode_length;
    env->flow_rows = rows;
    env->flow_speed = (float*)malloc(
        (size_t)rows * (size_t)lane * sizeof(float));
    env->flow_bin_count = (int*)malloc((size_t)lane * sizeof(int));
    if (!env->flow_speed || !env->flow_bin_count) {
        fprintf(stderr, "velocity diagram alloc failed\n");
        exit(1);
    }
    flow_diagram_clear(env);
}

// Closest centerline within reach meters. -1 if the car is farther out.
static int flow_agent_lane(Drive* env, float x, float y, float reach) {
    int nseg = env->flow_nseg;
    int nlanes = env->flow_nlanes;
    float limit = reach * reach;
    int best = -1;
    float best_lat = limit;
    float best_mid = 2.0f;
    for (int s = 0; s < nseg; s++) {
        FlowSeg* g = &env->flow_segs[s];
        float dx = g->bx - g->ax;
        float dy = g->by - g->ay;
        float len2 = dx * dx + dy * dy;
        if (len2 < 1.0e-8f) {
            continue;
        }
        float t = ((x - g->ax) * dx + (y - g->ay) * dy) / len2;
        if (t < 0.0f || t > 1.0f) {
            continue;
        }
        float px = g->ax + t * dx;
        float py = g->ay + t * dy;
        float lat = (x - px) * (x - px) + (y - py) * (y - py);
        if (lat > limit) {
            continue;
        }
        float mid = t - 0.5f;
        if (mid < 0.0f) {
            mid = -mid;
        }
        if (lat < best_lat || (lat == best_lat && mid < best_mid)) {
            best_lat = lat;
            best_mid = mid;
            best = s;
        }
    }
    if (best < 0) {
        return -1;
    }
    int lane = env->flow_segs[best].lane;
    if (lane < 0 || lane >= nlanes) {
        return -1;
    }
    return lane;
}

static int flow_place_lane(Drive* env, const Entity* e) {
    int lane = flow_agent_lane(env, e->x, e->y, FLOW_LATERAL);
    if (lane >= 0) {
        return lane;
    }
    return flow_agent_lane(env, e->x, e->y, FLOW_HIT_REACH);
}

// Other active car in this contact, or -1. The lower index owns the mark.
static int flow_hit_partner(Drive* env, int agent_idx) {
    Entity* a = &env->entities[agent_idx];
    int best = -1;
    for (int j = 0; j < env->active_agent_count; j++) {
        int idx = env->active_agent_indices[j];
        if (idx == agent_idx) {
            continue;
        }
        Entity* b = &env->entities[idx];
        if (b->collision_state != VEHICLE_COLLISION) {
            continue;
        }
        float dx = b->x - a->x;
        float dy = b->y - a->y;
        if (dx * dx + dy * dy > COLLISION_DIST_SQ) {
            continue;
        }
        if (!check_aabb_collision(a, b)) {
            continue;
        }
        if (best < 0 || idx < best) {
            best = idx;
        }
    }
    return best;
}

static void flow_hit_push(Drive* env, int lane, int kind) {
    int i = env->flow_hit_i;
    env->flow_hit_t[i] = env->timestep;
    env->flow_hit_lane[i] = lane;
    env->flow_hit_kind[i] = (unsigned char)kind;
    i++;
    if (i >= FLOW_HIT_MAX) {
        i = 0;
    }
    env->flow_hit_i = i;
    if (env->flow_hit_n < FLOW_HIT_MAX) {
        env->flow_hit_n++;
    }
}

// Mean speed of the vehicles on each lane. A lane is one road area.
// A new collision on that lane is stored for the chart beside the heatmap.
static void flow_diagram_record(Drive* env) {
    if (!env->flow_speed || !env->flow_segs) {
        return;
    }
    if (env->flow_rows <= 0 || env->flow_nlanes <= 0) {
        return;
    }
    int row = env->timestep;
    if (row < 0) {
        return;
    }
    if (env->sim_infinite) {
        row %= env->flow_rows;
    } else if (row >= env->flow_rows) {
        return;
    }
    int nlanes = env->flow_nlanes;
    float* sum = env->flow_speed + (size_t)row * (size_t)nlanes;
    int* count = env->flow_bin_count;
    for (int lane = 0; lane < nlanes; lane++) {
        sum[lane] = 0.0f;
        count[lane] = 0;
    }
    int nact = env->active_agent_count;
    if (nact > DRIVE_AGENT_MAX) {
        nact = DRIVE_AGENT_MAX;
    }
    unsigned char rising[DRIVE_AGENT_MAX];
    for (int i = 0; i < nact; i++) {
        Entity* e = &env->entities[env->active_agent_indices[i]];
        int hidden = e->x == INVALID_POSITION;
        if (!hidden) {
            int lane = flow_agent_lane(env, e->x, e->y, FLOW_LATERAL);
            if (lane >= 0) {
                float speed = sqrtf(e->vx * e->vx + e->vy * e->vy);
                sum[lane] += speed;
                count[lane] += 1;
            }
        }
        int hit = !hidden && e->collision_state > NO_COLLISION;
        rising[i] = (unsigned char)(hit && !env->flow_hit_prev[i]);
        env->flow_hit_prev[i] = (unsigned char)hit;
    }
    for (int lane = 0; lane < nlanes; lane++) {
        if (count[lane] == 0) {
            sum[lane] = -1.0f;
        } else {
            sum[lane] /= (float)count[lane];
        }
    }
    for (int i = 0; i < nact; i++) {
        if (!rising[i]) {
            continue;
        }
        int self = env->active_agent_indices[i];
        Entity* e = &env->entities[self];
        int other = -1;
        if (e->collision_state == VEHICLE_COLLISION) {
            other = flow_hit_partner(env, self);
            int slot = -1;
            if (other >= 0) {
                for (int j = 0; j < nact; j++) {
                    if (env->active_agent_indices[j] == other) {
                        slot = j;
                        break;
                    }
                }
            }
            if (other >= 0 && other < self && slot >= 0 && rising[slot]) {
                continue;
            }
        }
        int lane = flow_place_lane(env, e);
        if (lane < 0 && other >= 0) {
            lane = flow_place_lane(env, &env->entities[other]);
        }
        if (lane < 0) {
            continue;
        }
        flow_hit_push(env, lane, e->collision_state);
    }
}

// Initialization / Cleanup
static void drive_release_world(Drive* env) {
    route_free(env);
    free(env->flow_speed);
    free(env->flow_bin_count);
    free(env->flow_segs);
    env->flow_speed = NULL;
    env->flow_bin_count = NULL;
    env->flow_segs = NULL;
    env->flow_nseg = 0;
    env->flow_rows = 0;
    env->flow_nlanes = 0;
    env->flow_hit_n = 0;
    env->flow_hit_i = 0;
    memset(env->flow_hit_prev, 0, sizeof(env->flow_hit_prev));
    if (env->entities) {
        for (int i = 0; i < env->num_entities; i++) {
            free_entity(&env->entities[i]);
        }
        free(env->entities);
        env->entities = NULL;
    }
    env->num_entities = 0;
    env->num_objects = 0;
    env->num_roads = 0;
    free(env->active_agent_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    env->active_agent_indices = NULL;
    env->static_agent_indices = NULL;
    env->expert_static_agent_indices = NULL;
    env->active_agent_count = 0;
    env->static_agent_count = 0;
    env->expert_static_agent_count = 0;
    env->num_actors = 0;
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
}

static void drive_load_world(Drive* env) {
    env->entities = load_map_binary(env->map_name, env);
    if (!env->entities) {
        fprintf(stderr, "ERROR: Cannot load map %s\n",
            env->map_name ? env->map_name : "(null)");
        exit(1);
    }
    if (env->agent_initialization != AGENT_INIT_LOGS &&
        env->agent_initialization != AGENT_INIT_STRUCTURED) {
        fprintf(stderr,
            "env.agent_initialization must be 0 (logs) or 1 (structured)\n");
        exit(1);
    }
    // Structured placement is in network coordinates. set_means then centers
    // roads and vehicles together. Logged init leaves the binary objects alone.
    if (env->agent_initialization == AGENT_INIT_STRUCTURED) {
        spawn_flow_vehicles(env);
    }
    env->dynamics_model = CLASSIC;
    set_means(env);
    route_shift(env);
    init_grid_map(env);
    env->vision_range = VISION_RANGE;
    init_neighbor_offsets(env);
    free(env->neighbor_cache_indices);
    env->neighbor_cache_indices = (int*)calloc(
        (size_t)(env->grid_cols * env->grid_rows) + 1, sizeof(int));
    cache_neighbor_offsets(env);
    set_active_agents(env);
    // Logged agents are replayed once so a partner that overlaps the log can be
    // dropped. Structured vehicles have no future trajectory to replay.
    if (env->agent_initialization != AGENT_INIT_STRUCTURED) {
        remove_bad_trajectories(env);
    }
    set_start_position(env);
    if (env->agent_initialization == AGENT_INIT_STRUCTURED
        && env->active_agent_count <= 0) {
        fprintf(stderr, "structured init produced no agents for %s\n",
            env->map_name ? env->map_name : "(null)");
        exit(1);
    }
    free(env->logs);
    env->logs = (Log*)calloc((size_t)env->active_agent_count, sizeof(Log));
    flow_diagram_alloc(env);
    flow_diagram_record(env);
    if (env->sim_agents_on == 0) {
        int n = env->active_agent_count;
        if (n < 1) n = 1;
        if (n > DRIVE_AGENT_MAX) n = DRIVE_AGENT_MAX;
        env->sim_agents_req = n;
        env->sim_agents_on = n;
    }
}

void init(Drive* env) {
    env->timestep = 0;
    env->episode_length = TRAJECTORY_LENGTH;
    if (env->cfg) {
        DictItem* item = dict_find(env->cfg, "episode_length");
        if (item) {
            if (item->value < 1) {
                fprintf(stderr, "env.episode_length must be >= 1\n");
                exit(1);
            }
            env->episode_length = (int)item->value;
        }
    }
    drive_load_world(env);
}

void puf_close(Drive* env) {
    drive_release_world(env);
}

// Dynamics
void move_dynamics(Drive* env, int action_idx, int agent_idx) {
    if (env->dynamics_model != CLASSIC) return;

    Entity* agent = &env->entities[agent_idx];
    float* atn = env->agents[action_idx].actions;
    int acceleration_index = (int)atn[0];
    int steering_index = (int)atn[1];
    float acceleration = ACCELERATION_VALUES[acceleration_index];
    float steering = STEERING_VALUES[steering_index];

    float x = agent->x;
    float y = agent->y;
    float heading = agent->heading;
    float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);

    speed = speed + 0.5f * acceleration * SIM_DT;
    speed = clipSpeed(speed);
    // A reset target holds. Logged cars leave max_speed at 0.
    float cap = agent->max_speed;
    if (cap >= SIM_SPEED_MIN && speed > cap) {
        speed = cap;
    }

    float beta = tanh(0.5 * tanf(steering));
    float yaw_rate = (speed * cosf(beta) * tanf(steering)) / agent->length;
    float new_vx = speed * cosf(heading + beta);
    float new_vy = speed * sinf(heading + beta);

    x += new_vx * SIM_DT;
    y += new_vy * SIM_DT;
    heading += yaw_rate * SIM_DT;

    agent->x = x;
    agent->y = y;
    agent->heading = heading;
    agent->heading_x = cosf(heading);
    agent->heading_y = sinf(heading);
    agent->vx = new_vx;
    agent->vy = new_vy;
}

// Observations
void compute_observations(Drive* env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        float* obs = env->agents[i].observations;
        if (!obs) continue;
        memset(obs, 0, OBS_SIZE * sizeof(float));
        Entity* ego = &env->entities[env->active_agent_indices[i]];
        if (ego->type > CYCLIST) break;

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
        // Index 6 stays 0. Index 7 is the episode max speed / 100.
        obs[EGO_FEATURES - 1] = ego->max_speed / MAX_SPEED;

        // First MAX_OBS_AGENTS partners in index order, inside range.
        int obs_idx = EGO_FEATURES;
        int cars_seen = 0;
        int nscan = env->active_agent_count + env->static_agent_count;
        if (nscan > DRIVE_AGENT_MAX) nscan = DRIVE_AGENT_MAX;
        for (int j = 0; j < nscan && cars_seen < MAX_OBS_AGENTS; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j - env->active_agent_count < env->static_agent_count) {
                index = env->static_agent_indices[
                    j - env->active_agent_count];
            }
            if (index == -1) continue;
            if (env->entities[index].type > CYCLIST) break;
            if (index == env->active_agent_indices[i]) continue;

            Entity* other = &env->entities[index];

            float dx = other->x - ego->x;
            float dy = other->y - ego->y;
            if ((dx * dx + dy * dy) > OBS_DIST_SQ) continue;

            float rel_x = dx * cos_h + dy * sin_h;
            float rel_y = -dx * sin_h + dy * cos_h;

            obs[obs_idx + 0] = rel_x * OBS_POSITION_SCALE;
            obs[obs_idx + 1] = rel_y * OBS_POSITION_SCALE;
            obs[obs_idx + 2] = other->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other->length / MAX_VEH_LEN;
            obs[obs_idx + 4] = other->heading_x * ego->heading_x + other->heading_y * ego->heading_y;
            obs[obs_idx + 5] = other->heading_y * ego->heading_x - other->heading_x * ego->heading_y;
            float other_speed = sqrtf(other->vx * other->vx + other->vy * other->vy);
            obs[obs_idx + 6] = other_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += PARTNER_FEATURES;
        }
        int remaining_partner_obs =
            (MAX_OBS_AGENTS - cars_seen) * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;

        // Road observations
        int entity_list[MAX_ROAD_SEGMENT_OBSERVATIONS * 2];
        int grid_idx = getGridIndex(env, ego->x, ego->y);
        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k * 2];
            int geometry_idx = entity_list[k * 2 + 1];
            Entity* entity = &env->entities[entity_idx];

            float start_x = entity->traj_x[geometry_idx];
            float start_y = entity->traj_y[geometry_idx];
            float end_x = entity->traj_x[geometry_idx + 1];
            float end_y = entity->traj_y[geometry_idx + 1];
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
            float dx_norm = dx, dy_norm = dy;
            if (hypot > 0) { dx_norm /= hypot; dy_norm /= hypot; }

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
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * ROAD_FEATURES;
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }
}

void puf_reset(Drive* env) {
    env->timestep = 0;
    if (!env->entities || env->active_agent_count <= 0) return;
    set_start_position(env);
    assign_episode_goals(env);
    assign_episode_speeds(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].reached_goal = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].reached_goal_this_episode = 0;

        collision_check(env, agent_idx);
    }
    compute_observations(env);
    flow_diagram_clear(env);
    flow_diagram_record(env);
}

// Rebuild one map at n cars and reset. Observations must already be bound.
// Pause, sim_infinite, sim_follow, and the speed min and max stay.
void puf_set_population(Drive* env, int n) {
    if (n < 1) n = 1;
    if (n > DRIVE_AGENT_MAX) n = DRIVE_AGENT_MAX;
    env->max_agents = n;
    env->timestep = 0;
    drive_release_world(env);
    drive_load_world(env);
    env->sim_agents_on = n;
    env->sim_agents_apply = 0;
    env->num_agents = env->active_agent_count;
    if (env->num_agents > DRIVE_AGENT_MAX) {
        env->num_agents = DRIVE_AGENT_MAX;
    }
    if (env->human_agent_idx < 0
        || env->human_agent_idx >= env->num_agents) {
        env->human_agent_idx = 0;
    }
    if (env->agent_initialization == AGENT_INIT_STRUCTURED
        && env->active_agent_count != n) {
        fprintf(stderr, "agents per map: requested %d, placed %d\n",
            n, env->active_agent_count);
    }
    puf_reset(env);
}

// Back to this car's own spawn. The goal and target speed stay.
void respawn_agent(Drive* env, int agent_idx) {
    Entity* e = &env->entities[agent_idx];
    e->x = e->traj_x[0];
    e->y = e->traj_y[0];
    e->heading = e->traj_heading[0];
    e->heading_x = cosf(e->heading);
    e->heading_y = sinf(e->heading);
    float spd = e->max_speed;
    if (spd >= SIM_SPEED_MIN) {
        e->vx = spd * e->heading_x;
        e->vy = spd * e->heading_y;
    } else {
        e->vx = e->traj_vx[0];
        e->vy = e->traj_vy[0];
    }
    e->reached_goal = 0;
    e->respawn_timestep = env->timestep;
}

// Hold Left Shift + WASD/arrows. Tab cycles the human vehicle.
static void drive_human_controls(Drive *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (env->active_agent_count <= 0) {
        return;
    }
    if (env->human_agent_idx < 0 || env->human_agent_idx >= env->active_agent_count) {
        env->human_agent_idx = 0;
    }
    int idx = env->human_agent_idx;
    float *atn = env->agents[idx].actions;
    atn[0] = 3;
    atn[1] = 6;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        atn[0] += 2;
        if (atn[0] > 6) {
            atn[0] = 6;
        }
    }
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        atn[0] -= 2;
        if (atn[0] < 0) {
            atn[0] = 0;
        }
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        atn[1] += 4;
        if (atn[1] < 0) {
            atn[1] = 0;
        }
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        atn[1] -= 4;
        if (atn[1] > 12) {
            atn[1] = 12;
        }
    }
    if (IsKeyPressed(KEY_TAB)) {
        env->human_agent_idx = (env->human_agent_idx + 1) % env->active_agent_count;
    }
}

// The 5 m disk is easy to miss. Past the goal plane, or well beyond the
// trip, is the end of this lap. Finite scoring keeps the disk only.
static int infinite_passed_goal(const Entity* e) {
    float sx = e->traj_x[0];
    float sy = e->traj_y[0];
    float gx = e->goal_position_x - sx;
    float gy = e->goal_position_y - sy;
    float px = e->x - e->goal_position_x;
    float py = e->y - e->goal_position_y;
    if (px * gx + py * gy > 0.0f) {
        return 1;
    }
    float trip = sqrtf(gx * gx + gy * gy);
    float hx = e->x - sx;
    float hy = e->y - sy;
    float limit = trip + RESPAWN_OVERRUN;
    return (hx * hx + hy * hy) > (limit * limit);
}

void puf_step(Drive* env) {
    drive_human_controls(env);
    if (!env->entities || env->active_agent_count <= 0) return;
    for (int _i = 0; _i < env->active_agent_count; _i++) {
        if (!env->agents[_i].rewards) continue;
        env->agents[_i].rewards[0] = 0;
        env->agents[_i].terminals[0] = 0;
    }
    env->timestep++;

    // Infinite keeps driving past the configured length. Headless stays finite.
    if (!env->sim_infinite && env->timestep == env->episode_length) {
        add_log(env);
        puf_reset(env);
        return;
    }

    // Move expert static agents
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        if (env->entities[expert_idx].x == INVALID_POSITION) continue;
        move_expert(env, NULL, expert_idx);
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        Entity* agent = &env->entities[agent_idx];
        // Immunity expires so a respawned car can be penalized again.
        if (agent->respawn_timestep >= 0
            && env->timestep - agent->respawn_timestep >= RESPAWN_GRACE) {
            agent->respawn_timestep = -1;
        }
        agent->collision_state = NO_COLLISION;
        move_dynamics(env, i, agent_idx);
    }

    // Collision detection and rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = NO_COLLISION;
        collision_check(env, agent_idx);
        int collision_state = env->entities[agent_idx].collision_state;

        if (collision_state > NO_COLLISION) {
            if (collision_state == VEHICLE_COLLISION && env->entities[agent_idx].respawn_timestep == -1) {
                env->agents[i].rewards[0] = env->reward_vehicle_collision;
                env->logs[i].episode_return += env->reward_vehicle_collision;
                env->logs[i].clean_collision_rate = 1.0f;
                env->logs[i].collision_rate = 1.0f;
            } else if (collision_state == OFFROAD) {
                env->agents[i].rewards[0] = env->reward_offroad_collision;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].episode_return += env->reward_offroad_collision;
            }
            if (!env->entities[agent_idx].reached_goal_this_episode) {
                env->entities[agent_idx].collided_before_goal = 1;
            }
        }

        float distance_to_goal = relative_distance_2d(
            env->entities[agent_idx].x, env->entities[agent_idx].y,
            env->entities[agent_idx].goal_position_x, env->entities[agent_idx].goal_position_y);

        int at_goal = distance_to_goal < MIN_DISTANCE_TO_GOAL;
        if (!at_goal && env->sim_infinite) {
            at_goal = infinite_passed_goal(&env->entities[agent_idx]);
        }
        // Every arrival pays +1, including a later lap of the same goal.
        // reward_goal_post_respawn stays in the ini and is not applied.
        if (at_goal) {
            env->agents[i].rewards[0] += 1.0f;
            env->logs[i].episode_return += 1.0f;
            env->entities[agent_idx].reached_goal = 1;
            env->entities[agent_idx].reached_goal_this_episode = 1;
        }

        // Squared bin-steps of accel and steer. One adjacent-bin flip costs
        // reward_jerk. The first step of an episode compares against 0.
        float* atn = env->agents[i].actions;
        if (atn) {
            float accel = ACCELERATION_VALUES[(int)atn[0]];
            float steer = STEERING_VALUES[(int)atn[1]];
            float accel_step = ACCELERATION_VALUES[1] - ACCELERATION_VALUES[0];
            float steer_step = STEERING_VALUES[1] - STEERING_VALUES[0];
            float da = (accel - env->entities[agent_idx].prev_accel) / accel_step;
            float ds = (steer - env->entities[agent_idx].prev_steer) / steer_step;
            float jerk_penalty = env->reward_jerk * (da * da + ds * ds);
            env->agents[i].rewards[0] += jerk_penalty;
            env->logs[i].episode_return += jerk_penalty;
            env->entities[agent_idx].prev_accel = accel;
            env->entities[agent_idx].prev_steer = steer;
        }

        // 1 at the episode max speed, 0 at a stop or at twice that
        // speed. Logged maps leave max_speed at 0 and skip this.
        float vmax = env->entities[agent_idx].max_speed;
        if (vmax >= SIM_SPEED_MIN && env->reward_speed != 0.0f) {
            float vx = env->entities[agent_idx].vx;
            float vy = env->entities[agent_idx].vy;
            float v = sqrtf(vx * vx + vy * vy);
            float closeness = 1.0f - fabsf(v - vmax) / vmax;
            if (closeness < 0.0f) closeness = 0.0f;
            float bonus = env->reward_speed * closeness;
            env->agents[i].rewards[0] += bonus;
            env->logs[i].episode_return += bonus;
        }
    }

    // Record the arrival, then send the car back to its own start. It
    // stays in the episode until the step limit and can score again.
    flow_diagram_record(env);

    // Respawn agents that reached goal
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->entities[agent_idx].reached_goal) {
            respawn_agent(env, agent_idx);
        }
    }

    // Score used the overlapping pose. The policy and the picture see
    // the cars already pushed apart.
    bump_agents(env);

    compute_observations(env);
}

struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    int car_assignments[MAX_AGENTS];
    Vector3 default_camera_position;
    Vector3 default_camera_target;
    Texture2D flow_tex;
    Color* flow_pix;
    int flow_tex_w;
    int flow_tex_h;
};

Client* make_client(Drive* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    int recording = puf_record_enabled();
    if (recording) {
        puf_record_ensure_display();
    }
    unsigned int flags = FLAG_MSAA_4X_HINT;
    if (recording) {
        flags |= FLAG_WINDOW_HIDDEN;
    }
    SetConfigFlags(flags);
    InitWindow(client->width, client->height, "Pufferflow");
    if (!recording) {
        int mw = GetMonitorWidth(GetCurrentMonitor());
        int mh = GetMonitorHeight(GetCurrentMonitor());
        if (mw >= 1400 && mh >= 900) {
            int w = mw - 40;
            int h = mh - 160;
            if (w > mw - 16) {
                w = mw - 16;
            }
            if (h > mh - 96) {
                h = mh - 96;
            }
            SetWindowSize(w, h);
            SetWindowPosition((mw - w) / 2, 32);
            client->width = (float)GetScreenWidth();
            client->height = (float)GetScreenHeight();
            if (client->width < 100.0f || client->height < 100.0f) {
                client->width = (float)w;
                client->height = (float)h;
            }
        }
        SetTargetFPS(30);
    } else {
        rlDisableBackfaceCulling();
    }
    client->puffers = LoadTexture("resources/shared/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand_r(&env->rng) % 4) + 1;
    }
    // Get initial target position from first active agent
    float map_center_x = 0.0f;
    float map_center_y = 0.0f;
    if (env->map_corners) {
        map_center_x = (env->map_corners[0] + env->map_corners[2]) / 2.0f;
        map_center_y = (env->map_corners[1] + env->map_corners[3]) / 2.0f;
    }
    Vector3 target_pos = { map_center_x, map_center_y, 1.0f };
    client->default_camera_position = (Vector3){
        map_center_x,
        map_center_y + 120.0f,
        175.0f
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){ 0.0f, -1.0f, 0.0f };  // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

// Right-hand column: playback, episode length, follow, agents.
#define SIM_COL_W 132.0f
#define SIM_AGENT_H 46.0f
#define SIM_DIST_H 78.0f
#define SIM_OBS_W 220.0f
#define SIM_OBS_H 264.0f
#define SIM_OBS_RANGE 12.0f

enum {
    SIM_PLAY = 0,
    SIM_PAUSE = 1,
    SIM_RESET = 2,
    SIM_FINITE = 3,
    SIM_INFINITE = 4,
    SIM_FOLLOW = 5,
    SIM_NBTN = 6
};

static void sim_rects(float width, Rectangle r[SIM_NBTN]) {
    float col_x = width - 8.0f - SIM_COL_W;
    float x = col_x + 8.0f;
    float bw = SIM_COL_W - 16.0f;
    float bh = 26.0f;
    float y = 16.0f;
    float gap = 6.0f;
    float grp = 18.0f;
    r[SIM_PLAY] = (Rectangle){x, y, bw, bh};
    y += bh + gap;
    r[SIM_PAUSE] = (Rectangle){x, y, bw, bh};
    y += bh + gap;
    r[SIM_RESET] = (Rectangle){x, y, bw, bh};
    y += bh + grp;
    r[SIM_FINITE] = (Rectangle){x, y, bw, bh};
    y += bh + gap;
    r[SIM_INFINITE] = (Rectangle){x, y, bw, bh};
    y += bh + grp;
    r[SIM_FOLLOW] = (Rectangle){x, y, bw, bh};
}

static Rectangle sim_agents_rect(float width) {
    Rectangle r[SIM_NBTN];
    sim_rects(width, r);
    float y = r[SIM_FOLLOW].y + r[SIM_FOLLOW].height + 18.0f;
    return (Rectangle){
        r[SIM_FOLLOW].x, y, r[SIM_FOLLOW].width, SIM_AGENT_H
    };
}

static int sim_speed_shown(Drive* env) {
    return env->agent_initialization == AGENT_INIT_STRUCTURED
        && env->sim_speed_max >= SIM_SPEED_MIN;
}

static Rectangle sim_speed_rect(float width) {
    Rectangle agents = sim_agents_rect(width);
    float y = agents.y + agents.height + 18.0f;
    return (Rectangle){agents.x, y, agents.width, SIM_AGENT_H};
}

static Rectangle sim_hi_rect(float width) {
    Rectangle lo = sim_speed_rect(width);
    float y = lo.y + lo.height + 18.0f;
    return (Rectangle){lo.x, y, lo.width, SIM_AGENT_H};
}

static Rectangle sim_dist_rect(float width) {
    Rectangle hi = sim_hi_rect(width);
    float y = hi.y + hi.height + 2.0f;
    return (Rectangle){hi.x, y, hi.width, SIM_DIST_H};
}

static void sim_agents_set(Drive* env, int n) {
    if (n < 1) n = 1;
    if (n > DRIVE_AGENT_MAX) n = DRIVE_AGENT_MAX;
    env->sim_agents_req = n;
}

static int sim_agents_from_x(float mouse_x, Rectangle zone) {
    float t = 0.0f;
    if (zone.width > 1.0f) {
        t = (mouse_x - zone.x) / zone.width;
    }
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    int span = DRIVE_AGENT_MAX - 1;
    int n = 1 + (int)(t * (float)span + 0.5f);
    if (n < 1) n = 1;
    if (n > DRIVE_AGENT_MAX) n = DRIVE_AGENT_MAX;
    return n;
}

static float sim_speed_clamp(float s) {
    if (s < SIM_SPEED_MIN) s = SIM_SPEED_MIN;
    if (s > MAX_SPEED) s = MAX_SPEED;
    return s;
}

static void sim_speed_set(Drive* env, float s) {
    env->sim_speed_req = sim_speed_clamp(s);
}

static float sim_speed_from_x(float mouse_x, Rectangle zone) {
    float t = 0.0f;
    if (zone.width > 1.0f) {
        t = (mouse_x - zone.x) / zone.width;
    }
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    float span = MAX_SPEED - SIM_SPEED_MIN;
    float s = SIM_SPEED_MIN + t * span;
    s = roundf(s * 100.0f) / 100.0f;
    return sim_speed_clamp(s);
}

static int sim_speed_pending(Drive* env) {
    return fabsf(env->sim_speed_req - env->sim_speed_on) > 0.005f;
}

static void sim_hi_set(Drive* env, float s) {
    env->sim_hi_req = sim_speed_clamp(s);
}

static int sim_hi_pending(Drive* env) {
    return fabsf(env->sim_hi_req - env->sim_hi_on) > 0.005f;
}

static Rectangle sim_col_rect(Drive* env) {
    float width = env->client->width;
    Rectangle zone = sim_agents_rect(width);
    if (sim_speed_shown(env)) {
        zone = sim_dist_rect(width);
    }
    float x = width - 8.0f - SIM_COL_W;
    float bot = zone.y + zone.height + 8.0f;
    return (Rectangle){x, 8.0f, SIM_COL_W, bot - 8.0f};
}

// Bottom-left heatmap. More lanes make a taller panel, up to the window.
static Rectangle flow_panel_rect(float width, float height, int lanes) {
    if (lanes < 1) {
        lanes = 1;
    }
    float panel_h = FLOW_PANEL_HEAD + FLOW_PANEL_FOOT
        + FLOW_PANEL_ROW * (float)lanes;
    float max_h = height - 208.0f;
    if (panel_h > max_h) {
        panel_h = max_h;
    }
    float panel_w = FLOW_PANEL_W;
    float max_w = width - SIM_COL_W - 24.0f;
    if (panel_w > max_w) {
        panel_w = max_w;
    }
    float y = height - 8.0f - panel_h;
    if (y < 8.0f) {
        y = 8.0f;
    }
    return (Rectangle){8.0f, y, panel_w, panel_h};
}

// Beside the speed panel. Narrower by the colorbar so the time axes match.
static Rectangle flow_hit_panel(float width, float height, int lanes) {
    Rectangle speed = flow_panel_rect(width, height, lanes);
    float x = speed.x + speed.width + FLOW_PANEL_GAP;
    float max_r = width - SIM_COL_W - 16.0f;
    float w = speed.width - 40.0f;
    if (x + w > max_r) {
        w = max_r - x;
    }
    if (w < 0.0f) {
        w = 0.0f;
    }
    return (Rectangle){x, speed.y, w, speed.height};
}

// Left of the control column, top aligned. 12 m radius.
static Rectangle sim_obs_rect(float width) {
    float col_x = width - 8.0f - SIM_COL_W;
    float x = col_x - 8.0f - SIM_OBS_W;
    if (x < 8.0f) {
        x = 8.0f;
    }
    return (Rectangle){x, 8.0f, SIM_OBS_W, SIM_OBS_H};
}

static int sim_ui_hot(Drive* env) {
    Client* client = env->client;
    if (!client) {
        return 0;
    }
    Vector2 mouse = GetMousePosition();
    if (CheckCollisionPointRec(mouse, sim_col_rect(env))) {
        return 1;
    }
    if (!puf_record_enabled()
        && CheckCollisionPointRec(mouse, sim_obs_rect(client->width))) {
        return 1;
    }
    if (puf_record_enabled()) {
        return 0;
    }
    if (env->agent_initialization != AGENT_INIT_STRUCTURED) {
        return 0;
    }
    if (!env->flow_speed || env->flow_rows <= 0 || env->flow_nlanes <= 0) {
        return 0;
    }
    Rectangle panel = flow_panel_rect(client->width, client->height,
        env->flow_nlanes);
    float min_h = FLOW_PANEL_HEAD + FLOW_PANEL_FOOT + 8.0f;
    if (panel.width >= 200.0f && panel.height >= min_h
        && CheckCollisionPointRec(mouse, panel)) {
        return 1;
    }
    Rectangle hits = flow_hit_panel(client->width, client->height,
        env->flow_nlanes);
    if (hits.width >= 180.0f && hits.height >= min_h
        && CheckCollisionPointRec(mouse, hits)) {
        return 1;
    }
    return 0;
}

static void sim_draw_button(Rectangle r, const char* label, Color fill, Color edge) {
    DrawRectangleRec(r, fill);
    DrawRectangleLinesEx(r, 1.0f, edge);
    int tw = MeasureText(label, 16);
    int tx = (int)(r.x + (r.width - (float)tw) * 0.5f);
    DrawText(label, tx, (int)r.y + 5, 16, (Color){241, 241, 241, 255});
}

static void sim_slider_row(Rectangle zone, const char* label,
        const char* num, float t, int pending) {
    Color num_c = pending
        ? (Color){240, 196, 120, 255}
        : (Color){241, 241, 241, 255};
    DrawText(label, (int)zone.x, (int)zone.y, 16, PUFF_WHITE);
    int nw = MeasureText(num, 16);
    DrawText(num, (int)(zone.x + zone.width) - nw, (int)zone.y, 16, num_c);
    Rectangle track = {zone.x, zone.y + 22.0f, zone.width, 8.0f};
    DrawRectangleRec(track, (Color){28, 32, 42, 255});
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    float knob_w = 10.0f;
    float kx = zone.x + t * zone.width - knob_w * 0.5f;
    if (kx < zone.x) kx = zone.x;
    if (kx > zone.x + zone.width - knob_w) kx = zone.x + zone.width - knob_w;
    Color knob = pending
        ? (Color){240, 196, 120, 255}
        : (Color){36, 110, 72, 255};
    DrawRectangle((int)kx, (int)track.y - 4, (int)knob_w, 16, knob);
}

// Staged uniform of target speed. A zero-width band is a spike.
static void sim_dist_fill(Vector2 a, Vector2 b, Vector2 c, Color color) {
    DrawTriangle(a, b, c, color);
    DrawTriangle(a, c, b, color);
}

static void sim_dist_draw(Drive* env, Rectangle zone, int pending) {
    DrawRectangle((int)(zone.x - 8.0f), (int)zone.y, (int)SIM_COL_W,
        (int)zone.height, (Color){12, 14, 20, 255});
    Rectangle plot = {zone.x, zone.y + 2.0f, zone.width, zone.height - 18.0f};
    float base = plot.y + plot.height - 1.0f;
    float top = plot.y + 2.0f;
    float usable = base - top;
    if (usable < 4.0f) {
        return;
    }
    Color fill = pending
        ? (Color){240, 196, 120, 150}
        : (Color){36, 110, 72, 210};
    Color edge = pending
        ? (Color){240, 196, 120, 255}
        : (Color){150, 210, 160, 255};
    Color dim = {150, 156, 168, 255};
    float inner_x = plot.x + 3.0f;
    float inner_w = plot.width - 6.0f;
    if (inner_w < 8.0f) {
        inner_w = plot.width;
        inner_x = plot.x;
    }
    float lo = env->sim_speed_req;
    float hi = env->sim_hi_req;
    if (lo > hi) {
        float swap = lo;
        lo = hi;
        hi = swap;
    }
    float span = MAX_SPEED - SIM_SPEED_MIN;
    if (span < 1.0f) {
        span = 1.0f;
    }
    DrawLine((int)inner_x, (int)base,
        (int)(inner_x + inner_w), (int)base, (Color){70, 76, 88, 255});
    if (lo < SIM_SPEED_MIN) lo = SIM_SPEED_MIN;
    if (hi > MAX_SPEED) hi = MAX_SPEED;
    if (hi - lo <= 0.005f) {
        float t = (lo - SIM_SPEED_MIN) / span;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        float x = inner_x + t * inner_w;
        Vector2 tip = {x, top};
        Vector2 bl = {x - 5.0f, base};
        Vector2 br = {x + 5.0f, base};
        sim_dist_fill(tip, br, bl, fill);
        DrawLineEx(tip, bl, 1.25f, edge);
        DrawLineEx(tip, br, 1.25f, edge);
    } else {
        float t0 = (lo - SIM_SPEED_MIN) / span;
        float t1 = (hi - SIM_SPEED_MIN) / span;
        if (t0 < 0.0f) t0 = 0.0f;
        if (t1 > 1.0f) t1 = 1.0f;
        float x0 = inner_x + t0 * inner_w;
        float x1 = inner_x + t1 * inner_w;
        Vector2 a = {x0, top};
        Vector2 b = {x1, top};
        Vector2 c = {x1, base};
        Vector2 d = {x0, base};
        sim_dist_fill(d, c, b, fill);
        sim_dist_fill(d, b, a, fill);
        DrawLineEx(a, b, 1.25f, edge);
        DrawLineEx(a, d, 1.25f, edge);
        DrawLineEx(b, c, 1.25f, edge);
    }
    int nagt = env->active_agent_count;
    if (nagt > DRIVE_AGENT_MAX) nagt = DRIVE_AGENT_MAX;
    if (env->active_agent_indices && env->entities) {
        Color rug = {241, 241, 241, 200};
        for (int i = 0; i < nagt; i++) {
            int idx = env->active_agent_indices[i];
            if (idx < 0 || idx >= env->num_entities) continue;
            float m = env->entities[idx].max_speed;
            if (m < SIM_SPEED_MIN) continue;
            float rt = (m - SIM_SPEED_MIN) / span;
            if (rt < 0.0f) rt = 0.0f;
            if (rt > 1.0f) rt = 1.0f;
            int rx = (int)(inner_x + rt * inner_w);
            DrawLine(rx, (int)base + 2, rx, (int)base + 5, rug);
        }
    }
    DrawText("1", (int)inner_x, (int)base + 3, 10, dim);
    int tw = MeasureText("100", 10);
    DrawText("100", (int)(inner_x + inner_w) - tw, (int)base + 3, 10, dim);
}

static void sim_choice(Rectangle r, const char* label, int on, int hot, int amber) {
    Color fill = (Color){28, 32, 42, 255};
    Color edge = (Color){70, 76, 88, 255};
    if (on && amber) {
        fill = (Color){140, 78, 42, 255};
        edge = (Color){240, 196, 120, 255};
    } else if (on) {
        fill = (Color){36, 110, 72, 255};
        edge = (Color){150, 210, 160, 255};
    } else if (hot) {
        fill = (Color){46, 52, 66, 255};
    }
    if (hot) {
        edge = (Color){241, 241, 241, 255};
    }
    sim_draw_button(r, label, fill, edge);
}

static void sim_checkpoint_rejected(Drive* env) {
    const char* msg = env->checkpoint_error[0]
        ? env->checkpoint_error
        : "Checkpoint cannot be loaded, showing random behavior";
    fprintf(stderr, "%s\n", msg);
    fflush(stderr);
}

static void sim_use_finite(Drive* env) {
    int was_infinite = env->sim_infinite;
    env->sim_infinite = 0;
    if (was_infinite && env->timestep >= env->episode_length) {
        puf_reset(env);
    }
}

static void sim_finite_label(Drive* env, char* dst, int n) {
    float sec = (float)env->episode_length * SIM_DT;
    if (fabsf(sec - roundf(sec)) < 0.05f) {
        snprintf(dst, (size_t)n, "%.0f s", roundf(sec));
    } else {
        snprintf(dst, (size_t)n, "%.1f s", sec);
    }
}

// Ego meters to the crop. Forward is up, left is left.
static void sim_obs_seg(float cx, float cy, float ppm,
        float f0, float l0, float f1, float l1,
        float thick, Color color) {
    Vector2 a = {cx - l0 * ppm, cy - f0 * ppm};
    Vector2 b = {cx - l1 * ppm, cy - f1 * ppm};
    DrawLineEx(a, b, thick, color);
}

// vlen is meters of travel. The whole arrow, head included, grows with it.
static void sim_obs_arrow(float cx, float cy, float ppm,
        float f, float l, float c, float s, float vlen, Color color) {
    if (vlen < 0.15f) {
        return;
    }
    float f1 = f + c * vlen;
    float l1 = l + s * vlen;
    sim_obs_seg(cx, cy, ppm, f, l, f1, l1, 1.5f, color);
    float dx = -s;
    float dy = -c;
    float norm = sqrtf(dx * dx + dy * dy);
    if (norm < 1.0e-4f) {
        return;
    }
    dx /= norm;
    dy /= norm;
    float reach = vlen * ppm;
    float head = reach * 0.32f;
    if (head < 4.5f) {
        head = 4.5f;
    }
    if (head > 13.0f) {
        head = 13.0f;
    }
    float wing = head * 0.55f;
    float px = -dy;
    float py = dx;
    float tipx = cx - l1 * ppm;
    float tipy = cy - f1 * ppm;
    Vector2 tip = {tipx, tipy};
    Vector2 left = {
        tipx - dx * head + px * wing,
        tipy - dy * head + py * wing
    };
    Vector2 right = {
        tipx - dx * head - px * wing,
        tipy - dy * head - py * wing
    };
    sim_dist_fill(tip, left, right, color);
}

// Decode the selected car's observation. Empty slots stay blank.
static void sim_obs_draw(Drive* env) {
    if (!env->client || puf_record_enabled()) {
        return;
    }
    Rectangle rect = sim_obs_rect(env->client->width);
    DrawRectangleRec(rect, (Color){12, 14, 20, 255});
    DrawRectangleLinesEx(rect, 1.0f, (Color){70, 76, 88, 255});

    int n = env->active_agent_count;
    int slot = env->human_agent_idx;
    if (n < 0) {
        n = 0;
    }
    if (slot < 0 || slot >= n) {
        slot = 0;
    }
    const char* title = "Agent";
    if (n > 0) {
        title = TextFormat("Agent %d", slot);
    }
    DrawText(title, (int)rect.x + 10, (int)rect.y + 8, 16, PUFF_WHITE);

    float* obs = NULL;
    if (n > 0 && slot < DRIVE_AGENT_MAX) {
        obs = env->agents[slot].observations;
    }
    Color dim = {150, 156, 168, 255};
    if (!obs) {
        DrawText("no view", (int)rect.x + 10, (int)rect.y + 30, 14, dim);
        return;
    }
    float spd = obs[2] / OBS_SPEED_SCALE;
    float mx = obs[EGO_FEATURES - 1] * MAX_SPEED;
    int hit = obs[5] > 0.5f;
    float gf = obs[0] / OBS_GOAL_SCALE;
    float gl = obs[1] / OBS_GOAL_SCALE;
    float gd = sqrtf(gf * gf + gl * gl);
    char cap[64];
    snprintf(cap, sizeof(cap), "v %.1f  max %.1f%s",
        spd, mx, hit ? "  hit" : "");
    Color cap_c = hit ? (Color){210, 80, 70, 255} : PUFF_WHITE;
    DrawText(cap, (int)rect.x + 10, (int)rect.y + 28, 14, cap_c);
    char goal_txt[32];
    if (gd >= 10.0f) {
        snprintf(goal_txt, sizeof(goal_txt), "goal %.0f m", gd);
    } else {
        snprintf(goal_txt, sizeof(goal_txt), "goal %.1f m", gd);
    }
    DrawText(goal_txt, (int)rect.x + 10, (int)rect.y + 44, 14,
        (Color){90, 190, 110, 255});

    Rectangle plot = {
        rect.x + 10.0f, rect.y + 64.0f,
        rect.width - 20.0f, rect.height - 74.0f
    };
    float side = plot.width;
    if (plot.height < side) {
        side = plot.height;
    }
    float ppm = (side * 0.5f) / SIM_OBS_RANGE;
    float cx = plot.x + plot.width * 0.5f;
    float cy = plot.y + plot.height * 0.5f;

    BeginScissorMode((int)plot.x, (int)plot.y,
        (int)plot.width, (int)plot.height);
    DrawCircleLines((int)cx, (int)cy, ppm * (SIM_OBS_RANGE - 0.5f),
        (Color){70, 76, 88, 255});

    // One midpoint per segment. Edges are red, lanes and lines are white.
    int road0 = EGO_FEATURES + PARTNER_FEATURES * MAX_OBS_AGENTS;
    for (int pass = 0; pass < 2; pass++) {
        for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
            float* road = obs + road0 + k * ROAD_FEATURES;
            if (road[0] == 0.0f && road[1] == 0.0f && road[2] == 0.0f) {
                continue;
            }
            int edge = (int)(road[6] + 0.5f) >= 2;
            if (edge != pass) {
                continue;
            }
            float mid_f = road[0] / OBS_POSITION_SCALE;
            float mid_l = road[1] / OBS_POSITION_SCALE;
            Color pt = {241, 241, 241, 255};
            if (edge) {
                pt = (Color){210, 64, 64, 255};
            }
            DrawCircleV((Vector2){cx - mid_l * ppm, cy - mid_f * ppm},
                2.2f, pt);
        }
    }

    for (int j = 0; j < MAX_OBS_AGENTS; j++) {
        float* p = obs + EGO_FEATURES + j * PARTNER_FEATURES;
        if (p[0] == 0.0f && p[1] == 0.0f
            && p[2] == 0.0f && p[3] == 0.0f) {
            continue;
        }
        float f = p[0] / OBS_POSITION_SCALE;
        float l = p[1] / OBS_POSITION_SCALE;
        float hw = p[2] * MAX_VEH_WIDTH * 0.5f;
        float hl = p[3] * MAX_VEH_LEN * 0.5f;
        if (hw < 0.4f) {
            hw = 0.4f;
        }
        if (hl < 0.6f) {
            hl = 0.6f;
        }
        float ang = atan2f(p[5], p[4]);
        float c = cosf(ang);
        float s = sinf(ang);
        float lf = -s;
        float ll = c;
        Color car = {240, 150, 60, 255};
        float f0 = f + hl * c + hw * lf;
        float l0 = l + hl * s + hw * ll;
        float f1 = f + hl * c - hw * lf;
        float l1 = l + hl * s - hw * ll;
        float f2 = f - hl * c - hw * lf;
        float l2 = l - hl * s - hw * ll;
        float f3 = f - hl * c + hw * lf;
        float l3 = l - hl * s + hw * ll;
        sim_obs_seg(cx, cy, ppm, f0, l0, f1, l1, 1.25f, car);
        sim_obs_seg(cx, cy, ppm, f1, l1, f2, l2, 1.25f, car);
        sim_obs_seg(cx, cy, ppm, f2, l2, f3, l3, 1.25f, car);
        sim_obs_seg(cx, cy, ppm, f3, l3, f0, l0, 1.25f, car);
        // From the nose. Shorter than the selected car so it matches the box.
        sim_obs_arrow(cx, cy, ppm, f + c * hl, l + s * hl, c, s,
            p[6] * MAX_SPEED * 0.10f, PUFF_WHITE);
    }

    int outside = gd > SIM_OBS_RANGE;
    float df = gf;
    float dl = gl;
    Color goal = {90, 190, 110, 255};
    if (outside && gd > 1.0e-3f) {
        float rim = SIM_OBS_RANGE - 1.5f;
        df = gf * rim / gd;
        dl = gl * rim / gd;
    }
    DrawCircleV((Vector2){cx - dl * ppm, cy - df * ppm}, 3.5f, goal);
    if (outside && gd > 1.0e-3f) {
        float ux = gf / gd;
        float uy = gl / gd;
        sim_obs_seg(cx, cy, ppm,
            df - ux * 6.0f, dl - uy * 6.0f, df, dl, 1.5f, goal);
    }

    float ego_hw = obs[3] * MAX_VEH_WIDTH * 0.5f;
    float ego_hl = obs[4] * MAX_VEH_LEN * 0.5f;
    if (ego_hw < 0.4f) {
        ego_hw = 0.4f;
    }
    if (ego_hl < 0.6f) {
        ego_hl = 0.6f;
    }
    Color ego = hit ? (Color){210, 70, 70, 255} : (Color){241, 241, 241, 255};
    sim_obs_seg(cx, cy, ppm, ego_hl, ego_hw, ego_hl, -ego_hw, 1.25f, ego);
    sim_obs_seg(cx, cy, ppm, ego_hl, -ego_hw, -ego_hl, -ego_hw, 1.25f, ego);
    sim_obs_seg(cx, cy, ppm, -ego_hl, -ego_hw, -ego_hl, ego_hw, 1.25f, ego);
    sim_obs_seg(cx, cy, ppm, -ego_hl, ego_hw, ego_hl, ego_hw, 1.25f, ego);
    sim_obs_arrow(cx, cy, ppm, ego_hl, 0.0f, 1.0f, 0.0f,
        spd * 0.25f, PUFF_WHITE);
    EndScissorMode();
    DrawRectangleLinesEx(plot, 1.0f, (Color){70, 76, 88, 255});
}

static void sim_transport_draw(Drive* env) {
    if (!env->client) {
        return;
    }
    Rectangle r[SIM_NBTN];
    sim_rects(env->client->width, r);
    char finite[16];
    sim_finite_label(env, finite, (int)sizeof(finite));
    const char* labels[SIM_NBTN] = {
        "Play", "Pause", "Reset", finite, "Infinite", "Follow"
    };
    int on[SIM_NBTN] = {
        !env->sim_paused,
        env->sim_paused,
        0,
        !env->sim_infinite,
        env->sim_infinite,
        env->sim_follow
    };
    DrawRectangleRec(sim_col_rect(env), (Color){12, 14, 20, 230});
    Vector2 mouse = GetMousePosition();
    Color sep = (Color){70, 76, 88, 255};
    int y1 = (int)(r[SIM_RESET].y + r[SIM_RESET].height
        + r[SIM_FINITE].y) / 2;
    int x0 = (int)r[SIM_PLAY].x;
    int x1 = (int)(r[SIM_PLAY].x + r[SIM_PLAY].width);
    DrawLine(x0, y1, x1, y1, sep);
    int y_inf = (int)(r[SIM_INFINITE].y + r[SIM_INFINITE].height
        + r[SIM_FOLLOW].y) / 2;
    DrawLine(x0, y_inf, x1, y_inf, sep);
    for (int i = 0; i < SIM_NBTN; i++) {
        int hot = CheckCollisionPointRec(mouse, r[i]);
        sim_choice(r[i], labels[i], on[i], hot, i == SIM_PAUSE);
    }
    Rectangle zone = sim_agents_rect(env->client->width);
    int y3 = (int)(r[SIM_FOLLOW].y + r[SIM_FOLLOW].height + zone.y) / 2;
    DrawLine(x0, y3, x1, y3, sep);
    char num[8];
    snprintf(num, sizeof(num), "%d", env->sim_agents_req);
    float t = (float)(env->sim_agents_req - 1) / (float)(DRIVE_AGENT_MAX - 1);
    sim_slider_row(zone, "Agents", num, t,
        env->sim_agents_req != env->sim_agents_on);
    if (sim_speed_shown(env)) {
        Rectangle spd = sim_speed_rect(env->client->width);
        int y4 = (int)(zone.y + zone.height + spd.y) / 2;
        DrawLine(x0, y4, x1, y4, sep);
        char sn[16];
        snprintf(sn, sizeof(sn), "%.2f", env->sim_speed_req);
        float span = MAX_SPEED - SIM_SPEED_MIN;
        float st = 0.0f;
        if (span > 1e-3f) {
            st = (env->sim_speed_req - SIM_SPEED_MIN) / span;
        }
        sim_slider_row(spd, "Min", sn, st, sim_speed_pending(env));
        Rectangle sd = sim_hi_rect(env->client->width);
        int y5 = (int)(spd.y + spd.height + sd.y) / 2;
        DrawLine(x0, y5, x1, y5, sep);
        char dn[16];
        snprintf(dn, sizeof(dn), "%.2f", env->sim_hi_req);
        float dt = 0.0f;
        if (span > 1e-3f) {
            dt = (env->sim_hi_req - SIM_SPEED_MIN) / span;
        }
        sim_slider_row(sd, "Max", dn, dt, sim_hi_pending(env));
        int dist_pending = sim_speed_pending(env) || sim_hi_pending(env);
        sim_dist_draw(env, sim_dist_rect(env->client->width), dist_pending);
    }
    sim_obs_draw(env);
}

static void sim_camera_bev(Drive* env) {
    Client* client = env->client;
    if (!client) {
        return;
    }
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f};
    client->camera.fovy = 45.0f;
}

// Read once per frame, before the step, so a click applies on this tick.
static inline void puf_sim_poll(Drive* env) {
    if (!env->client || !IsWindowReady() || puf_record_enabled()) {
        env->sim_agents_drag = 0;
        env->sim_speed_drag = 0;
        env->sim_hi_drag = 0;
        return;
    }
    // Drop the one-frame hold before reading this frame's click.
    env->sim_hold_step = 0;
    if (IsKeyPressed(KEY_P)) {
        env->sim_paused = !env->sim_paused;
    }
    Rectangle zone = sim_agents_rect(env->client->width);
    Rectangle spd = sim_speed_rect(env->client->width);
    Rectangle sd = sim_hi_rect(env->client->width);
    int speed_on = sim_speed_shown(env);
    Vector2 mouse = GetMousePosition();
    int over = CheckCollisionPointRec(mouse, zone);
    int over_spd = speed_on && CheckCollisionPointRec(mouse, spd);
    int over_hi = speed_on && CheckCollisionPointRec(mouse, sd);
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && over) {
        env->sim_agents_drag = 1;
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && over_spd) {
        env->sim_speed_drag = 1;
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && over_hi) {
        env->sim_hi_drag = 1;
    }
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        env->sim_agents_drag = 0;
        env->sim_speed_drag = 0;
        env->sim_hi_drag = 0;
    }
    if (env->sim_agents_drag) {
        sim_agents_set(env, sim_agents_from_x(mouse.x, zone));
    }
    if (env->sim_speed_drag) {
        sim_speed_set(env, sim_speed_from_x(mouse.x, spd));
    }
    if (env->sim_hi_drag) {
        sim_hi_set(env, sim_speed_from_x(mouse.x, sd));
    }
    float wheel = GetMouseWheelMove();
    int dragging = env->sim_agents_drag || env->sim_speed_drag
        || env->sim_hi_drag;
    if (wheel != 0.0f && !dragging) {
        int step = 1;
        if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
            step = 10;
        }
        int dir = wheel > 0.0f ? 1 : -1;
        if (over) {
            sim_agents_set(env, env->sim_agents_req + dir * step);
        } else if (over_spd) {
            sim_speed_set(env, env->sim_speed_req + (float)(dir * step));
        } else if (over_hi) {
            sim_hi_set(env, env->sim_hi_req + (float)(dir * step));
        }
    }
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        return;
    }
    if (over || over_spd || over_hi || dragging) {
        return;
    }
    Rectangle r[SIM_NBTN];
    sim_rects(env->client->width, r);
    if (CheckCollisionPointRec(mouse, r[SIM_PLAY])) {
        if (env->sim_agents_req != env->sim_agents_on) {
            env->sim_agents_apply = 1;
        }
        if (sim_speed_pending(env)) {
            env->sim_speed_apply = 1;
        }
        if (sim_hi_pending(env)) {
            env->sim_hi_apply = 1;
        }
        env->sim_paused = 0;
    } else if (CheckCollisionPointRec(mouse, r[SIM_PAUSE])) {
        env->sim_paused = 1;
    } else if (CheckCollisionPointRec(mouse, r[SIM_RESET])) {
        // Manual reset is not a logged episode. Keep mode and pause.
        puf_reset(env);
        env->sim_hold_step = 1;
    } else if (CheckCollisionPointRec(mouse, r[SIM_FINITE])) {
        sim_use_finite(env);
    } else if (CheckCollisionPointRec(mouse, r[SIM_INFINITE])) {
        env->sim_infinite = 1;
    } else if (CheckCollisionPointRec(mouse, r[SIM_FOLLOW])) {
        env->sim_follow = !env->sim_follow;
        if (!env->sim_follow) {
            sim_camera_bev(env);
        }
    }
}

// Camera control functions
void handle_camera_controls(Drive* env) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    Client* client = env->client;
    if (!client) {
        return;
    }
    float camera_move_speed = 0.5f;

    // Follow owns the map position. A click on the panels must not pan.
    if (env->sim_follow) {
        is_dragging = false;
    } else if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        if (!sim_ui_hot(env)) {
            prev_mouse_pos = GetMousePosition();
            is_dragging = true;
        }
    }
    
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }
    
    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {
            (current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;
        
        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0 && !sim_ui_hot(env)) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z
        };
        
        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

// Keep the current overhead offset and aim it at the selected car.
static void sim_follow_camera(Drive* env) {
    Client* client = env->client;
    if (!env->sim_follow || !client || puf_record_enabled()) {
        return;
    }
    if (env->active_agent_count <= 0 || !env->entities
        || !env->active_agent_indices) {
        return;
    }
    int slot = env->human_agent_idx;
    if (slot < 0 || slot >= env->active_agent_count) {
        slot = 0;
    }
    int idx = env->active_agent_indices[slot];
    if (idx < 0 || idx >= env->num_entities) {
        return;
    }
    Entity* agent = &env->entities[idx];
    if (agent->x == INVALID_POSITION || agent->y == INVALID_POSITION) {
        return;
    }
    float ox = client->camera.position.x - client->camera.target.x;
    float oy = client->camera.position.y - client->camera.target.y;
    float oz = client->camera.position.z - client->camera.target.z;
    client->camera.target = (Vector3){agent->x, agent->y, 1.0f};
    client->camera.position = (Vector3){
        agent->x + ox, agent->y + oy, 1.0f + oz
    };
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f};
    client->camera.fovy = 45.0f;
}

void draw_agent_obs(Drive* env, int agent_index) {
    float diamond_height = 3.0f;
    float diamond_width = 1.5f;
    float diamond_z = 8.0f;

    Vector3 top = {0, 0, diamond_z + diamond_height / 2};
    Vector3 bot = {0, 0, diamond_z - diamond_height / 2};
    Vector3 fwd = {0, diamond_width / 2, diamond_z};
    Vector3 bck = {0, -diamond_width / 2, diamond_z};
    Vector3 lft = {-diamond_width / 2, 0, diamond_z};
    Vector3 rgt = {diamond_width / 2, 0, diamond_z};

    DrawTriangle3D(top, fwd, rgt, PUFF_CYAN);
    DrawTriangle3D(top, rgt, bck, PUFF_CYAN);
    DrawTriangle3D(top, bck, lft, PUFF_CYAN);
    DrawTriangle3D(top, lft, fwd, PUFF_CYAN);
    DrawTriangle3D(bot, rgt, fwd, PUFF_CYAN);
    DrawTriangle3D(bot, bck, rgt, PUFF_CYAN);
    DrawTriangle3D(bot, lft, bck, PUFF_CYAN);
    DrawTriangle3D(bot, fwd, lft, PUFF_CYAN);

    if (!IsKeyDown(KEY_LEFT_CONTROL)) return;

    if (agent_index < 0 || agent_index >= env->active_agent_count) return;
    if (!env->agents[agent_index].observations) return;
    float* agent_obs = env->agents[agent_index].observations;

    // Draw goal
    float goal_x = agent_obs[0] / OBS_GOAL_SCALE;
    float goal_y = agent_obs[1] / OBS_GOAL_SCALE;
    DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, GREEN);

    // Draw partner observations
    int obs_idx = EGO_FEATURES;
    for (int j = 0; j < MAX_OBS_AGENTS; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += PARTNER_FEATURES;
            continue;
        }
        float x = agent_obs[obs_idx] / OBS_POSITION_SCALE;
        float y = agent_obs[obs_idx + 1] / OBS_POSITION_SCALE;
        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, 1}, ORANGE);

        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float angle = atan2f(theta_y, theta_x);
        // Draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 7.5f;

        float ax = x + arrow_length * cosf(angle);
        float ay = y + arrow_length * sinf(angle);
        DrawLine3D((Vector3){x, y, 1}, (Vector3){ax, ay, 1}, PUFF_WHITE);

        float arrow_size = 2.0f;
        float dx = ax - x, dy = ay - y;
        float len = sqrtf(dx * dx + dy * dy);
        if (len > 0) {
            dx /= len; dy /= len;
            float px = -dy * arrow_size, py = dx * arrow_size;
            DrawLine3D((Vector3){ax, ay, 1}, (Vector3){ax - dx * arrow_size + px, ay - dy * arrow_size + py, 1}, PUFF_WHITE);
            DrawLine3D((Vector3){ax, ay, 1}, (Vector3){ax - dx * arrow_size - px, ay - dy * arrow_size - py, 1}, PUFF_WHITE);
        }
        obs_idx += PARTNER_FEATURES;
    }

    // Draw road edge observations
    int map_start_idx = EGO_FEATURES + PARTNER_FEATURES * MAX_OBS_AGENTS;
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
        int idx = map_start_idx + k * ROAD_FEATURES;
        if (agent_obs[idx] == 0 && agent_obs[idx + 1] == 0) continue;
        int entity_type = (int)agent_obs[idx + 6];
        if (entity_type + ROAD_LANE != ROAD_EDGE) continue;

        float x_mid = agent_obs[idx] / OBS_POSITION_SCALE;
        float y_mid = agent_obs[idx + 1] / OBS_POSITION_SCALE;
        float rel_angle = atan2f(agent_obs[idx + 5], agent_obs[idx + 4]);
        float seg_len = agent_obs[idx + 2] * MAX_ROAD_SEGMENT_LENGTH;

        float x_start = x_mid - seg_len * cosf(rel_angle);
        float y_start = y_mid - seg_len * sinf(rel_angle);
        float x_end = x_mid + seg_len * cosf(rel_angle);
        float y_end = y_mid + seg_len * sinf(rel_angle);

        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_mid, y_mid, 1}, PUFF_CYAN);
        DrawCube((Vector3){x_mid, y_mid, 1}, 0.5f, 0.5f, 0.5f, PUFF_CYAN);
        DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
    }
}

void draw_road_edge(Drive* env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};
    Color CURB_SIDE = (Color){180, 180, 180, 255};
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    float curb_height = 0.5f;
    float curb_width = 0.3f;

    Vector3 direction = {end_x - start_x, end_y - start_y, 0};
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);
    Vector3 nd = {direction.x / length, direction.y / length, 0};
    Vector3 perp = {-nd.y, nd.x, 0};

    Vector3 b1 = {start_x - perp.x * curb_width / 2, start_y - perp.y * curb_width / 2, 1.0f};
    Vector3 b2 = {start_x + perp.x * curb_width / 2, start_y + perp.y * curb_width / 2, 1.0f};
    Vector3 b3 = {end_x + perp.x * curb_width / 2, end_y + perp.y * curb_width / 2, 1.0f};
    Vector3 b4 = {end_x - perp.x * curb_width / 2, end_y - perp.y * curb_width / 2, 1.0f};

    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    DrawTriangle3D(b1, t1, b2, CURB_SIDE); DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE); DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE); DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE); DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

// The overview camera sits ~200 m off the map center, so a moving car is a
// few pixels and then leaves the frame. While recording, chase one agent
// (the configured human_agent_idx) from just behind and above.
static void drive_record_camera(Drive* env, Client* client) {
    if (env->active_agent_count <= 0 || !env->active_agent_indices || !env->entities) {
        return;
    }
    int slot = env->human_agent_idx;
    if (slot < 0 || slot >= env->active_agent_count) {
        slot = 0;
    }
    Entity* agent = &env->entities[env->active_agent_indices[slot]];
    float hx = cosf(agent->heading);
    float hy = sinf(agent->heading);
    const float back = 12.0f;
    const float height = 7.0f;
    const float ahead = 6.0f;
    client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    client->camera.fovy = 30.0f;
    client->camera.position = (Vector3){
        agent->x - back * hx,
        agent->y - back * hy,
        height
    };
    client->camera.target = (Vector3){
        agent->x + ahead * hx,
        agent->y + ahead * hy,
        1.0f
    };
}

// CIRCLES Fig. 7 bar. The first stop is 10 mph, the last is 80 mph.
static Color flow_mph_color(float mph) {
    static const unsigned char stop[FLOW_CMAP_N][3] = {
        {151, 27, 41}, {173, 44, 45}, {195, 60, 49}, {222, 107, 74},
        {235, 147, 94}, {244, 197, 126}, {251, 235, 165}, {255, 255, 197},
        {232, 244, 166}, {205, 230, 140}, {186, 220, 126}, {150, 202, 115},
        {114, 181, 105}, {78, 157, 91}, {57, 125, 73}, {44, 104, 61},
    };
    if (mph < FLOW_MPH_LO) {
        mph = FLOW_MPH_LO;
    }
    if (mph > FLOW_MPH_HI) {
        mph = FLOW_MPH_HI;
    }
    float x = (mph - FLOW_MPH_LO) / (FLOW_MPH_HI - FLOW_MPH_LO);
    x *= (FLOW_CMAP_N - 1);
    int i = (int)x;
    if (i >= FLOW_CMAP_N - 1) {
        i = FLOW_CMAP_N - 2;
        x = (float)(FLOW_CMAP_N - 1);
    }
    float f = x - (float)i;
    float r = stop[i][0] + f * ((int)stop[i + 1][0] - (int)stop[i][0]);
    float g = stop[i][1] + f * ((int)stop[i + 1][1] - (int)stop[i][1]);
    float b = stop[i][2] + f * ((int)stop[i + 1][2] - (int)stop[i][2]);
    return (Color){
        (unsigned char)(r + 0.5f),
        (unsigned char)(g + 0.5f),
        (unsigned char)(b + 0.5f),
        255
    };
}

static void flow_diagram_tex(Client* client, int w, int h) {
    if (client->flow_tex.id != 0 &&
            client->flow_tex_w == w && client->flow_tex_h == h) {
        return;
    }
    if (client->flow_tex.id != 0) {
        UnloadTexture(client->flow_tex);
        client->flow_tex.id = 0;
    }
    free(client->flow_pix);
    client->flow_pix = NULL;
    size_t n = (size_t)w * (size_t)h;
    client->flow_pix = (Color*)calloc(n, sizeof(Color));
    if (!client->flow_pix) {
        fprintf(stderr, "velocity diagram texture alloc failed\n");
        exit(1);
    }
    Image img = GenImageColor(w, h, (Color){20, 24, 32, 255});
    client->flow_tex = LoadTextureFromImage(img);
    UnloadImage(img);
    SetTextureFilter(client->flow_tex, TEXTURE_FILTER_POINT);
    client->flow_tex_w = w;
    client->flow_tex_h = h;
}

// One texture row per lane. Empty lane is the dark cell, not a slow speed.
static void flow_diagram_splat(Drive* env, Client* client) {
    int times = env->flow_rows;
    int lanes = env->flow_nlanes;
    int origin = 0;
    if (env->sim_infinite && env->timestep >= times) {
        origin = (env->timestep - times + 1) % times;
    }
    Color empty = {20, 24, 32, 255};
    for (int lane = 0; lane < lanes; lane++) {
        for (int t = 0; t < times; t++) {
            int src = origin + t;
            if (src >= times) {
                src -= times;
            }
            size_t si = (size_t)src * (size_t)lanes + (size_t)lane;
            size_t p = (size_t)lane * (size_t)times + (size_t)t;
            float spd = env->flow_speed[si];
            if (spd < 0.0f) {
                client->flow_pix[p] = empty;
            } else {
                client->flow_pix[p] = flow_mph_color(spd * FLOW_MPS_TO_MPH);
            }
        }
    }
}

// Lane index, top to bottom, same order as the numbered segments.
static Color flow_lane_color(int lane) {
    static const Color palette[4] = {
        {130, 180, 255, 255},
        {90, 210, 190, 255},
        {230, 190, 80, 255},
        {200, 140, 255, 255},
    };
    if (lane < 0) {
        lane = 0;
    }
    return palette[lane & 3];
}

// Same hue as the diagram chip, only a hint above the asphalt.
static Color flow_lane_road_color(int lane) {
    Color c = flow_lane_color(lane);
    float k = 0.12f;
    float r = ROAD_COLOR.r + (c.r - ROAD_COLOR.r) * k;
    float g = ROAD_COLOR.g + (c.g - ROAD_COLOR.g) * k;
    float b = ROAD_COLOR.b + (c.b - ROAD_COLOR.b) * k;
    return (Color){
        (unsigned char)r, (unsigned char)g, (unsigned char)b, 255
    };
}

static void flow_time_window(Drive* env, float* now, float* axis0, float* axis1) {
    int times = env->flow_rows;
    float duration = (float)times * SIM_DT;
    if (duration < SIM_DT) {
        duration = SIM_DT;
    }
    float tnow = env->timestep * SIM_DT;
    if (tnow < 0.0f) {
        tnow = 0.0f;
    }
    float a0 = 0.0f;
    float a1 = duration;
    if (env->sim_infinite && env->timestep >= times) {
        a0 = (float)(env->timestep - times + 1) * SIM_DT;
        a1 = tnow;
    } else if (tnow > duration) {
        tnow = duration;
    }
    *now = tnow;
    *axis0 = a0;
    *axis1 = a1;
}

static int flow_name_w(Drive* env, int lanes) {
    int name_w = MeasureText("Segment", 14);
    for (int lane = 0; lane < lanes; lane++) {
        int w = 14 + MeasureText(env->flow_lane_name[lane], 15);
        if (w > name_w) {
            name_w = w;
        }
    }
    return name_w;
}

static void flow_draw_time_axis(float plot_x, float plot_w, float bot,
    float axis0, float axis1) {
    float axis_span = axis1 - axis0;
    if (axis_span < SIM_DT) {
        axis_span = SIM_DT;
    }
    float tick = 5.0f;
    if (axis_span <= 12.0f) {
        tick = 2.0f;
    }
    if (axis_span <= 3.0f) {
        tick = 0.5f;
    }
    float t = ceilf((axis0 - 0.001f) / tick) * tick;
    if (t < axis0) {
        t = axis0;
    }
    for (; t <= axis1 + 0.001f; t += tick) {
        float draw_t = t;
        if (draw_t > axis1) {
            draw_t = axis1;
        }
        if (fabsf(draw_t) < 0.05f) {
            draw_t = 0.0f;
        }
        float x = plot_x + ((draw_t - axis0) / axis_span) * plot_w;
        const char* label;
        if (fabsf(draw_t - roundf(draw_t)) < 0.05f) {
            label = TextFormat("%.0fs", roundf(draw_t));
        } else {
            label = TextFormat("%.1fs", draw_t);
        }
        int tw = MeasureText(label, 14);
        int lx = (int)x - tw / 2;
        if (draw_t <= axis0 + 0.001f) {
            lx = (int)plot_x;
        }
        if (draw_t >= axis1 - 0.001f) {
            lx = (int)(plot_x + plot_w) - tw;
        }
        DrawText(label, lx, (int)bot + 4, 14, PUFF_WHITE);
        DrawLine((int)x, (int)bot, (int)x, (int)bot + 4, PUFF_WHITE);
        if (draw_t >= axis1 - 0.001f) {
            break;
        }
    }
    const char* axis = "Time";
    int axis_w = MeasureText(axis, 14);
    DrawText(axis, (int)(plot_x + plot_w * 0.5f) - axis_w / 2,
        (int)bot + 18, 14, PUFF_WHITE);
}

static void flow_draw_lane_keys(Drive* env, float panel_x, float top,
    float row_h, int lanes) {
    for (int lane = 0; lane < lanes; lane++) {
        float y = top + (float)lane * row_h;
        int ty = (int)(y + row_h * 0.5f) - 7;
        int chip = (int)panel_x + 8;
        DrawRectangle(chip, ty + 3, 10, 10, flow_lane_color(lane));
        DrawText(env->flow_lane_name[lane], chip + 14, ty, 15, PUFF_WHITE);
    }
}

// One heatmap. The y-axis segments are the lanes.
static void flow_diagram_draw(Drive* env, Client* client) {
    if (puf_record_enabled()) {
        return;
    }
    if (env->agent_initialization != AGENT_INIT_STRUCTURED) {
        return;
    }
    if (!env->flow_speed || env->flow_rows <= 0 || env->flow_nlanes <= 0) {
        return;
    }
    int times = env->flow_rows;
    int lanes = env->flow_nlanes;
    float now = 0.0f;
    float axis0 = 0.0f;
    float axis1 = 0.0f;
    flow_time_window(env, &now, &axis0, &axis1);
    float axis_span = axis1 - axis0;
    if (axis_span < SIM_DT) {
        axis_span = SIM_DT;
    }

    Rectangle panel = flow_panel_rect(client->width, client->height, lanes);
    float panel_x = panel.x;
    float panel_y = panel.y;
    float panel_w = panel.width;
    float panel_h = panel.height;
    float head = FLOW_PANEL_HEAD;
    float foot = FLOW_PANEL_FOOT;
    if (panel_w < 200.0f || panel_h < head + foot + 8.0f) {
        return;
    }
    DrawRectangle((int)panel_x, (int)panel_y, (int)panel_w, (int)panel_h,
        (Color){12, 14, 20, 255});

    // Wide enough for the "Segment" header. Numbers sit under that word.
    int name_w = flow_name_w(env, lanes);
    float bar_w = 12.0f;
    float plot_x = panel_x + 8.0f + (float)name_w + 14.0f;
    float plot_r = panel_x + panel_w - 8.0f - bar_w - 28.0f;
    float plot_w = plot_r - plot_x;
    if (plot_w < 40.0f) {
        return;
    }
    float top = panel_y + head;
    float bot = panel_y + panel_h - foot;
    float plot_h = bot - top;
    float row_h = plot_h / (float)lanes;

    flow_diagram_tex(client, times, lanes);
    if (client->flow_tex.id == 0 || !client->flow_pix) {
        return;
    }
    flow_diagram_splat(env, client);
    UpdateTexture(client->flow_tex, client->flow_pix);

    DrawText("Speed (mph)", (int)plot_x, (int)panel_y + 4, 18, PUFF_WHITE);
    DrawText("Segment", (int)panel_x + 8, (int)panel_y + 24, 14,
        (Color){210, 214, 220, 255});
    DrawText("Time step: 0.1 s", (int)plot_x, (int)panel_y + 24,
        14, PUFF_WHITE);
    const char* clock = TextFormat("%.1f s", now);
    int clock_w = MeasureText(clock, 18);
    DrawText(clock, (int)plot_r - clock_w, (int)panel_y + 4, 18, PUFF_WHITE);

    Rectangle src = {0.0f, 0.0f, (float)times, (float)lanes};
    Rectangle dst = {plot_x, top, plot_w, plot_h};
    DrawTexturePro(client->flow_tex, src, dst, (Vector2){0, 0}, 0, WHITE);
    Color frame = {70, 76, 88, 255};
    DrawRectangleLinesEx(dst, 1.0f, frame);
    float play_x = plot_x + ((now - axis0) / axis_span) * plot_w;
    for (int lane = 0; lane < lanes; lane++) {
        float y = top + (float)lane * row_h;
        if (lane > 0) {
            DrawLine((int)plot_x, (int)y, (int)(plot_x + plot_w), (int)y,
                frame);
        }
    }
    flow_draw_lane_keys(env, panel_x, top, row_h, lanes);
    DrawLine((int)play_x, (int)top, (int)play_x, (int)bot, WHITE);
    float first_top = top;
    float last_bot = bot;
    flow_draw_time_axis(plot_x, plot_w, last_bot, axis0, axis1);

    float bar_x = plot_r + 8.0f;
    int bars = (int)(last_bot - first_top);
    if (bars < 1) {
        bars = 1;
    }
    for (int i = 0; i < bars; i++) {
        float u = ((float)i + 0.5f) / (float)bars;
        float mph = FLOW_MPH_HI + (FLOW_MPH_LO - FLOW_MPH_HI) * u;
        DrawRectangle((int)bar_x, (int)first_top + i, (int)bar_w, 1,
            flow_mph_color(mph));
    }
    for (int mph = 80; mph >= 10; mph -= 10) {
        float u = ((float)mph - FLOW_MPH_HI) / (FLOW_MPH_LO - FLOW_MPH_HI);
        int ly = (int)(first_top + u * (last_bot - first_top)) - 6;
        if (ly < (int)first_top) {
            ly = (int)first_top;
        }
        if (ly > (int)last_bot - 12) {
            ly = (int)last_bot - 12;
        }
        const char* label = TextFormat("%d", mph);
        DrawText(label, (int)bar_x + (int)bar_w + 3, ly, 12, PUFF_WHITE);
    }
}

// Bars beside the speed graph. Red is a crash, yellow is off-road.
static void flow_hit_draw(Drive* env, Client* client) {
    if (puf_record_enabled()) {
        return;
    }
    if (env->agent_initialization != AGENT_INIT_STRUCTURED) {
        return;
    }
    if (!env->flow_speed || env->flow_rows <= 0 || env->flow_nlanes <= 0) {
        return;
    }
    int times = env->flow_rows;
    int lanes = env->flow_nlanes;
    float now = 0.0f;
    float axis0 = 0.0f;
    float axis1 = 0.0f;
    flow_time_window(env, &now, &axis0, &axis1);
    float axis_span = axis1 - axis0;
    if (axis_span < SIM_DT) {
        axis_span = SIM_DT;
    }
    Rectangle panel = flow_hit_panel(client->width, client->height, lanes);
    float panel_x = panel.x;
    float panel_y = panel.y;
    float panel_w = panel.width;
    float panel_h = panel.height;
    float head = FLOW_PANEL_HEAD;
    float foot = FLOW_PANEL_FOOT;
    if (panel_w < 180.0f || panel_h < head + foot + 8.0f) {
        return;
    }
    int name_w = flow_name_w(env, lanes);
    float plot_x = panel_x + 8.0f + (float)name_w + 14.0f;
    float plot_r = panel_x + panel_w - 8.0f;
    float plot_w = plot_r - plot_x;
    if (plot_w < 40.0f) {
        return;
    }
    float top = panel_y + head;
    float bot = panel_y + panel_h - foot;
    float plot_h = bot - top;
    float row_h = plot_h / (float)lanes;
    DrawRectangle((int)panel_x, (int)panel_y, (int)panel_w, (int)panel_h,
        (Color){12, 14, 20, 255});
    DrawText("Collisions", (int)plot_x, (int)panel_y + 4, 18, PUFF_WHITE);
    DrawText("Segment", (int)panel_x + 8, (int)panel_y + 24, 14,
        (Color){210, 214, 220, 255});
    const char* clock = TextFormat("%.1f s", now);
    int clock_w = MeasureText(clock, 18);
    int title_w = MeasureText("Collisions", 18);
    if (plot_x + (float)title_w + 8.0f < plot_r - (float)clock_w) {
        DrawText(clock, (int)plot_r - clock_w, (int)panel_y + 4,
            18, PUFF_WHITE);
    }
    Color frame = {70, 76, 88, 255};
    DrawRectangle((int)plot_x, (int)top, (int)plot_w, (int)plot_h,
        (Color){20, 24, 32, 255});
    for (int lane = 1; lane < lanes; lane++) {
        int y = (int)(top + (float)lane * row_h);
        DrawLine((int)plot_x, y, (int)(plot_x + plot_w), y, frame);
    }
    Rectangle plot = {plot_x, top, plot_w, plot_h};
    DrawRectangleLinesEx(plot, 1.0f, frame);
    flow_draw_lane_keys(env, panel_x, top, row_h, lanes);
    float play_x = plot_x + ((now - axis0) / axis_span) * plot_w;
    DrawLine((int)play_x, (int)top, (int)play_x, (int)bot, WHITE);
    flow_draw_time_axis(plot_x, plot_w, bot, axis0, axis1);

    int origin = 0;
    if (env->sim_infinite && env->timestep >= times) {
        origin = env->timestep - times + 1;
    }
    int n = env->flow_hit_n;
    int start = 0;
    if (n >= FLOW_HIT_MAX) {
        n = FLOW_HIT_MAX;
        start = env->flow_hit_i;
    }
    Color crash = {220, 45, 52, 255};
    Color offroad = {245, 208, 40, 255};
    int bh = 8;
    if ((float)bh > row_h * 0.4f) {
        bh = (int)(row_h * 0.4f);
    }
    if (bh < 4) {
        bh = 4;
    }
    int bw = 3;
    for (int k = 0; k < n; k++) {
        int idx = start + k;
        if (idx >= FLOW_HIT_MAX) {
            idx -= FLOW_HIT_MAX;
        }
        int tstep = env->flow_hit_t[idx];
        int lane = env->flow_hit_lane[idx];
        int col = tstep - origin;
        if (lane < 0 || lane >= lanes || col < 0 || col >= times) {
            continue;
        }
        int slot = 0;
        for (int j = 0; j < k; j++) {
            int jdx = start + j;
            if (jdx >= FLOW_HIT_MAX) {
                jdx -= FLOW_HIT_MAX;
            }
            if (env->flow_hit_t[jdx] == tstep
                && env->flow_hit_lane[jdx] == lane) {
                slot++;
            }
        }
        float row_top = top + (float)lane * row_h;
        float y = row_top + (row_h - (float)bh) * 0.5f;
        y += (float)((slot % 3) - 1) * 2.0f;
        if (y < row_top + 1.0f) {
            y = row_top + 1.0f;
        }
        float y_max = row_top + row_h - (float)bh - 1.0f;
        if (y > y_max) {
            y = y_max;
        }
        float cx = plot_x + ((float)col + 0.5f) / (float)times * plot_w;
        Color mark = crash;
        if (env->flow_hit_kind[idx] == OFFROAD) {
            mark = offroad;
        }
        DrawRectangle((int)(cx - (float)bw * 0.5f), (int)y, bw, bh, mark);
    }
}

// Lane centerlines. A quiet tint of the diagram chip, not a paint stripe.
static Vector2 flow_lane_screen[FLOW_LANE_MAX];
static int flow_lane_screen_ok[FLOW_LANE_MAX];

static void flow_lane_ribbon(float ax, float ay, float bx, float by, Color color) {
    float dx = bx - ax;
    float dy = by - ay;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 1.0e-4f) {
        return;
    }
    float half = 0.45f;
    float px = -dy / len * half;
    float py = dx / len * half;
    float z = 1.25f;
    Vector3 a = {ax + px, ay + py, z};
    Vector3 b = {ax - px, ay - py, z};
    Vector3 c = {bx - px, by - py, z};
    Vector3 d = {bx + px, by + py, z};
    DrawTriangle3D(a, b, c, color);
    DrawTriangle3D(a, c, b, color);
    DrawTriangle3D(a, c, d, color);
    DrawTriangle3D(a, d, c, color);
}

static void flow_draw_lanes(Drive* env, Client* client) {
    int n = env->flow_nlanes;
    if (n > FLOW_LANE_MAX) {
        n = FLOW_LANE_MAX;
    }
    for (int lane = 0; lane < FLOW_LANE_MAX; lane++) {
        flow_lane_screen_ok[lane] = 0;
    }
    if (puf_record_enabled() || !env->flow_segs || n <= 0) {
        return;
    }
    Rectangle diagram = flow_panel_rect(client->width, client->height, n);
    Rectangle col = sim_col_rect(env);
    float view_x = client->width * 0.40f;
    float view_y = diagram.y - 80.0f;
    if (view_y < 190.0f) {
        view_y = 190.0f;
    }
    float x_max = col.x - 12.0f;
    Rectangle eye = sim_obs_rect(client->width);
    for (int lane = 0; lane < n; lane++) {
        Color color = flow_lane_road_color(lane);
        int beg = env->flow_lane_beg[lane];
        int count = env->flow_lane_n[lane];
        float best = 1.0e30f;
        for (int s = 0; s < count; s++) {
            FlowSeg* g = &env->flow_segs[beg + s];
            flow_lane_ribbon(g->ax, g->ay, g->bx, g->by, color);
            float mx = 0.5f * (g->ax + g->bx);
            float my = 0.5f * (g->ay + g->by);
            Vector2 sp = GetWorldToScreen((Vector3){mx, my, 2.0f}, client->camera);
            if (sp.x < 8.0f || sp.y < 170.0f || sp.x > x_max) {
                continue;
            }
            if (sp.y > client->height - 12.0f) {
                continue;
            }
            if (sp.x > diagram.x - 8.0f && sp.y > diagram.y - 12.0f) {
                continue;
            }
            if (sp.x >= eye.x - 6.0f && sp.x <= eye.x + eye.width + 6.0f
                && sp.y >= eye.y - 4.0f
                && sp.y <= eye.y + eye.height + 4.0f) {
                continue;
            }
            float ddx = sp.x - view_x;
            float ddy = sp.y - view_y;
            float dist = ddx * ddx + ddy * ddy;
            if (dist < best) {
                best = dist;
                flow_lane_screen[lane] = sp;
                flow_lane_screen_ok[lane] = 1;
            }
        }
    }
}

static void flow_draw_lane_names(Drive* env, Client* client) {
    if (puf_record_enabled() || env->sim_follow || !env->flow_segs) {
        return;
    }
    int n = env->flow_nlanes;
    if (n > FLOW_LANE_MAX) {
        n = FLOW_LANE_MAX;
    }
    int x[FLOW_LANE_MAX];
    int y[FLOW_LANE_MAX];
    int tw[FLOW_LANE_MAX];
    int which[FLOW_LANE_MAX];
    int nuse = 0;
    for (int lane = 0; lane < n; lane++) {
        if (!flow_lane_screen_ok[lane]) {
            continue;
        }
        which[nuse] = lane;
        x[nuse] = (int)flow_lane_screen[lane].x;
        y[nuse] = (int)flow_lane_screen[lane].y;
        tw[nuse] = MeasureText(env->flow_lane_name[lane], 16);
        nuse++;
    }
    for (int pass = 0; pass < nuse; pass++) {
        for (int i = 0; i < nuse; i++) {
            for (int j = 0; j < i; j++) {
                int x_hit = x[i] < x[j] + tw[j] + 8 && x[j] < x[i] + tw[i] + 8;
                int dy = y[i] - y[j];
                if (dy < 0) {
                    dy = -dy;
                }
                if (!x_hit || dy >= 18) {
                    continue;
                }
                if (y[i] >= y[j]) {
                    y[i] = y[j] + 18;
                } else {
                    y[j] = y[i] + 18;
                }
            }
        }
    }
    Rectangle diagram = flow_panel_rect(client->width, client->height, n);
    Rectangle hits = flow_hit_panel(client->width, client->height, n);
    int y_floor = (int)client->height - 8;
    for (int i = 0; i < nuse; i++) {
        int y_lim = y_floor;
        if (x[i] < (int)(diagram.x + diagram.width) + 8) {
            int cap = (int)diagram.y - 20;
            if (cap < y_lim) {
                y_lim = cap;
            }
        }
        if (hits.width >= 180.0f && x[i] + tw[i] > (int)hits.x - 4
            && x[i] < (int)(hits.x + hits.width) + 4) {
            int cap = (int)hits.y - 20;
            if (cap < y_lim) {
                y_lim = cap;
            }
        }
        if (y[i] > y_lim) {
            y[i] = y_lim;
        }
        Rectangle eye = sim_obs_rect(client->width);
        if (x[i] + tw[i] > (int)eye.x - 4
            && x[i] < (int)(eye.x + eye.width) + 4) {
            int cover_top = (int)eye.y - 20;
            int cover_bot = (int)(eye.y + eye.height) + 2;
            if (y[i] > cover_top && y[i] < cover_bot) {
                y[i] = cover_bot;
            }
        }
        const char* name = env->flow_lane_name[which[i]];
        Color color = flow_lane_color(which[i]);
        DrawRectangle(x[i] - 2, y[i] - 1, tw[i] + 6, 18, (Color){12, 14, 20, 220});
        DrawText(name, x[i], y[i], 16, color);
    }
}

void puf_render(Drive* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    drive_human_controls(env);
    Client* client = env->client;
    if (puf_record_enabled()) {
        drive_record_camera(env, client);
    }
    BeginDrawing();
    ClearBackground(ROAD_COLOR);
    if (!puf_record_enabled()) {
        handle_camera_controls(env);
        sim_follow_camera(env);
    }
    BeginMode3D(client->camera);

    if (env->map_corners) {
        DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[1], 0}, PUFF_CYAN);
        DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[0], env->map_corners[3], 0}, PUFF_CYAN);
        DrawLine3D((Vector3){env->map_corners[2], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, PUFF_CYAN);
        DrawLine3D((Vector3){env->map_corners[0], env->map_corners[3], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, PUFF_CYAN);
    }

    for (int i = 0; i < env->num_entities; i++) {
        // Draw vehicles
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN) {
            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            if (!is_active_agent && !is_static_agent) {
                continue;
            }

            Vector3 position = {env->entities[i].x, env->entities[i].y, 1};
            float heading = env->entities[i].heading;
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            rlPushMatrix();
            rlTranslatef(position.x, position.y, position.z);
            rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);

            Model car_model = client->cars[5];
            if (is_active_agent) {
                car_model = client->cars[client->car_assignments[i % MAX_AGENTS]];
            }
            if (agent_index == env->human_agent_idx) {
                // Human-controlled agent uses default model
            }
            if (is_active_agent && env->entities[i].collision_state > NO_COLLISION) {
                car_model = client->cars[0];
            }

            // The marker floats 8 m above the car. Skip it in the chase shot
            // so it does not fill the zoomed frame.
            if (!puf_record_enabled()
                && agent_index == env->human_agent_idx
                && !env->entities[agent_index].reached_goal) {
                draw_agent_obs(env, agent_index);
            }

            BoundingBox bounds = GetModelBoundingBox(car_model);
            Vector3 model_size = {
                bounds.max.x - bounds.min.x,
                bounds.max.y - bounds.min.y,
                bounds.max.z - bounds.min.z
            };
            Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};
            DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
            rlPopMatrix();

            // Draw collision box
            float cos_h = env->entities[i].heading_x;
            float sin_h = env->entities[i].heading_y;
            float hl = env->entities[i].length * 0.5f;
            float hw = env->entities[i].width * 0.5f;
            Vector3 corners[4] = {
                {position.x + (hl * cos_h - hw * sin_h), position.y + (hl * sin_h + hw * cos_h), position.z},
                {position.x + (hl * cos_h + hw * sin_h), position.y + (hl * sin_h - hw * cos_h), position.z},
                {position.x + (-hl * cos_h - hw * sin_h), position.y + (-hl * sin_h + hw * cos_h), position.z},
                {position.x + (-hl * cos_h + hw * sin_h), position.y + (-hl * sin_h - hw * cos_h), position.z}
            };
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], PURPLE);
            }

            // FPV camera
            if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index) {
                if (env->entities[agent_index].reached_goal) {
                    env->human_agent_idx = rand_r(&env->rng) % env->active_agent_count;
                }
                client->camera.position = (Vector3){
                    position.x - 25.0f * cosf(heading),
                    position.y - 25.0f * sinf(heading),
                    position.z + 15
                };
                client->camera.target = (Vector3){
                    position.x + 40.0f * cosf(heading),
                    position.y + 40.0f * sinf(heading),
                    position.z - 5.0f
                };
                client->camera.up = (Vector3){0, 0, 1};
            }
            if (IsKeyReleased(KEY_SPACE)) {
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }

            if (!is_active_agent || env->entities[i].valid == 0) continue;
            if (!IsKeyDown(KEY_LEFT_CONTROL)) {
                DrawSphere((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, 1}, 0.5f, DARKGREEN);
            }
        }

        // Draw road elements
        if (env->entities[i].type < ROAD_LANE || env->entities[i].type > ROAD_EDGE) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            if (env->entities[i].type != ROAD_EDGE) continue;
            if (!IsKeyDown(KEY_LEFT_CONTROL)) {
                draw_road_edge(env,
                    env->entities[i].traj_x[j], env->entities[i].traj_y[j],
                    env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1]);
            }
        }
    }

    // The 5 m grid is a debug overlay. Skip it while recording so a full map
    // does not spend the frame drawing thousands of wires into the mp4.
    if (!puf_record_enabled()) {
        float grid_start_x = env->map_corners ? env->map_corners[0] : 0.0f;
        float grid_start_y = env->map_corners ? env->map_corners[1] : 0.0f;
        for (int i = 0; i < env->grid_cols; i++) {
            for (int j = 0; j < env->grid_rows; j++) {
                float x = grid_start_x + i * GRID_CELL_SIZE;
                float y = grid_start_y + j * GRID_CELL_SIZE;
                DrawCubeWires(
                    (Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 1},
                    GRID_CELL_SIZE, GRID_CELL_SIZE, 0.1f, PUFF_BACKGROUND2);
            }
        }
    }
    flow_draw_lanes(env, client);
    EndMode3D();
    flow_draw_lane_names(env, client);

    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)",
        client->camera.position.x, client->camera.position.y, client->camera.position.z), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)",
        client->camera.target.x, client->camera.target.y, client->camera.target.z), 10, 30, 20, PUFF_WHITE);
    const char* paused_mark = "";
    if (!puf_record_enabled() && env->sim_paused) {
        paused_mark = "   paused";
    }
    DrawText(TextFormat("Time: %.1f s%s", env->timestep * SIM_DT, paused_mark),
        10, 50, 20, PUFF_WHITE);
    if (!puf_record_enabled() && !env->checkpoint_ok) {
        DrawText("Checkpoint cannot be loaded, showing random behavior",
            10, 190, 20, (Color){240, 196, 120, 255});
    }
    int human_idx = -1;
    int accel = 0;
    int steer = 0;
    if (env->active_agent_count > 0 && env->active_agent_indices
        && env->human_agent_idx >= 0
        && env->human_agent_idx < env->active_agent_count) {
        human_idx = env->active_agent_indices[env->human_agent_idx];
        float* atn = env->agents[env->human_agent_idx].actions;
        if (atn) {
            accel = (int)atn[0];
            steer = (int)atn[1];
        }
    }
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);
    if (!puf_record_enabled()) {
        sim_transport_draw(env);
    }
    DrawText(TextFormat("Acceleration: %d", accel), 10, 110, 20, PUFF_WHITE);
    DrawText(TextFormat("Steering: %d", steer), 10, 130, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Rows: %d", env->grid_rows), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_cols), 10, 170, 20, PUFF_WHITE);
    flow_diagram_draw(env, client);
    flow_hit_draw(env, client);
    EndDrawing();
    if (puf_record_enabled()) {
        puf_record_capture((int)client->width, (int)client->height);
    }
    puf_web_vsync();
    // TEMP diag: first frames of the focused car. Removed after the check.
    if (!puf_record_enabled() && env->active_agent_count > 0) {
        static int diag_frame = 0;
        int slot = env->human_agent_idx;
        if (slot < 0 || slot >= env->active_agent_count) slot = 0;
        int idx = env->active_agent_indices[slot];
        Entity* e = &env->entities[idx];
        float lon = e->vx * e->heading_x + e->vy * e->heading_y;
        float lat = -e->vx * e->heading_y + e->vy * e->heading_x;
        float gx = e->goal_position_x - e->x;
        float gy = e->goal_position_y - e->y;
        float gf = gx * e->heading_x + gy * e->heading_y;
        float gl = -gx * e->heading_y + gy * e->heading_x;
        if (diag_frame < 8 || diag_frame == 20 || diag_frame == 40) {
            fprintf(stderr,
                "DIAG f=%d t=%d slot=%d spd=%.2f lon=%.2f lat=%.2f "
                "gf=%.1f gl=%.1f hdg=%.3f vx=%.2f vy=%.2f\n",
                diag_frame, env->timestep, slot,
                sqrtf(e->vx * e->vx + e->vy * e->vy), lon, lat,
                gf, gl, e->heading, e->vx, e->vy);
            TakeScreenshot(TextFormat("/tmp/obs_frame_%d.png", diag_frame));
        }
        diag_frame++;
        if (diag_frame > 45) {
            exit(0);
        }
    }
}

void close_client(Client* client) {
    if (client->flow_tex.id) {
        UnloadTexture(client->flow_tex);
    }
    free(client->flow_pix);
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}

#define MAP_RESOURCE_DIR "resources/drive"

static int drive_map_exists(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

// [env] data_dir is the dataset root (json/ and binaries/). Training reads binaries/.
// The web demo only preloads resources/drive/map_010.bin and map_942.bin.
static void drive_bin_dir(Dict* kwargs, char* out, size_t out_sz) {
    const char* data_dir = dict_get_str(kwargs, "data_dir");
    snprintf(out, out_sz, "%s/binaries", data_dir);
}

static void resolve_map_path(const char* bin_dir, int map_id, char* out, size_t out_sz) {
    snprintf(out, out_sz, "%s/map_%03d.bin", bin_dir, map_id);
    if (drive_map_exists(out)) return;
    snprintf(out, out_sz, "%s/map_%03d.bin", MAP_RESOURCE_DIR, map_id);
    if (drive_map_exists(out)) return;
    const char* fallbacks[] = {
        MAP_RESOURCE_DIR "/map_010.bin",
        MAP_RESOURCE_DIR "/map_942.bin",
    };
    for (int i = 0; i < 2; i++) {
        if (drive_map_exists(fallbacks[i])) {
            snprintf(out, out_sz, "%s", fallbacks[i]);
            return;
        }
    }
}

static int peek_map_agent_count(const char* filename, int max_agents,
        int agent_initialization, Dict* cfg) {
    Drive temp = {0};
    temp.map_name = (char*)filename;
    temp.max_agents = max_agents > 0 ? max_agents : MAX_AGENTS;
    temp.agent_initialization = agent_initialization;
    temp.cfg = cfg;
    temp.entities = load_map_binary(filename, &temp);
    if (!temp.entities) return 0;
    if (temp.agent_initialization == AGENT_INIT_STRUCTURED) {
        spawn_flow_vehicles(&temp);
    }
    set_means(&temp);
    set_active_agents(&temp);
    int n = temp.active_agent_count;
    if (n > DRIVE_AGENT_MAX) n = DRIVE_AGENT_MAX;
    puf_close(&temp);
    return n;
}

void puf_init(Env* env, Dict* kwargs) {
    env->human_agent_idx = dict_get(kwargs, "human_agent_idx");
    env->reward_vehicle_collision = dict_get(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = dict_get(kwargs, "reward_offroad_collision");
    env->reward_goal_post_respawn = dict_get(kwargs, "reward_goal_post_respawn");
    env->reward_vehicle_collision_post_respawn = dict_get(kwargs, "reward_vehicle_collision_post_respawn");
    env->reward_jerk = dict_get(kwargs, "reward_jerk");
    env->reward_speed = dict_get(kwargs, "reward_speed");
    env->agent_initialization = (int)dict_get(kwargs, "agent_initialization");
    env->cfg = kwargs;
    int map_id = dict_get(kwargs, "map_id");
    char bin_dir[480];
    char map_file[512];
    drive_bin_dir(kwargs, bin_dir, sizeof(bin_dir));
    resolve_map_path(bin_dir, map_id, map_file, sizeof(map_file));
    env->map_name = strdup(map_file);
    env->max_agents = MAX_AGENTS;
    init(env);
    env->num_agents = env->active_agent_count;
    if (env->num_agents > DRIVE_AGENT_MAX) env->num_agents = DRIVE_AGENT_MAX;
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

// 4.0 binding.c: scan maps, assign distinct map_ids, pack each buffer to
// exactly agents_per_buffer (full maps, then 1-agent leftovers).
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = dict_get(vec_kwargs, "total_agents");
    int num_buffers = dict_get(vec_kwargs, "num_buffers");
    int num_maps = dict_get(env_kwargs, "num_maps");
    int agents_per_buffer = total_agents / num_buffers;

    float reward_vehicle_collision = dict_get(env_kwargs, "reward_vehicle_collision");
    float reward_offroad_collision = dict_get(env_kwargs, "reward_offroad_collision");
    float reward_goal_post_respawn = dict_get(env_kwargs, "reward_goal_post_respawn");
    float reward_vehicle_collision_post_respawn = dict_get(env_kwargs, "reward_vehicle_collision_post_respawn");
    float reward_jerk = dict_get(env_kwargs, "reward_jerk");
    float reward_speed = dict_get(env_kwargs, "reward_speed");
    int human_agent_idx = dict_get(env_kwargs, "human_agent_idx");
    int agent_initialization = (int)dict_get(env_kwargs, "agent_initialization");

    char bin_dir[480];
    drive_bin_dir(env_kwargs, bin_dir, sizeof(bin_dir));

    char first_map[512];
    snprintf(first_map, sizeof(first_map), "%s/map_%03d.bin", bin_dir, 0);
    FILE* test_fp = fopen(first_map, "rb");
    if (!test_fp) {
        fprintf(stderr, "ERROR: Cannot find map files at %s/\n", bin_dir);
        fprintf(stderr, "Download WOMD JSON and run ocean/drive/dataset.py (see dataset.py header).\n");
        exit(1);
    }
    fclose(test_fp);

    int* agents_per_map = (int*)calloc(num_maps, sizeof(int));
    int* valid_map_ids = (int*)calloc(num_maps, sizeof(int));
    int num_valid_maps = 0;
    for (int m = 0; m < num_maps; m++) {
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", bin_dir, m);
        agents_per_map[m] = peek_map_agent_count(map_file, MAX_AGENTS,
            agent_initialization, env_kwargs);
        if (agents_per_map[m] > 0) {
            valid_map_ids[num_valid_maps++] = m;
        }
    }
    printf("Scanned %d maps from %s/, %d valid\n", num_maps, bin_dir, num_valid_maps);

    if (num_valid_maps == 0) {
        fprintf(stderr, "ERROR: No valid maps found in %s/\n", bin_dir);
        free(agents_per_map);
        free(valid_map_ids);
        exit(1);
    }

    int max_envs = agents_per_buffer * num_buffers;
    int* env_map_ids = (int*)malloc(max_envs * sizeof(int));
    int* env_max_agents = (int*)malloc(max_envs * sizeof(int));
    int total_envs = 0;
    int cursor = 0;

    for (int b = 0; b < num_buffers; b++) {
        buffer_env_starts[b] = total_envs;
        int buffer_agents = 0;
        while (buffer_agents < agents_per_buffer) {
            int m = valid_map_ids[cursor % num_valid_maps];
            int cap = agents_per_map[m];
            int remaining = agents_per_buffer - buffer_agents;
            if (cap <= remaining) {
                env_map_ids[total_envs] = m;
                env_max_agents[total_envs] = cap;
                buffer_agents += cap;
                total_envs++;
                cursor++;
            } else {
                while (buffer_agents < agents_per_buffer) {
                    int mm = valid_map_ids[cursor % num_valid_maps];
                    env_map_ids[total_envs] = mm;
                    env_max_agents[total_envs] = 1;
                    buffer_agents++;
                    total_envs++;
                    cursor++;
                }
            }
        }
        buffer_env_counts[b] = total_envs - buffer_env_starts[b];
    }

    printf("total envs: %d (%d maps cycled)\n", total_envs, cursor);

    Env* envs = (Env*)calloc(total_envs, sizeof(Env));
    for (int i = 0; i < total_envs; i++) {
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", bin_dir, env_map_ids[i]);
        Env* env = &envs[i];
        env->rng = i;
        env->agent_initialization = agent_initialization;
        env->cfg = env_kwargs;
        env->map_name = strdup(map_file);
        env->human_agent_idx = human_agent_idx;
        env->reward_vehicle_collision = reward_vehicle_collision;
        env->reward_offroad_collision = reward_offroad_collision;
        env->reward_goal_post_respawn = reward_goal_post_respawn;
        env->reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn;
        env->reward_jerk = reward_jerk;
        env->reward_speed = reward_speed;
        env->max_agents = env_max_agents[i];
        init(env);
        env->num_agents = env->active_agent_count;
        if (env->num_agents > env->max_agents) env->num_agents = env->max_agents;
        if (env->num_agents > DRIVE_AGENT_MAX) env->num_agents = DRIVE_AGENT_MAX;
        for (int a = 0; a < env->num_agents; a++) {
            env->agents[a].policy = 0;
            env->agents[a].action_mask = NULL;
        }
    }

    free(env_map_ids);
    free(env_max_agents);
    free(agents_per_map);
    free(valid_map_ids);

    printf("Created %d envs, target %d agents\n", total_envs, total_agents);

    *num_envs_out = total_envs;
    return envs;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "offroad_rate", log->offroad_rate);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "dnf_rate", log->dnf_rate);
    dict_set(out, "n", log->n);
    dict_set(out, "completion_rate", log->completion_rate);
    dict_set(out, "clean_collision_rate", log->clean_collision_rate);
}

