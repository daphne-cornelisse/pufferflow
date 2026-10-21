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
    // Point count of a road polyline. Cars stay at 0. Reset samples
    // their position and goal from the route.
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
    // Trip origin for the pass-the-goal test. Reset writes the sampled
    // start. Inference injections write the entrance.
    float origin_x;
    float origin_y;
    float prev_accel;
    float prev_steer;
    // Personal max speed in m/s. Below 1 the bicycle clip stays at MAX_SPEED.
    float max_speed;
};

void free_entity(Entity* entity) {
    free(entity->pts_x);
    free(entity->pts_y);
    entity->pts_x = NULL;
    entity->pts_y = NULL;
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
    // Cars on this map. Entity i is agent i. Roads follow the cars.
    int active_agent_count;
    int human_agent_idx;
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
    float world_mean_x;
    float world_mean_y;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float reward_jerk;
    float reward_speed;
    unsigned int rng;
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
    // Overhead camera stays on human_agent_idx. Off restores the map view.
    int sim_follow;
    // 1: cars keep overlapping. 0: bump them apart. Headless stays on bump.
    int sim_through;
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
    // 0 train, 1 inference. Set by the entry point before the first reset.
    int drive_mode;
    // Mainline veh/h. The ramp keeps the ini ratio. Live in the viewer.
    int sim_vph_req;
    int sim_vph_on;
    int sim_vph_ready;
    int sim_vph_drag;
    // Cars that left the network this episode, and cars inserted.
    // add_log folds the count into log.n. Their score is already in log.
    int stream_n;
    int stream_in;
    int stream_banner;
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
        Entity* e = &env->entities[i];
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
    if (env->drive_mode == DRIVE_MODE_INFERENCE
        && (env->stream_in > 0 || env->stream_n > 0)) {
        fprintf(stderr, "inference stream: %d entered, %d left\n",
            env->stream_in, env->stream_n);
    }
    env->log.n += (float)env->stream_n;
    env->stream_n = 0;
    env->stream_in = 0;
}

// Logged tracks sit in front of the roadgraph. Cars are placed from the
// network, so those tracks are skipped and only the polylines are kept.
static int skip_logged_object(FILE* file) {
    int type = 0;
    int size = 0;
    if (fread(&type, sizeof(int), 1, file) != 1) return -1;
    if (fread(&size, sizeof(int), 1, file) != 1) return -1;
    if (size < 0) return -1;
    (void)type;
    long bytes = (long)size * 7 * (long)sizeof(float);
    bytes += (long)size * (long)sizeof(int);
    bytes += 6 * (long)sizeof(float) + (long)sizeof(int);
    if (fseek(file, bytes, SEEK_CUR) != 0) return -1;
    return 0;
}

// Map loading. Returns the roadgraph. num_objects stays 0 until cars spawn.
Entity* load_map_binary(const char* filename, Drive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;

    int logged = 0;
    if (fread(&logged, sizeof(int), 1, file) != 1
        || fread(&env->num_roads, sizeof(int), 1, file) != 1
        || env->num_roads < 0 || logged < 0) {
        fclose(file);
        return NULL;
    }
    for (int i = 0; i < logged; i++) {
        if (skip_logged_object(file) != 0) {
            fclose(file);
            return NULL;
        }
    }
    env->num_objects = 0;
    env->num_entities = env->num_roads;
    Entity* entities = (Entity*)calloc((size_t)env->num_roads, sizeof(Entity));
    if (!entities && env->num_roads > 0) {
        fclose(file);
        return NULL;
    }
    for (int i = 0; i < env->num_roads; i++) {
        int size = 0;
        if (fread(&entities[i].type, sizeof(int), 1, file) != 1
            || fread(&size, sizeof(int), 1, file) != 1
            || size < 1) {
            fclose(file);
            free(entities);
            return NULL;
        }
        entities[i].array_size = size;
        entities[i].pts_x = (float*)malloc((size_t)size * sizeof(float));
        entities[i].pts_y = (float*)malloc((size_t)size * sizeof(float));
        float* z = (float*)malloc((size_t)size * sizeof(float));
        if (!entities[i].pts_x || !entities[i].pts_y || !z) {
            free(z);
            fclose(file);
            free(entities);
            return NULL;
        }
        if (fread(entities[i].pts_x, sizeof(float), size, file) != (size_t)size
            || fread(entities[i].pts_y, sizeof(float), size, file) != (size_t)size
            || fread(z, sizeof(float), size, file) != (size_t)size) {
            free(z);
            fclose(file);
            free(entities);
            return NULL;
        }
        free(z);
        int mark = 0;
        if (fread(&entities[i].width, sizeof(float), 1, file) != 1
            || fread(&entities[i].length, sizeof(float), 1, file) != 1
            || fread(&entities[i].height, sizeof(float), 1, file) != 1
            || fread(&entities[i].goal_position_x, sizeof(float), 1, file) != 1
            || fread(&entities[i].goal_position_y, sizeof(float), 1, file) != 1
            || fread(&entities[i].goal_position_z, sizeof(float), 1, file) != 1
            || fread(&mark, sizeof(int), 1, file) != 1) {
            fclose(file);
            free(entities);
            return NULL;
        }
        (void)mark;
    }
    fclose(file);
    return entities;
}

// Roads keep their first point. Cars are placed from the route at reset.
void set_start_position(Drive* env) {
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type < ROAD_LANE || !e->pts_x || e->array_size < 1) continue;
        e->x = e->pts_x[0];
        e->y = e->pts_y[0];
        e->z = 0.0f;
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
                if (e->pts_x[j] == INVALID_POSITION) continue;
                if (e->pts_y[j] == INVALID_POSITION) continue;
                if (!first_valid_point) {
                    top_left_x = bottom_right_x = e->pts_x[j];
                    top_left_y = bottom_right_y = e->pts_y[j];
                    first_valid_point = 1;
                    continue;
                }
                if (e->pts_x[j] < top_left_x) top_left_x = e->pts_x[j];
                if (e->pts_x[j] > bottom_right_x) bottom_right_x = e->pts_x[j];
                if (e->pts_y[j] < top_left_y) top_left_y = e->pts_y[j];
                if (e->pts_y[j] > bottom_right_y) bottom_right_y = e->pts_y[j];
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
                float x_center = (env->entities[i].pts_x[j] + env->entities[i].pts_x[j + 1]) / 2;
                float y_center = (env->entities[i].pts_y[j] + env->entities[i].pts_y[j + 1]) / 2;
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
        Entity* e = &env->entities[i];
        if (e->type < ROAD_LANE || !e->pts_x) continue;
        for (int j = 0; j < e->array_size; j++) {
            if (e->pts_x[j] == INVALID_POSITION) continue;
            if (e->pts_y[j] == INVALID_POSITION) continue;
            point_count++;
            mean_x += (e->pts_x[j] - mean_x) / point_count;
            mean_y += (e->pts_y[j] - mean_y) / point_count;
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;

    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type < ROAD_LANE || !e->pts_x) continue;
        for (int j = 0; j < e->array_size; j++) {
            if (e->pts_x[j] == INVALID_POSITION) continue;
            e->pts_x[j] -= mean_x;
            e->pts_y[j] -= mean_y;
        }
    }
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
            float ax = edge->pts_x[g];
            float ay = edge->pts_y[g];
            float bx = edge->pts_x[g + 1];
            float by = edge->pts_y[g + 1];
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
    int n = env->active_agent_count;
    if (n > DRIVE_AGENT_MAX) {
        n = DRIVE_AGENT_MAX;
    }
    float prev_x[DRIVE_AGENT_MAX];
    float prev_y[DRIVE_AGENT_MAX];
    for (int i = 0; i < n; i++) {
        Entity* e = &env->entities[i];
        // Respawn this step teleported. vx is the spawn speed, not the
        // displacement, so the sweep uses the pose already written.
        int stepped = e->respawn_timestep != env->timestep;
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
            for (int j = i + 1; j < n; j++) {
                bump_pair(env,
                    &env->entities[i], &env->entities[j],
                    prev_x[i], prev_y[i], prev_x[j], prev_y[j],
                    pass == 0);
            }
            bump_curb(env, &env->entities[i], prev_x[i], prev_y[i]);
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
        float start[2] = {entity->pts_x[geometry_idx], entity->pts_y[geometry_idx]};
        float end[2] = {entity->pts_x[geometry_idx + 1], entity->pts_y[geometry_idx + 1]};
        for (int k = 0; k < 4; k++) {
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], start, end)) {
                collided = OFFROAD;
                break;
            }
        }
        if (collided == OFFROAD) break;
    }

    // Check vehicle-vehicle collisions. Agent i is entity i.
    int nscan = env->active_agent_count;
    if (nscan > DRIVE_AGENT_MAX) nscan = DRIVE_AGENT_MAX;
    for (int index = 0; index < nscan; index++) {
        if (index == agent_idx) continue;
        Entity* entity = &env->entities[index];
        if (entity->x == INVALID_POSITION || entity->y == INVALID_POSITION) {
            continue;
        }
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
            float ax = e->pts_x[j];
            float ay = e->pts_y[j];
            float bx = e->pts_x[j + 1];
            float by = e->pts_y[j + 1];
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
    for (int idx = 0; idx < env->active_agent_count; idx++) {
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
    if (row < 0 || row >= env->flow_rows) {
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
        Entity* e = &env->entities[i];
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
        int self = i;
        Entity* e = &env->entities[self];
        int other = -1;
        if (e->collision_state == VEHICLE_COLLISION) {
            other = flow_hit_partner(env, self);
            if (other >= 0 && other < self && rising[other]) {
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
}

static void drive_load_world(Drive* env) {
    env->entities = load_map_binary(env->map_name, env);
    if (!env->entities) {
        fprintf(stderr, "ERROR: Cannot load map %s\n",
            env->map_name ? env->map_name : "(null)");
        exit(1);
    }
    // Cars are placed on the roadgraph in network coordinates. set_means
    // then centers roads and vehicles together.
    spawn_flow_vehicles(env);
    set_means(env);
    route_shift(env);
    init_grid_map(env);
    env->vision_range = VISION_RANGE;
    init_neighbor_offsets(env);
    free(env->neighbor_cache_indices);
    env->neighbor_cache_indices = (int*)calloc(
        (size_t)(env->grid_cols * env->grid_rows) + 1, sizeof(int));
    cache_neighbor_offsets(env);
    set_start_position(env);
    if (env->active_agent_count <= 0) {
        fprintf(stderr, "roadgraph init produced no agents for %s\n",
            env->map_name ? env->map_name : "(null)");
        exit(1);
    }
    env->num_agents = env->active_agent_count;
    if (env->num_agents > DRIVE_AGENT_MAX) {
        env->num_agents = DRIVE_AGENT_MAX;
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
    env->episode_length = 500;
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
    // A reset target holds. Below 1 m/s the clip stays at MAX_SPEED.
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
        Entity* ego = &env->entities[i];
        if (ego->type != VEHICLE) continue;
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
        // Index 6 stays 0. Index 7 is the episode max speed / 100.
        obs[EGO_FEATURES - 1] = ego->max_speed / MAX_SPEED;

        // First MAX_OBS_AGENTS partners in index order, inside range.
        int obs_idx = EGO_FEATURES;
        int cars_seen = 0;
        int nscan = env->active_agent_count;
        if (nscan > DRIVE_AGENT_MAX) nscan = DRIVE_AGENT_MAX;
        for (int index = 0; index < nscan && cars_seen < MAX_OBS_AGENTS; index++) {
            if (index == i) continue;

            Entity* other = &env->entities[index];
            if (other->x == INVALID_POSITION || other->y == INVALID_POSITION) {
                continue;
            }

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
    sample_initial_positions(env);
    assign_episode_goals(env);
    assign_episode_speeds(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = x;
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].reached_goal = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].reached_goal_this_episode = 0;

        collision_check(env, agent_idx);
    }
    compute_observations(env);
    flow_diagram_clear(env);
    flow_diagram_record(env);
    flow_schedule_reset(env);
    env->stream_n = 0;
    env->stream_in = 0;
}

// Rebuild one map at n cars and reset. Observations must already be bound.
// Pause, sim_follow, sim_through, and the speed ends stay.
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
    if (env->active_agent_count != n) {
        fprintf(stderr, "agents per map: requested %d, placed %d\n",
            n, env->active_agent_count);
    }
    puf_reset(env);
}

static int respawn_partner_xy(Drive* env, int self, float* xs, float* ys) {
    int nscan = env->active_agent_count;
    if (nscan > DRIVE_AGENT_MAX) {
        nscan = DRIVE_AGENT_MAX;
    }
    int n = 0;
    for (int index = 0; index < nscan; index++) {
        if (index == self) {
            continue;
        }
        Entity* o = &env->entities[index];
        if (o->x == INVALID_POSITION || o->y == INVALID_POSITION) {
            continue;
        }
        xs[n] = o->x;
        ys[n] = o->y;
        n++;
    }
    return n;
}

static float respawn_nearest_sq(float x, float y,
        const float* xs, const float* ys, int n) {
    float best = 1.0e30f;
    for (int i = 0; i < n; i++) {
        float dx = x - xs[i];
        float dy = y - ys[i];
        float d = dx * dx + dy * dy;
        if (d < best) {
            best = d;
        }
    }
    return best;
}

// Arc length of the path point closest to the goal.
static float respawn_goal_s(const FlowPath* path, float gx, float gy) {
    float walked = 0.0f;
    float best_d = 1.0e30f;
    float best_s = 0.0f;
    for (int p = 0; p < path->n; p++) {
        FlowPiece* pc = &path->pieces[p];
        float acc = 0.0f;
        for (int i = 0; i + 1 < pc->n; i++) {
            float dx = pc->x[i + 1] - pc->x[i];
            float dy = pc->y[i + 1] - pc->y[i];
            float seg2 = dx * dx + dy * dy;
            float t = 0.0f;
            if (seg2 > 1.0e-8f) {
                float num = (gx - pc->x[i]) * dx + (gy - pc->y[i]) * dy;
                t = num / seg2;
                if (t < 0.0f) {
                    t = 0.0f;
                }
                if (t > 1.0f) {
                    t = 1.0f;
                }
            }
            float qx = pc->x[i] + dx * t;
            float qy = pc->y[i] + dy * t;
            float ddx = qx - gx;
            float ddy = qy - gy;
            float d = ddx * ddx + ddy * ddy;
            float seg = sqrtf(seg2);
            if (d < best_d) {
                best_d = d;
                best_s = walked + acc + t * seg;
            }
            acc += seg;
        }
        walked += pc->len;
    }
    return best_s;
}

// Own spawn when it is clear. Otherwise the clear sample nearest that
// spawn, still short of the goal. If the route has no open point, the
// sample farthest from the nearest partner.
static void respawn_place(Drive* env, int agent, float* ox, float* oy,
        float* oh) {
    FlowRoutes* lanes = (FlowRoutes*)env->lane_state;
    if (!lanes || agent < 0 || agent >= lanes->nveh) {
        return;
    }
    int pi = lanes->path_of[agent];
    if (pi < 0 || pi >= lanes->npaths) {
        return;
    }
    FlowPath* path = &lanes->paths[pi];
    if (path->n <= 0 || path->len < 1.0f) {
        return;
    }
    Entity* e = &env->entities[agent];
    float s0 = lanes->s0[agent];
    float lim = RESPAWN_CLEAR * RESPAWN_CLEAR;
    float goal_lim = MIN_DISTANCE_TO_GOAL * MIN_DISTANCE_TO_GOAL;
    float gx = e->goal_position_x;
    float gy = e->goal_position_y;
    float s_max = respawn_goal_s(path, gx, gy) - MIN_DISTANCE_TO_GOAL;
    if (s_max > path->len) {
        s_max = path->len;
    }
    if (s_max < 0.0f) {
        s_max = 0.0f;
    }

    float xs[DRIVE_AGENT_MAX];
    float ys[DRIVE_AGENT_MAX];
    int np = respawn_partner_xy(env, agent, xs, ys);

    float x, y, h;
    if (s0 <= s_max + 1.0e-3f && path_sample(path, s0, &x, &y, &h)) {
        float gdx = x - gx;
        float gdy = y - gy;
        float nsq = respawn_nearest_sq(x, y, xs, ys, np);
        if (gdx * gdx + gdy * gdy >= goal_lim && nsq > lim) {
            *ox = x;
            *oy = y;
            *oh = h;
            return;
        }
    }

    int nstep = (int)(s_max / RESPAWN_STEP);
    if (nstep < 0) {
        nstep = 0;
    }
    if (nstep > 8000) {
        nstep = 8000;
    }
    int have_clear = 0;
    float clear_ad = 1.0e30f;
    float clear_x = *ox, clear_y = *oy, clear_h = *oh;
    int have_far = 0;
    float far_d = -1.0f;
    float far_x = *ox, far_y = *oy, far_h = *oh;
    for (int k = 0; k <= nstep; k++) {
        float ss = (k == nstep) ? s_max : (float)k * RESPAWN_STEP;
        if (!path_sample(path, ss, &x, &y, &h)) {
            continue;
        }
        float gdx = x - gx;
        float gdy = y - gy;
        if (gdx * gdx + gdy * gdy < goal_lim) {
            continue;
        }
        float nsq = respawn_nearest_sq(x, y, xs, ys, np);
        if (nsq > lim) {
            float ad = ss - s0;
            if (ad < 0.0f) {
                ad = -ad;
            }
            if (!have_clear || ad < clear_ad) {
                have_clear = 1;
                clear_ad = ad;
                clear_x = x;
                clear_y = y;
                clear_h = h;
            }
        } else if (!have_clear && nsq > far_d) {
            have_far = 1;
            far_d = nsq;
            far_x = x;
            far_y = y;
            far_h = h;
        }
    }
    if (have_clear) {
        *ox = clear_x;
        *oy = clear_y;
        *oh = clear_h;
    } else if (have_far) {
        *ox = far_x;
        *oy = far_y;
        *oh = far_h;
    }
}

// Back on this car's route. The goal and target speed stay.
void respawn_agent(Drive* env, int agent_idx) {
    Entity* e = &env->entities[agent_idx];
    float x = e->x;
    float y = e->y;
    float heading = e->heading;
    spawn_pose(env, agent_idx, &x, &y, &heading);
    heading = normalize_heading(heading);
    respawn_place(env, agent_idx, &x, &y, &heading);
    e->x = x;
    e->y = y;
    e->heading = heading;
    e->heading_x = cosf(heading);
    e->heading_y = sinf(heading);
    float spd = e->max_speed;
    if (spd >= SIM_SPEED_MIN) {
        e->vx = spd * e->heading_x;
        e->vy = spd * e->heading_y;
    } else {
        e->vx = 0.0f;
        e->vy = 0.0f;
    }
    e->reached_goal = 0;
    e->respawn_timestep = env->timestep;
}

// The car reached the route end. Count it, then free the slot. The next
// forward sees the terminal and clears that policy state.
static void flow_leave(Drive* env, int slot, int agent_idx) {
    Entity* e = &env->entities[agent_idx];
    Log* lg = env->logs ? &env->logs[slot] : NULL;
    if (e->reached_goal_this_episode) {
        env->log.completion_rate += 1.0f;
    }
    if (lg) {
        env->log.offroad_rate += lg->offroad_rate;
        env->log.collision_rate += lg->collision_rate;
        env->log.clean_collision_rate += lg->clean_collision_rate;
        env->log.episode_length += lg->episode_length;
        env->log.episode_return += lg->episode_return;
        *lg = (Log){0};
    }
    if (e->reached_goal_this_episode && !e->collided_before_goal) {
        env->log.score += 1.0f;
        env->log.perf += 1.0f;
    }
    env->stream_n += 1;
    e->x = INVALID_POSITION;
    e->y = INVALID_POSITION;
    e->vx = 0;
    e->vy = 0;
    e->reached_goal = 0;
    e->reached_goal_this_episode = 0;
    e->collided_before_goal = 0;
    e->respawn_timestep = -1;
    e->collision_state = NO_COLLISION;
    if (env->agents[slot].terminals) {
        env->agents[slot].terminals[0] = 1.0f;
    }
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
// trip, is the end of this lap. Training keeps the disk only. Inference
// uses this so a car that arrives can leave. origin is the sampled start,
// or the entrance after an insertion.
static int passed_goal(const Drive* env, const Entity* e) {
    (void)env;
    float sx = e->origin_x;
    float sy = e->origin_y;
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

    // The run is episode_length steps.
    if (env->timestep == env->episode_length) {
        add_log(env);
        puf_reset(env);
        return;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = i;
        Entity* agent = &env->entities[agent_idx];
        if (agent->x == INVALID_POSITION) {
            continue;
        }
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
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
        int agent_idx = i;
        if (env->entities[agent_idx].x == INVALID_POSITION) {
            continue;
        }
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
        if (!at_goal && env->drive_mode == DRIVE_MODE_INFERENCE) {
            at_goal = passed_goal(env, &env->entities[agent_idx]);
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
        // speed. Below 1 m/s there is no target, so this stays off.
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

    // Record the arrival. Train sends the car back to its own start.
    // Inference frees the slot and inserts whoever is due at the entrance.
    flow_diagram_record(env);

    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = i;
        if (!env->entities[agent_idx].reached_goal) {
            continue;
        }
        if (env->drive_mode == DRIVE_MODE_INFERENCE) {
            flow_leave(env, i, agent_idx);
        } else {
            respawn_agent(env, agent_idx);
        }
    }
    if (env->drive_mode == DRIVE_MODE_INFERENCE) {
        flow_inject(env);
    }

    // Score used the overlapping pose. Physics then pushes the cars apart.
    // Through leaves that overlap for the next step.
    if (!env->sim_through) {
        bump_agents(env);
    }

    compute_observations(env);
}

static void drive_bin_dir(Dict* kwargs, char* out, size_t out_sz) {
    const char* data_dir = dict_get_str(kwargs, "data_dir");
    snprintf(out, out_sz, "%s/binaries", data_dir);
}

static void resolve_map_path(const char* bin_dir, int map_id, char* out, size_t out_sz) {
    snprintf(out, out_sz, "%s/map_%03d.bin", bin_dir, map_id);
}

// Train is the default so the vectorized trainer, which only calls puf_init,
// samples curriculum goals. ./drive sets inference before the first reset.
void puf_init(Env* env, Dict* kwargs) {
    env->human_agent_idx = dict_get(kwargs, "human_agent_idx");
    env->reward_vehicle_collision = dict_get(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = dict_get(kwargs, "reward_offroad_collision");
    env->reward_goal_post_respawn = dict_get(kwargs, "reward_goal_post_respawn");
    env->reward_vehicle_collision_post_respawn = dict_get(kwargs, "reward_vehicle_collision_post_respawn");
    env->reward_jerk = dict_get(kwargs, "reward_jerk");
    env->reward_speed = dict_get(kwargs, "reward_speed");
    env->cfg = kwargs;
    env->drive_mode = DRIVE_MODE_TRAIN;
    int map_id = dict_get(kwargs, "map_id");
    char bin_dir[480];
    char map_file[512];
    drive_bin_dir(kwargs, bin_dir, sizeof(bin_dir));
    resolve_map_path(bin_dir, map_id, map_file, sizeof(map_file));
    env->map_name = strdup(map_file);
    env->max_agents = MAX_AGENTS;
    init(env);
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
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

// Window, camera, HUD, and the velocity diagram.
#include "render.h"

