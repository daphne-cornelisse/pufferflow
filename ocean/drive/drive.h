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

struct Log {
    float episode_return;
    float episode_length;
    float perf;
    float score;
    float offroad_rate;
    float collision_rate;
    float collision_count;
    float completion_rate;
    float n;
};

typedef struct Entity {
    int type;
    // Point count of a road polyline. Cars stay at 0. Reset samples
    // their position and goal from lane points.
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
    int collided_before_goal;
    int active_agent;
    float prev_accel;
    float prev_steer;
} Entity;

void free_entity(Entity* entity) {
    free(entity->pts_x);
    free(entity->pts_y);
    entity->pts_x = NULL;
    entity->pts_y = NULL;
}

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
    float world_mean_x;
    float world_mean_y;
    float reward_goal;
    float reward_jerk;
    unsigned int rng;
    // 1 after ./drive loads a checkpoint that matches this network.
    int checkpoint_ok;
    char checkpoint_error[256];
    // Straight-line goal band, meters. Reset samples a lane point in it.
    float goal_distance_min;
    float goal_distance_max;
    // Set from the window. Headless eval and recording never read these.
    int sim_paused;
    // Overhead camera stays on selected_agent_idx. Off restores the map view.
    int sim_follow;
    // One frame after Reset, so that click draws timestep 0.
    int sim_hold_step;
    Dict* cfg;
};

static int puf_sim_paused(Drive* env) {
    if (puf_record_enabled()) {
        return 0;
    }
    return env->sim_paused || env->sim_hold_step;
}

// Roads only.
Entity* load_map_binary(const char* filename, Drive* env) {
    FILE* file = fopen(filename, "rb");
    assert(file);
    assert(fread(&env->num_roads, sizeof(int), 1, file) == 1);
    assert(env->num_roads >= 0);
    env->num_objects = 0;
    env->num_entities = env->num_roads;
    Entity* entities = (Entity*)calloc((size_t)env->num_roads, sizeof(Entity));
    assert(entities || env->num_roads == 0);
    for (int i = 0; i < env->num_roads; i++) {
        int size = 0;
        assert(fread(&entities[i].type, sizeof(int), 1, file) == 1);
        assert(fread(&size, sizeof(int), 1, file) == 1);
        assert(entities[i].type >= ROAD_LANE && entities[i].type <= DRIVEWAY);
        assert(size >= 1);
        entities[i].array_size = size;
        entities[i].pts_x = (float*)malloc((size_t)size * sizeof(float));
        entities[i].pts_y = (float*)malloc((size_t)size * sizeof(float));
        float* z = (float*)malloc((size_t)size * sizeof(float));
        assert(entities[i].pts_x && entities[i].pts_y && z);
        assert(fread(entities[i].pts_x, sizeof(float), size, file)
            == (size_t)size);
        assert(fread(entities[i].pts_y, sizeof(float), size, file)
            == (size_t)size);
        assert(fread(z, sizeof(float), size, file) == (size_t)size);
        free(z);
        int mark = 0;
        assert(fread(&entities[i].width, sizeof(float), 1, file) == 1);
        assert(fread(&entities[i].length, sizeof(float), 1, file) == 1);
        assert(fread(&entities[i].height, sizeof(float), 1, file) == 1);
        assert(fread(&entities[i].goal_position_x, sizeof(float), 1, file) == 1);
        assert(fread(&entities[i].goal_position_y, sizeof(float), 1, file) == 1);
        assert(fread(&entities[i].goal_position_z, sizeof(float), 1, file) == 1);
        assert(fread(&mark, sizeof(int), 1, file) == 1);
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
            if (e->pts_x[j] == INVALID_POSITION) {
                continue;
            }
            if (e->pts_y[j] == INVALID_POSITION) {
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

// Mean Centering
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
            if (e->pts_x[j] == INVALID_POSITION) {
                continue;
            }
            if (e->pts_y[j] == INVALID_POSITION) {
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

// Collision detection
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

static void bump_pair(Drive* env, Entity* a, Entity* b,
        float pax, float pay, float pbx, float pby, int sweep) {
    if (a->x == INVALID_POSITION || b->x == INVALID_POSITION) {
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
        float* c0 = corners[k];
        float* c1 = corners[(k + 1) % 4];
        if (check_line_intersection(c0, c1, q0, q1)) {
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
        int n = check_neighbors(env, car->x, car->y, list,
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

static void bump_agents(Drive* env) {
    int n = env->active_agent_count;
    if (n > DRIVE_AGENT_MAX) {
        n = DRIVE_AGENT_MAX;
    }
    float prev_x[DRIVE_AGENT_MAX];
    float prev_y[DRIVE_AGENT_MAX];
    for (int i = 0; i < n; i++) {
        Entity* e = &env->entities[i];
        prev_x[i] = e->x - e->vx * SIM_DT;
        prev_y[i] = e->y - e->vy * SIM_DT;
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
    if (agent->x == INVALID_POSITION) {
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
                collided = OFFROAD;
                break;
            }
        }
        if (collided == OFFROAD) {
            break;
        }
    }

    // Check vehicle-vehicle collisions. Agent i is entity i.
    int nscan = env->active_agent_count;
    if (nscan > DRIVE_AGENT_MAX) {
        nscan = DRIVE_AGENT_MAX;
    }
    for (int index = 0; index < nscan; index++) {
        if (index == agent_idx) {
            continue;
        }
        Entity* entity = &env->entities[index];
        if (entity->x == INVALID_POSITION || entity->y == INVALID_POSITION) {
            continue;
        }
        float dx = entity->x - agent->x;
        float dy = entity->y - agent->y;
        if ((dx * dx + dy * dy) > COLLISION_DIST_SQ) {
            continue;
        }
        if (check_aabb_collision(agent, entity)) {
            collided = AGENT_COLLISION;
            car_collided_with_index = index;
            break;
        }
    }

    agent->collision_state = collided;
    return car_collided_with_index;
}


// Initialization / Cleanup
static void drive_release_world(Drive* env) {
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
    assert(env->entities);
    int n = env->max_agents;
    assert(n >= 1 && n <= DRIVE_AGENT_MAX);
    int nroads = env->num_roads;
    Entity* neu = (Entity*)calloc(
        (size_t)n + (size_t)nroads, sizeof(Entity));
    assert(neu);
    for (int i = 0; i < n; i++) {
        Entity* e = &neu[i];
        e->type = VEHICLE;
        e->width = CAR_WIDTH * COLLISION_BOX_SCALE;
        e->length = CAR_LENGTH * COLLISION_BOX_SCALE;
        e->height = CAR_HEIGHT;
        e->active_agent = 1;
        e->valid = 1;
        e->collision_state = NO_COLLISION;
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
    }
    if (nroads > 0) {
        memcpy(neu + n, env->entities, (size_t)nroads * sizeof(Entity));
    }
    free(env->entities);
    env->entities = neu;
    env->num_objects = n;
    env->num_entities = n + nroads;
    env->active_agent_count = n;
    env->num_agents = n;
    set_means(env);
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
}

void puf_close(Drive* env) {
    drive_release_world(env);
}

// Dynamics
void move_dynamics(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    float* atn = env->agents[agent_idx].actions;
    int acceleration_index = (int)atn[0];
    int steering_index = (int)atn[1];
    float acceleration = ACCELERATION_VALUES[acceleration_index];
    float steering = STEERING_VALUES[steering_index];

    float x = agent->x;
    float y = agent->y;
    float heading = agent->heading;
    float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);

    speed = speed + 0.5f * acceleration * SIM_DT;
    if (speed > MAX_SPEED) {
        speed = MAX_SPEED;
    }
    if (speed < -MAX_SPEED) {
        speed = -MAX_SPEED;
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
        if (!obs) {
            continue;
        }
        memset(obs, 0, OBS_SIZE * sizeof(float));
        Entity* ego = &env->entities[i];
        if (ego->type != VEHICLE) {
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
        // Index 6 and index 7 stay 0. Index 7 is open.

        // First MAX_AGENTS partners in index order, inside range.
        int obs_idx = EGO_FEATURES;
        int cars_seen = 0;
        int nscan = env->active_agent_count;
        if (nscan > DRIVE_AGENT_MAX) {
            nscan = DRIVE_AGENT_MAX;
        }
        for (int index = 0; index < nscan && cars_seen < MAX_AGENTS; index++) {
            if (index == i) {
                continue;
            }

            Entity* other = &env->entities[index];
            if (other->x == INVALID_POSITION || other->y == INVALID_POSITION) {
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
        int remaining_partner_obs =
            (MAX_AGENTS - cars_seen) * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;

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
        int remaining_obs =
            (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * ROAD_FEATURES;
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }
}

static int lane_point_count(Drive* env) {
    int n = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type != ROAD_LANE || !e->pts_x || e->array_size < 2) {
            continue;
        }
        n += e->array_size;
    }
    return n;
}

// Lane point k, in entity order. Heading follows the segment through it.
static void lane_point(Drive* env, int k, float* x, float* y, float* h) {
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type != ROAD_LANE || !e->pts_x || e->array_size < 2) {
            continue;
        }
        if (k >= e->array_size) {
            k -= e->array_size;
            continue;
        }
        int from = k;
        int to = k + 1;
        if (to >= e->array_size) {
            from = k - 1;
            to = k;
        }
        *x = e->pts_x[k];
        *y = e->pts_y[k];
        if (h) {
            *h = atan2f(e->pts_y[to] - e->pts_y[from],
                e->pts_x[to] - e->pts_x[from]);
        }
        return;
    }
    assert(0);
}

// Lane heading is the flow. One direction, so ahead of it is downstream.
static int goal_in_front(float x, float y, float hx, float hy,
        float gx, float gy) {
    float dx = gx - x;
    float dy = gy - y;
    return dx * hx + dy * hy > 0.0f;
}

static int pose_blocked(Drive* env, int self) {
    Entity* a = &env->entities[self];
    for (int i = 0; i < self; i++) {
        Entity* o = &env->entities[i];
        if (o->x == INVALID_POSITION || o->y == INVALID_POSITION) {
            continue;
        }
        if (check_aabb_collision(a, o)) {
            return 1;
        }
    }
    return 0;
}

// One lane point per car, then a lane point in the goal band and in
// front of the car. Velocity stays at zero.
void initialize_env(Drive* env) {
    int npts = lane_point_count(env);
    assert(npts > 1);
    int n = env->active_agent_count;
    float lo2 = env->goal_distance_min * env->goal_distance_min;
    float hi2 = env->goal_distance_max * env->goal_distance_max;
    for (int i = 0; i < n; i++) {
        env->entities[i].x = INVALID_POSITION;
        env->entities[i].y = INVALID_POSITION;
    }
    int tries = npts * 8;
    if (tries < 64) {
        tries = 64;
    }
    for (int i = 0; i < n; i++) {
        int placed = 0;
        for (int attempt = 0; attempt < tries && !placed; attempt++) {
            int k = (int)(rand_r(&env->rng) % (unsigned)npts);
            float x, y, h;
            lane_point(env, k, &x, &y, &h);
            if (x == INVALID_POSITION || y == INVALID_POSITION) {
                continue;
            }
            h = normalize_heading(h);
            Entity* e = &env->entities[i];
            e->x = x;
            e->y = y;
            e->z = 0.0f;
            e->heading = h;
            e->heading_x = cosf(h);
            e->heading_y = sinf(h);
            e->vx = 0.0f;
            e->vy = 0.0f;
            e->vz = 0.0f;
            e->prev_accel = 0.0f;
            e->prev_steer = 0.0f;
            e->collision_state = NO_COLLISION;
            e->reached_goal = 0;
            e->collided_before_goal = 0;
            e->valid = 1;
            if (pose_blocked(env, i)) {
                e->x = INVALID_POSITION;
                e->y = INVALID_POSITION;
                continue;
            }
            collision_check(env, i);
            if (e->collision_state != NO_COLLISION) {
                e->collision_state = NO_COLLISION;
                e->x = INVALID_POSITION;
                e->y = INVALID_POSITION;
                continue;
            }
            int goal_ok = 0;
            for (int gtry = 0; gtry < npts && !goal_ok; gtry++) {
                int gk = (int)(rand_r(&env->rng) % (unsigned)npts);
                float gx, gy;
                lane_point(env, gk, &gx, &gy, NULL);
                float dx = gx - x;
                float dy = gy - y;
                float d2 = dx * dx + dy * dy;
                if (d2 < lo2 || d2 > hi2) {
                    continue;
                }
                if (!goal_in_front(x, y, e->heading_x, e->heading_y,
                        gx, gy)) {
                    continue;
                }
                e->goal_position_x = gx;
                e->goal_position_y = gy;
                e->goal_position_z = 0.0f;
                goal_ok = 1;
            }
            if (!goal_ok) {
                e->x = INVALID_POSITION;
                e->y = INVALID_POSITION;
                continue;
            }
            placed = 1;
        }
        assert(placed);
    }
    for (int i = 0; i < n; i++) {
        collision_check(env, i);
        assert(env->entities[i].collision_state == NO_COLLISION);
        Entity* e = &env->entities[i];
        float dx = e->goal_position_x - e->x;
        float dy = e->goal_position_y - e->y;
        float d2 = dx * dx + dy * dy;
        assert(d2 >= lo2 && d2 <= hi2);
        assert(goal_in_front(e->x, e->y, e->heading_x, e->heading_y,
            e->goal_position_x, e->goal_position_y));
        assert(e->vx == 0.0f && e->vy == 0.0f);
        for (int j = i + 1; j < n; j++) {
            assert(!check_aabb_collision(e, &env->entities[j]));
        }
    }
}

void puf_reset(Drive* env) {
    env->timestep = 0;
    set_start_position(env);
    initialize_env(env);
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i] = (Log){0};
    }
    compute_observations(env);
}

static void drive_human_controls(Drive* env) {
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
        env->selected_agent_idx = (env->selected_agent_idx + 1) % env->active_agent_count;
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

static int point_early(Drive* env, float x) {
    if (!env->map_corners) {
        return 1;
    }
    return x <= env->map_corners[0] + EARLY_ROAD;
}

static void write_goal(Entity* e, float gx, float gy) {
    e->goal_position_x = gx;
    e->goal_position_y = gy;
    e->goal_position_z = 0.0f;
    e->reached_goal = 0;
}

// Uniform among lane points in front of (x, y) whose squared distance
// is in [lo2, hi2]. Returns 1 and writes the goal when one matches.
static int pick_lane_goal(Drive* env, Entity* e, float x, float y,
        float hx, float hy, float lo2, float hi2) {
    int npts = lane_point_count(env);
    float pick_x = 0.0f;
    float pick_y = 0.0f;
    int seen = 0;
    for (int gk = 0; gk < npts; gk++) {
        float gx, gy;
        lane_point(env, gk, &gx, &gy, NULL);
        float dx = gx - x;
        float dy = gy - y;
        float d2 = dx * dx + dy * dy;
        if (d2 < lo2 || d2 > hi2) {
            continue;
        }
        if (!goal_in_front(x, y, hx, hy, gx, gy)) {
            continue;
        }
        seen++;
        if ((int)(rand_r(&env->rng) % (unsigned)seen) == 0) {
            pick_x = gx;
            pick_y = gy;
        }
    }
    if (seen > 0) {
        write_goal(e, pick_x, pick_y);
        return 1;
    }
    return 0;
}

// Another lane point in front. The distance band first, then any point
// outside the arrival disk. Returns 0 when nothing is ahead.
static int sample_goal(Drive* env, Entity* e, float x, float y) {
    float lo2 = env->goal_distance_min * env->goal_distance_min;
    float hi2 = env->goal_distance_max * env->goal_distance_max;
    float hx = e->heading_x;
    float hy = e->heading_y;
    if (pick_lane_goal(env, e, x, y, hx, hy, lo2, hi2)) {
        return 1;
    }
    float min2 = MIN_DISTANCE_TO_GOAL * MIN_DISTANCE_TO_GOAL;
    return pick_lane_goal(env, e, x, y, hx, hy, min2, 1.0e30f);
}

// Early lane points first. Any free lane point if the entrance is full.
// A pose counts only when it has a goal in front.
static int try_respawn_pose(Drive* env, int agent_idx, int early_only) {
    Entity* e = &env->entities[agent_idx];
    int npts = lane_point_count(env);
    int tries = npts;
    if (tries < 64) {
        tries = 64;
    }
    for (int attempt = 0; attempt < tries; attempt++) {
        int k = (int)(rand_r(&env->rng) % (unsigned)npts);
        float x, y, h;
        lane_point(env, k, &x, &y, &h);
        if (early_only && !point_early(env, x)) {
            continue;
        }
        h = normalize_heading(h);
        e->x = x;
        e->y = y;
        e->z = 0.0f;
        e->heading = h;
        e->heading_x = cosf(h);
        e->heading_y = sinf(h);
        e->vx = 0.0f;
        e->vy = 0.0f;
        e->vz = 0.0f;
        e->prev_accel = 0.0f;
        e->prev_steer = 0.0f;
        e->collision_state = NO_COLLISION;
        if (spot_taken(env, agent_idx)) {
            continue;
        }
        collision_check(env, agent_idx);
        if (e->collision_state != NO_COLLISION) {
            e->collision_state = NO_COLLISION;
            continue;
        }
        if (!sample_goal(env, e, e->x, e->y)) {
            continue;
        }
        return 1;
    }
    return 0;
}

static void restore_pose(Entity* e, float ox, float oy, float oh,
        float ohx, float ohy, float ovx, float ovy) {
    e->x = ox;
    e->y = oy;
    e->heading = oh;
    e->heading_x = ohx;
    e->heading_y = ohy;
    e->vx = ovx;
    e->vy = ovy;
}

// Back to an early lane point, then a new goal in front. A full entrance
// keeps the arrival pose when that pose still has a goal ahead.
static void respawn_agent(Drive* env, int agent_idx) {
    Entity* e = &env->entities[agent_idx];
    float ox = e->x;
    float oy = e->y;
    float oh = e->heading;
    float ohx = e->heading_x;
    float ohy = e->heading_y;
    float ovx = e->vx;
    float ovy = e->vy;
    if (try_respawn_pose(env, agent_idx, 1)) {
        return;
    }
    restore_pose(e, ox, oy, oh, ohx, ohy, ovx, ovy);
    if (sample_goal(env, e, e->x, e->y)) {
        return;
    }
    if (try_respawn_pose(env, agent_idx, 0)) {
        return;
    }
    restore_pose(e, ox, oy, oh, ohx, ohy, ovx, ovy);
}

void puf_step(Drive* env) {
    drive_human_controls(env);
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
    env->timestep++;

    if (env->timestep == env->episode_length) {
        for (int i = 0; i < env->active_agent_count; i++) {
            env->log.completion_rate += env->logs[i].completion_rate;
            env->log.offroad_rate += env->logs[i].offroad_rate;
            env->log.collision_rate += env->logs[i].collision_rate;
            env->log.collision_count += env->logs[i].collision_count;
            env->log.score += env->logs[i].score;
            env->log.perf += env->logs[i].perf;
            env->log.episode_length += env->logs[i].episode_length;
            env->log.episode_return += env->logs[i].episode_return;
            env->log.n += 1;
        }
        puf_reset(env);
        return;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* agent = &env->entities[i];
        if (agent->x == INVALID_POSITION) {
            continue;
        }
        env->logs[i].episode_length += 1;
        agent->collision_state = NO_COLLISION;
        move_dynamics(env, i);
    }

    // Collision detection and rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* agent = &env->entities[i];
        if (agent->x == INVALID_POSITION) {
            continue;
        }
        agent->collision_state = NO_COLLISION;
        collision_check(env, i);
        int collision_state = agent->collision_state;
        if (collision_state == AGENT_COLLISION) {
            env->agents[i].rewards[0] = env->reward_vehicle_collision;
            env->logs[i].episode_return += env->reward_vehicle_collision;
            env->logs[i].collision_rate = 1.0f;
            env->logs[i].collision_count += 1.0f;
        } else if (collision_state == OFFROAD) {
            env->agents[i].rewards[0] = env->reward_offroad_collision;
            env->logs[i].offroad_rate = 1.0f;
            env->logs[i].episode_return += env->reward_offroad_collision;
        }
        if (collision_state > NO_COLLISION
                && env->logs[i].completion_rate == 0.0f) {
            agent->collided_before_goal = 1;
        }

        float distance_to_goal = relative_distance_2d(
            agent->x, agent->y,
            agent->goal_position_x, agent->goal_position_y);

        // Inside 2 m: pay reward_goal, then respawn with a new goal.
        int at_goal = distance_to_goal < MIN_DISTANCE_TO_GOAL;
        if (at_goal) {
            env->agents[i].rewards[0] += env->reward_goal;
            env->logs[i].episode_return += env->reward_goal;
            agent->reached_goal = 1;
            if (env->logs[i].completion_rate == 0.0f) {
                env->logs[i].completion_rate = 1.0f;
                if (!agent->collided_before_goal) {
                    env->logs[i].score = 1.0f;
                    env->logs[i].perf = 1.0f;
                }
            }
        }

        // Squared bin-steps of accel and steer. One adjacent-bin flip costs
        // reward_jerk. The first step of an episode compares against 0.
        float* atn = env->agents[i].actions;
        if (atn) {
            float accel = ACCELERATION_VALUES[(int)atn[0]];
            float steer = STEERING_VALUES[(int)atn[1]];
            float accel_step = ACCELERATION_VALUES[1] - ACCELERATION_VALUES[0];
            float steer_step = STEERING_VALUES[1] - STEERING_VALUES[0];
            float da = (accel - agent->prev_accel) / accel_step;
            float ds = (steer - agent->prev_steer) / steer_step;
            float jerk_penalty = env->reward_jerk * (da * da + ds * ds);
            env->agents[i].rewards[0] += jerk_penalty;
            env->logs[i].episode_return += jerk_penalty;
            agent->prev_accel = accel;
            agent->prev_steer = steer;
        }
        if (at_goal) {
            respawn_agent(env, i);
        }
    }

    bump_agents(env);

    compute_observations(env);
}

void puf_init(Env* env, Dict* kwargs) {
    env->selected_agent_idx = dict_get(kwargs, "selected_agent_idx");
    env->reward_vehicle_collision = dict_get(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = dict_get(kwargs, "reward_offroad_collision");
    env->reward_goal = dict_get(kwargs, "reward_goal");
    env->reward_jerk = dict_get(kwargs, "reward_jerk");
    env->cfg = kwargs;
    int map_id = dict_get(kwargs, "map_id");
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
    drive_load_world(env);
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
    dict_set(out, "collision_count", log->collision_count);
    dict_set(out, "n", log->n);
    dict_set(out, "completion_rate", log->completion_rate);
}

// Window, camera, and HUD.
#include "render.h"

