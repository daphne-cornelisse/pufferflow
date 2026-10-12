// Bird's-eye sanity renders of one loaded drive map.
//
// network.png is the roadgraph the simulator loaded: lane centers, road
// lines, road edges, and vehicles.
// observation.png is one agent's observation buffer, decoded the same way
// as PufferDrive 2.0 draw_agent_obs(..., mode=1) (drive.h around line 2434):
// world frame, and only road edges. Lane centers and road lines stay in the
// vector and are not drawn.
//
//   ./build/render_sanity <data_dir> <map_id|random> <output_dir> [agent_slot]
//
// agent_slot is the active-agent index. The default is human_agent_idx (the
// SDC, active slot 0).

#include "drive.h"

#include <sys/stat.h>
#include <time.h>

static const Color LANE_COLOR = {235, 190, 40, 255};
static const Color LINE_COLOR = {255, 255, 255, 255};
static const Color CURB_COLOR = {170, 176, 186, 255};
static const Color LIGHTBLUE = {167, 204, 255, 255};
static const Color EGO_COLOR = {220, 60, 60, 255};
static const Color PARTNER_COLOR = {255, 150, 40, 255};
static const Color EDGE_LINE = {70, 140, 255, 255};

static int ensure_dir(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(path, 0755) == 0;
}

static void ego_to_world(float px, float py, float hx, float hy,
                         float lx, float ly, float* wx, float* wy) {
    *wx = px + (lx * hx - ly * hy);
    *wy = py + (lx * hy + ly * hx);
}

static void draw_oriented_box(float x, float y, float hx, float hy,
                              float half_len, float half_w,
                              Color fill, Color wire, float z) {
    float lx[4] = {half_len, half_len, -half_len, -half_len};
    float ly[4] = {-half_w, half_w, half_w, -half_w};
    Vector3 c[4];
    for (int i = 0; i < 4; i++) {
        float wx, wy;
        ego_to_world(x, y, hx, hy, lx[i], ly[i], &wx, &wy);
        c[i] = (Vector3){wx, wy, z};
    }
    DrawTriangle3D(c[0], c[1], c[2], fill);
    DrawTriangle3D(c[0], c[2], c[3], fill);
    rlSetLineWidth(2.0f);
    for (int i = 0; i < 4; i++) {
        DrawLine3D(c[i], c[(i + 1) % 4], wire);
    }
    Vector3 nose = {
        x + hx * half_len * 1.6f,
        y + hy * half_len * 1.6f,
        z
    };
    DrawLine3D((Vector3){x, y, z}, nose, wire);
}

static int is_tracked_vehicle(Drive* env, int entity_index) {
    for (int j = 0; j < env->active_agent_count; j++) {
        if (env->active_agent_indices[j] == entity_index) return 1;
    }
    for (int j = 0; j < env->static_agent_count; j++) {
        if (env->static_agent_indices[j] == entity_index) return 1;
    }
    return 0;
}

static void draw_network(Drive* env, int agent_slot) {
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type < ROAD_LANE || e->array_size < 2) continue;
        for (int j = 0; j < e->array_size - 1; j++) {
            Vector3 a = {e->traj_x[j], e->traj_y[j], 0.05f};
            Vector3 b = {e->traj_x[j + 1], e->traj_y[j + 1], 0.05f};
            if (e->type == ROAD_EDGE) {
                draw_road_edge(env, a.x, a.y, b.x, b.y);
                rlSetLineWidth(2.0f);
                DrawLine3D(a, b, CURB_COLOR);
            } else if (e->type == ROAD_LANE) {
                rlSetLineWidth(2.0f);
                DrawLine3D(a, b, LANE_COLOR);
            } else if (e->type == ROAD_LINE) {
                rlSetLineWidth(2.0f);
                DrawLine3D(a, b, LINE_COLOR);
            } else if (e->type == STOP_SIGN) {
                rlSetLineWidth(2.0f);
                DrawLine3D(a, b, RED);
            } else if (e->type == CROSSWALK) {
                rlSetLineWidth(2.0f);
                DrawLine3D(a, b, ORANGE);
            } else if (e->type == SPEED_BUMP) {
                rlSetLineWidth(2.0f);
                DrawLine3D(a, b, YELLOW);
            } else if (e->type == DRIVEWAY) {
                rlSetLineWidth(2.0f);
                DrawLine3D(a, b, RED);
            }
        }
    }

    int ego_entity = env->active_agent_indices[agent_slot];
    for (int i = 0; i < env->num_entities; i++) {
        Entity* e = &env->entities[i];
        if (e->type != VEHICLE && e->type != PEDESTRIAN && e->type != CYCLIST) continue;
        if (!is_tracked_vehicle(env, i) || e->respawn_timestep != -1) continue;
        if (e->x == INVALID_POSITION || e->y == INVALID_POSITION) continue;
        Color color = (Color){150, 150, 150, 255};
        int active_slot = -1;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                active_slot = j;
                break;
            }
        }
        if (active_slot >= 0) color = BLUE;
        if (i == ego_entity) color = EGO_COLOR;
        draw_oriented_box(e->x, e->y, e->heading_x, e->heading_y,
                          e->length * 0.5f, e->width * 0.5f,
                          Fade(color, 0.55f), color, 0.6f);
        if (active_slot >= 0) {
            Color goal = (i == ego_entity) ? LIGHTBLUE : DARKGREEN;
            DrawSphere((Vector3){e->goal_position_x, e->goal_position_y, 0.4f}, 1.2f, goal);
        }
    }
}

// Decode agents[agent_slot].observations. Returns how far from the ego the
// drawn features reach, so the camera can contain the goal.
static float draw_observation(Drive* env, int agent_slot, int* edges_out, int* partners_out) {
    float* obs = env->agents[agent_slot].observations;
    int entity_index = env->active_agent_indices[agent_slot];
    Entity* ego = &env->entities[entity_index];
    float px = ego->x;
    float py = ego->y;
    float hx = ego->heading_x;
    float hy = ego->heading_y;
    float reach = 0.0f;

    int map_start = EGO_FEATURES + PARTNER_FEATURES * (MAX_AGENTS - 1);
    int edges = 0;
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
        int idx = map_start + k * ROAD_FEATURES;
        if (obs[idx] == 0.0f && obs[idx + 1] == 0.0f) continue;
        int entity_type = (int)obs[idx + 6];
        if (entity_type + ROAD_LANE != ROAD_EDGE) continue;

        float x_mid = obs[idx] / OBS_POSITION_SCALE;
        float y_mid = obs[idx + 1] / OBS_POSITION_SCALE;
        float rel_angle = atan2f(obs[idx + 5], obs[idx + 4]);
        float seg_len = obs[idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        float x_start = x_mid - seg_len * cosf(rel_angle);
        float y_start = y_mid - seg_len * sinf(rel_angle);
        float x_end = x_mid + seg_len * cosf(rel_angle);
        float y_end = y_mid + seg_len * sinf(rel_angle);

        float mx, my, sx, sy, ex, ey;
        ego_to_world(px, py, hx, hy, x_mid, y_mid, &mx, &my);
        ego_to_world(px, py, hx, hy, x_start, y_start, &sx, &sy);
        ego_to_world(px, py, hx, hy, x_end, y_end, &ex, &ey);
        DrawCube((Vector3){mx, my, 0.4f}, 0.5f, 0.5f, 0.5f, PUFF_CYAN);
        rlSetLineWidth(3.0f);
        DrawLine3D((Vector3){sx, sy, 0.4f}, (Vector3){ex, ey, 0.4f}, EDGE_LINE);
        float far = relative_distance_2d(px, py, sx, sy);
        float far_end = relative_distance_2d(px, py, ex, ey);
        if (far_end > far) far = far_end;
        if (far > reach) reach = far;
        edges++;
    }

    int partners = 0;
    int obs_idx = EGO_FEATURES;
    for (int j = 0; j < MAX_AGENTS - 1; j++) {
        if (obs[obs_idx] == 0.0f || obs[obs_idx + 1] == 0.0f) {
            obs_idx += PARTNER_FEATURES;
            continue;
        }
        float x = obs[obs_idx] / OBS_POSITION_SCALE;
        float y = obs[obs_idx + 1] / OBS_POSITION_SCALE;
        float half_w = 0.5f * obs[obs_idx + 2] * MAX_VEH_WIDTH;
        float half_l = 0.5f * obs[obs_idx + 3] * MAX_VEH_LEN;
        float theta_x = obs[obs_idx + 4];
        float theta_y = obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_h = cosf(partner_angle);
        float sin_h = sinf(partner_angle);
        float local[4][2] = {
            {x + (half_l * cos_h - half_w * sin_h), y + (half_l * sin_h + half_w * cos_h)},
            {x + (half_l * cos_h + half_w * sin_h), y + (half_l * sin_h - half_w * cos_h)},
            {x + (-half_l * cos_h + half_w * sin_h), y + (-half_l * sin_h - half_w * cos_h)},
            {x + (-half_l * cos_h - half_w * sin_h), y + (-half_l * sin_h + half_w * cos_h)},
        };
        Vector3 world[4];
        for (int c = 0; c < 4; c++) {
            float wx, wy;
            ego_to_world(px, py, hx, hy, local[c][0], local[c][1], &wx, &wy);
            world[c] = (Vector3){wx, wy, 0.5f};
            float far = relative_distance_2d(px, py, wx, wy);
            if (far > reach) reach = far;
        }
        DrawTriangle3D(world[0], world[1], world[2], Fade(PARTNER_COLOR, 0.45f));
        DrawTriangle3D(world[0], world[2], world[3], Fade(PARTNER_COLOR, 0.45f));
        rlSetLineWidth(2.0f);
        for (int c = 0; c < 4; c++) {
            DrawLine3D(world[c], world[(c + 1) % 4], PARTNER_COLOR);
        }
        float arrow_length = 2.5f;
        float ax = x + arrow_length * cos_h;
        float ay = y + arrow_length * sin_h;
        float pxw, pyw, axw, ayw;
        ego_to_world(px, py, hx, hy, x, y, &pxw, &pyw);
        ego_to_world(px, py, hx, hy, ax, ay, &axw, &ayw);
        DrawLine3D((Vector3){pxw, pyw, 0.5f}, (Vector3){axw, ayw, 0.5f}, PUFF_WHITE);
        partners++;
        obs_idx += PARTNER_FEATURES;
    }

    float goal_x = obs[0] / OBS_GOAL_SCALE;
    float goal_y = obs[1] / OBS_GOAL_SCALE;
    float gwx, gwy;
    ego_to_world(px, py, hx, hy, goal_x, goal_y, &gwx, &gwy);
    DrawSphere((Vector3){gwx, gwy, 0.4f}, 1.5f, LIGHTBLUE);
    float goal_far = relative_distance_2d(px, py, gwx, gwy);
    if (goal_far > reach) reach = goal_far;

    // Ego pose is not an observation slot. 2.0 still draws this car so the
    // decoded features have an anchor.
    draw_oriented_box(px, py, hx, hy, ego->length * 0.5f, ego->width * 0.5f,
                      Fade(EGO_COLOR, 0.55f), EGO_COLOR, 0.7f);

    *edges_out = edges;
    *partners_out = partners;
    return reach;
}

static void count_road_slots(const float* obs, int* lane, int* line, int* edge, int* other, int* empty) {
    *lane = *line = *edge = *other = *empty = 0;
    int map_start = EGO_FEATURES + PARTNER_FEATURES * (MAX_AGENTS - 1);
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
        int idx = map_start + k * ROAD_FEATURES;
        if (obs[idx] == 0.0f && obs[idx + 1] == 0.0f && obs[idx + 6] == 0.0f) {
            (*empty)++;
            continue;
        }
        int kind = (int)obs[idx + 6];
        if (kind == 0) (*lane)++;
        else if (kind == 1) (*line)++;
        else if (kind == 2) (*edge)++;
        else (*other)++;
    }
}

static void caption(const char* line1, const char* line2) {
    int width = MeasureText(line1, 18);
    int width2 = MeasureText(line2, 16);
    if (width2 > width) width = width2;
    DrawRectangle(8, 8, width + 20, 52, (Color){0, 0, 0, 190});
    DrawText(line1, 16, 14, 18, PUFF_WHITE);
    DrawText(line2, 16, 34, 16, (Color){210, 210, 210, 255});
}

static Camera3D ortho_camera(float cx, float cy, float fovy) {
    Camera3D camera = {0};
    camera.position = (Vector3){cx, cy, 400.0f};
    camera.target = (Vector3){cx, cy, 0.0f};
    camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    camera.projection = CAMERA_ORTHOGRAPHIC;
    camera.fovy = fovy;
    return camera;
}

static void export_texture(RenderTexture2D target, const char* path) {
    Image image = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&image);
    if (!ExportImage(image, path)) {
        fprintf(stderr, "failed to write %s\n", path);
        exit(1);
    }
    printf("wrote %s (%dx%d)\n", path, image.width, image.height);
    UnloadImage(image);
}

static void render_pair(Drive* env, int agent_slot, const char* network_path, const char* obs_path,
                        const char* title) {
    float min_x = env->map_corners[0];
    float min_y = env->map_corners[1];
    float max_x = env->map_corners[2];
    float max_y = env->map_corners[3];
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* e = &env->entities[env->active_agent_indices[i]];
        if (e->x < min_x) min_x = e->x;
        if (e->y < min_y) min_y = e->y;
        if (e->x > max_x) max_x = e->x;
        if (e->y > max_y) max_y = e->y;
        if (e->goal_position_x < min_x) min_x = e->goal_position_x;
        if (e->goal_position_y < min_y) min_y = e->goal_position_y;
        if (e->goal_position_x > max_x) max_x = e->goal_position_x;
        if (e->goal_position_y > max_y) max_y = e->goal_position_y;
    }
    float span_x = (max_x - min_x) * 1.08f;
    float span_y = (max_y - min_y) * 1.08f;
    if (span_x < 30.0f) span_x = 30.0f;
    if (span_y < 30.0f) span_y = 30.0f;
    float cx = (min_x + max_x) * 0.5f;
    float cy = (min_y + max_y) * 0.5f;

    int img_h = 900;
    int img_w = (int)(img_h * (span_x / span_y));
    if (img_w > 4096) {
        img_w = 4096;
        img_h = (int)(img_w * (span_y / span_x));
    }
    if (img_w < 960) img_w = 960;
    if (img_h < 320) img_h = 320;
    if (img_h > 1600) img_h = 1600;
    float img_aspect = (float)img_w / (float)img_h;
    float fovy = span_y;
    if (span_x / fovy > img_aspect) fovy = span_x / img_aspect;

    rlDisableBackfaceCulling();
    RenderTexture2D network = LoadRenderTexture(img_w, img_h);
    BeginTextureMode(network);
    ClearBackground(ROAD_COLOR);
    Camera3D network_cam = ortho_camera(cx, cy, fovy);
    BeginMode3D(network_cam);
    draw_network(env, agent_slot);
    EndMode3D();
    char line1[160];
    char line2[200];
    snprintf(line1, sizeof(line1), "%s   full network", title);
    snprintf(line2, sizeof(line2),
             "yellow lane centers   white road lines   gray road edges   red observed agent");
    caption(line1, line2);
    EndTextureMode();
    export_texture(network, network_path);
    UnloadRenderTexture(network);

    int edges = 0;
    int partners = 0;
    // Measure the observation before choosing the frame.
    float* agent_obs = env->agents[agent_slot].observations;
    int lane = 0, line = 0, edge_slots = 0, other = 0, empty = 0;
    count_road_slots(agent_obs, &lane, &line, &edge_slots, &other, &empty);
    float goal_x = agent_obs[0] / OBS_GOAL_SCALE;
    float goal_y = agent_obs[1] / OBS_GOAL_SCALE;
    float goal_reach = sqrtf(goal_x * goal_x + goal_y * goal_y);
    float half = goal_reach;
    if (half < VISION_RANGE * GRID_CELL_SIZE) half = VISION_RANGE * GRID_CELL_SIZE;
    half *= 1.15f;

    int obs_px = 1280;
    RenderTexture2D obs_target = LoadRenderTexture(obs_px, obs_px);
    Entity* ego = &env->entities[env->active_agent_indices[agent_slot]];
    BeginTextureMode(obs_target);
    ClearBackground(ROAD_COLOR);
    Camera3D obs_cam = ortho_camera(ego->x, ego->y, half * 2.0f);
    BeginMode3D(obs_cam);
    float drawn_reach = draw_observation(env, agent_slot, &edges, &partners);
    EndMode3D();
    snprintf(line1, sizeof(line1), "%s   agent %d observation", title, agent_slot);
    snprintf(line2, sizeof(line2),
             "drawn: %d road edges, %d partners, goal   omitted: %d lane centers, %d road lines",
             edges, partners, lane, line);
    caption(line1, line2);
    EndTextureMode();
    export_texture(obs_target, obs_path);
    UnloadRenderTexture(obs_target);

    printf("agent_slot=%d entity=%d pos=(%.1f, %.1f) heading=%.3f\n",
           agent_slot, env->active_agent_indices[agent_slot], ego->x, ego->y, ego->heading);
    printf("obs road slots: lane=%d line=%d edge=%d other=%d empty=%d drawn_edges=%d partners=%d\n",
           lane, line, edge_slots, other, empty, edges, partners);
    printf("goal_ego=(%.1f, %.1f) reach=%.1f m  drawn_reach=%.1f m  frame_diameter=%.1f m\n",
           goal_x, goal_y, goal_reach, drawn_reach, half * 2.0f);
    if (drawn_reach > half) {
        printf("note: a drawn feature sits outside the frame (reach %.1f > half %.1f)\n",
               drawn_reach, half);
    }
}

static int map_file_ok(const char* data_dir, int map_id) {
    char path[512];
    snprintf(path, sizeof(path), "%s/binaries/map_%03d.bin", data_dir, map_id);
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

static int pick_random_map(const char* data_dir) {
    srand(20261012);
    int start = rand() % 1000;
    for (int k = 0; k < 1000; k++) {
        int id = (start + k) % 1000;
        if (map_file_ok(data_dir, id)) return id;
    }
    fprintf(stderr, "no map binaries under %s/binaries\n", data_dir);
    exit(1);
}

static void load_env(Drive* env, const char* data_dir, int map_id) {
    Ini ini = {0};
    puf_ini_load_env(&ini, "drive", 0, NULL);
    char map_buf[32];
    snprintf(map_buf, sizeof(map_buf), "%d", map_id);
    puf_ini_put(&ini, "env.data_dir", data_dir);
    puf_ini_put(&ini, "env.map_id", map_buf);
    Dict* env_sec = puf_ini_section(&ini, "env", 0);
    memset(env, 0, sizeof(*env));
    puf_init(env, env_sec);

    size_t n_obs = (size_t)env->num_agents * (size_t)OBS_SIZE;
    obs_t* observations = calloc(n_obs > 0 ? n_obs : 1, sizeof(obs_t));
    float* actions = calloc((size_t)env->num_agents * NUM_ATNS + 1, sizeof(float));
    float* rewards = calloc((size_t)env->num_agents + 1, sizeof(float));
    float* terminals = calloc((size_t)env->num_agents + 1, sizeof(float));
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = observations + (size_t)i * OBS_SIZE;
        env->agents[i].actions = actions + (size_t)i * NUM_ATNS;
        env->agents[i].rewards = rewards + i;
        env->agents[i].terminals = terminals + i;
    }
    puf_reset(env);
    if (env->active_agent_count <= 0 || !env->agents[0].observations) {
        fprintf(stderr, "map %d has no active agents\n", map_id);
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <data_dir> <map_id|random> <output_dir> [agent_slot]\n", argv[0]);
        return 1;
    }
    const char* data_dir = argv[1];
    const char* out_dir = argv[3];
    if (!ensure_dir(out_dir)) {
        fprintf(stderr, "cannot create %s\n", out_dir);
        return 1;
    }

    int map_id;
    int random_map = strcmp(argv[2], "random") == 0;
    if (random_map) {
        map_id = pick_random_map(data_dir);
    } else {
        map_id = atoi(argv[2]);
    }

    Drive env = {0};
    load_env(&env, data_dir, map_id);
    int agent_slot = env.human_agent_idx;
    if (argc >= 5) agent_slot = atoi(argv[4]);
    if (agent_slot < 0 || agent_slot >= env.active_agent_count) {
        fprintf(stderr, "agent_slot %d outside 0..%d\n", agent_slot, env.active_agent_count - 1);
        return 1;
    }
    env.human_agent_idx = agent_slot;

    const char* slash = strrchr(data_dir, '/');
    const char* base = slash ? slash + 1 : data_dir;
    char stem[64];
    if (random_map || strncmp(base, "womd", 4) == 0) {
        snprintf(stem, sizeof(stem), "waymo_map_%03d", map_id);
    } else {
        snprintf(stem, sizeof(stem), "%s", base);
    }
    char network_path[512];
    char obs_path[512];
    snprintf(network_path, sizeof(network_path), "%s/%s_network.png", out_dir, stem);
    snprintf(obs_path, sizeof(obs_path), "%s/%s_observation.png", out_dir, stem);

    char title[128];
    snprintf(title, sizeof(title), "%s  map_%03d", base, map_id);
    printf("map=%s agents=%d human_slot=%d\n", env.map_name, env.active_agent_count, agent_slot);

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIDDEN);
    InitWindow(64, 64, "drive sanity");
    if (!IsWindowReady()) {
        fprintf(stderr, "raylib window failed\n");
        return 1;
    }
    render_pair(&env, agent_slot, network_path, obs_path, title);
    CloseWindow();
    return 0;
}
