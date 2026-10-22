// Viewer drawing, camera, and the on-screen controls.
// Included from drive.h after the simulation. Eval can still render.
// The step, rewards, and reset stay in drive.h.

#pragma once

#include "raylib.h"
#define float3 rl_float3
#define double3 rl_double3
#include "raymath.h"
#undef float3
#undef double3
#include "rlgl.h"

struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[2];
    int car_assignments[DRIVE_AGENT_MAX];
    Vector3 default_camera_position;
    Vector3 default_camera_target;
    // Headless record: left panel is the scene BEV, right is one agent's view.
    // The top-right inset is that agent's observation.
    RenderTexture2D record_bev;
    RenderTexture2D record_pov;
    int record_views;
    Font font;
    int font_ready;
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
        SetTargetFPS(PUF_STEPS_PER_SEC);
    } else {
        rlDisableBackfaceCulling();
    }
    client->puffers = LoadTexture("resources/shared/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/Sedan.glb");
    client->cars[1] = LoadModel("resources/drive/Compact.glb");
    for (int i = 0; i < DRIVE_AGENT_MAX; i++) {
        client->car_assignments[i] = rand_r(&env->rng) % 2;
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
    // Nimbus Sans is the Helvetica clone, but it is CFF and will not load.
    // Liberation Sans is the TrueType stand-in.
    const char* sans_paths[] = {
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    };
    for (int i = 0; i < 2; i++) {
        if (!FileExists(sans_paths[i])) {
            continue;
        }
        Font font = LoadFontEx(sans_paths[i], 64, NULL, 0);
        if (font.glyphCount <= 0) {
            continue;
        }
        SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
        client->font = font;
        client->font_ready = 1;
        break;
    }
    return client;
}

// Right-hand column: playback and follow.
enum {
    SIM_PLAY = 0,
    SIM_PAUSE = 1,
    SIM_RESET = 2,
    SIM_FOLLOW = 3,
    SIM_NBTN = 4
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
    r[SIM_FOLLOW] = (Rectangle){x, y, bw, bh};
}

static Rectangle sim_col_rect(Drive* env) {
    float width = env->client->width;
    Rectangle r[SIM_NBTN];
    sim_rects(width, r);
    float x = width - 8.0f - SIM_COL_W;
    float bot = r[SIM_FOLLOW].y + r[SIM_FOLLOW].height + 8.0f;
    return (Rectangle){x, 8.0f, SIM_COL_W, bot - 8.0f};
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
    return 0;
}

static const Color PAPER = {255, 255, 255, 255};
static const Color CARD = {255, 255, 255, 232};
static const Color WELL = {240, 242, 243, 255};
static const Color INK = {21, 23, 25, 255};
static const Color MUTED = {85, 85, 85, 255};
static const Color LINE = {215, 218, 221, 255};
static const Color GRID = {207, 212, 215, 255};
static const Color HOVER = {232, 237, 240, 255};
static const Color SAGE = {127, 192, 163, 255};
static const Color SAGE_INK = {4, 120, 87, 255};
static const Color STEEL = {92, 143, 197, 255};
static const Color STEEL_INK = {53, 94, 121, 255};
static const Color CORAL = {242, 152, 111, 255};
static const Color AMBER = {230, 186, 64, 255};
static const Color CRIMSON = {185, 28, 28, 255};

static void sim_draw_button(Rectangle r, const char* label, Color fill, Color edge) {
    DrawRectangleRec(r, fill);
    DrawRectangleLinesEx(r, 1.0f, edge);
    int tw = MeasureText(label, 16);
    int tx = (int)(r.x + (r.width - (float)tw) * 0.5f);
    DrawText(label, tx, (int)r.y + 5, 16, INK);
}

static void sim_choice(Rectangle r, const char* label, int on, int hot, int amber) {
    Color fill = WELL;
    Color edge = LINE;
    if (on && amber) {
        fill = CORAL;
        edge = CORAL;
    } else if (on) {
        fill = SAGE;
        edge = SAGE;
    } else if (hot) {
        fill = HOVER;
    }
    if (hot) {
        edge = INK;
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

// Ego meters to the crop. Forward is up, left is left.
static void sim_obs_seg(float cx, float cy, float ppm,
        float f0, float l0, float f1, float l1,
        float thick, Color color) {
    Vector2 a = {cx - l0 * ppm, cy - f0 * ppm};
    Vector2 b = {cx - l1 * ppm, cy - f1 * ppm};
    DrawLineEx(a, b, thick, color);
}

static Font sim_font(Drive* env) {
    if (env->client && env->client->font_ready) {
        return env->client->font;
    }
    return GetFontDefault();
}

static void sim_sans(Drive* env, const char* text, float x, float y,
        float size, Color color) {
    DrawTextEx(sim_font(env), text, (Vector2){x, y}, size, 0.4f, color);
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
    DrawTriangle(tip, left, right, color);
    DrawTriangle(tip, right, left, color);
}

// Box in the ego frame. (c, s) is the car's heading, forward then left.
static void sim_obs_box(float cx, float cy, float ppm,
        float f, float l, float hl, float hw,
        float c, float s, float thick, Color color) {
    float lf = -s;
    float ll = c;
    float f0 = f + hl * c + hw * lf;
    float l0 = l + hl * s + hw * ll;
    float f1 = f + hl * c - hw * lf;
    float l1 = l + hl * s - hw * ll;
    float f2 = f - hl * c - hw * lf;
    float l2 = l - hl * s - hw * ll;
    float f3 = f - hl * c + hw * lf;
    float l3 = l - hl * s + hw * ll;
    sim_obs_seg(cx, cy, ppm, f0, l0, f1, l1, thick, color);
    sim_obs_seg(cx, cy, ppm, f1, l1, f2, l2, thick, color);
    sim_obs_seg(cx, cy, ppm, f2, l2, f3, l3, thick, color);
    sim_obs_seg(cx, cy, ppm, f3, l3, f0, l0, thick, color);
}

// Decode the selected car's observation. Empty slots stay blank.
static void sim_obs_draw(Drive* env) {
    if (!env->client || puf_record_enabled()) {
        return;
    }
    Rectangle rect = sim_obs_rect(env->client->width);
    DrawRectangleRec(rect, CARD);
    DrawRectangleLinesEx(rect, 1.0f, LINE);

    int n = env->active_agent_count;
    int slot = env->selected_agent_idx;
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
    DrawText(title, (int)rect.x + 10, (int)rect.y + 8, 16, INK);

    float* obs = NULL;
    if (n > 0 && slot < DRIVE_AGENT_MAX) {
        obs = env->agents[slot].observations;
    }
    Color dim = MUTED;
    if (!obs) {
        DrawText("no view", (int)rect.x + 10, (int)rect.y + 30, 14, dim);
        return;
    }
    float spd = obs[2] / OBS_SPEED_SCALE;
    int hit = obs[5] > 0.5f;
    float gf = obs[0] / OBS_GOAL_SCALE;
    float gl = obs[1] / OBS_GOAL_SCALE;
    float gd = sqrtf(gf * gf + gl * gl);
    char cap[64];
    snprintf(cap, sizeof(cap), "v %.1f%s", spd, hit ? "  hit" : "");
    Color cap_c = hit ? CRIMSON : INK;
    DrawText(cap, (int)rect.x + 10, (int)rect.y + 28, 14, cap_c);
    char goal_txt[32];
    if (gd >= 10.0f) {
        snprintf(goal_txt, sizeof(goal_txt), "goal %.0f m", gd);
    } else {
        snprintf(goal_txt, sizeof(goal_txt), "goal %.1f m", gd);
    }
    DrawText(goal_txt, (int)rect.x + 10, (int)rect.y + 44, 14, SAGE_INK);

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
        LINE);

    int road0 = EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS;
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
            Color pt = STEEL;
            if (edge) {
                pt = CORAL;
            }
            DrawCircleV((Vector2){cx - mid_l * ppm, cy - mid_f * ppm},
                2.2f, pt);
        }
    }

    for (int j = 0; j < MAX_AGENTS; j++) {
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
        Color car = CORAL;
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
            p[6] * MAX_SPEED * 0.10f, STEEL_INK);
    }

    int outside = gd > SIM_OBS_RANGE;
    float df = gf;
    float dl = gl;
    Color goal = SAGE_INK;
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
    Color ego = hit ? CRIMSON : INK;
    sim_obs_seg(cx, cy, ppm, ego_hl, ego_hw, ego_hl, -ego_hw, 1.25f, ego);
    sim_obs_seg(cx, cy, ppm, ego_hl, -ego_hw, -ego_hl, -ego_hw, 1.25f, ego);
    sim_obs_seg(cx, cy, ppm, -ego_hl, -ego_hw, -ego_hl, ego_hw, 1.25f, ego);
    sim_obs_seg(cx, cy, ppm, -ego_hl, ego_hw, ego_hl, ego_hw, 1.25f, ego);
    // Speed over 100, fitted to the crop so the head is not clipped off.
    float head_m = 13.0f / ppm + 0.3f;
    float room = SIM_OBS_RANGE - ego_hl - head_m;
    if (room < 1.0f) {
        room = 1.0f;
    }
    float vlen = (spd / MAX_SPEED) * room;
    sim_obs_arrow(cx, cy, ppm, ego_hl, 0.0f, 1.0f, 0.0f,
        vlen, STEEL_INK);
    EndScissorMode();
    DrawRectangleLinesEx(plot, 1.0f, LINE);
}

static void sim_transport_draw(Drive* env) {
    if (!env->client) {
        return;
    }
    Rectangle r[SIM_NBTN];
    sim_rects(env->client->width, r);
    const char* labels[SIM_NBTN] = {
        "Play", "Pause", "Reset", "Follow"
    };
    int on[SIM_NBTN] = {
        !env->sim_paused,
        env->sim_paused,
        0,
        env->sim_follow
    };
    Rectangle col = sim_col_rect(env);
    DrawRectangleRec(col, CARD);
    DrawRectangleLinesEx(col, 1.0f, LINE);
    Vector2 mouse = GetMousePosition();
    Color sep = LINE;
    int y1 = (int)(r[SIM_RESET].y + r[SIM_RESET].height
        + r[SIM_FOLLOW].y) / 2;
    int x0 = (int)r[SIM_PLAY].x;
    int x1 = (int)(r[SIM_PLAY].x + r[SIM_PLAY].width);
    DrawLine(x0, y1, x1, y1, sep);
    for (int i = 0; i < SIM_NBTN; i++) {
        int hot = CheckCollisionPointRec(mouse, r[i]);
        sim_choice(r[i], labels[i], on[i], hot, i == SIM_PAUSE);
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
        return;
    }
    // Drop the one-frame hold before reading this frame's click.
    env->sim_hold_step = 0;
    if (IsKeyPressed(KEY_P)) {
        env->sim_paused = !env->sim_paused;
    }
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        return;
    }
    Vector2 mouse = GetMousePosition();
    Rectangle r[SIM_NBTN];
    sim_rects(env->client->width, r);
    if (CheckCollisionPointRec(mouse, r[SIM_PLAY])) {
        env->sim_paused = 0;
    } else if (CheckCollisionPointRec(mouse, r[SIM_PAUSE])) {
        env->sim_paused = 1;
    } else if (CheckCollisionPointRec(mouse, r[SIM_RESET])) {
        // Manual reset is not a logged episode.
        puf_reset(env);
        env->sim_hold_step = 1;
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
    if (env->active_agent_count <= 0 || !env->entities) {
        return;
    }
    int slot = env->selected_agent_idx;
    if (slot < 0 || slot >= env->active_agent_count) {
        slot = 0;
    }
    Entity* agent = &env->entities[slot];
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

    DrawTriangle3D(top, fwd, rgt, STEEL);
    DrawTriangle3D(top, rgt, bck, STEEL);
    DrawTriangle3D(top, bck, lft, STEEL);
    DrawTriangle3D(top, lft, fwd, STEEL);
    DrawTriangle3D(bot, rgt, fwd, STEEL);
    DrawTriangle3D(bot, bck, rgt, STEEL);
    DrawTriangle3D(bot, lft, bck, STEEL);
    DrawTriangle3D(bot, fwd, lft, STEEL);

    if (!IsKeyDown(KEY_LEFT_CONTROL)) return;

    if (agent_index < 0 || agent_index >= env->active_agent_count) return;
    if (!env->agents[agent_index].observations) return;
    float* agent_obs = env->agents[agent_index].observations;

    // Draw goal
    float goal_x = agent_obs[0] / OBS_GOAL_SCALE;
    float goal_y = agent_obs[1] / OBS_GOAL_SCALE;
    DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, SAGE_INK);

    // Draw partner observations
    int obs_idx = EGO_FEATURES;
    for (int j = 0; j < MAX_AGENTS; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += PARTNER_FEATURES;
            continue;
        }
        float x = agent_obs[obs_idx] / OBS_POSITION_SCALE;
        float y = agent_obs[obs_idx + 1] / OBS_POSITION_SCALE;
        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, 1}, CORAL);

        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float angle = atan2f(theta_y, theta_x);
        // Draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 7.5f;

        float ax = x + arrow_length * cosf(angle);
        float ay = y + arrow_length * sinf(angle);
        DrawLine3D((Vector3){x, y, 1}, (Vector3){ax, ay, 1}, INK);

        float arrow_size = 2.0f;
        float dx = ax - x, dy = ay - y;
        float len = sqrtf(dx * dx + dy * dy);
        if (len > 0) {
            dx /= len; dy /= len;
            float px = -dy * arrow_size, py = dx * arrow_size;
            DrawLine3D((Vector3){ax, ay, 1}, (Vector3){ax - dx * arrow_size + px, ay - dy * arrow_size + py, 1}, INK);
            DrawLine3D((Vector3){ax, ay, 1}, (Vector3){ax - dx * arrow_size - px, ay - dy * arrow_size - py, 1}, INK);
        }
        obs_idx += PARTNER_FEATURES;
    }

    // Draw road edge observations
    int map_start_idx = EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS;
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

        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_mid, y_mid, 1}, STEEL);
        DrawCube((Vector3){x_mid, y_mid, 1}, 0.5f, 0.5f, 0.5f, STEEL);
        DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, STEEL_INK);
    }
}

void draw_road_edge(Drive* env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){168, 173, 178, 255};
    Color CURB_SIDE = (Color){120, 125, 130, 255};
    Color CURB_BOTTOM = (Color){90, 95, 100, 255};
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

static void draw_world(Drive* env, Client* client);
static void puf_render_hud(Drive* env, Client* client);

static int selected_slot(Drive* env) {
    int slot = env->selected_agent_idx;
    if (slot < 0 || slot >= env->active_agent_count) {
        slot = 0;
    }
    return slot;
}

// Orthographic top-down of the whole map. fovy is the vertical span;
// widen it when the map is wider than the panel.
static Camera3D bev_camera(Drive* env, float aspect) {
    float x0 = -40.0f;
    float y0 = -40.0f;
    float x1 = 40.0f;
    float y1 = 40.0f;
    if (env->map_corners) {
        x0 = env->map_corners[0];
        y0 = env->map_corners[1];
        x1 = env->map_corners[2];
        y1 = env->map_corners[3];
    }
    float map_w = fabsf(x1 - x0);
    float map_h = fabsf(y1 - y0);
    if (map_w < 1.0f) {
        map_w = 1.0f;
    }
    if (map_h < 1.0f) {
        map_h = 1.0f;
    }
    if (aspect < 0.05f) {
        aspect = 0.05f;
    }
    float fovy = map_h;
    if (map_w / aspect > fovy) {
        fovy = map_w / aspect;
    }
    Camera3D cam = {0};
    cam.position = (Vector3){0.5f * (x0 + x1), 0.5f * (y0 + y1), 400.0f};
    cam.target = (Vector3){cam.position.x, cam.position.y, 0.0f};
    cam.up = (Vector3){0.0f, -1.0f, 0.0f};
    cam.fovy = fovy * 1.06f;
    cam.projection = CAMERA_ORTHOGRAPHIC;
    return cam;
}

// Chase the selected agent from behind and above. Same pose as the
// interactive space-bar view.
static Camera3D pov_camera(Drive* env) {
    Camera3D cam = {0};
    cam.up = (Vector3){0.0f, 0.0f, 1.0f};
    cam.fovy = 60.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    cam.position = (Vector3){0.0f, -25.0f, 15.0f};
    cam.target = (Vector3){0.0f, 40.0f, 1.0f};
    if (env->active_agent_count <= 0 || !env->entities) {
        return cam;
    }
    Entity* agent = &env->entities[selected_slot(env)];
    if (agent->x == INVALID_POSITION || agent->y == INVALID_POSITION) {
        return cam;
    }
    float hx = cosf(agent->heading);
    float hy = sinf(agent->heading);
    cam.position = (Vector3){
        agent->x - 25.0f * hx,
        agent->y - 25.0f * hy,
        15.0f
    };
    cam.target = (Vector3){
        agent->x + 40.0f * hx,
        agent->y + 40.0f * hy,
        1.0f
    };
    return cam;
}

// Top-right of a recorded frame. The selected agent's observation, in
// that car's frame. Forward is up and left is left. The crop is a fixed
// 20 m half-width centered on the car, so it does not zoom. Lane points
// are muted, edges are amber. Road lines are not drawn. The blue box and
// the arrow are the ego car. Partner boxes are the cars in view.
static Rectangle record_agent_view(Drive* env) {
    const float box_w = 300.0f;
    const float box_h = 348.0f;
    float screen_w = 1280.0f;
    if (env->client && env->client->width > box_w + 24.0f) {
        screen_w = env->client->width;
    }
    Rectangle rect = {screen_w - 12.0f - box_w, 12.0f, box_w, box_h};
    DrawRectangleRec(rect, CARD);
    DrawRectangleLinesEx(rect, 1.0f, LINE);

    int slot = 0;
    if (env->active_agent_count > 0) {
        slot = selected_slot(env);
    }
    float* obs = NULL;
    if (env->active_agent_count > 0 && slot < DRIVE_AGENT_MAX) {
        obs = env->agents[slot].observations;
    }
    if (!obs) {
        sim_sans(env, "no view", rect.x + 10.0f, rect.y + 12.0f, 16, MUTED);
        return rect;
    }

    float spd = obs[2] / OBS_SPEED_SCALE;
    int hit = obs[5] > 0.5f;
    float gf = obs[0] / OBS_GOAL_SCALE;
    float gl = obs[1] / OBS_GOAL_SCALE;
    float gd = sqrtf(gf * gf + gl * gl);
    char cap[64];
    snprintf(cap, sizeof(cap), "v %.1f%s", spd, hit ? "  hit" : "");
    sim_sans(env, cap, rect.x + 10.0f, rect.y + 8.0f, 16,
        hit ? CRIMSON : INK);
    char goal_txt[64];
    if (gd >= 10.0f) {
        snprintf(goal_txt, sizeof(goal_txt),
            "distance to next destination %.0f m", gd);
    } else {
        snprintf(goal_txt, sizeof(goal_txt),
            "distance to next destination %.1f m", gd);
    }
    sim_sans(env, goal_txt, rect.x + 10.0f, rect.y + 26.0f, 13, SAGE_INK);
    char step_txt[32];
    snprintf(step_txt, sizeof(step_txt), "t %d", env->timestep);
    sim_sans(env, step_txt, rect.x + 10.0f, rect.y + 44.0f, 16, INK);

    int road0 = EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS;
    const float range = 20.0f;

    Rectangle plot = {
        rect.x + 8.0f, rect.y + 66.0f,
        rect.width - 16.0f, rect.height - 74.0f
    };
    float side = plot.width;
    if (plot.height < side) {
        side = plot.height;
    }
    float ppm = (side * 0.5f) / range;
    float cx = plot.x + plot.width * 0.5f;
    float cy = plot.y + plot.height * 0.5f;

    BeginScissorMode((int)plot.x, (int)plot.y,
        (int)plot.width, (int)plot.height);
    for (int pass = 0; pass < 2; pass++) {
        for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
            float* road = obs + road0 + k * ROAD_FEATURES;
            if (road[0] == 0.0f && road[1] == 0.0f && road[2] == 0.0f) {
                continue;
            }
            int kind = (int)(road[6] + 0.5f);
            int edge = kind >= 2;
            if (edge != pass) {
                continue;
            }
            if (kind == 1) {
                continue;
            }
            float mid_f = road[0] / OBS_POSITION_SCALE;
            float mid_l = road[1] / OBS_POSITION_SCALE;
            Color col = MUTED;
            if (edge) {
                col = AMBER;
            }
            DrawCircleV((Vector2){cx - mid_l * ppm, cy - mid_f * ppm},
                3.4f, col);
        }
    }
    for (int j = 0; j < MAX_AGENTS; j++) {
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
        sim_obs_box(cx, cy, ppm, f, l, hl, hw,
            cosf(ang), sinf(ang), 1.6f, CORAL);
    }
    float ego_hw = obs[3] * MAX_VEH_WIDTH * 0.5f;
    float ego_hl = obs[4] * MAX_VEH_LEN * 0.5f;
    if (ego_hw < 0.4f) {
        ego_hw = 0.4f;
    }
    if (ego_hl < 0.6f) {
        ego_hl = 0.6f;
    }
    sim_obs_box(cx, cy, ppm, 0.0f, 0.0f, ego_hl, ego_hw,
        1.0f, 0.0f, 2.0f, STEEL);
    sim_obs_arrow(cx, cy, ppm, 0.0f, 0.0f, 1.0f, 0.0f,
        22.0f / ppm, STEEL_INK);
    EndScissorMode();
    DrawRectangleLinesEx(plot, 1.0f, LINE);
    return rect;
}

// One framebuffer: scene BEV on the left, the selected agent's view on
// the right. Each panel is its own target so the projection aspect matches
// the panel instead of the full window. The top-right inset is the
// observation, drawn after the panels so it sits on the chase view.
static void record_views(Drive* env) {
    Client* client = env->client;
    int w = (int)client->width;
    int h = (int)client->height;
    if (w < 2 || h < 2) {
        return;
    }
    int left_w = w / 2;
    int right_w = w - left_w;
    if (!client->record_views) {
        client->record_bev = LoadRenderTexture(left_w, h);
        client->record_pov = LoadRenderTexture(right_w, h);
        client->record_views = 1;
    }
    rlSetClipPlanes(0.05, 8000.0);
    BeginTextureMode(client->record_bev);
    ClearBackground(PAPER);
    BeginMode3D(bev_camera(env, (float)left_w / (float)h));
    draw_world(env, client);
    EndMode3D();
    EndTextureMode();

    BeginTextureMode(client->record_pov);
    ClearBackground(PAPER);
    BeginMode3D(pov_camera(env));
    draw_world(env, client);
    EndMode3D();
    EndTextureMode();

    BeginDrawing();
    ClearBackground(PAPER);
    DrawTexturePro(client->record_bev.texture,
        (Rectangle){0, 0, (float)left_w, (float)-h},
        (Rectangle){0, 0, (float)left_w, (float)h},
        (Vector2){0, 0}, 0.0f, WHITE);
    DrawTexturePro(client->record_pov.texture,
        (Rectangle){0, 0, (float)right_w, (float)-h},
        (Rectangle){(float)left_w, 0, (float)right_w, (float)h},
        (Vector2){0, 0}, 0.0f, WHITE);
    DrawRectangle(left_w - 1, 0, 2, h, INK);
    Rectangle obs_box = record_agent_view(env);
    sim_sans(env, "Full simulator state", 16.0f, 14.0f, 20, INK);
    const char* who = TextFormat("Selected agent %d", selected_slot(env));
    Vector2 who_sz = MeasureTextEx(sim_font(env), who, 20, 0.4f);
    sim_sans(env, who, obs_box.x - who_sz.x - 12.0f, 14.0f, 20, INK);
    EndDrawing();
    puf_record_capture(w, h);
}

static void draw_goal_disc(float x, float y, float radius) {
    const int n = 48;
    const float z = 0.3f;
    Color fill = SAGE;
    fill.a = 96;
    Vector3 c = {x, y, z};
    float inner = radius - 0.25f;
    if (inner < 0.0f) {
        inner = 0.0f;
    }
    for (int k = 0; k < n; k++) {
        float a0 = (float)k * 2.0f * PI / (float)n;
        float a1 = (float)(k + 1) * 2.0f * PI / (float)n;
        float c0 = cosf(a0);
        float s0 = sinf(a0);
        float c1 = cosf(a1);
        float s1 = sinf(a1);
        Vector3 p0 = {x + c0 * radius, y + s0 * radius, z};
        Vector3 p1 = {x + c1 * radius, y + s1 * radius, z};
        Vector3 q0 = {x + c0 * inner, y + s0 * inner, z};
        Vector3 q1 = {x + c1 * inner, y + s1 * inner, z};
        DrawTriangle3D(c, p0, p1, fill);
        DrawTriangle3D(q0, p0, p1, SAGE_INK);
        DrawTriangle3D(q0, p1, q1, SAGE_INK);
    }
}

static void draw_goal_markers(Drive* env) {
    if (IsKeyDown(KEY_LEFT_CONTROL)) {
        return;
    }
    rlDisableDepthTest();
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity* e = &env->entities[i];
        if (e->type != VEHICLE || e->valid == 0) {
            continue;
        }
        if (e->goal_position_x == INVALID_POSITION
            || e->goal_position_y == INVALID_POSITION) {
            continue;
        }
        draw_goal_disc(e->goal_position_x, e->goal_position_y,
            MIN_DISTANCE_TO_GOAL);
    }
    rlEnableDepthTest();
}

void puf_render(Drive* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    if (puf_record_enabled()) {
        record_views(env);
        return;
    }
    human_controls(env);
    Client* client = env->client;
    BeginDrawing();
    ClearBackground(PAPER);
    handle_camera_controls(env);
    sim_follow_camera(env);
    BeginMode3D(client->camera);
    draw_world(env, client);
    EndMode3D();
    puf_render_hud(env, client);
}

static void draw_world(Drive* env, Client* client) {
    if (env->map_corners) {
        DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[1], 0}, STEEL);
        DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[0], env->map_corners[3], 0}, STEEL);
        DrawLine3D((Vector3){env->map_corners[2], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, STEEL);
        DrawLine3D((Vector3){env->map_corners[0], env->map_corners[3], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, STEEL);
    }

    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type < ROAD_LANE
            || env->entities[i].type > ROAD_EDGE) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            float ax = env->entities[i].pts_x[j];
            float ay = env->entities[i].pts_y[j];
            float bx = env->entities[i].pts_x[j + 1];
            float by = env->entities[i].pts_y[j + 1];
            if (env->entities[i].type == ROAD_EDGE) {
                if (!IsKeyDown(KEY_LEFT_CONTROL)) {
                    draw_road_edge(env, ax, ay, bx, by);
                }
                continue;
            }
            Color line = (Color){120, 120, 120, 255};
            if (env->entities[i].type == ROAD_LANE) {
                line = (Color){180, 180, 180, 255};
            }
            DrawLine3D(
                (Vector3){ax, ay, 0.05f},
                (Vector3){bx, by, 0.05f},
                line);
        }
    }

    // One line per cell edge, under the lane marks. A wire box on every
    // cell sat on the cars and took a draw call per cell.
    if (!puf_record_enabled() && env->map_corners) {
        float x0 = env->map_corners[0];
        float y0 = env->map_corners[1];
        float x1 = env->map_corners[2];
        float y1 = env->map_corners[3];
        float z = 0.02f;
        for (int i = 0; i <= env->grid_cols; i++) {
            float x = x0 + i * GRID_CELL_SIZE;
            DrawLine3D((Vector3){x, y0, z}, (Vector3){x, y1, z}, GRID);
        }
        for (int j = 0; j <= env->grid_rows; j++) {
            float y = y0 + j * GRID_CELL_SIZE;
            DrawLine3D((Vector3){x0, y, z}, (Vector3){x1, y, z}, GRID);
        }
    }
    draw_goal_markers(env);

    for (int i = 0; i < env->num_entities; i++) {
        // Draw vehicles
        if (i < env->active_agent_count && env->entities[i].type == VEHICLE) {
            int agent_index = i;
            if (env->entities[i].x == INVALID_POSITION
                || env->entities[i].y == INVALID_POSITION) {
                continue;
            }

            Vector3 position = {env->entities[i].x, env->entities[i].y, 1};
            float heading = env->entities[i].heading;
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            rlPushMatrix();
            rlTranslatef(position.x, position.y, position.z);
            rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);

            Model car_model = client->cars[client->car_assignments[i]];
            Color car_tint = WHITE;
            if (env->entities[i].collision_state > NO_COLLISION) {
                car_tint = CRIMSON;
            } else if (puf_record_enabled()
                    && agent_index == env->selected_agent_idx) {
                car_tint = STEEL;
            }

            // The marker floats 8 m above the car. Skip it in the chase shot
            // so it does not fill the zoomed frame.
            if (!puf_record_enabled()
                && agent_index == env->selected_agent_idx
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
            DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, car_tint);
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
                DrawLine3D(corners[j], corners[(j + 1) % 4], STEEL_INK);
            }
            if (puf_record_enabled() && agent_index == env->selected_agent_idx) {
                DrawCircle3D((Vector3){position.x, position.y, 0.35f},
                    2.4f, (Vector3){0.0f, 0.0f, 1.0f}, 90.0f, STEEL_INK);
            }

            // FPV camera
            if (IsKeyDown(KEY_SPACE) && env->selected_agent_idx == agent_index) {
                if (env->entities[agent_index].reached_goal) {
                    env->selected_agent_idx = rand_r(&env->rng) % env->active_agent_count;
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

        }
    }
}

static void puf_render_hud(Drive* env, Client* client) {
    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)",
        client->camera.position.x, client->camera.position.y, client->camera.position.z), 10, 10, 20, INK);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)",
        client->camera.target.x, client->camera.target.y, client->camera.target.z), 10, 30, 20, INK);
    const char* paused_mark = "";
    if (!puf_record_enabled() && env->sim_paused) {
        paused_mark = "   paused";
    }
    DrawText(TextFormat("Time: %.1f s%s", env->timestep * SIM_DT, paused_mark),
        10, 50, 20, INK);
    if (!puf_record_enabled() && !env->checkpoint_ok) {
        DrawText("Checkpoint cannot be loaded, showing random behavior",
            10, 190, 20, CRIMSON);
    }
    int human_idx = -1;
    int accel = 0;
    int steer = 0;
    if (env->active_agent_count > 0
        && env->selected_agent_idx >= 0
        && env->selected_agent_idx < env->active_agent_count) {
        human_idx = env->selected_agent_idx;
        float* atn = env->agents[env->selected_agent_idx].actions;
        if (atn) {
            accel = (int)atn[0];
            steer = (int)atn[1];
        }
    }
    DrawText(TextFormat("Controlling Agent: %d", env->selected_agent_idx), 10, 70, 20, INK);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, INK);
    if (!puf_record_enabled()) {
        sim_transport_draw(env);
    }
    DrawText(TextFormat("Acceleration: %d", accel), 10, 110, 20, INK);
    DrawText(TextFormat("Steering: %d", steer), 10, 130, 20, INK);
    DrawText(TextFormat("Grid Rows: %d", env->grid_rows), 10, 150, 20, INK);
    DrawText(TextFormat("Grid Cols: %d", env->grid_cols), 10, 170, 20, INK);
    EndDrawing();
    if (puf_record_enabled()) {
        puf_record_capture((int)client->width, (int)client->height);
    }
    puf_web_vsync();
}

void close_client(Client* client) {
    if (client->record_views) {
        UnloadRenderTexture(client->record_bev);
        UnloadRenderTexture(client->record_pov);
        client->record_views = 0;
    }
    for (int i = 0; i < 2; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    if (client->font_ready) {
        UnloadFont(client->font);
    }
    CloseWindow();
    free(client);
}

