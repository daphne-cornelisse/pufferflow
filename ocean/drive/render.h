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

// Top-down crop, matching the highway GIFs: ±40 m on the short side.
// Recordings sit a little closer, and at a higher resolution.
#define SIM_CROP 80.0f
#define SIM_RECORD_ZOOM 0.70f
#define SIM_TRACE 512
// One square sheet per recording. DRIVE_RECORD_WHICH picks which car.
#define SIM_RECORD_AGENTS 5
#define SIM_RECORD_SIDE 960
#define SIM_RECORD_MARGIN 44
// The sheet is drawn at this multiple and scaled down, which is the
// antialiasing the one-pixel strokes never had.
#define SIM_RECORD_SS 2
static RenderTexture2D sheet_hi = {0};
static int sheet_hi_side = 0;

static void sheet_hi_ensure(int side) {
    if (sheet_hi.id != 0 && sheet_hi_side == side) {
        return;
    }
    if (sheet_hi.id != 0) {
        UnloadRenderTexture(sheet_hi);
    }
    sheet_hi = LoadRenderTexture(side, side);
    SetTextureFilter(sheet_hi.texture, TEXTURE_FILTER_BILINEAR);
    sheet_hi_side = side;
}
static int sheet_origin_x = 0;
static int sheet_origin_y = 0;

struct Client {
    float width;
    float height;
    Vector3 camera_target;
    float camera_zoom;
    // Path of the followed car. A new sample clears it.
    float trace_x[SIM_TRACE];
    float trace_y[SIM_TRACE];
    int trace_n;
    int trace_i;
    int trace_agent;
    int trace_stamp;
    int trace_ready;
    // Recording follows this many policy cars, one square each.
    int record_n;
    int record_slot[SIM_RECORD_AGENTS];
    int record_trace_n[SIM_RECORD_AGENTS];
    int record_trace_i[SIM_RECORD_AGENTS];
    float record_trace_x[SIM_RECORD_AGENTS][SIM_TRACE];
    float record_trace_y[SIM_RECORD_AGENTS][SIM_TRACE];
    unsigned int pick_rng;
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

// Street view for a town. A highway longer than 700 m is framed
// from farther back so the whole roadway fits.
static void sim_camera_distance(Drive* env, float* back, float* up) {
    float cam_back = 120.0f;
    float cam_up = 175.0f;
    float span = 0.0f;
    if (env->sim_user_size) {
        cam_back = 48.0f;
        cam_up = 72.0f;
    } else if (env->map_corners) {
        float span_x = env->map_corners[2] - env->map_corners[0];
        float span_y = env->map_corners[3] - env->map_corners[1];
        span = span_x > span_y ? span_x : span_y;
        if (span > 700.0f) {
            float dist = span * 0.85f;
            cam_back = dist * 0.50f;
            cam_up = dist * 0.90f;
        }
    }
    *back = cam_back;
    *up = cam_up;
}

static void sim_frame_camera(Drive* env) {
    Client* client = env->client;
    float cx = 0.0f;
    float cy = 0.0f;
    float back = 120.0f;
    float up = 175.0f;
    if (!client || !env->map_corners) {
        return;
    }
    cx = (env->map_corners[0] + env->map_corners[2]) * 0.5f;
    cy = (env->map_corners[1] + env->map_corners[3]) * 0.5f;
    sim_camera_distance(env, &back, &up);
    client->default_camera_target = (Vector3){cx, cy, 1.0f};
    client->default_camera_position = (Vector3){cx, cy + back, up};
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera.target = client->default_camera_target;
    client->camera.position = client->default_camera_position;
    if (env->sim_follow) {
        back = 120.0f;
        up = 175.0f;
        if (env->sim_user_size) {
            back = 48.0f;
            up = 72.0f;
        }
        client->camera.position = (Vector3){cx, cy + back, up};
    }
}

Client* make_client(Drive* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->trace_agent = -1;
    client->pick_rng = env->rng ^ 0x9E3779B9u;
    if (client->pick_rng == 0) {
        client->pick_rng = 1;
    }
    client->width = 1280;
    client->height = 704;
    if (env->sim_user_size) {
        client->height = 880;
    }
    int recording = puf_record_enabled();
    if (recording) {
        // Even axes for yuv420. Three squares across, BC contact width.
        client->width = SIM_RECORD_SIDE + SIM_RECORD_MARGIN * 2;
        client->height = client->width;
        puf_record_ensure_display((int)client->width, (int)client->height);
    }
    unsigned int flags = FLAG_MSAA_4X_HINT;
    if (recording) {
        flags |= FLAG_WINDOW_HIDDEN;
    }
    if (env->sim_user_size) {
        flags |= FLAG_WINDOW_RESIZABLE;
    }
    SetConfigFlags(flags);
    InitWindow(client->width, client->height, "Pufferflow");
    if (!recording) {
        if (env->sim_user_size) {
            SetWindowMinSize(1080, 720);
        }
        int mw = GetMonitorWidth(GetCurrentMonitor());
        int mh = GetMonitorHeight(GetCurrentMonitor());
        if (!env->sim_user_size && mw >= 1400 && mh >= 900) {
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
    client->cars[0] = LoadModel("resources/drive/Sedan.glb");
    client->cars[1] = LoadModel("resources/drive/Compact.glb");
    for (int i = 0; i < DRIVE_AGENT_MAX; i++) {
        client->car_assignments[i] = rand_r(&env->rng) % 2;
    }
    env->client = client;
    client->camera_zoom = 1.0f;
    sim_frame_camera(env);
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

static Rectangle sim_scene_rect(Drive* env) {
    float width = env->client->width;
    float col_x = width - 8.0f - SIM_COL_W;
    float x = col_x + 8.0f;
    float bw = SIM_COL_W - 16.0f;
    return (Rectangle){x, 16.0f, bw, 26.0f};
}

static void sim_rects(Drive* env, Rectangle r[SIM_NBTN]) {
    float width = env->client->width;
    float col_x = width - 8.0f - SIM_COL_W;
    float x = col_x + 8.0f;
    float bw = SIM_COL_W - 16.0f;
    float bh = 26.0f;
    float y = 16.0f;
    if (env->sim_scene) {
        y += bh + 6.0f;
    }
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
    sim_rects(env, r);
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
    if (env->ui_hot && env->ui_hot(env)) {
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
// Lane markings. Kind 1 in the observation, distinct from center lanes.
static const Color ROAD_LINE_PT = {126, 110, 176, 255};
// Highway sheet. Policy cars are dark blue, the followed car is lighter,
// and every log-replayed car, including ghosts, is grey.
static const Color POLICY_BLUE = {27, 79, 140, 255};
static const Color EGO_BLUE = {126, 178, 222, 255};
static const Color REPLAY_GREY = {176, 176, 176, 255};
// Matplotlib sheet: black box edges, grey lanes, yellow markings, black edges.
static const Color BOX_EDGE = {34, 34, 34, 255};
// A vehicle overlap is red. A road-edge overlap is orange.
static const Color HIT_RED = {214, 39, 40, 255};
static const Color HIT_RED_EDGE = {140, 18, 18, 255};
static const Color OFFROAD_ORANGE = {255, 127, 14, 255};
static const Color OFFROAD_ORANGE_EDGE = {176, 78, 0, 255};
static const Color LANE_YELLOW = {230, 195, 92, 255};
static const Color LANE_GREY = {176, 182, 189, 255};
static const Color EDGE_GREY = {34, 34, 34, 255};
// Logged pose of a policy car, drawn under the controlled box.
static const Color GHOST_GREEN = {176, 222, 190, 255};
static const Color GHOST_GREEN_EDGE = {118, 176, 140, 255};

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

// Signed speed within speed_band of the target. A target of 0 never matches.
static int obs_speed_in_band(Drive* env, int slot) {
    if (env->target_speed <= 0.0f || slot < 0
            || slot >= env->active_agent_count) {
        return 0;
    }
    Entity* ego = &env->entities[slot];
    float signed_speed = ego->vx * ego->heading_x + ego->vy * ego->heading_y;
    float err = signed_speed - env->target_speed;
    if (err < 0.0f) {
        err = -err;
    }
    return err <= env->speed_band;
}

// Road feature 6 is type minus ROAD_LANE: 0 center lane, 1 line, 2+ edge.
// Lines draw first, then center lanes, then edges.
static int obs_road_layer(int kind) {
    if (kind == 1) {
        return 0;
    }
    if (kind == 0) {
        return 1;
    }
    if (kind >= 2) {
        return 2;
    }
    return -1;
}

static Color obs_road_color(int kind, int recorded) {
    if (kind == 1) {
        return ROAD_LINE_PT;
    }
    if (kind >= 2) {
        return recorded ? AMBER : CORAL;
    }
    return recorded ? MUTED : STEEL;
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
    float target = obs[EGO_TARGET_SPEED] / OBS_SPEED_SCALE;
    char target_txt[64];
    snprintf(target_txt, sizeof(target_txt), "target %.1f", target);
    Color target_c = obs_speed_in_band(env, slot) ? SAGE_INK : CRIMSON;
    DrawText(target_txt, (int)rect.x + 10, (int)rect.y + 44, 14, target_c);
    char goal_txt[32];
    if (gd >= 10.0f) {
        snprintf(goal_txt, sizeof(goal_txt), "goal %.0f m", gd);
    } else {
        snprintf(goal_txt, sizeof(goal_txt), "goal %.1f m", gd);
    }
    DrawText(goal_txt, (int)rect.x + 10, (int)rect.y + 60, 14, SAGE_INK);

    // The critic's value sits under the crop.
    const float value_foot = 20.0f;
    Rectangle plot = {
        rect.x + 10.0f, rect.y + 78.0f,
        rect.width - 20.0f, rect.height - 88.0f - value_foot
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
    for (int pass = 0; pass < 3; pass++) {
        for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {
            float* road = obs + road0 + k * ROAD_FEATURES;
            if (road[0] == 0.0f && road[1] == 0.0f && road[2] == 0.0f) {
                continue;
            }
            int kind = (int)(road[6] + 0.5f);
            if (obs_road_layer(kind) != pass) {
                continue;
            }
            float mid_f = road[0] / OBS_POSITION_SCALE;
            float mid_l = road[1] / OBS_POSITION_SCALE;
            DrawCircleV((Vector2){cx - mid_l * ppm, cy - mid_f * ppm},
                2.2f, obs_road_color(kind, 0));
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
    char value_txt[32];
    if (env->predicted_ready) {
        snprintf(value_txt, sizeof(value_txt), "value %.2f", env->predicted_value);
    } else {
        snprintf(value_txt, sizeof(value_txt), "value --");
    }
    DrawText(value_txt, (int)rect.x + 10,
        (int)(plot.y + plot.height + 3.0f), 14, INK);
}

static void sim_transport_draw(Drive* env) {
    if (!env->client) {
        return;
    }
    Rectangle r[SIM_NBTN];
    sim_rects(env, r);
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
    if (env->sim_scene) {
        Rectangle scene = sim_scene_rect(env);
        int scene_hot = CheckCollisionPointRec(mouse, scene);
        sim_choice(scene, "New scene", 0, scene_hot, 0);
    }
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
    sim_rects(env, r);
    if (env->sim_scene) {
        Rectangle scene = sim_scene_rect(env);
        if (CheckCollisionPointRec(mouse, scene)) {
            env->sim_scene(env);
            env->sim_hold_step = 1;
            return;
        }
    }
    if (CheckCollisionPointRec(mouse, r[SIM_PLAY])) {
        env->sim_paused = 0;
    } else if (CheckCollisionPointRec(mouse, r[SIM_PAUSE])) {
        env->sim_paused = 1;
    } else if (CheckCollisionPointRec(mouse, r[SIM_RESET])) {
        // Manual reset is not a logged episode.
        if (env->sim_reset) {
            env->sim_reset(env);
        } else {
            puf_reset(env);
        }
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

    // Handle mouse wheel for zoom. Follow keeps an orthographic span.
    float wheel = GetMouseWheelMove();
    if (wheel != 0 && !sim_ui_hot(env) && env->sim_follow) {
        client->camera_zoom *= 1.0f - wheel * 0.1f;
        if (client->camera_zoom < 0.25f) {
            client->camera_zoom = 0.25f;
        }
        if (client->camera_zoom > 6.0f) {
            client->camera_zoom = 6.0f;
        }
        return;
    }
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

static int sim_body_drawn(Drive* env, int slot) {
    if (!env->entities || slot < 0 || slot >= env->body_count) {
        return 0;
    }
    Entity* agent = &env->entities[slot];
    if (agent->type != VEHICLE && agent->type != TRUCK) {
        return 0;
    }
    if (agent->valid == 0) {
        return 0;
    }
    if (agent->x == INVALID_POSITION || agent->y == INVALID_POSITION) {
        return 0;
    }
    return 1;
}

static int sim_agent_live(Drive* env, int slot) {
    if (slot >= env->active_agent_count) {
        return 0;
    }
    return sim_body_drawn(env, slot);
}

// The off-road flag stays clear when a car drives past the last road
// edge. A car more than 30 m from every road point is off the sheet.
static int sim_near_road(Drive* env, int slot) {
    Entity* car = &env->entities[slot];
    float max2 = 30.0f * 30.0f;
    for (int i = 0; i < env->num_entities; i++) {
        Entity* road = &env->entities[i];
        if (road->type < ROAD_LANE || road->type > ROAD_EDGE) {
            continue;
        }
        if (!road->pts_x || !road->pts_y || road->array_size < 1) {
            continue;
        }
        for (int j = 0; j < road->array_size; j += 3) {
            float dx = road->pts_x[j] - car->x;
            float dy = road->pts_y[j] - car->y;
            if (dx * dx + dy * dy <= max2) {
                return 1;
            }
        }
    }
    return 0;
}

// Leaving the roadway does not clear valid or the position, so a
// departed car would stay selected and hide the rest of the fleet.
static int sim_agent_onroad(Drive* env, int slot) {
    if (!sim_agent_live(env, slot)) {
        return 0;
    }
    if (env->entities[slot].offroad_state != 0) {
        return 0;
    }
    return sim_near_road(env, slot);
}

// Policy cars still on the road, then a log car, then a departed
// policy car. The draw rng is not the simulation rng. A recording
// passes allow_replay 0 so the sheet never follows a log car.
static int sim_sample_agent(Drive* env, int allow_replay) {
    int live = 0;
    int onroad = 0;
    int replay = 0;
    int n = env->active_agent_count;
    for (int i = 0; i < n; i++) {
        if (!sim_agent_live(env, i)) {
            continue;
        }
        live++;
        if (sim_agent_onroad(env, i)) {
            onroad++;
        }
    }
    for (int i = n; i < env->body_count; i++) {
        if (sim_body_drawn(env, i) && sim_near_road(env, i)) {
            replay++;
        }
    }
    int mode = onroad > 0 ? 0 : ((allow_replay && replay > 0) ? 1 : 2);
    int pool = mode == 0 ? onroad : (mode == 1 ? replay : live);
    if (pool <= 0) {
        return 0;
    }
    int pick = (int)(rand_r(&env->client->pick_rng) % (unsigned)pool);
    int begin = mode == 1 ? n : 0;
    int end = mode == 1 ? env->body_count : n;
    for (int i = begin; i < end; i++) {
        if (!sim_body_drawn(env, i)) {
            continue;
        }
        if (mode == 0 && !sim_agent_onroad(env, i)) {
            continue;
        }
        if (mode == 1 && !sim_near_road(env, i)) {
            continue;
        }
        if (pick == 0) {
            return i;
        }
        pick--;
    }
    return 0;
}

static void sim_trace_push(Client* client, float x, float y) {
    int i = client->trace_i;
    client->trace_x[i] = x;
    client->trace_y[i] = y;
    client->trace_i = (i + 1) % SIM_TRACE;
    if (client->trace_n < SIM_TRACE) {
        client->trace_n++;
    }
}

static void sim_trace_clear(Client* client, int slot) {
    client->trace_n = 0;
    client->trace_i = 0;
    client->trace_agent = slot;
}

// On-road policy cars for a recording. DRIVE_RECORD_WHICH picks one.
static void sim_record_pick(Drive* env) {
    Client* client = env->client;
    int pool[DRIVE_AGENT_MAX];
    int n = 0;
    int active = env->active_agent_count;
    if (active > DRIVE_AGENT_MAX) {
        active = DRIVE_AGENT_MAX;
    }
    for (int i = 0; i < active; i++) {
        if (sim_agent_onroad(env, i)) {
            pool[n++] = i;
        }
    }
    if (n < SIM_RECORD_AGENTS) {
        for (int i = 0; i < active; i++) {
            if (!sim_agent_live(env, i)) {
                continue;
            }
            int seen = 0;
            for (int j = 0; j < n; j++) {
                if (pool[j] == i) {
                    seen = 1;
                }
            }
            if (!seen) {
                pool[n++] = i;
            }
        }
    }
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rand_r(&client->pick_rng) % (unsigned)(i + 1));
        int tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
    }
    int take = n < SIM_RECORD_AGENTS ? n : SIM_RECORD_AGENTS;
    int which = 0;
    const char* which_env = getenv("DRIVE_RECORD_WHICH");
    if (which_env && which_env[0]) {
        which = atoi(which_env);
    }
    if (which < 0) {
        which = 0;
    }
    if (which >= take) {
        which = take - 1;
    }
    client->record_n = take > 0 ? 1 : 0;
    if (take > 0) {
        client->record_slot[0] = pool[which];
        client->record_trace_n[0] = 0;
        client->record_trace_i[0] = 0;
        env->selected_agent_idx = pool[which];
        printf("record agent %d\n", pool[which]);
        fflush(stdout);
    }
}

static void sim_record_push(Client* client, int k, float x, float y) {
    int i = client->record_trace_i[k];
    client->record_trace_x[k][i] = x;
    client->record_trace_y[k][i] = y;
    client->record_trace_i[k] = (i + 1) % SIM_TRACE;
    if (client->record_trace_n[k] < SIM_TRACE) {
        client->record_trace_n[k]++;
    }
}

static void sim_record_install(Client* client, int k) {
    client->trace_n = client->record_trace_n[k];
    client->trace_i = client->record_trace_i[k];
    client->trace_agent = client->record_slot[k];
    memcpy(client->trace_x, client->record_trace_x[k], sizeof(client->trace_x));
    memcpy(client->trace_y, client->record_trace_y[k], sizeof(client->trace_y));
}

// On a log map, follow a policy car that is still on the road. Once
// those are gone, follow a log car so the sheet stays on the highway.
// A recording follows three policy cars until each one arrives.
// The path starts over when the followed car changes.
static void sim_choose_ego(Drive* env) {
    Client* client = env->client;
    if (!client) {
        return;
    }
    int seen = client->trace_stamp;
    int ready = client->trace_ready;
    int reset = ready && env->timestep < seen;
    if (puf_record_enabled() && env->log_spawn) {
        if (client->record_n <= 0 || reset) {
            sim_record_pick(env);
        }
        for (int k = 0; k < client->record_n; k++) {
            int slot = client->record_slot[k];
            if (!sim_body_drawn(env, slot)) {
                continue;
            }
            int same = client->record_trace_n[k] > 0 && ready
                && env->timestep == seen;
            if (same) {
                continue;
            }
            Entity* agent = &env->entities[slot];
            sim_record_push(client, k, agent->x, agent->y);
        }
        client->trace_stamp = env->timestep;
        client->trace_ready = 1;
        if (client->record_n > 0) {
            env->selected_agent_idx = client->record_slot[0];
        }
        return;
    }
    int slot = env->selected_agent_idx;
    int n = env->active_agent_count;
    int onroad_left = sim_agent_onroad(env, slot);
    int replay_left = 0;
    if (!onroad_left) {
        for (int i = 0; i < n; i++) {
            if (sim_agent_onroad(env, i)) {
                onroad_left = 1;
                break;
            }
        }
    }
    if (!onroad_left) {
        for (int i = n; i < env->body_count; i++) {
            if (sim_body_drawn(env, i) && sim_near_road(env, i)) {
                replay_left = 1;
                break;
            }
        }
    }
    int ok = 0;
    if (onroad_left) {
        ok = sim_agent_onroad(env, slot);
    } else if (replay_left) {
        ok = slot >= n && sim_body_drawn(env, slot) && sim_near_road(env, slot);
    } else {
        ok = sim_agent_live(env, slot);
    }
    if (env->log_spawn && (client->trace_agent < 0 || reset || !ok)) {
        int next = sim_sample_agent(env, 1);
        if (client->trace_agent < 0 || reset || next != slot) {
            env->selected_agent_idx = next;
            sim_trace_clear(client, next);
        }
        slot = next;
    }
    if (slot != client->trace_agent) {
        sim_trace_clear(client, slot);
    }
    client->trace_stamp = env->timestep;
    client->trace_ready = 1;
    if (!sim_body_drawn(env, slot)) {
        return;
    }
    Entity* agent = &env->entities[slot];
    int same = client->trace_n > 0 && ready && env->timestep == seen;
    if (!same) {
        sim_trace_push(client, agent->x, agent->y);
    }
}

// Orthographic sheet centered on the followed car. fovy is the
// vertical span. The short side of the window is SIM_CROP meters.
static Camera3D ego_camera(Drive* env, float aspect) {
    float x = 0.0f;
    float y = 0.0f;
    if (env->map_corners) {
        x = 0.5f * (env->map_corners[0] + env->map_corners[2]);
        y = 0.5f * (env->map_corners[1] + env->map_corners[3]);
    }
    int slot = env->selected_agent_idx;
    if (sim_body_drawn(env, slot)) {
        x = env->entities[slot].x;
        y = env->entities[slot].y;
    } else if (puf_record_enabled() && env->client && env->client->trace_n > 0) {
        // The followed car was just parked at its goal. Hold the sheet
        // on the last point so that frame shows the removal.
        int i = env->client->trace_i - 1;
        if (i < 0) {
            i = SIM_TRACE - 1;
        }
        x = env->client->trace_x[i];
        y = env->client->trace_y[i];
    }
    if (aspect < 0.05f) {
        aspect = 1.0f;
    }
    float zoom = 1.0f;
    if (env->client && env->client->camera_zoom > 0.0f) {
        zoom = env->client->camera_zoom;
    }
    float span = SIM_CROP * zoom;
    float fovy = span;
    if (aspect < 1.0f) {
        fovy = span / aspect;
    }
    Camera3D cam = {0};
    cam.position = (Vector3){x, y, 400.0f};
    cam.target = (Vector3){x, y, 0.0f};
    cam.up = (Vector3){0.0f, -1.0f, 0.0f};
    cam.fovy = fovy;
    cam.projection = CAMERA_ORTHOGRAPHIC;
    return cam;
}

static void sim_follow_camera(Drive* env) {
    Client* client = env->client;
    if (!env->sim_follow || !client || puf_record_enabled()) {
        return;
    }
    float aspect = 1.0f;
    if (client->height > 1.0f) {
        aspect = client->width / client->height;
    }
    client->camera = ego_camera(env, aspect);
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

static void draw_world(Drive* env, Client* client);
static void draw_sheet(Drive* env);
static void draw_sheet_overlay(Drive* env, Camera3D camera, int sw, int sh);
static void puf_render_hud(Drive* env, Client* client);

// One square sheet. The clip ends when this car reaches its goal.
static void record_views(Drive* env) {
    Client* client = env->client;
    int w = (int)client->width;
    int h = (int)client->height;
    if (w < 2 || h < 2 || client->record_n <= 0) {
        return;
    }
    int side = SIM_RECORD_SIDE;
    int saved = env->selected_agent_idx;
    env->selected_agent_idx = client->record_slot[0];
    sim_record_install(client, 0);
    int margin = SIM_RECORD_MARGIN;
    int tex = side * SIM_RECORD_SS;
    sheet_hi_ensure(tex);
    Camera3D sheet = ego_camera(env, 1.0f);
    sheet.fovy *= SIM_RECORD_ZOOM;
    BeginTextureMode(sheet_hi);
    ClearBackground(PAPER);
    BeginMode3D(sheet);
    draw_sheet(env);
    EndMode3D();
    EndTextureMode();
    BeginDrawing();
    ClearBackground(PAPER);
    DrawTexturePro(sheet_hi.texture,
        (Rectangle){0.0f, 0.0f, (float)tex, -(float)tex},
        (Rectangle){(float)margin, (float)margin, (float)side, (float)side},
        (Vector2){0.0f, 0.0f}, 0.0f, WHITE);
    sheet_origin_x = margin;
    sheet_origin_y = margin;
    draw_sheet_overlay(env, sheet, side, side);
    sheet_origin_x = 0;
    sheet_origin_y = 0;
    DrawRectangleLinesEx((Rectangle){
        (float)margin + 0.5f, (float)margin + 0.5f,
        (float)side - 1.0f, (float)side - 1.0f
    }, 1.0f, BOX_EDGE);
    env->selected_agent_idx = saved;
    EndDrawing();
    puf_record_capture(w, h);
}

void puf_render(Drive* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    if (env->sim_user_size && IsWindowReady()) {
        int sw = GetScreenWidth();
        int sh = GetScreenHeight();
        if (sw >= 100 && sh >= 100) {
            env->client->width = (float)sw;
            env->client->height = (float)sh;
        }
    }
    sim_choose_ego(env);
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
    if (env->sim_follow) {
        BeginMode3D(client->camera);
        draw_sheet(env);
        EndMode3D();
        draw_sheet_overlay(env, client->camera,
            (int)client->width, (int)client->height);
    } else {
        BeginMode3D(client->camera);
        draw_world(env, client);
        EndMode3D();
    }
    puf_render_hud(env, client);
}

static Color road_color(int type) {
    if (type == ROAD_LINE) {
        return LANE_YELLOW;
    }
    if (type == ROAD_EDGE) {
        return EDGE_GREY;
    }
    return LANE_GREY;
}

static int paint_agent_hit(Drive* env, int i) {
    if (env->entities[i].collision_state == AGENT_COLLISION) {
        return 1;
    }
    int n = env->active_agent_count;
    if (n > DRIVE_AGENT_MAX) {
        n = DRIVE_AGENT_MAX;
    }
    for (int j = 0; j < n; j++) {
        if (j == i) {
            continue;
        }
        if (env->entities[j].collision_state == AGENT_COLLISION
                && drive_hit_with[j] == i) {
            return 1;
        }
    }
    return 0;
}

static int paint_edge_hit(Drive* env, int i) {
    Entity* car = &env->entities[i];
    if (car->offroad_state) {
        return 1;
    }
    return car->collision_state == OFFROAD;
}

static void car_paint(Drive* env, int i, Color* fill, Color* edge) {
    if (paint_agent_hit(env, i)) {
        *fill = HIT_RED;
        *edge = HIT_RED_EDGE;
        return;
    }
    if (paint_edge_hit(env, i)) {
        *fill = OFFROAD_ORANGE;
        *edge = OFFROAD_ORANGE_EDGE;
        return;
    }
    *edge = BOX_EDGE;
    if (i == env->selected_agent_idx) {
        *fill = EGO_BLUE;
        return;
    }
    if (i < env->active_agent_count) {
        *fill = POLICY_BLUE;
        return;
    }
    *fill = REPLAY_GREY;
}

static void draw_flat_car(float x, float y, float heading,
        float length, float width, Color fill, Color edge) {
    float hx = cosf(heading);
    float hy = sinf(heading);
    float px = -hy;
    float py = hx;
    float hl = length * 0.5f;
    float hw = width * 0.5f;
    if (hl < 0.8f) {
        hl = 0.8f;
    }
    if (hw < 0.4f) {
        hw = 0.4f;
    }
    float fx[4] = {hl, hl, -hl, -hl};
    float fy[4] = {hw, -hw, -hw, hw};
    Vector3 c[4];
    for (int k = 0; k < 4; k++) {
        c[k] = (Vector3){
            x + fx[k] * hx + fy[k] * px,
            y + fx[k] * hy + fy[k] * py,
            0.4f
        };
    }
    DrawTriangle3D(c[0], c[1], c[2], fill);
    DrawTriangle3D(c[0], c[2], c[3], fill);
    DrawTriangle3D(c[0], c[2], c[1], fill);
    DrawTriangle3D(c[0], c[3], c[2], fill);
    for (int k = 0; k < 4; k++) {
        DrawLine3D(c[k], c[(k + 1) % 4], edge);
    }
}

// Thin stroke under the followed car, about the width of a lane line.
static void draw_trace_segment(float x0, float y0, float x1, float y1) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 1.0e-3f) {
        return;
    }
    float half = 0.16f;
    float px = -dy / len * half;
    float py = dx / len * half;
    float z = 0.22f;
    Vector3 a = {x0 + px, y0 + py, z};
    Vector3 b = {x0 - px, y0 - py, z};
    Vector3 c = {x1 - px, y1 - py, z};
    Vector3 d = {x1 + px, y1 + py, z};
    DrawTriangle3D(a, b, c, EGO_BLUE);
    DrawTriangle3D(a, c, d, EGO_BLUE);
    DrawTriangle3D(a, c, b, EGO_BLUE);
    DrawTriangle3D(a, d, c, EGO_BLUE);
}

static void draw_ego_trace(Client* client) {
    if (!client || client->trace_n < 2) {
        return;
    }
    int start = 0;
    if (client->trace_n == SIM_TRACE) {
        start = client->trace_i;
    }
    for (int k = 1; k < client->trace_n; k++) {
        int a = (start + k - 1) % SIM_TRACE;
        int b = (start + k) % SIM_TRACE;
        draw_trace_segment(
            client->trace_x[a], client->trace_y[a],
            client->trace_x[b], client->trace_y[b]);
    }
}

static void draw_world(Drive* env, Client* client) {
    for (int i = 0; i < env->num_entities; i++) {
        int type = env->entities[i].type;
        if (type < ROAD_LANE || type > ROAD_EDGE) {
            continue;
        }
        Color line = road_color(type);
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            float ax = env->entities[i].pts_x[j];
            float ay = env->entities[i].pts_y[j];
            float bx = env->entities[i].pts_x[j + 1];
            float by = env->entities[i].pts_y[j + 1];
            if (ax == INVALID_POSITION || ay == INVALID_POSITION
                    || bx == INVALID_POSITION || by == INVALID_POSITION) {
                continue;
            }
            DrawLine3D(
                (Vector3){ax, ay, 0.05f},
                (Vector3){bx, by, 0.05f},
                line);
        }
    }
    rlDisableDepthTest();
    draw_ego_trace(client);
    rlDisableBackfaceCulling();
    for (int i = 0; i < env->body_count; i++) {
        Entity* car = &env->entities[i];
        int kind = car->type == VEHICLE || car->type == TRUCK;
        if (!kind || car->valid == 0
                || car->x == INVALID_POSITION
                || car->y == INVALID_POSITION) {
            continue;
        }
        Color fill = REPLAY_GREY;
        Color edge = BOX_EDGE;
        car_paint(env, i, &fill, &edge);
        draw_flat_car(car->x, car->y, car->heading,
            car->length, car->width, fill, edge);
    }
    rlEnableBackfaceCulling();
    rlEnableDepthTest();
}

static Vector2 sheet_point(Camera3D camera, int sw, int sh, float x, float y) {
    Vector2 p = GetWorldToScreenEx((Vector3){x, y, 0.0f}, camera, sw, sh);
    p.x += (float)sheet_origin_x;
    p.y += (float)sheet_origin_y;
    return p;
}

static float sheet_ppm(Camera3D camera, int sw, int sh) {
    Vector2 a = sheet_point(camera, sw, sh, camera.target.x, camera.target.y);
    Vector2 b = sheet_point(camera, sw, sh, camera.target.x + 1.0f, camera.target.y);
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    return sqrtf(dx * dx + dy * dy);
}

static int sheet_on_screen(Vector2 p, int sw, int sh) {
    float x = p.x - (float)sheet_origin_x;
    float y = p.y - (float)sheet_origin_y;
    return x >= -80.0f && y >= -80.0f && x <= (float)sw + 80.0f
        && y <= (float)sh + 80.0f;
}

static void tri_both(Vector3 a, Vector3 b, Vector3 c, Color color) {
    DrawTriangle3D(a, b, c, color);
    DrawTriangle3D(a, c, b, color);
}

static void quad_z(float x0, float y0, float x1, float y1,
        float x2, float y2, float x3, float y3, float z, Color color) {
    Vector3 a = {x0, y0, z};
    Vector3 b = {x1, y1, z};
    Vector3 c = {x2, y2, z};
    Vector3 d = {x3, y3, z};
    tri_both(a, b, c, color);
    tri_both(a, c, d, color);
}

// Offset from a polyline vertex. Sharp corners are clamped so the join
// stays a short bevel instead of a spike.
static void ribbon_sides(float x0, float y0, float x1, float y1, float x2, float y2,
        int has_prev, int has_next, float half,
        float* lx, float* ly, float* rx, float* ry) {
    float ax = 0.0f;
    float ay = 0.0f;
    float bx = 0.0f;
    float by = 0.0f;
    if (has_prev) {
        float dx = x1 - x0;
        float dy = y1 - y0;
        float len = sqrtf(dx * dx + dy * dy);
        if (len > 1.0e-5f) {
            ax = dx / len;
            ay = dy / len;
        }
    }
    if (has_next) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len = sqrtf(dx * dx + dy * dy);
        if (len > 1.0e-5f) {
            bx = dx / len;
            by = dy / len;
        }
    }
    if (!has_prev) {
        ax = bx;
        ay = by;
    }
    if (!has_next) {
        bx = ax;
        by = ay;
    }
    float nax = -ay;
    float nay = ax;
    float nbx = -by;
    float nby = bx;
    float mx = nax + nbx;
    float my = nay + nby;
    float mlen = sqrtf(mx * mx + my * my);
    if (mlen < 1.0e-4f) {
        mx = nax;
        my = nay;
    } else {
        mx /= mlen;
        my /= mlen;
    }
    float dot = nax * mx + nay * my;
    if (dot < 0.35f) {
        dot = 0.35f;
    }
    float scale = half / dot;
    *lx = x1 + mx * scale;
    *ly = y1 + my * scale;
    *rx = x1 - mx * scale;
    *ry = y1 - my * scale;
}

static void draw_ribbon(const float* xs, const float* ys, int n,
        float width, float z, Color color) {
    if (n < 2 || width <= 0.0f) {
        return;
    }
    float half = width * 0.5f;
    float plx = 0.0f;
    float ply = 0.0f;
    float prx = 0.0f;
    float pry = 0.0f;
    for (int i = 0; i < n; i++) {
        float x0 = xs[i];
        float y0 = ys[i];
        float x2 = xs[i];
        float y2 = ys[i];
        if (i > 0) {
            x0 = xs[i - 1];
            y0 = ys[i - 1];
        }
        if (i + 1 < n) {
            x2 = xs[i + 1];
            y2 = ys[i + 1];
        }
        float lx, ly, rx, ry;
        ribbon_sides(x0, y0, xs[i], ys[i], x2, y2, i > 0, i + 1 < n, half,
            &lx, &ly, &rx, &ry);
        if (i > 0) {
            quad_z(plx, ply, prx, pry, rx, ry, lx, ly, z, color);
        }
        plx = lx;
        ply = ly;
        prx = rx;
        pry = ry;
    }
}

static float rib_x[8192];
static float rib_y[8192];

static void draw_sheet_roads(Drive* env) {
    for (int i = 0; i < env->num_entities; i++) {
        int type = env->entities[i].type;
        if (type < ROAD_LANE || type > ROAD_EDGE) {
            continue;
        }
        float width = 0.16f;
        float z = 0.05f;
        if (type == ROAD_LINE) {
            width = 0.13f;
            z = 0.06f;
        } else if (type == ROAD_EDGE) {
            width = 0.28f;
            z = 0.07f;
        }
        Color line = road_color(type);
        int n = 0;
        int count = env->entities[i].array_size;
        for (int j = 0; j < count; j++) {
            float x = env->entities[i].pts_x[j];
            float y = env->entities[i].pts_y[j];
            int bad = x == INVALID_POSITION || y == INVALID_POSITION;
            if (bad || n == 8192) {
                draw_ribbon(rib_x, rib_y, n, width, z, line);
                n = 0;
                if (bad) {
                    continue;
                }
            }
            rib_x[n] = x;
            rib_y[n] = y;
            n++;
        }
        draw_ribbon(rib_x, rib_y, n, width, z, line);
    }
}

static void draw_oriented_box(float x, float y, float heading,
        float length, float width, float z, Color color) {
    float hx = cosf(heading);
    float hy = sinf(heading);
    float px = -hy;
    float py = hx;
    float hl = length * 0.5f;
    float hw = width * 0.5f;
    float fx[4] = {hl, hl, -hl, -hl};
    float fy[4] = {hw, -hw, -hw, hw};
    Vector3 c[4];
    for (int k = 0; k < 4; k++) {
        c[k] = (Vector3){
            x + fx[k] * hx + fy[k] * px,
            y + fx[k] * hy + fy[k] * py,
            z
        };
    }
    tri_both(c[0], c[1], c[2], color);
    tri_both(c[0], c[2], c[3], color);
}

// Filled box and a filled heading arrow. The outline is a slightly
// larger box under the fill, so the edge stays a constant width.
static void draw_sheet_car(Drive* env, int i) {
    Entity* car = &env->entities[i];
    Color fill = REPLAY_GREY;
    Color edge = BOX_EDGE;
    car_paint(env, i, &fill, &edge);
    float length = car->length;
    float width = car->width;
    if (length < 0.6f) {
        length = 0.6f;
    }
    if (width < 0.5f) {
        width = 0.5f;
    }
    float outline = 0.10f;
    draw_oriented_box(car->x, car->y, car->heading,
        length + outline * 2.0f, width + outline * 2.0f, 0.80f, edge);
    draw_oriented_box(car->x, car->y, car->heading,
        length, width, 0.90f, fill);
    float hx = cosf(car->heading);
    float hy = sinf(car->heading);
    float half = length * 0.5f;
    float z = 1.15f;
    Color mark = INK;
    float head = 0.70f;
    float tip_x = car->x + hx * (half + 1.05f);
    float tip_y = car->y + hy * (half + 1.05f);
    float base_x = tip_x - hx * head;
    float base_y = tip_y - hy * head;
    float wing = head * 0.42f;
    float shaft = 0.12f;
    float sx = -hy * shaft;
    float sy = hx * shaft;
    float tail = half * 0.15f;
    quad_z(car->x + hx * tail + sx, car->y + hy * tail + sy,
        car->x + hx * tail - sx, car->y + hy * tail - sy,
        base_x - sx, base_y - sy, base_x + sx, base_y + sy, z, mark);
    Vector3 tip = {tip_x, tip_y, z};
    Vector3 left = {base_x - hy * wing, base_y + hx * wing, z};
    Vector3 right = {base_x + hy * wing, base_y - hx * wing, z};
    tri_both(tip, left, right, mark);
}

static void draw_disk(float x, float y, float radius, float z, int seg, Color color) {
    if (radius <= 0.0f) {
        return;
    }
    Vector3 center = {x, y, z};
    float step = 2.0f * PI / (float)seg;
    for (int i = 0; i < seg; i++) {
        float a0 = step * (float)i;
        float a1 = step * (float)(i + 1);
        Vector3 p0 = {x + cosf(a0) * radius, y + sinf(a0) * radius, z};
        Vector3 p1 = {x + cosf(a1) * radius, y + sinf(a1) * radius, z};
        tri_both(center, p0, p1, color);
    }
}

static void draw_annulus(float x, float y, float inner, float outer,
        float z, int seg, Color color) {
    if (outer <= inner || inner < 0.0f) {
        return;
    }
    float step = 2.0f * PI / (float)seg;
    for (int i = 0; i < seg; i++) {
        float a0 = step * (float)i;
        float a1 = step * (float)(i + 1);
        float c0 = cosf(a0);
        float s0 = sinf(a0);
        float c1 = cosf(a1);
        float s1 = sinf(a1);
        quad_z(
            x + c0 * inner, y + s0 * inner,
            x + c0 * outer, y + s0 * outer,
            x + c1 * outer, y + s1 * outer,
            x + c1 * inner, y + s1 * inner,
            z, color);
    }
}

// Goal of the followed car. It stays after that car is parked.
static void draw_sheet_goal(Drive* env) {
    int slot = env->selected_agent_idx;
    if (!env->entities || slot < 0 || slot >= env->active_agent_count) {
        return;
    }
    Entity* car = &env->entities[slot];
    if (car->goals_sampled <= 0) {
        return;
    }
    float gx = car->goal_position_x;
    float gy = car->goal_position_y;
    if (gx == INVALID_POSITION || gy == INVALID_POSITION) {
        return;
    }
    float reach = MIN_DISTANCE_TO_GOAL;
    if (env->goal_radius > 0.0f) {
        reach = env->goal_radius;
    }
    float z = 1.25f;
    draw_annulus(gx, gy, reach - 0.16f, reach + 0.06f, z, 96, EGO_BLUE);
    draw_disk(gx, gy, 0.55f, z, 24, EGO_BLUE);
}

// Logged pose of the followed car, under its controlled box.
static void draw_sheet_ghosts(Drive* env) {
    if (!env->gt_x || env->gt_n <= 0 || env->gt_len <= 0) {
        return;
    }
    int i = env->selected_agent_idx;
    if (i < 0 || i >= env->gt_n || i >= env->active_agent_count) {
        return;
    }
    int frame = env->init_steps + env->timestep;
    if (frame < 0 || frame >= env->gt_len) {
        return;
    }
    int k = i * env->gt_len + frame;
    if (!env->gt_v[k]) {
        return;
    }
    float x = env->gt_x[k];
    float y = env->gt_y[k];
    if (x == INVALID_POSITION || y == INVALID_POSITION) {
        return;
    }
    Entity* car = &env->entities[i];
    float length = car->length;
    float width = car->width;
    if (length < 0.6f) {
        length = 0.6f;
    }
    if (width < 0.5f) {
        width = 0.5f;
    }
    float outline = 0.10f;
    float heading = env->gt_h[k];
    draw_oriented_box(x, y, heading,
        length + outline * 2.0f, width + outline * 2.0f,
        0.50f, GHOST_GREEN_EDGE);
    draw_oriented_box(x, y, heading, length, width, 0.60f, GHOST_GREEN);
}

// Top-down sheet drawn inside BeginMode3D. Marks are filled ribbons
// so the joins stay smooth. The palette stays the light one.
static void draw_sheet(Drive* env) {
    rlDisableBackfaceCulling();
    rlDisableDepthTest();
    draw_sheet_roads(env);
    draw_ego_trace(env->client);
    draw_sheet_ghosts(env);
    for (int i = 0; i < env->body_count; i++) {
        Entity* car = &env->entities[i];
        int kind = car->type == VEHICLE || car->type == TRUCK;
        if (!kind || car->valid == 0
                || car->x == INVALID_POSITION
                || car->y == INVALID_POSITION) {
            continue;
        }
        draw_sheet_car(env, i);
    }
    draw_sheet_goal(env);
    rlEnableDepthTest();
    rlEnableBackfaceCulling();
}

static void draw_halo(Drive* env, const char* text, float x, float y, float size) {
    float pad = size * 0.06f;
    if (pad < 1.0f) {
        pad = 1.0f;
    }
    sim_sans(env, text, x - pad, y, size, PAPER);
    sim_sans(env, text, x + pad, y, size, PAPER);
    sim_sans(env, text, x, y - pad, size, PAPER);
    sim_sans(env, text, x, y + pad, size, PAPER);
    sim_sans(env, text, x, y, size, INK);
}

// Screen-space labels. The record path never reaches the HUD.
static void draw_sheet_overlay(Drive* env, Camera3D camera, int sw, int sh) {
    float ppm = sheet_ppm(camera, sw, sh);
    Font font = sim_font(env);
    float size = ppm * 0.72f;
    if (size < 14.0f) {
        size = 14.0f;
    }
    if (size > 36.0f) {
        size = 36.0f;
    }
    for (int i = 0; i < env->body_count; i++) {
        Entity* car = &env->entities[i];
        int kind = car->type == VEHICLE || car->type == TRUCK;
        if (!kind || car->valid == 0
                || car->x == INVALID_POSITION
                || car->y == INVALID_POSITION) {
            continue;
        }
        float lx = car->x;
        float ly = car->y;
        float spd = sqrtf(car->vx * car->vx + car->vy * car->vy);
        float back = car->length * 0.5f;
        if (back < 0.3f) {
            back = 0.3f;
        }
        back += size / ppm + 0.35f;
        if (spd > 0.5f) {
            lx -= car->vx / spd * back;
            ly -= car->vy / spd * back;
        }
        Vector2 s = sheet_point(camera, sw, sh, lx, ly);
        if (!sheet_on_screen(s, sw, sh)) {
            continue;
        }
        char label[16];
        snprintf(label, sizeof(label), "%d", i);
        Vector2 tw = MeasureTextEx(font, label, size, 0.4f);
        draw_halo(env, label, s.x - tw.x * 0.5f, s.y - tw.y * 0.5f, size);
    }
    char step[32];
    snprintf(step, sizeof(step), "t = %d", env->timestep);
    float title = (float)sh * 0.055f;
    if (title < 40.0f) {
        title = 40.0f;
    }
    draw_halo(env, step, 18.0f + (float)sheet_origin_x,
        14.0f + (float)sheet_origin_y, title);
}

static void puf_render_hud(Drive* env, Client* client) {
    if (!puf_record_enabled() && !env->checkpoint_ok) {
        DrawText("Checkpoint cannot be loaded, showing random behavior",
            10, 10, 20, CRIMSON);
    }
    if (!puf_record_enabled()) {
        sim_transport_draw(env);
    }
    if (env->hud_extra) {
        env->hud_extra(env);
    }
    EndDrawing();
    if (puf_record_enabled()) {
        puf_record_capture((int)client->width, (int)client->height);
    }
    puf_web_vsync();
}

void close_client(Client* client) {
    if (sheet_hi.id != 0) {
        UnloadRenderTexture(sheet_hi);
        sheet_hi.id = 0;
        sheet_hi_side = 0;
    }
    if (client->record_views) {
        UnloadRenderTexture(client->record_bev);
        UnloadRenderTexture(client->record_pov);
        client->record_views = 0;
    }
    for (int i = 0; i < 2; i++) {
        UnloadModel(client->cars[i]);
    }
    if (client->font_ready) {
        UnloadFont(client->font);
    }
    CloseWindow();
    free(client);
}

