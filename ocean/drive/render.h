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

// Right-hand column: playback, follow, agents.
enum {
    SIM_PLAY = 0,
    SIM_PAUSE = 1,
    SIM_RESET = 2,
    SIM_FOLLOW = 3,
    SIM_THROUGH = 4,
    SIM_PHYSICS = 5,
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
    r[SIM_FOLLOW] = (Rectangle){x, y, bw, bh};
    y += bh + grp;
    r[SIM_THROUGH] = (Rectangle){x, y, bw, bh};
    y += bh + gap;
    r[SIM_PHYSICS] = (Rectangle){x, y, bw, bh};
}

static Rectangle sim_agents_rect(float width) {
    Rectangle r[SIM_NBTN];
    sim_rects(width, r);
    float y = r[SIM_PHYSICS].y + r[SIM_PHYSICS].height + 18.0f;
    return (Rectangle){
        r[SIM_PHYSICS].x, y, r[SIM_PHYSICS].width, SIM_AGENT_H
    };
}

static int sim_speed_shown(Drive* env) {
    return env->sim_speed_max >= SIM_SPEED_MIN;
}

// Live mainline rate. Shown in train too; only inference inserts.
static int sim_vph_shown(Drive* env) {
    return env->lane_state != NULL;
}

static Rectangle sim_vph_rect(float width) {
    Rectangle agents = sim_agents_rect(width);
    float y = agents.y + agents.height + 18.0f;
    return (Rectangle){agents.x, y, agents.width, SIM_AGENT_H};
}

static Rectangle sim_speed_rect(Drive* env, float width) {
    Rectangle above = sim_agents_rect(width);
    if (sim_vph_shown(env)) {
        above = sim_vph_rect(width);
    }
    float y = above.y + above.height + 18.0f;
    return (Rectangle){above.x, y, above.width, SIM_AGENT_H};
}

static Rectangle sim_hi_rect(Drive* env, float width) {
    Rectangle lo = sim_speed_rect(env, width);
    float y = lo.y + lo.height + 18.0f;
    return (Rectangle){lo.x, y, lo.width, SIM_AGENT_H};
}

static Rectangle sim_dist_rect(Drive* env, float width) {
    Rectangle hi = sim_hi_rect(env, width);
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

// Linear across 0..DRIVE_VPH_MAX. 0 stops insertion.
static int sim_vph_from_x(float mouse_x, Rectangle zone) {
    float t = 0.0f;
    if (zone.width > 1.0f) {
        t = (mouse_x - zone.x) / zone.width;
    }
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    int n = (int)(t * (float)DRIVE_VPH_MAX + 0.5f);
    if (n < 0) n = 0;
    if (n > DRIVE_VPH_MAX) n = DRIVE_VPH_MAX;
    return n;
}

static Rectangle sim_col_rect(Drive* env) {
    float width = env->client->width;
    Rectangle zone = sim_agents_rect(width);
    if (sim_vph_shown(env)) {
        zone = sim_vph_rect(width);
    }
    if (sim_speed_shown(env)) {
        zone = sim_dist_rect(env, width);
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
    if (env->entities) {
        Color rug = {241, 241, 241, 200};
        for (int i = 0; i < nagt; i++) {
            float m = env->entities[i].max_speed;
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
    // Speed over 100, fitted to the crop so the head is not clipped off.
    float head_m = 13.0f / ppm + 0.3f;
    float room = SIM_OBS_RANGE - ego_hl - head_m;
    if (room < 1.0f) {
        room = 1.0f;
    }
    float vlen = (spd / MAX_SPEED) * room;
    sim_obs_arrow(cx, cy, ppm, ego_hl, 0.0f, 1.0f, 0.0f,
        vlen, PUFF_WHITE);
    EndScissorMode();
    DrawRectangleLinesEx(plot, 1.0f, (Color){70, 76, 88, 255});
}

static void sim_transport_draw(Drive* env) {
    if (!env->client) {
        return;
    }
    Rectangle r[SIM_NBTN];
    sim_rects(env->client->width, r);
    const char* labels[SIM_NBTN] = {
        "Play", "Pause", "Reset", "Follow", "Through", "Physics"
    };
    int on[SIM_NBTN] = {
        !env->sim_paused,
        env->sim_paused,
        0,
        env->sim_follow,
        env->sim_through,
        !env->sim_through
    };
    DrawRectangleRec(sim_col_rect(env), (Color){12, 14, 20, 230});
    Vector2 mouse = GetMousePosition();
    Color sep = (Color){70, 76, 88, 255};
    int y1 = (int)(r[SIM_RESET].y + r[SIM_RESET].height
        + r[SIM_FOLLOW].y) / 2;
    int x0 = (int)r[SIM_PLAY].x;
    int x1 = (int)(r[SIM_PLAY].x + r[SIM_PLAY].width);
    DrawLine(x0, y1, x1, y1, sep);
    int y_col = (int)(r[SIM_FOLLOW].y + r[SIM_FOLLOW].height
        + r[SIM_THROUGH].y) / 2;
    DrawLine(x0, y_col, x1, y_col, sep);
    for (int i = 0; i < SIM_NBTN; i++) {
        int hot = CheckCollisionPointRec(mouse, r[i]);
        sim_choice(r[i], labels[i], on[i], hot, i == SIM_PAUSE);
    }
    Rectangle zone = sim_agents_rect(env->client->width);
    int y3 = (int)(r[SIM_PHYSICS].y + r[SIM_PHYSICS].height + zone.y) / 2;
    DrawLine(x0, y3, x1, y3, sep);
    char num[8];
    snprintf(num, sizeof(num), "%d", env->sim_agents_req);
    float t = (float)(env->sim_agents_req - 1) / (float)(DRIVE_AGENT_MAX - 1);
    sim_slider_row(zone, "Agents", num, t,
        env->sim_agents_req != env->sim_agents_on);
    Rectangle above = zone;
    if (sim_vph_shown(env)) {
        Rectangle vph = sim_vph_rect(env->client->width);
        int yv = (int)(above.y + above.height + vph.y) / 2;
        DrawLine(x0, yv, x1, yv, sep);
        char vn[8];
        snprintf(vn, sizeof(vn), "%d", env->sim_vph_on);
        float vt = 0.0f;
        if (DRIVE_VPH_MAX > 0) {
            vt = (float)env->sim_vph_on / (float)DRIVE_VPH_MAX;
        }
        sim_slider_row(vph, "veh/h", vn, vt, 0);
        above = vph;
    }
    if (sim_speed_shown(env)) {
        Rectangle spd = sim_speed_rect(env, env->client->width);
        int y4 = (int)(above.y + above.height + spd.y) / 2;
        DrawLine(x0, y4, x1, y4, sep);
        char sn[16];
        snprintf(sn, sizeof(sn), "%.2f", env->sim_speed_req);
        float span = MAX_SPEED - SIM_SPEED_MIN;
        float st = 0.0f;
        if (span > 1e-3f) {
            st = (env->sim_speed_req - SIM_SPEED_MIN) / span;
        }
        sim_slider_row(spd, "Min", sn, st, sim_speed_pending(env));
        Rectangle sd = sim_hi_rect(env, env->client->width);
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
        sim_dist_draw(env, sim_dist_rect(env, env->client->width),
            dist_pending);
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
        env->sim_vph_drag = 0;
        return;
    }
    // Drop the one-frame hold before reading this frame's click.
    env->sim_hold_step = 0;
    if (IsKeyPressed(KEY_P)) {
        env->sim_paused = !env->sim_paused;
    }
    Rectangle zone = sim_agents_rect(env->client->width);
    Rectangle vph = sim_vph_rect(env->client->width);
    Rectangle spd = sim_speed_rect(env, env->client->width);
    Rectangle sd = sim_hi_rect(env, env->client->width);
    int vph_on = sim_vph_shown(env);
    int speed_on = sim_speed_shown(env);
    Vector2 mouse = GetMousePosition();
    int over = CheckCollisionPointRec(mouse, zone);
    int over_vph = vph_on && CheckCollisionPointRec(mouse, vph);
    int over_spd = speed_on && CheckCollisionPointRec(mouse, spd);
    int over_hi = speed_on && CheckCollisionPointRec(mouse, sd);
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && over) {
        env->sim_agents_drag = 1;
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && over_vph) {
        env->sim_vph_drag = 1;
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
        env->sim_vph_drag = 0;
    }
    if (env->sim_agents_drag) {
        sim_agents_set(env, sim_agents_from_x(mouse.x, zone));
    }
    if (env->sim_vph_drag) {
        flow_set_vph(env, sim_vph_from_x(mouse.x, vph));
    }
    if (env->sim_speed_drag) {
        sim_speed_set(env, sim_speed_from_x(mouse.x, spd));
    }
    if (env->sim_hi_drag) {
        sim_hi_set(env, sim_speed_from_x(mouse.x, sd));
    }
    float wheel = GetMouseWheelMove();
    int dragging = env->sim_agents_drag || env->sim_vph_drag
        || env->sim_speed_drag || env->sim_hi_drag;
    if (wheel != 0.0f && !dragging) {
        int step = 1;
        if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
            step = 10;
        }
        int dir = wheel > 0.0f ? 1 : -1;
        if (over) {
            sim_agents_set(env, env->sim_agents_req + dir * step);
        } else if (over_vph) {
            int vstep = 100;
            if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
                vstep = 1000;
            }
            flow_set_vph(env, env->sim_vph_on + dir * vstep);
        } else if (over_spd) {
            sim_speed_set(env, env->sim_speed_req + (float)(dir * step));
        } else if (over_hi) {
            sim_hi_set(env, env->sim_hi_req + (float)(dir * step));
        }
    }
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        return;
    }
    if (over || over_vph || over_spd || over_hi || dragging) {
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
    } else if (CheckCollisionPointRec(mouse, r[SIM_FOLLOW])) {
        env->sim_follow = !env->sim_follow;
        if (!env->sim_follow) {
            sim_camera_bev(env);
        }
    } else if (CheckCollisionPointRec(mouse, r[SIM_THROUGH])) {
        env->sim_through = 1;
    } else if (CheckCollisionPointRec(mouse, r[SIM_PHYSICS])) {
        env->sim_through = 0;
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
    int slot = env->human_agent_idx;
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
    if (env->active_agent_count <= 0 || !env->entities) {
        return;
    }
    int slot = env->human_agent_idx;
    if (slot < 0 || slot >= env->active_agent_count) {
        slot = 0;
    }
    Entity* agent = &env->entities[slot];
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
    Color empty = {20, 24, 32, 255};
    for (int lane = 0; lane < lanes; lane++) {
        for (int t = 0; t < times; t++) {
            size_t si = (size_t)t * (size_t)lanes + (size_t)lane;
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
    if (tnow > duration) {
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
        int col = tstep;
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

            Model car_model = client->cars[client->car_assignments[i % MAX_AGENTS]];
            if (env->entities[i].collision_state > NO_COLLISION) {
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

            if (env->entities[i].valid == 0) continue;
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
                    env->entities[i].pts_x[j], env->entities[i].pts_y[j],
                    env->entities[i].pts_x[j + 1], env->entities[i].pts_y[j + 1]);
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
    if (env->active_agent_count > 0
        && env->human_agent_idx >= 0
        && env->human_agent_idx < env->active_agent_count) {
        human_idx = env->human_agent_idx;
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

