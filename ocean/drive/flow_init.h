// Structured vehicle initialization (env.agent_initialization = 1).
//
// The [env] keys play the role of a SUMO route file:
//   vtype_N = id,length,width,height,maxSpeed
//   route_N = id,edge,edge,...
//   flow_N  = id,type,begin,end,vehsPerHour,route,departLane,departSpeed
//   network_file = path to a SUMO .net.xml
// Indices start at 0 and have no gaps. Three-field vtypes
// (id,length,maxSpeed) use SUMO's default width 1.8 and height 1.5.
//
// A flow emits one vehicle every 3600/vehsPerHour seconds from begin up to
// end. Placement puts down the vehicles that would still be on the route at
// time `end` if each had held depart speed with no car-following and no lane
// changes. departSpeed "desired" and "max" use min(vType maxSpeed, lane speed)
// on each lane; there is no leader yet, so "max" matches "desired". departPos
// is SUMO's default "base" (front bumper at length + 0.1 m).
//
// Train keeps that warm start. Each reset samples a goal from 40 m ahead
// to the route end, and a car that arrives respawns at its own spawn.
// Inference (./drive) keeps the warm start too, puts every goal at the
// route end, and keeps inserting at the entrance on the same clock.
// A car that reaches the end leaves, which frees its slot for the next
// insertion. The veh/h slider scales the mainline rate. Other flows keep
// the ini ratio. The clock is SUMO's: depart_k = begin + k * 3600/vph,
// and the first insertion is the first depart at or after `end`.
//
// The learned policy's bicycle model moves every car. The routes stay so
// a reset can restore each spawn. A train respawn copies the spawn pose
// back and keeps the goal and the max speed.
//
// departLane "random" and "free" assign departure lanes round-robin across
// the first edge, shifted by env->rng, so parallel envs differ but the count
// does not. "best" keeps the lane that follows the route farthest. "first"
// and "allowed" use lane 0. A non-negative integer selects that lane.

#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FLOW_MAX 16
#define FLOW_MAX_EDGES 16
#define FLOW_MAX_LANES 16
#define FLOW_NAME 64
#define DEPART_POS_EPS 0.1f
// Center-to-center gap required at the entrance. length 4.3 + minGap 2.8.
#define FLOW_INSERT_GAP 8.0f
// Departures held while the entrance is blocked or every slot is full.
#define FLOW_QUEUE 32

enum {
    LANE_SPLIT = 0,
    LANE_FIXED = 1,
    LANE_BEST = 2
};

enum {
    SPEED_DESIRED = 0,
    SPEED_LIMIT = 1,
    SPEED_VALUE = 2
};

typedef struct {
    char id[FLOW_NAME];
    float length;
    float width;
    float height;
    float max_speed;
} VTypeSpec;

typedef struct {
    char id[FLOW_NAME];
    int nedges;
    char edges[FLOW_MAX_EDGES][FLOW_NAME];
} RouteSpec;

typedef struct {
    char id[FLOW_NAME];
    char type_id[FLOW_NAME];
    char route_id[FLOW_NAME];
    char lane_token[FLOW_NAME];
    char speed_token[FLOW_NAME];
    double begin;
    double end;
    double vehs_per_hour;
    int lane_mode;
    int lane_index;
    int speed_mode;
    float speed_value;
} FlowSpec;

typedef struct {
    char id[FLOW_NAME];
    char edge[FLOW_NAME];
    int index;
    int internal;
    float speed;
    int n;
    float* x;
    float* y;
} SumoLane;

typedef struct {
    char from[FLOW_NAME];
    char to[FLOW_NAME];
    char via[FLOW_NAME];
    int from_lane;
    int to_lane;
} SumoConn;

typedef struct {
    SumoLane* lanes;
    int nlanes;
    int lane_cap;
    SumoConn* conns;
    int nconns;
    int conn_cap;
} SumoNet;

typedef struct {
    float* x;
    float* y;
    int n;
    float speed;
    float len;
} FlowPiece;

typedef struct {
    FlowPiece* pieces;
    int n;
    float len;
} FlowPath;

typedef struct {
    FlowPath paths[FLOW_MAX_LANES];
    int lane_of[FLOW_MAX_LANES];
    int npaths;
    double horizon;
    int i0;
    int i1;
    const VTypeSpec* vtype;
} FlowPlan;

typedef struct {
    float x, y, heading, speed, max_speed;
    float goal_x, goal_y;
    int flow;
    int path;
    float along;
    double age;
    const VTypeSpec* vtype;
} SpawnVeh;

// Routes kept after placement. s0 is the spawn arc. Goals are sampled
// from it at reset. The bicycle model does not integrate along the path.
typedef struct {
    FlowPath* paths;
    int npaths;
    int* path_of;
    float* s0;
    int nveh;
    // Insertion clock. base_vph is the ini rate. vph follows the slider.
    int nflows;
    double base_vph[FLOW_MAX];
    double vph[FLOW_MAX];
    double begin[FLOW_MAX];
    double end[FLOW_MAX];
    double next_depart[FLOW_MAX];
    float length[FLOW_MAX];
    int path0[FLOW_MAX];
    int npath[FLOW_MAX];
    int pending[FLOW_MAX];
    int cursor[FLOW_MAX];
} FlowRoutes;

static void flow_die(const char* msg) {
    fprintf(stderr, "structured init: %s\n", msg);
    exit(1);
}

static const char* cfg_str(Dict* cfg, const char* key) {
    DictItem* item = dict_find(cfg, key);
    if (!item || !item->str || !item->str[0]) return NULL;
    return item->str;
}

static int parse_f(const char* s, double* out) {
    char* end = NULL;
    double v = strtod(s, &end);
    if (!s[0] || end == s || *end) return 0;
    *out = v;
    return 1;
}

static int parse_i(const char* s, int* out) {
    char* end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || end == s || *end) return 0;
    *out = (int)v;
    return 1;
}

static int split_csv(const char* s, char toks[][FLOW_NAME], int max) {
    int n = 0;
    const char* p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (n >= max) return -1;
        int i = 0;
        while (*p && *p != ',') {
            if (i + 1 >= FLOW_NAME) return -1;
            toks[n][i++] = *p++;
        }
        while (i > 0 && (toks[n][i - 1] == ' ' || toks[n][i - 1] == '\t')) i--;
        toks[n][i] = 0;
        if (i == 0) return -1;
        n++;
        if (*p == ',') p++;
    }
    return n;
}

static int file_ok(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

static void resolve_network(Drive* env, char* out, int cap) {
    const char* given = cfg_str(env->cfg, "network_file");
    if (given) {
        if (!file_ok(given)) {
            fprintf(stderr, "structured init: cannot open network_file %s\n", given);
            exit(1);
        }
        snprintf(out, cap, "%s", given);
        return;
    }
    const char* data_dir = dict_get_str(env->cfg, "data_dir");
    snprintf(out, cap, "%s/onramp.net.xml", data_dir);
    if (file_ok(out)) return;
    snprintf(out, cap, "%s/smallnet.net.xml", data_dir);
    if (file_ok(out)) return;
    flow_die("set env.network_file to a SUMO .net.xml");
}

static char* read_whole(const char* path, size_t* n_out) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0 || n > 50L * 1024L * 1024L) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf) {
        perror("malloc");
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    buf[n] = 0;
    fclose(fp);
    *n_out = (size_t)n;
    return buf;
}

static const char* tag_end(const char* start, const char* end) {
    for (const char* p = start; p < end; p++) {
        if (*p == '>') return p;
    }
    return NULL;
}

static int attr_copy(const char* start, const char* end, const char* key,
        char* out, int cap) {
    char pat[96];
    int pn = snprintf(pat, sizeof(pat), "%s=\"", key);
    if (pn <= 0 || pn >= (int)sizeof(pat)) return 0;
    for (const char* p = start; p + pn < end; p++) {
        int boundary = (p == start || p[-1] == ' ' || p[-1] == '\n' ||
            p[-1] == '\t' || p[-1] == '\r');
        if (!boundary || memcmp(p, pat, (size_t)pn) != 0) continue;
        p += pn;
        int i = 0;
        int truncated = 0;
        while (p < end && *p != '"') {
            if (i + 1 >= cap) {
                truncated = 1;
                break;
            }
            out[i++] = *p++;
        }
        out[i] = 0;
        if (truncated || p >= end || *p != '"') return 0;
        return 1;
    }
    return 0;
}

static int parse_shape(const char* s, float** ox, float** oy) {
    int cap = 8;
    int n = 0;
    float* xs = (float*)malloc((size_t)cap * sizeof(float));
    float* ys = (float*)malloc((size_t)cap * sizeof(float));
    if (!xs || !ys) {
        perror("malloc");
        exit(1);
    }
    const char* p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        char* end = NULL;
        float x = strtof(p, &end);
        if (end == p || *end != ',') {
            free(xs);
            free(ys);
            return -1;
        }
        p = end + 1;
        float y = strtof(p, &end);
        if (end == p) {
            free(xs);
            free(ys);
            return -1;
        }
        p = end;
        if (n == cap) {
            cap *= 2;
            xs = (float*)realloc(xs, (size_t)cap * sizeof(float));
            ys = (float*)realloc(ys, (size_t)cap * sizeof(float));
            if (!xs || !ys) {
                perror("realloc");
                exit(1);
            }
        }
        xs[n] = x;
        ys[n] = y;
        n++;
        if (n > 100000) flow_die("lane shape is too long");
    }
    *ox = xs;
    *oy = ys;
    return n;
}

static void net_add_lane(SumoNet* net, const SumoLane* lane) {
    if (net->nlanes == net->lane_cap) {
        net->lane_cap = net->lane_cap ? net->lane_cap * 2 : 16;
        net->lanes = (SumoLane*)realloc(net->lanes, (size_t)net->lane_cap * sizeof(SumoLane));
        if (!net->lanes) {
            perror("realloc");
            exit(1);
        }
    }
    net->lanes[net->nlanes++] = *lane;
}

static void net_add_conn(SumoNet* net, const SumoConn* conn) {
    if (net->nconns == net->conn_cap) {
        net->conn_cap = net->conn_cap ? net->conn_cap * 2 : 16;
        net->conns = (SumoConn*)realloc(net->conns, (size_t)net->conn_cap * sizeof(SumoConn));
        if (!net->conns) {
            perror("realloc");
            exit(1);
        }
    }
    net->conns[net->nconns++] = *conn;
}

static void parse_net(SumoNet* net, const char* xml, size_t n) {
    memset(net, 0, sizeof(*net));
    const char* end = xml + n;
    const char* p = xml;
    char edge_id[FLOW_NAME];
    int in_edge = 0;
    int internal = 0;
    edge_id[0] = 0;
    while (p < end) {
        if ((size_t)(end - p) >= 6 && memcmp(p, "<edge ", 6) == 0) {
            const char* gt = tag_end(p, end);
            if (!gt) flow_die("unclosed <edge> in network");
            if (!attr_copy(p, gt, "id", edge_id, FLOW_NAME)) {
                flow_die("network edge is missing id");
            }
            char func[32];
            internal = attr_copy(p, gt, "function", func, (int)sizeof(func)) &&
                strcmp(func, "internal") == 0;
            in_edge = 1;
            p = gt + 1;
            continue;
        }
        if ((size_t)(end - p) >= 7 && memcmp(p, "</edge>", 7) == 0) {
            in_edge = 0;
            p += 7;
            continue;
        }
        if (in_edge && (size_t)(end - p) >= 6 && memcmp(p, "<lane ", 6) == 0) {
            const char* gt = tag_end(p, end);
            if (!gt) flow_die("unclosed <lane> in network");
            SumoLane lane;
            memset(&lane, 0, sizeof(lane));
            snprintf(lane.edge, sizeof(lane.edge), "%s", edge_id);
            lane.internal = internal;
            if (!attr_copy(p, gt, "id", lane.id, FLOW_NAME)) {
                flow_die("network lane is missing id");
            }
            char buf[4096];
            if (!attr_copy(p, gt, "index", buf, (int)sizeof(buf)) || !parse_i(buf, &lane.index)) {
                flow_die("network lane is missing index");
            }
            if (!attr_copy(p, gt, "speed", buf, (int)sizeof(buf))) {
                flow_die("network lane is missing speed");
            }
            double speed = 0;
            if (!parse_f(buf, &speed) || speed <= 0) flow_die("network lane has a bad speed");
            lane.speed = (float)speed;
            if (!attr_copy(p, gt, "shape", buf, (int)sizeof(buf))) {
                flow_die("network lane is missing shape");
            }
            lane.n = parse_shape(buf, &lane.x, &lane.y);
            if (lane.n < 0) flow_die("network lane has a bad shape");
            net_add_lane(net, &lane);
            p = gt + 1;
            continue;
        }
        if ((size_t)(end - p) >= 12 && memcmp(p, "<connection ", 12) == 0) {
            const char* gt = tag_end(p, end);
            if (!gt) flow_die("unclosed <connection> in network");
            SumoConn conn;
            memset(&conn, 0, sizeof(conn));
            char buf[64];
            if (!attr_copy(p, gt, "from", conn.from, FLOW_NAME) ||
                !attr_copy(p, gt, "to", conn.to, FLOW_NAME) ||
                !attr_copy(p, gt, "fromLane", buf, (int)sizeof(buf)) ||
                !parse_i(buf, &conn.from_lane) ||
                !attr_copy(p, gt, "toLane", buf, (int)sizeof(buf)) ||
                !parse_i(buf, &conn.to_lane)) {
                flow_die("network connection is missing endpoints");
            }
            attr_copy(p, gt, "via", conn.via, FLOW_NAME);
            net_add_conn(net, &conn);
            p = gt + 1;
            continue;
        }
        p++;
    }
    if (net->nlanes == 0) flow_die("network has no lanes");
}

static void net_free(SumoNet* net) {
    for (int i = 0; i < net->nlanes; i++) {
        free(net->lanes[i].x);
        free(net->lanes[i].y);
    }
    free(net->lanes);
    free(net->conns);
    memset(net, 0, sizeof(*net));
}

static const SumoLane* find_lane_id(const SumoNet* net, const char* id) {
    for (int i = 0; i < net->nlanes; i++) {
        if (strcmp(net->lanes[i].id, id) == 0) return &net->lanes[i];
    }
    return NULL;
}

static const SumoLane* find_edge_lane(const SumoNet* net, const char* edge, int index) {
    for (int i = 0; i < net->nlanes; i++) {
        const SumoLane* lane = &net->lanes[i];
        if (lane->internal) continue;
        if (lane->index == index && strcmp(lane->edge, edge) == 0) return lane;
    }
    return NULL;
}

static int edge_exists(const SumoNet* net, const char* edge) {
    for (int i = 0; i < net->nlanes; i++) {
        if (!net->lanes[i].internal && strcmp(net->lanes[i].edge, edge) == 0) return 1;
    }
    return 0;
}

static int collect_lane_indices(const SumoNet* net, const char* edge, int* indices, int max) {
    int n = 0;
    for (int i = 0; i < net->nlanes; i++) {
        const SumoLane* lane = &net->lanes[i];
        if (lane->internal || strcmp(lane->edge, edge) != 0) continue;
        if (n >= max) flow_die("departure edge has too many lanes");
        indices[n++] = lane->index;
    }
    for (int i = 1; i < n; i++) {
        int v = indices[i];
        int j = i;
        while (j > 0 && indices[j - 1] > v) {
            indices[j] = indices[j - 1];
            j--;
        }
        indices[j] = v;
    }
    return n;
}

static const SumoConn* find_conn(const SumoNet* net, const char* from, int from_lane,
        const char* to) {
    for (int i = 0; i < net->nconns; i++) {
        const SumoConn* c = &net->conns[i];
        if (c->from_lane == from_lane && strcmp(c->from, from) == 0 &&
            strcmp(c->to, to) == 0) {
            return c;
        }
    }
    return NULL;
}

static void path_push(FlowPath* path, const SumoLane* lane, float speed) {
    if (lane->n < 2 || speed <= 0.05f) return;
    int n = 0;
    float* xs = (float*)malloc((size_t)lane->n * sizeof(float));
    float* ys = (float*)malloc((size_t)lane->n * sizeof(float));
    if (!xs || !ys) {
        perror("malloc");
        exit(1);
    }
    for (int i = 0; i < lane->n; i++) {
        if (n > 0) {
            float dx = lane->x[i] - xs[n - 1];
            float dy = lane->y[i] - ys[n - 1];
            if (dx * dx + dy * dy < 0.02f * 0.02f) continue;
        }
        xs[n] = lane->x[i];
        ys[n] = lane->y[i];
        n++;
    }
    if (n < 2) {
        free(xs);
        free(ys);
        return;
    }
    float len = 0;
    for (int i = 0; i + 1 < n; i++) {
        float dx = xs[i + 1] - xs[i];
        float dy = ys[i + 1] - ys[i];
        len += sqrtf(dx * dx + dy * dy);
    }
    if (len < 1e-3f) {
        free(xs);
        free(ys);
        return;
    }
    if (path->n == 0) {
        path->pieces = NULL;
    }
    path->pieces = (FlowPiece*)realloc(path->pieces, (size_t)(path->n + 1) * sizeof(FlowPiece));
    if (!path->pieces) {
        perror("realloc");
        exit(1);
    }
    FlowPiece* pc = &path->pieces[path->n++];
    pc->x = xs;
    pc->y = ys;
    pc->n = n;
    pc->speed = speed;
    pc->len = len;
    path->len += len;
}

static void path_free(FlowPath* path) {
    for (int i = 0; i < path->n; i++) {
        free(path->pieces[i].x);
        free(path->pieces[i].y);
    }
    free(path->pieces);
    memset(path, 0, sizeof(*path));
}

static float piece_speed(const VTypeSpec* vt, const FlowSpec* flow, const SumoLane* lane) {
    float limit = lane->speed;
    float desired = vt->max_speed < limit ? vt->max_speed : limit;
    if (flow->speed_mode == SPEED_VALUE) {
        float v = flow->speed_value < vt->max_speed ? flow->speed_value : vt->max_speed;
        return v;
    }
    if (flow->speed_mode == SPEED_LIMIT) {
        return limit < vt->max_speed ? limit : vt->max_speed;
    }
    return desired;
}

static int build_path(const SumoNet* net, const RouteSpec* route, int lane_index,
        const VTypeSpec* vt, const FlowSpec* flow, FlowPath* path) {
    memset(path, 0, sizeof(*path));
    const SumoLane* lane = find_edge_lane(net, route->edges[0], lane_index);
    if (!lane) return 0;
    path_push(path, lane, piece_speed(vt, flow, lane));
    int cur = lane_index;
    for (int e = 0; e + 1 < route->nedges; e++) {
        const SumoConn* c = find_conn(net, route->edges[e], cur, route->edges[e + 1]);
        if (!c) break;
        if (c->via[0]) {
            const SumoLane* via = find_lane_id(net, c->via);
            if (!via) {
                fprintf(stderr, "structured init: missing via lane %s\n", c->via);
                exit(1);
            }
            path_push(path, via, piece_speed(vt, flow, via));
        }
        const SumoLane* to = find_edge_lane(net, c->to, c->to_lane);
        if (!to) break;
        path_push(path, to, piece_speed(vt, flow, to));
        cur = c->to_lane;
    }
    return path->n > 0 && path->len > 1.0f;
}

static double path_time_to(const FlowPath* path, float dist) {
    if (dist < 0) dist = 0;
    double walked = 0;
    double time = 0;
    for (int i = 0; i < path->n; i++) {
        if (walked >= dist) break;
        double take = path->pieces[i].len;
        if (walked + take > dist) take = (double)dist - walked;
        time += take / (double)path->pieces[i].speed;
        walked += take;
    }
    return time;
}

static double path_horizon(const FlowPath* path, float center0) {
    float s_max = path->len - MIN_DISTANCE_TO_GOAL - 1.0f;
    if (s_max <= center0 + 0.05f) return -1;
    return path_time_to(path, s_max) - path_time_to(path, center0);
}

// A personal max cannot outrun the piece. 0 keeps the piece speed.
static float piece_travel(const FlowPiece* pc, float vmax) {
    float v = pc->speed;
    if (vmax > 0.0f && vmax < v) {
        v = vmax;
    }
    return v;
}

static int path_move(const FlowPath* path, float s0, double time,
        float vmax, float* s_out) {
    double left = time;
    double walked = 0;
    double s = s0 < 0 ? 0 : s0;
    for (int i = 0; i < path->n; i++) {
        FlowPiece* pc = &path->pieces[i];
        double piece_end = walked + pc->len;
        if (s >= piece_end - 1e-4) {
            walked = piece_end;
            continue;
        }
        double s_in = s > walked ? s : walked;
        double room = piece_end - s_in;
        double spd = (double)piece_travel(pc, vmax);
        double dt = room / spd;
        if (left <= dt + 1e-7) {
            *s_out = (float)(s_in + left * spd);
            return 1;
        }
        left -= dt;
        s = piece_end;
        walked = piece_end;
    }
    return 0;
}

static int path_sample(const FlowPath* path, float dist, float* ox, float* oy, float* oh) {
    if (path->n <= 0) return 0;
    if (dist < 0) dist = 0;
    if (dist > path->len) dist = path->len;
    float walked = 0;
    for (int p = 0; p < path->n; p++) {
        FlowPiece* pc = &path->pieces[p];
        float local = dist - walked;
        if (local > pc->len + 1e-3f && p + 1 < path->n) {
            walked += pc->len;
            continue;
        }
        if (local < 0) local = 0;
        float acc = 0;
        for (int i = 0; i + 1 < pc->n; i++) {
            float dx = pc->x[i + 1] - pc->x[i];
            float dy = pc->y[i + 1] - pc->y[i];
            float seg = sqrtf(dx * dx + dy * dy);
            if (seg < 1e-6f) continue;
            if (acc + seg + 1e-4f >= local || i + 2 >= pc->n) {
                float t = (local - acc) / seg;
                if (t < 0) t = 0;
                if (t > 1) t = 1;
                *ox = pc->x[i] + dx * t;
                *oy = pc->y[i] + dy * t;
                *oh = atan2f(dy, dx);
                return 1;
            }
            acc += seg;
        }
        walked += pc->len;
    }
    return 0;
}

static float path_speed_at(const FlowPath* path, float s, float vmax) {
    double walked = 0;
    float spd = piece_travel(&path->pieces[0], vmax);
    for (int i = 0; i < path->n; i++) {
        spd = piece_travel(&path->pieces[i], vmax);
        walked += path->pieces[i].len;
        if (s <= (float)walked + 1e-3f) return spd;
    }
    return spd;
}

static const VTypeSpec* find_vtype(const VTypeSpec* vtypes, int n, const char* id) {
    for (int i = 0; i < n; i++) {
        if (strcmp(vtypes[i].id, id) == 0) return &vtypes[i];
    }
    return NULL;
}

static const RouteSpec* find_route(const RouteSpec* routes, int n, const char* id) {
    for (int i = 0; i < n; i++) {
        if (strcmp(routes[i].id, id) == 0) return &routes[i];
    }
    return NULL;
}

static void read_demand(Dict* cfg, VTypeSpec* vtypes, int* nv,
        RouteSpec* routes, int* nr, FlowSpec* flows, int* nf) {
    *nv = *nr = *nf = 0;
    for (int i = 0; i < FLOW_MAX; i++) {
        char key[32];
        snprintf(key, sizeof(key), "vtype_%d", i);
        const char* spec = cfg_str(cfg, key);
        if (!spec) break;
        char tok[8][FLOW_NAME];
        int n = split_csv(spec, tok, 8);
        VTypeSpec* vt = &vtypes[(*nv)++];
        memset(vt, 0, sizeof(*vt));
        double length = 0, width = 1.8, height = 1.5, max_speed = 0;
        if (n == 3) {
            if (!parse_f(tok[1], &length) || !parse_f(tok[2], &max_speed)) n = -1;
        } else if (n == 5) {
            if (!parse_f(tok[1], &length) || !parse_f(tok[2], &width) ||
                !parse_f(tok[3], &height) || !parse_f(tok[4], &max_speed)) n = -1;
        } else {
            n = -1;
        }
        if (n < 0 || length <= 0 || width <= 0 || height <= 0 || max_speed <= 0) {
            fprintf(stderr, "structured init: bad %s (want id,length,width,height,maxSpeed)\n", key);
            exit(1);
        }
        snprintf(vt->id, sizeof(vt->id), "%s", tok[0]);
        vt->length = (float)length;
        vt->width = (float)width;
        vt->height = (float)height;
        vt->max_speed = (float)max_speed;
    }
    for (int i = 0; i < FLOW_MAX; i++) {
        char key[32];
        snprintf(key, sizeof(key), "route_%d", i);
        const char* spec = cfg_str(cfg, key);
        if (!spec) break;
        char tok[1 + FLOW_MAX_EDGES][FLOW_NAME];
        int n = split_csv(spec, tok, 1 + FLOW_MAX_EDGES);
        if (n < 2) {
            fprintf(stderr, "structured init: bad %s (want id,edge,...)\n", key);
            exit(1);
        }
        RouteSpec* route = &routes[(*nr)++];
        memset(route, 0, sizeof(*route));
        snprintf(route->id, sizeof(route->id), "%s", tok[0]);
        route->nedges = n - 1;
        for (int e = 0; e < route->nedges; e++) {
            snprintf(route->edges[e], FLOW_NAME, "%s", tok[e + 1]);
        }
    }
    for (int i = 0; i < FLOW_MAX; i++) {
        char key[32];
        snprintf(key, sizeof(key), "flow_%d", i);
        const char* spec = cfg_str(cfg, key);
        if (!spec) break;
        char tok[8][FLOW_NAME];
        int n = split_csv(spec, tok, 8);
        if (n != 8) {
            fprintf(stderr,
                "structured init: bad %s (want id,type,begin,end,vehsPerHour,route,departLane,departSpeed)\n",
                key);
            exit(1);
        }
        FlowSpec* flow = &flows[(*nf)++];
        memset(flow, 0, sizeof(*flow));
        snprintf(flow->id, sizeof(flow->id), "%s", tok[0]);
        snprintf(flow->type_id, sizeof(flow->type_id), "%s", tok[1]);
        snprintf(flow->route_id, sizeof(flow->route_id), "%s", tok[5]);
        snprintf(flow->lane_token, sizeof(flow->lane_token), "%s", tok[6]);
        snprintf(flow->speed_token, sizeof(flow->speed_token), "%s", tok[7]);
        if (!parse_f(tok[2], &flow->begin) || !parse_f(tok[3], &flow->end) ||
            !parse_f(tok[4], &flow->vehs_per_hour) ||
            !(flow->end > flow->begin) || !(flow->vehs_per_hour > 0) || flow->begin < 0) {
            fprintf(stderr, "structured init: bad times or vehsPerHour in %s\n", key);
            exit(1);
        }
        if (strcmp(tok[6], "random") == 0 || strcmp(tok[6], "free") == 0) {
            flow->lane_mode = LANE_SPLIT;
        } else if (strcmp(tok[6], "best") == 0) {
            flow->lane_mode = LANE_BEST;
        } else if (strcmp(tok[6], "first") == 0 || strcmp(tok[6], "allowed") == 0) {
            flow->lane_mode = LANE_FIXED;
            flow->lane_index = 0;
        } else if (parse_i(tok[6], &flow->lane_index) && flow->lane_index >= 0) {
            flow->lane_mode = LANE_FIXED;
        } else {
            fprintf(stderr, "structured init: unsupported departLane '%s' in %s\n", tok[6], key);
            exit(1);
        }
        double speed_value = 0;
        if (strcmp(tok[7], "desired") == 0 || strcmp(tok[7], "max") == 0) {
            flow->speed_mode = SPEED_DESIRED;
        } else if (strcmp(tok[7], "speedLimit") == 0) {
            flow->speed_mode = SPEED_LIMIT;
        } else if (parse_f(tok[7], &speed_value) && speed_value > 0) {
            flow->speed_mode = SPEED_VALUE;
            flow->speed_value = (float)speed_value;
        } else {
            fprintf(stderr, "structured init: unsupported departSpeed '%s' in %s\n", tok[7], key);
            exit(1);
        }
        if (!find_vtype(vtypes, *nv, flow->type_id)) {
            fprintf(stderr, "structured init: flow %s type '%s' is not defined\n",
                flow->id, flow->type_id);
            exit(1);
        }
        if (!find_route(routes, *nr, flow->route_id)) {
            fprintf(stderr, "structured init: flow %s route '%s' is not defined\n",
                flow->id, flow->route_id);
            exit(1);
        }
    }
    if (*nf == 0) flow_die("set flow_0 (and route_0, vtype_0) in the ini");
}

static void allocate_slots(const int* natural, int n, int cap, int* out) {
    int total = 0;
    for (int i = 0; i < n; i++) total += natural[i];
    if (total <= cap) {
        for (int i = 0; i < n; i++) out[i] = natural[i];
        return;
    }
    double frac[FLOW_MAX];
    int used = 0;
    for (int i = 0; i < n; i++) {
        double exact = (double)natural[i] * (double)cap / (double)total;
        out[i] = (int)floor(exact);
        if (out[i] > natural[i]) out[i] = natural[i];
        frac[i] = exact - out[i];
        used += out[i];
    }
    while (used < cap) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (out[i] >= natural[i]) continue;
            if (best < 0 || frac[i] > frac[best]) best = i;
        }
        if (best < 0) break;
        out[best]++;
        frac[best] = 0;
        used++;
    }
}

static float* one_float(float v) {
    float* p = (float*)malloc(sizeof(float));
    if (!p) {
        perror("malloc");
        exit(1);
    }
    *p = v;
    return p;
}

static int* one_int(int v) {
    int* p = (int*)malloc(sizeof(int));
    if (!p) {
        perror("malloc");
        exit(1);
    }
    *p = v;
    return p;
}

static void fill_vehicle(Entity* e, const SpawnVeh* s) {
    memset(e, 0, sizeof(*e));
    const VTypeSpec* vt = s->vtype;
    float heading = normalize_heading(s->heading);
    e->type = VEHICLE;
    e->array_size = 1;
    e->traj_x = one_float(s->x);
    e->traj_y = one_float(s->y);
    e->traj_z = one_float(0);
    e->traj_vx = one_float(s->speed * cosf(heading));
    e->traj_vy = one_float(s->speed * sinf(heading));
    e->traj_vz = one_float(0);
    e->traj_heading = one_float(heading);
    e->traj_valid = one_int(1);
    e->width = vt->width;
    e->length = vt->length;
    e->height = vt->height;
    e->goal_position_x = s->goal_x;
    e->goal_position_y = s->goal_y;
    e->goal_position_z = 0;
    e->mark_as_expert = 0;
    e->respawn_timestep = -1;
    e->max_speed = s->max_speed;
}

static float flow_goal_distance(Drive* env) {
    float goal = 150.0f;
    DictItem* item = dict_find(env->cfg, "flow_goal_distance");
    if (item && item->value > 0) goal = (float)item->value;
    if (goal < MIN_DISTANCE_TO_GOAL) goal = MIN_DISTANCE_TO_GOAL;
    return goal;
}

static void plan_flow(const SumoNet* net, const FlowSpec* flow, const RouteSpec* route,
        const VTypeSpec* vt, FlowPlan* plan) {
    memset(plan, 0, sizeof(*plan));
    plan->vtype = vt;
    int indices[FLOW_MAX_LANES];
    int nlanes = collect_lane_indices(net, route->edges[0], indices, FLOW_MAX_LANES);
    if (nlanes <= 0) {
        fprintf(stderr, "structured init: route %s edge %s has no lanes\n",
            route->id, route->edges[0]);
        exit(1);
    }
    if (flow->lane_mode == LANE_FIXED) {
        int ok = 0;
        for (int i = 0; i < nlanes; i++) ok = ok || indices[i] == flow->lane_index;
        if (!ok) {
            fprintf(stderr, "structured init: flow %s departLane %d is outside 0..%d\n",
                flow->id, flow->lane_index, nlanes - 1);
            exit(1);
        }
        indices[0] = flow->lane_index;
        nlanes = 1;
    }
    float center0 = vt->length * 0.5f + DEPART_POS_EPS;
    int best = 0;
    double best_h = -1;
    for (int i = 0; i < nlanes; i++) {
        if (!build_path(net, route, indices[i], vt, flow, &plan->paths[plan->npaths])) {
            fprintf(stderr, "structured init: flow %s cannot follow lane %d of %s\n",
                flow->id, indices[i], route->edges[0]);
            exit(1);
        }
        plan->lane_of[plan->npaths] = indices[i];
        double h = path_horizon(&plan->paths[plan->npaths], center0);
        if (flow->lane_mode == LANE_BEST && h > best_h) {
            best_h = h;
            best = plan->npaths;
        }
        plan->npaths++;
    }
    if (flow->lane_mode == LANE_BEST) {
        for (int i = 0; i < plan->npaths; i++) {
            if (i != best) path_free(&plan->paths[i]);
        }
        if (best != 0) {
            plan->paths[0] = plan->paths[best];
            memset(&plan->paths[best], 0, sizeof(plan->paths[best]));
        }
        plan->lane_of[0] = plan->lane_of[best];
        plan->npaths = 1;
    }
    plan->horizon = 1e300;
    for (int i = 0; i < plan->npaths; i++) {
        double h = path_horizon(&plan->paths[i], center0);
        if (h < plan->horizon) plan->horizon = h;
    }
    if (plan->horizon > 1e-3) plan->horizon -= 1e-3;
    plan->i0 = 0;
    plan->i1 = 0;
    if (!(plan->horizon > 0)) return;
    double period = 3600.0 / flow->vehs_per_hour;
    double t_lo = flow->end - plan->horizon;
    if (t_lo < flow->begin) t_lo = flow->begin;
    double t_hi = flow->end;
    if (!(t_hi > t_lo)) return;
    int i0 = (int)ceil((t_lo - flow->begin) / period - 1e-9);
    int i1 = (int)ceil((t_hi - flow->begin) / period - 1e-9);
    if (i0 < 0) i0 = 0;
    if (i1 < i0) i1 = i0;
    if (i1 - i0 > 1000000) {
        fprintf(stderr, "structured init: flow %s places more than 1000000 vehicles\n", flow->id);
        exit(1);
    }
    plan->i0 = i0;
    plan->i1 = i1;
}

static void plan_free(FlowPlan* plan) {
    for (int i = 0; i < plan->npaths; i++) path_free(&plan->paths[i]);
    plan->npaths = 0;
}

static FlowPath path_copy(const FlowPath* src) {
    FlowPath dst;
    memset(&dst, 0, sizeof(dst));
    dst.n = src->n;
    dst.len = src->len;
    if (src->n <= 0) return dst;
    dst.pieces = (FlowPiece*)calloc((size_t)src->n, sizeof(FlowPiece));
    if (!dst.pieces) {
        perror("calloc");
        exit(1);
    }
    for (int i = 0; i < src->n; i++) {
        FlowPiece* sp = &src->pieces[i];
        FlowPiece* dp = &dst.pieces[i];
        dp->n = sp->n;
        dp->speed = sp->speed;
        dp->len = sp->len;
        dp->x = (float*)malloc((size_t)sp->n * sizeof(float));
        dp->y = (float*)malloc((size_t)sp->n * sizeof(float));
        if (!dp->x || !dp->y) {
            perror("malloc");
            exit(1);
        }
        memcpy(dp->x, sp->x, (size_t)sp->n * sizeof(float));
        memcpy(dp->y, sp->y, (size_t)sp->n * sizeof(float));
    }
    return dst;
}

static void route_bind(Drive* env, FlowPlan* plans, int nf, SpawnVeh* spawns, int nspawn,
        const FlowSpec* flows, const double* base_vph) {
    int npaths = 0;
    for (int f = 0; f < nf; f++) npaths += plans[f].npaths;
    if (npaths <= 0) flow_die("structured init has no routes");
    FlowRoutes* lanes = (FlowRoutes*)calloc(1, sizeof(FlowRoutes));
    if (!lanes) {
        perror("calloc");
        exit(1);
    }
    lanes->npaths = npaths;
    lanes->nveh = nspawn;
    lanes->paths = (FlowPath*)calloc((size_t)npaths, sizeof(FlowPath));
    lanes->path_of = (int*)calloc((size_t)nspawn, sizeof(int));
    lanes->s0 = (float*)calloc((size_t)nspawn, sizeof(float));
    if (!lanes->paths || !lanes->path_of || !lanes->s0) {
        perror("calloc");
        exit(1);
    }
    int k = 0;
    lanes->nflows = nf;
    for (int f = 0; f < nf; f++) {
        lanes->base_vph[f] = base_vph[f];
        lanes->begin[f] = flows[f].begin;
        lanes->end[f] = flows[f].end;
        lanes->length[f] = plans[f].vtype ? plans[f].vtype->length : 4.3f;
        lanes->path0[f] = k;
        lanes->npath[f] = plans[f].npaths;
        for (int i = 0; i < plans[f].npaths; i++) {
            lanes->paths[k++] = path_copy(&plans[f].paths[i]);
        }
    }
    for (int i = 0; i < nspawn; i++) {
        if (spawns[i].path < 0 || spawns[i].path >= npaths) {
            flow_die("vehicle route index is out of range");
        }
        lanes->path_of[i] = spawns[i].path;
        lanes->s0[i] = spawns[i].along;
    }
    env->lane_state = lanes;
}

static void route_free(Drive* env) {
    FlowRoutes* lanes = (FlowRoutes*)env->lane_state;
    if (!lanes) return;
    for (int p = 0; p < lanes->npaths; p++) path_free(&lanes->paths[p]);
    free(lanes->paths);
    free(lanes->path_of);
    free(lanes->s0);
    free(lanes);
    env->lane_state = NULL;
}

static void route_shift(Drive* env) {
    FlowRoutes* lanes = (FlowRoutes*)env->lane_state;
    if (!lanes) return;
    float mx = env->world_mean_x;
    float my = env->world_mean_y;
    for (int p = 0; p < lanes->npaths; p++) {
        FlowPath* path = &lanes->paths[p];
        for (int i = 0; i < path->n; i++) {
            FlowPiece* pc = &path->pieces[i];
            for (int k = 0; k < pc->n; k++) {
                pc->x[k] -= mx;
                pc->y[k] -= my;
            }
        }
    }
}

// Train draws a goal from 40 m ahead to the route end, so some trips
// are short and some run the length of the road. Inference puts every
// goal on the last point of that car's route. path_sample is centered.
static void assign_episode_goals(Drive* env) {
    FlowRoutes* lanes = (FlowRoutes*)env->lane_state;
    if (!lanes || env->agent_initialization != AGENT_INIT_STRUCTURED) {
        return;
    }
    int inference = env->drive_mode == DRIVE_MODE_INFERENCE;
    int n = env->active_agent_count;
    if (n > lanes->nveh) {
        n = lanes->nveh;
    }
    for (int i = 0; i < n; i++) {
        int agent = env->active_agent_indices[i];
        if (agent < 0 || agent >= lanes->nveh) {
            continue;
        }
        FlowPath* path = &lanes->paths[lanes->path_of[agent]];
        float s0 = lanes->s0[agent];
        float s_far = path->len;
        float goal_s = s_far;
        if (!inference) {
            float s_near = s0 + 40.0f;
            if (s_near > s_far) {
                s_near = s_far;
            }
            float u = (float)(rand_r(&env->rng) % 10001) / 10000.0f;
            goal_s = s_near + u * (s_far - s_near);
        }
        float gx, gy, gh;
        if (!path_sample(path, goal_s, &gx, &gy, &gh)) {
            continue;
        }
        Entity* e = &env->entities[agent];
        e->goal_position_x = gx;
        e->goal_position_y = gy;
        e->goal_position_z = 0.0f;
    }
}

static int plan_demand(const SumoNet* net, FlowSpec* flows, int nf,
        const RouteSpec* routes, int nr, const VTypeSpec* vtypes, int nv,
        FlowPlan* plans, int* natural) {
    int demand = 0;
    for (int f = 0; f < nf; f++) {
        const RouteSpec* route = find_route(routes, nr, flows[f].route_id);
        const VTypeSpec* vt = find_vtype(vtypes, nv, flows[f].type_id);
        plan_flow(net, &flows[f], route, vt, &plans[f]);
        natural[f] = plans[f].i1 - plans[f].i0;
        if (natural[f] < 0) natural[f] = 0;
        demand += natural[f];
    }
    return demand;
}

// Natural demand is the t=end snapshot. A larger request tightens headway
// on the same routes and replans, then allocate_slots trims to cap.
static void scale_flows_to_cap(const SumoNet* net, FlowSpec* flows, int nf,
        const RouteSpec* routes, int nr, const VTypeSpec* vtypes, int nv,
        FlowPlan* plans, int* natural, int* demand, int cap) {
    if (*demand <= 0 || *demand >= cap) return;
    FlowSpec base[FLOW_MAX];
    memcpy(base, flows, (size_t)nf * sizeof(FlowSpec));
    double scale = (double)cap / (double)(*demand);
    for (int attempt = 0; attempt < 6 && *demand < cap; attempt++) {
        for (int f = 0; f < nf; f++) {
            plan_free(&plans[f]);
            flows[f] = base[f];
            flows[f].vehs_per_hour = base[f].vehs_per_hour * scale;
        }
        *demand = plan_demand(net, flows, nf, routes, nr, vtypes, nv,
            plans, natural);
        if (*demand <= 0) break;
        if (*demand < cap) {
            double next = scale * (double)cap / (double)(*demand);
            if (next < scale * 1.01) next = scale * 1.05;
            scale = next;
        }
    }
}

// Uniform on [lo, hi], cut to 1..100. A zero-width band returns lo
// and does not touch the rng, so goal draws stay put.
static float sample_target_speed(Drive* env, float lo, float hi) {
    if (lo > hi) {
        float t = lo;
        lo = hi;
        hi = t;
    }
    if (lo < SIM_SPEED_MIN) lo = SIM_SPEED_MIN;
    if (hi > MAX_SPEED) hi = MAX_SPEED;
    if (hi < lo) hi = lo;
    if (hi - lo <= 0.005f) return lo;
    float u = (float)(rand_r(&env->rng) % 10001) / 10000.0f;
    return lo + (hi - lo) * u;
}

// One target per car for the episode. The car starts at that speed.
// A respawn keeps the target and returns to it.
static void assign_episode_speeds(Drive* env) {
    FlowRoutes* lanes = (FlowRoutes*)env->lane_state;
    if (!lanes || env->agent_initialization != AGENT_INIT_STRUCTURED) {
        return;
    }
    float lo = env->sim_speed_on;
    float hi = env->sim_hi_on;
    int n = env->active_agent_count;
    if (n > lanes->nveh) n = lanes->nveh;
    for (int i = 0; i < n; i++) {
        int agent = env->active_agent_indices[i];
        if (agent < 0 || agent >= lanes->nveh) continue;
        Entity* e = &env->entities[agent];
        float v = sample_target_speed(env, lo, hi);
        e->max_speed = v;
        e->vx = v * e->heading_x;
        e->vy = v * e->heading_y;
    }
}

// Slider rate onto each flow. Flow 0 is the mainline. The rest keep
// base_vph[f] / base_vph[0].
static void flow_write_rates(FlowRoutes* lanes, int main_vph) {
    double base0 = lanes->nflows > 0 ? lanes->base_vph[0] : 0;
    double scale = 0;
    if (base0 > 1e-6 && main_vph > 0) {
        scale = (double)main_vph / base0;
    }
    for (int f = 0; f < lanes->nflows; f++) {
        lanes->vph[f] = lanes->base_vph[f] * scale;
    }
}

// First depart is the first SUMO slot at or after the warm-start time.
// That is one period after the youngest car already on the road.
static void flow_schedule_reset(Drive* env) {
    FlowRoutes* lanes = (FlowRoutes*)env->lane_state;
    if (!lanes || lanes->nflows <= 0) {
        return;
    }
    int main = env->sim_vph_on;
    if (main < 0) {
        main = 0;
    }
    if (main > DRIVE_VPH_MAX) {
        main = DRIVE_VPH_MAX;
    }
    flow_write_rates(lanes, main);
    for (int f = 0; f < lanes->nflows; f++) {
        lanes->pending[f] = 0;
        lanes->cursor[f] = 0;
        double vph = lanes->vph[f];
        if (vph <= 1e-6) {
            lanes->next_depart[f] = 1e300;
            continue;
        }
        double period = 3600.0 / vph;
        double span = lanes->end[f] - lanes->begin[f];
        double k = ceil(span / period - 1e-9);
        double next = lanes->begin[f] + k * period;
        double delay = next - lanes->end[f];
        if (delay < 0) {
            delay = 0;
        }
        lanes->next_depart[f] = delay;
    }
    if (env->drive_mode == DRIVE_MODE_INFERENCE && !env->stream_banner) {
        env->stream_banner = 1;
        fprintf(stderr, "inference: goals at the route end, insert");
        for (int f = 0; f < lanes->nflows; f++) {
            fprintf(stderr, " %.0f", lanes->vph[f]);
        }
        fprintf(stderr, " veh/h\n");
    }
}

// Live demand change. The cars already on the road stay. The next
// entrance attempt is one period from now, so a drag does not dump a queue.
static void flow_set_vph(Drive* env, int main_vph) {
    if (main_vph < 0) {
        main_vph = 0;
    }
    if (main_vph > DRIVE_VPH_MAX) {
        main_vph = DRIVE_VPH_MAX;
    }
    env->sim_vph_req = main_vph;
    env->sim_vph_on = main_vph;
    FlowRoutes* lanes = (FlowRoutes*)env->lane_state;
    if (!lanes || lanes->nflows <= 0) {
        return;
    }
    flow_write_rates(lanes, main_vph);
    double now = (double)env->timestep * (double)SIM_DT;
    for (int f = 0; f < lanes->nflows; f++) {
        lanes->pending[f] = 0;
        double vph = lanes->vph[f];
        if (vph <= 1e-6) {
            lanes->next_depart[f] = 1e300;
            continue;
        }
        lanes->next_depart[f] = now + 3600.0 / vph;
    }
}

static int flow_gap_clear(Drive* env, float x, float y) {
    float lim = FLOW_INSERT_GAP * FLOW_INSERT_GAP;
    int n = env->active_agent_count;
    if (n > DRIVE_AGENT_MAX) {
        n = DRIVE_AGENT_MAX;
    }
    for (int i = 0; i < n; i++) {
        int idx = env->active_agent_indices[i];
        if (idx < 0) {
            continue;
        }
        Entity* o = &env->entities[idx];
        if (o->x == INVALID_POSITION || o->y == INVALID_POSITION) {
            continue;
        }
        float dx = o->x - x;
        float dy = o->y - y;
        if (dx * dx + dy * dy < lim) {
            return 0;
        }
    }
    return 1;
}

static int flow_free_slot(Drive* env) {
    int n = env->active_agent_count;
    if (n > DRIVE_AGENT_MAX) {
        n = DRIVE_AGENT_MAX;
    }
    for (int i = 0; i < n; i++) {
        int idx = env->active_agent_indices[i];
        if (idx < 0) {
            continue;
        }
        if (env->entities[idx].x == INVALID_POSITION) {
            return i;
        }
    }
    return -1;
}

// One due vehicle per flow, at the route start, if a slot is free and
// the entrance has FLOW_INSERT_GAP of room. SUMO would queue the rest.
static void flow_inject(Drive* env) {
    FlowRoutes* lanes = (FlowRoutes*)env->lane_state;
    if (!lanes || env->drive_mode != DRIVE_MODE_INFERENCE) {
        return;
    }
    double now = (double)env->timestep * (double)SIM_DT;
    for (int f = 0; f < lanes->nflows; f++) {
        double vph = lanes->vph[f];
        if (vph <= 1e-6 || lanes->npath[f] <= 0) {
            continue;
        }
        double period = 3600.0 / vph;
        while (lanes->pending[f] < FLOW_QUEUE
                && now + 1e-9 >= lanes->next_depart[f]) {
            lanes->pending[f]++;
            lanes->next_depart[f] += period;
        }
        if (lanes->pending[f] <= 0) {
            continue;
        }
        int slot = flow_free_slot(env);
        if (slot < 0) {
            continue;
        }
        int npath = lanes->npath[f];
        for (int k = 0; k < npath; k++) {
            int lane_i = (lanes->cursor[f] + k) % npath;
            int pi = lanes->path0[f] + lane_i;
            if (pi < 0 || pi >= lanes->npaths) {
                continue;
            }
            FlowPath* path = &lanes->paths[pi];
            float center0 = lanes->length[f] * 0.5f + DEPART_POS_EPS;
            float x, y, heading;
            if (!path_sample(path, center0, &x, &y, &heading)) {
                continue;
            }
            if (!flow_gap_clear(env, x, y)) {
                continue;
            }
            float gx, gy, gh;
            if (!path_sample(path, path->len, &gx, &gy, &gh)) {
                continue;
            }
            (void)gh;
            int agent = env->active_agent_indices[slot];
            Entity* e = &env->entities[agent];
            heading = normalize_heading(heading);
            float spd = sample_target_speed(env, env->sim_speed_on, env->sim_hi_on);
            e->x = x;
            e->y = y;
            e->z = 0;
            e->heading = heading;
            e->heading_x = cosf(heading);
            e->heading_y = sinf(heading);
            e->max_speed = spd;
            e->vx = spd * e->heading_x;
            e->vy = spd * e->heading_y;
            e->vz = 0;
            e->goal_position_x = gx;
            e->goal_position_y = gy;
            e->goal_position_z = 0;
            e->origin_x = x;
            e->origin_y = y;
            e->reached_goal = 0;
            e->collided_before_goal = 0;
            e->reached_goal_this_episode = 0;
            e->collision_state = NO_COLLISION;
            e->prev_accel = 0;
            e->prev_steer = 0;
            e->respawn_timestep = env->timestep;
            e->valid = 1;
            if (agent < lanes->nveh) {
                lanes->path_of[agent] = pi;
                lanes->s0[agent] = center0;
            }
            lanes->cursor[f] = lane_i + 1;
            lanes->pending[f]--;
            env->stream_in++;
            break;
        }
    }
}

static void spawn_flow_vehicles(Drive* env) {
    if (!env->cfg) flow_die("missing env config");
    char net_path[512];
    resolve_network(env, net_path, (int)sizeof(net_path));
    size_t xml_n = 0;
    char* xml = read_whole(net_path, &xml_n);
    if (!xml) {
        fprintf(stderr, "structured init: cannot read %s\n", net_path);
        exit(1);
    }
    SumoNet net;
    parse_net(&net, xml, xml_n);
    free(xml);

    VTypeSpec vtypes[FLOW_MAX];
    RouteSpec routes[FLOW_MAX];
    FlowSpec flows[FLOW_MAX];
    int nv = 0, nr = 0, nf = 0;
    read_demand(env->cfg, vtypes, &nv, routes, &nr, flows, &nf);
    // First load records the vtype cap and the uniform ends.
    // Later loads keep whatever Play committed.
    if (env->sim_speed_max <= 0.0f && nv > 0) {
        float cap = vtypes[0].max_speed;
        if (cap < SIM_SPEED_MIN) cap = SIM_SPEED_MIN;
        if (cap > MAX_SPEED) cap = MAX_SPEED;
        env->sim_speed_max = cap;
        float lo = cap;
        float hi = cap;
        if (env->cfg) {
            DictItem* lo_item = dict_find(env->cfg, "target_speed_min");
            if (lo_item) lo = (float)lo_item->value;
            DictItem* hi_item = dict_find(env->cfg, "target_speed_max");
            if (hi_item) hi = (float)hi_item->value;
        }
        if (lo < SIM_SPEED_MIN) lo = SIM_SPEED_MIN;
        if (lo > MAX_SPEED) lo = MAX_SPEED;
        if (hi < SIM_SPEED_MIN) hi = SIM_SPEED_MIN;
        if (hi > MAX_SPEED) hi = MAX_SPEED;
        env->sim_speed_req = lo;
        env->sim_speed_on = lo;
        env->sim_hi_req = hi;
        env->sim_hi_on = hi;
    }
    for (int r = 0; r < nr; r++) {
        for (int e = 0; e < routes[r].nedges; e++) {
            if (!edge_exists(&net, routes[r].edges[e])) {
                fprintf(stderr, "structured init: route %s edge '%s' is not in %s\n",
                    routes[r].id, routes[r].edges[e], net_path);
                exit(1);
            }
        }
    }

    double base_vph[FLOW_MAX];
    for (int f = 0; f < nf; f++) {
        base_vph[f] = flows[f].vehs_per_hour;
    }
    if (!env->sim_vph_ready && nf > 0) {
        int v = (int)(base_vph[0] + 0.5);
        if (v < 0) {
            v = 0;
        }
        if (v > DRIVE_VPH_MAX) {
            v = DRIVE_VPH_MAX;
        }
        env->sim_vph_on = v;
        env->sim_vph_req = v;
        env->sim_vph_ready = 1;
    }
    FlowPlan plans[FLOW_MAX];
    memset(plans, 0, sizeof(plans));
    int natural[FLOW_MAX];
    int demand = plan_demand(&net, flows, nf, routes, nr, vtypes, nv,
        plans, natural);
    if (demand <= 0) {
        flow_die("flows place no vehicles on the network");
    }

    int cap = env->max_agents > 0 ? env->max_agents : MAX_AGENTS;
    if (cap < 1) cap = 1;
    if (cap > DRIVE_AGENT_MAX) cap = DRIVE_AGENT_MAX;
    scale_flows_to_cap(&net, flows, nf, routes, nr, vtypes, nv,
        plans, natural, &demand, cap);
    int alloc[FLOW_MAX];
    allocate_slots(natural, nf, cap, alloc);

    int nspawn = 0;
    for (int f = 0; f < nf; f++) nspawn += alloc[f];
    if (nspawn <= 0) flow_die("agent cap left no vehicles");
    SpawnVeh* spawns = (SpawnVeh*)calloc((size_t)nspawn, sizeof(SpawnVeh));
    if (!spawns) {
        perror("calloc");
        exit(1);
    }
    float goal_dist = flow_goal_distance(env);
    int path_off[FLOW_MAX];
    int path_count = 0;
    for (int f = 0; f < nf; f++) {
        path_off[f] = path_count;
        path_count += plans[f].npaths;
    }
    int written = 0;
    for (int f = 0; f < nf; f++) {
        if (alloc[f] <= 0) continue;
        FlowPlan* plan = &plans[f];
        const VTypeSpec* vt = plan->vtype;
        float center0 = vt->length * 0.5f + DEPART_POS_EPS;
        double period = 3600.0 / flows[f].vehs_per_hour;
        int keep_from = plan->i1 - alloc[f];
        if (keep_from < plan->i0) keep_from = plan->i0;
        int offset = plan->npaths > 0 ? (int)(env->rng % (unsigned)plan->npaths) : 0;
        for (int i = keep_from; i < plan->i1; i++) {
            double age = flows[f].end - (flows[f].begin + (double)i * period);
            int lane_i = (flows[f].lane_mode == LANE_SPLIT) ? (i + offset) % plan->npaths : 0;
            FlowPath* path = &plan->paths[lane_i];
            // Lane speed places the car. Reset assigns the target.
            float center_s = 0;
            if (!path_move(path, center0, age, MAX_SPEED, &center_s)) continue;
            float goal_s = center_s + goal_dist;
            if (goal_s > path->len) goal_s = path->len;
            float x, y, heading, gx, gy, goal_heading;
            if (!path_sample(path, center_s, &x, &y, &heading) ||
                !path_sample(path, goal_s, &gx, &gy, &goal_heading)) {
                continue;
            }
            (void)goal_heading;
            float dx = gx - x;
            float dy = gy - y;
            if (sqrtf(dx * dx + dy * dy) < MIN_DISTANCE_TO_GOAL) continue;
            SpawnVeh* s = &spawns[written++];
            s->x = x;
            s->y = y;
            s->heading = heading;
            s->speed = path_speed_at(path, center_s, MAX_SPEED);
            s->max_speed = 0.0f;
            s->goal_x = gx;
            s->goal_y = gy;
            s->flow = f;
            s->path = path_off[f] + lane_i;
            s->along = center_s;
            s->age = age;
            s->vtype = vt;
        }
    }
    if (written <= 0) flow_die("agent cap left no vehicles");
    if (written != nspawn) {
        fprintf(stderr, "structured init: placed %d of %d vehicles\n",
            written, nspawn);
        nspawn = written;
    }

    static int flow_init_logged = 0;
    if (!flow_init_logged && cap == MAX_AGENTS) {
        flow_init_logged = 1;
        printf("structured init: %d vehicles (demand %d, cap %d) from %s\n",
            nspawn, demand, cap, net_path);
        for (int f = 0; f < nf; f++) {
            printf("  flow %s: %d/%d  %.4g veh/h  route %s  t=%.4g..%.4g  departLane %s  departSpeed %s\n",
                flows[f].id, alloc[f], natural[f], flows[f].vehs_per_hour, flows[f].route_id,
                flows[f].begin, flows[f].end, flows[f].lane_token, flows[f].speed_token);
        }
    }

    int nroads = env->num_roads;
    Entity* neu = (Entity*)calloc((size_t)nspawn + (size_t)nroads, sizeof(Entity));
    if (!neu) {
        perror("calloc");
        exit(1);
    }
    for (int i = 0; i < nspawn; i++) fill_vehicle(&neu[i], &spawns[i]);
    int primary = 0;
    for (int i = 1; i < nspawn; i++) {
        int better = spawns[i].flow < spawns[primary].flow ||
            (spawns[i].flow == spawns[primary].flow && spawns[i].age < spawns[primary].age);
        if (better) primary = i;
    }
    if (primary != nspawn - 1) {
        Entity tmp = neu[primary];
        neu[primary] = neu[nspawn - 1];
        neu[nspawn - 1] = tmp;
        SpawnVeh stmp = spawns[primary];
        spawns[primary] = spawns[nspawn - 1];
        spawns[nspawn - 1] = stmp;
    }
    // Keep the routes. Reset samples a goal on each one.
    route_bind(env, plans, nf, spawns, nspawn, flows, base_vph);
    if (nroads > 0) {
        memcpy(neu + nspawn, env->entities + env->num_objects, (size_t)nroads * sizeof(Entity));
    }
    for (int i = 0; i < env->num_objects; i++) free_entity(&env->entities[i]);
    free(env->entities);
    env->entities = neu;
    env->num_objects = nspawn;
    env->num_entities = nspawn + nroads;

    free(spawns);
    for (int f = 0; f < nf; f++) plan_free(&plans[f]);
    net_free(&net);
}
