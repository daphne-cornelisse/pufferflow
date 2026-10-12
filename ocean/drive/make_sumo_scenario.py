"""Build a drive scenario from smallnet.net.xml.

Writes data/bottleneck/json/smallnet.json, converts it with dataset.py into
data/bottleneck/binaries/map_000.bin, and saves a top-down preview.

The trainer loads this from config/drive.ini:
  ./puffer train --headless
"""

import math
import os
import xml.etree.ElementTree as ET

from ocean.drive.dataset import (
    LANE_FREEWAY,
    ROAD_EDGE_BOUNDARY,
    ROAD_LINE_BROKEN_SINGLE_WHITE,
    save_map_binary,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NET_PATH = os.path.join(ROOT, "smallnet.net.xml")
JSON_DIR = os.path.join(ROOT, "data", "bottleneck", "json")
BIN_DIR = os.path.join(ROOT, "data", "bottleneck", "binaries")
PREVIEW_PATH = os.path.join(ROOT, "data", "bottleneck", "preview.png")

LANE_WIDTH = 3.2
HALF = LANE_WIDTH / 2.0
SPACING = 5.0
MAX_PTS = 10
TRAJECTORY_LENGTH = 91
DT = 0.1
CAR = {"width": 2.0, "length": 4.8, "height": 1.5}


def parse_shape(text):
    pts = []
    for token in text.split():
        x, y = token.split(",")
        pts.append((float(x), float(y)))
    return pts


def dedupe(pts, tol=0.05):
    out = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append(p)
    return out


def path_length(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def resample(pts, spacing):
    pts = dedupe(pts)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    acc = 0.0
    target = spacing
    for a, b in zip(pts, pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-9:
            continue
        ux, uy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
        while acc + seg >= target - 1e-6:
            d = target - acc
            out.append((a[0] + ux * d, a[1] + uy * d))
            target += spacing
        acc += seg
    if math.hypot(out[-1][0] - pts[-1][0], out[-1][1] - pts[-1][1]) > 0.05:
        out.append(pts[-1])
    return out


def left_normals(pts):
    dirs = []
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        dirs.append((dx / length, dy / length))
    normals = []
    for i in range(len(pts)):
        if i == 0:
            dx, dy = dirs[0]
        elif i == len(pts) - 1:
            dx, dy = dirs[-1]
        else:
            dx = dirs[i - 1][0] + dirs[i][0]
            dy = dirs[i - 1][1] + dirs[i][1]
            length = math.hypot(dx, dy) or 1.0
            dx, dy = dx / length, dy / length
        normals.append((-dy, dx))
    return normals


def offset(pts, dist):
    return [
        (p[0] + dist * n[0], p[1] + dist * n[1])
        for p, n in zip(pts, left_normals(pts))
    ]


def sample_at(pts, distance):
    distance = max(0.0, min(distance, path_length(pts)))
    walked = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if walked + seg >= distance - 1e-8:
            t = 0.0 if seg < 1e-9 else (distance - walked) / seg
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            heading = math.atan2(b[1] - a[1], b[0] - a[0])
            return (x, y), heading
        walked += seg
    a, b = pts[-2], pts[-1]
    return b, math.atan2(b[1] - a[1], b[0] - a[0])


def chunks(pts):
    pts = resample(dedupe(pts), SPACING)
    if len(pts) < 2:
        return []
    pieces = []
    i = 0
    last = len(pts) - 1
    while i < last:
        j = min(i + MAX_PTS - 1, last)
        pieces.append(pts[i : j + 1])
        if j == last:
            break
        i = j
    return pieces


def load_lanes(path):
    root = ET.parse(path).getroot()
    lanes = {}
    for edge in root.findall("edge"):
        for lane in edge.findall("lane"):
            shape = dedupe(parse_shape(lane.attrib["shape"]))
            lanes[lane.attrib["id"]] = {
                "id": lane.attrib["id"],
                "edge": edge.attrib["id"],
                "index": int(lane.attrib["index"]),
                "speed": float(lane.attrib["speed"]),
                "shape": shape,
                "length": path_length(shape),
            }
    return lanes


def chain(lanes, ids):
    pts = []
    for lane_id in ids:
        pts.extend(lanes[lane_id]["shape"])
    return dedupe(pts)


def point(x, y, z=0.0):
    return {"x": float(x), "y": float(y), "z": float(z)}


def road_features(polylines, kind, element_id, next_id):
    features = []
    for pts in polylines:
        for piece in chunks(pts):
            features.append(
                {
                    "id": next_id,
                    "type": kind,
                    "map_element_id": element_id,
                    "geometry": [point(x, y) for x, y in piece],
                }
            )
            next_id += 1
    return features, next_id


def point_segment_distance(px, py, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - a[0], py - a[1])
    t = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / length_sq))
    return math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))


def polyline_distance(point, pts):
    return min(point_segment_distance(point[0], point[1], a, b) for a, b in zip(pts, pts[1:]))


def same_boundary(a, b, tol=0.45):
    """True when two lane sides are the same painted boundary, not just a shared corner."""
    if len(a) < 2 or len(b) < 2:
        return False
    close = sum(1 for p in a if polyline_distance(p, b) < tol)
    return close >= 2 and close / len(a) > 0.6


def lane_sides(lanes):
    """Left and right boundaries of every drivable lane, in Waymo's classes.

    A side shared with a same-direction neighbor is a broken white lane line.
    A side with no neighbor is a road-edge boundary. Both are offset from the
    lane center by half a lane, so the classes meet.
    """
    sides = []
    for lane in lanes.values():
        if lane["length"] < 1.0 or len(lane["shape"]) < 2:
            continue
        sampled = resample(lane["shape"], SPACING)
        if len(sampled) < 2:
            continue
        sides.append(offset(sampled, HALF))
        sides.append(offset(sampled, -HALF))

    used = set()
    lines = []
    edges = []
    for i, side in enumerate(sides):
        if i in used:
            continue
        partner = None
        for j in range(i + 1, len(sides)):
            if j in used:
                continue
            if same_boundary(side, sides[j]):
                partner = j
                break
        if partner is None:
            edges.append(side)
        else:
            used.add(partner)
            lines.append(side)
    return lines, edges


def make_agent(path, s0, speed, goal_dist, agent_id):
    total = path_length(path)
    goal_s = min(s0 + goal_dist, total - 1.0)
    goal, _ = sample_at(path, goal_s)
    positions, velocities, headings, valids = [], [], [], []
    for i in range(TRAJECTORY_LENGTH):
        s = s0 + speed * DT * i
        if s >= total - 0.5:
            positions.append(point(-10000, -10000, -10000))
            velocities.append(point(0, 0, 0))
            headings.append(0.0)
            valids.append(0)
            continue
        (x, y), heading = sample_at(path, s)
        positions.append(point(x, y))
        velocities.append(point(speed * math.cos(heading), speed * math.sin(heading)))
        headings.append(heading)
        valids.append(1)
    return {
        "id": agent_id,
        "type": "vehicle",
        "width": CAR["width"],
        "length": CAR["length"],
        "height": CAR["height"],
        "position": positions,
        "velocity": velocities,
        "heading": headings,
        "valid": valids,
        "goalPosition": point(*goal),
        "mark_as_expert": 0,
        "_s0": s0,
        "_speed": speed,
        "_goal_dist": goal_s - s0,
    }


def build():
    lanes = load_lanes(NET_PATH)
    left = chain(lanes, ["E0_1", "E1_1", ":J2_1_1", "E2_2", ":J3_0_1", "E4_1"])
    right = chain(lanes, ["E0_0", "E1_0", ":J2_1_0", "E2_1", ":J3_0_0", "E4_0"])
    accel = chain(lanes, ["ramp_0", ":J2_0_0", "E2_0"])

    # Waymo classes, matched to this net. Every center is a freeway lane.
    # Neighboring lanes share a broken-white line. The outer side of a lane
    # is a road-edge boundary. Caps close the dead ends.
    lane_lines = [
        lane["shape"]
        for lane in lanes.values()
        if lane["length"] >= 1.0 and len(lane["shape"]) >= 2
    ]
    markings, edges = lane_sides(lanes)
    ramp_left = offset(resample(lanes["ramp_0"]["shape"], SPACING), HALF)
    ramp_right = offset(resample(lanes["ramp_0"]["shape"], SPACING), -HALF)
    # Dead-end caps, plus the boundary that closes the acceleration lane at J3.
    # That lane has no successor, so its outer edge must meet the through lane's edge.
    edges.extend(
        [
            [(0.0, 0.0), (0.0, -6.4)],
            [(1300.0, 0.0), (1300.0, -6.4)],
            [ramp_left[0], ramp_right[0]],
            [(1096.0, -9.6), (1104.0, -6.4)],
        ]
    )

    roads = []
    next_id = 1
    made, next_id = road_features(lane_lines, "lane", LANE_FREEWAY, next_id)
    roads.extend(made)
    made, next_id = road_features(markings, "road_line", ROAD_LINE_BROKEN_SINGLE_WHITE, next_id)
    roads.extend(made)
    made, next_id = road_features(edges, "road_edge", ROAD_EDGE_BOUNDARY, next_id)
    roads.extend(made)

    # Last object is the primary agent (the env treats num_objects - 1 as the SDC).
    specs = [
        (right, 70.0, 20.0, 150.0),
        (right, 180.0, 20.0, 150.0),
        (right, 420.0, 22.0, 160.0),
        (left, 40.0, 22.0, 160.0),
        (left, 130.0, 22.0, 160.0),
        (accel, 80.0, 13.0, 110.0),
        (accel, path_length(lanes["ramp_0"]["shape"]) + 40.0, 18.0, 140.0),
        (left, 480.0, 22.0, 170.0),
    ]
    objects = []
    for i, (path, s0, speed, goal_dist) in enumerate(specs):
        objects.append(make_agent(path, s0, speed, goal_dist, 1000 + i))

    scenario = {
        "name": "smallnet",
        "scenario_id": "smallnet",
        "objects": [{k: v for k, v in obj.items() if not k.startswith("_")} for obj in objects],
        "roads": roads,
        "tl_states": {},
        "metadata": {
            "sdc_track_index": len(objects) - 1,
            "objects_of_interest": [],
            "tracks_to_predict": [],
        },
    }
    return scenario, objects, {
        "lanes": lanes,
        "left": left,
        "right": right,
        "accel": accel,
        "edges": edges,
        "markings": markings,
        "lane_lines": lane_lines,
        "roads": roads,
    }


def grid_pressure(roads):
    xs, ys = [], []
    segments = []
    for road in roads:
        if road["type"] not in ("lane", "road_line", "road_edge"):
            continue
        geom = road["geometry"]
        for a, b in zip(geom, geom[1:]):
            xs.extend([a["x"], b["x"]])
            ys.extend([a["y"], b["y"]])
            segments.append((a, b))
    min_x, min_y = min(xs), min(ys)
    counts = {}
    for a, b in segments:
        mx = (a["x"] + b["x"]) / 2.0
        my = (a["y"] + b["y"]) / 2.0
        key = (int((mx - min_x) / 5.0), int((my - min_y) / 5.0))
        counts[key] = counts.get(key, 0) + 1
    hottest = max(counts.values()) if counts else 0
    return hottest, sum(1 for n in counts.values() if n > 10)


def save_preview(scene, objects, path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [1.2, 1.3, 1.0]}
    )
    views = [
        (axes[0], "smallnet", None),
        (axes[1], "merge at J2", (600, 860, -70, 12)),
        (axes[2], "lane drop at J3", (1040, 1180, -16, 8)),
    ]
    edge_color = "#f2f2f2"
    lane_color = "#8fd18f"
    mark_color = "#e6d36a"
    legend = None

    for ax, title, window in views:
        edge_plot = None
        for pts in scene["edges"]:
            xs, ys = zip(*pts)
            edge_plot, = ax.plot(xs, ys, color=edge_color, linewidth=1.6, solid_capstyle="round")
        lane_plot = None
        for pts in scene["lane_lines"]:
            xs, ys = zip(*resample(pts, 20))
            lane_plot, = ax.plot(xs, ys, color=lane_color, linewidth=0.8, alpha=0.9)
        line_plot = None
        for pts in scene["markings"]:
            xs, ys = zip(*pts)
            line_plot, = ax.plot(xs, ys, color=mark_color, linewidth=0.7, linestyle="--")
        if legend is None and edge_plot and lane_plot and line_plot:
            legend = ax.legend(
                [lane_plot, line_plot, edge_plot],
                [
                    "lane / freeway (1)",
                    "road line / broken white (6)",
                    "road edge / boundary (15)",
                ],
                loc="lower right",
                fontsize=8,
                frameon=False,
                labelcolor="white",
            )
        for i, obj in enumerate(objects):
            p = obj["position"][0]
            h = obj["heading"][0]
            length, width = obj["length"], obj["width"]
            c, s = math.cos(h), math.sin(h)
            hl, hw = length / 2, width / 2
            corners = [
                (hl * c - hw * s, hl * s + hw * c),
                (hl * c + hw * s, hl * s - hw * c),
                (-hl * c + hw * s, -hl * s - hw * c),
                (-hl * c - hw * s, -hl * s + hw * c),
            ]
            poly = [(p["x"] + dx, p["y"] + dy) for dx, dy in corners]
            color = "#ff5a5a" if i == len(objects) - 1 else "#4aa3ff"
            ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="white", lw=0.4))
            g = obj["goalPosition"]
            ax.scatter([g["x"]], [g["y"]], s=18, c="#39d353", zorder=4)
            ax.annotate(
                str(i),
                (p["x"], p["y"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                color="white",
                fontsize=7,
            )
        ax.set_title(title, color="white", loc="left", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#1b1d22")
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444444")
        if window:
            ax.set_xlim(window[0], window[1])
            ax.set_ylim(window[2], window[3])
        else:
            ax.set_xlim(-30, 1340)
            ax.set_ylim(-75, 20)
        ax.set_xlabel("x (m)", color="#bbbbbb")
        ax.set_ylabel("y (m)", color="#bbbbbb")

    fig.patch.set_facecolor("#111215")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Observation layout from ocean/drive/drive.h. Road slots are the first 75
# segments in the vision spiral, not every segment inside the radius.
GRID_CELL_SIZE = 5.0
MAX_ENTITIES_PER_CELL = 10
VISION_RANGE = 21
MAX_ROAD_SEGMENT_OBSERVATIONS = 75
PARTNER_FEATURES = 7
EGO_FEATURES = 7
ROAD_FEATURES = 7
MAX_AGENTS = 64
OBS_GOAL_SCALE = 0.005
OBS_SPEED_SCALE = 0.01
OBS_POSITION_SCALE = 0.02
MAX_SPEED = 100.0
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
MAX_ROAD_SEGMENT_LENGTH = 100.0
MAX_ROAD_SCALE = 100.0
OBS_DIST_SQ = 2500.0
def main():
    scenario, objects, scene = build()
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(BIN_DIR, exist_ok=True)
    json_path = os.path.join(JSON_DIR, "smallnet.json")
    import json

    with open(json_path, "w") as f:
        json.dump(scenario, f)
    binary_path = os.path.join(BIN_DIR, "map_000.bin")
    save_map_binary(scenario, binary_path, 0)
    hottest, overflow = grid_pressure(scenario["roads"])
    save_preview(scene, objects, PREVIEW_PATH)

    print(f"json {json_path}")
    print(f"binary {binary_path}")
    print(f"preview {PREVIEW_PATH}")
    print(f"roads {len(scenario['roads'])} objects {len(objects)}")
    from collections import Counter

    print("classes", dict(Counter((r["type"], r["map_element_id"]) for r in scenario["roads"])))
    print(f"grid max segments/cell {hottest} overflow_cells {overflow}")
    for i, obj in enumerate(objects):
        p = obj["position"][0]
        g = obj["goalPosition"]
        dist = math.hypot(g["x"] - p["x"], g["y"] - p["y"])
        print(
            f"agent {i} id {obj['id']} start ({p['x']:.1f},{p['y']:.1f}) "
            f"heading {obj['heading'][0]:.3f} speed {obj['_speed']:.1f} "
            f"goal ({g['x']:.1f},{g['y']:.1f}) dist {dist:.1f}m "
            f"eta {dist / obj['_speed']:.1f}s valid {sum(obj['valid'])}"
        )


if __name__ == "__main__":
    main()
