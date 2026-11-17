#!/usr/bin/env python3
"""Agent density on the four Town maps.

Initializes the drive sim at 64, 128, 256, and 512 agents on Town01,
Town02, Town06, and Town10HD. Reports cars per square meter, the share
of the lane centerline taken by car length, the share of road covered
by the boxes, and that road share after growing every box by 20 cm on
each side. One panel per metric, with agent count on x.

The Town01 map figure is written to logs/maps/town01_density.png.

The map binary stores lane width as 0, so road area is not width times
length. It is the surface enclosed by ROAD_EDGE polylines: rasterize
those edges, thicken them by one cell so small gaps do not leak, flood
out from the lane centerlines, then give back the inner cell the
thickening had covered. Off-road in the sim is a car rectangle crossing
an edge, and this fill is the area inside those edges.

Needs build/libdrive_density.so compiled with -DDRIVE_AGENT_MAX=512.
Does not load or replace ./puffer.

    python3 scripts/plot_town_density.py
"""

import argparse
import ctypes
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libdrive_density.so"
OUT = ROOT / "logs" / "maps" / "town01_density.png"
METRICS_OUT = ROOT / "logs" / "maps" / "town_density_metrics.png"
DATA_DIR = ROOT / "data" / "mixed"
MAP_ID = 0
# map_id matches data/mixed/binaries/map_%03d.bin.
TOWNS = (
    (0, "Town01"),
    (1, "Town02"),
    (2, "Town06"),
    (3, "Town10HD"),
)
COUNTS = (64, 128, 256, 512)
# Extra space each car needs, on every side of the plan box.
PAD_M = 0.20
CELL_M = 0.25
GOAL_MIN = 5.0
GOAL_MAX = 50.0
EPISODE_LENGTH = 300
ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6
# Body-frame corners, same order as ocean/drive/constants.h offsets[].
CORNERS = np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])


def load_lib():
    if not LIB_PATH.is_file():
        raise SystemExit(f"missing {LIB_PATH}")
    lib = ctypes.CDLL(str(LIB_PATH))
    handle = ctypes.c_void_p
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    lib.pf_create_sim.restype = handle
    lib.pf_create_sim.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
    ]
    lib.pf_close.restype = None
    lib.pf_close.argtypes = [handle]
    lib.pf_reset.restype = None
    lib.pf_reset.argtypes = [handle, ctypes.c_uint]
    lib.pf_n.restype = ctypes.c_int
    lib.pf_n.argtypes = [handle]
    lib.pf_agent.restype = None
    lib.pf_agent.argtypes = [handle, ctypes.c_int, c_float_p]
    lib.pf_road_counts.restype = None
    lib.pf_road_counts.argtypes = [handle, c_int_p, c_int_p]
    lib.pf_roads.restype = None
    lib.pf_roads.argtypes = [handle, c_int_p, c_int_p, c_float_p, c_float_p]
    return lib


def create_sim(lib, num_agents, map_id=MAP_ID):
    sim = lib.pf_create_sim(
        str(DATA_DIR).encode(), int(map_id), int(num_agents), EPISODE_LENGTH,
        GOAL_MIN, GOAL_MAX,
    )
    if not sim:
        raise RuntimeError(f"pf_create_sim failed for map {map_id}, {num_agents} agents")
    # puf_init only loads the map. Placement happens in puf_reset.
    lib.pf_reset(sim, 0)
    got = int(lib.pf_n(sim))
    if got != num_agents:
        lib.pf_close(sim)
        raise RuntimeError(
            f"map {map_id} initialized {got} agents, asked for {num_agents}"
        )
    return sim


def read_agents(lib, sim):
    n = int(lib.pf_n(sim))
    rows = []
    for i in range(n):
        out = (ctypes.c_float * 11)()
        lib.pf_agent(sim, i, out)
        rows.append(np.array(out, dtype=np.float64))
    agents = np.stack(rows)
    valid = agents[:, 10] > 0.5
    if int(valid.sum()) != n:
        raise RuntimeError(f"{n - int(valid.sum())} agents failed to spawn")
    return agents


def read_roads(lib, sim):
    nroads = ctypes.c_int()
    npts = ctypes.c_int()
    lib.pf_road_counts(sim, ctypes.byref(nroads), ctypes.byref(npts))
    offsets = np.zeros(nroads.value + 1, dtype=np.int32)
    kinds = np.zeros(nroads.value, dtype=np.int32)
    xs = np.zeros(npts.value, dtype=np.float32)
    ys = np.zeros(npts.value, dtype=np.float32)
    lib.pf_roads(
        sim,
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    roads = []
    for i in range(int(nroads.value)):
        a = int(offsets[i])
        b = int(offsets[i + 1])
        roads.append((int(kinds[i]), xs[a:b].astype(np.float64), ys[a:b].astype(np.float64)))
    return roads


def _mark(mask, row, col):
    if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
        mask[row, col] = True


def _raster_segment(mask, x0, y0, x1, y1, xmin, ymin, cell):
    """Mark every cell the segment crosses, including both cells at a corner."""
    gx0 = (x0 - xmin) / cell
    gy0 = (y0 - ymin) / cell
    gx1 = (x1 - xmin) / cell
    gy1 = (y1 - ymin) / cell
    x = math.floor(gx0)
    y = math.floor(gy0)
    x_end = math.floor(gx1)
    y_end = math.floor(gy1)
    dx = gx1 - gx0
    dy = gy1 - gy0
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
    _mark(mask, y, x)
    if step_x == 0 and step_y == 0:
        return
    t_delta_x = abs(1.0 / dx) if step_x else math.inf
    t_delta_y = abs(1.0 / dy) if step_y else math.inf
    if step_x > 0:
        t_max_x = (math.floor(gx0) + 1 - gx0) * t_delta_x
    elif step_x < 0:
        t_max_x = (gx0 - math.floor(gx0)) * t_delta_x
    else:
        t_max_x = math.inf
    if step_y > 0:
        t_max_y = (math.floor(gy0) + 1 - gy0) * t_delta_y
    elif step_y < 0:
        t_max_y = (gy0 - math.floor(gy0)) * t_delta_y
    else:
        t_max_y = math.inf
    guard = int(abs(dx) + abs(dy)) + 4
    for _ in range(guard):
        if x == x_end and y == y_end:
            return
        if abs(t_max_x - t_max_y) <= 1e-9:
            _mark(mask, y, x + step_x)
            _mark(mask, y + step_y, x)
            t_max_x += t_delta_x
            t_max_y += t_delta_y
            x += step_x
            y += step_y
        elif t_max_x < t_max_y:
            t_max_x += t_delta_x
            x += step_x
        else:
            t_max_y += t_delta_y
            y += step_y
        _mark(mask, y, x)


def road_area(roads, cell):
    xs = np.concatenate([rx for _, rx, _ in roads])
    ys = np.concatenate([ry for _, _, ry in roads])
    pad = 20.0
    xmin = float(xs.min()) - pad
    ymin = float(ys.min()) - pad
    cols = int(math.ceil((float(xs.max()) + pad - xmin) / cell))
    rows = int(math.ceil((float(ys.max()) + pad - ymin) / cell))
    barrier = np.zeros((rows, cols), dtype=bool)
    for kind, rx, ry in roads:
        if kind != ROAD_EDGE or len(rx) < 2:
            continue
        for x0, y0, x1, y1 in zip(rx[:-1], ry[:-1], rx[1:], ry[1:]):
            _raster_segment(barrier, x0, y0, x1, y1, xmin, ymin, cell)
    wall = ndimage.binary_dilation(barrier)
    seeds = np.zeros_like(barrier)
    for kind, rx, ry in roads:
        if kind != ROAD_LANE:
            continue
        col = np.floor((rx - xmin) / cell).astype(np.int32)
        row = np.floor((ry - ymin) / cell).astype(np.int32)
        keep = (
            (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
            & ~wall[row.clip(0, rows - 1), col.clip(0, cols - 1)]
        )
        seeds[row[keep], col[keep]] = True
    if not seeds.any():
        raise RuntimeError("no lane seed inside the road edges")
    road = seeds
    steps = 0
    while True:
        grown = ndimage.binary_dilation(road, mask=~wall)
        steps += 1
        if np.array_equal(grown, road):
            break
        road = grown
        if steps > rows + cols:
            raise RuntimeError("road flood did not stop")
    touched = road[0].any() or road[-1].any() or road[:, 0].any() or road[:, -1].any()
    if touched:
        raise RuntimeError("road flood reached the grid border")
    # The wall is one cell thicker than the drawn edge. Give that inner
    # cell back. It does not cross the edge raster itself.
    band = ndimage.binary_dilation(road) & ~barrier & ~road
    full = road | band
    extent = (xmin, xmin + cols * cell, ymin, ymin + rows * cell)
    raw_m2 = float(road.sum()) * cell * cell
    full_m2 = float(full.sum()) * cell * cell
    return full, extent, full_m2, raw_m2, steps


def car_polygons(agents):
    polys = []
    for x, y, heading, width, length in agents[:, [0, 1, 2, 7, 8]]:
        body = CORNERS * np.array([length * 0.5, width * 0.5])
        c = math.cos(heading)
        s = math.sin(heading)
        rot = np.array([[c, -s], [s, c]])
        polys.append(body @ rot.T + np.array([x, y]))
    return polys


def box_area(agents):
    """Plan area of the collision boxes. Length is the long side."""
    return float(np.sum(agents[:, 7] * agents[:, 8]))


def padded_box_area(agents, pad):
    """Same boxes, grown by `pad` meters on every side."""
    return float(np.sum((agents[:, 7] + 2.0 * pad) * (agents[:, 8] + 2.0 * pad)))


def density_text(count, area):
    return f"{count / area:.5f} cars/m$^2$"


def draw(roads, populations, mask, extent, area, out):
    xs = np.concatenate([rx for _, rx, _ in roads])
    ys = np.concatenate([ry for _, _, ry in roads])
    pad = 8.0
    x0, x1 = float(xs.min()) - pad, float(xs.max()) + pad
    y0, y1 = float(ys.min()) - pad, float(ys.max()) + pad
    fig, axes = plt.subplots(
        1, len(populations), figsize=(16.8, 5.8),
        sharex=True, sharey=True, constrained_layout=True,
    )
    tint = np.zeros((*mask.shape, 4), dtype=np.float32)
    tint[mask, 0] = 0.76
    tint[mask, 1] = 0.84
    tint[mask, 2] = 0.72
    tint[mask, 3] = 0.55
    for ax, (count, agents) in zip(axes, populations):
        ax.imshow(
            tint, origin="lower", extent=extent, interpolation="nearest", zorder=0,
        )
        for kind, rx, ry in roads:
            if kind == ROAD_EDGE:
                ax.plot(rx, ry, color="#1a1a1a", lw=0.7, solid_capstyle="round", zorder=2)
            elif kind == ROAD_LINE:
                ax.plot(rx, ry, color="#8d8d8d", lw=0.35, zorder=1)
            elif kind == ROAD_LANE:
                ax.plot(rx, ry, color="#4c78a8", lw=0.45, zorder=1)
        cars = PolyCollection(
            car_polygons(agents),
            facecolor="#e6550d", edgecolor="#7f2704",
            linewidths=0.15, zorder=3,
        )
        ax.add_collection(cars)
        covered = 100.0 * box_area(agents) / area
        lane_m = 0.0
        for kind, rx, ry in roads:
            if kind == ROAD_LANE and len(rx) >= 2:
                lane_m += float(np.hypot(np.diff(rx), np.diff(ry)).sum())
        centerline = 100.0 * float(np.sum(agents[:, 8])) / lane_m
        ax.set_title(
            f"{count} agents\n{density_text(count, area)}\n"
            f"{covered:.2f}% of road\n{centerline:.2f}% of centerline",
            fontsize=11,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Town01  ·  road area {area:,.0f} m$^2$"
        f"  ·  cars/m$^2$ = agents / road area"
        f"  ·  coverage = box width $\\times$ length / road area",
        fontsize=12,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def measure_case(map_id, count):
    lib = load_lib()
    sim = create_sim(lib, count, map_id)
    try:
        roads = read_roads(lib, sim)
        agents = read_agents(lib, sim)
    finally:
        lib.pf_close(sim)
    _, _, area, _, _ = road_area(roads, CELL_M)
    boxes = box_area(agents)
    padded = padded_box_area(agents, PAD_M)
    lane_m = 0.0
    for kind, rx, ry in roads:
        if kind == ROAD_LANE and len(rx) >= 2:
            lane_m += float(np.hypot(np.diff(rx), np.diff(ry)).sum())
    return {
        "map_id": int(map_id),
        "agents": int(count),
        "road_m2": area,
        "lane_m": lane_m,
        "width_m": float(agents[0, 7]),
        "length_m": float(agents[0, 8]),
        "box_m2": boxes,
        "padded_m2": padded,
        "cars_per_m2": count / area,
        "coverage_pct": 100.0 * boxes / area,
        "padded_pct": 100.0 * padded / area,
        "centerline_pct": 100.0 * float(np.sum(agents[:, 8])) / lane_m,
    }


def run_case(map_id, count):
    """One init per process. A failed goal assert must not kill the others."""
    script = str(Path(__file__).resolve())
    try:
        proc = subprocess.run(
            [sys.executable, script, "--case", str(map_id), str(count)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return None, "timed out"
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if proc.returncode != 0 or line is None:
        err = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        reason = err[-1] if err else f"exit {proc.returncode}"
        return None, reason
    return json.loads(line[len("RESULT "):]), None


def draw_metrics(rows, out):
    by_town = {name: [] for _, name in TOWNS}
    for row in rows:
        by_town[row["town"]].append(row)
    panels = (
        ("cars_per_m2", "Cars / m²", "{:.5f}"),
        ("centerline_pct", "Centerline coverage", "{:.2f}"),
        ("coverage_pct", "Road coverage", "{:.2f}"),
        ("padded_pct", "Road coverage, 20 cm around each car", "{:.2f}"),
    )
    y_labels = ("cars/m²", "% of centerline", "% of road", "% of road")
    colors = {
        "Town01": "#4c78a8",
        "Town02": "#f58518",
        "Town06": "#54a24b",
        "Town10HD": "#e45756",
    }
    markers = {"Town01": "o", "Town02": "s", "Town06": "D", "Town10HD": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
    for ax, (key, title, fmt), ylabel in zip(axes.ravel(), panels, y_labels):
        for _, name in TOWNS:
            series = sorted(by_town[name], key=lambda row: row["agents"])
            if not series:
                continue
            ax.plot(
                [row["agents"] for row in series],
                [row[key] for row in series],
                color=colors[name],
                marker=markers[name],
                lw=1.6,
                ms=6,
                label=name,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("agents")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(COUNTS))
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p, fmt=fmt: fmt.format(v)))
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        "Centerline coverage is car length / lane centerline."
        "  Road coverage uses the box area inside the road edges.",
        fontsize=10, y=0.93,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def report(rows, failed):
    print(
        f"box {rows[0]['width_m']:.3f} m by {rows[0]['length_m']:.3f} m"
        f"  padded {rows[0]['width_m'] + 2 * PAD_M:.3f} m"
        f" by {rows[0]['length_m'] + 2 * PAD_M:.3f} m",
        flush=True,
    )
    print(
        f"{'town':8} {'n':4} {'road m2':>10} {'lane m':>8} {'cars/m2':>10}"
        f" {'lane %':>8} {'road %':>8} {'pad %':>8}",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['town']:8} {row['agents']:4d} {row['road_m2']:10.1f}"
            f" {row['lane_m']:8.1f} {row['cars_per_m2']:10.5f}"
            f" {row['centerline_pct']:8.2f} {row['coverage_pct']:8.2f}"
            f" {row['padded_pct']:8.2f}",
            flush=True,
        )
    for name, count, reason in failed:
        print(f"FAILED {name} {count}: {reason}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--metrics-out", type=Path, default=METRICS_OUT)
    parser.add_argument("--case", nargs=2, type=int, metavar=("MAP", "AGENTS"))
    parser.add_argument("--town01-map", action="store_true")
    args = parser.parse_args()
    if args.case is not None:
        payload = measure_case(args.case[0], args.case[1])
        print("RESULT " + json.dumps(payload), flush=True)
        return
    if args.town01_map:
        plot_town01(args.out)
        return
    rows = []
    failed = []
    for map_id, name in TOWNS:
        for count in COUNTS:
            row, reason = run_case(map_id, count)
            if row is None:
                failed.append((name, count, reason))
                print(f"failed {name} {count}: {reason}", flush=True)
                continue
            row["town"] = name
            rows.append(row)
            print(
                f"{name} {count}: {row['cars_per_m2']:.5f} cars/m2"
                f"  centerline {row['centerline_pct']:.2f}%"
                f"  road {row['coverage_pct']:.2f}%"
                f"  padded {row['padded_pct']:.2f}%",
                flush=True,
            )
    if not rows:
        raise SystemExit("no town initialized")
    report(rows, failed)
    draw_metrics(rows, args.metrics_out)
    print(f"wrote {args.metrics_out}", flush=True)


def plot_town01(out):
    lib = load_lib()
    populations = []
    roads = None
    for count in COUNTS:
        sim = create_sim(lib, count)
        try:
            if roads is None:
                roads = read_roads(lib, sim)
            populations.append((count, read_agents(lib, sim)))
        finally:
            lib.pf_close(sim)
        print(f"initialized {count} agents", flush=True)
    mask, extent, area, raw, steps = road_area(roads, CELL_M)
    print(
        f"road area {area:.1f} m^2 "
        f"(inside the edge, before reclaim {raw:.1f}, flood steps {steps})",
        flush=True,
    )
    lane_m = 0.0
    for kind, rx, ry in roads:
        if kind != ROAD_LANE or len(rx) < 2:
            continue
        lane_m += float(np.hypot(np.diff(rx), np.diff(ry)).sum())
    width = float(populations[0][1][0, 7])
    length = float(populations[0][1][0, 8])
    print(f"box {width:.3f} m by {length:.3f} m", flush=True)
    for count, agents in populations:
        boxes = box_area(agents)
        print(
            f"{count:4d}  {count / area:.5f} cars/m^2"
            f"  boxes {boxes:.0f} m^2 ({100.0 * boxes / area:.1f}% of road)"
            f"  {lane_m / count:.1f} m of lane per car",
            flush=True,
        )
    draw(roads, populations, mask, extent, area, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
