"""Five cars on map 194. Bicycle and delta-local steps from action_grid.json."""

import ctypes
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_bind as drv
import train_bc_map as t

MAP_ID = 194
N_CARS = 5
# Continuous bicycle. Wide bounds, no bin snap.
CONT = (-50.0, 50.0, 0, -float(np.pi), float(np.pi), 0)
OUT = drv.ROOT / "analysis" / "offline" / "dynamics_rollouts.png"
GRID_JSON = drv.ROOT / "analysis" / "offline" / "action_grid.json"
ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6


def _specs():
    summary = json.loads(GRID_JSON.read_text())
    best = summary["best"]
    delta = summary["best_discrete_delta"]
    bike = (
        float(best["accel_min"]), float(best["accel_max"]), int(best["n_accel"]),
        float(best["steer_min"]), float(best["steer_max"]), int(best["n_steer"]),
    )
    return best, bike, (
        int(delta["n_dx"]), int(delta["n_dy"]), int(delta["n_yaw"]),
    )


def _trace_bike(lib, h, grid):
    n = lib.pf_log_n(h)
    traj = lib.pf_log_traj(h)
    xs = np.zeros(n * traj, dtype=np.float32)
    ys = np.zeros(n * traj, dtype=np.float32)
    on = np.zeros(n * traj, dtype=np.uint8)
    lib.pf_quantized_trace(
        h, grid[0], grid[1], grid[2], grid[3], grid[4], grid[5],
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        on.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    shape = (n, traj)
    return xs.reshape(shape), ys.reshape(shape), on.reshape(shape)


def _trace_delta(lib, h, counts):
    n = lib.pf_log_n(h)
    traj = lib.pf_log_traj(h)
    xs = np.zeros(n * traj, dtype=np.float32)
    ys = np.zeros(n * traj, dtype=np.float32)
    on = np.zeros(n * traj, dtype=np.uint8)
    lib.pf_delta_trace(
        h, counts[0], counts[1], counts[2],
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        on.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    shape = (n, traj)
    return xs.reshape(shape), ys.reshape(shape), on.reshape(shape)


def _log(lib, h, agent):
    traj = lib.pf_log_traj(h)
    xs = np.zeros(traj, dtype=np.float64)
    ys = np.zeros(traj, dtype=np.float64)
    valid = np.zeros(traj, dtype=bool)
    for frame in range(traj):
        if not lib.pf_log_valid(h, agent, frame):
            continue
        x, y = drv.log_xy(lib, h, agent, frame)
        xs[frame] = x
        ys[frame] = y
        valid[frame] = True
    return xs, ys, valid


def _ade(sim_x, sim_y, sim_on, log_x, log_y, log_on):
    err = []
    for frame in range(1, len(log_on)):
        if not (sim_on[frame] and log_on[frame]):
            continue
        dx = float(sim_x[frame] - log_x[frame])
        dy = float(sim_y[frame] - log_y[frame])
        err.append((dx * dx + dy * dy) ** 0.5)
    if not err:
        return 0.0
    return float(np.mean(err))


def _pieces(xs, ys, mask):
    cuts = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for a, b in cuts.reshape(-1, 2):
        if b - a >= 2:
            yield xs[a:b], ys[a:b]


def _roads_in(roads, x0, x1, y0, y1):
    kept = []
    for kind, rx, ry in roads:
        if len(rx) == 0:
            continue
        inside = (
            (rx >= x0) & (rx <= x1) & (ry >= y0) & (ry <= y1)
        ).any()
        if inside:
            kept.append((kind, rx, ry))
    return kept


def main():
    best, bike_grid, delta_grid = _specs()
    lib = drv.load()
    h = drv.create_scenario(lib, t.DATA, MAP_ID, t.JSONS[MAP_ID])
    try:
        fit = [i for i in range(lib.pf_log_n(h)) if not lib.pf_base_unfit(h, i)]
        sdc = int(lib.pf_sdc(h))
        logs = {i: _log(lib, h, i) for i in fit}
        ranked = sorted(fit, key=lambda i: int(logs[i][2].sum()), reverse=True)
        chosen = ranked[:N_CARS]
        if sdc in fit and sdc not in chosen and int(logs[sdc][2].sum()) >= 20:
            chosen = chosen[:-1] + [sdc]
        disc_x, disc_y, disc_on = _trace_bike(lib, h, bike_grid)
        cont_x, cont_y, cont_on = _trace_bike(lib, h, CONT)
        delta_x, delta_y, delta_on = _trace_delta(lib, h, delta_grid)
        road = drv.roads(lib, h)
    finally:
        lib.pf_close(h)

    rows = []
    for agent in chosen:
        lx, ly, lv = logs[agent]
        rows.append({
            "agent": agent,
            "sdc": agent == sdc,
            "log": (lx, ly, lv),
            "delta_ade": _ade(delta_x[agent], delta_y[agent], delta_on[agent], lx, ly, lv),
            "disc_ade": _ade(disc_x[agent], disc_y[agent], disc_on[agent], lx, ly, lv),
            "cont_ade": _ade(cont_x[agent], cont_y[agent], cont_on[agent], lx, ly, lv),
            "delta": (delta_x[agent], delta_y[agent], delta_on[agent].astype(bool)),
            "disc": (disc_x[agent], disc_y[agent], disc_on[agent].astype(bool)),
            "cont": (cont_x[agent], cont_y[agent], cont_on[agent].astype(bool)),
        })

    xs = []
    ys = []
    for row in rows:
        for key in ("log", "delta", "disc", "cont"):
            x, y, m = row[key]
            xs.append(x[m])
            ys.append(y[m])
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    pad = 18.0
    x0, x1 = float(all_x.min()) - pad, float(all_x.max()) + pad
    y0, y1 = float(all_y.min()) - pad, float(all_y.max()) + pad
    near = _roads_in(road, x0 - 30.0, x1 + 30.0, y0 - 30.0, y1 + 30.0)

    t.apply_figure_style()
    fig, ax = plt.subplots(figsize=(7.6, 7.2), constrained_layout=True)
    drew_lane = False
    drew_edge = False
    for kind, rx, ry in near:
        if kind == ROAD_EDGE:
            ax.plot(rx, ry, color="#4a4a4a", lw=1.05, zorder=1, solid_capstyle="round",
                label="road edge" if not drew_edge else None)
            drew_edge = True
        elif kind == ROAD_LINE:
            ax.plot(rx, ry, color="#b5b5b5", lw=0.55, zorder=0)
        else:
            ax.plot(rx, ry, color="#d2d2d2", lw=0.8, zorder=0,
                label="lane" if kind == ROAD_LANE and not drew_lane else None)
            if kind == ROAD_LANE:
                drew_lane = True
    colors = sns.color_palette("deep", n_colors=len(rows))
    for color, row in zip(colors, rows):
        lx, ly, lv = row["log"]
        for px, py in _pieces(lx, ly, lv):
            ax.plot(px, py, color="black", lw=0.9, zorder=2, solid_capstyle="round")
        dx, dy, dm = row["delta"]
        for px, py in _pieces(dx, dy, dm):
            ax.plot(px, py, color=color, lw=1.9, zorder=3, solid_capstyle="round")
        cx, cy, cm = row["cont"]
        for px, py in _pieces(cx, cy, cm):
            ax.plot(px, py, color=color, lw=1.35, ls=(0, (1.0, 1.3)), zorder=4)
        bx, by, bm = row["disc"]
        for px, py in _pieces(bx, by, bm):
            ax.plot(px, py, color=color, lw=1.55, ls=(0, (3.4, 1.6)), zorder=5)
        start = int(np.flatnonzero(lv)[0])
        ax.scatter([lx[start]], [ly[start]], s=18, color=color, zorder=6, linewidths=0)
    delta_mean = float(np.mean([row["delta_ade"] for row in rows]))
    disc_mean = float(np.mean([row["disc_ade"] for row in rows]))
    cont_mean = float(np.mean([row["cont_ade"] for row in rows]))
    bike_label = (
        f"discrete bicycle ±{best['accel_max']:g}/{best['n_accel']}"
        f" ±{best['steer_max']:g}/{best['n_steer']}"
    )
    handles = [
        Line2D([0], [0], color="#4a4a4a", lw=1.05, label="road edge"),
        Line2D([0], [0], color="#d2d2d2", lw=1.4, label="lane"),
        Line2D([0], [0], color="black", lw=0.9, label="log"),
        Line2D([0], [0], color="#333333", lw=1.9,
            label=f"delta-local {delta_grid[0]}/{delta_grid[1]}/{delta_grid[2]}"),
        Line2D([0], [0], color="#333333", lw=1.35, ls=(0, (1.0, 1.3)),
            label="continuous bicycle"),
        Line2D([0], [0], color="#333333", lw=1.55, ls=(0, (3.4, 1.6)),
            label=bike_label),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="best")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"map {MAP_ID}   one color per car\n"
        f"discrete bicycle {disc_mean:.2f} m    "
        f"continuous {cont_mean:.2f} m    "
        f"delta-local {delta_mean:.2f} m"
    )
    t.style_ax(ax)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT}", flush=True)
    print(f"json {t.JSONS[MAP_ID].name}", flush=True)
    for row in rows:
        mark = " sdc" if row["sdc"] else ""
        print(
            f"agent {row['agent']:3d}{mark}  "
            f"delta {row['delta_ade']:.3f} m  "
            f"discrete {row['disc_ade']:.3f} m  "
            f"continuous {row['cont_ade']:.3f} m",
            flush=True,
        )


if __name__ == "__main__":
    main()
