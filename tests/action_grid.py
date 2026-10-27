"""Log inverse snapped onto a grid and stepped with drive.h.

Bin count 0 clips and does not snap. Frame 0 is the spawn and is not in the ADE.
"""

import ctypes
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_bind as drv
import train_bc_map as t

N_MAPS = 50
BINS = (32, 64, 128, 256, 512, 1024)
ACCEL_BOUNDS = (2.0, 4.0, 8.0)
STEER_BOUNDS = (1.0, 1.5, 4.0)
# Wide enough that the continuous bicycle is the model, not the clip.
BIKE_CONTINUOUS = (-50.0, 50.0, 0, -float(np.pi), float(np.pi), 0)
# Paper grid is asymmetric. Equal counts share the bicycle x-axis.
DELTA_PAPER = (51, 51, 127)
OUT = drv.ROOT / "analysis" / "offline" / "action_grid"

def _ade(lib, h, amin, amax, na, smin, smax, ns):
    ade = ctypes.c_float()
    n_agents = ctypes.c_int()
    clip_a = ctypes.c_float()
    clip_s = ctypes.c_float()
    lib.pf_quantized_ade(
        h, float(amin), float(amax), int(na),
        float(smin), float(smax), int(ns),
        ctypes.byref(ade), ctypes.byref(n_agents),
        ctypes.byref(clip_a), ctypes.byref(clip_s),
    )
    return float(ade.value), int(n_agents.value), float(clip_a.value), float(clip_s.value)

def _delta(lib, h, n_dx, n_dy, n_yaw):
    ade = ctypes.c_float()
    n_agents = ctypes.c_int()
    clip_x = ctypes.c_float()
    clip_y = ctypes.c_float()
    clip_yaw = ctypes.c_float()
    lib.pf_delta_ade(
        h, int(n_dx), int(n_dy), int(n_yaw),
        ctypes.byref(ade), ctypes.byref(n_agents),
        ctypes.byref(clip_x), ctypes.byref(clip_y), ctypes.byref(clip_yaw),
    )
    return (
        float(ade.value), int(n_agents.value),
        float(clip_x.value), float(clip_y.value), float(clip_yaw.value),
    )

def _delta_row(n_dx, n_dy, n_yaw, per_map, clips, kind):
    vals = np.asarray(per_map, dtype=np.float64)
    clip = np.asarray(clips, dtype=np.float64)
    return {
        "kind": kind,
        "n_dx": int(n_dx),
        "n_dy": int(n_dy),
        "n_yaw": int(n_yaw),
        "ade_mean": float(vals.mean()),
        "ade_std_maps": float(vals.std()),
        "clip_dx": float(clip[:, 0].mean()),
        "clip_dy": float(clip[:, 1].mean()),
        "clip_yaw": float(clip[:, 2].mean()),
        "agents_per_map": float(clip[:, 3].mean()),
    }

def main():
    lib = drv.load()
    grids = []
    for ab in ACCEL_BOUNDS:
        for sb in STEER_BOUNDS:
            for n in BINS:
                grids.append((-ab, ab, n, -sb, sb, n))
    grids.append((-4.0, 4.0, 21, -1.0, 1.0, 31))
    delta_grids = [(0, 0, 0)]
    delta_grids.append(DELTA_PAPER)
    for n in BINS:
        delta_grids.append((n, n, n))
    per_map = {g: [] for g in grids}
    clips = {g: [] for g in grids}
    delta_map = {g: [] for g in delta_grids}
    delta_clips = {g: [] for g in delta_grids}
    cont_map = []
    cont_clips = []
    agent_counts = []
    for map_id in range(N_MAPS):
        h = drv.create_scenario(lib, t.DATA, map_id, t.JSONS[map_id], physics="through")
        try:
            for g in grids:
                amin, amax, na, smin, smax, ns = g
                ade, n_agents, clip_a, clip_s = _ade(
                    lib, h, amin, amax, na, smin, smax, ns)
                per_map[g].append(ade)
                clips[g].append((clip_a, clip_s, n_agents))
            ade, n_agents, clip_a, clip_s = _ade(lib, h, *BIKE_CONTINUOUS)
            cont_map.append(ade)
            cont_clips.append((clip_a, clip_s, n_agents))
            for g in delta_grids:
                ade, n_agents, cx, cy, cyaw = _delta(lib, h, *g)
                delta_map[g].append(ade)
                delta_clips[g].append((cx, cy, cyaw, n_agents))
            agent_counts.append(clips[grids[0]][-1][2])
        finally:
            lib.pf_close(h)
        print(
            f"map {map_id:3d} agents {agent_counts[-1]}"
            f"  cont {cont_map[-1]:.3f}",
            flush=True,
        )
    rows = []
    for g in grids:
        vals = np.asarray(per_map[g], dtype=np.float64)
        clip = np.asarray(clips[g], dtype=np.float64)
        amin, amax, na, smin, smax, ns = g
        rows.append({
            "accel_min": amin,
            "accel_max": amax,
            "n_accel": na,
            "steer_min": smin,
            "steer_max": smax,
            "n_steer": ns,
            "ade_mean": float(vals.mean()),
            "ade_std_maps": float(vals.std()),
            "clip_accel": float(clip[:, 0].mean()),
            "clip_steer": float(clip[:, 1].mean()),
            "agents_per_map": float(clip[:, 2].mean()),
            "reference": bool(na == 21 and ns == 31),
        })
    rows.sort(key=lambda r: r["ade_mean"])
    delta_rows = []
    for g in delta_grids:
        if g == (0, 0, 0):
            kind = "continuous"
        elif g == DELTA_PAPER:
            kind = "paper"
        else:
            kind = "equal"
        delta_rows.append(_delta_row(g[0], g[1], g[2], delta_map[g], delta_clips[g], kind))
    cont_vals = np.asarray(cont_map, dtype=np.float64)
    cont_clip = np.asarray(cont_clips, dtype=np.float64)
    continuous_bicycle = {
        "accel_min": BIKE_CONTINUOUS[0],
        "accel_max": BIKE_CONTINUOUS[1],
        "n_accel": 0,
        "steer_min": BIKE_CONTINUOUS[3],
        "steer_max": BIKE_CONTINUOUS[4],
        "n_steer": 0,
        "ade_mean": float(cont_vals.mean()),
        "ade_std_maps": float(cont_vals.std()),
        "ade_median_maps": float(np.median(cont_vals)),
        "clip_accel": float(cont_clip[:, 0].mean()),
        "clip_steer": float(cont_clip[:, 1].mean()),
        "agents_per_map": float(cont_clip[:, 2].mean()),
    }
    discrete_delta = [
        r for r in delta_rows if r["kind"] in ("equal", "paper")
    ]
    discrete_delta.sort(key=lambda r: r["ade_mean"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "maps": N_MAPS,
        "maps_are": "sorted json index 0..49",
        "agents": "base_unfit == 0, one closed-loop rollout, all of them together",
        "model": (
            "kinematic bicycle in drive.h: v += a dt, then the circular arc "
            "at the new speed. Wheelbase is max(0.6 * length, 1 m). "
            "Steer is atan(L dtheta / v1 dt) from the logged heading change."
        ),
        "action": (
            "log inverse from the logged pose, unclamped, then nearest "
            "uniform bin inside the bound. Even bin counts have no exact zero. "
            "Bin count 0 clips and does not snap."
        ),
        "error": "mean over cars of mean distance to the log after each step",
        "accel_bounds_m_s2": list(ACCEL_BOUNDS),
        "steer_bounds_rad": list(STEER_BOUNDS),
        "bins": list(BINS),
        "best": rows[0],
        "continuous_bicycle": continuous_bicycle,
        "best_discrete_delta": discrete_delta[0],
        "rows": rows,
        "delta": {
            "dx_m": 3.5,
            "dy_m": 0.1,
            "dyaw_rad": float(np.pi / 6.0),
            "a_long_max_m_s2": 8.0,
            "max_steer_rad": 0.7,
            "rows": delta_rows,
        },
    }
    json_path = OUT.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {json_path}", flush=True)
    print(
        f"best ade {rows[0]['ade_mean']:.3f} m  "
        f"accel ±{rows[0]['accel_max']:g} x {rows[0]['n_accel']}  "
        f"steer ±{rows[0]['steer_max']:g} x {rows[0]['n_steer']}",
        flush=True,
    )
    print(
        f"continuous bicycle {continuous_bicycle['ade_mean']:.3f} ± "
        f"{continuous_bicycle['ade_std_maps']:.3f}",
        flush=True,
    )
    for r in delta_rows:
        print(
            f"delta {r['kind']:10s} {r['n_dx']:3d}/{r['n_dy']:3d}/{r['n_yaw']:3d}"
            f"  ade {r['ade_mean']:.3f} ± {r['ade_std_maps']:.3f}"
            f"  clip dx {r['clip_dx']:.3f} dy {r['clip_dy']:.3f} yaw {r['clip_yaw']:.3f}",
            flush=True,
        )

def _default_bound_rows(rows):
    series = []
    for row in rows:
        if row.get("reference"):
            continue
        if abs(float(row["accel_max"]) - 8.0) > 1e-6:
            continue
        if abs(float(row["steer_max"]) - 1.5) > 1e-6:
            continue
        if int(row["n_accel"]) != int(row["n_steer"]):
            continue
        if int(row["n_accel"]) <= 0:
            continue
        series.append(row)
    series.sort(key=lambda row: int(row["n_accel"]))
    return series

def plot_upper_bound(summary, path):
    series = _default_bound_rows(summary["rows"])
    if not series:
        raise SystemExit("action_grid.json has no ±8 m/s², ±1.5 rad bicycle rows")
    cont_bike = float(summary["continuous_bicycle"]["ade_mean"])
    disc_delta = float(summary["best_discrete_delta"]["ade_mean"])
    cont_delta = float(next(
        row["ade_mean"] for row in summary["delta"]["rows"] if row["kind"] == "continuous"
    ))
    disc_bins = (
        int(summary["best_discrete_delta"]["n_dx"]),
        int(summary["best_discrete_delta"]["n_dy"]),
        int(summary["best_discrete_delta"]["n_yaw"]),
    )
    if len(set(disc_bins)) != 1:
        raise SystemExit(f"discrete delta bins are not equal: {disc_bins}")
    n_delta = disc_bins[0]
    xs = [int(row["n_accel"]) for row in series]
    ades = [float(row["ade_mean"]) for row in series]
    t.apply_figure_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.axhline(
        cont_bike, color="0.2", lw=1.15, ls="--", zorder=1,
        label=f"continuous bicycle  {cont_bike:.2f} m",
    )
    ax.axhline(
        cont_delta, color="#6b3fa0", lw=1.15, ls=":", zorder=1,
        label=f"continuous delta-local  {cont_delta:.2f} m",
    )
    ax.scatter(
        xs, ades, s=64, marker="o", color="#4c78a8",
        edgecolors="0.15", linewidths=0.5, zorder=3,
        label="discrete bicycle  ±8 m/s², ±1.5 rad",
    )
    ax.scatter(
        [n_delta], [disc_delta], s=70, marker="D", color="#6b3fa0",
        edgecolors="0.15", linewidths=0.5, zorder=4,
        label=f"discrete delta-local  {n_delta}/{n_delta}/{n_delta}",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(n) for n in xs])
    ax.set_xlim(xs[0] / 1.35, xs[-1] * 1.45)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("bins on each head")
    ax.set_ylabel("ADE (m)")
    ax.set_title("Bicycle ±8 m/s², ±1.5 rad")
    t.style_ax(ax)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.suptitle(
        "Replay upper bound, 50 maps. Inferred log command, closed loop"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    return series

def plot_upper_bound_from_json(json_path=None, png_path=None):
    json_path = Path(json_path) if json_path else OUT.with_suffix(".json")
    png_path = Path(png_path) if png_path else OUT.parent / "replay_upper_bound.png"
    summary = json.loads(json_path.read_text())
    return plot_upper_bound(summary, png_path)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--overview":
        plot_upper_bound_from_json()
    else:
        main()
