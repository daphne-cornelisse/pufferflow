#!/usr/bin/env python3
"""Sweep panel for the v1 advantage curriculum.

Reads sweep.log and the per-run ini files. The top row is the sweep
score, explained variance against shared-encoder Spearman rho, the
trunk gradient ratio, and the top-20% gradient share. The next row is
standard against hard, or the logged anchor against pool. One panel
plots the hard fraction and the easy fraction against the final score.
Another plots each copy's hardness against its mean absolute advantage.
One panel is the final score on each map against centerline density.
The panel beside it is hard fraction against wall time, colored by score.
The remaining panels are the swept hyperparameters against the final score.

    python scripts/plot_curr_sweep.py logs/drive-currv1
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RUN_LINE = re.compile(
    r"sweep run=(\d+) score=([0-9.eE+-]+) cost=([0-9.eE+-]+)"
)
BEST_COLOR = "#55A868"

# default.ini sweep keys, then the two drive.ini extras.
HYPERS = (
    ("total_timesteps", "train", "total_timesteps", True),
    ("hidden_size", "policy", "hidden_size", False),
    ("num_layers", "policy", "num_layers", False),
    ("total_agents", "vec", "total_agents", False),
    ("num_buffers", "vec", "num_buffers", False),
    ("horizon", "train", "horizon", False),
    ("minibatch_size", "train", "minibatch_size", False),
    ("learning_rate", "train", "learning_rate", True),
    ("ent_coef", "train", "ent_coef", True),
    ("gamma", "train", "gamma", False),
    ("gae_lambda", "train", "gae_lambda", False),
    ("clip_coef", "train", "clip_coef", False),
    ("vf_clip_coef", "train", "vf_clip_coef", False),
    ("vf_coef", "train", "vf_coef", False),
    ("max_grad_norm", "train", "max_grad_norm", False),
    ("momentum", "train", "momentum", False),
    ("anchor_coef", "train", "anchor_coef", True),
    ("reward_goal", "env", "reward_goal", False),
    ("hard_frac", "env", "hard_frac", False),
)

# The finished currv1 logs use anchor/pool. Later runs use standard/hard.
SPLIT_SETS = (
    (
        ("standard", "hard"),
        (
            ("env/standard_mean_reward", "env/hard_mean_reward", "mean reward"),
            ("env/standard_collision_rate", "env/hard_collision_rate", "collision rate"),
            ("env/standard_episode_length", "env/hard_episode_length", "episode length"),
            ("env/standard_mean_abs_adv", "env/hard_mean_abs_adv", "mean |A|"),
        ),
    ),
    (
        ("anchor", "pool"),
        (
            ("env/anchor_mean_reward", "env/pool_mean_reward", "mean reward"),
            ("env/anchor_collision_rate", "env/pool_collision_rate", "collision rate"),
            ("env/anchor_episode_length", "env/pool_episode_length", "episode length"),
            ("env/anchor_mean_abs_adv", "env/pool_mean_abs_adv", "mean |A|"),
        ),
    ),
)
SPLIT_KEYS = tuple(
    key for _names, rows in SPLIT_SETS for row in rows for key in row[:2]
)
MAPS = (
    ("env/score_town_01", "Town01"),
    ("env/score_town_02", "Town02"),
    ("env/score_town_06", "Town06"),
    ("env/score_town_10", "Town10HD"),
    ("env/score_bottleneck", "Bottleneck"),
)


def _floats(line):
    parts = [part.strip() for part in line.split("=", 1)[1].split(",")]
    return [float(part) for part in parts if part]


def metric_series(text, key):
    prefix = key + " ="
    for line in text.splitlines():
        if line.startswith(prefix):
            return _floats(line)
    return None


def section_series(text, section, key):
    """Comma list under one [section]. Bare keys repeat across sections."""
    prefix = key + " ="
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped[1:-1] == section
            continue
        if in_section and line.startswith(prefix):
            return _floats(line)
    return None


def section_value(text, section, key):
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped[1:-1] == section
            continue
        if not in_section or "=" not in stripped or stripped.startswith("#"):
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            try:
                return float(value.strip().split()[0])
            except ValueError:
                return None
    return None


def sweep_scores(log_path):
    """Run id to (standard score, wall-clock seconds). Later lines win."""
    logged = {}
    paths = [log_path]
    if log_path.name == "sweep.log":
        paths.append(log_path.with_name("sweep.out"))
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            match = RUN_LINE.search(line)
            if match:
                logged[int(match.group(1))] = (
                    float(match.group(2)),
                    float(match.group(3)),
                )
    return logged


def load_runs(metrics_dir):
    runs = []
    for path in sorted(metrics_dir.glob("sweep_*.ini")):
        run = int(path.stem.rsplit("_", 1)[-1])
        text = path.read_text(errors="replace")
        steps = metric_series(text, "agent_steps")
        score = metric_series(text, "env/standard_score")
        score_name = "standard score"
        if not score or not steps or len(score) != len(steps):
            score = metric_series(text, "env/score")
            score_name = "score"
        if not steps or not score or len(steps) != len(score):
            continue
        overall = metric_series(text, "env/score")
        default_score = overall[-1] if overall else score[-1]
        row = {
            "run": run,
            "path": path,
            "steps": steps,
            "score": score,
            "score_name": score_name,
            "final_score": score[-1],
            "default_score": default_score,
        }
        for key in (
            "value/explained_var",
            "adv/spearman_shared",
            "adv/ratio_trunk",
            "adv/share_top",
        ):
            series = metric_series(text, key)
            row[key] = series if series and len(series) == len(steps) else None
        diag_step = section_series(text, "adv_diag", "step")
        row["diag_step"] = diag_step
        for key in (
            "explained_var",
            "spearman_shared",
            "ratio_trunk",
            "share_top",
        ):
            series = section_series(text, "adv_diag", key)
            row[key] = (
                series if series and diag_step and len(series) == len(diag_step)
                else None
            )
        for key in SPLIT_KEYS:
            series = metric_series(text, key)
            row[key] = series if series and len(series) == len(steps) else None
        for key, _title in MAPS:
            series = metric_series(text, key)
            row[key] = series if series and len(series) == len(steps) else None
        for label, section, key, _log in HYPERS:
            row[label] = section_value(text, section, key)
        hard_d = section_series(text, "hardness", "d")
        hard_abs = section_series(text, "hardness", "abs")
        if hard_d and hard_abs and len(hard_d) == len(hard_abs):
            row["hard_d"] = hard_d
            row["hard_abs"] = hard_abs
        else:
            row["hard_d"] = None
            row["hard_abs"] = None
        runs.append(row)
    return runs


def choose_split(runs):
    """Prefer standard/hard. Fall back to the logged anchor/pool series."""
    for names, rows in SPLIT_SETS:
        left, right, _label = rows[0]
        if any(row.get(left) or row.get(right) for row in runs):
            return names, rows
    return SPLIT_SETS[0]


def apply_style():
    sns.set_style("ticks")
    sns.set_palette("deep")


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


HI_SCORE = 0.95


def scatter_scores(ax, x, y, score, c=None, cmap=None, color=None, **kw):
    """Circles at or below 0.95, triangles above. `score` picks the marker."""
    x = list(x)
    y = list(y)
    score = list(score)
    n = len(x)
    colors = None
    values = None
    if c is not None and not isinstance(c, str):
        values = list(c)
    elif color is not None and not isinstance(color, str):
        try:
            if len(color) == n:
                colors = list(color)
        except TypeError:
            colors = None
    norm = None
    if values:
        finite = [v for v in values if v == v]
        if finite:
            norm = plt.Normalize(min(finite), max(finite))
    artist = None
    hi = [s > HI_SCORE for s in score]
    for flag, marker in ((False, "o"), (True, "^")):
        idx = [i for i, flag_i in enumerate(hi) if flag_i is flag]
        if not idx:
            continue
        call = dict(kw)
        call["marker"] = marker
        xs = [x[i] for i in idx]
        ys = [y[i] for i in idx]
        if values is not None:
            artist = ax.scatter(
                xs, ys, c=[values[i] for i in idx],
                cmap=cmap, norm=norm, **call,
            )
        else:
            if colors is not None:
                call["c"] = [colors[i] for i in idx]
            elif color is not None:
                call["c"] = color
            artist = ax.scatter(xs, ys, **call)
    return artist


def restyle_swarm(ax, scores, color, size, zorder, edgecolor, linewidth):
    """Swap a just-drawn swarm so scores above 0.95 are triangles."""
    coll = ax.collections[-1]
    offsets = coll.get_offsets()
    scores = list(scores)
    coll.remove()
    if len(offsets) != len(scores):
        ax.scatter(
            offsets[:, 0], offsets[:, 1], color=color, s=size ** 2,
            linewidths=linewidth, edgecolors=edgecolor, zorder=zorder,
        )
        return
    for flag, marker in ((False, "o"), (True, "^")):
        idx = [i for i, s in enumerate(scores) if (s > HI_SCORE) is flag]
        if not idx:
            continue
        ax.scatter(
            offsets[idx, 0], offsets[idx, 1],
            color=color, s=size ** 2, marker=marker,
            linewidths=linewidth, edgecolors=edgecolor, zorder=zorder,
        )
    del facecolor


def town_densities():
    """128-car centerline share, from config/town_density.json."""
    path = Path(__file__).resolve().parents[1] / "config" / "town_density.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if int(payload.get("agents", 0)) != 128:
        return {}
    out = {}
    for row in payload.get("towns", []):
        if int(row.get("agents", 0)) != 128:
            continue
        if "centerline_pct" not in row or "town" not in row:
            continue
        out[row["town"]] = float(row["centerline_pct"])
    return out


def draw_town_density(ax, runs, best):
    """Final map score against centerline coverage. One point per run."""
    density = town_densities()
    rows = []
    for row in runs:
        for key, title in MAPS:
            ys = row.get(key)
            if not ys or title not in density:
                continue
            rows.append({
                "run": row["run"],
                "town": title,
                "density": density[title],
                "score": float(ys[-1]),
                "best": row["run"] == best,
            })
    if not rows:
        ax.set_title("Town score")
        ax.set_xlabel("% of centerline")
        ax.set_ylabel("score")
        return
    frame = pd.DataFrame(rows)
    placed = (
        frame[["town", "density"]]
        .drop_duplicates()
        .sort_values("density")
    )
    densities = [float(value) for value in placed["density"]]
    gaps = [b - a for a, b in zip(densities, densities[1:])]
    span = min(gaps) if gaps else densities[0] * 0.4
    sns.violinplot(
        data=frame, x="density", y="score",
        native_scale=True, fill=False, inner=None,
        color="0.35", linewidth=1.0, width=0.8,
        cut=0, density_norm="width",
        ax=ax, legend=False, zorder=1,
    )
    other = frame[~frame["best"]]
    marked = frame[frame["best"]]
    if len(other):
        sns.swarmplot(
            data=other, x="density", y="score",
            color=sns.color_palette("deep")[0],
            native_scale=True, size=2.4, ax=ax, legend=False,
            edgecolor="white", linewidth=0.2, zorder=3,
        )
        restyle_swarm(
            ax, other["score"], sns.color_palette("deep")[0],
            2.4, 3, "white", 0.2,
        )
    if len(marked):
        sns.swarmplot(
            data=marked, x="density", y="score",
            color=BEST_COLOR, native_scale=True, size=3.2,
            ax=ax, legend=False, zorder=4,
        )
        restyle_swarm(ax, marked["score"], BEST_COLOR, 3.2, 4, "white", 0.0)
    half = 0.32 * span
    for dens, scores in frame.groupby("density")["score"]:
        mean = float(scores.mean())
        ax.plot(
            [float(dens) - half, float(dens) + half],
            [mean, mean],
            color="0.1", linewidth=1.8, solid_capstyle="butt", zorder=5,
        )
    pad = min(gaps) * 0.75 if gaps else densities[0] * 0.2
    ax.set_xlim(densities[0] - pad, densities[-1] + pad)
    ax.set_autoscalex_on(False)
    ax.set_xticks(densities)
    ax.set_xticklabels([
        f"{town}\n{dens:.2f}"
        for town, dens in zip(placed["town"], densities)
    ])
    ax.tick_params(axis="x", labelsize=8)
    labels = ax.get_xticklabels()
    for i in range(1, len(densities)):
        if densities[i] - densities[i - 1] >= 1.5:
            continue
        shift = ScaledTranslation(0, -26 / 72, ax.figure.dpi_scale_trans)
        labels[i].set_transform(labels[i].get_transform() + shift)
    finite = frame["score"].dropna()
    if len(finite) == 0 or float(finite.max()) <= 1.0:
        ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("% of centerline")
    ax.set_ylabel("score")
    ax.set_title("Town score vs density")


def line_runs(ax, runs, best, ykey, use_diag=False):
    color = sns.color_palette("deep")[0]
    xkey = "diag_step" if use_diag else "steps"
    for row in runs:
        ys = row.get(ykey)
        xs = row.get(xkey)
        if not ys or not xs:
            continue
        mark = row["run"] == best
        ax.plot(
            xs, ys,
            color=BEST_COLOR if mark else color,
            linewidth=2.2 if mark else 1.0,
            alpha=1.0 if mark else 0.45,
            zorder=3 if mark else 2,
        )


def draw(runs, out):
    apply_style()
    best = max(runs, key=lambda row: row["final_score"])["run"]
    fig = plt.figure(figsize=(16.0, 22.0), dpi=300)
    grid = fig.add_gridspec(
        8, 4, hspace=0.72, wspace=0.52,
        height_ratios=[1, 1, 1.45, 1, 1, 1, 1, 1],
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    split_axes = [fig.add_subplot(grid[1, i]) for i in range(4)]
    density_ax = fig.add_subplot(grid[2, 0:2])
    time_ax = fig.add_subplot(grid[2, 2:4])
    hyper_axes = [
        fig.add_subplot(grid[3 + i // 4, i % 4]) for i in range(len(HYPERS))
    ]

    line_runs(axes[0], runs, best, "score")
    score_name = runs[0].get("score_name", "score")
    axes[0].set_title(score_name.capitalize())
    axes[0].set_xlabel("steps")
    axes[0].set_ylabel(score_name)
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ev_ax = axes[1]
    points = []
    for row in runs:
        xs = row.get("explained_var")
        ys = row.get("spearman_shared")
        steps = row.get("diag_step")
        if not xs or not ys or not steps:
            continue
        ratios = row.get("ratio_trunk") or []
        for i, (x, y, step) in enumerate(zip(xs, ys, steps)):
            # Capture stores explained variance and leaves the grad fields at 0.
            if i == 0 and y == 0.0 and (not ratios or ratios[0] == 0.0):
                continue
            points.append((x, y, step))
    if points:
        frame = pd.DataFrame(points, columns=["ev", "rho", "step"])
        sc = ev_ax.scatter(
            frame["ev"], frame["rho"], c=frame["step"], cmap="viridis",
            s=18, linewidths=0, zorder=3,
        )
        bar = fig.colorbar(sc, ax=ev_ax, pad=0.02, fraction=0.046)
        bar.set_label("training step")
        bar.formatter.set_scientific(True)
        bar.formatter.set_powerlimits((0, 0))
    ev_ax.add_patch(Rectangle(
        (0.60, 0.70), 0.45, 0.35,
        facecolor="#55A868", edgecolor="#55A868", alpha=0.18,
        linewidth=1.0, zorder=1,
    ))
    ev_ax.axvline(0.60, color="#55A868", linestyle="--", linewidth=0.8, zorder=2)
    ev_ax.axhline(0.70, color="#55A868", linestyle="--", linewidth=0.8, zorder=2)
    ev_ax.set_title("EV vs Spearman ρ")
    ev_ax.set_xlabel("explained variance")
    ev_ax.set_ylabel("spearman ρ")

    idx_ax = axes[2]
    idx_rows = [row for row in runs if row.get("wall_min") is not None]
    if idx_rows:
        sc = scatter_scores(
            idx_ax,
            [row["run"] for row in idx_rows],
            [row["default_score"] for row in idx_rows],
            [row["default_score"] for row in idx_rows],
            c=[row["wall_min"] for row in idx_rows],
            cmap="viridis", s=36, linewidths=0.4,
            edgecolors="white", zorder=3,
        )
        cax = make_axes_locatable(idx_ax).append_axes(
            "right", size="5%", pad=0.08,
        )
        bar = fig.colorbar(sc, cax=cax)
        bar.set_label("wall time (min)")
    idx_ax.set_title("Default score")
    idx_ax.set_xlabel("sweep index")
    idx_ax.set_ylabel("default score")

    line_runs(axes[3], runs, best, "share_top", use_diag=True)
    axes[3].axhline(0.70, color="0.35", linestyle="--", linewidth=0.9, zorder=1)
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_title("Share top 20%")
    axes[3].set_xlabel("steps")
    axes[3].set_ylabel("share")
    axes[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    split_names, split_rows = choose_split(runs)
    left_c, right_c = sns.color_palette("deep")[:2]
    for ax, (left, right, label) in zip(split_axes, split_rows):
        for row in runs:
            mark = row["run"] == best
            for key, color in ((left, left_c), (right, right_c)):
                ys = row.get(key)
                if not ys:
                    continue
                ax.plot(
                    row["steps"], ys, color=color,
                    linewidth=2.2 if mark else 1.0,
                    alpha=1.0 if mark else 0.4,
                    zorder=3 if mark else 2,
                )
        ax.set_title(label)
        ax.set_xlabel("steps")
        ax.set_ylabel(label)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    split_axes[0].plot([], [], color=left_c, label=split_names[0])
    split_axes[0].plot([], [], color=right_c, label=split_names[1])
    split_axes[0].legend(frameon=False, loc="best")

    draw_town_density(density_ax, runs, best)
    timed = [
        row for row in runs
        if row.get("hard_frac") is not None and row.get("wall_min") is not None
    ]
    if timed:
        sc = scatter_scores(
            time_ax,
            [row["hard_frac"] for row in timed],
            [row["wall_min"] for row in timed],
            [row["final_score"] for row in timed],
            c=[row["final_score"] for row in timed],
            cmap="Greens", s=36, linewidths=0.4,
            edgecolors="white", zorder=3,
        )
        bar = fig.colorbar(sc, ax=time_ax, pad=0.02, fraction=0.046)
        bar.set_label(score_name)
    time_ax.set_xlim(-0.05, 1.05)
    time_ax.set_xlabel("hard fraction")
    time_ax.set_ylabel("wall time (min)")
    time_ax.set_title("Hard fraction vs wall time")

    summary = pd.DataFrame(runs)
    easy_c, hard_c = sns.color_palette("deep")[:2]
    for ax, (label, _section, _key, log_x) in zip(hyper_axes, HYPERS):
        part = summary.dropna(subset=[label])
        if label == "hard_frac":
            timed = part.dropna(subset=["wall_min"]) if "wall_min" in part else part
            if len(timed):
                sc = scatter_scores(
                    ax, timed[label], timed["final_score"],
                    timed["final_score"], c=timed["wall_min"],
                    cmap="viridis", s=36, linewidths=0.4,
                    edgecolors="white", zorder=3,
                )
                bar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
                bar.set_label("wall time (min)")
            ax.set_xlim(-0.05, 1.05)
            ax.set_xlabel("hard fraction")
        elif len(part):
            colors = [
                BEST_COLOR if int(run) == best else sns.color_palette("deep")[0]
                for run in part["run"]
            ]
            scatter_scores(
                ax, part[label], part["final_score"], part["final_score"],
                color=colors, s=28, linewidths=0.4,
                edgecolors="white", zorder=3,
            )
            ax.set_xlabel(label.replace("_", " "))
        else:
            ax.set_xlabel(label.replace("_", " "))
        ax.set_ylabel(score_name)
        ax.tick_params(axis="x", labelsize=7)
        if log_x and len(part) and float(part[label].min()) > 0:
            ax.set_xscale("log")

    hard_ax = fig.add_subplot(grid[7, 3])
    for row in runs:
        xs = row.get("hard_d")
        ys = row.get("hard_abs")
        if not xs or not ys:
            continue
        mark = row["run"] == best
        hard_ax.scatter(
            xs, ys,
            color=BEST_COLOR if mark else easy_c,
            s=16 if mark else 10,
            linewidths=0,
            alpha=1.0 if mark else 0.35,
            zorder=3 if mark else 2,
        )
    hard_ax.set_xlim(-0.05, 1.05)
    hard_ax.set_xlabel("hardness")
    hard_ax.set_ylabel("advantage magnitude")
    hard_ax.set_title("Hardness vs |A|")

    for ax in list(axes) + list(split_axes) + [density_ax, time_ax] + list(hyper_axes) + [hard_ax]:
        style_ax(ax)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path, default=ROOT / "logs" / "drive-currv1",
        help="sweep directory with sweep.log and sweep_*.ini",
    )
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    metrics = root / "drive" if (root / "drive").is_dir() else root
    log_path = root / "sweep.log"
    if not log_path.is_file() and (metrics / "sweep.log").is_file():
        log_path = metrics / "sweep.log"
    runs = load_runs(metrics)
    if not runs:
        raise SystemExit(f"no finished runs in {metrics}")
    logged = sweep_scores(log_path)
    for row in runs:
        if row["run"] in logged:
            score, wall_s = logged[row["run"]]
            row["final_score"] = score
            row["wall_min"] = wall_s / 60.0
    out = args.out or (root / "sweep-panels.png")
    best = draw(runs, out)
    print(f"wrote {out} runs={len(runs)} best={best}")


if __name__ == "__main__":
    main()
