#!/usr/bin/env python3
"""Plot a protein sweep from its log and per-run metrics.

The figure is three rows. Episode return against sweep index is colored by
trial wall clock. Every other panel colors runs with ColorBrewer Set1.
Panels are return curves, goal score, collision rate, score against
jerk when those series exist (otherwise collision count), and parameter count. The bottom row is a boxplot of the swept
hyperparameters with one swarm point per run. Each hyperparameter is scaled
to its own min and max.

    python scripts/plot_sweep.py logs/drive/sweep.log -o logs/drive/sweep-panels.png

Wall clock is the trial cost in the sweep log, in minutes. Goal score and
collision rate come from the binned env/ metrics written at the end of each
run. Runs that failed before writing metrics are left out. Figures go under
logs/drive.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RUN_LINE = re.compile(
    r"sweep run=(\d+) score=\S+ cost=([0-9.eE+-]+)"
)
# Drive observation is 8 ego + 64*7 partners + 75*7 road. Decoder is
# 7 accel logits, 13 steer logits, and one value output.
POLICY_PARAMS_PER_HIDDEN = 8 + 7 * 64 + 7 * 75 + 7 + 13 + 1
# Swept settings shown on the bottom row. gae_lambda is the GAE discount.
HYPER_FIELDS = (
    ("gamma", "train", "gamma"),
    ("gae_lambda", "train", "gae_lambda"),
    ("learning_rate", "train", "learning_rate"),
    ("ent_coef", "train", "ent_coef"),
    ("replay_ratio", "train", "replay_ratio"),
    ("clip_coef", "train", "clip_coef"),
    ("vf_coef", "train", "vf_coef"),
    ("vf_clip_coef", "train", "vf_clip_coef"),
    ("momentum", "train", "momentum"),
    ("max_grad_norm", "train", "max_grad_norm"),
    ("hidden_size", "policy", "hidden_size"),
    ("num_layers", "policy", "num_layers"),
    ("horizon", "train", "horizon"),
    ("minibatch_size", "train", "minibatch_size"),
    ("total_agents", "vec", "total_agents"),
    ("total_timesteps", "train", "total_timesteps"),
)


def parse_sweep_log(path):
    runs = []
    for line in path.read_text().splitlines():
        match = RUN_LINE.search(line)
        if match:
            runs.append((int(match.group(1)), float(match.group(2)) / 60.0))
    if not runs:
        raise SystemExit(f"no finished runs in {path}")
    return runs


def metric_series(text, key):
    prefix = key + " ="
    for line in text.splitlines():
        if line.startswith(prefix):
            return [float(part) for part in line.split("=", 1)[1].split(",") if part.strip()]
    return None


def section_value(text, section, key):
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped[1:-1] == section
            continue
        if not in_section or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            return float(value.strip())
    return None


def policy_params(hidden, layers):
    """Weight count. num_layers is truncated to int, same as training."""
    hidden = int(hidden)
    layers = int(layers)
    return hidden * POLICY_PARAMS_PER_HIDDEN + layers * 3 * hidden * hidden


def metrics_file(metrics_dir, run, log_mtime):
    """Newest ini for this run index that belongs to this sweep log."""
    hits = []
    for path in metrics_dir.glob(f"sweep_*_{run:04d}.ini"):
        if path.stat().st_mtime <= log_mtime + 1.0:
            hits.append(path)
    if not hits:
        return None
    return max(hits, key=lambda path: path.stat().st_mtime)


def load_frames(log_path, metrics_dir):
    log_mtime = log_path.stat().st_mtime
    returns = []
    collisions = []
    counts = []
    summary = []
    missing = []
    for run, wall_min in parse_sweep_log(log_path):
        path = metrics_file(metrics_dir, run, log_mtime)
        if path is None:
            missing.append(run)
            continue
        text = path.read_text(errors="replace")
        return_vals = metric_series(text, "env/episode_return")
        score_vals = metric_series(text, "env/score")
        collision_vals = metric_series(text, "env/collision_rate")
        jerk_vals = metric_series(text, "env/jerk")
        smooth_vals = metric_series(text, "env/smooth")
        count_vals = metric_series(text, "env/collision_count")
        steps = metric_series(text, "agent_steps")
        hidden = section_value(text, "policy", "hidden_size")
        layers = section_value(text, "policy", "num_layers")
        if (not return_vals or not score_vals or not collision_vals
                or not steps or hidden is None or layers is None
                or len(steps) != len(return_vals) or len(steps) != len(collision_vals)):
            missing.append(run)
            continue
        for step, value in zip(steps, return_vals):
            returns.append({"run": run, "steps": step, "episode_return": value})
        for step, value in zip(steps, collision_vals):
            collisions.append({"run": run, "steps": step, "collision_rate": value})
        if count_vals and len(count_vals) == len(steps):
            for step, value in zip(steps, count_vals):
                counts.append({"run": run, "steps": step, "collision_count": value})
        summary.append({
            "run": run,
            "score": score_vals[-1],
            "episode_return": return_vals[-1],
            "jerk": jerk_vals[-1] if jerk_vals else float("nan"),
            "smooth": smooth_vals[-1] if smooth_vals else float("nan"),
            "wall_min": wall_min,
            "collision_rate": collision_vals[-1],
            "collision_count": (
                count_vals[-1] if count_vals and len(count_vals) == len(steps) else float("nan")
            ),
            "parameters": policy_params(hidden, layers),
        })
        for label, section, key in HYPER_FIELDS:
            value = section_value(text, section, key)
            summary[-1][label] = float(value) if value is not None else float("nan")
    if not summary:
        raise SystemExit(f"no metrics for runs in {log_path}")
    return (
        pd.DataFrame(returns),
        pd.DataFrame(collisions),
        pd.DataFrame(counts),
        pd.DataFrame(summary),
        missing,
    )


def series_key(run):
    return f"{int(run):04d}"


def distinct_colors(count):
    """ColorBrewer Set1. Extra runs repeat the nine colors."""
    base = [tuple(color) for color in sns.color_palette("Set1")]
    if count <= 0:
        return []
    return [base[i % len(base)] for i in range(count)]


def categorical_palette(run_ids):
    """Set1 color per run. The same run keeps that color on every panel."""
    ordered = list(dict.fromkeys(int(run) for run in run_ids))
    colors = distinct_colors(len(ordered))
    return {series_key(run): colors[i] for i, run in enumerate(ordered)}


def with_series(frame):
    out = frame.copy()
    out["series"] = out["run"].map(series_key)
    return out


def style_steps(ax):
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_xlabel("steps")


def figure_axes():
    """Two rows of result panels and a full-width hyperparameter row."""
    fig = plt.figure(figsize=(16.8, 12.4), dpi=150)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.2])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    return fig, axes, fig.add_subplot(grid[2, :])


def draw_hypers(ax, summary, palette):
    """Boxplot of each swept hyperparameter, with one swarm point per run.

    Values are scaled to the min and max of that hyperparameter in this figure.
    """
    rows = []
    for record in summary.to_dict("records"):
        for label, _section, _key in HYPER_FIELDS:
            value = record.get(label)
            if value is None or value != value:
                continue
            rows.append({"series": record["series"], "hyper": label, "value": float(value)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        ax.set_axis_off()
        ax.set_title("No hyperparameters")
        return
    def scale(values):
        lo = float(values.min())
        hi = float(values.max())
        if hi == lo:
            return pd.Series(0.5, index=values.index)
        return (values - lo) / (hi - lo)
    frame["scaled"] = frame.groupby("hyper", sort=False)["value"].transform(scale)
    order = [label for label, _section, _key in HYPER_FIELDS if label in set(frame["hyper"])]
    sns.boxplot(
        data=frame, x="hyper", y="scaled", order=order, ax=ax,
        color="#d0d0d0", fliersize=0, width=0.62, linewidth=0.8,
    )
    sns.swarmplot(
        data=frame, x="hyper", y="scaled", order=order, hue="series",
        palette=palette, ax=ax, legend=False, size=3.2,
    )
    ax.set_xlabel("")
    ax.set_ylabel("scaled value")
    ax.set_title("Hyperparameters")
    ax.set_ylim(-0.08, 1.08)
    ax.tick_params(axis="x", labelrotation=32)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def draw(returns, collisions, counts, summary, out):
    sns.set_theme(style="whitegrid", context="notebook")
    palette = categorical_palette(summary["run"])
    summary = with_series(summary)
    returns = with_series(returns)
    collisions = with_series(collisions)
    if len(counts):
        counts = with_series(counts)

    fig, axes, hyper_ax = figure_axes()

    lo = float(summary["wall_min"].min())
    hi = float(summary["wall_min"].max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    norm = plt.Normalize(lo, hi)
    sns.scatterplot(
        data=summary, x="run", y="episode_return", hue="wall_min",
        hue_norm=norm, palette="viridis", s=58, ax=axes[0],
        legend=False, edgecolor="white", linewidth=0.6,
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[0], pad=0.02,
    )
    colorbar.set_label("wall clock (min)")
    axes[0].set_xscale("linear")
    axes[0].set_yscale("linear")
    axes[0].set_xlabel("sweep index")
    axes[0].set_ylabel("episode return")
    axes[0].set_title("Episode return vs sweep index")

    sns.lineplot(
        data=returns, x="steps", y="episode_return", hue="series",
        palette=palette, estimator=None, linewidth=1.3, ax=axes[1], legend=False,
    )
    axes[1].set_xscale("linear")
    axes[1].set_yscale("linear")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[1].ticklabel_format(axis="y", style="plain")
    axes[1].set_xlabel("steps")
    axes[1].set_ylabel("episode return")
    axes[1].set_title("Episode return vs steps")

    sns.scatterplot(
        data=summary, x="run", y="score", hue="series",
        palette=palette, s=58, ax=axes[2], legend=False,
        edgecolor="white", linewidth=0.6,
    )
    axes[2].set_xlabel("sweep index")
    axes[2].set_ylabel("score")
    axes[2].set_title("Score")
    axes[2].set_ylim(-0.03, 1.03)

    sns.lineplot(
        data=collisions, x="steps", y="collision_rate", hue="series",
        palette=palette, estimator=None, linewidth=1.3, ax=axes[3], legend=False,
    )
    style_steps(axes[3])
    axes[3].set_ylabel("collision rate")
    axes[3].set_title("Collision rate")
    axes[3].set_ylim(-0.03, 1.03)

    has_jerk = bool(summary["jerk"].notna().all() and summary["smooth"].notna().all())
    if has_jerk:
        sns.scatterplot(
            data=summary, x="jerk", y="score", hue="series",
            palette=palette, s=58, ax=axes[4], legend=False,
            edgecolor="white", linewidth=0.6,
        )
        axes[4].set_xlabel("jerk")
        axes[4].set_ylabel("score")
        axes[4].set_title("Score vs jerk")
        axes[4].set_ylim(-0.03, 1.03)
    elif len(counts):
        sns.lineplot(
            data=counts, x="steps", y="collision_count", hue="series",
            palette=palette, estimator=None, linewidth=1.3, ax=axes[4], legend=False,
        )
        style_steps(axes[4])
        axes[4].set_ylabel("collision count")
        axes[4].set_title("Collision count")
    else:
        axes[4].set_axis_off()
        axes[4].set_title("No jerk or collision count")

    sns.scatterplot(
        data=summary, x="run", y="parameters", hue="series",
        palette=palette, s=58, ax=axes[5], legend=False,
        edgecolor="white", linewidth=0.6,
    )
    axes[5].set_yscale("log")
    axes[5].axhline(2_000_000, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
    axes[5].set_xlabel("sweep index")
    axes[5].set_ylabel("parameters")
    axes[5].set_title("Network parameters")

    draw_hypers(hyper_ax, summary, palette)
    for ax in list(axes) + [hyper_ax]:
        sns.despine(ax=ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def tag_phase(frame, phase):
    if frame is None or len(frame) == 0:
        return frame
    out = frame.copy()
    out["phase"] = phase
    out["series"] = [f"{phase}-{int(run):04d}" for run in out["run"]]
    return out


def series_palette(keys):
    """Set1 color per run. Phase is part of the key so through and bump differ."""
    ordered = list(dict.fromkeys(keys))
    colors = distinct_colors(len(ordered))
    return {key: colors[i] for i, key in enumerate(ordered)}


def draw_overview(returns, collisions, counts, summary, out):
    """Both physics phases. Return vs run order is colored by wall clock.

    Through runs come first, then bump. Other panels use Set1, one color per
    run. Circles and solid lines are through. Squares and dashed lines are bump.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    palette = series_palette(summary["series"])
    fig, axes, hyper_ax = figure_axes()
    lo = float(summary["wall_min"].min())
    hi = float(summary["wall_min"].max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    norm = plt.Normalize(lo, hi)
    markers = {"through": "o", "bump": "s"}
    for phase, marker in markers.items():
        part = summary[summary["phase"] == phase]
        if len(part) == 0:
            continue
        sns.scatterplot(
            data=part, x="idx", y="episode_return", hue="wall_min",
            hue_norm=norm, palette="viridis", marker=marker, s=58, ax=axes[0],
            legend=False, edgecolor="white", linewidth=0.6,
        )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[0], pad=0.02,
    )
    colorbar.set_label("wall clock (min)")
    bump_idx = summary.loc[summary["phase"] == "bump", "idx"]
    if len(bump_idx):
        axes[0].axvline(float(bump_idx.min()) - 0.5, color="0.4", linestyle=":", linewidth=0.9)
    axes[0].set_xscale("linear")
    axes[0].set_yscale("linear")
    axes[0].set_xlabel("run order")
    axes[0].set_ylabel("episode return")
    axes[0].set_title("Episode return vs sweep index")

    for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
        part = returns[returns["phase"] == phase] if len(returns) else returns
        if len(part) == 0:
            continue
        sns.lineplot(
            data=part, x="steps", y="episode_return", hue="series",
            palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
            ax=axes[1], legend=False,
        )
    axes[1].set_xscale("linear")
    axes[1].set_yscale("linear")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[1].ticklabel_format(axis="y", style="plain")
    axes[1].set_xlabel("steps")
    axes[1].set_ylabel("episode return")
    axes[1].set_title("Episode return vs steps")

    for phase, marker in markers.items():
        part = summary[summary["phase"] == phase]
        if len(part) == 0:
            continue
        sns.scatterplot(
            data=part, x="idx", y="score", hue="series",
            palette=palette, marker=marker, s=58, ax=axes[2], legend=False,
            edgecolor="white", linewidth=0.6,
        )
    axes[2].set_xlabel("run order")
    axes[2].set_ylabel("score")
    axes[2].set_title("Score")
    axes[2].set_ylim(-0.03, 1.03)

    for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
        part = collisions[collisions["phase"] == phase] if len(collisions) else collisions
        if len(part) == 0:
            continue
        sns.lineplot(
            data=part, x="steps", y="collision_rate", hue="series",
            palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
            ax=axes[3], legend=False,
        )
    style_steps(axes[3])
    axes[3].set_ylabel("collision rate")
    axes[3].set_title("Collision rate")
    axes[3].set_ylim(-0.03, 1.03)

    has_jerk = bool(summary["jerk"].notna().all() and summary["smooth"].notna().all())
    if has_jerk:
        for phase, marker in markers.items():
            part = summary[summary["phase"] == phase]
            if len(part) == 0:
                continue
            sns.scatterplot(
                data=part, x="jerk", y="score", hue="series",
                palette=palette, marker=marker, s=58, ax=axes[4], legend=False,
                edgecolor="white", linewidth=0.6,
            )
        axes[4].set_xlabel("jerk")
        axes[4].set_ylabel("score")
        axes[4].set_title("Score vs jerk")
        axes[4].set_ylim(-0.03, 1.03)
    elif len(counts):
        for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
            part = counts[counts["phase"] == phase]
            if len(part) == 0:
                continue
            sns.lineplot(
                data=part, x="steps", y="collision_count", hue="series",
                palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
                ax=axes[4], legend=False,
            )
        style_steps(axes[4])
        axes[4].set_ylabel("collision count")
        axes[4].set_title("Collision count")
    else:
        axes[4].set_axis_off()
        axes[4].set_title("No jerk or collision count")

    for phase, marker in markers.items():
        part = summary[summary["phase"] == phase]
        if len(part) == 0:
            continue
        sns.scatterplot(
            data=part, x="idx", y="parameters", hue="series",
            palette=palette, marker=marker, s=58, ax=axes[5], legend=False,
            edgecolor="white", linewidth=0.6,
        )
    axes[5].set_yscale("log")
    axes[5].axhline(2_000_000, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
    axes[5].set_xlabel("run order")
    axes[5].set_ylabel("parameters")
    axes[5].set_title("Network parameters")

    draw_hypers(hyper_ax, summary, palette)
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.2", linestyle="None", label="through"),
            Line2D([0], [0], marker="s", color="0.2", linestyle="None", label="bump"),
        ],
        loc="upper right", frameon=False,
    )
    for ax in list(axes) + [hyper_ax]:
        sns.despine(ax=ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def load_overview(phases):
    """phases is a list of (name, log_path, metrics_dir)."""
    returns = []
    collisions = []
    counts = []
    summaries = []
    missing = []
    for name, log_path, metrics_dir in phases:
        log_path = Path(log_path)
        if not log_path.exists() or not log_path.read_text().strip():
            continue
        try:
            part_returns, part_collisions, part_counts, part_summary, part_missing = load_frames(
                log_path, Path(metrics_dir),
            )
        except SystemExit:
            continue
        returns.append(tag_phase(part_returns, name))
        collisions.append(tag_phase(part_collisions, name))
        counts.append(tag_phase(part_counts, name))
        summaries.append(tag_phase(part_summary, name))
        missing.extend(f"{name}:{run}" for run in part_missing)
    summaries = [frame for frame in summaries if frame is not None and len(frame)]
    if not summaries:
        raise SystemExit("no finished runs to plot")
    summary = pd.concat(summaries, ignore_index=True)
    summary["idx"] = range(len(summary))
    returns = pd.concat(
        [frame for frame in returns if frame is not None and len(frame)],
        ignore_index=True,
    )
    collisions = pd.concat(
        [frame for frame in collisions if frame is not None and len(frame)],
        ignore_index=True,
    )
    count_frames = [frame for frame in counts if frame is not None and len(frame)]
    counts = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    return returns, collisions, counts, summary, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", nargs="?", type=Path, default=ROOT / "logs" / "drive" / "sweep.log",
        help="sweep log with 'sweep run=' lines",
    )
    parser.add_argument(
        "--phase", action="append", nargs=3, metavar=("NAME", "LOG", "METRICS"),
        help="add a through or bump phase: name, sweep log, metrics directory",
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=ROOT / "logs" / "drive",
        help="directory of per-run ini files",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=ROOT / "logs" / "drive" / "sweep-panels.png",
        help="output image under logs/drive",
    )
    args = parser.parse_args()
    out = args.out
    if out.parent != ROOT / "logs" / "drive":
        out = ROOT / "logs" / "drive" / out.name
    if args.phase:
        returns, collisions, counts, summary, missing = load_overview(args.phase)
        draw_overview(returns, collisions, counts, summary, out)
    else:
        returns, collisions, counts, summary, missing = load_frames(args.log, args.metrics_dir)
        draw(returns, collisions, counts, summary, out)
    print(f"wrote {out} runs={len(summary)}")
    if missing:
        print("no metrics for runs:", " ".join(str(run) for run in missing))


if __name__ == "__main__":
    main()
