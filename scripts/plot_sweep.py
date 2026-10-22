#!/usr/bin/env python3
"""Plot a protein sweep from its log and per-run metrics.

The figure is four rows. Score quality against sweep index is the first
panel, colored by run. Completion rate against jerk is next. Every panel
colors runs with ColorBrewer Set1, except episode return against sweep
index, which is colored by trial wall clock. The other panels are return
curves, goal score, collision rate, score against jerk when those series
exist (otherwise collision count), and parameter count. Under that, the
rate-like hyperparameters share one boxplot, each scaled to its own min
and max. Hidden size, layer count, horizon, minibatch size, agent count,
and timestep budget each have their own panel, in the units they were
swept, with one swarm point per run.

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
# Swept settings. gae_lambda is the GAE discount. The rate-like fields share
# a scaled axis. The count fields are drawn one panel each, unscaled.
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
SCALED_HYPERS = (
    "gamma",
    "gae_lambda",
    "learning_rate",
    "ent_coef",
    "replay_ratio",
    "clip_coef",
    "vf_coef",
    "vf_clip_coef",
    "momentum",
    "max_grad_norm",
)
SIZE_HYPERS = (
    "hidden_size",
    "num_layers",
    "minibatch_size",
    "total_agents",
    "horizon",
    "total_timesteps",
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
        quality_vals = metric_series(text, "env/score_quality")
        completion_vals = metric_series(text, "env/completion_rate")
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
            "score_quality": quality_vals[-1] if quality_vals else float("nan"),
            "completion_rate": (
                completion_vals[-1] if completion_vals else float("nan")
            ),
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
    """Result panels, one scaled coefficient row, then one panel per size."""
    fig = plt.figure(figsize=(22.4, 16.4), dpi=150, layout="constrained")
    outer = fig.add_gridspec(
        4, 1, height_ratios=[1.0, 1.0, 1.05, 1.15],
    )
    top = outer[0].subgridspec(1, 4, wspace=0.28)
    mid = outer[1].subgridspec(1, 4, wspace=0.28)
    axes = [fig.add_subplot(top[0, col]) for col in range(4)]
    axes += [fig.add_subplot(mid[0, col]) for col in range(4)]
    coef_ax = fig.add_subplot(outer[2])
    size_grid = outer[3].subgridspec(1, len(SIZE_HYPERS), wspace=0.42)
    size_axes = [
        fig.add_subplot(size_grid[0, i]) for i in range(len(SIZE_HYPERS))
    ]
    return fig, axes, coef_ax, size_axes


def draw_hypers(ax, summary, palette, fields):
    """Boxplot of rate-like hyperparameters, one swarm point per run.

    Values are scaled to the min and max of that hyperparameter in this figure.
    """
    rows = []
    for record in summary.to_dict("records"):
        for label in fields:
            value = record.get(label)
            if value is None or value != value:
                continue
            rows.append({
                "series": record["series"], "hyper": label, "value": float(value),
            })
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
    order = [label for label in fields if label in set(frame["hyper"])]
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
    ax.set_title("Coefficients")
    ax.set_ylim(-0.08, 1.08)
    ax.tick_params(axis="x", labelrotation=32)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def draw_size_panel(ax, summary, palette, label):
    """One size hyperparameter in the units it was swept, not rescaled."""
    if label not in summary.columns or not summary[label].notna().any():
        ax.set_axis_off()
        return
    part = summary.loc[summary[label].notna(), ["series", label]].copy()
    part = part.rename(columns={label: "value"})
    part["hyper"] = label
    sns.boxplot(
        data=part, x="hyper", y="value", ax=ax,
        color="#d0d0d0", fliersize=0, width=0.55, linewidth=0.8,
    )
    sns.swarmplot(
        data=part, x="hyper", y="value", hue="series",
        palette=palette, ax=ax, legend=False, size=4.0,
    )
    lo = float(part["value"].min())
    hi = float(part["value"].max())
    span = hi - lo
    pad = span * 0.22 if span > 0 else max(abs(lo) * 0.08, 0.5)
    ax.set_ylim(lo - pad, hi + pad)
    if max(abs(lo), abs(hi)) >= 1.0e4:
        # Tick text is the value itself. The default sci offset sits on the title.
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _pos: f"{value:.2e}")
        )
        ax.yaxis.get_offset_text().set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(label.replace("_", " "))
    ax.tick_params(axis="x", labelbottom=False)


def draw(returns, collisions, counts, summary, out):
    sns.set_theme(style="whitegrid", context="notebook")
    palette = categorical_palette(summary["run"])
    summary = with_series(summary)
    returns = with_series(returns)
    collisions = with_series(collisions)
    if len(counts):
        counts = with_series(counts)

    fig, axes, hyper_ax, size_axes = figure_axes()

    if summary["score_quality"].notna().any():
        sns.scatterplot(
            data=summary, x="run", y="score_quality", hue="series",
            palette=palette, s=58, ax=axes[0], legend=False,
            edgecolor="white", linewidth=0.6,
        )
        axes[0].set_ylim(-0.03, 1.03)
    else:
        axes[0].set_axis_off()
    axes[0].set_xlabel("sweep index")
    axes[0].set_ylabel("score quality")
    axes[0].set_title("Score quality")

    if summary["completion_rate"].notna().any() and summary["jerk"].notna().any():
        sns.scatterplot(
            data=summary, x="jerk", y="completion_rate", hue="series",
            palette=palette, s=58, ax=axes[1], legend=False,
            edgecolor="white", linewidth=0.6,
        )
        axes[1].set_ylim(-0.03, 1.03)
    else:
        axes[1].set_axis_off()
    axes[1].set_xlabel("jerk")
    axes[1].set_ylabel("completion rate")
    axes[1].set_title("Completion rate vs jerk")

    lo = float(summary["wall_min"].min())
    hi = float(summary["wall_min"].max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    norm = plt.Normalize(lo, hi)
    sns.scatterplot(
        data=summary, x="run", y="episode_return", hue="wall_min",
        hue_norm=norm, palette="viridis", s=58, ax=axes[2],
        legend=False, edgecolor="white", linewidth=0.6,
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[2], pad=0.02,
    )
    colorbar.set_label("wall clock (min)")
    axes[2].set_xscale("linear")
    axes[2].set_yscale("linear")
    axes[2].set_xlabel("sweep index")
    axes[2].set_ylabel("episode return")
    axes[2].set_title("Episode return vs sweep index")

    sns.lineplot(
        data=returns, x="steps", y="episode_return", hue="series",
        palette=palette, estimator=None, linewidth=1.3, ax=axes[3], legend=False,
    )
    axes[3].set_xscale("linear")
    axes[3].set_yscale("linear")
    axes[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[3].ticklabel_format(axis="y", style="plain")
    axes[3].set_xlabel("steps")
    axes[3].set_ylabel("episode return")
    axes[3].set_title("Episode return vs steps")

    sns.scatterplot(
        data=summary, x="run", y="score", hue="series",
        palette=palette, s=58, ax=axes[4], legend=False,
        edgecolor="white", linewidth=0.6,
    )
    axes[4].set_xlabel("sweep index")
    axes[4].set_ylabel("score")
    axes[4].set_title("Score")
    axes[4].set_ylim(-0.03, 1.03)

    sns.lineplot(
        data=collisions, x="steps", y="collision_rate", hue="series",
        palette=palette, estimator=None, linewidth=1.3, ax=axes[5], legend=False,
    )
    style_steps(axes[5])
    axes[5].set_ylabel("collision rate")
    axes[5].set_title("Collision rate")
    axes[5].set_ylim(-0.03, 1.03)

    has_jerk = bool(summary["jerk"].notna().all() and summary["smooth"].notna().all())
    if has_jerk:
        sns.scatterplot(
            data=summary, x="jerk", y="score", hue="series",
            palette=palette, s=58, ax=axes[6], legend=False,
            edgecolor="white", linewidth=0.6,
        )
        axes[6].set_xlabel("jerk")
        axes[6].set_ylabel("score")
        axes[6].set_title("Score vs jerk")
        axes[6].set_ylim(-0.03, 1.03)
    elif len(counts):
        sns.lineplot(
            data=counts, x="steps", y="collision_count", hue="series",
            palette=palette, estimator=None, linewidth=1.3, ax=axes[6], legend=False,
        )
        style_steps(axes[6])
        axes[6].set_ylabel("collision count")
        axes[6].set_title("Collision count")
    else:
        axes[6].set_axis_off()
        axes[6].set_title("No jerk or collision count")

    sns.scatterplot(
        data=summary, x="run", y="parameters", hue="series",
        palette=palette, s=58, ax=axes[7], legend=False,
        edgecolor="white", linewidth=0.6,
    )
    axes[7].set_yscale("log")
    axes[7].axhline(2_000_000, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
    axes[7].set_xlabel("sweep index")
    axes[7].set_ylabel("parameters")
    axes[7].set_title("Network parameters")

    draw_hypers(hyper_ax, summary, palette, SCALED_HYPERS)
    for ax, label in zip(size_axes, SIZE_HYPERS):
        draw_size_panel(ax, summary, palette, label)
    for ax in list(axes) + [hyper_ax] + list(size_axes):
        sns.despine(ax=ax)
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


def scatter_phases(summary, x, y, ax, palette, markers):
    for phase, marker in markers.items():
        part = summary[summary["phase"] == phase]
        if len(part) == 0:
            continue
        sns.scatterplot(
            data=part, x=x, y=y, hue="series",
            palette=palette, marker=marker, s=58, ax=ax, legend=False,
            edgecolor="white", linewidth=0.6,
        )


def draw_overview(returns, collisions, counts, summary, out):
    """Both physics phases. Score quality is colored by sweep index.

    Through runs come first, then bump. Panels use Set1, one color per run,
    except episode return against run order, which is colored by wall clock.
    Circles and solid lines are through. Squares and dashed lines are bump.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    palette = series_palette(summary["series"])
    fig, axes, hyper_ax, size_axes = figure_axes()
    markers = {"through": "o", "bump": "s"}

    if summary["score_quality"].notna().any():
        scatter_phases(summary, "idx", "score_quality", axes[0], palette, markers)
        axes[0].set_ylim(-0.03, 1.03)
    else:
        axes[0].set_axis_off()
    axes[0].set_xlabel("run order")
    axes[0].set_ylabel("score quality")
    axes[0].set_title("Score quality")

    if summary["completion_rate"].notna().any() and summary["jerk"].notna().any():
        scatter_phases(summary, "jerk", "completion_rate", axes[1], palette, markers)
        axes[1].set_ylim(-0.03, 1.03)
    else:
        axes[1].set_axis_off()
    axes[1].set_xlabel("jerk")
    axes[1].set_ylabel("completion rate")
    axes[1].set_title("Completion rate vs jerk")

    lo = float(summary["wall_min"].min())
    hi = float(summary["wall_min"].max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    norm = plt.Normalize(lo, hi)
    for phase, marker in markers.items():
        part = summary[summary["phase"] == phase]
        if len(part) == 0:
            continue
        sns.scatterplot(
            data=part, x="idx", y="episode_return", hue="wall_min",
            hue_norm=norm, palette="viridis", marker=marker, s=58, ax=axes[2],
            legend=False, edgecolor="white", linewidth=0.6,
        )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[2], pad=0.02,
    )
    colorbar.set_label("wall clock (min)")
    bump_idx = summary.loc[summary["phase"] == "bump", "idx"]
    if len(bump_idx):
        axes[2].axvline(float(bump_idx.min()) - 0.5, color="0.4", linestyle=":", linewidth=0.9)
    axes[2].set_xscale("linear")
    axes[2].set_yscale("linear")
    axes[2].set_xlabel("run order")
    axes[2].set_ylabel("episode return")
    axes[2].set_title("Episode return vs sweep index")

    for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
        part = returns[returns["phase"] == phase] if len(returns) else returns
        if len(part) == 0:
            continue
        sns.lineplot(
            data=part, x="steps", y="episode_return", hue="series",
            palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
            ax=axes[3], legend=False,
        )
    axes[3].set_xscale("linear")
    axes[3].set_yscale("linear")
    axes[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[3].ticklabel_format(axis="y", style="plain")
    axes[3].set_xlabel("steps")
    axes[3].set_ylabel("episode return")
    axes[3].set_title("Episode return vs steps")

    scatter_phases(summary, "idx", "score", axes[4], palette, markers)
    axes[4].set_xlabel("run order")
    axes[4].set_ylabel("score")
    axes[4].set_title("Score")
    axes[4].set_ylim(-0.03, 1.03)

    for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
        part = collisions[collisions["phase"] == phase] if len(collisions) else collisions
        if len(part) == 0:
            continue
        sns.lineplot(
            data=part, x="steps", y="collision_rate", hue="series",
            palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
            ax=axes[5], legend=False,
        )
    style_steps(axes[5])
    axes[5].set_ylabel("collision rate")
    axes[5].set_title("Collision rate")
    axes[5].set_ylim(-0.03, 1.03)

    has_jerk = bool(summary["jerk"].notna().all() and summary["smooth"].notna().all())
    if has_jerk:
        scatter_phases(summary, "jerk", "score", axes[6], palette, markers)
        axes[6].set_xlabel("jerk")
        axes[6].set_ylabel("score")
        axes[6].set_title("Score vs jerk")
        axes[6].set_ylim(-0.03, 1.03)
    elif len(counts):
        for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
            part = counts[counts["phase"] == phase]
            if len(part) == 0:
                continue
            sns.lineplot(
                data=part, x="steps", y="collision_count", hue="series",
                palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
                ax=axes[6], legend=False,
            )
        style_steps(axes[6])
        axes[6].set_ylabel("collision count")
        axes[6].set_title("Collision count")
    else:
        axes[6].set_axis_off()
        axes[6].set_title("No jerk or collision count")

    scatter_phases(summary, "idx", "parameters", axes[7], palette, markers)
    axes[7].set_yscale("log")
    axes[7].axhline(2_000_000, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
    axes[7].set_xlabel("run order")
    axes[7].set_ylabel("parameters")
    axes[7].set_title("Network parameters")

    draw_hypers(hyper_ax, summary, palette, SCALED_HYPERS)
    for ax, label in zip(size_axes, SIZE_HYPERS):
        draw_size_panel(ax, summary, palette, label)
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.2", linestyle="None", label="through"),
            Line2D([0], [0], marker="s", color="0.2", linestyle="None", label="bump"),
        ],
        loc="upper right", frameon=False,
    )
    for ax in list(axes) + [hyper_ax] + list(size_axes):
        sns.despine(ax=ax)
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
