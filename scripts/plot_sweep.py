#!/usr/bin/env python3
"""Plot a protein sweep from its log and per-run metrics.

The figure is six rows. The sweep objective against sweep index is the
first panel, colored by run. Episode return against sweep index is next,
colored by trial wall clock, then the return curve. Every other panel
uses the default seaborn palette. The best sweep objective is green on
every panel and a star on each scatter. An ini with sweep.goal = minimize
picks the lowest objective; otherwise the highest wins. The next row is anchor loss over training steps,
score against jerk when those series exist (otherwise meters per
infraction), and parameter count.
The next row is final meters per infraction against score, completion
rate against score, then meters driven before the first vehicle overlap
or off-road (env/distance_before_first_failure) over training steps.
Older inis that still say env/distance_before_failure are read as the
same series.
A panel stays blank when the run ini has no such series. The next row is
training score against anchor coefficient beside town score against
centerline density at 128 cars. Each panel is the same width as a top-row
subplot. Density is the share of lane centerline taken by car length, from
config/town_density.json. Every run is one point on each town, in one
swarm so points that share a density do not stack. Towns are ordered by
that density. A town is skipped when that run has no score for it.
Under that,
the rate-like hyperparameters share one boxplot, each scaled to its own
min and max. Hidden size, layer count, horizon, minibatch size, and
timestep budget each have their own panel, in the units they
were swept, with one swarm point per run.

A benchmark report can still name the sweep objective. That report is a
[benchmark] section in the run ini, or logs/drive/<run_id>.benchmark.txt
written by ./drive --benchmark.

    python scripts/plot_sweep.py logs/drive/sweep.log -o logs/drive/sweep-panels.png

Wall clock is the trial cost in the sweep log, in minutes. Goal score and
collision rate come from the binned env/ metrics written at the end of each
run. Runs that failed before writing metrics are left out. Figures go under
logs/drive, or logs/replay_sweep when that directory is the output.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.transforms import ScaledTranslation
import pandas as pd
import seaborn as sns

# Seaborn's deep green. Other runs skip this color so only the best is green.
BEST_COLOR = "#55A868"
BEST_STAR_SIZE = 220

ROOT = Path(__file__).resolve().parents[1]
RUN_LINE = re.compile(
    r"sweep run=(\d+) score=([0-9.eE+-]+) cost=([0-9.eE+-]+)"
)
# Names written by ./drive --benchmark. Anything else is shown as-is.
OBJECTIVE_LABELS = {
    "score": "score",
    "replay_score": "replay score",
    "replay_collision": "replay collision",
    "replay_offroad": "replay off-road",
    "replay_completion": "replay completion",
    "replay_ade_lat": "lateral ADE",
    "replay_ade_lon": "longitudinal ADE",
    "horizon_collision": "horizon collisions",
    "horizon_score": "steps before collision",
}
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
    "horizon",
    "total_timesteps",
)
BENCH_KEYS = (
    "replay_collision",
    "replay_score",
    "horizon_score",
)
# Final env/score on each CARLA map. Town10HD is score_town_10.
TOWN_FIELDS = (
    ("score_town_01", "Town01"),
    ("score_town_02", "Town02"),
    ("score_town_06", "Town06"),
    ("score_town_10", "Town10HD"),
)


def parse_sweep_log(path):
    """Finished trials: run index, protein objective, wall-clock minutes."""
    runs = []
    for line in path.read_text().splitlines():
        match = RUN_LINE.search(line)
        if match:
            runs.append((
                int(match.group(1)),
                float(match.group(2)),
                float(match.group(3)) / 60.0,
            ))
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


def section_text(text, section, key):
    """Raw ini value. Comments and list-looking lines are skipped."""
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
            return value.strip().strip("'\"")
    return None


def parse_report(path):
    """Metric floats, plus the `sweep <name> <value>` objective when present."""
    out = {}
    sweep_name = None
    for line in path.read_text(errors="replace").splitlines():
        bits = line.split()
        if len(bits) >= 3 and bits[0] == "sweep":
            try:
                out["sweep"] = float(bits[2])
            except ValueError:
                continue
            sweep_name = bits[1]
            continue
        if len(bits) < 2:
            continue
        try:
            out[bits[0]] = float(bits[1])
        except ValueError:
            continue
    return out, sweep_name


def benchmark_values(ini_path, text):
    """Replay metrics, and the benchmark objective name when a report has one."""
    found = {}
    sweep_name = None
    for key in BENCH_KEYS:
        value = section_value(text, "benchmark", key)
        if value is not None:
            found[key] = value
    names = (
        ini_path.with_suffix(".benchmark.txt"),
        ROOT / "logs" / "drive" / f"{ini_path.stem}.benchmark.txt",
    )
    for path in names:
        if not path.is_file():
            continue
        parsed, name = parse_report(path)
        if name and sweep_name is None:
            sweep_name = name
        if "sweep" in parsed and "sweep_value" not in found:
            found["sweep_value"] = parsed["sweep"]
        for key in BENCH_KEYS:
            if key not in found and key in parsed:
                found[key] = parsed[key]
    return found, sweep_name


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
    losses = []
    counts = []
    failures = []
    summary = []
    missing = []
    for run, objective, wall_min in parse_sweep_log(log_path):
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
        completion_vals = metric_series(text, "env/completion_rate")
        count_vals = metric_series(text, "env/collision_count")
        gap_vals = metric_series(text, "env/avg_distance_per_infraction")
        loss_vals = metric_series(text, "loss/anchor")
        distance_vals = metric_series(text, "env/distance_before_first_failure")
        if not distance_vals:
            distance_vals = metric_series(text, "env/distance_before_failure")
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
        if loss_vals and len(loss_vals) == len(steps):
            for step, value in zip(steps, loss_vals):
                losses.append({
                    "run": run, "steps": step, "anchor_loss": value,
                })
        if gap_vals and len(gap_vals) == len(steps):
            for step, value in zip(steps, gap_vals):
                counts.append({
                    "run": run,
                    "steps": step,
                    "avg_distance_per_infraction": value,
                })
        if distance_vals and len(distance_vals) == len(steps):
            for step, value in zip(steps, distance_vals):
                failures.append({
                    "run": run,
                    "steps": step,
                    "distance_before_first_failure": value,
                })
        summary.append({
            "run": run,
            "score": score_vals[-1],
            "objective": objective,
            "objective_name": "score",
            "completion_rate": (
                completion_vals[-1] if completion_vals else float("nan")
            ),
            "episode_return": return_vals[-1],
            "jerk": jerk_vals[-1] if jerk_vals else float("nan"),
            "smooth": smooth_vals[-1] if smooth_vals else float("nan"),
            "wall_min": wall_min,
            "collision_count": (
                count_vals[-1] if count_vals and len(count_vals) == len(steps) else float("nan")
            ),
            "distance_before_first_failure": (
                distance_vals[-1]
                if distance_vals and len(distance_vals) == len(steps)
                else float("nan")
            ),
            "avg_distance_per_infraction": (
                gap_vals[-1]
                if gap_vals and len(gap_vals) == len(steps)
                else float("nan")
            ),
            "parameters": policy_params(hidden, layers),
        })
        for key, _title in TOWN_FIELDS:
            vals = metric_series(text, "env/" + key)
            summary[-1][key] = vals[-1] if vals else float("nan")
        for label, section, key in HYPER_FIELDS:
            value = section_value(text, section, key)
            summary[-1][label] = float(value) if value is not None else float("nan")
        bench, sweep_name = benchmark_values(path, text)
        for key in BENCH_KEYS:
            summary[-1][key] = bench.get(key, float("nan"))
        if sweep_name:
            summary[-1]["objective_name"] = sweep_name
            if "sweep_value" in bench:
                summary[-1]["objective"] = bench["sweep_value"]
        else:
            train_metric = section_text(text, "sweep", "metric")
            if train_metric:
                summary[-1]["objective_name"] = train_metric
        coef = section_value(text, "train", "anchor_coef")
        summary[-1]["anchor_coef"] = (
            float(coef) if coef is not None else float("nan")
        )
        goal = section_text(text, "sweep", "goal")
        summary[-1]["goal"] = goal if goal in ("maximize", "minimize") else "maximize"
    if not summary:
        raise SystemExit(f"no metrics for runs in {log_path}")
    return (
        pd.DataFrame(returns),
        pd.DataFrame(losses),
        pd.DataFrame(counts),
        pd.DataFrame(failures),
        pd.DataFrame(summary),
        missing,
    )


def series_key(run):
    return f"{int(run):04d}"


def apply_figure_style():
    sns.set_style("ticks")
    sns.set_palette("deep")


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def same_color(color, target):
    try:
        return mcolors.to_rgb(color) == mcolors.to_rgb(target)
    except (ValueError, TypeError):
        return False


def distinct_colors(count):
    """Default seaborn palette without its green. Extra runs repeat."""
    base = [
        tuple(color) for color in sns.color_palette("deep")
        if not same_color(color, BEST_COLOR)
    ]
    if count <= 0 or not base:
        return []
    return [base[i % len(base)] for i in range(count)]


def assign_palette(ordered, best):
    """One seaborn color per series. The best series is green."""
    best = set(best or ())
    others = [key for key in ordered if key not in best]
    colors = distinct_colors(len(others))
    palette = {key: colors[i] for i, key in enumerate(others)}
    green = mcolors.to_rgb(BEST_COLOR)
    for key in ordered:
        if key in best:
            palette[key] = green
    return palette


def categorical_palette(run_ids, best=()):
    """One seaborn color per run. The same run keeps that color on every panel."""
    ordered = list(dict.fromkeys(series_key(run) for run in run_ids))
    return assign_palette(ordered, best)


def best_series(summary):
    """Series keys that hold the best sweep objective."""
    if summary is None or len(summary) == 0 or "objective" not in summary.columns:
        return set()
    if "series" not in summary.columns:
        return set()
    finite = summary.dropna(subset=["objective"])
    if finite.empty:
        return set()
    if "goal" in finite.columns:
        minimize = bool(finite["goal"].map(lambda goal: goal == "minimize").all())
    else:
        minimize = False
    target = float(finite["objective"].min() if minimize else finite["objective"].max())
    close = (finite["objective"] - target).abs() <= 1e-8
    return set(finite.loc[close, "series"])


def split_best(summary, best):
    best = set(best or ())
    empty = summary.iloc[0:0]
    if not best or summary is None or len(summary) == 0 or "series" not in summary.columns:
        return empty, summary
    stars = summary[summary["series"].isin(best)]
    rest = summary[~summary["series"].isin(best)]
    return stars, rest


def scatter_stars(frame, x, y, ax):
    """Best run on a scatter panel."""
    if frame is None or len(frame) == 0:
        return
    plotted = frame.dropna(subset=[x, y])
    if plotted.empty:
        return
    ax.scatter(
        plotted[x], plotted[y],
        marker="*", s=BEST_STAR_SIZE, c=[mcolors.to_rgb(BEST_COLOR)],
        edgecolors="0.1", linewidths=0.45, zorder=6,
    )


def lift_best(ax):
    """Keep the green series above the other runs, and heavier so it stays visible."""
    for line in ax.lines:
        if same_color(line.get_color(), BEST_COLOR):
            line.set_zorder(5)
            line.set_linewidth(2.4)
    for coll in ax.collections:
        facecolors = coll.get_facecolors()
        if len(facecolors) == 0:
            continue
        if all(same_color(color, BEST_COLOR) for color in facecolors):
            coll.set_zorder(6)


def with_series(frame):
    out = frame.copy()
    out["series"] = out["run"].map(series_key)
    return out


def style_steps(ax):
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_xlabel("steps")


def fit_score_axis(ax, values):
    """Unit window while score is a rate. Counts may rise above 1."""
    finite = values.dropna()
    if len(finite) == 0 or float(finite.max()) <= 1.0:
        ax.set_ylim(-0.03, 1.03)
        return
    hi = float(finite.max())
    pad = max(hi * 0.08, 0.5)
    ax.set_ylim(-0.03, hi + pad)


def fit_failure_axis(ax, values):
    """Steps before the first collision or off-road. Zero is a first-step failure."""
    finite = values.dropna()
    if len(finite) == 0:
        return
    hi = float(finite.max())
    pad = max(hi * 0.08, 1.0)
    ax.set_ylim(-0.03 * pad, hi + pad)


def blank_panel(ax, title, message):
    ax.set_axis_off()
    ax.set_title(title)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", transform=ax.transAxes, color="0.45",
    )


def figure_axes():
    """Result panels, failure row, anchor and density, coefficients, sizes."""
    fig = plt.figure(figsize=(22.4, 24.5), dpi=100, layout="constrained")
    outer = fig.add_gridspec(
        6, 1, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.05, 1.15],
    )
    top = outer[0].subgridspec(1, 3, wspace=0.28)
    mid = outer[1].subgridspec(1, 3, wspace=0.28)
    axes = [fig.add_subplot(top[0, col]) for col in range(3)]
    axes += [fig.add_subplot(mid[0, col]) for col in range(3)]
    failure_grid = outer[2].subgridspec(1, 3, wspace=0.28)
    failure_axes = [fig.add_subplot(failure_grid[0, col]) for col in range(3)]
    pair = outer[3].subgridspec(1, 3, wspace=0.28)
    anchor_ax = fig.add_subplot(pair[0, 0])
    density_ax = fig.add_subplot(pair[0, 1])
    coef_ax = fig.add_subplot(outer[4])
    size_grid = outer[5].subgridspec(1, len(SIZE_HYPERS), wspace=0.42)
    size_axes = [
        fig.add_subplot(size_grid[0, i]) for i in range(len(SIZE_HYPERS))
    ]
    return fig, axes, failure_axes, anchor_ax, density_ax, coef_ax, size_axes


def scatter_metric(summary, x, y, ax, palette, markers=None, best=()):
    stars, rest = split_best(summary, best)
    if len(rest):
        if markers is None:
            sns.scatterplot(
                data=rest, x=x, y=y, hue="series",
                palette=palette, s=58, ax=ax, legend=False,
                edgecolor="white", linewidth=0.6,
            )
        else:
            scatter_phases(rest, x, y, ax, palette, markers)
    scatter_stars(stars, x, y, ax)


def draw_score_vs_anchor(ax, summary, palette, markers=None, best=()):
    """Training score against the anchor coefficient."""
    title = "Score vs anchor coef"
    present = (
        "score" in summary.columns
        and "anchor_coef" in summary.columns
        and bool(summary["score"].notna().any())
        and bool(summary["anchor_coef"].notna().any())
    )
    if not present:
        ax.set_axis_off()
        ax.set_title(title)
        ax.text(
            0.5, 0.5, "no anchor coef",
            ha="center", va="center", transform=ax.transAxes, color="0.45",
        )
        return
    scatter_metric(summary, "anchor_coef", "score", ax, palette, markers, best)
    ax.set_xscale("linear")
    fit_score_axis(ax, summary["score"])
    ax.set_xlabel("anchor coef")
    ax.set_ylabel("score")
    ax.set_title(title)


def draw_gap_vs_score(ax, summary, palette, markers=None, best=()):
    """Final meters per infraction against the training score."""
    title = "Distance per infraction (m) vs score"
    key = "avg_distance_per_infraction"
    present = (
        key in summary.columns
        and "score" in summary.columns
        and bool(summary[key].notna().any())
        and bool(summary["score"].notna().any())
    )
    if not present:
        blank_panel(ax, title, "no distance per infraction")
        return
    scatter_metric(summary, "score", key, ax, palette, markers, best)
    scores = summary["score"].dropna()
    if len(scores) and float(scores.max()) <= 1.0:
        ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("score")
    ax.set_ylabel("m / infraction")
    ax.set_title(title)
    fit_failure_axis(ax, summary[key])


def draw_completion_vs_score(ax, summary, palette, markers=None, best=()):
    """Final completion rate against the training score."""
    title = "Completion rate vs score"
    key = "completion_rate"
    present = (
        key in summary.columns
        and "score" in summary.columns
        and bool(summary[key].notna().any())
        and bool(summary["score"].notna().any())
    )
    if not present:
        blank_panel(ax, title, "no completion rate")
        return
    scatter_metric(summary, "score", key, ax, palette, markers, best)
    scores = summary["score"].dropna()
    if len(scores) and float(scores.max()) <= 1.0:
        ax.set_xlim(-0.03, 1.03)
    rates = summary[key].dropna()
    if len(rates) and float(rates.max()) <= 1.0:
        ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("score")
    ax.set_ylabel("completion rate")
    ax.set_title(title)


def draw_failure_metric(scatter_ax, curve_ax, summary, curves, key, ylabel,
        title, message, x, palette, markers=None, draw_scatter=True, best=()):
    """Final value against the sweep, then the same series over training."""
    xlabel = "sweep index" if x == "run" else "run order"
    present = key in summary.columns and bool(summary[key].notna().any())
    if draw_scatter and not present:
        blank_panel(scatter_ax, title, message)
    elif draw_scatter:
        scatter_metric(summary, x, key, scatter_ax, palette, markers, best)
        scatter_ax.set_xlabel(xlabel)
        scatter_ax.set_ylabel(ylabel)
        scatter_ax.set_title(title)
        fit_failure_axis(scatter_ax, summary[key])
    has_curve = (
        curves is not None and len(curves) > 0 and key in curves.columns
        and bool(curves[key].notna().any())
    )
    if not has_curve:
        blank_panel(curve_ax, title, message)
        return
    plotted = curves.dropna(subset=[key])
    if markers is None:
        sns.lineplot(
            data=plotted, x="steps", y=key, hue="series",
            palette=palette, estimator=None, linewidth=1.3,
            ax=curve_ax, legend=False,
        )
    else:
        for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
            part = plotted[plotted["phase"] == phase]
            if len(part) == 0:
                continue
            sns.lineplot(
                data=part, x="steps", y=key, hue="series",
                palette=palette, estimator=None, linewidth=1.3,
                linestyle=linestyle, ax=curve_ax, legend=False,
            )
    style_steps(curve_ax)
    curve_ax.set_ylabel(ylabel)
    curve_ax.set_title(title)
    fit_failure_axis(curve_ax, plotted[key])


def draw_failure(axes, summary, curves, palette, x, markers=None, best=()):
    """Meters per infraction and completion against score, then meters before failure."""
    draw_gap_vs_score(axes[0], summary, palette, markers, best)
    draw_completion_vs_score(axes[1], summary, palette, markers, best)
    draw_failure_metric(
        axes[0], axes[2], summary, curves,
        "distance_before_first_failure", "meters before first failure",
        "Distance before first failure", "no distance before first failure",
        x, palette, markers, draw_scatter=False, best=best,
    )


def load_town_density():
    """128-car centerline density for each town, from config/town_density.json."""
    path = ROOT / "config" / "town_density.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    if int(payload.get("agents", 0)) != 128:
        return []
    score_key = {title: key for key, title in TOWN_FIELDS}
    rows = []
    for row in payload.get("towns", []):
        if int(row.get("agents", 0)) != 128:
            continue
        town = row.get("town")
        if town not in score_key or "centerline_pct" not in row:
            continue
        rows.append({
            "town": town,
            "score_key": score_key[town],
            "density": float(row["centerline_pct"]),
        })
    return rows


def draw_town_density(ax, summary, palette):
    """Town score against centerline coverage at 128 cars.

    An unfilled violin sits behind each swarm. A dark line marks the mean.
    """
    title = "Performance across towns, sorted by density"
    towns = load_town_density()
    if not towns:
        blank_panel(ax, title, "no town density")
        return
    rows = []
    for record in summary.to_dict("records"):
        for town in towns:
            score = record.get(town["score_key"])
            if score is None or score != score:
                continue
            rows.append({
                "series": record["series"],
                "town": town["town"],
                "density": town["density"],
                "score": float(score),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        blank_panel(ax, title, "no town score")
        return
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
    sns.swarmplot(
        data=frame, x="density", y="score", hue="series",
        palette=palette, dodge=False, native_scale=True,
        size=3, ax=ax, legend=False,
        edgecolor="white", linewidth=0.3, zorder=3,
    )
    half = 0.32 * span
    for density, scores in frame.groupby("density")["score"]:
        mean = float(scores.mean())
        ax.plot(
            [float(density) - half, float(density) + half],
            [mean, mean],
            color="0.1", linewidth=1.8, solid_capstyle="butt", zorder=4,
        )
    pad = min(gaps) * 0.75 if gaps else densities[0] * 0.2
    ax.set_xlim(densities[0] - pad, densities[-1] + pad)
    ax.set_autoscalex_on(False)
    ax.set_xticks(densities)
    ax.set_xticklabels([
        f"{town}\n{density:.2f}"
        for town, density in zip(placed["town"], densities)
    ])
    ax.tick_params(axis="x", labelsize=8)
    labels = ax.get_xticklabels()
    for i in range(1, len(densities)):
        if densities[i] - densities[i - 1] >= 1.5:
            continue
        shift = ScaledTranslation(0, -26 / 72, ax.figure.dpi_scale_trans)
        labels[i].set_transform(labels[i].get_transform() + shift)
    fit_score_axis(ax, frame["score"])
    ax.set_xlabel("% of centerline")
    ax.set_ylabel("score")
    ax.set_title(title)


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


def objective_label(names):
    """One ylabel when every trial optimized the same metric."""
    cleaned = []
    for name in names:
        if name is None or name != name:
            continue
        text = str(name).strip()
        if text:
            cleaned.append(text)
    if len(set(cleaned)) != 1:
        return "objective"
    name = cleaned[0]
    if name in OBJECTIVE_LABELS:
        return OBJECTIVE_LABELS[name]
    return name.replace("_", " ")


def draw_objective(ax, summary, palette, x, markers=None, best=()):
    """Protein's trial score. The benchmark report names it when one exists."""
    title = "Sweep objective"
    present = (
        "objective" in summary.columns
        and bool(summary["objective"].notna().any())
    )
    if not present:
        ax.set_axis_off()
        ax.set_title(title)
        return
    scatter_metric(summary, x, "objective", ax, palette, markers, best)
    values = summary["objective"].dropna()
    if len(values) and bool(((values >= 0.0) & (values <= 1.0)).all()):
        ax.set_ylim(-0.03, 1.03)
    xlabel = "sweep index" if x == "run" else "run order"
    names = []
    if "objective_name" in summary.columns:
        names = summary["objective_name"].tolist()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(objective_label(names))
    ax.set_title(title)


def fit_loss_axis(ax, values):
    """Pad the observed anchor-loss range. Zero is not a meaningful floor."""
    finite = values.dropna()
    if len(finite) == 0:
        return
    lo = float(finite.min())
    hi = float(finite.max())
    span = hi - lo
    pad = span * 0.08 if span > 0 else max(abs(hi) * 0.08, 0.1)
    ax.set_ylim(lo - pad, hi + pad)


def draw_anchor_loss(ax, losses, palette, markers=None):
    """Anchor loss over training steps. Blank when the ini has no series."""
    title = "Anchor loss"
    column = "anchor_loss"
    has_curve = (
        losses is not None and len(losses) > 0
        and column in losses.columns
        and bool(losses[column].notna().any())
    )
    if not has_curve:
        blank_panel(ax, title, "no anchor loss")
        return
    plotted = losses.dropna(subset=[column])
    if markers is None:
        sns.lineplot(
            data=plotted, x="steps", y=column, hue="series",
            palette=palette, estimator=None, linewidth=1.3, ax=ax, legend=False,
        )
    else:
        for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
            part = plotted[plotted["phase"] == phase]
            if len(part) == 0:
                continue
            sns.lineplot(
                data=part, x="steps", y=column, hue="series",
                palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
                ax=ax, legend=False,
            )
    style_steps(ax)
    ax.set_ylabel("anchor loss")
    ax.set_title(title)
    fit_loss_axis(ax, plotted[column])


def draw_distance_per_infraction(ax, counts, palette, markers=None):
    """Meters driven per infraction step. Blank when the series is absent."""
    title = "Distance per infraction (m)"
    column = "avg_distance_per_infraction"
    has_curve = (
        counts is not None and len(counts) > 0
        and column in counts.columns
        and bool(counts[column].notna().any())
    )
    if not has_curve:
        blank_panel(ax, title, "no distance per infraction")
        return
    plotted = counts.dropna(subset=[column])
    if markers is None:
        sns.lineplot(
            data=plotted, x="steps", y=column, hue="series",
            palette=palette, estimator=None, linewidth=1.3, ax=ax, legend=False,
        )
    else:
        for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
            part = plotted[plotted["phase"] == phase]
            if len(part) == 0:
                continue
            sns.lineplot(
                data=part, x="steps", y=column, hue="series",
                palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
                ax=ax, legend=False,
            )
    style_steps(ax)
    ax.set_ylabel("m / infraction")
    ax.set_title(title)
    fit_failure_axis(ax, plotted[column])


def scatter_wallclock(summary, x, y, ax, norm, markers=None, best=()):
    """Episode return colored by wall clock. The best run is still a green star."""
    stars, rest = split_best(summary, best)
    if len(rest):
        if markers is None:
            sns.scatterplot(
                data=rest, x=x, y=y, hue="wall_min",
                hue_norm=norm, palette="viridis", s=58, ax=ax,
                legend=False, edgecolor="white", linewidth=0.6,
            )
        else:
            for phase, marker in markers.items():
                part = rest[rest["phase"] == phase]
                if len(part) == 0:
                    continue
                sns.scatterplot(
                    data=part, x=x, y=y, hue="wall_min",
                    hue_norm=norm, palette="viridis", marker=marker, s=58, ax=ax,
                    legend=False, edgecolor="white", linewidth=0.6,
                )
    scatter_stars(stars, x, y, ax)


def draw(returns, losses, counts, failures, summary, out):
    apply_figure_style()
    summary = with_series(summary)
    returns = with_series(returns)
    if len(losses):
        losses = with_series(losses)
    if len(counts):
        counts = with_series(counts)
    if len(failures):
        failures = with_series(failures)
    best = best_series(summary)
    palette = categorical_palette(summary["run"], best)

    fig, axes, failure_axes, anchor_ax, density_ax, hyper_ax, size_axes = figure_axes()

    draw_objective(axes[0], summary, palette, "run", best=best)

    lo = float(summary["wall_min"].min())
    hi = float(summary["wall_min"].max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    norm = plt.Normalize(lo, hi)
    scatter_wallclock(summary, "run", "episode_return", axes[1], norm, best=best)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[1], pad=0.02,
    )
    colorbar.set_label("wall clock (min)")
    axes[1].set_xscale("linear")
    axes[1].set_yscale("linear")
    axes[1].set_xlabel("sweep index")
    axes[1].set_ylabel("episode return")
    axes[1].set_title("Episode return vs sweep index")

    sns.lineplot(
        data=returns, x="steps", y="episode_return", hue="series",
        palette=palette, estimator=None, linewidth=1.3, ax=axes[2], legend=False,
    )
    axes[2].set_xscale("linear")
    axes[2].set_yscale("linear")
    axes[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[2].ticklabel_format(axis="y", style="plain")
    axes[2].set_xlabel("steps")
    axes[2].set_ylabel("episode return")
    axes[2].set_title("Episode return vs steps")

    draw_anchor_loss(axes[3], losses, palette)

    has_jerk = bool(summary["jerk"].notna().all() and summary["smooth"].notna().all())
    if has_jerk:
        scatter_metric(summary, "jerk", "score", axes[4], palette, best=best)
        axes[4].set_xlabel("jerk")
        axes[4].set_ylabel("score")
        axes[4].set_title("Score vs jerk")
        axes[4].set_ylim(-0.03, 1.03)
    else:
        draw_distance_per_infraction(axes[4], counts, palette)

    scatter_metric(summary, "run", "parameters", axes[5], palette, best=best)
    axes[5].set_yscale("log")
    axes[5].axhline(2_000_000, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
    axes[5].set_xlabel("sweep index")
    axes[5].set_ylabel("parameters")
    axes[5].set_title("Network parameters")

    draw_failure(failure_axes, summary, failures, palette, "run", best=best)
    draw_score_vs_anchor(anchor_ax, summary, palette, best=best)
    draw_town_density(density_ax, summary, palette)
    draw_hypers(hyper_ax, summary, palette, SCALED_HYPERS)
    for ax, label in zip(size_axes, SIZE_HYPERS):
        draw_size_panel(ax, summary, palette, label)
    for ax in (
        list(axes) + list(failure_axes) + [anchor_ax, density_ax]
        + [hyper_ax] + list(size_axes)
    ):
        style_ax(ax)
        lift_best(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def tag_phase(frame, phase):
    if frame is None or len(frame) == 0:
        return frame
    out = frame.copy()
    out["phase"] = phase
    out["series"] = [f"{phase}-{int(run):04d}" for run in out["run"]]
    return out


def series_palette(keys, best=()):
    """One seaborn color per series. Phase is part of the key so through and bump differ."""
    ordered = list(dict.fromkeys(keys))
    return assign_palette(ordered, best)


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


def draw_overview(returns, losses, counts, failures, summary, out):
    """Both physics phases. The sweep objective is the first panel.

    Through runs come first, then bump. Panels use the default seaborn palette, one color per run,
    except episode return against run order, which is colored by wall clock.
    Circles and solid lines are through. Squares and dashed lines are bump.
    The best objective is green, and a star on each scatter.
    """
    apply_figure_style()
    best = best_series(summary)
    palette = series_palette(summary["series"], best)
    fig, axes, failure_axes, anchor_ax, density_ax, hyper_ax, size_axes = figure_axes()
    markers = {"through": "o", "bump": "s"}

    draw_objective(axes[0], summary, palette, "idx", markers, best)

    lo = float(summary["wall_min"].min())
    hi = float(summary["wall_min"].max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    norm = plt.Normalize(lo, hi)
    scatter_wallclock(
        summary, "idx", "episode_return", axes[1], norm, markers, best,
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[1], pad=0.02,
    )
    colorbar.set_label("wall clock (min)")
    bump_idx = summary.loc[summary["phase"] == "bump", "idx"]
    if len(bump_idx):
        axes[1].axvline(float(bump_idx.min()) - 0.5, color="0.4", linestyle=":", linewidth=0.9)
    axes[1].set_xscale("linear")
    axes[1].set_yscale("linear")
    axes[1].set_xlabel("run order")
    axes[1].set_ylabel("episode return")
    axes[1].set_title("Episode return vs sweep index")

    for phase, linestyle in (("through", "solid"), ("bump", "dashed")):
        part = returns[returns["phase"] == phase] if len(returns) else returns
        if len(part) == 0:
            continue
        sns.lineplot(
            data=part, x="steps", y="episode_return", hue="series",
            palette=palette, estimator=None, linewidth=1.3, linestyle=linestyle,
            ax=axes[2], legend=False,
        )
    axes[2].set_xscale("linear")
    axes[2].set_yscale("linear")
    axes[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[2].ticklabel_format(axis="y", style="plain")
    axes[2].set_xlabel("steps")
    axes[2].set_ylabel("episode return")
    axes[2].set_title("Episode return vs steps")

    draw_anchor_loss(axes[3], losses, palette, markers)

    has_jerk = bool(summary["jerk"].notna().all() and summary["smooth"].notna().all())
    if has_jerk:
        scatter_metric(summary, "jerk", "score", axes[4], palette, markers, best)
        axes[4].set_xlabel("jerk")
        axes[4].set_ylabel("score")
        axes[4].set_title("Score vs jerk")
        axes[4].set_ylim(-0.03, 1.03)
    else:
        draw_distance_per_infraction(axes[4], counts, palette, markers)

    scatter_metric(summary, "idx", "parameters", axes[5], palette, markers, best)
    axes[5].set_yscale("log")
    axes[5].axhline(2_000_000, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
    axes[5].set_xlabel("run order")
    axes[5].set_ylabel("parameters")
    axes[5].set_title("Network parameters")

    draw_failure(failure_axes, summary, failures, palette, "idx", markers, best)
    draw_score_vs_anchor(anchor_ax, summary, palette, markers, best)
    draw_town_density(density_ax, summary, palette)
    draw_hypers(hyper_ax, summary, palette, SCALED_HYPERS)
    for ax, label in zip(size_axes, SIZE_HYPERS):
        draw_size_panel(ax, summary, palette, label)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="0.2", linestyle="None", label="through"),
        Line2D([0], [0], marker="s", color="0.2", linestyle="None", label="bump"),
        Line2D(
            [0], [0], marker="*", color=BEST_COLOR, linestyle="None",
            markersize=12, label="best",
        ),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    for ax in (
        list(axes) + list(failure_axes) + [anchor_ax, density_ax]
        + [hyper_ax] + list(size_axes)
    ):
        style_ax(ax)
        lift_best(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def load_overview(phases):
    """phases is a list of (name, log_path, metrics_dir)."""
    returns = []
    losses = []
    counts = []
    failures = []
    summaries = []
    missing = []
    for name, log_path, metrics_dir in phases:
        log_path = Path(log_path)
        if not log_path.exists() or not log_path.read_text().strip():
            continue
        try:
            (
                part_returns, part_losses, part_counts, part_failures,
                part_summary, part_missing,
            ) = load_frames(log_path, Path(metrics_dir))
        except SystemExit:
            continue
        returns.append(tag_phase(part_returns, name))
        losses.append(tag_phase(part_losses, name))
        counts.append(tag_phase(part_counts, name))
        failures.append(tag_phase(part_failures, name))
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
    loss_frames = [frame for frame in losses if frame is not None and len(frame)]
    losses = pd.concat(loss_frames, ignore_index=True) if loss_frames else pd.DataFrame()
    count_frames = [frame for frame in counts if frame is not None and len(frame)]
    counts = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    failure_frames = [frame for frame in failures if frame is not None and len(frame)]
    failures = (
        pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()
    )
    return returns, losses, counts, failures, summary, missing


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
        help="output image under logs/drive or logs/replay_sweep",
    )
    args = parser.parse_args()
    out = args.out
    allowed = (
        (ROOT / "logs" / "drive").resolve(),
        (ROOT / "logs" / "replay_sweep").resolve(),
    )
    if out.parent.resolve() not in allowed:
        out = ROOT / "logs" / "drive" / out.name
    if args.phase:
        returns, losses, counts, failures, summary, missing = load_overview(args.phase)
        draw_overview(returns, losses, counts, failures, summary, out)
    else:
        returns, losses, counts, failures, summary, missing = load_frames(
            args.log, args.metrics_dir,
        )
        draw(returns, losses, counts, failures, summary, out)
    print(f"wrote {out} runs={len(summary)}")
    if missing:
        print("no metrics for runs:", " ".join(str(run) for run in missing))


if __name__ == "__main__":
    main()
