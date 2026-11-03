#!/usr/bin/env python3
"""Generalization panel for one drive checkpoint.

The top row is density. Cars per town go from the training count through
twice and four times that. Each town has its own color and marker.
The left panel is against the car count, so the training fleet is one
vertical line. The other two panels use centerline density from
config/town_density.json scaled by agents / 128. An open marker is the
training fleet. Score and average distance show the standard deviation
across rollouts.

The bottom row is horizon. Episode length goes from the training length
through twice and four times that, at the training car count. Average
infractions are the episode sum per agent, with the standard deviation
across rollouts. The last panel is one 1024-step episode. Agents are not reset
between bins. Each bar is vehicle collisions per agent in that
128-step window, so the height is the rate.

    python scripts/plot_eval.py logs/drive/eval.json logs/drive/eval-panel.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
TOWNS = ("Town01", "Town02", "Town06", "Town10HD")
TOWN_MARKERS = {
    "Town01": "o",
    "Town02": "s",
    "Town06": "D",
    "Town10HD": "^",
}
HORIZONS = ("256", "512", "1024")


def apply_figure_style():
    sns.set_style("ticks")
    sns.set_palette("deep")


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    if ax.legend_ is not None:
        ax.legend_.set_frame_on(False)


def load_density(town_json):
    data = json.loads(town_json.read_text())
    base = {row["map_id"]: row["centerline_pct"] for row in data["towns"]}
    names = {row["map_id"]: row["town"] for row in data["towns"]}
    return data["agents"], base, names


def rollout_sd(values):
    """Mean ± sample standard deviation across rollouts."""
    values = np.asarray(values, dtype=float)
    center = float(np.mean(values))
    if values.size < 2:
        return center, center
    spread = float(np.std(values, ddof=1))
    return center - spread, center + spread


def draw_lines(ax, frame, x, y, xlabel, ylabel, title, legend, error=False):
    kwargs = dict(
        data=frame,
        x=x,
        y=y,
        hue="town",
        style="town",
        hue_order=list(TOWNS),
        style_order=list(TOWNS),
        palette="deep",
        markers=TOWN_MARKERS,
        dashes=False,
        estimator="mean",
        errorbar=rollout_sd if error else None,
        ax=ax,
        legend=legend,
    )
    if error:
        kwargs["err_style"] = "bars"
        kwargs["err_kws"] = {"linewidth": 1.0, "capsize": 3}
    sns.lineplot(**kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_ax(ax)
    if legend and ax.legend_ is not None:
        ax.legend_.set_title("town")


def mark_training(ax, frame, x, y, train_agents, guide):
    """Open marker on the training fleet. The car-count axis gets a line."""
    rows = frame[frame["agents"] == train_agents]
    means = rows.groupby(["town", x], as_index=False)[y].mean()
    for town, marker in TOWN_MARKERS.items():
        part = means[means["town"] == town]
        if part.empty:
            continue
        ax.scatter(
            part[x], part[y],
            marker=marker, s=110,
            facecolors="none", edgecolors="black",
            linewidths=1.3, zorder=6,
        )
    if not guide:
        return
    ax.axvline(
        train_agents, color="black",
        linestyle=(0, (2, 1.5)), linewidth=0.8, zorder=1,
    )
    ax.text(
        train_agents, 0.98, " training",
        transform=ax.get_xaxis_transform(),
        fontsize=7, color="black", ha="left", va="top",
    )


def draw_density(axes, rows, ref_agents, base, names, train_agents):
    frame = pd.DataFrame(rows)
    frame["town"] = frame["map_id"].map(names)
    frame["density"] = frame["map_id"].map(base) * frame["agents"] / ref_agents
    specs = (
        ("agents", "score", "score", "Score vs cars", "num cars", True),
        (
            "density",
            "avg_distance_per_infraction",
            "average distance between infractions (m)",
            "Meters per infraction",
            "centerline density (%)",
            True,
        ),
        (
            "density",
            "collision_count",
            "collision count",
            "Collisions vs density",
            "centerline density (%)",
            False,
        ),
    )
    for ax, (x, key, ylabel, title, xlabel, error) in zip(axes, specs):
        draw_lines(
            ax, frame, x, key, xlabel, ylabel, title,
            ax is axes[0], error,
        )
        mark_training(ax, frame, x, key, train_agents, x == "agents")


def draw_horizon(axes, rows):
    frame = pd.DataFrame(rows)
    ticks = sorted(frame["episode_length"].unique())
    specs = (
        (
            "total_infractions",
            "Average infractions per agent",
            "Infractions vs horizon",
            True,
        ),
        ("collision_count", "collision count", "Collisions vs horizon", False),
    )
    for ax, (key, ylabel, title, error) in zip(axes, specs):
        draw_lines(
            ax, frame, "episode_length", key,
            "episode length", ylabel, title, False, error,
        )
        ax.set_xticks(ticks)


def draw_failures(ax, rows, bin_steps):
    """Collision rate inside one continuous 1024-step episode."""
    longest = 1024
    width = 128
    assert width % bin_steps == 0
    flat = []
    seen = False
    for row in rows:
        if int(row["episode_length"]) != longest:
            continue
        seen = True
        agents = row["agents"]
        counts = {}
        for item in row["bins"]:
            start = (int(item["start"]) // width) * width
            counts[start] = counts.get(start, 0) + item["collisions"]
        for start in range(0, longest, width):
            flat.append({
                "town": row["town"],
                "start": start,
                "collisions": counts.get(start, 0) / agents,
            })
    assert seen
    frame = pd.DataFrame(flat)
    sns.barplot(
        data=frame,
        x="start",
        y="collisions",
        hue="town",
        hue_order=list(TOWNS),
        palette="deep",
        ax=ax,
        errorbar=None,
    )
    ax.set_title(f"Collision rate ({longest} steps)")
    ax.set_xlabel(f"step ({width}-step bins)")
    ax.set_ylabel("collisions per agent")
    style_ax(ax)
    if ax.legend_ is not None:
        ax.legend_.set_title("town")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def keep_first(pairs):
    """cJSON used to emit episode_length twice. The first is the horizon."""
    obj = {}
    for key, value in pairs:
        if key not in obj:
            obj[key] = value
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("png_path")
    args = parser.parse_args()
    data = json.loads(Path(args.json_path).read_text(),
        object_pairs_hook=keep_first)
    ref_agents, base, names = load_density(ROOT / "config" / "town_density.json")
    train_agents = {int(row["agents"]) for row in data["horizon"]}
    assert len(train_agents) == 1
    apply_figure_style()
    fig, axes = plt.subplots(2, 3, figsize=(22.4, 10.4), dpi=100,
        layout="constrained")
    draw_density(
        axes[0], data["density"], ref_agents, base, names,
        train_agents.pop(),
    )
    draw_horizon(axes[1], data["horizon"])
    draw_failures(axes[1, 2], data["horizon"], data.get("failure_bin", 64))
    fig.savefig(args.png_path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
