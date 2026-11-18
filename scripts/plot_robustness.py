#!/usr/bin/env python3
"""Center density panel, in kilometers, against the Gigaflow level.

Same towns, markers, and centerline-density axis as the middle plot of
the eval panel's top row. The vertical axis is a log scale so a few
kilometers and 3 million km can share the figure. An open marker is the
training fleet of 128 cars. One star at 3 million km marks the
Gigaflow level, at the mean centerline density of the 50-car fleet.

    python scripts/plot_robustness.py logs/drive/best35_3b-eval.json \\
        logs/drive/eval-extra.json logs/drive/best35_3b-robustness.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_eval as pe

ROOT = Path(__file__).resolve().parents[1]
GIGA_KM = 3_000_000


def load_rows(paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text(), object_pairs_hook=pe.keep_first)
        rows.extend(data["density"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ref_agents, base, names = pe.load_density(ROOT / "config" / "town_density.json")
    frame = pd.DataFrame(load_rows(args.json_paths))
    frame["town"] = frame["map_id"].map(names)
    frame["density"] = frame["map_id"].map(base) * frame["agents"] / ref_agents
    frame["km"] = frame["avg_distance_per_infraction"] / 1000.0
    pe.apply_figure_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200, layout="constrained")
    ax.set_yscale("log")
    pe.draw_lines(
        ax, frame, "density", "km",
        "centerline density (%)",
        "average distance between infractions (km)",
        "Distance per infraction",
        True, True,
    )
    if ax.legend_ is not None:
        ax.legend_.set_loc("center right")
    pe.mark_training(ax, frame, "density", "km", 128, False)
    mark_gigaflow(ax, frame)
    positive = frame.loc[frame["km"] > 0, "km"]
    ax.set_ylim(float(positive.min()) / 2.0, GIGA_KM * 4.0)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)


def mark_gigaflow(ax, frame):
    """One star at 3 million km. Gigaflow reports that level at 50 agents."""
    fifty = frame.loc[frame["agents"] == 50, "density"]
    x = float(fifty.mean())
    ax.axvline(
        x, color="black", linestyle=(0, (2, 2)),
        linewidth=0.9, zorder=1,
    )
    ax.scatter(
        [x], [GIGA_KM],
        marker="*", s=280, color="black", zorder=7,
    )
    ax.annotate(
        f"Gigaflow robustness level\n50 agents, {x:.1f}% density",
        xy=(x, GIGA_KM),
        xytext=(12, -6),
        textcoords="offset points",
        ha="left", va="top", fontsize=14, color="black",
    )


if __name__ == "__main__":
    main()
