#!/usr/bin/env python3
"""Town06 collisions per agent on a density by horizon grid.

One episode per car count. Each horizon column is one time bin
(failure_bin steps, 100 by default). Counts in the log are fleet
totals. Both panels divide by the number of cars. The right panel is
the mean collisions per agent inside that bin. The left panel is the
running mean through the end of the bin.

    python scripts/plot_benchmark.py logs/drive/benchmark.json \
        logs/drive/benchmark-summary.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize


def window_end(bin_i, width, length):
    """Last dynamics step in this bin. The closing tick only resets."""
    last = length - 2
    end = min((bin_i + 1) * width - 1, last)
    assert end >= bin_i * width
    return end


def tables(runs, width):
    per_bin = []
    cumulative = []
    order = []
    for run in runs:
        agents = int(run["agents"])
        length = int(run["episode_length"])
        counts = [int(v) for v in run["collisions"]]
        nbins = length // width
        assert len(counts) == nbins
        total = 0
        for bin_i, count in enumerate(counts):
            start = bin_i * width
            end = window_end(bin_i, width, length)
            label = f"{start}–{end}"
            if label not in order:
                order.append(label)
            total += count
            per_bin.append({
                "density": agents,
                "horizon": label,
                "collisions": count / agents,
            })
            cumulative.append({
                "density": agents,
                "horizon": label,
                "collisions": total / agents,
            })
    return per_bin, cumulative, order


def pivot(rows, columns):
    densities = []
    for row in rows:
        if row["density"] not in densities:
            densities.append(row["density"])
    return (
        pd.DataFrame(rows)
        .pivot(index="density", columns="horizon", values="collisions")
        .reindex(index=list(reversed(densities)), columns=columns)
    )


def draw(ax, table, cmap, cbar_label):
    norm = Normalize(
        vmin=float(table.min().min()),
        vmax=float(table.max().max()),
    )
    sns.heatmap(
        table,
        ax=ax,
        cmap=cmap,
        vmin=norm.vmin,
        vmax=norm.vmax,
        cbar_kws={"label": cbar_label},
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
    )
    for i, density in enumerate(table.index):
        for j, horizon in enumerate(table.columns):
            value = float(table.loc[density, horizon])
            red, green, blue, _ = cmap(norm(value))
            lum = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                j + 0.5, i + 0.5, f"{value:.2f}",
                ha="center", va="center", fontsize=5,
                color="black" if lum > 0.55 else "white",
            )
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("center")
    return norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("png_path")
    args = parser.parse_args()
    data = json.loads(Path(args.json_path).read_text())
    width = int(data["failure_bin"])
    per_bin, cumulative, order = tables(data["runs"], width)
    left = pivot(cumulative, order)
    right = pivot(per_bin, order)
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 2, figsize=(32.0, 7.2), dpi=200)
    draw(axes[0], left, cmap, "cumulative collisions per agent")
    draw(axes[1], right, cmap, "collisions per agent in bin")
    axes[0].set_xlabel(
        "horizon (mean cumulative collisions per agent through end of bin)"
    )
    axes[1].set_xlabel(
        f"horizon bin (mean collisions per agent over {width} steps)"
    )
    axes[0].set_ylabel("density (cars)")
    axes[1].set_ylabel("density (cars)")
    axes[0].set_title("Cumulative, per agent")
    axes[1].set_title("Per bin, per agent")
    fig.suptitle(f"Vehicle collisions, {data['town']}", y=1.02)
    fig.tight_layout()
    fig.savefig(args.png_path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
