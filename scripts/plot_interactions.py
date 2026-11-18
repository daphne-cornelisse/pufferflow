#!/usr/bin/env python3
"""Useful and wasteful transitions in one Gigaflow-sized rollout.

The four towns are averaged. A useful transition is an agent step within
2 m of another car. The rest of the 5 million transitions are wasteful.

    python scripts/plot_interactions.py logs/drive/interactions.json \
        logs/drive/best35_3b-interactions.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Gigaflow rollout: 38,400 environments for 128 steps.
BATCH = 5_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("out")
    args = parser.parse_args()
    data = json.loads(Path(args.json_path).read_text())
    useful_share = data["fraction_agent_steps_in_interaction"]
    useful = useful_share * BATCH
    wasteful = BATCH - useful
    label = "total transitions per rollout (5 million)"
    frame = pd.DataFrame([
        {"rollout": label, "kind": "useful", "count": useful},
        {"rollout": label, "kind": "wasteful", "count": wasteful},
    ])
    sns.set_style("ticks")
    sns.set_palette("deep")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=100, layout="constrained")
    sns.barplot(data=frame, x="rollout", y="count", hue="kind", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("transitions")
    ax.set_title("Average across towns")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _p: f"{v / 1_000_000:.0f}M" if v else "0")
    )
    ax.set_ylim(0, BATCH * 1.18)
    for patch, count in zip(ax.patches, (useful, wasteful)):
        if patch.get_height() == 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{count:,.0f}\n{100 * count / BATCH:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    ax.legend(title="")
    fig.savefig(args.out, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
