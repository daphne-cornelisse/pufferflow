#!/usr/bin/env python3
"""Meta-micro realism bars for I24 scenes 1 and 3.

A green dashed line at 1 is ground truth. Bars are the BC anchor, the
regularized 1B run (spiced self-play), and the unregularized 1B run.

    python scripts/plot_dreal_meta.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "analysis" / "offline" / "i24"
OUT = REPORT_DIR / "dreal_sweep_scenes13_meta.png"
GT_GREEN = "#14532d"

# Anchor is shared by the finished reports. The two policies are the 1B runs.
SOURCES = (
    ("anchor", "Anchor", "dreal_sweep_scenes13.json", "#d62728"),
    ("spiced", "Spiced self-play", "dreal_sweep_scenes13_spiced.json", "#722F37"),
    ("regular", "Unregularized self-play", "dreal_sweep_scenes13_regular.json", "#111111"),
)


def composite(block):
    if "meta" in block:
        return block["meta"]
    return block["eta"]


def load_method(path, key):
    doc = json.loads(path.read_text())
    if key not in doc["methods"]:
        raise SystemExit(f"{path.name} has no {key}")
    return composite(doc["methods"][key])


def main():
    rows = []
    for key, label, name, color in SOURCES:
        path = REPORT_DIR / name
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        stat = load_method(path, key)
        rows.append({
            "method": label,
            "meta": stat["mean"],
            "std": stat["std"],
            "n": stat["n"],
            "color": color,
        })
    frame = pd.DataFrame(rows)
    labels = list(frame["method"])
    palette = dict(zip(frame["method"], frame["color"]))
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    sns.barplot(
        data=frame,
        x="method",
        y="meta",
        hue="method",
        palette=palette,
        ax=ax,
        legend=False,
        order=labels,
    )
    ax.errorbar(
        x=range(len(frame)),
        y=frame["meta"],
        yerr=frame["std"],
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        capthick=1,
    )
    ax.axhline(
        1.0, color=GT_GREEN, linestyle=(0, (6, 3)), linewidth=1.8,
        zorder=3, clip_on=False,
    )
    ax.text(
        0.99, 0.975,
        "Ceiling (ground-truth trajectories)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=GT_GREEN,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_axisbelow(True)
    for tick in (0.2, 0.4, 0.6, 0.8):
        ax.axhline(tick, color="#9a9a9a", alpha=0.35, linewidth=0.9, zorder=0)
    for patch in ax.patches:
        patch.set_zorder(2)
    ax.set_xlabel("")
    ax.set_ylabel("meta-micro realism score", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.annotate(
        "",
        xy=(0, 1.0),
        xytext=(0, 0.93),
        xycoords=ax.get_yaxis_transform(),
        arrowprops=dict(
            arrowstyle="-|>", color="black", lw=1.1, mutation_scale=9,
            shrinkA=0, shrinkB=0,
        ),
        annotation_clip=False,
        zorder=5,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")
    for row in rows:
        print(f"{row['method']}: {row['meta']:.4f} ± {row['std']:.4f}")


if __name__ == "__main__":
    main()
