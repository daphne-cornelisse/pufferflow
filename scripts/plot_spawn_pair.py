#!/usr/bin/env python3
"""Standard and hard spawn on the same road graph.

The placement code does not read difficulty. Both panels use the same
seed. The hard panel is d = 1: a 10-step episode and a 0.35 s headway.
The mark in front of each hard car is that headway at the spawn speed.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6


def apply_style():
    sns.set_style("ticks")
    sns.set_palette("deep")


def draw_roads(ax, roads):
    for road in roads:
        pts = road["xy"]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        kind = road["type"]
        if kind == ROAD_EDGE:
            ax.plot(xs, ys, color="0.45", linewidth=0.8, zorder=1)
        elif kind == ROAD_LINE:
            ax.plot(xs, ys, color="0.75", linewidth=0.4, zorder=1)
        else:
            ax.plot(xs, ys, color="0.62", linewidth=0.5, zorder=1)


def draw_cars(ax, cars, color, headway):
    for car in cars:
        ax.annotate(
            "",
            xy=(car["x"] + 4.0 * car["hx"], car["y"] + 4.0 * car["hy"]),
            xytext=(car["x"], car["y"]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
            zorder=3,
        )
        ax.plot(
            [car["x"], car["gx"]], [car["y"], car["gy"]],
            color=color, linewidth=0.4, alpha=0.45, zorder=2,
        )
        ax.scatter(
            [car["gx"]], [car["gy"]], s=6, color=color,
            linewidths=0, zorder=3,
        )
        if headway and car["tau"] > 0 and car["speed"] > 0.05:
            dist = car["tau"] * car["speed"]
            ax.plot(
                [car["x"], car["x"] + dist * car["hx"]],
                [car["y"], car["y"] + dist * car["hy"]],
                color="#C44E52", linewidth=1.0, zorder=4,
            )


def main():
    apply_style()
    payload = json.loads(Path("/tmp/spawn_pair.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), dpi=100)
    color = sns.color_palette("deep")[0]
    for ax, key in zip(axes, ("standard", "hard")):
        mode = payload[key]
        draw_roads(ax, payload["roads"])
        draw_cars(ax, mode["cars"], color, headway=(key == "hard"))
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if key == "standard":
            title = f"Standard: limit {mode['episode_limit']} steps, no headway"
        else:
            title = (
                f"Hard: limit {mode['episode_limit']} steps, "
                f"headway {mode['headway_mean']:.2f} s"
            )
        ax.set_title(title)
    out = ROOT / "logs" / "drive-currv1" / "spawn_standard_hard.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
