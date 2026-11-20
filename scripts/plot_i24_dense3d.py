"""3D views of the two densest I24 windows.

Each vehicle is its logged box. The arrow at the base of the box points
along the logged heading, and its length is the logged speed. A 20 m/s
car draws a 10 m arrow. The view is a low oblique look down the freeway.

  python scripts/plot_i24_dense3d.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "data" / "i24" / "json"
OUT_DIR = ROOT / "data" / "i24" / "assets"

WINDOW_M = 100.0
# Meters of arrow per m/s. 20 m/s draws a 5 m arrow, about one car length.
ARROW_M_PER_MPS = 0.25
# Almost flat, the way the GPUDrive viewer draws a vehicle.
Z_SCALE = 0.45
SCENES = (
    ("i24_00002.json", 3, "i24_dense_scene3_3d.png"),
    ("i24_00000.json", 1, "i24_dense_scene1_3d.png"),
)

# GPUDrive draws a pale road, gray lanes, a darker edge, and a muted
# center line. The selected vehicle is red. Everyone else is white.
ROAD_COLOR = {
    "lane": "#b0b0b0",
    "road_line": "#c6a15a",
    "road_edge": "#5c5c5c",
}
ROAD_WIDTH = {"lane": 0.6, "road_line": 0.9, "road_edge": 1.35}
ASPHALT = "#e4e4e4"
# Seaborn deep blue. Every logged box uses it.
BOX_COLOR = "#4c72b0"


def load(name: str) -> dict:
    return json.loads((JSON_DIR / name).read_text())


def speed(sample: dict) -> float:
    return math.hypot(float(sample["x"]), float(sample["y"]))


def densest(doc: dict) -> tuple[int, float]:
    """Time index and window start with the most valid vehicles."""
    objects = doc["objects"]
    states = len(objects[0]["valid"])
    best = (0, 0, 0.0)
    for t in range(0, states, 5):
        xs = [
            float(obj["position"][t]["x"])
            for obj in objects
            if obj["valid"][t]
        ]
        if len(xs) < 5:
            continue
        xs_arr = np.asarray(xs)
        for x0 in np.arange(xs_arr.min(), xs_arr.max(), 20.0):
            count = int(np.sum((xs_arr >= x0) & (xs_arr < x0 + WINDOW_M)))
            if count > best[0]:
                best = (count, t, float(x0))
    return best[1], best[2]


def arrow_head(x, y, z, heading, reach):
    """A flat chevron at the tip of a speed arrow, in the road plane."""
    c = math.cos(heading)
    s = math.sin(heading)
    tip_x = x + c * reach
    tip_y = y + s * reach
    back = min(1.3, reach * 0.28)
    flare = back * 0.32
    return [
        (tip_x, tip_y, z),
        (tip_x - c * back - s * flare, tip_y - s * back + c * flare, z),
        (tip_x - c * back + s * flare, tip_y - s * back - c * flare, z),
    ]


def box_faces(x, y, heading, length, width, height):
    c = math.cos(heading)
    s = math.sin(heading)
    hl = length * 0.5
    hw = width * 0.5
    local = (
        (hl, hw, 0.0),
        (hl, -hw, 0.0),
        (-hl, -hw, 0.0),
        (-hl, hw, 0.0),
        (hl, hw, height),
        (hl, -hw, height),
        (-hl, -hw, height),
        (-hl, hw, height),
    )
    corners = []
    for dx, dy, dz in local:
        corners.append((
            x + dx * c - dy * s,
            y + dx * s + dy * c,
            dz * Z_SCALE,
        ))
    faces_idx = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    return [[corners[i] for i in face] for face in faces_idx]


def asphalt(doc: dict, x0: float, x1: float):
    """A slab between the two outermost road edges, sampled along x."""
    edges = [road for road in doc["roads"] if road["type"] == "road_edge"]
    if len(edges) < 2:
        return None
    edges.sort(key=lambda road: float(np.mean([pt["y"] for pt in road["geometry"]])))
    low, high = edges[0], edges[-1]

    def y_at(road, x):
        pts = sorted(road["geometry"], key=lambda pt: float(pt["x"]))
        xs = [float(pt["x"]) for pt in pts]
        ys = [float(pt["y"]) for pt in pts]
        return float(np.interp(x, xs, ys))

    samples = np.linspace(x0 - 4, x1 + 4, 40)
    forward = [(float(x), y_at(low, x), 0.0) for x in samples]
    back = [(float(x), y_at(high, x), 0.0) for x in samples[::-1]]
    return forward + back


def draw(doc: dict, scene: int, t: int, x0: float, path: Path) -> None:
    x1 = x0 + WINDOW_M
    objects = []
    for obj in doc["objects"]:
        if not obj["valid"][t]:
            continue
        pos = obj["position"][t]
        x = float(pos["x"])
        if x < x0 or x >= x1:
            continue
        objects.append(obj)
    objects.sort(key=lambda obj: float(obj["position"][t]["y"]))

    sns.set_theme(style="white", context="paper")

    fig = plt.figure(figsize=(8.6, 4.8), dpi=200)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    elev, azim = 62, -70
    elev_r = math.radians(elev)
    azim_r = math.radians(azim)
    cam = (
        math.cos(elev_r) * math.cos(azim_r),
        math.cos(elev_r) * math.sin(azim_r),
        math.sin(elev_r),
    )

    def depth(x, y, z):
        return x * cam[0] + y * cam[1] + z * cam[2]
    y_lo = min(pt["y"] for road in doc["roads"] for pt in road["geometry"])
    y_hi = max(pt["y"] for road in doc["roads"] for pt in road["geometry"])
    pad_y = 1.2
    slab = asphalt(doc, x0, x1)
    if slab is not None:
        ax.add_collection3d(Poly3DCollection(
            [slab], facecolor=ASPHALT, edgecolor="none", alpha=1.0, zorder=0,
        ))

    for road in doc["roads"]:
        pts = [
            pt for pt in road["geometry"]
            if x0 - 10 <= float(pt["x"]) <= x1 + 10
        ]
        if len(pts) < 2:
            continue
        kind = road["type"]
        ax.plot(
            [float(pt["x"]) for pt in pts],
            [float(pt["y"]) for pt in pts],
            np.full(len(pts), 0.02),
            color=ROAD_COLOR.get(kind, "#888888"),
            linewidth=ROAD_WIDTH.get(kind, 0.8),
            solid_capstyle="round",
            zorder=1,
        )

    # Farther vehicles first, so a nearer box paints over the one behind it.
    objects.sort(key=lambda obj: depth(
        float(obj["position"][t]["x"]),
        float(obj["position"][t]["y"]),
        0.0,
    ))
    heads = []
    shafts = []
    for rank, obj in enumerate(objects):
        pos = obj["position"][t]
        x = float(pos["x"])
        y = float(pos["y"])
        heading = float(obj["heading"][t])
        length = float(obj["length"])
        width = float(obj["width"])
        height = float(obj["height"])
        color = BOX_COLOR
        faces = box_faces(x, y, heading, length, width, height)
        ax.add_collection3d(Poly3DCollection(
            faces[1:],
            facecolor=color,
            edgecolor="#222222",
            linewidths=0.45,
            alpha=1.0,
            zorder=10 + rank,
        ))
        base = 0.12
        vel = speed(obj["velocity"][t])
        reach = vel * ARROW_M_PER_MPS
        if reach < 0.4:
            continue
        c = math.cos(heading)
        s = math.sin(heading)
        shafts.append((
            [x, x + c * reach],
            [y, y + s * reach],
            [base, base],
            rank,
        ))
        heads.append((arrow_head(x, y, base, heading, reach), rank))
    for xs, ys, zs, order in shafts:
        ax.plot(xs, ys, zs, color="#1a1a1a", linewidth=0.6, solid_capstyle="butt", zorder=order)
    for head, order in heads:
        ax.add_collection3d(Poly3DCollection(
            [head], facecolor="#1a1a1a", edgecolor="#1a1a1a", linewidths=0.15,
            zorder=order,
        ))

    ax.set_xlim(x0 - 6, x1 + 6)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
    z_top = 1.6
    ax.set_zlim(0.0, z_top)
    ax.set_box_aspect((
        (x1 - x0) + 8,
        (y_hi - y_lo) + 2 * pad_y,
        3.2,
    ))
    ax.view_init(elev=elev, azim=azim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.set_axis_off()
    ax.set_title(
        f"I-24 scene {scene}    {t / 10:.1f} s    {len(objects)} vehicles\n"
        "arrow length is speed,  5 m = 20 m/s",
        fontsize=11,
        pad=2,
        color="#222222",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}  t={t}  vehicles={len(objects)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, scene, filename in SCENES:
        doc = load(name)
        t, x0 = densest(doc)
        draw(doc, scene, t, x0, OUT_DIR / filename)


if __name__ == "__main__":
    main()
