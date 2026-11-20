"""Generate time-space diagrams for the I24-3D ground-truth trajectories.

The script writes two figures:

1. Eastbound and westbound trajectories in separate panels for scenes 1-3.
2. Scene 3 westbound trajectories together and split by predominant lane.

Lane membership is inferred from the lateral boundaries recovered from each
scene's ``space_pts`` calibration grid. A vehicle is assigned to the lane in
which it spends the most trajectory samples, so a lane-changing trajectory is
shown in exactly one lane panel.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

if __package__:
    from .i24_3d_to_spiced import (
        MAJOR_QUALITY_FLAGS,
        SAMPLE_HZ,
        normalize_scene,
    )
else:
    from i24_3d_to_spiced import (
        MAJOR_QUALITY_FLAGS,
        SAMPLE_HZ,
        normalize_scene,
    )


# Continuous red-yellow-green gradient: stopped traffic is red, the midpoint
# is yellow, and free-flow traffic is green.
SPEED_COLORMAP = matplotlib.colormaps["RdYlGn"].resampled(512)
LANE_DESCRIPTORS = {
    1: "median-side",
    2: "inner-middle",
    3: "outer-middle",
    4: "shoulder-side",
}


def clean_objects(scene: dict[str, Any], include_major_flags: bool) -> list[dict[str, Any]]:
    """Return all objects, or remove objects carrying a major quality flag."""
    if include_major_flags:
        return scene["objects"]
    return [
        obj
        for obj in scene["objects"]
        if not (set(obj.get("quality_flags", [])) & MAJOR_QUALITY_FLAGS)
    ]


def time_bounds(objects: list[dict[str, Any]]) -> tuple[int, int]:
    """Return the first and last 10 Hz ticks represented by a set of objects."""
    first_tick = min(int(obj["start_tick"]) for obj in objects)
    last_tick = max(
        int(obj["start_tick"]) + len(obj["position_m"]) - 1 for obj in objects
    )
    return first_tick, last_tick


def trajectory_time(obj: dict[str, Any], scene_start_tick: int) -> list[float]:
    """Return time in seconds relative to a scene's first retained sample."""
    return [
        (int(obj["start_tick"]) + index - scene_start_tick) / SAMPLE_HZ
        for index in range(len(obj["position_m"]))
    ]


def speed_norm(speed_max_mps: float) -> Normalize:
    """Return the shared speed normalization used by every panel."""
    return Normalize(vmin=0.0, vmax=speed_max_mps, clip=True)


def add_speed_colored_trajectory(
    axis: Any,
    obj: dict[str, Any],
    scene_start_tick: int,
    normalization: Normalize,
    linewidth: float,
    alpha: float,
) -> None:
    """Draw one trajectory with each segment colored by its local speed."""
    times = trajectory_time(obj, scene_start_tick)
    positions = [point[0] for point in obj["position_m"]]
    if len(times) < 2:
        return

    points = np.column_stack((times, positions))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    state_speeds = np.asarray(
        [(vx * vx + vy * vy) ** 0.5 for vx, vy in obj["velocity_mps"]]
    )
    segment_speeds = 0.5 * (state_speeds[:-1] + state_speeds[1:])
    collection = LineCollection(
        segments,
        cmap=SPEED_COLORMAP,
        norm=normalization,
        linewidths=linewidth,
        alpha=alpha,
    )
    collection.set_array(segment_speeds)
    axis.add_collection(collection)


def add_speed_colorbar(
    figure: Any,
    normalization: Normalize,
    speed_max_mps: float,
) -> None:
    """Add a shared red-yellow-green speed legend below a figure."""
    colorbar_axis = figure.add_axes((0.35, 0.018, 0.30, 0.018))
    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=SPEED_COLORMAP),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    midpoint = speed_max_mps / 2.0
    colorbar.set_ticks((0.0, midpoint, speed_max_mps))
    colorbar.set_ticklabels(("0", f"{midpoint:g}", f"≥{speed_max_mps:g}"))
    colorbar.set_label("Local speed (m/s)")


def style_axis(
    axis: Any,
    duration: float,
    x_limits: tuple[float, float],
) -> None:
    """Apply common time-space axis limits and labels."""
    axis.set_xlim(0.0, duration)
    axis.set_ylim(*x_limits)
    axis.set_xlabel("Time from scene start (s)")
    axis.set_ylabel("Longitudinal position, x (m)")
    axis.grid(True, color="#D8D8D8", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.75)


def configure_matplotlib() -> None:
    """Set compact, consistent typography for both figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def plot_time_space_by_direction(
    scenes: list[dict[str, Any]],
    output_path: Path,
    include_major_flags: bool,
    speed_max_mps: float,
    dpi: int,
) -> None:
    """Plot eastbound and westbound trajectories separately for scenes 1-3."""
    rows = []
    for scene in scenes:
        objects = clean_objects(scene, include_major_flags)
        first_tick, last_tick = time_bounds(objects)
        rows.append((scene, objects, first_tick, last_tick))

    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.3))
    figure.subplots_adjust(
        left=0.072,
        right=0.985,
        bottom=0.10,
        top=0.905,
        hspace=0.42,
        wspace=0.16,
    )
    figure.suptitle(
        "I-24-3D time-space diagrams by travel direction",
        fontsize=17,
        fontweight="bold",
        y=0.976,
    )
    if include_major_flags:
        quality_text = "major quality flags included"
    else:
        exclusions = ", ".join(
            f"Scene {scene['scene']}: {scene['object_count'] - len(objects)}"
            for scene, objects, _, _ in rows
        )
        quality_text = f"major-quality exclusions — {exclusions}"
    figure.text(
        0.5,
        0.945,
        f"Full 10 Hz ground-truth trajectories · {quality_text}",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )

    direction_labels = {1: "Eastbound (+x)", -1: "Westbound (−x)"}
    normalization = speed_norm(speed_max_mps)
    for row_index, (scene, objects, first_tick, last_tick) in enumerate(rows):
        duration = (last_tick - first_tick) / SAMPLE_HZ
        all_x = [point[0] for obj in objects for point in obj["position_m"]]
        x_limits = (min(all_x), max(all_x))

        for column_index, direction in enumerate((1, -1)):
            axis = axes[row_index, column_index]
            subset = [obj for obj in objects if int(obj["direction"]) == direction]
            for obj in subset:
                add_speed_colored_trajectory(
                    axis,
                    obj,
                    first_tick,
                    normalization,
                    linewidth=0.68,
                    alpha=0.38,
                )
            axis.set_title(
                f"Scene {scene['scene']} · {direction_labels[direction]} · "
                f"{len(subset)} trajectories",
                loc="left",
                pad=7,
                fontweight="semibold",
            )
            style_axis(axis, duration, x_limits)

    add_speed_colorbar(figure, normalization, speed_max_mps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def carriageway_boundaries(scene: dict[str, Any], direction: int) -> list[float]:
    """Return lateral lane boundaries for one side of the median."""
    boundaries = scene["map"]["boundary_y_m"]
    gaps = [right - left for left, right in zip(boundaries, boundaries[1:])]
    median_index = max(range(len(gaps)), key=gaps.__getitem__)
    if direction == 1:
        return boundaries[: median_index + 1]
    if direction == -1:
        return boundaries[median_index + 1 :]
    raise ValueError(f"Direction must be 1 or -1, not {direction}")


def lane_centers(scene: dict[str, Any], direction: int) -> list[float]:
    """Return lateral center coordinates ordered from median to shoulder."""
    boundaries = carriageway_boundaries(scene, direction)
    centers = [(left + right) / 2.0 for left, right in zip(boundaries, boundaries[1:])]
    if direction == 1:
        centers.reverse()
    return centers


def sample_lane(lateral_position: float, centers: list[float]) -> int:
    """Assign one sample to its nearest calibrated lane center."""
    return min(range(len(centers)), key=lambda index: abs(lateral_position - centers[index])) + 1


def trajectory_lanes(obj: dict[str, Any], centers: list[float]) -> list[int]:
    """Return the nearest lane for every point in a trajectory."""
    return [sample_lane(point[1], centers) for point in obj["position_m"]]


def predominant_lane(obj: dict[str, Any], centers: list[float]) -> int:
    """Assign a trajectory to the lane containing most of its samples."""
    counts = Counter(trajectory_lanes(obj, centers))
    return min(counts, key=lambda lane: (-counts[lane], lane))


def plot_scene3_westbound_by_lane(
    scene: dict[str, Any],
    output_path: Path,
    include_major_flags: bool,
    speed_max_mps: float,
    dpi: int,
) -> None:
    """Compare combined Scene 3 westbound traffic with lane-specific panels."""
    objects = clean_objects(scene, include_major_flags)
    westbound = [obj for obj in objects if int(obj["direction"]) == -1]
    first_tick, last_tick = time_bounds(objects)
    duration = (last_tick - first_tick) / SAMPLE_HZ
    centers = lane_centers(scene, direction=-1)
    if len(centers) != 4:
        raise ValueError(f"Expected four Scene 3 westbound lanes, found {len(centers)}")

    lane_of = {int(obj["id"]): predominant_lane(obj, centers) for obj in westbound}
    assigned = {
        lane: [obj for obj in westbound if lane_of[int(obj["id"])] == lane]
        for lane in range(1, 5)
    }
    lane_changers = sum(
        len(set(trajectory_lanes(obj, centers))) > 1 for obj in westbound
    )
    all_x = [point[0] for obj in westbound for point in obj["position_m"]]
    x_limits = (min(all_x), max(all_x))

    figure = plt.figure(figsize=(13.4, 11.6))
    grid = GridSpec(
        3,
        2,
        figure=figure,
        height_ratios=[1.16, 1.0, 1.0],
        hspace=0.45,
        wspace=0.17,
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.89,
    )
    combined_axis = figure.add_subplot(grid[0, :])
    lane_axes = {
        1: figure.add_subplot(grid[1, 0]),
        2: figure.add_subplot(grid[1, 1]),
        3: figure.add_subplot(grid[2, 0]),
        4: figure.add_subplot(grid[2, 1]),
    }
    figure.suptitle(
        "I-24-3D Scene 3 westbound: combined versus separated by lane",
        fontsize=17,
        fontweight="bold",
        y=0.972,
    )
    figure.text(
        0.5,
        0.94,
        f"Predominant lane from calibrated lateral position · "
        f"{lane_changers} of {len(westbound)} trajectories change lanes",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )

    normalization = speed_norm(speed_max_mps)
    for obj in westbound:
        add_speed_colored_trajectory(
            combined_axis,
            obj,
            first_tick,
            normalization,
            alpha=0.31,
            linewidth=0.68,
        )
    combined_axis.set_title(
        f"All westbound trajectories together · {len(westbound)} trajectories",
        loc="left",
        pad=7,
        fontweight="semibold",
    )
    style_axis(combined_axis, duration, x_limits)

    for lane, axis in lane_axes.items():
        for obj in assigned[lane]:
            add_speed_colored_trajectory(
                axis,
                obj,
                first_tick,
                normalization,
                alpha=0.48,
                linewidth=0.72,
            )
        axis.set_title(
            f"Lane {lane} ({LANE_DESCRIPTORS[lane]}) · "
            f"{len(assigned[lane])} trajectories",
            loc="left",
            pad=7,
            fontweight="semibold",
        )
        style_axis(axis, duration, x_limits)

    add_speed_colorbar(figure, normalization, speed_max_mps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    repo_root = Path(__file__).resolve().parents[2]
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--i24-root",
        type=Path,
        default=repo_root.parent / "i24_3d_dataset" / "I24-3D",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures" / "i24_time_space",
    )
    argument_parser.add_argument(
        "--include-major-flags",
        action="store_true",
        help="Include trajectories with extreme speed or calibrated-range flags",
    )
    argument_parser.add_argument(
        "--speed-max-mps",
        type=float,
        default=35.0,
        help="Speed mapped to green; larger values are clipped (default: 35)",
    )
    argument_parser.add_argument("--dpi", type=int, default=150)
    return argument_parser


def main() -> None:
    """Load the I24-3D data and write both requested time-space figures."""
    args = parser().parse_args()
    if args.speed_max_mps <= 0:
        raise SystemExit("--speed-max-mps must be positive")
    configure_matplotlib()
    scenes = [
        normalize_scene(args.i24_root.resolve(), scene, stabilize_headings=False)
        for scene in (1, 2, 3)
    ]
    plot_time_space_by_direction(
        scenes,
        args.output_dir / "i24_time_space_by_direction.png",
        include_major_flags=args.include_major_flags,
        speed_max_mps=args.speed_max_mps,
        dpi=args.dpi,
    )
    plot_scene3_westbound_by_lane(
        scenes[2],
        args.output_dir / "i24_scene3_westbound_by_lane.png",
        include_major_flags=args.include_major_flags,
        speed_max_mps=args.speed_max_mps,
        dpi=args.dpi,
    )
    print(f"Wrote {args.output_dir / 'i24_time_space_by_direction.png'}")
    print(f"Wrote {args.output_dir / 'i24_scene3_westbound_by_lane.png'}")


if __name__ == "__main__":
    main()
