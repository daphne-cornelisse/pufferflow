"""Generate Eulerian speed-field plots from I24-3D trajectories.

Each field estimates local mean speed v(x, t) on a regular grid of time and
longitudinal position. Native 10 Hz trajectory samples are assigned to grid
cells, their speed magnitudes are averaged, and the sample-weighted numerator
and denominator can be Gaussian-smoothed. Cells outside the finite smoothing
support of an observation remain blank.

The script writes two figures matching the layouts of
``plot_i24_time_space.py``:

1. Eastbound and westbound speed fields for scenes 1-3.
2. Scene 3 westbound speed fields together and by predominant lane.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

if __package__:
    from .i24_3d_to_spiced import SAMPLE_HZ, normalize_scene
    from .plot_i24_time_space import (
        LANE_DESCRIPTORS,
        SPEED_COLORMAP,
        clean_objects,
        lane_centers,
        predominant_lane,
        time_bounds,
    )
else:
    from i24_3d_to_spiced import SAMPLE_HZ, normalize_scene
    from plot_i24_time_space import (
        LANE_DESCRIPTORS,
        SPEED_COLORMAP,
        clean_objects,
        lane_centers,
        predominant_lane,
        time_bounds,
    )


def regular_edges(lower: float, upper: float, step: float) -> np.ndarray:
    """Return step-aligned bin edges that contain the closed input interval."""
    if step <= 0:
        raise ValueError("Grid step must be positive")
    start = math.floor(lower / step) * step
    stop = math.ceil(upper / step) * step
    if stop <= start:
        stop = start + step
    count = round((stop - start) / step)
    return start + np.arange(count + 1, dtype=float) * step


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    """Return a normalized one-dimensional kernel truncated at three sigma."""
    if sigma_bins <= 0:
        return np.ones(1, dtype=float)
    radius = max(1, math.ceil(3.0 * sigma_bins))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    return kernel / kernel.sum()


def _convolve_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve one array axis while preserving its exact shape."""
    if len(kernel) == 1:
        return values.copy()
    radius = len(kernel) // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="constant")
    return np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="valid"), axis, padded
    )


def _smooth_grid(
    values: np.ndarray,
    time_sigma_bins: float,
    space_sigma_bins: float,
) -> np.ndarray:
    """Apply a separable Gaussian filter to a [space, time] array."""
    smoothed = _convolve_axis(values, _gaussian_kernel(space_sigma_bins), axis=0)
    return _convolve_axis(smoothed, _gaussian_kernel(time_sigma_bins), axis=1)


def build_speed_field(
    objects: list[dict[str, Any]],
    scene_start_tick: int,
    time_edges: np.ndarray,
    position_edges: np.ndarray,
    min_samples: int,
    smooth_time_seconds: float,
    smooth_space_meters: float,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Bin trajectory samples and return mean speed plus raw sample counts.

    The returned arrays are indexed ``[position_bin, time_bin]``. Smoothing is
    sample-weighted: speed sums and counts are filtered independently and then
    divided, so a lightly observed cell cannot outweigh a densely observed one.
    """
    shape = (len(position_edges) - 1, len(time_edges) - 1)
    if not objects:
        return np.ma.masked_all(shape, dtype=float), np.zeros(shape, dtype=float)

    times: list[float] = []
    positions: list[float] = []
    speeds: list[float] = []
    for obj in objects:
        start_tick = int(obj["start_tick"])
        for index, (position, velocity) in enumerate(
            zip(obj["position_m"], obj["velocity_mps"])
        ):
            times.append((start_tick + index - scene_start_tick) / SAMPLE_HZ)
            positions.append(float(position[0]))
            speeds.append(math.hypot(float(velocity[0]), float(velocity[1])))

    sample_count, _, _ = np.histogram2d(
        positions, times, bins=(position_edges, time_edges)
    )
    speed_sum, _, _ = np.histogram2d(
        positions, times, bins=(position_edges, time_edges), weights=speeds
    )

    eligible = sample_count >= min_samples
    numerator = np.where(eligible, speed_sum, 0.0)
    denominator = np.where(eligible, sample_count, 0.0)
    time_bin_seconds = float(time_edges[1] - time_edges[0])
    space_bin_meters = float(position_edges[1] - position_edges[0])
    numerator = _smooth_grid(
        numerator,
        smooth_time_seconds / time_bin_seconds,
        smooth_space_meters / space_bin_meters,
    )
    denominator = _smooth_grid(
        denominator,
        smooth_time_seconds / time_bin_seconds,
        smooth_space_meters / space_bin_meters,
    )

    field = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).eps,
    )
    masked = np.ma.masked_where(denominator <= np.finfo(float).eps, field)
    return masked, sample_count


def _scene_grid(
    objects: list[dict[str, Any]],
    first_tick: int,
    last_tick: int,
    time_bin_seconds: float,
    space_bin_meters: float,
) -> tuple[np.ndarray, np.ndarray]:
    duration = (last_tick - first_tick) / SAMPLE_HZ
    all_x = [float(point[0]) for obj in objects for point in obj["position_m"]]
    return (
        regular_edges(0.0, duration, time_bin_seconds),
        regular_edges(min(all_x), max(all_x), space_bin_meters),
    )


def _draw_field(
    axis: Any,
    objects: list[dict[str, Any]],
    first_tick: int,
    time_edges: np.ndarray,
    position_edges: np.ndarray,
    normalization: Normalize,
    min_samples: int,
    smooth_time_seconds: float,
    smooth_space_meters: float,
) -> float:
    field, _ = build_speed_field(
        objects,
        first_tick,
        time_edges,
        position_edges,
        min_samples,
        smooth_time_seconds,
        smooth_space_meters,
    )
    axis.pcolormesh(
        time_edges,
        position_edges,
        field,
        cmap=SPEED_COLORMAP,
        norm=normalization,
        shading="flat",
        rasterized=True,
    )
    axis.set_xlim(time_edges[0], time_edges[-1])
    axis.set_ylim(position_edges[0], position_edges[-1])
    axis.set_xlabel("Time from scene start (s)")
    axis.set_ylabel("Longitudinal position, x (m)")
    observed = np.ma.count(field)
    return observed / field.size if field.size else 0.0


def _add_colorbar(figure: Any, normalization: Normalize, speed_max_mps: float) -> None:
    mappable = ScalarMappable(norm=normalization, cmap=SPEED_COLORMAP)
    mappable.set_array([])
    colorbar = figure.colorbar(
        mappable,
        ax=figure.axes,
        orientation="horizontal",
        fraction=0.025,
        pad=0.055,
        aspect=45,
    )
    colorbar.set_ticks([0.0, speed_max_mps / 2.0, speed_max_mps])
    colorbar.set_ticklabels(["0", f"{speed_max_mps / 2:g}", f"≥{speed_max_mps:g}"])
    colorbar.set_label("Mean local speed (m/s)")


def _method_text(args: argparse.Namespace) -> str:
    smoothing = (
        f"Gaussian σ=({args.smooth_time_seconds:g} s, "
        f"{args.smooth_space_meters:g} m)"
        if args.smooth_time_seconds > 0 or args.smooth_space_meters > 0
        else "no smoothing"
    )
    return (
        f"Mean of 10 Hz samples · {args.time_bin_seconds:g} s × "
        f"{args.space_bin_meters:g} m cells · {smoothing}"
    )


def plot_speed_fields_by_direction(
    scenes: list[dict[str, Any]], output_path: Path, args: argparse.Namespace
) -> None:
    """Plot eastbound and westbound speed fields for scenes 1-3."""
    rows = []
    for scene in scenes:
        objects = clean_objects(scene, args.include_major_flags)
        first_tick, last_tick = time_bounds(objects)
        time_edges, position_edges = _scene_grid(
            objects,
            first_tick,
            last_tick,
            args.time_bin_seconds,
            args.space_bin_meters,
        )
        rows.append((scene, objects, first_tick, time_edges, position_edges))

    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.3))
    figure.subplots_adjust(
        left=0.072,
        right=0.985,
        bottom=0.13,
        top=0.895,
        hspace=0.42,
        wspace=0.16,
    )
    figure.suptitle(
        "I-24-3D longitudinal speed fields by travel direction",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.94,
        _method_text(args),
        ha="center",
        fontsize=10.5,
        color="#555555",
    )

    normalization = Normalize(vmin=0.0, vmax=args.speed_max_mps, clip=True)
    direction_labels = {1: "Eastbound (+x)", -1: "Westbound (−x)"}
    for row_index, row in enumerate(rows):
        scene, objects, first_tick, time_edges, position_edges = row
        for column_index, direction in enumerate((1, -1)):
            axis = axes[row_index, column_index]
            subset = [obj for obj in objects if int(obj["direction"]) == direction]
            coverage = _draw_field(
                axis,
                subset,
                first_tick,
                time_edges,
                position_edges,
                normalization,
                args.min_samples,
                args.smooth_time_seconds,
                args.smooth_space_meters,
            )
            axis.set_title(
                f"Scene {scene['scene']} · {direction_labels[direction]} · "
                f"{len(subset)} trajectories · {coverage:.0%} support",
                loc="left",
                pad=7,
                fontweight="semibold",
            )

    _add_colorbar(figure, normalization, args.speed_max_mps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def plot_scene3_westbound_speed_fields_by_lane(
    scene: dict[str, Any], output_path: Path, args: argparse.Namespace
) -> None:
    """Plot the combined Scene 3 westbound field and four lane fields."""
    objects = clean_objects(scene, args.include_major_flags)
    westbound = [obj for obj in objects if int(obj["direction"]) == -1]
    first_tick, last_tick = time_bounds(objects)
    time_edges, position_edges = _scene_grid(
        westbound,
        first_tick,
        last_tick,
        args.time_bin_seconds,
        args.space_bin_meters,
    )
    centers = lane_centers(scene, direction=-1)
    if len(centers) != 4:
        raise ValueError(f"Expected four Scene 3 westbound lanes, found {len(centers)}")
    lane_of = {int(obj["id"]): predominant_lane(obj, centers) for obj in westbound}
    assigned = {
        lane: [obj for obj in westbound if lane_of[int(obj["id"])] == lane]
        for lane in range(1, 5)
    }

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
        bottom=0.13,
        top=0.88,
    )
    combined_axis = figure.add_subplot(grid[0, :])
    lane_axes = {
        1: figure.add_subplot(grid[1, 0]),
        2: figure.add_subplot(grid[1, 1]),
        3: figure.add_subplot(grid[2, 0]),
        4: figure.add_subplot(grid[2, 1]),
    }
    figure.suptitle(
        "I-24-3D Scene 3 westbound longitudinal speed fields",
        fontsize=17,
        fontweight="bold",
        y=0.972,
    )
    figure.text(
        0.5,
        0.935,
        _method_text(args),
        ha="center",
        fontsize=10.5,
        color="#555555",
    )

    normalization = Normalize(vmin=0.0, vmax=args.speed_max_mps, clip=True)
    coverage = _draw_field(
        combined_axis,
        westbound,
        first_tick,
        time_edges,
        position_edges,
        normalization,
        args.min_samples,
        args.smooth_time_seconds,
        args.smooth_space_meters,
    )
    combined_axis.set_title(
        f"All westbound lanes · {len(westbound)} trajectories · {coverage:.0%} support",
        loc="left",
        pad=7,
        fontweight="semibold",
    )

    for lane, axis in lane_axes.items():
        coverage = _draw_field(
            axis,
            assigned[lane],
            first_tick,
            time_edges,
            position_edges,
            normalization,
            args.min_samples,
            args.smooth_time_seconds,
            args.smooth_space_meters,
        )
        axis.set_title(
            f"Lane {lane} ({LANE_DESCRIPTORS[lane]}) · "
            f"{len(assigned[lane])} trajectories · {coverage:.0%} support",
            loc="left",
            pad=7,
            fontweight="semibold",
        )

    _add_colorbar(figure, normalization, args.speed_max_mps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
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
        default=repo_root / "figures" / "i24_speed_field",
    )
    argument_parser.add_argument("--include-major-flags", action="store_true")
    argument_parser.add_argument("--speed-max-mps", type=float, default=35.0)
    argument_parser.add_argument("--time-bin-seconds", type=float, default=1.0)
    argument_parser.add_argument("--space-bin-meters", type=float, default=15.0)
    argument_parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum raw samples in a cell before it contributes to smoothing",
    )
    argument_parser.add_argument("--smooth-time-seconds", type=float, default=1.5)
    argument_parser.add_argument("--smooth-space-meters", type=float, default=30.0)
    argument_parser.add_argument("--dpi", type=int, default=150)
    return argument_parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--speed-max-mps": args.speed_max_mps,
        "--time-bin-seconds": args.time_bin_seconds,
        "--space-bin-meters": args.space_bin_meters,
        "--dpi": args.dpi,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    if args.min_samples < 1:
        raise SystemExit("--min-samples must be at least 1")
    if args.smooth_time_seconds < 0 or args.smooth_space_meters < 0:
        raise SystemExit("Smoothing scales cannot be negative")


def main() -> None:
    args = parser().parse_args()
    _validate_args(args)
    scenes = [
        normalize_scene(args.i24_root.resolve(), scene, stabilize_headings=False)
        for scene in (1, 2, 3)
    ]
    direction_path = args.output_dir / "i24_speed_fields_by_direction.png"
    lane_path = args.output_dir / "i24_scene3_westbound_speed_fields_by_lane.png"
    plot_speed_fields_by_direction(scenes, direction_path, args)
    plot_scene3_westbound_speed_fields_by_lane(scenes[2], lane_path, args)
    print(f"Wrote {direction_path}")
    print(f"Wrote {lane_path}")


if __name__ == "__main__":
    main()
