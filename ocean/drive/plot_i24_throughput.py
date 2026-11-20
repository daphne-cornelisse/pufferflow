"""Visualize total-roadway throughput from I24-3D trajectory crossings.

The high-fidelity view counts exact, interpolated vehicle crossings at a dense
array of virtual cross-sections. The ideal-loop view uses three perfect
cross-sectional detector stations per direction and aggregates their identical
crossing events into coarser time intervals. Neither measurement requires lane
assignment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if __package__:
    from .i24_3d_to_spiced import SAMPLE_HZ, normalize_scene
    from .plot_i24_time_space import clean_objects, configure_matplotlib, time_bounds
else:
    from i24_3d_to_spiced import SAMPLE_HZ, normalize_scene
    from plot_i24_time_space import clean_objects, configure_matplotlib, time_bounds


DIRECTION_LABELS = {1: "Eastbound", -1: "Westbound"}
STATION_COLORS = ("#5B4B8A", "#008C8C", "#D95F02")
TRUTH_COLOR = "#5B4B8A"
LOOP_COLOR = "#008C8C"


class CrossingEvent(NamedTuple):
    scene: int
    direction: int
    gate_x_m: float
    vehicle_id: int
    time_s: float


def trajectory_crossing_time_tick(
    obj: dict[str, Any], gate_x_m: float
) -> float | None:
    """Return the first direction-aligned crossing time, interpolated in ticks."""
    direction = int(obj["direction"])
    x_m = np.asarray(obj["position_m"], dtype=float)[:, 0]
    aligned_position = direction * x_m
    aligned_gate = direction * gate_x_m
    increments = np.diff(aligned_position)
    crossing_indices = np.flatnonzero(
        (aligned_position[:-1] < aligned_gate)
        & (aligned_position[1:] >= aligned_gate)
        & (increments > 0.0)
    )
    if not len(crossing_indices):
        return None
    index = int(crossing_indices[0])
    fraction = (aligned_gate - aligned_position[index]) / increments[index]
    return float(obj["start_tick"]) + index + float(fraction)


def dense_gate_positions(scene: dict[str, Any], spacing_m: float) -> np.ndarray:
    """Return regularly spaced cross-sections inside the calibrated x range."""
    lower, upper = map(float, scene["map"]["x_range_m"])
    first = math.ceil(lower / spacing_m) * spacing_m
    last = math.floor(upper / spacing_m) * spacing_m
    gates = np.arange(first, last + 0.5 * spacing_m, spacing_m)
    if len(gates) < 3:
        raise ValueError("Dense-gate spacing leaves fewer than three cross-sections")
    return gates


def ideal_loop_stations(
    scene: dict[str, Any], direction: int
) -> list[tuple[str, float]]:
    """Place upstream, middle, and downstream stations along travel direction."""
    lower, upper = map(float, scene["map"]["x_range_m"])
    physical_positions = [lower + fraction * (upper - lower) for fraction in (0.25, 0.5, 0.75)]
    if direction == -1:
        physical_positions.reverse()
    return list(zip(("Upstream", "Middle", "Downstream"), physical_positions))


def collect_crossing_events(
    scene: dict[str, Any],
    direction: int,
    gates_x_m: list[float] | np.ndarray,
    include_major_flags: bool,
    scene_start_tick: int,
) -> list[CrossingEvent]:
    """Collect one event per observed vehicle crossing per gate."""
    objects = [
        obj
        for obj in clean_objects(scene, include_major_flags)
        if int(obj["direction"]) == direction
    ]
    events: list[CrossingEvent] = []
    for obj in objects:
        for gate_x_m in gates_x_m:
            crossing_tick = trajectory_crossing_time_tick(obj, float(gate_x_m))
            if crossing_tick is None:
                continue
            events.append(
                CrossingEvent(
                    scene=int(scene["scene"]),
                    direction=direction,
                    gate_x_m=float(gate_x_m),
                    vehicle_id=int(obj["id"]),
                    time_s=(crossing_tick - scene_start_tick) / SAMPLE_HZ,
                )
            )
    events.sort(key=lambda event: (event.time_s, event.gate_x_m, event.vehicle_id))
    return events


def time_bin_edges(duration_s: float, interval_s: float) -> np.ndarray:
    """Return interval edges, preserving a possibly partial final interval."""
    edges = np.arange(0.0, duration_s, interval_s)
    if not len(edges) or edges[0] != 0.0:
        edges = np.insert(edges, 0, 0.0)
    if duration_s - edges[-1] > 1e-9:
        edges = np.append(edges, duration_s)
    if len(edges) == 1:
        edges = np.append(edges, duration_s)
    return edges


def complete_time_bin_edges(duration_s: float, interval_s: float) -> np.ndarray:
    """Return only complete fixed-duration intervals."""
    complete_interval_count = math.floor((duration_s + 1e-9) / interval_s)
    if complete_interval_count < 1:
        return np.asarray([0.0, duration_s])
    return np.arange(complete_interval_count + 1, dtype=float) * interval_s


def events_at_gate(events: list[CrossingEvent], gate_x_m: float) -> np.ndarray:
    """Return sorted crossing times at one gate."""
    return np.asarray(
        sorted(
            event.time_s
            for event in events
            if math.isclose(event.gate_x_m, gate_x_m, abs_tol=1e-7)
        ),
        dtype=float,
    )


def binned_gate_counts(
    events: list[CrossingEvent], gates_x_m: np.ndarray, edges_s: np.ndarray
) -> np.ndarray:
    """Return time-bin by gate counts."""
    counts = np.zeros((len(edges_s) - 1, len(gates_x_m)), dtype=int)
    for gate_index, gate_x_m in enumerate(gates_x_m):
        times = events_at_gate(events, float(gate_x_m))
        counts[:, gate_index] = np.histogram(times, bins=edges_s)[0]
    return counts


def position_edges(positions_m: np.ndarray) -> np.ndarray:
    """Convert cross-section centers to plotting-cell edges."""
    midpoints = 0.5 * (positions_m[:-1] + positions_m[1:])
    return np.concatenate(
        (
            [positions_m[0] - (midpoints[0] - positions_m[0])],
            midpoints,
            [positions_m[-1] + (positions_m[-1] - midpoints[-1])],
        )
    )


def build_analysis(
    scenes: list[dict[str, Any]],
    include_major_flags: bool,
    dense_spacing_m: float,
) -> list[dict[str, Any]]:
    """Build crossing-event datasets for all scene-direction groups."""
    analyses: list[dict[str, Any]] = []
    for scene in scenes:
        objects = clean_objects(scene, include_major_flags)
        first_tick, last_tick = time_bounds(objects)
        duration_s = (last_tick - first_tick) / SAMPLE_HZ
        dense_gates = dense_gate_positions(scene, dense_spacing_m)
        for direction in (1, -1):
            stations = ideal_loop_stations(scene, direction)
            dense_events = collect_crossing_events(
                scene,
                direction,
                dense_gates,
                include_major_flags,
                first_tick,
            )
            station_events = collect_crossing_events(
                scene,
                direction,
                [position for _, position in stations],
                include_major_flags,
                first_tick,
            )
            analyses.append(
                {
                    "scene": int(scene["scene"]),
                    "direction": direction,
                    "duration_s": duration_s,
                    "dense_gates_x_m": dense_gates,
                    "dense_events": dense_events,
                    "stations": stations,
                    "station_events": station_events,
                }
            )
    return analyses


def _group_title(analysis: dict[str, Any]) -> str:
    return f"Scene {analysis['scene']} · {DIRECTION_LABELS[analysis['direction']]}"


def plot_truth_flow_heatmap(
    analyses: list[dict[str, Any]],
    output_path: Path,
    truth_interval_s: float,
    flow_max_vph: float | None,
    dpi: int,
) -> None:
    """Plot dense-gate trajectory-truth flow over position and time."""
    fields: list[tuple[np.ndarray, np.ndarray]] = []
    all_positive: list[float] = []
    for analysis in analyses:
        edges_s = complete_time_bin_edges(
            analysis["duration_s"], truth_interval_s
        )
        counts = binned_gate_counts(
            analysis["dense_events"], analysis["dense_gates_x_m"], edges_s
        )
        flow_vph = counts / np.diff(edges_s)[:, None] * 3600.0
        fields.append((edges_s, flow_vph))
        all_positive.extend(flow_vph[flow_vph > 0.0].tolist())
    if flow_max_vph is None:
        percentile = float(np.percentile(all_positive, 99)) if all_positive else 1.0
        flow_max_vph = max(500.0, math.ceil(percentile / 500.0) * 500.0)
    normalization = Normalize(vmin=0.0, vmax=flow_max_vph, clip=True)
    colormap = matplotlib.colormaps["viridis"]

    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.4))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.88, hspace=0.42, wspace=0.16)
    figure.suptitle("I-24-3D trajectory-truth total-roadway flow field", fontsize=17, fontweight="bold", y=0.975)
    figure.text(
        0.5,
        0.94,
        f"Exact vehicle crossings · gates every {analyses[0]['dense_gates_x_m'][1] - analyses[0]['dense_gates_x_m'][0]:g} m · "
        f"complete {truth_interval_s:g} s flow intervals · no lane assignment",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    for axis, analysis, (edges_s, flow_vph) in zip(axes.flat, analyses, fields):
        gates = analysis["dense_gates_x_m"]
        mesh = axis.pcolormesh(
            position_edges(gates),
            edges_s,
            flow_vph,
            cmap=colormap,
            norm=normalization,
            shading="flat",
            rasterized=True,
        )
        del mesh
        for role, position in analysis["stations"]:
            axis.axvline(position, color="white", linewidth=0.75, alpha=0.72)
            axis.text(position, 0.985, role[0], transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color="white")
        axis.set_title(_group_title(analysis), loc="left", fontweight="semibold")
        axis.set_xlabel("Longitudinal position, x (m)")
        axis.set_ylabel("Time from scene start (s)")
    colorbar_axis = figure.add_axes((0.33, 0.035, 0.34, 0.018))
    colorbar = figure.colorbar(ScalarMappable(norm=normalization, cmap=colormap), cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Interval flow rate (vehicles/hour, all lanes)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _cumulative_step(times_s: np.ndarray, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(([0.0], times_s, [duration_s]))
    y = np.concatenate(([0], np.arange(1, len(times_s) + 1), [len(times_s)]))
    return x, y


def plot_cumulative_counts(
    analyses: list[dict[str, Any]], output_path: Path, dpi: int
) -> None:
    """Plot cumulative exact counts at the three ideal-loop locations."""
    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.3))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.89, hspace=0.40, wspace=0.17)
    figure.suptitle("I-24-3D cumulative total-roadway counts", fontsize=17, fontweight="bold", y=0.976)
    figure.text(0.5, 0.94, "Exact trajectory crossings at the three ideal-loop stations", ha="center", fontsize=10.5, color="#555555")
    for axis, analysis in zip(axes.flat, analyses):
        for station_index, (role, position) in enumerate(analysis["stations"]):
            times = events_at_gate(analysis["station_events"], position)
            x, y = _cumulative_step(times, analysis["duration_s"])
            axis.step(
                x,
                y,
                where="post",
                color=STATION_COLORS[station_index],
                linewidth=1.8,
                label=f"{role} · x={position:.0f} m · total={len(times)}",
            )
        axis.set_xlim(0.0, analysis["duration_s"])
        axis.set_ylim(bottom=0.0)
        axis.set_title(_group_title(analysis), loc="left", fontweight="semibold")
        axis.set_xlabel("Time from scene start (s)")
        axis.set_ylabel("Cumulative vehicles")
        axis.grid(color="#DDDDDD", linewidth=0.5)
        axis.legend(loc="upper left", frameon=False, fontsize=7.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_ideal_loop_counts(
    analyses: list[dict[str, Any]],
    output_path: Path,
    loop_interval_s: float,
    dpi: int,
) -> None:
    """Plot ideal-loop raw vehicle counts in coarse intervals."""
    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.3))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.89, hspace=0.40, wspace=0.17)
    figure.suptitle("I-24-3D ideal-loop interval counts", fontsize=17, fontweight="bold", y=0.976)
    figure.text(
        0.5,
        0.94,
        f"Perfect cross-sectional detection · complete {loop_interval_s:g} s intervals · raw counts, all lanes",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    for axis, analysis in zip(axes.flat, analyses):
        edges = complete_time_bin_edges(
            analysis["duration_s"], loop_interval_s
        )
        for station_index, (role, position) in enumerate(analysis["stations"]):
            times = events_at_gate(analysis["station_events"], position)
            counts = np.histogram(times, bins=edges)[0]
            axis.step(
                edges,
                np.append(counts, counts[-1]),
                where="post",
                color=STATION_COLORS[station_index],
                linewidth=1.9,
                label=f"{role} · x={position:.0f} m",
            )
            axis.plot(
                0.5 * (edges[:-1] + edges[1:]),
                counts,
                linestyle="none",
                marker="o",
                markersize=3.2,
                color=STATION_COLORS[station_index],
            )
        axis.set_xlim(0.0, edges[-1])
        axis.set_ylim(bottom=0.0)
        axis.set_title(_group_title(analysis), loc="left", fontweight="semibold")
        axis.set_xlabel("Time from scene start (s)")
        axis.set_ylabel(f"Vehicles per {loop_interval_s:g} s interval")
        axis.grid(color="#DDDDDD", linewidth=0.5)
        axis.legend(loc="upper right", frameon=False, fontsize=7.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_truth_vs_loop(
    analyses: list[dict[str, Any]],
    output_path: Path,
    truth_interval_s: float,
    loop_interval_s: float,
    dpi: int,
) -> None:
    """Compare fine truth and coarse ideal-loop rates at the middle station."""
    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.3))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.89, hspace=0.40, wspace=0.17)
    figure.suptitle("I-24-3D trajectory truth versus ideal-loop aggregation", fontsize=17, fontweight="bold", y=0.976)
    figure.text(
        0.5,
        0.94,
        f"Same exact middle-station crossings over complete {loop_interval_s:g} s windows; only temporal resolution differs",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    for axis, analysis in zip(axes.flat, analyses):
        _, middle_position = analysis["stations"][1]
        times = events_at_gate(analysis["station_events"], middle_position)
        loop_edges = complete_time_bin_edges(
            analysis["duration_s"], loop_interval_s
        )
        comparison_duration_s = float(loop_edges[-1])
        truth_edges = complete_time_bin_edges(
            comparison_duration_s, truth_interval_s
        )
        truth_counts = np.histogram(times, bins=truth_edges)[0]
        loop_counts = np.histogram(times, bins=loop_edges)[0]
        truth_rates = truth_counts / np.diff(truth_edges) * 3600.0
        loop_rates = loop_counts / np.diff(loop_edges) * 3600.0
        axis.bar(
            0.5 * (truth_edges[:-1] + truth_edges[1:]),
            truth_rates,
            width=0.78 * np.diff(truth_edges),
            color=TRUTH_COLOR,
            alpha=0.44,
            label=f"Trajectory truth · {truth_interval_s:g} s",
        )
        axis.step(
            loop_edges,
            np.append(loop_rates, loop_rates[-1]),
            where="post",
            color=LOOP_COLOR,
            linewidth=2.1,
            label=f"Ideal loop · {loop_interval_s:g} s",
        )
        axis.set_xlim(0.0, comparison_duration_s)
        axis.set_ylim(bottom=0.0)
        axis.set_title(f"{_group_title(analysis)} · middle x={middle_position:.0f} m", loc="left", fontweight="semibold")
        axis.set_xlabel("Time from scene start (s)")
        axis.set_ylabel("Annualized interval flow (vehicles/hour)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.legend(loc="upper right", frameon=False, fontsize=7.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_segment_accumulation(
    analyses: list[dict[str, Any]], output_path: Path, dpi: int
) -> None:
    """Plot cumulative entering minus leaving counts between outer stations."""
    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.3))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.89, hspace=0.40, wspace=0.17)
    figure.suptitle("I-24-3D observed segment accumulation", fontsize=17, fontweight="bold", y=0.976)
    figure.text(
        0.5,
        0.94,
        "Upstream cumulative count minus downstream cumulative count · change since scene start",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    for axis, analysis in zip(axes.flat, analyses):
        upstream_position = analysis["stations"][0][1]
        downstream_position = analysis["stations"][2][1]
        upstream_times = events_at_gate(analysis["station_events"], upstream_position)
        downstream_times = events_at_gate(analysis["station_events"], downstream_position)
        grid_s = np.arange(0.0, analysis["duration_s"] + 1.0, 1.0)
        grid_s[-1] = min(grid_s[-1], analysis["duration_s"])
        accumulation = np.searchsorted(upstream_times, grid_s, side="right") - np.searchsorted(downstream_times, grid_s, side="right")
        axis.step(grid_s, accumulation, where="post", color=LOOP_COLOR, linewidth=1.9)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.fill_between(grid_s, 0.0, accumulation, step="post", color=LOOP_COLOR, alpha=0.16)
        axis.set_xlim(0.0, analysis["duration_s"])
        axis.set_title(_group_title(analysis), loc="left", fontweight="semibold")
        axis.set_xlabel("Time from scene start (s)")
        axis.set_ylabel("Net vehicles accumulated")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def analysis_manifest(
    analyses: list[dict[str, Any]],
    truth_interval_s: float,
    loop_interval_s: float,
    dense_spacing_m: float,
    include_major_flags: bool,
) -> dict[str, Any]:
    """Return a JSON-ready summary of stations, counts, and accumulation."""
    groups = []
    for analysis in analyses:
        loop_edges = complete_time_bin_edges(
            analysis["duration_s"], loop_interval_s
        )
        stations = []
        for role, position in analysis["stations"]:
            times = events_at_gate(analysis["station_events"], position)
            stations.append(
                {
                    "role": role,
                    "x_m": position,
                    "total_crossings": len(times),
                    "mean_flow_vph_over_scene": len(times) / analysis["duration_s"] * 3600.0,
                    "interval_edges_seconds": loop_edges.tolist(),
                    "interval_counts": np.histogram(times, bins=loop_edges)[0].tolist(),
                }
            )
        groups.append(
            {
                "scene": analysis["scene"],
                "direction": analysis["direction"],
                "duration_seconds": analysis["duration_s"],
                "dense_gate_count": len(analysis["dense_gates_x_m"]),
                "dense_gate_positions_m": analysis["dense_gates_x_m"].tolist(),
                "stations": stations,
                "net_accumulation_at_end": stations[0]["total_crossings"] - stations[2]["total_crossings"],
            }
        )
    return {
        "method": {
            "measurement": "first direction-aligned rear-center crossing, linearly interpolated between 10 Hz samples",
            "roadway_scope": "all lanes combined; no lane assignment",
            "dense_gate_spacing_meters": dense_spacing_m,
            "trajectory_truth_interval_seconds": truth_interval_s,
            "ideal_loop_interval_seconds": loop_interval_s,
            "ideal_loop_positions": "25%, 50%, and 75% of calibrated x range, ordered by travel direction",
            "major_quality_flags_included": include_major_flags,
        },
        "groups": groups,
    }


def parser() -> argparse.ArgumentParser:
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
        default=repo_root / "figures" / "i24_throughput",
    )
    argument_parser.add_argument("--include-major-flags", action="store_true")
    argument_parser.add_argument("--dense-gate-spacing-meters", type=float, default=25.0)
    argument_parser.add_argument("--truth-interval-seconds", type=float, default=10.0)
    argument_parser.add_argument("--loop-interval-seconds", type=float, default=30.0)
    argument_parser.add_argument("--flow-max-vph", type=float, default=0.0)
    argument_parser.add_argument("--dpi", type=int, default=150)
    return argument_parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "dense_gate_spacing_meters",
        "truth_interval_seconds",
        "loop_interval_seconds",
        "dpi",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.flow_max_vph < 0:
        raise SystemExit("--flow-max-vph cannot be negative")


def main() -> None:
    args = parser().parse_args()
    _validate_args(args)
    configure_matplotlib()
    scenes = [
        normalize_scene(args.i24_root.resolve(), scene, stabilize_headings=False)
        for scene in (1, 2, 3)
    ]
    analyses = build_analysis(
        scenes, args.include_major_flags, args.dense_gate_spacing_meters
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.output_dir / "i24_trajectory_truth_flow_heatmap.png"
    cumulative_path = args.output_dir / "i24_cumulative_counts.png"
    loop_path = args.output_dir / "i24_ideal_loop_counts.png"
    comparison_path = args.output_dir / "i24_truth_vs_ideal_loop.png"
    accumulation_path = args.output_dir / "i24_segment_accumulation.png"
    manifest_path = args.output_dir / "i24_throughput_analysis.json"

    plot_truth_flow_heatmap(
        analyses,
        heatmap_path,
        args.truth_interval_seconds,
        args.flow_max_vph or None,
        args.dpi,
    )
    plot_cumulative_counts(analyses, cumulative_path, args.dpi)
    plot_ideal_loop_counts(
        analyses, loop_path, args.loop_interval_seconds, args.dpi
    )
    plot_truth_vs_loop(
        analyses,
        comparison_path,
        args.truth_interval_seconds,
        args.loop_interval_seconds,
        args.dpi,
    )
    plot_segment_accumulation(analyses, accumulation_path, args.dpi)
    with manifest_path.open("w") as file:
        json.dump(
            analysis_manifest(
                analyses,
                args.truth_interval_seconds,
                args.loop_interval_seconds,
                args.dense_gate_spacing_meters,
                args.include_major_flags,
            ),
            file,
            indent=2,
        )
        file.write("\n")
    for path in (
        heatmap_path,
        cumulative_path,
        loop_path,
        comparison_path,
        accumulation_path,
        manifest_path,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
