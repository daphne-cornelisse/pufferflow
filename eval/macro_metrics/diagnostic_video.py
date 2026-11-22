#!/usr/bin/env python3
"""Render a fixed-road I-24 rollout video with matched ground-truth ghosts."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.macro_metrics.rollout import (
    ground_truth_arrays,
    load_scenario,
    simulate_rollout,
)
from eval.macro_metrics.runtime import (
    I24_POLICY_HIDDEN,
    I24_POLICY_LAYERS,
    DriveRuntime,
    PolicySession,
)

SAMPLE_HZ = 10.0
SIM_BLUE = "#2f6da8"
SIM_BLUE_EDGE = "#174a78"
GT_GREEN = "#2a9d68"
ROAD_FILL = "#e7e7e7"
ROAD_EDGE = "#555555"
ROAD_LINE = "#ffffff"


def oriented_box_vertices(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Return four corners per oriented vehicle box, shape ``(n, 4, 2)``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    half_length = np.asarray(length, dtype=np.float64) * (0.5 * scale)
    half_width = np.asarray(width, dtype=np.float64) * (0.5 * scale)
    forward = np.stack((np.cos(heading), np.sin(heading)), axis=1)
    lateral = np.stack((-np.sin(heading), np.cos(heading)), axis=1)
    center = np.stack((x, y), axis=1)
    return np.stack(
        (
            center + half_length[:, None] * forward + half_width[:, None] * lateral,
            center + half_length[:, None] * forward - half_width[:, None] * lateral,
            center - half_length[:, None] * forward - half_width[:, None] * lateral,
            center - half_length[:, None] * forward + half_width[:, None] * lateral,
        ),
        axis=1,
    )


def _ensure_runtime(library: Path, policy_act: Path) -> None:
    if not library.is_file() or not policy_act.is_file():
        subprocess.run([str(ROOT / "eval" / "macro_metrics" / "build_runtime.sh")], check=True)
    if not library.is_file() or not policy_act.is_file():
        raise FileNotFoundError("Macro runtime build did not produce both required files")


def capture_rollout(
    *,
    scenario_path: Path,
    data_dir: Path,
    map_id: int,
    checkpoint: Path,
    target_rollout: int,
    seed: int,
    output_path: Path,
    library: Path,
    policy_act: Path,
    hidden: int,
    layers: int,
) -> Path:
    """Reproduce one independently seeded rollout and save its trajectory."""
    if target_rollout < 0:
        raise ValueError("target_rollout must be nonnegative")
    _ensure_runtime(library, policy_act)
    scenario = load_scenario(scenario_path)
    runtime = DriveRuntime(library)
    handle = runtime.create_scenario(data_dir, map_id, scenario_path)
    agents = int(runtime.lib.pf_n(handle))
    observation_size = int(runtime.lib.pf_obs_size(handle))
    rollout_seed = seed + target_rollout
    started = time.perf_counter()
    try:
        with PolicySession(
            policy_act,
            checkpoint,
            agents,
            observation_size,
            hidden=hidden,
            layers=layers,
            seed=rollout_seed,
        ) as policy:
            selected = simulate_rollout(
                runtime,
                handle,
                scenario,
                policy,
                seed=rollout_seed,
            )
            elapsed = time.perf_counter() - started
            print(
                f"rollout r{target_rollout:02d} complete in {elapsed:.1f} s",
                flush=True,
            )
    finally:
        runtime.lib.pf_close(handle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **selected,
        scene=np.int32(scenario["metadata"]["source_scene"]),
        rollout=np.int32(target_rollout),
        policy_seed=np.int32(rollout_seed),
        checkpoint=np.asarray(str(checkpoint)),
        policy_hidden_size=np.int32(hidden),
        policy_num_layers=np.int32(layers),
    )
    print(
        f"saved {output_path} after {time.perf_counter() - started:.1f} s",
        flush=True,
    )
    return output_path


def _direction_mask(trajectory: dict[str, np.ndarray], direction: int) -> np.ndarray:
    return np.asarray(trajectory["direction"]) == direction


def _road_bounds(
    scenario: dict[str, Any], ground_truth: dict[str, np.ndarray], direction: int
) -> tuple[float, float]:
    direction_rows = _direction_mask(ground_truth, direction)
    valid = ground_truth["valid"] & direction_rows[:, None]
    y = ground_truth["y"][valid]
    low_data = float(np.min(y))
    high_data = float(np.max(y))
    edges = sorted(
        float(np.median([point["y"] for point in road["geometry"]]))
        for road in scenario["roads"]
        if road["type"] == "road_edge"
    )
    lower = [value for value in edges if value <= low_data]
    upper = [value for value in edges if value >= high_data]
    low = max(lower) if lower else low_data - 1.5
    high = min(upper) if upper else high_data + 1.5
    return low - 0.6, high + 0.6


def _longitudinal_segments(
    ground_truth: dict[str, np.ndarray], direction: int, count: int = 3
) -> np.ndarray:
    direction_rows = _direction_mask(ground_truth, direction)
    valid = ground_truth["valid"] & direction_rows[:, None]
    x = ground_truth["x"][valid]
    low = math.floor(float(np.min(x)) / 50.0) * 50.0
    high = math.ceil(float(np.max(x)) / 50.0) * 50.0
    return np.linspace(low, high, count + 1)


def _draw_roads(
    axis: plt.Axes,
    scenario: dict[str, Any],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    axis.axhspan(y_bounds[0], y_bounds[1], color=ROAD_FILL, zorder=0)
    for road in scenario["roads"]:
        road_type = road["type"]
        if road_type not in ("road_line", "road_edge"):
            continue
        x = np.asarray([point["x"] for point in road["geometry"]], dtype=float)
        y = np.asarray([point["y"] for point in road["geometry"]], dtype=float)
        if np.nanmax(y) < y_bounds[0] or np.nanmin(y) > y_bounds[1]:
            continue
        if road_type == "road_edge":
            axis.plot(x, y, color=ROAD_EDGE, linewidth=1.4, zorder=1)
        else:
            axis.plot(
                x,
                y,
                color=ROAD_LINE,
                linewidth=1.3,
                linestyle=(0, (9, 7)),
                zorder=1,
            )
    axis.set_xlim(*x_bounds)
    axis.set_ylim(*y_bounds)
    axis.grid(axis="x", color="#b8b8b8", alpha=0.35, linewidth=0.6)
    axis.set_facecolor(ROAD_FILL)


def _trajectory_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def render_video(
    *,
    scenario_path: Path,
    trajectory_path: Path,
    output_path: Path,
    direction: int,
    frame_stride: int,
    fps: int,
    dpi: int,
    metric_label: str,
    ffmpeg: Path | None = None,
) -> Path:
    """Render three fixed roadway strips with simulation and matched GT ghosts."""
    if frame_stride < 1 or fps < 1:
        raise ValueError("frame_stride and fps must be positive")
    ffmpeg_path = str(ffmpeg) if ffmpeg is not None else shutil.which("ffmpeg")
    if ffmpeg_path is None or not Path(ffmpeg_path).is_file():
        raise FileNotFoundError("ffmpeg is required to encode the diagnostic video")
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    scenario = load_scenario(scenario_path)
    ground_truth = ground_truth_arrays(scenario)
    simulation = _trajectory_archive(trajectory_path)
    for key in ("ids", "direction", "x", "y", "heading", "valid", "length", "width"):
        if key not in simulation:
            raise ValueError(f"Trajectory archive is missing {key}")
    if simulation["x"].shape != ground_truth["x"].shape:
        raise ValueError("Simulation and ground truth use different agent/frame shapes")

    direction_name = "westbound" if direction == -1 else "eastbound"
    direction_rows = _direction_mask(ground_truth, direction)
    segment_edges = _longitudinal_segments(ground_truth, direction)
    y_bounds = _road_bounds(scenario, ground_truth, direction)
    # At the default 110 DPI this is 1584 × 880, satisfying yuv420p's
    # even-dimension requirement without an ffmpeg padding filter.
    figure, axes = plt.subplots(3, 1, figsize=(14.4, 8.0), sharey=True)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.86, hspace=0.34)
    sim_collections: list[PolyCollection] = []
    gt_collections: list[PolyCollection] = []
    for index, axis in enumerate(axes):
        bounds = (float(segment_edges[index]), float(segment_edges[index + 1]))
        _draw_roads(axis, scenario, bounds, y_bounds)
        sim_collection = PolyCollection(
            [], facecolors=SIM_BLUE, edgecolors=SIM_BLUE_EDGE,
            linewidths=0.65, alpha=0.82, zorder=3,
        )
        gt_collection = PolyCollection(
            [], facecolors="none", edgecolors=GT_GREEN,
            linewidths=1.25, alpha=0.95, zorder=4,
        )
        axis.add_collection(sim_collection)
        axis.add_collection(gt_collection)
        sim_collections.append(sim_collection)
        gt_collections.append(gt_collection)
        axis.set_title(
            f"Road segment {bounds[0]:.0f}–{bounds[1]:.0f} m",
            loc="left",
            fontsize=10,
            fontweight="semibold",
        )
        axis.set_ylabel("Lateral y (m)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Longitudinal position x (m)")

    checkpoint = str(np.asarray(simulation.get("checkpoint", "checkpoint")).item())
    rollout = int(np.asarray(simulation.get("rollout", 0)).item())
    figure.suptitle(
        f"I-24-3D Scene {scenario['metadata']['source_scene']} {direction_name} · "
        f"{Path(checkpoint).stem} · rollout r{rollout:02d}",
        fontsize=15,
        y=0.965,
    )
    clock = figure.text(
        0.5,
        0.91,
        "",
        ha="center",
        va="center",
        fontsize=11,
    )
    if metric_label:
        figure.text(0.985, 0.91, metric_label, ha="right", va="center", fontsize=9, color="#555555")
    figure.legend(
        handles=(
            Patch(facecolor=SIM_BLUE, edgecolor=SIM_BLUE_EDGE, label="Simulated vehicle"),
            Patch(facecolor="none", edgecolor=GT_GREEN, linewidth=1.5, label="Ground-truth ghost"),
        ),
        loc="upper left",
        bbox_to_anchor=(0.065, 0.925),
        frameon=False,
        ncol=2,
    )

    frames = simulation["x"].shape[1]
    frame_indices = list(range(0, frames, frame_stride))
    if frame_indices[-1] != frames - 1:
        frame_indices.append(frames - 1)
    length = np.asarray(simulation["length"], dtype=float)
    width = np.asarray(simulation["width"], dtype=float)

    def update(frame: int):
        sim_active = (
            direction_rows
            & simulation["valid"][:, frame].astype(bool)
            & np.isfinite(simulation["x"][:, frame])
            & np.isfinite(simulation["y"][:, frame])
            & np.isfinite(simulation["heading"][:, frame])
        )
        gt_active = (
            direction_rows
            & ground_truth["valid"][:, frame]
            & np.isfinite(ground_truth["x"][:, frame])
            & np.isfinite(ground_truth["y"][:, frame])
            & np.isfinite(ground_truth["heading"][:, frame])
        )
        sim_boxes = oriented_box_vertices(
            simulation["x"][sim_active, frame],
            simulation["y"][sim_active, frame],
            simulation["heading"][sim_active, frame],
            length[sim_active],
            width[sim_active],
        )
        gt_boxes = oriented_box_vertices(
            ground_truth["x"][gt_active, frame],
            ground_truth["y"][gt_active, frame],
            ground_truth["heading"][gt_active, frame],
            ground_truth["length"][gt_active],
            ground_truth["width"][gt_active],
            scale=1.08,
        )
        for collection in sim_collections:
            collection.set_verts(sim_boxes)
        for collection in gt_collections:
            collection.set_verts(gt_boxes)
        sim_in_view = (
            sim_active
            & (simulation["x"][:, frame] >= segment_edges[0])
            & (simulation["x"][:, frame] <= segment_edges[-1])
            & (simulation["y"][:, frame] >= y_bounds[0])
            & (simulation["y"][:, frame] <= y_bounds[1])
        )
        clock.set_text(
            f"Scene time {frame / SAMPLE_HZ:05.1f} s  ·  "
            f"active WB tracks: {int(np.sum(gt_active))}  ·  "
            f"sim boxes in roadway view: {int(np.sum(sim_in_view))}"
        )
        return [*sim_collections, *gt_collections, clock]

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=1000.0 / fps,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3000,
        extra_args=(
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ),
    )
    started = time.perf_counter()
    animation.save(output_path, writer=writer, dpi=dpi)
    plt.close(figure)
    print(f"rendered {output_path} in {time.perf_counter() - started:.1f} s", flush=True)
    return output_path


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--scenario", type=Path, required=True)
    argument_parser.add_argument("--data-dir", type=Path, required=True)
    argument_parser.add_argument("--map-id", type=int, default=0)
    argument_parser.add_argument("--checkpoint", type=Path, required=True)
    argument_parser.add_argument("--target-rollout", type=int, default=0)
    argument_parser.add_argument("--seed", type=int, default=1)
    argument_parser.add_argument("--trajectory-out", type=Path, required=True)
    argument_parser.add_argument("--video-out", type=Path, required=True)
    argument_parser.add_argument("--render-only", action="store_true")
    argument_parser.add_argument("--capture-only", action="store_true")
    argument_parser.add_argument("--direction", choices=("westbound", "eastbound"), default="westbound")
    argument_parser.add_argument("--frame-stride", type=int, default=2)
    argument_parser.add_argument("--fps", type=int, default=5)
    argument_parser.add_argument("--dpi", type=int, default=110)
    argument_parser.add_argument("--metric-label", default="")
    argument_parser.add_argument(
        "--ffmpeg", type=Path, default=None,
        help="ffmpeg executable; otherwise resolve it from PATH",
    )
    argument_parser.add_argument("--hidden", type=int, default=I24_POLICY_HIDDEN)
    argument_parser.add_argument("--layers", type=int, default=I24_POLICY_LAYERS)
    argument_parser.add_argument("--library", type=Path, default=ROOT / "build" / "libdrive_offline.so")
    argument_parser.add_argument("--policy-act", type=Path, default=ROOT / "build" / "i24_policy_act")
    return argument_parser


def main() -> int:
    args = parser().parse_args()
    if args.render_only and args.capture_only:
        raise SystemExit("--render-only and --capture-only are mutually exclusive")
    if not args.render_only:
        capture_rollout(
            scenario_path=args.scenario,
            data_dir=args.data_dir,
            map_id=args.map_id,
            checkpoint=args.checkpoint,
            target_rollout=args.target_rollout,
            seed=args.seed,
            output_path=args.trajectory_out,
            library=args.library,
            policy_act=args.policy_act,
            hidden=args.hidden,
            layers=args.layers,
        )
    if not args.capture_only:
        render_video(
            scenario_path=args.scenario,
            trajectory_path=args.trajectory_out,
            output_path=args.video_out,
            direction=-1 if args.direction == "westbound" else 1,
            frame_stride=args.frame_stride,
            fps=args.fps,
            dpi=args.dpi,
            metric_label=args.metric_label,
            ffmpeg=args.ffmpeg,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
