"""Analyze I24-3D assignment and net distance-headway distributions.

Scenes 1 and 3 use the calibrated lane boundaries. Scene 2 instead infers the
dominant paths followed by the observed trajectories, which accommodates its
snow-rut traffic streams. Proposed Scene 2 leaders must also overlap the
follower laterally after both vehicles are expressed relative to their shared
effective-stream centerline.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

if __package__:
    from .i24_3d_to_spiced import SAMPLE_HZ, normalize_scene
    from .plot_i24_time_space import (
        carriageway_boundaries,
        clean_objects,
        configure_matplotlib,
        lane_centers,
        sample_lane,
        time_bounds,
    )
else:
    from i24_3d_to_spiced import SAMPLE_HZ, normalize_scene
    from plot_i24_time_space import (
        carriageway_boundaries,
        clean_objects,
        configure_matplotlib,
        lane_centers,
        sample_lane,
        time_bounds,
    )


ALL_COLOR = "#5B4B8A"
UNAMBIGUOUS_COLOR = "#008C8C"
RAW_COLOR = "#D95F02"
CONFIRMED_COLOR = "#1B9E77"
DIRECTION_LABELS = {1: "Eastbound", -1: "Westbound"}


class LaneState(NamedTuple):
    lane: int
    ambiguous: bool
    outside: bool


class HeadwayObservation(NamedTuple):
    scene: int
    direction: int
    lane: int
    tick: int
    follower_id: int
    leader_id: int
    gap_m: float
    ambiguous_pair: bool


class EffectiveStreamModel(NamedTuple):
    """Polynomial centerlines for data-derived traffic streams."""

    direction: int
    x_reference_m: float
    x_scale_m: float
    coefficients: tuple[tuple[float, ...], ...]
    density_peaks_m: tuple[float, ...]

    def centers_at(self, longitudinal_position_m: float) -> np.ndarray:
        z = (longitudinal_position_m - self.x_reference_m) / self.x_scale_m
        return np.asarray(
            [np.polyval(coefficient, z) for coefficient in self.coefficients]
        )


class EffectiveStreamState(NamedTuple):
    lane: int
    ambiguous: bool
    outside: bool
    residual_m: float
    smoothed_y_m: float


def smooth_lateral_positions(
    obj: dict[str, Any], smoothing_seconds: float
) -> np.ndarray:
    """Return a centered moving-average lateral trajectory."""
    values = np.asarray(obj["position_m"], dtype=float)[:, 1]
    window = max(1, round(smoothing_seconds * SAMPLE_HZ))
    if window <= 1 or len(values) <= 2:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window <= 1:
        return values.copy()
    half_window = window // 2
    padded = np.pad(values, half_window, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _density_peaks(
    lateral_positions_m: np.ndarray,
    limits_m: tuple[float, float],
    bandwidth_m: float,
    minimum_spacing_m: float,
    minimum_prominence_fraction: float,
) -> np.ndarray:
    """Locate repeatable lateral traffic-stream modes in a smoothed histogram."""
    bin_width_m = 0.05
    bin_count = max(40, math.ceil((limits_m[1] - limits_m[0]) / bin_width_m))
    counts, edges = np.histogram(lateral_positions_m, bins=bin_count, range=limits_m)
    actual_bin_width = edges[1] - edges[0]
    density = gaussian_filter1d(
        counts.astype(float), sigma=max(0.5, bandwidth_m / actual_bin_width)
    )
    peak_indices, _ = find_peaks(
        density,
        distance=max(1, round(minimum_spacing_m / actual_bin_width)),
        prominence=minimum_prominence_fraction * float(density.max()),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[peak_indices]


def _fit_binned_polynomial(
    x_m: np.ndarray,
    y_m: np.ndarray,
    x_reference_m: float,
    x_scale_m: float,
    degree: int,
    bin_count: int = 24,
) -> tuple[float, ...]:
    """Fit a centerline to lateral medians, limiting density-related bias."""
    edges = np.linspace(float(x_m.min()), float(x_m.max()), bin_count + 1)
    bin_indices = np.clip(np.digitize(x_m, edges) - 1, 0, bin_count - 1)
    median_x: list[float] = []
    median_y: list[float] = []
    weights: list[float] = []
    for bin_index in range(bin_count):
        selected = bin_indices == bin_index
        if np.count_nonzero(selected) < 8:
            continue
        median_x.append(float(np.median(x_m[selected])))
        median_y.append(float(np.median(y_m[selected])))
        weights.append(math.sqrt(float(np.count_nonzero(selected))))
    if not median_x:
        return (float(np.median(y_m)),)
    fit_degree = min(degree, len(median_x) - 1)
    normalized_x = (np.asarray(median_x) - x_reference_m) / x_scale_m
    return tuple(
        float(value)
        for value in np.polyfit(
            normalized_x,
            np.asarray(median_y),
            fit_degree,
            w=np.asarray(weights),
        )
    )


def infer_effective_streams(
    scene: dict[str, Any],
    include_major_flags: bool,
    smoothing_seconds: float = 1.0,
    density_bandwidth_m: float = 0.35,
    minimum_stream_spacing_m: float = 2.5,
    minimum_peak_prominence_fraction: float = 0.08,
    polynomial_degree: int = 2,
) -> tuple[dict[int, EffectiveStreamModel], dict[int, np.ndarray]]:
    """Infer Scene 2 traffic-stream centerlines from smoothed trajectories."""
    objects = clean_objects(scene, include_major_flags)
    smoothed_y_by_id = {
        int(obj["id"]): smooth_lateral_positions(obj, smoothing_seconds)
        for obj in objects
    }
    models: dict[int, EffectiveStreamModel] = {}
    for direction in (1, -1):
        subset = [obj for obj in objects if int(obj["direction"]) == direction]
        x_m = np.concatenate(
            [np.asarray(obj["position_m"], dtype=float)[:, 0] for obj in subset]
        )
        y_m = np.concatenate([smoothed_y_by_id[int(obj["id"])] for obj in subset])
        limits = carriageway_boundaries(scene, direction)
        peaks = _density_peaks(
            y_m,
            (limits[0], limits[-1]),
            density_bandwidth_m,
            minimum_stream_spacing_m,
            minimum_peak_prominence_fraction,
        )
        if len(peaks) < 2:
            raise ValueError(
                f"Could not infer at least two Scene 2 streams for direction {direction}"
            )
        x_reference_m = float(np.median(x_m))
        x_scale_m = max(1.0, 0.5 * float(np.ptp(x_m)))
        coefficients: list[tuple[float, ...]] = [(float(peak),) for peak in peaks]
        for _ in range(6):
            z = (x_m - x_reference_m) / x_scale_m
            predictions = np.column_stack(
                [np.polyval(coefficient, z) for coefficient in coefficients]
            )
            assignments = np.argmin(np.abs(y_m[:, None] - predictions), axis=1)
            updated: list[tuple[float, ...]] = []
            for stream_index in range(len(coefficients)):
                selected = assignments == stream_index
                if np.count_nonzero(selected) < 50:
                    updated.append(coefficients[stream_index])
                else:
                    updated.append(
                        _fit_binned_polynomial(
                            x_m[selected],
                            y_m[selected],
                            x_reference_m,
                            x_scale_m,
                            polynomial_degree,
                        )
                    )
            coefficients = updated

        reference_values = np.asarray(
            [np.polyval(coefficient, 0.0) for coefficient in coefficients]
        )
        ordering = np.argsort(reference_values)
        if direction == 1:
            ordering = ordering[::-1]
        ordered_coefficients = tuple(coefficients[index] for index in ordering)
        ordered_peaks = tuple(float(peaks[index]) for index in ordering)
        models[direction] = EffectiveStreamModel(
            direction=direction,
            x_reference_m=x_reference_m,
            x_scale_m=x_scale_m,
            coefficients=ordered_coefficients,
            density_peaks_m=ordered_peaks,
        )
    return models, smoothed_y_by_id


def effective_stream_state(
    scene: dict[str, Any],
    obj: dict[str, Any],
    sample_index: int,
    model: EffectiveStreamModel,
    smoothed_y_m: np.ndarray,
    maximum_center_offset_m: float = 1.5,
    minimum_assignment_advantage_m: float = 0.75,
) -> EffectiveStreamState:
    """Assign a sample to an inferred path and report confidence/residual."""
    x_m = float(obj["position_m"][sample_index][0])
    y_m = float(smoothed_y_m[sample_index])
    centers = model.centers_at(x_m)
    distances = np.abs(centers - y_m)
    best = int(np.argmin(distances))
    second_distance = (
        float(np.partition(distances, 1)[1]) if len(distances) > 1 else math.inf
    )
    boundaries = carriageway_boundaries(scene, int(obj["direction"]))
    outside = not (boundaries[0] <= y_m <= boundaries[-1])
    ambiguous = (
        float(distances[best]) > maximum_center_offset_m
        or second_distance - float(distances[best])
        < minimum_assignment_advantage_m
    )
    return EffectiveStreamState(
        lane=best + 1,
        ambiguous=ambiguous,
        outside=outside,
        residual_m=y_m - float(centers[best]),
        smoothed_y_m=y_m,
    )


def lane_state(
    scene: dict[str, Any],
    obj: dict[str, Any],
    sample_index: int,
    boundary_margin_m: float = 0.0,
) -> LaneState:
    """Return hard lane membership and footprint-overlap ambiguity."""
    direction = int(obj["direction"])
    boundaries = carriageway_boundaries(scene, direction)
    centers = lane_centers(scene, direction)
    lateral_position = float(obj["position_m"][sample_index][1])
    half_width = 0.5 * float(obj["width_m"]) + boundary_margin_m
    lane = sample_lane(lateral_position, centers)
    ambiguous = any(
        lateral_position - half_width <= boundary <= lateral_position + half_width
        for boundary in boundaries[1:-1]
    )
    outside = not (boundaries[0] <= lateral_position <= boundaries[-1])
    return LaneState(lane=lane, ambiguous=ambiguous, outside=outside)


def _run_lengths(values: list[Any]) -> list[tuple[Any, int]]:
    if not values:
        return []
    runs: list[tuple[Any, int]] = []
    current = values[0]
    length = 1
    for value in values[1:]:
        if value == current:
            length += 1
        else:
            runs.append((current, length))
            current = value
            length = 1
    runs.append((current, length))
    return runs


def short_return_flickers(lanes: list[int], max_samples: int) -> int:
    """Count short A-B-A lane runs in a hard-assignment sequence."""
    runs = _run_lengths(lanes)
    return sum(
        previous_lane == next_lane
        and middle_lane != previous_lane
        and middle_length <= max_samples
        for (previous_lane, _), (middle_lane, middle_length), (next_lane, _) in zip(
            runs, runs[1:], runs[2:]
        )
    )


def confirmed_switches(states: list[LaneState]) -> int:
    """Count lane changes after removing boundary-overlap and outside samples."""
    lanes = [state.lane for state in states if not state.ambiguous and not state.outside]
    return max(0, len(_run_lengths(lanes)) - 1)


def analyze_lane_assignment(
    scenes: list[dict[str, Any]],
    include_major_flags: bool,
    boundary_margin_m: float,
    flicker_max_seconds: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Summarize ambiguity, raw switches, and short-return flicker by group."""
    summaries: list[dict[str, Any]] = []
    all_episode_durations: list[float] = []
    max_flicker_samples = max(1, round(flicker_max_seconds * SAMPLE_HZ))
    for scene in scenes:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for obj in clean_objects(scene, include_major_flags):
            grouped[int(obj["direction"])].append(obj)
        for direction in (1, -1):
            summary = {
                "scene": int(scene["scene"]),
                "direction": direction,
                "vehicle_count": len(grouped[direction]),
                "sample_count": 0,
                "ambiguous_sample_count": 0,
                "outside_sample_count": 0,
                "raw_switch_count": 0,
                "confirmed_switch_count": 0,
                "short_return_flicker_count": 0,
                "vehicles_with_raw_switch": 0,
                "vehicles_with_confirmed_switch": 0,
                "vehicles_with_short_return_flicker": 0,
                "ambiguous_episode_count": 0,
            }
            for obj in grouped[direction]:
                states = [
                    lane_state(scene, obj, index, boundary_margin_m)
                    for index in range(len(obj["position_m"]))
                ]
                lanes = [state.lane for state in states]
                raw_switch_count = sum(
                    left != right for left, right in zip(lanes, lanes[1:])
                )
                confirmed_switch_count = confirmed_switches(states)
                flicker_count = short_return_flickers(lanes, max_flicker_samples)
                ambiguous_runs = [
                    length / SAMPLE_HZ
                    for value, length in _run_lengths(
                        [state.ambiguous for state in states]
                    )
                    if value
                ]

                summary["sample_count"] += len(states)
                summary["ambiguous_sample_count"] += sum(
                    state.ambiguous for state in states
                )
                summary["outside_sample_count"] += sum(state.outside for state in states)
                summary["raw_switch_count"] += raw_switch_count
                summary["confirmed_switch_count"] += confirmed_switch_count
                summary["short_return_flicker_count"] += flicker_count
                summary["ambiguous_episode_count"] += len(ambiguous_runs)
                summary["vehicles_with_raw_switch"] += raw_switch_count > 0
                summary["vehicles_with_confirmed_switch"] += confirmed_switch_count > 0
                summary["vehicles_with_short_return_flicker"] += flicker_count > 0
                all_episode_durations.extend(ambiguous_runs)

            sample_count = int(summary["sample_count"])
            vehicle_count = int(summary["vehicle_count"])
            exposure_minutes = sample_count / SAMPLE_HZ / 60.0
            summary["ambiguous_sample_fraction"] = (
                summary["ambiguous_sample_count"] / sample_count
                if sample_count
                else 0.0
            )
            summary["raw_switches_per_100_vehicle_minutes"] = (
                100.0 * summary["raw_switch_count"] / exposure_minutes
                if exposure_minutes
                else 0.0
            )
            summary["confirmed_switches_per_100_vehicle_minutes"] = (
                100.0 * summary["confirmed_switch_count"] / exposure_minutes
                if exposure_minutes
                else 0.0
            )
            for key in (
                "vehicles_with_raw_switch",
                "vehicles_with_confirmed_switch",
                "vehicles_with_short_return_flicker",
            ):
                summary[f"{key}_fraction"] = (
                    summary[key] / vehicle_count if vehicle_count else 0.0
                )
            summaries.append(summary)
    return summaries, all_episode_durations


def collect_headway_observations(
    scene: dict[str, Any],
    include_major_flags: bool,
    boundary_margin_m: float,
    sample_interval_seconds: float,
) -> list[HeadwayObservation]:
    """Infer adjacent same-lane pairs at regularly spaced scene ticks."""
    objects = clean_objects(scene, include_major_flags)
    first_tick, last_tick = time_bounds(objects)
    stride = max(1, round(sample_interval_seconds * SAMPLE_HZ))
    observations: list[HeadwayObservation] = []
    for tick in range(first_tick, last_tick + 1, stride):
        groups: dict[tuple[int, int], list[tuple[float, dict[str, Any], LaneState]]] = (
            defaultdict(list)
        )
        for obj in objects:
            sample_index = tick - int(obj["start_tick"])
            if not 0 <= sample_index < len(obj["position_m"]):
                continue
            state = lane_state(scene, obj, sample_index, boundary_margin_m)
            if state.outside:
                continue
            direction = int(obj["direction"])
            rear_s = direction * float(obj["position_m"][sample_index][0])
            groups[(direction, state.lane)].append((rear_s, obj, state))

        for (direction, lane), vehicles in groups.items():
            vehicles.sort(key=lambda entry: entry[0])
            for follower, leader in zip(vehicles, vehicles[1:]):
                follower_s, follower_obj, follower_state = follower
                leader_s, leader_obj, leader_state = leader
                gap_m = leader_s - (
                    follower_s + float(follower_obj["length_m"])
                )
                observations.append(
                    HeadwayObservation(
                        scene=int(scene["scene"]),
                        direction=direction,
                        lane=lane,
                        tick=tick,
                        follower_id=int(follower_obj["id"]),
                        leader_id=int(leader_obj["id"]),
                        gap_m=gap_m,
                        ambiguous_pair=(
                            follower_state.ambiguous or leader_state.ambiguous
                        ),
                    )
                )
    return observations


def collect_effective_headway_observations(
    scene: dict[str, Any],
    include_major_flags: bool,
    sample_interval_seconds: float,
    models: dict[int, EffectiveStreamModel],
    smoothed_y_by_id: dict[int, np.ndarray],
    maximum_center_offset_m: float,
    minimum_assignment_advantage_m: float,
    minimum_lateral_overlap_m: float,
) -> tuple[list[HeadwayObservation], dict[str, int]]:
    """Collect Scene 2 gaps using effective streams plus lateral compatibility."""
    objects = clean_objects(scene, include_major_flags)
    first_tick, last_tick = time_bounds(objects)
    stride = max(1, round(sample_interval_seconds * SAMPLE_HZ))
    observations: list[HeadwayObservation] = []
    diagnostics = {
        "candidate_pair_count": 0,
        "accepted_pair_count": 0,
        "lateral_incompatibility_count": 0,
        "ambiguous_accepted_pair_count": 0,
    }
    for tick in range(first_tick, last_tick + 1, stride):
        groups: dict[
            tuple[int, int],
            list[tuple[float, dict[str, Any], EffectiveStreamState]],
        ] = defaultdict(list)
        for obj in objects:
            sample_index = tick - int(obj["start_tick"])
            if not 0 <= sample_index < len(obj["position_m"]):
                continue
            direction = int(obj["direction"])
            state = effective_stream_state(
                scene,
                obj,
                sample_index,
                models[direction],
                smoothed_y_by_id[int(obj["id"])],
                maximum_center_offset_m,
                minimum_assignment_advantage_m,
            )
            if state.outside:
                continue
            rear_s = direction * float(obj["position_m"][sample_index][0])
            groups[(direction, state.lane)].append((rear_s, obj, state))

        for (direction, lane), vehicles in groups.items():
            vehicles.sort(key=lambda entry: entry[0])
            for follower, leader in zip(vehicles, vehicles[1:]):
                diagnostics["candidate_pair_count"] += 1
                follower_s, follower_obj, follower_state = follower
                leader_s, leader_obj, leader_state = leader
                lateral_overlap_m = (
                    0.5
                    * (
                        float(follower_obj["width_m"])
                        + float(leader_obj["width_m"])
                    )
                    - abs(follower_state.residual_m - leader_state.residual_m)
                )
                if lateral_overlap_m < minimum_lateral_overlap_m:
                    diagnostics["lateral_incompatibility_count"] += 1
                    continue
                ambiguous_pair = follower_state.ambiguous or leader_state.ambiguous
                diagnostics["accepted_pair_count"] += 1
                diagnostics["ambiguous_accepted_pair_count"] += ambiguous_pair
                gap_m = leader_s - (
                    follower_s + float(follower_obj["length_m"])
                )
                observations.append(
                    HeadwayObservation(
                        scene=int(scene["scene"]),
                        direction=direction,
                        lane=lane,
                        tick=tick,
                        follower_id=int(follower_obj["id"]),
                        leader_id=int(leader_obj["id"]),
                        gap_m=gap_m,
                        ambiguous_pair=ambiguous_pair,
                    )
                )
    return observations, diagnostics


def leader_relationship_diagnostics(
    observations: list[HeadwayObservation], sample_interval_seconds: float
) -> dict[str, float | int | None]:
    """Summarize persistence of sampled leader identities."""
    stride = max(1, round(sample_interval_seconds * SAMPLE_HZ))
    by_follower: dict[tuple[int, int], list[HeadwayObservation]] = defaultdict(list)
    for observation in observations:
        by_follower[(observation.direction, observation.follower_id)].append(
            observation
        )
    consecutive_transition_count = 0
    leader_change_count = 0
    run_durations: list[float] = []
    for follower_observations in by_follower.values():
        follower_observations.sort(key=lambda observation: observation.tick)
        run_length = 1
        for previous, current in zip(
            follower_observations, follower_observations[1:]
        ):
            consecutive = current.tick - previous.tick == stride
            same_leader = current.leader_id == previous.leader_id
            if consecutive:
                consecutive_transition_count += 1
                leader_change_count += not same_leader
            if consecutive and same_leader:
                run_length += 1
            else:
                run_durations.append(run_length * sample_interval_seconds)
                run_length = 1
        run_durations.append(run_length * sample_interval_seconds)
    exposure_minutes = (
        consecutive_transition_count * sample_interval_seconds / 60.0
    )
    return {
        "observation_count": len(observations),
        "follower_count": len(by_follower),
        "consecutive_transition_count": consecutive_transition_count,
        "leader_change_count": leader_change_count,
        "leader_changes_per_100_follower_minutes": (
            100.0 * leader_change_count / exposure_minutes
            if exposure_minutes
            else 0.0
        ),
        "median_relationship_duration_seconds": (
            float(np.median(run_durations)) if run_durations else None
        ),
        "p90_relationship_duration_seconds": (
            float(np.percentile(run_durations, 90)) if run_durations else None
        ),
    }


def summarize_headways(
    observations: list[HeadwayObservation],
) -> list[dict[str, Any]]:
    """Summarize all and unambiguous nonnegative headways by scene/direction."""
    summaries: list[dict[str, Any]] = []
    for scene in (1, 2, 3):
        for direction in (1, -1):
            group = [
                obs
                for obs in observations
                if obs.scene == scene and obs.direction == direction
            ]
            all_valid = [obs.gap_m for obs in group if obs.gap_m >= 0.0]
            clean_valid = [
                obs.gap_m
                for obs in group
                if obs.gap_m >= 0.0 and not obs.ambiguous_pair
            ]
            all_negative = sum(obs.gap_m < 0.0 for obs in group)
            clean_negative = sum(
                obs.gap_m < 0.0 and not obs.ambiguous_pair for obs in group
            )
            summaries.append(
                {
                    "scene": scene,
                    "direction": direction,
                    "all_valid_count": len(all_valid),
                    "unambiguous_valid_count": len(clean_valid),
                    "retained_fraction": (
                        len(clean_valid) / len(all_valid) if all_valid else 0.0
                    ),
                    "all_negative_count": all_negative,
                    "unambiguous_negative_count": clean_negative,
                    "all_median_m": (
                        float(np.median(all_valid)) if all_valid else None
                    ),
                    "unambiguous_median_m": (
                        float(np.median(clean_valid)) if clean_valid else None
                    ),
                    "all_values_m": all_valid,
                    "unambiguous_values_m": clean_valid,
                }
            )
    return summaries


def _group_labels() -> list[str]:
    return [
        f"S{scene}\n{'EB' if direction == 1 else 'WB'}"
        for scene in (1, 2, 3)
        for direction in (1, -1)
    ]


def plot_headway_distributions(
    summaries: list[dict[str, Any]],
    output_path: Path,
    sample_interval_seconds: float,
    max_gap_meters: float,
    dpi: int,
) -> None:
    """Plot selected hybrid-method distributions with confidence filtering."""
    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.3))
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.08,
        top=0.89,
        hspace=0.40,
        wspace=0.17,
    )
    figure.suptitle(
        "I-24-3D net distance-headway distributions",
        fontsize=17,
        fontweight="bold",
        y=0.976,
    )
    figure.text(
        0.5,
        0.94,
        f"{sample_interval_seconds:g} s sampling · bumper-to-bumper gap · "
        "Scene 2 uses inferred snow-rut streams · negative gaps excluded",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    bins = np.linspace(0.0, max_gap_meters, 41)
    for axis, summary in zip(axes.flat, summaries):
        all_values = np.asarray(summary["all_values_m"], dtype=float)
        clean_values = np.asarray(summary["unambiguous_values_m"], dtype=float)
        if summary["scene"] == 2:
            labels = (
                "Laterally compatible effective-stream pairs",
                "High-confidence effective-stream pairs",
            )
        else:
            labels = (
                "Forced lane for transitional samples",
                "Both vehicles unambiguous",
            )
        for values, color, label in (
            (all_values, ALL_COLOR, labels[0]),
            (clean_values, UNAMBIGUOUS_COLOR, labels[1]),
        ):
            if len(values):
                axis.hist(
                    values,
                    bins=bins,
                    weights=np.full(len(values), 1.0 / len(values)),
                    histtype="step",
                    linewidth=1.8,
                    color=color,
                    label=f"{label} · n={len(values):,} · median={np.median(values):.1f} m",
                )
        axis.set_xlim(0.0, max_gap_meters)
        axis.set_title(
            f"Scene {summary['scene']} · {DIRECTION_LABELS[summary['direction']]}",
            loc="left",
            fontweight="semibold",
        )
        axis.set_xlabel("Net bumper-to-bumper gap (m)")
        axis.set_ylabel("Fraction of observations per bin")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.legend(loc="upper right", fontsize=7.5, frameon=False)
        all_tail = np.mean(all_values > max_gap_meters) if len(all_values) else 0.0
        clean_tail = (
            np.mean(clean_values > max_gap_meters) if len(clean_values) else 0.0
        )
        axis.text(
            0.99,
            0.54,
            f">{max_gap_meters:g} m: {all_tail:.1%} / {clean_tail:.1%}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#555555",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_headway_cdfs(
    summaries: list[dict[str, Any]],
    output_path: Path,
    sample_interval_seconds: float,
    max_gap_meters: float,
    dpi: int,
) -> None:
    """Plot empirical CDFs for the same observations as the histograms."""
    figure, axes = plt.subplots(3, 2, figsize=(13.4, 10.3))
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.08,
        top=0.89,
        hspace=0.40,
        wspace=0.17,
    )
    figure.suptitle(
        "I-24-3D net distance-headway empirical CDFs",
        fontsize=17,
        fontweight="bold",
        y=0.976,
    )
    figure.text(
        0.5,
        0.94,
        f"{sample_interval_seconds:g} s repeated observations across full trajectories · "
        "Scene 2 uses inferred snow-rut streams",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    for axis, summary in zip(axes.flat, summaries):
        all_values = np.sort(np.asarray(summary["all_values_m"], dtype=float))
        clean_values = np.sort(
            np.asarray(summary["unambiguous_values_m"], dtype=float)
        )
        if summary["scene"] == 2:
            labels = (
                "Laterally compatible effective-stream pairs",
                "High-confidence effective-stream pairs",
            )
        else:
            labels = (
                "Forced lane for transitional samples",
                "Both vehicles unambiguous",
            )
        for values, color, label in (
            (all_values, ALL_COLOR, labels[0]),
            (clean_values, UNAMBIGUOUS_COLOR, labels[1]),
        ):
            if len(values):
                x = np.concatenate(([0.0], values))
                cumulative_fraction = np.concatenate(
                    ([0.0], np.arange(1, len(values) + 1) / len(values))
                )
                axis.step(
                    x,
                    cumulative_fraction,
                    where="post",
                    linewidth=1.8,
                    color=color,
                    label=f"{label} · n={len(values):,} · median={np.median(values):.1f} m",
                )
        axis.axhline(0.5, color="#999999", linewidth=0.7, linestyle=":")
        axis.set_xlim(0.0, max_gap_meters)
        axis.set_ylim(0.0, 1.01)
        axis.set_title(
            f"Scene {summary['scene']} · {DIRECTION_LABELS[summary['direction']]}",
            loc="left",
            fontweight="semibold",
        )
        axis.set_xlabel("Net bumper-to-bumper gap (m)")
        axis.set_ylabel("Cumulative fraction ≤ gap")
        axis.grid(color="#DDDDDD", linewidth=0.5)
        axis.legend(loc="lower right", fontsize=7.5, frameon=False)
        all_tail = np.mean(all_values > max_gap_meters) if len(all_values) else 0.0
        clean_tail = (
            np.mean(clean_values > max_gap_meters) if len(clean_values) else 0.0
        )
        if all_tail or clean_tail:
            axis.text(
                0.99,
                0.55,
                f">{max_gap_meters:g} m: {all_tail:.1%} / {clean_tail:.1%}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#555555",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_scene2_effective_streams(
    scene: dict[str, Any],
    models: dict[int, EffectiveStreamModel],
    smoothed_y_by_id: dict[int, np.ndarray],
    include_major_flags: bool,
    output_path: Path,
    dpi: int,
) -> None:
    """Show Scene 2 trajectories, calibrated boundaries, and inferred paths."""
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.82, wspace=0.15)
    figure.suptitle(
        "Scene 2 effective traffic streams inferred from snowy-day trajectories",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        "Thin traces: 1 s-smoothed vehicle paths · solid: inferred stream centerlines · "
        "dashed: calibration-grid lane boundaries",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    colors = ("#5B4B8A", "#008C8C", "#D95F02", "#1B9E77", "#7570B3")
    objects = clean_objects(scene, include_major_flags)
    for axis, direction in zip(axes, (1, -1)):
        subset = [obj for obj in objects if int(obj["direction"]) == direction]
        all_x = np.concatenate(
            [np.asarray(obj["position_m"], dtype=float)[:, 0] for obj in subset]
        )
        for obj in subset:
            x_m = np.asarray(obj["position_m"], dtype=float)[:, 0]
            y_m = smoothed_y_by_id[int(obj["id"])]
            axis.plot(x_m, y_m, color="#A6A6A6", linewidth=0.45, alpha=0.24)
        x_grid = np.linspace(float(all_x.min()), float(all_x.max()), 300)
        model = models[direction]
        for lane_index, coefficient in enumerate(model.coefficients, start=1):
            z = (x_grid - model.x_reference_m) / model.x_scale_m
            axis.plot(
                x_grid,
                np.polyval(coefficient, z),
                color=colors[(lane_index - 1) % len(colors)],
                linewidth=2.1,
                label=f"Effective stream {lane_index}",
            )
        for boundary in carriageway_boundaries(scene, direction)[1:-1]:
            axis.axhline(boundary, color="#666666", linestyle="--", linewidth=0.8, alpha=0.65)
        axis.set_xlim(float(all_x.min()), float(all_x.max()))
        boundaries = carriageway_boundaries(scene, direction)
        axis.set_ylim(boundaries[0], boundaries[-1])
        axis.set_xlabel("Longitudinal position, x (m)")
        axis.set_ylabel("Lateral position, y (m)")
        axis.set_title(
            f"{DIRECTION_LABELS[direction]} · {len(model.coefficients)} inferred streams",
            loc="left",
            fontweight="semibold",
        )
        axis.grid(color="#DDDDDD", linewidth=0.45)
        axis.legend(loc="best", frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _scene_direction_summary(
    summaries: list[dict[str, Any]], scene: int, direction: int
) -> dict[str, Any]:
    return next(
        summary
        for summary in summaries
        if summary["scene"] == scene and summary["direction"] == direction
    )


def plot_scene2_method_comparison(
    calibrated_summaries: list[dict[str, Any]],
    effective_summaries: list[dict[str, Any]],
    leader_diagnostics: list[dict[str, Any]],
    output_path: Path,
    max_gap_meters: float,
    dpi: int,
) -> None:
    """Compare calibrated lanes with the Scene 2 effective-stream hybrid."""
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.5))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.11, top=0.86, hspace=0.38, wspace=0.22)
    figure.suptitle(
        "Scene 2: calibrated lanes versus inferred effective streams",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.92,
        "High-confidence, nonnegative bumper-to-bumper gaps · 1 s sampling",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    bins = np.linspace(0.0, max_gap_meters, 41)
    for axis, direction in zip(axes[0], (1, -1)):
        calibrated = _scene_direction_summary(calibrated_summaries, 2, direction)
        effective = _scene_direction_summary(effective_summaries, 2, direction)
        for summary, color, label in (
            (calibrated, ALL_COLOR, "Calibrated lanes"),
            (effective, UNAMBIGUOUS_COLOR, "Effective streams + overlap check"),
        ):
            values = np.asarray(summary["unambiguous_values_m"], dtype=float)
            if len(values):
                axis.hist(
                    values,
                    bins=bins,
                    weights=np.full(len(values), 1.0 / len(values)),
                    histtype="step",
                    linewidth=2.0,
                    color=color,
                    label=f"{label} · n={len(values):,} · median={np.median(values):.1f} m",
                )
        axis.set_xlim(0.0, max_gap_meters)
        axis.set_xlabel("Net bumper-to-bumper gap (m)")
        axis.set_ylabel("Fraction of observations per bin")
        axis.set_title(DIRECTION_LABELS[direction], loc="left", fontweight="semibold")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)

    labels = ["EB", "WB"]
    x = np.arange(2)
    width = 0.34
    calibrated_medians = []
    effective_medians = []
    for direction in (1, -1):
        calibrated_medians.append(
            _scene_direction_summary(calibrated_summaries, 2, direction)[
                "unambiguous_median_m"
            ]
            or 0.0
        )
        effective_medians.append(
            _scene_direction_summary(effective_summaries, 2, direction)[
                "unambiguous_median_m"
            ]
            or 0.0
        )
    axes[1, 0].bar(x - width / 2, calibrated_medians, width, color=ALL_COLOR, label="Calibrated")
    axes[1, 0].bar(x + width / 2, effective_medians, width, color=UNAMBIGUOUS_COLOR, label="Effective")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Median net gap (m)")
    axes[1, 0].set_title("Median of retained headways", loc="left")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", color="#DDDDDD", linewidth=0.5)

    calibrated_changes = [
        row["calibrated"]["leader_changes_per_100_follower_minutes"]
        for row in leader_diagnostics
    ]
    effective_changes = [
        row["effective"]["leader_changes_per_100_follower_minutes"]
        for row in leader_diagnostics
    ]
    axes[1, 1].bar(x - width / 2, calibrated_changes, width, color=ALL_COLOR, label="Calibrated")
    axes[1, 1].bar(x + width / 2, effective_changes, width, color=UNAMBIGUOUS_COLOR, label="Effective")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Changes per 100 follower-min")
    axes[1, 1].set_title("Sampled leader-identity changes", loc="left")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", color="#DDDDDD", linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_lane_diagnostics(
    lane_summaries: list[dict[str, Any]],
    headway_summaries: list[dict[str, Any]],
    episode_durations: list[float],
    output_path: Path,
    flicker_max_seconds: float,
    dpi: int,
) -> None:
    """Visualize ambiguity prevalence, switching, flicker, and sample retention."""
    labels = _group_labels()
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.7))
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.86,
        hspace=0.36,
        wspace=0.25,
    )
    figure.suptitle(
        "I-24-3D instantaneous lane-assignment diagnostics",
        fontsize=17,
        fontweight="bold",
        y=0.972,
    )
    if episode_durations:
        episode_array = np.asarray(episode_durations)
        duration_text = (
            f"Boundary-overlap episodes: n={len(episode_array):,}, "
            f"median={np.median(episode_array):.1f} s, "
            f"95th={np.percentile(episode_array, 95):.1f} s"
        )
    else:
        duration_text = "No boundary-overlap episodes"
    figure.text(0.5, 0.925, duration_text, ha="center", fontsize=10.5, color="#555555")

    ambiguity = [100.0 * row["ambiguous_sample_fraction"] for row in lane_summaries]
    axes[0, 0].bar(x, ambiguity, color=ALL_COLOR)
    axes[0, 0].set_title("Vehicle-time marked transitional/ambiguous", loc="left")
    axes[0, 0].set_ylabel("Trajectory samples (%)")

    raw_switches = [row["raw_switches_per_100_vehicle_minutes"] for row in lane_summaries]
    confirmed = [
        row["confirmed_switches_per_100_vehicle_minutes"] for row in lane_summaries
    ]
    axes[0, 1].bar(x - width / 2, raw_switches, width, color=RAW_COLOR, label="Raw center assignment")
    axes[0, 1].bar(x + width / 2, confirmed, width, color=CONFIRMED_COLOR, label="After overlap exclusion")
    axes[0, 1].set_title("Lane-switch rate", loc="left")
    axes[0, 1].set_ylabel("Switches per 100 vehicle-min")
    axes[0, 1].legend(frameon=False, fontsize=8)

    flicker_vehicle_fraction = [
        100.0 * row["vehicles_with_short_return_flicker_fraction"]
        for row in lane_summaries
    ]
    confirmed_vehicle_fraction = [
        100.0 * row["vehicles_with_confirmed_switch_fraction"]
        for row in lane_summaries
    ]
    axes[1, 0].bar(
        x - width / 2,
        flicker_vehicle_fraction,
        width,
        color=RAW_COLOR,
        label=f"Short A-B-A ≤{flicker_max_seconds:g} s",
    )
    axes[1, 0].bar(
        x + width / 2,
        confirmed_vehicle_fraction,
        width,
        color=CONFIRMED_COLOR,
        label="Confirmed lane change",
    )
    axes[1, 0].set_title("Vehicles affected", loc="left")
    axes[1, 0].set_ylabel("Vehicles (%)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    retention = [100.0 * row["retained_fraction"] for row in headway_summaries]
    axes[1, 1].bar(x, retention, color=UNAMBIGUOUS_COLOR)
    axes[1, 1].set_title("Headway observations retained", loc="left")
    axes[1, 1].set_ylabel("Nonnegative hard-assignment pairs (%)")
    axes[1, 1].set_ylim(0.0, 105.0)
    for index, row in enumerate(headway_summaries):
        before = row["all_median_m"]
        after = row["unambiguous_median_m"]
        if before is not None and after is not None:
            axes[1, 1].text(
                index,
                retention[index] + 2.0,
                f"{before:.1f}→{after:.1f} m",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _json_ready_headway_summaries(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in summary.items()
            if key not in {"all_values_m", "unambiguous_values_m"}
        }
        for summary in summaries
    ]


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
        default=repo_root / "figures" / "i24_headways",
    )
    argument_parser.add_argument("--include-major-flags", action="store_true")
    argument_parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    argument_parser.add_argument("--boundary-margin-meters", type=float, default=0.0)
    argument_parser.add_argument("--flicker-max-seconds", type=float, default=1.0)
    argument_parser.add_argument("--max-gap-meters", type=float, default=120.0)
    argument_parser.add_argument("--cdf-max-gap-meters", type=float, default=400.0)
    argument_parser.add_argument("--effective-smoothing-seconds", type=float, default=1.0)
    argument_parser.add_argument("--effective-density-bandwidth-meters", type=float, default=0.35)
    argument_parser.add_argument("--effective-min-stream-spacing-meters", type=float, default=2.5)
    argument_parser.add_argument("--effective-min-peak-prominence", type=float, default=0.08)
    argument_parser.add_argument("--effective-polynomial-degree", type=int, default=2)
    argument_parser.add_argument("--effective-max-center-offset-meters", type=float, default=1.5)
    argument_parser.add_argument("--effective-min-assignment-advantage-meters", type=float, default=0.75)
    argument_parser.add_argument("--effective-min-lateral-overlap-meters", type=float, default=0.25)
    argument_parser.add_argument("--dpi", type=int, default=150)
    return argument_parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "sample_interval_seconds",
        "flicker_max_seconds",
        "max_gap_meters",
        "cdf_max_gap_meters",
        "effective_smoothing_seconds",
        "effective_density_bandwidth_meters",
        "effective_min_stream_spacing_meters",
        "effective_max_center_offset_meters",
        "effective_min_assignment_advantage_meters",
        "dpi",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.boundary_margin_meters < 0:
        raise SystemExit("--boundary-margin-meters cannot be negative")
    if not 0.0 < args.effective_min_peak_prominence < 1.0:
        raise SystemExit("--effective-min-peak-prominence must be between 0 and 1")
    if args.effective_polynomial_degree not in (0, 1, 2, 3):
        raise SystemExit("--effective-polynomial-degree must be 0, 1, 2, or 3")
    if args.effective_min_lateral_overlap_meters < 0:
        raise SystemExit("--effective-min-lateral-overlap-meters cannot be negative")


def main() -> None:
    args = parser().parse_args()
    _validate_args(args)
    configure_matplotlib()
    scenes = [
        normalize_scene(args.i24_root.resolve(), scene, stabilize_headings=False)
        for scene in (1, 2, 3)
    ]
    lane_summaries, episode_durations = analyze_lane_assignment(
        scenes,
        args.include_major_flags,
        args.boundary_margin_meters,
        args.flicker_max_seconds,
    )
    normal_observations = [
        observation
        for scene in scenes
        if int(scene["scene"]) != 2
        for observation in collect_headway_observations(
            scene,
            args.include_major_flags,
            args.boundary_margin_meters,
            args.sample_interval_seconds,
        )
    ]
    scene2 = scenes[1]
    effective_models, smoothed_y_by_id = infer_effective_streams(
        scene2,
        args.include_major_flags,
        args.effective_smoothing_seconds,
        args.effective_density_bandwidth_meters,
        args.effective_min_stream_spacing_meters,
        args.effective_min_peak_prominence,
        args.effective_polynomial_degree,
    )
    effective_observations, effective_pair_diagnostics = (
        collect_effective_headway_observations(
            scene2,
            args.include_major_flags,
            args.sample_interval_seconds,
            effective_models,
            smoothed_y_by_id,
            args.effective_max_center_offset_meters,
            args.effective_min_assignment_advantage_meters,
            args.effective_min_lateral_overlap_meters,
        )
    )
    calibrated_scene2_observations = collect_headway_observations(
        scene2,
        args.include_major_flags,
        args.boundary_margin_meters,
        args.sample_interval_seconds,
    )
    observations = normal_observations + effective_observations
    headway_summaries = summarize_headways(observations)
    calibrated_scene2_summaries = summarize_headways(calibrated_scene2_observations)
    effective_scene2_summaries = summarize_headways(effective_observations)
    scene2_leader_diagnostics: list[dict[str, Any]] = []
    for direction in (1, -1):
        calibrated_clean = [
            observation
            for observation in calibrated_scene2_observations
            if observation.direction == direction and not observation.ambiguous_pair
        ]
        effective_clean = [
            observation
            for observation in effective_observations
            if observation.direction == direction and not observation.ambiguous_pair
        ]
        scene2_leader_diagnostics.append(
            {
                "direction": direction,
                "calibrated": leader_relationship_diagnostics(
                    calibrated_clean, args.sample_interval_seconds
                ),
                "effective": leader_relationship_diagnostics(
                    effective_clean, args.sample_interval_seconds
                ),
            }
        )

    distribution_path = args.output_dir / "i24_distance_headway_distributions.png"
    cdf_path = args.output_dir / "i24_distance_headway_cdfs.png"
    stream_path = args.output_dir / "i24_scene2_effective_streams.png"
    comparison_path = args.output_dir / "i24_scene2_method_comparison.png"
    manifest_path = args.output_dir / "i24_headway_analysis.json"
    plot_headway_distributions(
        headway_summaries,
        distribution_path,
        args.sample_interval_seconds,
        args.max_gap_meters,
        args.dpi,
    )
    plot_headway_cdfs(
        headway_summaries,
        cdf_path,
        args.sample_interval_seconds,
        args.cdf_max_gap_meters,
        args.dpi,
    )
    plot_scene2_effective_streams(
        scene2,
        effective_models,
        smoothed_y_by_id,
        args.include_major_flags,
        stream_path,
        args.dpi,
    )
    plot_scene2_method_comparison(
        calibrated_scene2_summaries,
        effective_scene2_summaries,
        scene2_leader_diagnostics,
        comparison_path,
        args.max_gap_meters,
        args.dpi,
    )
    manifest = {
        "method": {
            "headway": "net bumper-to-bumper gap",
            "sample_interval_seconds": args.sample_interval_seconds,
            "scene_1_and_3_assignment": "nearest calibrated lane center",
            "scene_1_and_3_ambiguous_rule": "vehicle lateral footprint overlaps an internal lane boundary",
            "scene_2_assignment": "nearest trajectory-inferred polynomial effective-stream centerline",
            "scene_2_pair_rule": "adjacent within effective stream and minimum residual-coordinate lateral overlap",
            "boundary_margin_meters": args.boundary_margin_meters,
            "flicker_rule": f"short A-B-A middle run <= {args.flicker_max_seconds:g} seconds",
            "negative_gap_policy": "reported but excluded from distributions",
            "major_quality_flags_included": args.include_major_flags,
        },
        "lane_assignment": lane_summaries,
        "headways": _json_ready_headway_summaries(headway_summaries),
        "scene_2_calibrated_comparison": _json_ready_headway_summaries(
            calibrated_scene2_summaries
        ),
        "scene_2_effective_streams": {
            "parameters": {
                "smoothing_seconds": args.effective_smoothing_seconds,
                "density_bandwidth_meters": args.effective_density_bandwidth_meters,
                "minimum_stream_spacing_meters": args.effective_min_stream_spacing_meters,
                "minimum_peak_prominence_fraction": args.effective_min_peak_prominence,
                "polynomial_degree": args.effective_polynomial_degree,
                "maximum_center_offset_meters": args.effective_max_center_offset_meters,
                "minimum_assignment_advantage_meters": args.effective_min_assignment_advantage_meters,
                "minimum_lateral_overlap_meters": args.effective_min_lateral_overlap_meters,
            },
            "models": {
                str(direction): {
                    "stream_count": len(model.coefficients),
                    "x_reference_m": model.x_reference_m,
                    "x_scale_m": model.x_scale_m,
                    "density_peaks_m": model.density_peaks_m,
                    "coefficients": model.coefficients,
                }
                for direction, model in effective_models.items()
            },
            "pair_diagnostics": effective_pair_diagnostics,
            "leader_diagnostics": scene2_leader_diagnostics,
        },
        "ambiguous_episode_durations_seconds": {
            "count": len(episode_durations),
            "median": (
                float(np.median(episode_durations)) if episode_durations else None
            ),
            "p95": (
                float(np.percentile(episode_durations, 95))
                if episode_durations
                else None
            ),
            "max": max(episode_durations) if episode_durations else None,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    print(f"Wrote {distribution_path}")
    print(f"Wrote {cdf_path}")
    print(f"Wrote {stream_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
