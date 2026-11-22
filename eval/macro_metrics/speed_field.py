"""Eulerian speed fields and cell-wise velocity error metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldGrid:
    time_edges: np.ndarray
    position_edges: np.ndarray


@dataclass(frozen=True)
class SpeedField:
    values: np.ndarray
    sample_count: np.ndarray

    @property
    def support(self) -> np.ndarray:
        return np.isfinite(self.values)


def regular_edges(lower: float, upper: float, step: float) -> np.ndarray:
    """Step-aligned edges containing the closed interval [lower, upper]."""
    if step <= 0:
        raise ValueError("Grid step must be positive")
    start = math.floor(lower / step) * step
    stop = math.ceil(upper / step) * step
    if stop <= start:
        stop = start + step
    count = round((stop - start) / step)
    return start + np.arange(count + 1, dtype=np.float64) * step


def field_grid(
    x: np.ndarray,
    valid: np.ndarray,
    duration_seconds: float,
    time_bin_seconds: float = 1.0,
    space_bin_meters: float = 15.0,
) -> FieldGrid:
    points = np.asarray(x, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    if points.size == 0:
        raise ValueError("Cannot construct a field grid without valid positions")
    return FieldGrid(
        regular_edges(0.0, duration_seconds, time_bin_seconds),
        regular_edges(float(points.min()), float(points.max()), space_bin_meters),
    )


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.ones(1, dtype=np.float64)
    radius = max(1, math.ceil(3.0 * sigma_bins))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    return kernel / kernel.sum()


def _convolve_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if len(kernel) == 1:
        return values.copy()
    radius = len(kernel) // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="constant")
    return np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="valid"), axis, padded
    )


def _smooth(values: np.ndarray, time_sigma_bins: float, space_sigma_bins: float) -> np.ndarray:
    values = _convolve_axis(values, _gaussian_kernel(space_sigma_bins), axis=0)
    return _convolve_axis(values, _gaussian_kernel(time_sigma_bins), axis=1)


def build_speed_field(
    times: np.ndarray,
    positions: np.ndarray,
    speeds: np.ndarray,
    valid: np.ndarray,
    grid: FieldGrid,
    *,
    min_samples: int = 1,
    smooth_time_seconds: float = 0.0,
    smooth_space_meters: float = 0.0,
) -> SpeedField:
    """Average samples per cell, optionally smoothing sums and counts.

    Smoothing is sample-weighted: the speed numerator and observation count
    are convolved independently and divided afterward.  Empty cells remain
    NaN; they are never interpreted as stopped traffic.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be at least one")
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (times.shape == positions.shape == speeds.shape == valid.shape):
        raise ValueError("Field sample arrays must have matching shapes")
    usable = valid & np.isfinite(times) & np.isfinite(positions) & np.isfinite(speeds)
    shape = (len(grid.position_edges) - 1, len(grid.time_edges) - 1)
    if not np.any(usable):
        return SpeedField(np.full(shape, np.nan), np.zeros(shape))
    count, _, _ = np.histogram2d(
        positions[usable], times[usable], bins=(grid.position_edges, grid.time_edges)
    )
    speed_sum, _, _ = np.histogram2d(
        positions[usable],
        times[usable],
        bins=(grid.position_edges, grid.time_edges),
        weights=speeds[usable],
    )
    eligible = count >= min_samples
    numerator = np.where(eligible, speed_sum, 0.0)
    denominator = np.where(eligible, count, 0.0)
    dt = float(grid.time_edges[1] - grid.time_edges[0])
    dx = float(grid.position_edges[1] - grid.position_edges[0])
    numerator = _smooth(
        numerator, smooth_time_seconds / dt, smooth_space_meters / dx
    )
    denominator = _smooth(
        denominator, smooth_time_seconds / dt, smooth_space_meters / dx
    )
    values = np.full(shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=values, where=denominator > np.finfo(float).eps)
    return SpeedField(values, count)


def velocity_error(
    ground_truth: SpeedField,
    simulation: SpeedField,
    *,
    zero_tolerance_mps: float = 1.0e-6,
    stabilized_floor_mps: float = 1.0,
) -> dict[str, float | int]:
    """Velocity errors on common support plus explicit support diagnostics.

    Conventional MAPE excludes exactly-zero ground-truth cells because their
    percentage error is undefined.  ``stabilized_mape_percent`` includes all
    common cells with a 1 m/s denominator floor by default.
    """
    gt = np.asarray(ground_truth.values, dtype=np.float64)
    sim = np.asarray(simulation.values, dtype=np.float64)
    if gt.shape != sim.shape:
        raise ValueError("Ground-truth and simulated fields use different grids")
    gt_support = np.isfinite(gt)
    sim_support = np.isfinite(sim)
    common = gt_support & sim_support
    gt_n = int(gt_support.sum())
    sim_n = int(sim_support.sum())
    common_n = int(common.sum())
    if common_n == 0:
        raise ValueError("Fields have no common support")
    truth = gt[common]
    predicted = sim[common]
    delta = predicted - truth
    nonzero = np.abs(truth) > zero_tolerance_mps
    mape = float(np.mean(np.abs(delta[nonzero]) / np.abs(truth[nonzero])) * 100.0) \
        if np.any(nonzero) else float("nan")
    stabilized = float(
        np.mean(np.abs(delta) / np.maximum(np.abs(truth), stabilized_floor_mps)) * 100.0
    )
    return {
        "mape_percent": mape,
        "stabilized_mape_percent": stabilized,
        "mae_mps": float(np.mean(np.abs(delta))),
        "rmse_mps": float(np.sqrt(np.mean(delta * delta))),
        "bias_mps": float(np.mean(delta)),
        "common_cells": common_n,
        "mape_cells": int(nonzero.sum()),
        "zero_gt_cells": int((~nonzero).sum()),
        "gt_support_cells": gt_n,
        "sim_support_cells": sim_n,
        "missing_support_cells": int((gt_support & ~sim_support).sum()),
        "extra_support_cells": int((sim_support & ~gt_support).sum()),
        "support_recall": common_n / gt_n if gt_n else float("nan"),
        "support_precision": common_n / sim_n if sim_n else float("nan"),
        "stabilized_floor_mps": float(stabilized_floor_mps),
    }


def pool_direction_errors(
    ground_truth: dict[int, SpeedField], simulation: dict[int, SpeedField], **kwargs
) -> dict[str, float | int]:
    """Pool EB/WB cells before computing metrics (cell-weighted, not direction-weighted)."""
    directions = sorted(set(ground_truth) & set(simulation))
    if not directions:
        raise ValueError("No matching travel directions")
    gt_values = np.concatenate([ground_truth[d].values.ravel() for d in directions])
    sim_values = np.concatenate([simulation[d].values.ravel() for d in directions])
    gt_count = np.concatenate([ground_truth[d].sample_count.ravel() for d in directions])
    sim_count = np.concatenate([simulation[d].sample_count.ravel() for d in directions])
    return velocity_error(
        SpeedField(gt_values, gt_count), SpeedField(sim_values, sim_count), **kwargs
    )

