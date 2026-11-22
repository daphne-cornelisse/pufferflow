"""Shared speed-field construction and scoring for macro rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .speed_field import (
    FieldGrid,
    SpeedField,
    build_speed_field,
    pool_direction_errors,
    velocity_error,
)

SAMPLE_HZ = 10
DIRECTIONS = (1, -1)
MODES = {
    "raw": (0.0, 0.0),
    "smoothed": (1.5, 30.0),
}


def trajectory_fields(
    trajectory: dict[str, np.ndarray], grid: FieldGrid, mode: str
) -> dict[int, SpeedField]:
    """Build one eastbound and westbound field on a shared grid."""
    sigma_t, sigma_x = MODES[mode]
    agents, frames = trajectory["x"].shape
    times = np.broadcast_to(
        np.arange(frames, dtype=np.float64)[None, :] / SAMPLE_HZ,
        (agents, frames),
    )
    result = {}
    for direction in DIRECTIONS:
        direction_mask = trajectory["direction"][:, None] == direction
        result[direction] = build_speed_field(
            times,
            trajectory["x"],
            trajectory["speed"],
            trajectory["valid"] & direction_mask,
            grid,
            smooth_time_seconds=sigma_t,
            smooth_space_meters=sigma_x,
        )
    return result


def score_fields(
    ground_truth: dict[int, SpeedField], simulated: dict[int, SpeedField]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return pooled and direction-specific velocity errors."""
    pooled = pool_direction_errors(ground_truth, simulated)
    by_direction = {
        ("eastbound" if direction == 1 else "westbound"): velocity_error(
            ground_truth[direction], simulated[direction]
        )
        for direction in DIRECTIONS
    }
    return pooled, by_direction
