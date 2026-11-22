"""Durable, resumable trajectory archives for expensive policy rollouts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE_VERSION = 1
TRAJECTORY_KEYS = (
    "ids",
    "direction",
    "x",
    "y",
    "heading",
    "speed",
    "valid",
    "length",
    "width",
    "collision_state",
    "action_accel_bin",
    "action_steer_bin",
)


def archive_metadata(
    *,
    scene: int,
    rollout: int,
    policy_seed: int,
    checkpoint: str,
    checkpoint_fingerprint: str,
    scenario_fingerprint: str,
    implementation_fingerprint: str,
    hidden: int,
    layers: int,
) -> dict[str, Any]:
    return {
        "archive_version": np.int32(ARCHIVE_VERSION),
        "scene": np.int32(scene),
        "rollout": np.int32(rollout),
        "policy_seed": np.int64(policy_seed),
        "checkpoint": np.asarray(checkpoint),
        "checkpoint_fingerprint": np.asarray(checkpoint_fingerprint),
        "scenario_fingerprint": np.asarray(scenario_fingerprint),
        "implementation_fingerprint": np.asarray(implementation_fingerprint),
        "policy_hidden_size": np.int32(hidden),
        "policy_num_layers": np.int32(layers),
    }


def _scalar(value: np.ndarray) -> Any:
    return np.asarray(value).item()


def load_rollout_archive(
    path: Path, expected_metadata: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Load an archive only when its provenance matches the requested job."""
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in (*TRAJECTORY_KEYS, *expected_metadata) if key not in archive]
        if missing:
            raise ValueError(f"Rollout archive {path} is missing {missing}")
        mismatches = []
        for key, expected in expected_metadata.items():
            actual = _scalar(archive[key])
            wanted = _scalar(np.asarray(expected))
            if actual != wanted:
                mismatches.append(f"{key}={actual!r} (expected {wanted!r})")
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(f"Rollout archive {path} has incompatible provenance: {joined}")
        return {key: archive[key].copy() for key in TRAJECTORY_KEYS}


def write_rollout_archive(
    path: Path,
    trajectory: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Atomically write one rollout so interrupted jobs remain resumable."""
    missing = [key for key in TRAJECTORY_KEYS if key not in trajectory]
    if missing:
        raise ValueError(f"Trajectory is missing archive arrays {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                **{key: trajectory[key] for key in TRAJECTORY_KEYS},
                **metadata,
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
