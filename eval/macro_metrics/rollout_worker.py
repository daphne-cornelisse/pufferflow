"""One-process worker for an independently seeded macro rollout."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .field_evaluation import MODES, score_fields, trajectory_fields
from .rollout import ground_truth_arrays, load_scenario, simulate_rollout
from .rollout_archive import (
    archive_metadata,
    load_rollout_archive,
    write_rollout_archive,
)
from .runtime import DriveRuntime, PolicySession
from .speed_field import field_grid


@dataclass(frozen=True)
class RolloutJob:
    scenario_path: Path
    data_dir: Path
    map_id: int
    checkpoint: Path
    checkpoint_name: str
    checkpoint_fingerprint: str
    scenario_fingerprint: str
    implementation_fingerprint: str
    scene: int
    rollout: int
    seed: int
    archive_path: Path
    library: Path
    policy_act: Path
    hidden: int
    layers: int
    time_bin_seconds: float
    space_bin_meters: float
    resume: bool = True


def _metadata(job: RolloutJob) -> dict[str, Any]:
    return archive_metadata(
        scene=job.scene,
        rollout=job.rollout,
        policy_seed=job.seed,
        checkpoint=job.checkpoint_name,
        checkpoint_fingerprint=job.checkpoint_fingerprint,
        scenario_fingerprint=job.scenario_fingerprint,
        implementation_fingerprint=job.implementation_fingerprint,
        hidden=job.hidden,
        layers=job.layers,
    )


def run_rollout_job(job: RolloutJob) -> dict[str, Any]:
    """Simulate or resume one rollout, then build all configured fields."""
    started = time.perf_counter()
    scenario = load_scenario(job.scenario_path)
    source_scene = int(scenario["metadata"]["source_scene"])
    if source_scene != job.scene:
        raise ValueError(f"Scenario scene {source_scene} does not match job scene {job.scene}")
    metadata = _metadata(job)
    reused = job.resume and job.archive_path.is_file()
    if reused:
        trajectory = load_rollout_archive(job.archive_path, metadata)
    else:
        runtime = DriveRuntime(job.library)
        handle = runtime.create_scenario(job.data_dir, job.map_id, job.scenario_path)
        agents = int(runtime.lib.pf_n(handle))
        observation_size = int(runtime.lib.pf_obs_size(handle))
        try:
            with PolicySession(
                job.policy_act,
                job.checkpoint,
                agents,
                observation_size,
                hidden=job.hidden,
                layers=job.layers,
                seed=job.seed,
            ) as policy:
                trajectory = simulate_rollout(
                    runtime,
                    handle,
                    scenario,
                    policy,
                    seed=job.seed,
                )
        finally:
            runtime.lib.pf_close(handle)
        write_rollout_archive(job.archive_path, trajectory, metadata)

    ground_truth = ground_truth_arrays(scenario)
    duration = float(scenario["metadata"]["duration_seconds"])
    grid = field_grid(
        ground_truth["x"],
        ground_truth["valid"],
        duration,
        time_bin_seconds=job.time_bin_seconds,
        space_bin_meters=job.space_bin_meters,
    )
    ground_truth_fields = {
        mode: trajectory_fields(ground_truth, grid, mode) for mode in MODES
    }
    field_arrays: dict[str, np.ndarray] = {}
    rows = []
    for mode in MODES:
        simulated = trajectory_fields(trajectory, grid, mode)
        pooled, by_direction = score_fields(ground_truth_fields[mode], simulated)
        rows.append(
            {
                "checkpoint": job.checkpoint_name,
                "scene": job.scene,
                "rollout": job.rollout,
                "mode": mode,
                **pooled,
                "by_direction": by_direction,
            }
        )
        for direction, suffix in ((1, "eb"), (-1, "wb")):
            field_arrays[f"sim_{mode}_{suffix}"] = simulated[direction].values
            field_arrays[f"sim_count_{mode}_{suffix}"] = simulated[
                direction
            ].sample_count

    return {
        "rollout": job.rollout,
        "rows": rows,
        "fields": field_arrays,
        "archive": str(job.archive_path),
        "reused": reused,
        "elapsed_seconds": time.perf_counter() - started,
    }
