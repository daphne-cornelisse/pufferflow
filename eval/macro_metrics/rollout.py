"""Dynamic-entry, all-RL full-scene rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import DriveRuntime, PolicySession


def load_scenario(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def ground_truth_arrays(scenario: dict[str, Any]) -> dict[str, np.ndarray]:
    objects = scenario["objects"]
    x = np.asarray(
        [[point["x"] for point in obj["position"]] for obj in objects], dtype=np.float32
    )
    y = np.asarray(
        [[point["y"] for point in obj["position"]] for obj in objects], dtype=np.float32
    )
    vx = np.asarray(
        [[point["x"] for point in obj["velocity"]] for obj in objects], dtype=np.float32
    )
    vy = np.asarray(
        [[point["y"] for point in obj["velocity"]] for obj in objects], dtype=np.float32
    )
    heading = np.asarray(
        [obj["heading"] for obj in objects], dtype=np.float32
    )
    return {
        "ids": np.asarray([obj["id"] for obj in objects], dtype=np.int64),
        "direction": np.asarray([obj["source_direction"] for obj in objects], dtype=np.int8),
        "x": x,
        "y": y,
        "heading": heading,
        "speed": np.hypot(vx, vy).astype(np.float32),
        "valid": np.asarray([obj["valid"] for obj in objects], dtype=bool),
        "length": np.asarray([obj["length"] for obj in objects], dtype=np.float32),
        "width": np.asarray([obj["width"] for obj in objects], dtype=np.float32),
    }


def simulate_rollout(
    runtime: DriveRuntime,
    handle,
    scenario: dict[str, Any],
    policy: PolicySession,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run one scene clock, controlling every source-valid vehicle with RL."""
    source = ground_truth_arrays(scenario)
    source_valid = source["valid"]
    agents, frames = source_valid.shape
    if int(runtime.lib.pf_n(handle)) != agents or int(runtime.lib.pf_log_n(handle)) != agents:
        raise RuntimeError("Runtime slots do not match the full-scene JSON")
    if int(runtime.lib.pf_log_traj(handle)) != frames:
        raise RuntimeError("Runtime clock does not match the full-scene JSON")
    runtime.lib.pf_allow_all_logs(handle)
    runtime.lib.pf_reset(handle, int(seed))
    mean_x, mean_y = runtime.mean_xy(handle)
    x = np.full((agents, frames), np.nan, dtype=np.float32)
    y = np.full((agents, frames), np.nan, dtype=np.float32)
    heading = np.full((agents, frames), np.nan, dtype=np.float32)
    speed = np.full((agents, frames), np.nan, dtype=np.float32)
    collision_state = np.full((agents, frames), -1, dtype=np.int8)
    action_accel_bin = np.full((agents, frames), np.nan, dtype=np.float32)
    action_steer_bin = np.full((agents, frames), np.nan, dtype=np.float32)
    active_out = np.zeros((agents, frames), dtype=bool)
    previous_active = np.zeros(agents, dtype=bool)

    for frame in range(frames):
        runtime.lib.pf_set_timestep(handle, frame)
        runtime.lib.pf_censor_invalid(handle, frame)
        runtime.lib.pf_spawn_due(handle)
        runtime.lib.pf_recompute(handle)
        states = runtime.states(handle)
        active = states[:, 10] > 0.5
        if not np.array_equal(active, source_valid[:, frame]):
            missing = np.flatnonzero(source_valid[:, frame] & ~active)
            extra = np.flatnonzero(~source_valid[:, frame] & active)
            raise RuntimeError(
                f"Lifecycle mismatch at frame {frame}: missing={missing[:5]}, extra={extra[:5]}"
            )
        x[active, frame] = states[active, 0] + mean_x
        y[active, frame] = states[active, 1] + mean_y
        heading[active, frame] = states[active, 2]
        speed[active, frame] = np.hypot(states[active, 3], states[active, 4])
        collision_state[active, frame] = states[active, 9].astype(np.int8)
        active_out[:, frame] = active
        if frame + 1 == frames:
            break

        observations = runtime.observations(handle)
        terminals = ((~active) | (active & ~previous_active)).astype(np.float32)
        actions = policy.act(observations, terminals)
        action_accel_bin[active, frame] = actions[active, 0]
        action_steer_bin[active, frame] = actions[active, 1]
        runtime.set_actions(handle, actions)
        runtime.lib.pf_integrate(handle, 0)
        previous_active = active

    return {
        "ids": source["ids"],
        "direction": source["direction"],
        "x": x,
        "y": y,
        "heading": heading,
        "speed": speed,
        "collision_state": collision_state,
        "action_accel_bin": action_accel_bin,
        "action_steer_bin": action_steer_bin,
        "valid": active_out,
        "length": source["length"],
        "width": source["width"],
    }
