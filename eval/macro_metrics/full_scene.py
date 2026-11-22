"""Build full-clock I-24 scenarios for macro evaluation.

Unlike ``utils.i24.build_scenario``, these scenarios are not centered on a
single focal vehicle.  Every retained trajectory occupies one simulator slot,
enters on its first source-valid frame, and is removed after its last one.
Scene 1 uses the converter's 90 second source-clock clip; scenes 2 and 3 use
their complete source clocks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from utils.common import DRIVE_LOG_MAX, SAMPLE_HZ, write_log_binary, write_sdc_index
from utils.i24 import (
    MAJOR_QUALITY_FLAGS,
    ROAD_SAMPLE_METERS,
    STATIONARY_MAX_METERS,
    _point,
    _position_travel_m,
    _sample_polyline,
    normalize_scene,
)

MIN_VALID_STATES = 10
MAX_FULL_SCENE_AGENTS = 512


def _eligible(obj: dict[str, Any], exclude_major_flags: bool) -> tuple[bool, str | None]:
    positions = obj.get("position_m", [])
    if len(positions) < MIN_VALID_STATES:
        return False, "fewer_than_10_states"
    points = [(float(point[0]), float(point[1])) for point in positions]
    if _position_travel_m(points) < STATIONARY_MAX_METERS:
        return False, "stationary"
    flags = set(obj.get("quality_flags", []))
    if exclude_major_flags and flags & MAJOR_QUALITY_FLAGS:
        return False, "major_quality_flag"
    return True, None


def eligible_objects(
    scene_data: dict[str, Any], exclude_major_flags: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return policy-controlled tracks and a machine-readable exclusion list."""
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for obj in scene_data["objects"]:
        ok, reason = _eligible(obj, exclude_major_flags)
        if ok:
            kept.append(obj)
        else:
            excluded.append(
                {
                    "id": int(obj["id"]),
                    "reason": reason,
                    "quality_flags": sorted(obj.get("quality_flags", [])),
                }
            )
    kept.sort(key=lambda item: int(item["id"]))
    return kept, excluded


def _translated_roads(
    roads: list[dict[str, Any]], origin_x: float, x_bounds: tuple[float, float]
) -> list[dict[str, Any]]:
    translated = []
    for road in roads:
        geometry = [
            _point(point["x"] - origin_x, point["y"], point.get("z", 0.0))
            for point in road["geometry"]
        ]
        if len(geometry) == 2:
            low = min(geometry[0]["x"], geometry[1]["x"], x_bounds[0])
            high = max(geometry[0]["x"], geometry[1]["x"], x_bounds[1])
            if geometry[0]["x"] <= geometry[1]["x"]:
                geometry[0]["x"], geometry[1]["x"] = low, high
            else:
                geometry[0]["x"], geometry[1]["x"] = high, low
        translated.append(
            {
                **{key: value for key, value in road.items() if key != "geometry"},
                "geometry": _sample_polyline(geometry, ROAD_SAMPLE_METERS),
            }
        )
    return translated


def build_full_scene(
    scene_data: dict[str, Any], exclude_major_flags: bool = True
) -> dict[str, Any]:
    """Create one global-clock scenario with no replayed background vehicles."""
    selected, excluded = eligible_objects(scene_data, exclude_major_flags)
    if not selected:
        raise ValueError(f"Scene {scene_data['scene']} has no eligible trajectories")
    if len(selected) > MAX_FULL_SCENE_AGENTS:
        raise ValueError(
            f"Scene {scene_data['scene']} needs {len(selected)} slots; "
            f"the macro runtime supports {MAX_FULL_SCENE_AGENTS}"
        )

    start_tick = min(int(obj["start_tick"]) for obj in selected)
    end_tick = max(int(obj["start_tick"]) + len(obj["position_m"]) - 1 for obj in selected)
    state_count = end_tick - start_tick + 1
    if not 2 <= state_count <= DRIVE_LOG_MAX:
        raise ValueError(f"Full scene has unsupported state count {state_count}")

    all_x = [float(point[0]) for obj in selected for point in obj["position_m"]]
    # A 15 m-aligned origin makes downstream position bins easy to interpret.
    origin_x = math.floor(min(all_x) / 15.0) * 15.0
    objects = []
    for obj in selected:
        offset = int(obj["start_tick"]) - start_tick
        count = len(obj["position_m"])
        positions = []
        velocities = []
        headings = []
        valid = []
        for frame in range(state_count):
            local = frame - offset
            present = 0 <= local < count
            source_index = max(0, min(local, count - 1))
            point = obj["position_m"][source_index]
            velocity = obj["velocity_mps"][source_index] if present else (0.0, 0.0)
            positions.append(_point(float(point[0]) - origin_x, float(point[1])))
            velocities.append(_point(float(velocity[0]), float(velocity[1])))
            headings.append(float(obj["heading_rad"][source_index]))
            valid.append(bool(present))
        final_index = offset + count - 1
        objects.append(
            {
                "id": int(obj["id"]),
                "type": obj["type"],
                "source_direction": int(obj["direction"]),
                "length": float(obj["length_m"]),
                "width": float(obj["width_m"]),
                "height": float(obj["height_m"]),
                "position": positions,
                "velocity": velocities,
                "heading": headings,
                "valid": valid,
                "goalPosition": positions[final_index],
                "mark_as_expert": 0,
            }
        )

    local_x = [value - origin_x for value in all_x]
    roads = _translated_roads(
        scene_data["map"]["roads"],
        origin_x,
        (min(local_x) - 50.0, max(local_x) + 50.0),
    )
    scene = int(scene_data["scene"])
    scenario_id = f"i24_scene{scene}_full"
    return {
        "name": scenario_id,
        "scenario_id": scenario_id,
        "objects": objects,
        "roads": roads,
        "tl_states": [],
        "metadata": {
            # Required by the existing loader, but not privileged in this eval.
            "sdc_track_index": 0,
            "frame_interval_micros": 100_000,
            "frame_count": state_count,
            "duration_seconds": (state_count - 1) / SAMPLE_HZ,
            "source_scene": scene,
            "source_start_tick": start_tick,
            "source_end_tick": end_tick,
            "coordinate_origin_m": {"x": origin_x, "y": 0.0},
            "all_active_vehicles_policy_controlled": True,
            "eligible_vehicle_count": len(objects),
            "excluded_vehicle_count": len(excluded),
            "excluded_vehicles": excluded,
            "exclude_major_quality_flags": bool(exclude_major_flags),
            "scene_clock_clip": scene_data.get("scene_clock_clip"),
        },
    }


def _validate_full_scene(scenario: dict[str, Any]) -> None:
    objects = scenario["objects"]
    states = int(scenario["metadata"]["frame_count"])
    if not objects or not scenario["roads"]:
        raise ValueError("Full-scene scenario needs vehicles and roads")
    if len(objects) > MAX_FULL_SCENE_AGENTS:
        raise ValueError("Full-scene scenario exceeds the 512-agent runtime")
    for obj in objects:
        for key in ("position", "velocity", "heading", "valid"):
            if len(obj[key]) != states:
                raise ValueError(f"Object {obj['id']} has a malformed {key} series")
        if sum(bool(value) for value in obj["valid"]) < MIN_VALID_STATES:
            raise ValueError(f"Object {obj['id']} is shorter than the eligibility rule")
        indices = [index for index, value in enumerate(obj["valid"]) if value]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"Object {obj['id']} has a non-contiguous valid interval")


def write_full_scene_dataset(
    scenarios: Iterable[dict[str, Any]], output_dir: Path
) -> list[Path]:
    """Write JSON and log binaries in the layout expected by the C runtime."""
    json_dir = output_dir / "json"
    binary_dir = output_dir / "binaries"
    json_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for map_index, scenario in enumerate(scenarios):
        _validate_full_scene(scenario)
        path = json_dir / f"i24_full_scene{scenario['metadata']['source_scene']}.json"
        with path.open("w") as file:
            json.dump(scenario, file)
            file.write("\n")
        write_log_binary(scenario, binary_dir / f"map_{map_index:03d}.bin")
        paths.append(path)
    write_sdc_index(binary_dir / "sdc.bin", [0] * len(paths))
    return paths


def generate_full_scene_dataset(
    i24_root: Path,
    output_dir: Path,
    scenes: Iterable[int] = (1, 2, 3),
    exclude_major_flags: bool = True,
) -> list[Path]:
    scenarios = [
        build_full_scene(
            normalize_scene(i24_root, int(scene)),
            exclude_major_flags=exclude_major_flags,
        )
        for scene in scenes
    ]
    return write_full_scene_dataset(scenarios, output_dir)

