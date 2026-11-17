"""Convert I24-3D ground-truth splines into PufferDrive logs.

A log is one self-driving car's track at 10 Hz, both endpoints
included, capped at the longest I24 scene (90 s, 901 states).
The record is the one in utils/common.py:

  data/<name>/json/<id>.json
  data/<name>/binaries/map_%03d.bin
  data/<name>/binaries/sdc.bin

Sorted json order is the map index. The binary is the log record
load_map_binary reads (objects, then roads). The trajectory length in
that header is this scenario's state count. sdc.bin is the object
index of the self-driving car, one int32 per map. Appendix A.4 of
arXiv 2606.19370 then infers actions and replays them in the
simulator. This script only builds the logs.

Source positions and sizes are feet. Exported distances are meters.
Only the Python standard library is required.

  python utils/i24.py --i24-root /path/to/I24-3D --output-dir data/i24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.common import (  # noqa: E402
    CYCLIST,
    DRIVE_LOG_MAX,
    LANE_FREEWAY,
    MAX_OBJECTS,
    OBJECT_TYPES,
    PEDESTRIAN,
    ROAD_EDGE_BOUNDARY,
    ROAD_EDGE_MEDIAN,
    ROAD_LINE_BROKEN_SINGLE_WHITE,
    SAMPLE_HZ,
    VEHICLE,
    drive_road_type,
    validate_scenario,
    write_log_binary,
    write_sdc_index,
)


FEET_TO_METERS = 0.3048
# The simulator stores one observation per segment midpoint, in 5 m
# grid cells, and only cells within about 50 m of the agent. A single
# chord across the whole highway has one midpoint, often far from the
# car, so the observation is empty. Sample every road at the cell size.
ROAD_SAMPLE_METERS = 5.0
# Longest I24 scene clock. Inclusive 10 Hz states: t = 0 through t = 90.
MAX_SCENARIO_SECONDS = 90
MAX_TRAJECTORY_STATES = MAX_SCENARIO_SECONDS * SAMPLE_HZ + 1
assert MAX_TRAJECTORY_STATES == DRIVE_LOG_MAX
HEADING_WINDOW_HALF_SECONDS = 0.5
HEADING_MIN_RELIABLE_SPEED_MPS = 0.5
MAJOR_QUALITY_FLAGS = {
    "speed_above_60_mps",
    "more_than_200m_outside_calibrated_x_range",
    "outside_calibrated_y_range",
}

# I24-3D body styles. PufferDrive stores every one of these as a vehicle.
# Scene 3 labels one car-sized track (14.8 x 6.0 ft) as "cs".
I24_VEHICLE_CLASSES = {
    "sedan",
    "midsize",
    "pickup",
    "van",
    "semi",
    "truck",
    "cs",
}


def drive_object_type(source_class: str) -> str:
    """Map an I24 label onto a PufferDrive object category."""
    name = source_class.strip().lower()
    if name in I24_VEHICLE_CLASSES or name in ("", "vehicle"):
        return OBJECT_TYPES[VEHICLE]
    if name in (OBJECT_TYPES[PEDESTRIAN], OBJECT_TYPES[CYCLIST]):
        return name
    raise ValueError(f"I24 class {source_class!r} is not a PufferDrive object type")


def _point(x: float, y: float, z: float = 0.0) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z)}


def _cluster(values: list[float], tolerance: float = 0.1) -> list[float]:
    """Merge nearly identical calibration coordinates from overlapping cameras."""
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or abs(value - sum(groups[-1]) / len(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def load_road_grid(homography_path: Path) -> dict[str, Any]:
    """Derive lane geometry from the 12-foot I24-3D calibration grid."""
    with homography_path.open() as file:
        homographies = json.load(file)

    points: list[tuple[float, float]] = []
    for direction in ("EB", "WB"):
        for camera in homographies[direction].values():
            points.extend((float(x), float(y)) for x, y in camera["space_pts"])

    x_values = [point[0] for point in points]
    y_values = _cluster([point[1] for point in points])
    if len(y_values) < 4:
        raise ValueError(f"Could not recover lane boundaries from {homography_path}")

    gaps = [right - left for left, right in zip(y_values, y_values[1:])]
    median_index = max(range(len(gaps)), key=gaps.__getitem__)
    carriageways = [y_values[: median_index + 1], y_values[median_index + 1 :]]
    if any(len(boundaries) < 2 for boundaries in carriageways):
        raise ValueError(f"Could not split the two carriageways in {homography_path}")

    x_min_ft, x_max_ft = min(x_values), max(x_values)
    roads: list[dict[str, Any]] = []
    road_id = 0
    for carriageway_index, boundaries_ft in enumerate(carriageways):
        # The lower-y carriageway is eastbound (+x). Westbound centerlines
        # run the other way so they follow traffic.
        direction = 1 if carriageway_index == 0 else -1
        start_x, end_x = (x_min_ft, x_max_ft) if direction == 1 else (x_max_ft, x_min_ft)

        for lower, upper in zip(boundaries_ft, boundaries_ft[1:]):
            center_y = 0.5 * (lower + upper)
            roads.append(
                {
                    "id": road_id,
                    "type": "lane",
                    "map_element_id": LANE_FREEWAY,
                    "geometry": [
                        _point(start_x * FEET_TO_METERS, center_y * FEET_TO_METERS),
                        _point(end_x * FEET_TO_METERS, center_y * FEET_TO_METERS),
                    ],
                }
            )
            road_id += 1

        for divider_y in boundaries_ft[1:-1]:
            roads.append(
                {
                    "id": road_id,
                    "type": "road_line",
                    "map_element_id": ROAD_LINE_BROKEN_SINGLE_WHITE,
                    "geometry": [
                        _point(x_min_ft * FEET_TO_METERS, divider_y * FEET_TO_METERS),
                        _point(x_max_ft * FEET_TO_METERS, divider_y * FEET_TO_METERS),
                    ],
                }
            )
            road_id += 1

        for boundary_index in (0, len(boundaries_ft) - 1):
            boundary_y = boundaries_ft[boundary_index]
            is_median = (
                carriageway_index == 0 and boundary_index == len(boundaries_ft) - 1
            ) or (carriageway_index == 1 and boundary_index == 0)
            roads.append(
                {
                    "id": road_id,
                    "type": "road_edge",
                    "map_element_id": ROAD_EDGE_MEDIAN if is_median else ROAD_EDGE_BOUNDARY,
                    "geometry": [
                        _point(x_min_ft * FEET_TO_METERS, boundary_y * FEET_TO_METERS),
                        _point(x_max_ft * FEET_TO_METERS, boundary_y * FEET_TO_METERS),
                    ],
                }
            )
            road_id += 1

    return {
        "source": homography_path.name,
        "boundary_y_m": [value * FEET_TO_METERS for value in y_values],
        "x_range_m": [x_min_ft * FEET_TO_METERS, x_max_ft * FEET_TO_METERS],
        "roads": roads,
    }


def _interpolate(
    timestamps: list[float], values: list[float], target: float, index: int
) -> tuple[float, int]:
    while index + 1 < len(timestamps) and timestamps[index + 1] < target:
        index += 1
    if index + 1 >= len(timestamps):
        return values[-1], len(timestamps) - 2
    left_t, right_t = timestamps[index], timestamps[index + 1]
    if right_t <= left_t:
        return values[index], index
    fraction = max(0.0, min(1.0, (target - left_t) / (right_t - left_t)))
    return values[index] + fraction * (values[index + 1] - values[index]), index


def _stabilized_headings(
    positions: list[list[float]], fallback_heading: float
) -> list[float]:
    """Infer heading from a local regression, holding it while motion is ambiguous."""
    radius = round(HEADING_WINDOW_HALF_SECONDS * SAMPLE_HZ)
    candidates: list[Optional[float]] = []
    for center in range(len(positions)):
        begin = max(0, center - radius)
        end = min(len(positions), center + radius + 1)
        mean_index = sum(range(begin, end)) / (end - begin)
        denominator = sum((index - mean_index) ** 2 for index in range(begin, end))
        if denominator == 0:
            candidates.append(None)
            continue
        vx = (
            sum((index - mean_index) * positions[index][0] for index in range(begin, end))
            / denominator
            * SAMPLE_HZ
        )
        vy = (
            sum((index - mean_index) * positions[index][1] for index in range(begin, end))
            / denominator
            * SAMPLE_HZ
        )
        if math.hypot(vx, vy) >= HEADING_MIN_RELIABLE_SPEED_MPS:
            candidates.append(math.atan2(vy, vx))
        else:
            candidates.append(None)

    first_reliable = next((value for value in candidates if value is not None), None)
    if first_reliable is None:
        return [fallback_heading] * len(positions)

    current = first_reliable
    headings = []
    for candidate in candidates:
        if candidate is not None:
            current = candidate
        headings.append(current)
    return headings


def _heading_estimation_metadata(stabilize_headings: bool) -> dict[str, Any]:
    if stabilize_headings:
        return {
            "stabilized": True,
            "method": "centered_linear_regression_on_position",
            "window_half_width_seconds": HEADING_WINDOW_HALF_SECONDS,
            "minimum_reliable_speed_mps": HEADING_MIN_RELIABLE_SPEED_MPS,
            "low_speed_policy": "hold_last_reliable_heading",
            "initial_low_speed_policy": "backfill_first_reliable_heading",
            "positions_modified": False,
            "velocities_modified": False,
        }
    return {
        "stabilized": False,
        "method": "centered_finite_difference_velocity",
        "minimum_speed_mps": 0.05,
        "low_speed_policy": "source_direction_fallback",
        "positions_modified": False,
        "velocities_modified": False,
    }


def normalize_object(
    raw: dict[str, Any], road_grid: dict[str, Any], stabilize_headings: bool = False
) -> dict[str, Any]:
    """Resample one native spline at 10 Hz. Positions are not smoothed."""
    timestamps = [float(value) for value in raw["timestamp"]]
    xs = [float(value) for value in raw["x_position"]]
    ys = [float(value) for value in raw["y_position"]]
    if not (len(timestamps) == len(xs) == len(ys)) or len(timestamps) < 2:
        raise ValueError(f"Trajectory {raw.get('id')} has inconsistent or insufficient samples")

    triples = sorted(zip(timestamps, xs, ys))
    deduped: list[tuple[float, float, float]] = []
    for sample in triples:
        if deduped and sample[0] <= deduped[-1][0]:
            if sample[0] == deduped[-1][0]:
                deduped[-1] = sample
                continue
            raise ValueError(f"Trajectory {raw.get('id')} has decreasing timestamps")
        deduped.append(sample)
    timestamps, xs, ys = map(list, zip(*deduped))

    first_tick = math.ceil(timestamps[0] * SAMPLE_HZ - 1e-5)
    last_tick = math.floor(timestamps[-1] * SAMPLE_HZ + 1e-5)
    if last_tick < first_tick:
        raise ValueError(f"Trajectory {raw.get('id')} is shorter than one 10 Hz state")

    positions: list[list[float]] = []
    source_index = 0
    for tick in range(first_tick, last_tick + 1):
        target = tick / SAMPLE_HZ
        x, source_index = _interpolate(timestamps, xs, target, source_index)
        y, source_index = _interpolate(timestamps, ys, target, source_index)
        positions.append([x * FEET_TO_METERS, y * FEET_TO_METERS])

    velocities: list[list[float]] = []
    for index in range(len(positions)):
        if len(positions) == 1:
            vx = vy = 0.0
        elif index == 0:
            vx = (positions[1][0] - positions[0][0]) * SAMPLE_HZ
            vy = (positions[1][1] - positions[0][1]) * SAMPLE_HZ
        elif index == len(positions) - 1:
            vx = (positions[-1][0] - positions[-2][0]) * SAMPLE_HZ
            vy = (positions[-1][1] - positions[-2][1]) * SAMPLE_HZ
        else:
            vx = (positions[index + 1][0] - positions[index - 1][0]) * SAMPLE_HZ / 2
            vy = (positions[index + 1][1] - positions[index - 1][1]) * SAMPLE_HZ / 2
        velocities.append([vx, vy])

    fallback_heading = 0.0 if int(raw.get("direction", 1)) >= 0 else math.pi
    if stabilize_headings:
        headings = _stabilized_headings(positions, fallback_heading)
    else:
        headings = [
            math.atan2(vy, vx) if math.hypot(vx, vy) > 0.05 else fallback_heading
            for vx, vy in velocities
        ]
    speeds = [math.hypot(vx, vy) for vx, vy in velocities]

    quality_flags: list[str] = []
    duration = (last_tick - first_tick) / SAMPLE_HZ
    # Descriptive only. Short tracks are still exported at their own length.
    if duration < 9.0:
        quality_flags.append("shorter_than_9_seconds")
    if max(speeds, default=0.0) > 60.0:
        quality_flags.append("speed_above_60_mps")
    map_x_min, map_x_max = road_grid["x_range_m"]
    map_y_min = road_grid["boundary_y_m"][0]
    map_y_max = road_grid["boundary_y_m"][-1]
    # Vehicles enter and leave beyond the overlapping camera grid. Only
    # flag extreme extrapolation, not that run-in and run-out.
    margin = 200.0
    if any(x < map_x_min - margin or x > map_x_max + margin for x, _ in positions):
        quality_flags.append("more_than_200m_outside_calibrated_x_range")
    if any(y < map_y_min - margin or y > map_y_max + margin for _, y in positions):
        quality_flags.append("outside_calibrated_y_range")

    length_m = float(raw.get("l", 0.0)) * FEET_TO_METERS
    width_m = float(raw.get("w", 0.0)) * FEET_TO_METERS
    height_m = float(raw.get("h", 0.0)) * FEET_TO_METERS
    if length_m < 0.5:
        length_m = 4.5
    if width_m < 0.2:
        width_m = 1.8
    if height_m <= 0.0:
        height_m = 1.5

    source_class = str(raw.get("class", "vehicle"))
    normalized = {
        "id": int(raw["id"]),
        "source_class": source_class,
        "type": drive_object_type(source_class),
        "direction": int(raw.get("direction", 0)),
        "length_m": length_m,
        "width_m": width_m,
        "height_m": height_m,
        "start_tick": first_tick,
        "sample_hz": SAMPLE_HZ,
        "position_m": positions,
        "velocity_mps": velocities,
        "heading_rad": headings,
        "source_sample_count": len(raw["timestamp"]),
    }
    if quality_flags:
        normalized["quality_flags"] = quality_flags
    return normalized


def normalize_scene(
    i24_root: Path, scene: int, stabilize_headings: bool = False
) -> dict[str, Any]:
    splines_path = i24_root / "data" / "spl_obj" / f"scene{scene}_splobj.json"
    homography_path = i24_root / "data" / "hg" / f"scene{scene}_hg.json"
    with splines_path.open() as file:
        raw_objects = json.load(file)
    road_grid = load_road_grid(homography_path)
    objects = [
        normalize_object(raw, road_grid, stabilize_headings=stabilize_headings)
        for raw in raw_objects
    ]
    return {
        "schema": "i24_3d_full_trajectory_v1",
        "scene": scene,
        "sample_hz": SAMPLE_HZ,
        "units": {
            "position": "meters",
            "velocity": "meters_per_second",
            "heading": "radians",
        },
        "source": str(splines_path),
        "object_count": len(objects),
        "heading_estimation": _heading_estimation_metadata(stabilize_headings),
        "map": road_grid,
        "objects": objects,
    }


def _object_end_tick(obj: dict[str, Any]) -> int:
    return int(obj["start_tick"]) + len(obj["position_m"]) - 1


def choose_sdc(scene_data: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        obj
        for obj in scene_data["objects"]
        if len(obj["position_m"]) >= 2
        and "speed_above_60_mps" not in obj.get("quality_flags", [])
        and "outside_calibrated_y_range" not in obj.get("quality_flags", [])
    ]
    if not candidates:
        raise ValueError("No trajectory has two or more states")
    candidates.sort(
        key=lambda obj: (len(obj.get("quality_flags", [])), -len(obj["position_m"]), obj["id"])
    )
    return candidates[0]


def _state_at_tick(obj: dict[str, Any], tick: int) -> tuple[list[float], list[float], float, int]:
    index = tick - int(obj["start_tick"])
    valid = 0 <= index < len(obj["position_m"])
    clamped = max(0, min(index, len(obj["position_m"]) - 1))
    position = obj["position_m"][clamped]
    velocity = obj["velocity_mps"][clamped] if valid else [0.0, 0.0]
    heading = obj["heading_rad"][clamped]
    return position, velocity, heading, int(valid)


def _sample_polyline(geometry: list[dict[str, float]], spacing: float) -> list[dict[str, float]]:
    """Keep both endpoints and insert points every `spacing` meters."""
    if len(geometry) < 2 or spacing <= 0:
        return geometry
    sampled = [geometry[0]]
    for start, end in zip(geometry, geometry[1:]):
        dx = end["x"] - start["x"]
        dy = end["y"] - start["y"]
        dz = end.get("z", 0.0) - start.get("z", 0.0)
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        steps = int(length // spacing)
        for step in range(1, steps + 1):
            t = (step * spacing) / length
            if t >= 1.0:
                break
            sampled.append(_point(
                start["x"] + t * dx,
                start["y"] + t * dy,
                start.get("z", 0.0) + t * dz,
            ))
        sampled.append(_point(end["x"], end["y"], end.get("z", 0.0)))
    return sampled


def _translated_roads(
    roads: list[dict[str, Any]],
    origin: list[float],
    x_bounds: Optional[tuple[float, float]] = None,
) -> list[dict[str, Any]]:
    translated = []
    for road in roads:
        geometry = [
            _point(point["x"] - origin[0], point["y"] - origin[1], point.get("z", 0.0))
            for point in road["geometry"]
        ]
        if x_bounds is not None and len(geometry) == 2:
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


def build_scenario(
    scene_data: dict[str, Any],
    sdc_id: int,
    start_tick: int,
    window_index: int,
    max_objects: int,
    state_count: int,
    scenario_id: Optional[str] = None,
    full_range_context: bool = False,
) -> dict[str, Any]:
    """One PufferDrive scenario. The self-driving car is object 0."""
    objects_by_id = {obj["id"]: obj for obj in scene_data["objects"]}
    if sdc_id not in objects_by_id:
        raise ValueError(f"No object {sdc_id} in scene {scene_data['scene']}")
    sdc = objects_by_id[sdc_id]
    end_tick = start_tick + state_count - 1
    if start_tick < sdc["start_tick"] or end_tick > _object_end_tick(sdc):
        raise ValueError(f"SDC {sdc_id} does not cover the requested {state_count}-state range")

    origin, _, _, _ = _state_at_tick(sdc, start_tick)
    candidates = []
    for obj in scene_data["objects"]:
        if obj["id"] == sdc_id:
            continue
        if full_range_context:
            overlap_start = max(start_tick, int(obj["start_tick"]))
            overlap_end = min(end_tick, _object_end_tick(obj))
            if overlap_start > overlap_end:
                continue
            distance_sq = min(
                (obj["position_m"][tick - obj["start_tick"]][0]
                 - sdc["position_m"][tick - sdc["start_tick"]][0]) ** 2
                + (obj["position_m"][tick - obj["start_tick"]][1]
                   - sdc["position_m"][tick - sdc["start_tick"]][1]) ** 2
                for tick in range(overlap_start, overlap_end + 1)
            )
        else:
            position, _, _, valid = _state_at_tick(obj, start_tick)
            if not valid:
                continue
            distance_sq = (position[0] - origin[0]) ** 2 + (position[1] - origin[1]) ** 2
        candidates.append((distance_sq, obj["id"], obj))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = [sdc] + [item[2] for item in candidates[: max_objects - 1]]

    scenario_objects = []
    for obj in selected:
        positions = []
        velocities = []
        headings = []
        valids = []
        for tick in range(start_tick, end_tick + 1):
            position, velocity, heading, valid = _state_at_tick(obj, tick)
            positions.append(_point(position[0] - origin[0], position[1] - origin[1]))
            velocities.append(_point(velocity[0], velocity[1]))
            headings.append(float(heading))
            valids.append(bool(valid))
        final_valid_index = max(index for index, valid in enumerate(valids) if valid)
        scenario_objects.append(
            {
                "id": int(obj["id"]),
                "type": obj["type"],
                "length": obj["length_m"],
                "width": obj["width_m"],
                "height": obj["height_m"],
                "position": positions,
                "velocity": velocities,
                "heading": headings,
                "valid": valids,
                "goalPosition": positions[final_valid_index],
                "mark_as_expert": 0,
            }
        )

    if scenario_id is None:
        scenario_id = f"i24s{scene_data['scene']}v{sdc_id}w{window_index}"
    all_x = [
        point["x"]
        for obj in scenario_objects
        for point, valid in zip(obj["position"], obj["valid"])
        if valid and math.isfinite(point["x"])
    ]
    all_y = [
        point["y"]
        for obj in scenario_objects
        for point, valid in zip(obj["position"], obj["valid"])
        if valid and math.isfinite(point["y"])
    ]
    # The calibration grid covers the camera overlap. These scenes are
    # straight, so extend the recovered lines over the exported window.
    roads = _translated_roads(
        scene_data["map"]["roads"], origin, (min(all_x) - 50.0, max(all_x) + 50.0)
    )
    return {
        "name": scenario_id,
        "scenario_id": scenario_id,
        "objects": scenario_objects,
        "roads": roads,
        "tl_states": [],
        "metadata": {
            "sdc_track_index": 0,
            "frame_interval_micros": 100_000,
            "frame_count": state_count,
            "duration_seconds": (state_count - 1) / SAMPLE_HZ,
            "source_scene": scene_data["scene"],
            "source_sdc_id": sdc_id,
            "source_start_time_unix": start_tick / SAMPLE_HZ,
            "heading_estimation": scene_data.get(
                "heading_estimation", _heading_estimation_metadata(False)
            ),
            "coordinate_origin_m": {"x": origin[0], "y": origin[1]},
        },
    }


def build_track_scenario(
    scene_data: dict[str, Any], sdc_id: int, max_objects: int
) -> dict[str, Any]:
    """One scenario for the self-driving car's whole track, capped at 90 s."""
    objects_by_id = {obj["id"]: obj for obj in scene_data["objects"]}
    if sdc_id not in objects_by_id:
        raise ValueError(f"No object {sdc_id} in scene {scene_data['scene']}")
    sdc = objects_by_id[sdc_id]
    state_count = len(sdc["position_m"])
    if state_count < 2:
        raise ValueError(f"SDC {sdc_id} has fewer than two states")
    if state_count > MAX_TRAJECTORY_STATES:
        state_count = MAX_TRAJECTORY_STATES
    return build_scenario(
        scene_data,
        sdc_id=sdc_id,
        start_tick=int(sdc["start_tick"]),
        window_index=0,
        max_objects=max_objects,
        state_count=state_count,
        scenario_id=f"i24s{scene_data['scene']}v{sdc_id}",
        full_range_context=True,
    )


def write_dataset(scenarios: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Write json/, binaries/map_%03d.bin, and binaries/sdc.bin.

    Map i is the i-th scenario. Filenames sort in that order. Each
    binary stores that scenario's own state count.
    """
    json_dir = output_dir / "json"
    binary_dir = output_dir / "binaries"
    json_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sdc_indices: list[int] = []
    for map_index, scenario in enumerate(scenarios):
        validate_scenario(scenario)
        name = f"i24_{map_index:05d}.json"
        path = json_dir / name
        scenario = dict(scenario)
        scenario["name"] = name
        scenario["scenario_id"] = Path(name).stem
        _write_json(path, scenario)
        write_log_binary(scenario, binary_dir / f"map_{map_index:03d}.bin")
        sdc_indices.append(int(scenario["metadata"]["sdc_track_index"]))
        written.append(path)
    if sdc_indices:
        write_sdc_index(binary_dir / "sdc.bin", sdc_indices)
    return written


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(value, file)
        file.write("\n")


def _selected_sdcs(scene_data: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_sdcs:
        return sorted(scene_data["objects"], key=lambda obj: obj["id"])
    if args.sdc_id is not None:
        matching = [obj for obj in scene_data["objects"] if obj["id"] == args.sdc_id]
        if not matching:
            raise ValueError(f"No object {args.sdc_id} in scene {scene_data['scene']}")
        return matching
    return [choose_sdc(scene_data)]


def convert(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = (1, 2, 3) if args.all_scenes else (args.scene,)
    scenarios: list[dict[str, Any]] = []
    selected_sdcs: list[dict[str, int]] = []
    ineligible_sdcs: list[dict[str, Any]] = []
    excluded_sdcs: list[dict[str, Any]] = []
    flag_counts: Counter = Counter()
    scene_summaries = []
    source_object_count = 0

    for scene in scenes:
        scene_data = normalize_scene(
            args.i24_root.resolve(), scene, stabilize_headings=args.stabilize_headings
        )
        source_object_count += scene_data["object_count"]
        scene_flag_counts = Counter(
            flag for obj in scene_data["objects"] for flag in obj.get("quality_flags", [])
        )
        flag_counts.update(scene_flag_counts)
        if args.write_full:
            _write_json(
                args.output_dir / f"scene{scene}_full_10hz.json", scene_data
            )

        scene_eligible = 0
        scene_exported = 0
        scene_excluded = 0
        scene_scenarios_before = len(scenarios)
        for sdc in _selected_sdcs(scene_data, args):
            if len(sdc["position_m"]) < 2:
                if args.all_sdcs:
                    ineligible_sdcs.append(
                        {"scene": scene, "id": sdc["id"], "reason": "shorter_than_two_states"}
                    )
                    continue
                raise ValueError(f"SDC {sdc['id']} has fewer than two states")
            scene_eligible += 1

            major_flags = sorted(MAJOR_QUALITY_FLAGS & set(sdc.get("quality_flags", [])))
            if args.exclude_major_flags and major_flags:
                excluded_sdcs.append({"scene": scene, "id": sdc["id"], "flags": major_flags})
                scene_excluded += 1
                continue

            selected_sdcs.append({"scene": scene, "id": sdc["id"]})
            scene_exported += 1
            scenario = build_track_scenario(scene_data, sdc["id"], args.max_objects)
            validate_scenario(scenario, args.max_objects)
            scenarios.append(scenario)

        scene_summaries.append(
            {
                "scene": scene,
                "source_object_count": scene_data["object_count"],
                "eligible_sdc_count": scene_eligible,
                "exported_sdc_count": scene_exported,
                "ineligible_sdc_count": sum(
                    1 for item in ineligible_sdcs if item["scene"] == scene
                ),
                "excluded_major_quality_sdc_count": scene_excluded,
                "scenario_count": len(scenarios) - scene_scenarios_before,
                "quality_flag_counts": dict(sorted(scene_flag_counts.items())),
            }
        )

    written = write_dataset(scenarios, args.output_dir)
    frame_counts = [int(scenario["metadata"]["frame_count"]) for scenario in scenarios]
    manifest = {
        "schema": "i24_pufferdrive_v1",
        "layout": "pufferdrive",
        "scenes": list(scenes),
        "source_object_count": source_object_count,
        "sample_hz": SAMPLE_HZ,
        "max_scenario_seconds": MAX_SCENARIO_SECONDS,
        "max_trajectory_states": MAX_TRAJECTORY_STATES,
        "min_frame_count": min(frame_counts) if frame_counts else 0,
        "max_frame_count": max(frame_counts) if frame_counts else 0,
        "heading_estimation": _heading_estimation_metadata(args.stabilize_headings),
        "eligible_sdc_count": len(selected_sdcs) + len(excluded_sdcs),
        "exported_sdc_count": len(selected_sdcs),
        "ineligible_sdc_count": len(ineligible_sdcs),
        "excluded_major_quality_sdc_count": len(excluded_sdcs),
        "scenario_count": len(scenarios),
        "log_binary_count": len(scenarios),
        "selected_sdcs": selected_sdcs,
        "ineligible_sdcs": ineligible_sdcs,
        "excluded_sdcs": excluded_sdcs,
        "files": [str(path.relative_to(args.output_dir)) for path in written],
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "scene_summaries": scene_summaries,
        "notes": [
            "json/ and binaries/ are PufferDrive data.",
            "map_i.bin is the i-th json. Its header length is that scenario's state count.",
            "sdc.bin holds that map's self-driving-car object index.",
            "Each track keeps its own length, capped at 90 s (901 states).",
            "Goals are each track's last valid logged position.",
            "The road model is the scene's 12-foot calibration grid, extended along the roadway and sampled every 5 m.",
            "Action labels are the simulator inverse on these logs, as in arXiv 2606.19370 A.4.",
            "Heading stabilization never modifies exported positions or velocities.",
        ],
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_i24_root = repo_root.parent / "i24_3d_dataset" / "I24-3D"
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--i24-root", type=Path, default=default_i24_root)
    argument_parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "data" / "i24"
    )
    argument_parser.add_argument("--scene", type=int, choices=(1, 2, 3), default=2)
    argument_parser.add_argument(
        "--all-scenes", action="store_true", help="Process scenes 1, 2, and 3"
    )
    argument_parser.add_argument(
        "--sdc-id", type=int, default=None,
        help="Default: the longest clean trajectory",
    )
    argument_parser.add_argument(
        "--all-sdcs", action="store_true",
        help="Export every track that has at least two states",
    )
    argument_parser.add_argument(
        "--exclude-major-flags", action="store_true",
        help="Skip focal tracks with extreme speed or range flags",
    )
    argument_parser.add_argument(
        "--stabilize-headings", action="store_true",
        help=(
            "Estimate headings over a 1.0 s position window and hold the last "
            "reliable orientation below 0.5 m/s"
        ),
    )
    argument_parser.add_argument("--max-objects", type=int, default=128)
    argument_parser.add_argument(
        "--write-full", action="store_true",
        help="Also write the resampled full-scene JSON",
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    if args.all_scenes and not args.all_sdcs:
        raise SystemExit("--all-scenes requires --all-sdcs")
    if args.all_sdcs and args.sdc_id is not None:
        raise SystemExit("--all-sdcs and --sdc-id are mutually exclusive")
    if not 1 <= args.max_objects <= MAX_OBJECTS:
        raise SystemExit(f"--max-objects must be between 1 and {MAX_OBJECTS}")
    if not args.i24_root.is_dir():
        raise SystemExit(f"I24 root does not exist: {args.i24_root}")
    manifest = convert(args)
    summary = {
        "scenes": manifest["scenes"],
        "source_object_count": manifest["source_object_count"],
        "scenario_count": manifest["scenario_count"],
        "log_binary_count": manifest["log_binary_count"],
        "exported_sdc_count": manifest["exported_sdc_count"],
        "manifest": str(args.output_dir / "manifest.json"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
