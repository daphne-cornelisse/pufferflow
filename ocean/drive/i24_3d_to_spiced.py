"""Prototype converter from I24-3D ground-truth splines to spiced self-play.

The converter deliberately has two outputs:

1. A scene-level, full-length 10 Hz file that retains every ground-truth spline.
2. One or more 91-state (9.0 second) scenarios accepted by spiced_self_play,
   or full-SDC scenarios for exact PufferDrive replay.

Only the Python standard library is required. Source distances and dimensions are
in feet; all exported distances are in meters.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Tuple


FEET_TO_METERS = 0.3048
SAMPLE_HZ = 10
WINDOW_STATES = 91
WINDOW_SECONDS = (WINDOW_STATES - 1) / SAMPLE_HZ
MAX_SPICED_OBJECTS = 32
HEADING_WINDOW_HALF_SECONDS = 0.5
HEADING_MIN_RELIABLE_SPEED_MPS = 0.5
MAJOR_QUALITY_FLAGS = {
    "speed_above_60_mps",
    "more_than_200m_outside_calibrated_x_range",
    "outside_calibrated_y_range",
}


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
        # The lower-y carriageway is eastbound (+x); westbound geometry is
        # reversed so its lane centerlines follow traffic direction.
        direction = 1 if carriageway_index == 0 else -1
        start_x, end_x = (x_min_ft, x_max_ft) if direction == 1 else (x_max_ft, x_min_ft)

        for lower, upper in zip(boundaries_ft, boundaries_ft[1:]):
            center_y = 0.5 * (lower + upper)
            roads.append(
                {
                    "id": road_id,
                    "type": "lane",
                    "map_element_id": 1,  # Waymo freeway lane
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
                    "map_element_id": 6,  # broken single white
                    "geometry": [
                        _point(x_min_ft * FEET_TO_METERS, divider_y * FEET_TO_METERS),
                        _point(x_max_ft * FEET_TO_METERS, divider_y * FEET_TO_METERS),
                    ],
                }
            )
            road_id += 1

        for boundary_index in (0, len(boundaries_ft) - 1):
            boundary_y = boundaries_ft[boundary_index]
            is_median = (carriageway_index == 0 and boundary_index == len(boundaries_ft) - 1) or (
                carriageway_index == 1 and boundary_index == 0
            )
            roads.append(
                {
                    "id": road_id,
                    "type": "road_edge",
                    "map_element_id": 16 if is_median else 15,
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
        indices = range(begin, end)
        mean_index = sum(indices) / (end - begin)
        denominator = sum((index - mean_index) ** 2 for index in range(begin, end))
        if denominator == 0:
            candidates.append(None)
            continue
        vx = (
            sum(
                (index - mean_index) * positions[index][0]
                for index in range(begin, end)
            )
            / denominator
            * SAMPLE_HZ
        )
        vy = (
            sum(
                (index - mean_index) * positions[index][1]
                for index in range(begin, end)
            )
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

    # Backfill an initially stationary interval, then retain the last
    # confident orientation whenever displacement is below the noise floor.
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
    """Resample one native 30 Hz spline at 10 Hz without smoothing it."""
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
    if duration < WINDOW_SECONDS:
        quality_flags.append("shorter_than_9_seconds")
    if max(speeds, default=0.0) > 60.0:
        quality_flags.append("speed_above_60_mps")
    map_x_min, map_x_max = road_grid["x_range_m"]
    map_y_min, map_y_max = road_grid["boundary_y_m"][0], road_grid["boundary_y_m"][-1]
    # Vehicles legitimately enter and leave beyond the overlapping camera
    # calibration points. Only flag extreme extrapolation, not that normal
    # run-in/run-out region.
    margin = 200.0
    if any(x < map_x_min - margin or x > map_x_max + margin for x, _ in positions):
        quality_flags.append("more_than_200m_outside_calibrated_x_range")
    if any(y < map_y_min - margin or y > map_y_max + margin for _, y in positions):
        quality_flags.append("outside_calibrated_y_range")

    normalized = {
        "id": int(raw["id"]),
        "source_class": str(raw.get("class", "vehicle")),
        "type": "vehicle",
        "direction": int(raw.get("direction", 0)),
        "length_m": float(raw.get("l", 0.0)) * FEET_TO_METERS,
        "width_m": float(raw.get("w", 0.0)) * FEET_TO_METERS,
        "height_m": float(raw.get("h", 0.0)) * FEET_TO_METERS,
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
        "units": {"position": "meters", "velocity": "meters_per_second", "heading": "radians"},
        "source": str(splines_path),
        "object_count": len(objects),
        "heading_estimation": _heading_estimation_metadata(stabilize_headings),
        "map": road_grid,
        "objects": objects,
    }


def _object_end_tick(obj: dict[str, Any]) -> int:
    return int(obj["start_tick"]) + len(obj["position_m"]) - 1


def choose_sdc(scene_data: dict[str, Any], required_seconds: float) -> dict[str, Any]:
    required_ticks = math.ceil(required_seconds * SAMPLE_HZ)
    candidates = [
        obj
        for obj in scene_data["objects"]
        if len(obj["position_m"]) - 1 >= required_ticks
        and "speed_above_60_mps" not in obj.get("quality_flags", [])
        and "outside_calibrated_y_range" not in obj.get("quality_flags", [])
    ]
    if not candidates:
        raise ValueError(f"No trajectory spans the requested {required_seconds:.1f} seconds")
    # Prefer the longest clean trajectory, with source ID as deterministic tie-breaker.
    candidates.sort(key=lambda obj: (len(obj.get("quality_flags", [])), -len(obj["position_m"]), obj["id"]))
    return candidates[0]


def _state_at_tick(obj: dict[str, Any], tick: int) -> tuple[list[float], list[float], float, int]:
    index = tick - int(obj["start_tick"])
    valid = 0 <= index < len(obj["position_m"])
    clamped = max(0, min(index, len(obj["position_m"]) - 1))
    position = obj["position_m"][clamped]
    velocity = obj["velocity_mps"][clamped] if valid else [0.0, 0.0]
    heading = obj["heading_rad"][clamped]
    return position, velocity, heading, int(valid)


def _translated_roads(
    roads: list[dict[str, Any]], origin: list[float], x_bounds: Optional[Tuple[float, float]] = None
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
                "geometry": geometry,
            }
        )
    return translated


def build_spiced_scenario(
    scene_data: dict[str, Any],
    sdc_id: int,
    start_tick: int,
    window_index: int,
    max_objects: int,
    state_count: int = WINDOW_STATES,
    scenario_id: Optional[str] = None,
    full_range_context: bool = False,
) -> dict[str, Any]:
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
    for selected_index, obj in enumerate(selected):
        positions = []
        velocities = []
        headings = []
        valids = []
        for tick in range(start_tick, end_tick + 1):
            position, velocity, heading, valid = _state_at_tick(obj, tick)
            positions.append(_point(position[0] - origin[0], position[1] - origin[1]))
            velocities.append(_point(velocity[0], velocity[1]))
            headings.append(float(heading))
            valids.append(valid)
        final_valid_index = max(index for index, valid in enumerate(valids) if valid)
        scenario_objects.append(
            {
                "id": int(obj["id"]),
                "source_class": obj["source_class"],
                "type": "vehicle",
                "length": obj["length_m"],
                "width": obj["width_m"],
                "height": obj["height_m"],
                "position": positions,
                "velocity": velocities,
                "heading": headings,
                "valid": valids,
                "goalPosition": positions[final_valid_index],
                "mark_as_expert": 0 if selected_index == 0 else 1,
            }
        )

    if scenario_id is None:
        scenario_id = f"i24s{scene_data['scene']}v{sdc_id}w{window_index}"
    if len(scenario_id.encode("utf-8")) > 16:
        raise ValueError(f"Scenario ID is too long for spiced_self_play: {scenario_id}")
    all_x = [point["x"] for obj in scenario_objects for point in obj["position"] if math.isfinite(point["x"])]
    all_y = [point["y"] for obj in scenario_objects for point in obj["position"] if math.isfinite(point["y"])]
    # The calibration grid covers the cameras' overlap, while annotated
    # vehicles can legitimately enter/exit beyond it. I-24 is straight in
    # these scenes, so extend the recovered horizontal lines to keep the full
    # exported window on-road.
    roads = _translated_roads(
        scene_data["map"]["roads"], origin, (min(all_x) - 50.0, max(all_x) + 50.0)
    )
    return {
        "name": scenario_id,
        "scenario_id": scenario_id,
        "objects": scenario_objects,
        "roads": roads,
        "tl_states": [],
        "tracks_to_predict": [{"track_index": 0, "difficulty": 0, "object_type": 1}],
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0, "difficulty": 0, "object_type": 1}],
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
            "bounds": {"minX": min(all_x), "maxX": max(all_x), "minY": min(all_y), "maxY": max(all_y)},
        },
    }


def build_full_length_scenario(
    scene_data: dict[str, Any], sdc_id: int, max_objects: int
) -> dict[str, Any]:
    """Build one PufferDrive scenario spanning the SDC's complete 10 Hz track."""
    objects_by_id = {obj["id"]: obj for obj in scene_data["objects"]}
    if sdc_id not in objects_by_id:
        raise ValueError(f"No object {sdc_id} in scene {scene_data['scene']}")
    sdc = objects_by_id[sdc_id]
    scenario_id = f"i24s{scene_data['scene']}v{sdc_id}full"
    return build_spiced_scenario(
        scene_data,
        sdc_id=sdc_id,
        start_tick=int(sdc["start_tick"]),
        window_index=0,
        max_objects=max_objects,
        state_count=len(sdc["position_m"]),
        scenario_id=scenario_id,
        full_range_context=True,
    )


def validate_spiced_scenario(scenario: dict[str, Any], max_objects: int = MAX_SPICED_OBJECTS) -> None:
    """Validate the assumptions used by spiced_self_play's binary exporter."""
    required_top_level = {"name", "scenario_id", "objects", "roads", "tl_states", "metadata"}
    missing = required_top_level - scenario.keys()
    if missing:
        raise ValueError(f"Scenario is missing keys: {sorted(missing)}")
    if len(scenario["scenario_id"].encode("utf-8")) > 16:
        raise ValueError("scenario_id exceeds spiced_self_play's 16-byte binary field")
    if not 1 <= len(scenario["objects"]) <= max_objects:
        raise ValueError(f"Expected 1..{max_objects} objects")
    if scenario["metadata"].get("sdc_track_index") != 0:
        raise ValueError("Prototype scenarios must put the SDC at object index 0")
    expected_states = int(scenario["metadata"].get("frame_count", WINDOW_STATES))
    if expected_states < 2:
        raise ValueError("Scenario must contain at least two states")
    for object_index, obj in enumerate(scenario["objects"]):
        for field in ("position", "velocity", "heading", "valid"):
            if len(obj.get(field, [])) != expected_states:
                raise ValueError(
                    f"Object {object_index} field {field} does not have {expected_states} states"
                )
        if any(not math.isfinite(point[axis]) for point in obj["position"] for axis in ("x", "y", "z")):
            raise ValueError(f"Object {object_index} contains non-finite positions")
    if not all(scenario["objects"][0]["valid"]):
        raise ValueError("The SDC must be valid for the full 91-state window")
    if not scenario["roads"] or any(len(road.get("geometry", [])) < 2 for road in scenario["roads"]):
        raise ValueError("Each scenario needs non-empty road geometry")


def build_preview_html(scenario: dict[str, Any]) -> str:
    """Build a self-contained animated, top-down preview of one scenario."""
    preview_data = {
        "name": scenario["name"],
        "sample_hz": SAMPLE_HZ,
        "objects": [
            {
                "id": obj["id"],
                "length": round(float(obj["length"]), 3),
                "width": round(float(obj["width"]), 3),
                "is_sdc": index == scenario["metadata"]["sdc_track_index"],
                "x": [round(float(point["x"]), 3) for point in obj["position"]],
                "y": [round(float(point["y"]), 3) for point in obj["position"]],
                "heading": [round(float(value), 5) for value in obj["heading"]],
                "valid": [int(value) for value in obj["valid"]],
                "speed": [
                    round(math.hypot(float(value["x"]), float(value["y"])), 2)
                    for value in obj["velocity"]
                ],
            }
            for index, obj in enumerate(scenario["objects"])
        ],
        "roads": [
            {
                "type": road["type"],
                "geometry": [
                    [round(float(point["x"]), 3), round(float(point["y"]), 3)]
                    for point in road["geometry"]
                ],
            }
            for road in scenario["roads"]
        ],
    }
    data_json = json.dumps(preview_data, separators=(",", ":")).replace("</", "<\\/")
    template = """<div id="i24-scenario-viz" aria-label="Animated top-down I-24 trajectory preview">
  <style>
    #i24-scenario-viz {
      color-scheme: light dark;
      --i24-bg: light-dark(#f7f8fa, #101317);
      --i24-fg: light-dark(#17202a, #edf2f7);
      --i24-muted: light-dark(#66717d, #98a5b3);
      --i24-border: light-dark(#c9d1d9, #3a4652);
      --i24-lane: light-dark(#aab4bf, #566270);
      --i24-edge: light-dark(#596674, #aab6c4);
      --i24-sdc: light-dark(#d84a3a, #ff7666);
      --i24-traffic: light-dark(#2276b7, #5eb5ee);
      --i24-trail: light-dark(#9c352b, #c95f53);
      font: 14px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--i24-fg);
      width: 100%;
    }
    #i24-scenario-viz .i24-head,
    #i24-scenario-viz .i24-controls,
    #i24-scenario-viz .i24-legend {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    #i24-scenario-viz .i24-head { justify-content: space-between; margin-bottom: 8px; }
    #i24-scenario-viz h2 { font-size: 16px; line-height: 1.2; margin: 0; font-weight: 650; }
    #i24-scenario-viz .i24-readout { color: var(--i24-muted); font-variant-numeric: tabular-nums; }
    #i24-scenario-viz .i24-stage {
      height: clamp(190px, 34vw, 280px);
      border: 1px solid var(--i24-border);
      background: var(--i24-bg);
      overflow: hidden;
    }
    #i24-scenario-viz canvas { width: 100%; height: 100%; display: block; }
    #i24-scenario-viz .i24-controls { margin-top: 10px; }
    #i24-scenario-viz button,
    #i24-scenario-viz select {
      border: 1px solid var(--i24-border);
      background: var(--i24-bg);
      color: var(--i24-fg);
      border-radius: 5px;
      padding: 6px 10px;
      font: inherit;
    }
    #i24-scenario-viz button { min-width: 66px; cursor: pointer; }
    #i24-scenario-viz input[type="range"] { flex: 1 1 260px; min-width: 150px; accent-color: var(--i24-sdc); }
    #i24-scenario-viz .i24-legend { margin-top: 8px; color: var(--i24-muted); font-size: 12px; }
    #i24-scenario-viz .i24-key { display: inline-flex; align-items: center; gap: 5px; }
    #i24-scenario-viz .i24-swatch { width: 16px; height: 3px; background: var(--i24-lane); }
    #i24-scenario-viz .i24-swatch.sdc { height: 9px; background: var(--i24-sdc); }
    #i24-scenario-viz .i24-swatch.traffic { height: 9px; background: var(--i24-traffic); }
    @media (max-width: 440px) {
      #i24-scenario-viz .i24-stage { height: 190px; }
      #i24-scenario-viz .i24-readout { width: 100%; }
    }
  </style>
  <div class="i24-head">
    <h2></h2>
    <div class="i24-readout" aria-live="polite"></div>
  </div>
  <div class="i24-stage">
    <canvas role="img" aria-label="Top-down road animation with the focal vehicle and surrounding traffic"></canvas>
  </div>
  <div class="i24-controls">
    <button type="button" class="i24-play" aria-pressed="false">Play</button>
    <input class="i24-time" type="range" min="0" max="90" step="1" value="0" aria-label="Trajectory frame">
    <label>Speed
      <select class="i24-speed" aria-label="Playback speed">
        <option value="0.5">0.5×</option>
        <option value="1" selected>1×</option>
        <option value="2">2×</option>
      </select>
    </label>
  </div>
  <div class="i24-legend">
    <span class="i24-key"><span class="i24-swatch sdc"></span>Focal SDC</span>
    <span class="i24-key"><span class="i24-swatch traffic"></span>Surrounding traffic</span>
    <span class="i24-key"><span class="i24-swatch"></span>Lane geometry</span>
  </div>
  <script>
  (() => {
    const root = document.getElementById("i24-scenario-viz");
    const data = __I24_PREVIEW_DATA__;
    const canvas = root.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const title = root.querySelector("h2");
    const readout = root.querySelector(".i24-readout");
    const playButton = root.querySelector(".i24-play");
    const timeSlider = root.querySelector(".i24-time");
    const speedSelect = root.querySelector(".i24-speed");
    const sdc = data.objects.find(object => object.is_sdc);
    let frame = 0;
    let playing = false;
    let lastStep = 0;
    let css = {};

    title.textContent = `I-24 scene preview · ${data.name}`;
    timeSlider.max = String(sdc.x.length - 1);

    const validPoints = [];
    data.objects.forEach(object => object.x.forEach((x, index) => {
      if (object.valid[index]) validPoints.push([x, object.y[index]]);
    }));
    const roadYs = data.roads.flatMap(road => road.geometry.map(point => point[1]));
    const xValues = validPoints.map(point => point[0]);
    const yValues = validPoints.map(point => point[1]).concat(roadYs);
    const bounds = {
      minX: Math.min(...xValues) - 25,
      maxX: Math.max(...xValues) + 25,
      minY: Math.min(...yValues) - 5,
      maxY: Math.max(...yValues) + 5
    };

    function refreshColors() {
      const probe = document.createElement("span");
      probe.style.cssText = "position:absolute;visibility:hidden;pointer-events:none";
      root.appendChild(probe);
      const resolve = name => {
        probe.style.color = `var(${name})`;
        return getComputedStyle(probe).color;
      };
      css = {
        bg: resolve("--i24-bg"),
        fg: resolve("--i24-fg"),
        muted: resolve("--i24-muted"),
        lane: resolve("--i24-lane"),
        edge: resolve("--i24-edge"),
        sdc: resolve("--i24-sdc"),
        traffic: resolve("--i24-traffic"),
        trail: resolve("--i24-trail")
      };
      probe.remove();
    }

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function projection() {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const scale = Math.min(
        (width - 28) / (bounds.maxX - bounds.minX),
        (height - 28) / (bounds.maxY - bounds.minY)
      );
      const usedWidth = (bounds.maxX - bounds.minX) * scale;
      const usedHeight = (bounds.maxY - bounds.minY) * scale;
      const offsetX = (width - usedWidth) / 2;
      const offsetY = (height - usedHeight) / 2;
      return (x, y) => [
        offsetX + (x - bounds.minX) * scale,
        height - offsetY - (y - bounds.minY) * scale,
        scale
      ];
    }

    function drawRoads(project) {
      data.roads.forEach(road => {
        ctx.beginPath();
        road.geometry.forEach((point, index) => {
          const [x, y] = project(point[0], point[1]);
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        if (road.type === "road_edge") {
          ctx.strokeStyle = css.edge;
          ctx.lineWidth = 2;
          ctx.setLineDash([]);
        } else if (road.type === "road_line") {
          ctx.strokeStyle = css.lane;
          ctx.lineWidth = 1;
          ctx.setLineDash([8, 7]);
        } else {
          ctx.strokeStyle = css.lane;
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 7]);
        }
        ctx.stroke();
      });
      ctx.setLineDash([]);
    }

    function drawTrail(object, project) {
      ctx.beginPath();
      let started = false;
      for (let index = 0; index <= frame; index += 1) {
        if (!object.valid[index]) continue;
        const [x, y] = project(object.x[index], object.y[index]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = object.is_sdc ? css.trail : css.traffic;
      ctx.globalAlpha = object.is_sdc ? 0.65 : 0.16;
      ctx.lineWidth = object.is_sdc ? 2 : 1;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    function drawVehicle(object, project) {
      if (!object.valid[frame]) return;
      const [x, y, scale] = project(object.x[frame], object.y[frame]);
      const length = Math.max(7, object.length * scale);
      const width = Math.max(4, object.width * scale);
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(-object.heading[frame]);
      ctx.fillStyle = object.is_sdc ? css.sdc : css.traffic;
      ctx.globalAlpha = object.is_sdc ? 1 : 0.84;
      ctx.fillRect(-length / 2, -width / 2, length, width);
      ctx.fillStyle = css.bg;
      ctx.globalAlpha = 0.75;
      ctx.fillRect(length * 0.18, -width * 0.32, Math.max(2, length * 0.15), width * 0.64);
      ctx.restore();
      ctx.globalAlpha = 1;
      if (object.is_sdc) {
        ctx.fillStyle = css.fg;
        ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(`SDC ${object.id}`, x + 8, y - 8);
      }
    }

    function draw() {
      if (!canvas.clientWidth || !canvas.clientHeight) return;
      const project = projection();
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      ctx.fillStyle = css.bg;
      ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawRoads(project);
      data.objects.forEach(object => drawTrail(object, project));
      data.objects.filter(object => !object.is_sdc).forEach(object => drawVehicle(object, project));
      drawVehicle(sdc, project);
      const active = data.objects.filter(object => object.valid[frame]).length;
      readout.textContent = `${(frame / data.sample_hz).toFixed(1)} s · ${sdc.speed[frame].toFixed(1)} m/s · ${active} vehicles`;
      timeSlider.value = String(frame);
    }

    function setPlaying(next) {
      playing = next;
      playButton.textContent = playing ? "Pause" : "Play";
      playButton.setAttribute("aria-pressed", String(playing));
      lastStep = performance.now();
      if (playing) requestAnimationFrame(animate);
    }

    function animate(now) {
      if (!playing) return;
      const playbackSpeed = Number(speedSelect.value);
      if (now - lastStep >= 1000 / (data.sample_hz * playbackSpeed)) {
        frame = frame >= Number(timeSlider.max) ? 0 : frame + 1;
        lastStep = now;
        draw();
      }
      requestAnimationFrame(animate);
    }

    playButton.addEventListener("click", () => setPlaying(!playing));
    timeSlider.addEventListener("input", event => {
      frame = Number(event.target.value);
      draw();
    });
    const observer = new ResizeObserver(resize);
    observer.observe(canvas.parentElement);
    refreshColors();
    resize();
  })();
  </script>
</div>
"""
    return template.replace("__I24_PREVIEW_DATA__", data_json)


def write_json(path: Path, value: Any, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        if compact:
            json.dump(value, file, separators=(",", ":"))
        else:
            json.dump(value, file, indent=2)
            file.write("\n")


def window_start_ticks(
    obj: dict[str, Any], sdc_offset: float, window_stride: float, num_windows: int
) -> list[int]:
    """Return every requested complete 91-state window start for one SDC."""
    first_start_tick = obj["start_tick"] + round(sdc_offset * SAMPLE_HZ)
    stride_ticks = round(window_stride * SAMPLE_HZ)
    available_ticks = _object_end_tick(obj) - first_start_tick
    if available_ticks < WINDOW_STATES - 1:
        return []
    possible_windows = 1 + (available_ticks - (WINDOW_STATES - 1)) // stride_ticks
    window_count = possible_windows if num_windows == 0 else min(num_windows, possible_windows)
    return [first_start_tick + index * stride_ticks for index in range(window_count)]


def _selected_sdcs(scene_data: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_sdcs:
        return sorted(scene_data["objects"], key=lambda obj: obj["id"])
    if args.sdc_id is not None:
        matching = [obj for obj in scene_data["objects"] if obj["id"] == args.sdc_id]
        if not matching:
            raise ValueError(f"No object {args.sdc_id} in scene {scene_data['scene']}")
        return matching
    required = args.sdc_offset + WINDOW_SECONDS
    if args.num_windows > 0:
        required += (args.num_windows - 1) * args.window_stride
    return [choose_sdc(scene_data, required)]


def convert(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = (1, 2, 3) if args.all_scenes else (args.scene,)
    scenario_paths: list[str] = []
    scenario_files: list[str] = []
    preview_paths: list[str] = []
    preview_files: list[str] = []
    full_paths: list[str] = []
    full_files: list[str] = []
    object_counts: list[int] = []
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

        full_path = args.output_dir / f"scene{scene}_full_10hz.json"
        if not args.skip_full:
            write_json(full_path, scene_data, compact=True)
            full_paths.append(str(full_path))
            full_files.append(str(full_path.relative_to(args.output_dir)))

        scene_eligible = 0
        scene_exported = 0
        scene_excluded = 0
        scene_scenarios_before = len(scenario_paths)
        for sdc in _selected_sdcs(scene_data, args):
            starts = (
                [int(sdc["start_tick"])]
                if args.full_length
                else window_start_ticks(sdc, args.sdc_offset, args.window_stride, args.num_windows)
            )
            if not starts:
                if args.all_sdcs:
                    ineligible_sdcs.append(
                        {"scene": scene, "id": sdc["id"], "reason": "no_complete_91_state_window"}
                    )
                    continue
                raise ValueError("The requested SDC offset leaves no complete 91-state window")
            scene_eligible += 1

            major_flags = sorted(MAJOR_QUALITY_FLAGS & set(sdc.get("quality_flags", [])))
            if args.exclude_major_flags and major_flags:
                excluded_sdcs.append({"scene": scene, "id": sdc["id"], "flags": major_flags})
                scene_excluded += 1
                continue

            selected_sdcs.append({"scene": scene, "id": sdc["id"]})
            scene_exported += 1
            for window_index, start_tick in enumerate(starts):
                scenario = (
                    build_full_length_scenario(scene_data, sdc["id"], args.max_objects)
                    if args.full_length
                    else build_spiced_scenario(
                        scene_data, sdc["id"], start_tick, window_index, args.max_objects
                    )
                )
                validate_spiced_scenario(scenario, args.max_objects)
                scenario_path = args.output_dir / "scenarios" / f"{scenario['scenario_id']}.json"
                write_json(scenario_path, scenario)
                scenario_paths.append(str(scenario_path))
                scenario_files.append(str(scenario_path.relative_to(args.output_dir)))
                if not args.skip_preview:
                    preview_path = args.output_dir / "previews" / f"{scenario['scenario_id']}.html"
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    preview_path.write_text(build_preview_html(scenario))
                    preview_paths.append(str(preview_path))
                    preview_files.append(str(preview_path.relative_to(args.output_dir)))
                object_counts.append(len(scenario["objects"]))

        scene_summaries.append(
            {
                "scene": scene,
                "source_object_count": scene_data["object_count"],
                "eligible_sdc_count": scene_eligible,
                "exported_sdc_count": scene_exported,
                "ineligible_sdc_count": sum(1 for item in ineligible_sdcs if item["scene"] == scene),
                "excluded_major_quality_sdc_count": scene_excluded,
                "scenario_count": len(scenario_paths) - scene_scenarios_before,
                "quality_flag_counts": dict(sorted(scene_flag_counts.items())),
            }
        )

    is_batch = args.all_sdcs or args.all_scenes
    manifest = {
        "schema": "i24_3d_conversion_manifest_v2",
        "mode": "batch" if is_batch else "single_sdc",
        "scenes": list(scenes),
        "source_object_count": source_object_count,
        "sample_hz": SAMPLE_HZ,
        "heading_estimation": _heading_estimation_metadata(args.stabilize_headings),
        "full_trajectory_outputs": full_paths,
        "full_trajectory_files": full_files,
        "eligible_sdc_count": len(selected_sdcs) + len(excluded_sdcs),
        "exported_sdc_count": len(selected_sdcs),
        "ineligible_sdc_count": len(ineligible_sdcs),
        "excluded_major_quality_sdc_count": len(excluded_sdcs),
        "scenario_count": len(scenario_paths),
        "selected_sdcs": selected_sdcs,
        "ineligible_sdcs": ineligible_sdcs,
        "excluded_sdcs": excluded_sdcs,
        "scenario_outputs": scenario_paths,
        "scenario_files": scenario_files,
        "preview_outputs": preview_paths,
        "preview_files": preview_files,
        "scenario_object_counts": object_counts,
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "scene_summaries": scene_summaries,
        "options": {
            "all_scenes": args.all_scenes,
            "all_sdcs": args.all_sdcs,
            "full_length": args.full_length,
            "num_windows": args.num_windows,
            "window_stride_seconds": args.window_stride,
            "sdc_offset_seconds": args.sdc_offset,
            "max_objects": args.max_objects,
            "exclude_major_flags": args.exclude_major_flags,
            "stabilize_headings": args.stabilize_headings,
        },
        "notes": [
            "Quality flags are warnings; the prototype retains flagged source trajectories.",
            "Source trajectories shorter than 9.0 seconds remain in the full output but cannot be a complete 91-state SDC window.",
            "The road model comes from the scene's 12-foot calibration grid and is extended longitudinally; it is not a surveyed lane map.",
            "Each 91-state file spans 9.0 seconds because both endpoints are included at 10 Hz.",
            "Full-length scenarios are for PufferDrive replay; the current spiced self-play training pipeline still expects 91 states.",
            "Heading stabilization is opt-in and never modifies exported positions or velocities.",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_i24_root = repo_root.parent / "i24_3d_dataset" / "I24-3D"
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--i24-root", type=Path, default=default_i24_root)
    argument_parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "i24_3d_prototype")
    argument_parser.add_argument("--scene", type=int, choices=(1, 2, 3), default=2)
    argument_parser.add_argument("--all-scenes", action="store_true", help="Process scenes 1, 2, and 3")
    argument_parser.add_argument("--sdc-id", type=int, default=None, help="Default: automatically choose a long, clean trajectory")
    argument_parser.add_argument("--all-sdcs", action="store_true", help="Export every SDC with a complete 91-state window")
    argument_parser.add_argument(
        "--exclude-major-flags",
        action="store_true",
        help="Skip SDCs with extreme speed or calibrated-range quality flags",
    )
    argument_parser.add_argument("--sdc-offset", type=float, default=0.0, help="Seconds after the SDC trajectory begins")
    argument_parser.add_argument("--num-windows", type=int, default=1, help="Use 0 to export all complete windows")
    argument_parser.add_argument("--window-stride", type=float, default=9.0, help="Seconds between window starts")
    argument_parser.add_argument(
        "--full-length",
        action="store_true",
        help="Export each selected SDC's complete track as one PufferDrive replay scenario",
    )
    argument_parser.add_argument(
        "--stabilize-headings",
        action="store_true",
        help=(
            "Estimate headings over a 1.0 s position window and hold the last reliable "
            "orientation below 0.5 m/s; positions and velocities are unchanged"
        ),
    )
    argument_parser.add_argument("--max-objects", type=int, default=MAX_SPICED_OBJECTS)
    argument_parser.add_argument("--skip-full", action="store_true", help="Do not write the full-length 10 Hz intermediate")
    argument_parser.add_argument("--skip-preview", action="store_true", help="Do not write animated HTML previews")
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    if args.all_scenes and not args.all_sdcs:
        raise SystemExit("--all-scenes currently requires --all-sdcs")
    if args.all_sdcs and args.sdc_id is not None:
        raise SystemExit("--all-sdcs and --sdc-id are mutually exclusive")
    if args.full_length and args.sdc_offset != 0:
        raise SystemExit("--sdc-offset cannot be combined with --full-length")
    if args.full_length and args.num_windows != 1:
        raise SystemExit("--num-windows cannot be combined with --full-length")
    if args.num_windows < 0:
        raise SystemExit("--num-windows must be nonnegative")
    if args.sdc_offset < 0:
        raise SystemExit("--sdc-offset must be nonnegative")
    if args.window_stride <= 0:
        raise SystemExit("--window-stride must be positive")
    if not 1 <= args.max_objects <= MAX_SPICED_OBJECTS:
        raise SystemExit(f"--max-objects must be between 1 and {MAX_SPICED_OBJECTS}")
    manifest = convert(args)
    summary_keys = (
        "schema",
        "mode",
        "scenes",
        "source_object_count",
        "eligible_sdc_count",
        "exported_sdc_count",
        "ineligible_sdc_count",
        "excluded_major_quality_sdc_count",
        "scenario_count",
        "scene_summaries",
    )
    console_summary = {key: manifest[key] for key in summary_keys}
    console_summary["manifest"] = str(args.output_dir / "manifest.json")
    print(json.dumps(console_summary, indent=2))


if __name__ == "__main__":
    main()
