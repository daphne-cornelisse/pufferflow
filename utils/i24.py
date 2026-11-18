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
# Annotated boxes are treated as slightly larger than the body that
# drove the lane. Exported length, width, and height use this factor.
BOX_SCALE = 0.90
# The calibration edge is the outer lane line. No I24 vehicle is
# treated as truly overlapping that line, so each edge is moved 1.5 ft
# away from the lanes.
ROAD_EDGE_OUTWARD_M = 1.5 * FEET_TO_METERS
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
# A track that drives this far against its carriageway is removed.
# The carriageway direction is the majority of tracks there, and the
# lower carriageway is eastbound (+x) when that vote is tied.
WRONG_WAY_MIN_METERS = 10.0
# Intersection area above half the smaller box, on more than this
# many timesteps, drops the smaller track. Pairs are collected first.
DEEP_OVERLAP_FRACTION = 0.5
DEEP_OVERLAP_MIN_STEPS = 5
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


def _edge_y_m(boundary_y_ft: float, lane_mid_ft: float) -> float:
    """Lane-line y, shifted 1.5 ft away from the carriageway center."""
    outward = -1.0 if boundary_y_ft < lane_mid_ft else 1.0
    return boundary_y_ft * FEET_TO_METERS + outward * ROAD_EDGE_OUTWARD_M


def _carriageway_split_y(boundary_y_m: list[float]) -> float:
    """Y of the median, the largest gap between calibration lines."""
    gaps = [right - left for left, right in zip(boundary_y_m, boundary_y_m[1:])]
    if not gaps:
        raise ValueError("The road grid has no carriageway split")
    median_index = max(range(len(gaps)), key=gaps.__getitem__)
    return 0.5 * (boundary_y_m[median_index] + boundary_y_m[median_index + 1])


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

        lane_mid_ft = 0.5 * (boundaries_ft[0] + boundaries_ft[-1])
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
                        _point(x_min_ft * FEET_TO_METERS, _edge_y_m(boundary_y, lane_mid_ft)),
                        _point(x_max_ft * FEET_TO_METERS, _edge_y_m(boundary_y, lane_mid_ft)),
                    ],
                }
            )
            road_id += 1

    boundary_y_m = [value * FEET_TO_METERS for value in y_values]
    return {
        "source": homography_path.name,
        "boundary_y_m": boundary_y_m,
        "carriageway_split_y_m": _carriageway_split_y(boundary_y_m),
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
    raw: dict[str, Any], road_grid: dict[str, Any], stabilize_headings: bool = True
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
    length_m *= BOX_SCALE
    width_m *= BOX_SCALE
    height_m *= BOX_SCALE

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
    i24_root: Path, scene: int, stabilize_headings: bool = True
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


def _box_corners(
    x: float, y: float, heading: float, length: float, width: float
) -> list[tuple[float, float]]:
    """Clockwise corners. Length is along heading."""
    half_length = length * 0.5
    half_width = width * 0.5
    cosine = math.cos(heading)
    sine = math.sin(heading)
    corners = []
    for along, across in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        corners.append((
            x + along * cosine - across * sine,
            y + along * sine + across * cosine,
        ))
    return corners


def _clip_polygon(
    subject: list[tuple[float, float]],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> list[tuple[float, float]]:
    """Keep the part of subject on the inside of a clockwise edge."""
    if not subject:
        return []

    def cross(point: tuple[float, float]) -> float:
        return (
            (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
        )

    def intersection(
        start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, float]:
        start_cross = cross(start)
        end_cross = cross(end)
        denom = start_cross - end_cross
        if abs(denom) < 1e-12:
            return end
        weight = start_cross / denom
        return (
            start[0] + weight * (end[0] - start[0]),
            start[1] + weight * (end[1] - start[1]),
        )

    output: list[tuple[float, float]] = []
    previous = subject[-1]
    previous_cross = cross(previous)
    for current in subject:
        current_cross = cross(current)
        if current_cross <= 1e-9:
            if previous_cross > 1e-9:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_cross <= 1e-9:
            output.append(intersection(previous, current))
        previous = current
        previous_cross = current_cross
    return output


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(polygon):
        nxt = polygon[(index + 1) % len(polygon)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) * 0.5


def _intersection_area(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> float:
    """Area shared by two convex boxes."""
    polygon = first
    for index, start in enumerate(second):
        polygon = _clip_polygon(polygon, start, second[(index + 1) % len(second)])
        if len(polygon) < 3:
            return 0.0
    return _polygon_area(polygon)


def _valid_xy(obj: dict[str, Any]) -> list[tuple[float, float]]:
    return [
        (float(point["x"]), float(point["y"]))
        for point, valid in zip(obj["position"], obj["valid"])
        if valid
    ]


def _opposing_meters(points: list[tuple[float, float]], legal_sign: int) -> float:
    """Path length traveled against legal_sign (+1 east, -1 west)."""
    opposing = 0.0
    previous_x = None
    for x_pos, _y_pos in points:
        if previous_x is not None:
            step = x_pos - previous_x
            if step * legal_sign < 0.0:
                opposing += abs(step)
        previous_x = x_pos
    return opposing


def _legal_directions(
    tracks: list[tuple[float, list[tuple[float, float]]]],
) -> dict[str, int]:
    """Majority of net travel on each side of the median.

    tracks are (mean_y, valid xy). A tie uses eastbound below the
    median and westbound above it. The caller passes mean_y already
    compared against the split by the sign of the stored side key.
    """
    votes = {"lower": [0, 0], "upper": [0, 0]}
    for side, points in tracks:
        if len(points) < 2:
            continue
        net = points[-1][0] - points[0][0]
        if abs(net) < WRONG_WAY_MIN_METERS:
            continue
        votes[side][1 if net > 0.0 else 0] += 1
    legal: dict[str, int] = {}
    for side, fallback in (("lower", 1), ("upper", -1)):
        west_votes, east_votes = votes[side]
        if east_votes == west_votes:
            legal[side] = fallback
        else:
            legal[side] = 1 if east_votes > west_votes else -1
    return legal


def _window_points(obj: dict[str, Any], start_tick: int, end_tick: int) -> list[tuple[float, float]]:
    points = []
    for tick in range(start_tick, end_tick + 1):
        position, _velocity, _heading, valid = _state_at_tick(obj, tick)
        if valid:
            points.append((float(position[0]), float(position[1])))
    return points


def _wrong_way_ids(
    objects: list[dict[str, Any]],
    start_tick: int,
    end_tick: int,
    split_y: float,
    sdc_id: int,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    """Tracks that drive WRONG_WAY_MIN_METERS against their carriageway.

    The focal car is reported but not listed for deletion.
    """
    samples = []
    for obj in objects:
        points = _window_points(obj, start_tick, end_tick)
        if len(points) < 2:
            continue
        mean_y = sum(point[1] for point in points) / len(points)
        side = "lower" if mean_y < split_y else "upper"
        samples.append((obj, points, side, mean_y))
    legal = _legal_directions([(side, points) for _obj, points, side, _mean in samples])
    removed = []
    focal_opposing = 0.0
    for obj, points, side, _mean_y in samples:
        opposing = _opposing_meters(points, legal[side])
        if int(obj["id"]) == int(sdc_id):
            focal_opposing = opposing
            continue
        if opposing > WRONG_WAY_MIN_METERS:
            removed.append(
                {
                    "id": int(obj["id"]),
                    "carriageway": "eastbound" if side == "lower" else "westbound",
                    "opposing_m": round(opposing, 2),
                }
            )
    removed.sort(key=lambda item: item["id"])
    return removed, legal, focal_opposing


def _deep_overlap_ids(
    objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Smaller car of each pair that overlaps on more than 5 timesteps.

    Overlap means the intersection is more than half the smaller box.
    Pairs are marked together, then the smaller ids are dropped once.
    """
    count = len(objects)
    if count < 2:
        return [], []
    state_count = len(objects[0]["valid"])
    areas = [float(obj["length"]) * float(obj["width"]) for obj in objects]
    deep_steps = [[0 for _ in range(count)] for _ in range(count)]
    for tick in range(state_count):
        alive = [index for index in range(count) if objects[index]["valid"][tick]]
        boxes = {}
        centers = {}
        reach = {}
        for index in alive:
            obj = objects[index]
            point = obj["position"][tick]
            centers[index] = (float(point["x"]), float(point["y"]))
            reach[index] = 0.5 * math.hypot(float(obj["length"]), float(obj["width"]))
            boxes[index] = _box_corners(
                float(point["x"]),
                float(point["y"]),
                float(obj["heading"][tick]),
                float(obj["length"]),
                float(obj["width"]),
            )
        for left_pos, left in enumerate(alive):
            for right in alive[left_pos + 1 :]:
                dx = centers[left][0] - centers[right][0]
                dy = centers[left][1] - centers[right][1]
                limit = reach[left] + reach[right]
                if dx * dx + dy * dy > limit * limit:
                    continue
                shared = _intersection_area(boxes[left], boxes[right])
                smaller = min(areas[left], areas[right])
                if smaller > 0.0 and shared > DEEP_OVERLAP_FRACTION * smaller:
                    deep_steps[left][right] += 1
    pairs = []
    focal_smaller = []
    for left in range(count):
        for right in range(left + 1, count):
            steps = deep_steps[left][right]
            if steps <= DEEP_OVERLAP_MIN_STEPS:
                continue
            if areas[left] < areas[right] or (
                areas[left] == areas[right] and int(objects[left]["id"]) > int(objects[right]["id"])
            ):
                small, large = left, right
            else:
                small, large = right, left
            record = {
                "id": int(objects[small]["id"]),
                "other_id": int(objects[large]["id"]),
                "timesteps": int(steps),
                "area_m2": round(areas[small], 3),
                "other_area_m2": round(areas[large], 3),
            }
            if small == 0:
                focal_smaller.append(record)
                continue
            pairs.append(record)
    pairs.sort(key=lambda item: (item["id"], item["other_id"]))
    return pairs, focal_smaller


def _cleanup_tracks(
    scenario_objects: list[dict[str, Any]],
    scene_objects: list[dict[str, Any]],
    start_tick: int,
    end_tick: int,
    split_y: float,
    sdc_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop wrong-way tracks, then the smaller car of each deep overlap."""
    wrong_way, legal, focal_opposing = _wrong_way_ids(
        scene_objects, start_tick, end_tick, split_y, sdc_id
    )
    wrong_ids = {item["id"] for item in wrong_way}
    kept = [obj for obj in scenario_objects if int(obj["id"]) not in wrong_ids]
    overlaps, focal_smaller = _deep_overlap_ids(kept)
    overlap_ids = {item["id"] for item in overlaps}
    cleaned = [obj for obj in kept if int(obj["id"]) not in overlap_ids]
    report = {
        "wrong_way_min_m": WRONG_WAY_MIN_METERS,
        "deep_overlap_fraction": DEEP_OVERLAP_FRACTION,
        "deep_overlap_min_steps": DEEP_OVERLAP_MIN_STEPS,
        "legal_direction": {
            "eastbound_lower": "east" if legal["lower"] > 0 else "west",
            "westbound_upper": "east" if legal["upper"] > 0 else "west",
        },
        "focal_opposing_m": round(focal_opposing, 2),
        "focal_kept_despite_wrong_way": focal_opposing > WRONG_WAY_MIN_METERS,
        "removed_wrong_way": wrong_way,
        "removed_deep_overlap": overlaps,
        "focal_smaller_overlaps_kept": focal_smaller,
    }
    return cleaned, report


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
    road_grid = scene_data["map"]
    if "carriageway_split_y_m" in road_grid:
        split_y = float(road_grid["carriageway_split_y_m"])
    else:
        split_y = _carriageway_split_y(road_grid["boundary_y_m"])
    wrong_way, _legal, _focal_opposing = _wrong_way_ids(
        scene_data["objects"], start_tick, end_tick, split_y, sdc_id
    )
    wrong_ids = {item["id"] for item in wrong_way}
    candidates = []
    for obj in scene_data["objects"]:
        if obj["id"] == sdc_id or obj["id"] in wrong_ids:
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
    # Roads keep the pre-cleanup extent. Wrong-way ids were already
    # left out of the nearest-object cut; this pass drops deep overlaps
    # and any wrong-way track that still got in.
    scenario_objects, cleanup = _cleanup_tracks(
        scenario_objects,
        scene_data["objects"],
        start_tick,
        end_tick,
        split_y,
        sdc_id,
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
            "track_cleanup": cleanup,
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
            "The road model is the scene's 12-foot calibration grid, extended along the roadway and sampled every 5 m. Road edges are 1.5 ft outward from the lane lines. Boxes are 0.90 of the annotated size.",
            "Action labels are the simulator inverse on these logs, as in arXiv 2606.19370 A.4.",
            "Heading stabilization never modifies exported positions or velocities.",
            "A track that travels more than 10 m against its carriageway is removed. The carriageway direction is the majority of tracks on that side of the median.",
            "When two boxes overlap by more than half the smaller box for more than 5 timesteps, the smaller track is removed. Pairs are marked before any of those tracks are deleted. The focal track is kept.",
        ],
    }
    manifest["track_cleanup"] = [
        {
            "scenario_id": scenario["scenario_id"],
            "source_scene": scenario["metadata"]["source_scene"],
            "source_sdc_id": scenario["metadata"]["source_sdc_id"],
            "object_count": len(scenario["objects"]),
            **scenario["metadata"].get("track_cleanup", {}),
        }
        for scenario in scenarios
    ]
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
        "--stabilize-headings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Estimate headings over a 1.0 s position window and hold the last "
            "reliable orientation below 0.5 m/s. This is the default. "
            "--no-stabilize-headings keeps the per-frame velocity heading."
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
