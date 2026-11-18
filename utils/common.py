"""PufferDrive data. Every converter writes this layout.

  data/<name>/json/<id>.json
  data/<name>/binaries/map_%03d.bin
  data/<name>/binaries/sdc.bin

Sorted json order is the map index. config/drive.ini [env] data_dir
points at data/<name>. Two binary records share one road encoding:

  Road map. int road count, then road records. No vehicles.
  load_map_binary treats a file as a road map when the first four
  ints are not a log header.

  Log. int object count, int road count, then each object, then the
  roads. The first object's type and trajectory length sit where a
  road map would have its first road header. The ints after each
  object are the next object's type and length, and after the last
  object they are the first road header. sdc.bin is int count, then
  one int32 self-driving-car index per map.

A log's trajectory length is that scenario's own state count. One
step is 0.1 s. A log is 2..DRIVE_LOG_MAX states. Object and road
class ids match ocean/drive/constants.h. Source roadgraph ids below
are stored as roads[].map_element_id, then collapsed to those classes.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

# One step is 0.1 s. DRIVE_LOG_MAX is the longest log load_map_binary
# accepts, matching constants.h. A Waymo Open Motion segment is 91 states.
SAMPLE_HZ = 10
WAYMO_TRAJECTORY_STATES = 91
MIN_TRAJECTORY_STATES = 2
DRIVE_LOG_MAX = 90 * SAMPLE_HZ + 1
# Export cap. DRIVE_AGENT_MAX is 256. load_map_binary accepts 512.
MAX_OBJECTS = 256

# ocean/drive/constants.h entity types. Partners are 1..4. Roads follow,
# so a type below ROAD_LANE is a body and a truck is not a lane.
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 3
TRUCK = 4
# Exported length. A vehicle-like body at least this long is a truck,
# including a tractor-trailer. Pedestrians and cyclists stay themselves.
TRUCK_MIN_LENGTH_M = 7.5
ROAD_LANE = 5
ROAD_LINE = 6
ROAD_EDGE = 7
STOP_SIGN_CLASS = 8
CROSSWALK_CLASS = 9
SPEED_BUMP_CLASS = 10
DRIVEWAY_CLASS = 11

# Source roadgraph ids, stored as roads[].map_element_id before
# drive_road_type. The numbers follow the Waymo Open Motion catalog:
# https://waymo.com/open/data/motion/tfexample/
LANE_UNKNOWN = 0
LANE_FREEWAY = 1
LANE_SURFACE_STREET = 2
LANE_BIKE = 3
ROAD_LINE_UNKNOWN = 5
ROAD_LINE_BROKEN_SINGLE_WHITE = 6
ROAD_LINE_SOLID_SINGLE_WHITE = 7
ROAD_LINE_SOLID_DOUBLE_WHITE = 8
ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
ROAD_LINE_SOLID_SINGLE_YELLOW = 11
ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
ROAD_EDGE_UNKNOWN = 14
ROAD_EDGE_BOUNDARY = 15
ROAD_EDGE_MEDIAN = 16
STOP_SIGN = 17
CROSSWALK = 18
SPEED_BUMP = 19
DRIVEWAY = 20

OBJECT_TYPES = {
    VEHICLE: "vehicle",
    PEDESTRIAN: "pedestrian",
    CYCLIST: "cyclist",
    TRUCK: "truck",
}
# Source labels that become vehicle or truck, chosen by length.
VEHICLE_LABELS = {
    "",
    "vehicle",
    "sedan",
    "midsize",
    "pickup",
    "van",
    "cs",
    "truck",
    "semi",
    "trailer",
}
# roads[].type strings in the scenario JSON.
ROAD_TYPES = (
    "lane",
    "road_line",
    "road_edge",
    "stop_sign",
    "crosswalk",
    "speed_bump",
    "driveway",
)


def drive_road_type(map_element_id: int) -> int:
    """Collapse a source roadgraph id to the PufferDrive road class."""
    if LANE_UNKNOWN <= map_element_id <= LANE_BIKE:
        return ROAD_LANE
    if ROAD_LINE_UNKNOWN <= map_element_id <= ROAD_LINE_PASSING_DOUBLE_YELLOW:
        return ROAD_LINE
    if ROAD_EDGE_UNKNOWN <= map_element_id <= ROAD_EDGE_MEDIAN:
        return ROAD_EDGE
    if map_element_id == STOP_SIGN:
        return STOP_SIGN_CLASS
    if map_element_id == CROSSWALK:
        return CROSSWALK_CLASS
    if map_element_id == SPEED_BUMP:
        return SPEED_BUMP_CLASS
    if map_element_id == DRIVEWAY:
        return DRIVEWAY_CLASS
    return map_element_id


def classify_partner(source_type: str, length_m: float) -> str:
    """Partner class shared by Waymo and I24.

    Pedestrians and cyclists stay those classes. Any other vehicle-like
    body of TRUCK_MIN_LENGTH_M or longer is a truck. A shorter one is a
    vehicle. Object 0 may be either a vehicle or a truck.
    """
    name = str(source_type or "").strip().lower()
    if name in (OBJECT_TYPES[PEDESTRIAN], OBJECT_TYPES[CYCLIST]):
        return name
    if name not in VEHICLE_LABELS:
        raise ValueError(f"Unknown partner class {source_type!r}")
    if float(length_m) >= TRUCK_MIN_LENGTH_M:
        return OBJECT_TYPES[TRUCK]
    return OBJECT_TYPES[VEHICLE]


def object_type_code(type_name: str) -> int:
    """Object name to the integer stored in a PufferDrive log."""
    for code, name in OBJECT_TYPES.items():
        if name == type_name:
            return code
    raise ValueError(f"Unknown PufferDrive object type {type_name!r}")


def validate_scenario(scenario: dict[str, Any], max_objects: int = MAX_OBJECTS) -> None:
    """Check the fields offline.h and load_map_binary require of a log."""
    required = {"name", "scenario_id", "objects", "roads", "metadata"}
    missing = required - scenario.keys()
    if missing:
        raise ValueError(f"Scenario is missing keys: {sorted(missing)}")
    if not 1 <= len(scenario["objects"]) <= max_objects:
        raise ValueError(f"Expected 1..{max_objects} objects")
    if scenario["metadata"].get("sdc_track_index") != 0:
        raise ValueError("The self-driving car must be object 0")
    expected_states = int(scenario["metadata"]["frame_count"])
    if expected_states < MIN_TRAJECTORY_STATES or expected_states > DRIVE_LOG_MAX:
        raise ValueError(
            f"Scenario length must be {MIN_TRAJECTORY_STATES}..{DRIVE_LOG_MAX} states"
        )
    for object_index, obj in enumerate(scenario["objects"]):
        if obj.get("type") not in OBJECT_TYPES.values():
            raise ValueError(f"Object {object_index} is not a PufferDrive object type")
        if object_index == 0 and obj.get("type") not in (
            OBJECT_TYPES[VEHICLE],
            OBJECT_TYPES[TRUCK],
        ):
            raise ValueError("The self-driving car must be a vehicle or a truck")
        for field in ("position", "velocity", "heading", "valid"):
            if len(obj.get(field, [])) != expected_states:
                raise ValueError(
                    f"Object {object_index} field {field} does not have {expected_states} states"
                )
        for key in ("width", "length", "goalPosition"):
            if key not in obj:
                raise ValueError(f"Object {object_index} is missing {key}")
        if any(
            not math.isfinite(point[axis])
            for point in obj["position"]
            for axis in ("x", "y", "z")
        ):
            raise ValueError(f"Object {object_index} contains non-finite positions")
    if not all(scenario["objects"][0]["valid"]):
        raise ValueError("The self-driving car must be valid on every state")
    if not scenario["roads"]:
        raise ValueError("Each scenario needs non-empty road geometry")
    for road in scenario["roads"]:
        if road.get("type") not in ROAD_TYPES:
            raise ValueError(f"Road type {road.get('type')!r} is not a PufferDrive road category")
        if len(road.get("geometry", [])) < 2:
            raise ValueError("Each scenario needs non-empty road geometry")
        collapsed = drive_road_type(int(road.get("map_element_id", -1)))
        if collapsed < ROAD_LANE or collapsed > DRIVEWAY_CLASS:
            raise ValueError(
                f"Road map_element_id {road.get('map_element_id')} is not a PufferDrive road category"
            )


def write_road(file, road: dict[str, Any]) -> None:
    """One road record. map_element_id is collapsed to the drive class."""
    geometry = road.get("geometry") or []
    road_type = drive_road_type(int(road.get("map_element_id", 0)))
    file.write(struct.pack("ii", road_type, len(geometry)))
    for coord in ("x", "y", "z"):
        file.write(struct.pack(
            f"{len(geometry)}f",
            *[float(point.get(coord, 0.0)) for point in geometry],
        ))
    file.write(struct.pack(
        "fff",
        float(road.get("width", 0.0)),
        float(road.get("length", 0.0)),
        float(road.get("height", 0.0)),
    ))
    goal = road.get("goalPosition") or {"x": 0.0, "y": 0.0, "z": 0.0}
    file.write(struct.pack(
        "fff",
        float(goal.get("x", 0.0)),
        float(goal.get("y", 0.0)),
        float(goal.get("z", 0.0)),
    ))
    file.write(struct.pack("i", int(road.get("mark_as_expert", 0) or 0)))


def write_road_map(roads: list[dict[str, Any]], path: Path | str) -> None:
    """Road-only binary. The file has no vehicle records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(struct.pack("i", len(roads)))
        for road in roads:
            write_road(file, road)


def write_log_binary(scenario: dict[str, Any], path: Path | str) -> None:
    """Log record load_map_binary detects. Every object shares one length."""
    objects = scenario["objects"]
    roads = scenario["roads"]
    if not objects or not roads:
        raise ValueError("A log binary needs objects and roads")
    state_count = len(objects[0]["position"])
    if state_count < MIN_TRAJECTORY_STATES or state_count > DRIVE_LOG_MAX:
        raise ValueError(
            f"Log binaries are {MIN_TRAJECTORY_STATES}..{DRIVE_LOG_MAX} states, got {state_count}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(struct.pack("ii", len(objects), len(roads)))
        for obj in objects:
            if len(obj["position"]) != state_count:
                raise ValueError("Every object in one binary must share the trajectory length")
            code = object_type_code(obj["type"])
            file.write(struct.pack("ii", code, state_count))
            for coord in ("x", "y", "z"):
                file.write(struct.pack(
                    f"{state_count}f",
                    *[float(point.get(coord, 0.0)) for point in obj["position"]],
                ))
            for coord in ("x", "y", "z"):
                file.write(struct.pack(
                    f"{state_count}f",
                    *[float(sample.get(coord, 0.0)) for sample in obj["velocity"]],
                ))
            file.write(struct.pack(
                f"{state_count}f",
                *[float(value) for value in obj["heading"]],
            ))
            file.write(struct.pack(
                f"{state_count}i",
                *[1 if flag else 0 for flag in obj["valid"]],
            ))
            goal = obj["goalPosition"]
            file.write(struct.pack(
                "fff",
                float(obj["width"]),
                float(obj["length"]),
                float(obj.get("height", 1.5)),
            ))
            file.write(struct.pack(
                "fff",
                float(goal["x"]),
                float(goal["y"]),
                float(goal.get("z", 0.0)),
            ))
            file.write(struct.pack("i", 1 if obj.get("mark_as_expert") else 0))
        for road in roads:
            write_road(file, road)


def write_sdc_index(path: Path | str, indices: list[int]) -> None:
    """int count, then one int32 object index per map. Matches read_map_sdc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(struct.pack("i", len(indices)))
        file.write(struct.pack(f"{len(indices)}i", *indices))
