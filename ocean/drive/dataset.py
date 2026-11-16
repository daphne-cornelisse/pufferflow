"""Convert Waymo Open Motion Dataset (WOMD) JSON maps to binary format for PufferLib drive env.

Each dataset lives at data/<name>/ with json/ and binaries/.
config/drive.ini [env] data_dir points at the dataset the trainer loads.

Step 0: Download the preprocessed JSON scenarios from HuggingFace e.g.,
  https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_train_1000

  uv pip install huggingface_hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='daphne-cornelisse/pufferdrive_womd_train_1000', repo_type='dataset', local_dir='data/womd_train_1000')"

Step 1: Unzip the scenarios into json/
    mkdir -p data/womd_train_1000/json
    tar xzf data/womd_train_1000/pufferdrive_womd_train_1000.tar.gz --strip-components=1 -C data/womd_train_1000/json/

Step 2: Process to map binaries
  python ocean/drive/dataset.py --data_folder data/womd_train_1000/json --output_dir data/womd_train_1000/binaries
"""

import json
import struct
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

TRAJECTORY_LENGTH = 91

# Waymo roadgraph_samples/type, stored as roads[].map_element_id.
# https://waymo.com/open/data/motion/tfexample/
# The drive env collapses these to the classes in ocean/drive/drive.h:
#   lane centers 0-3 -> ROAD_LANE, road lines 5-13 -> ROAD_LINE,
#   road edges 14-16 -> ROAD_EDGE, then stop sign, crosswalk, speed bump, driveway.
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


def drive_road_type(map_element_id):
    """Collapse a Waymo roadgraph type to the drive env's road class."""
    if LANE_UNKNOWN <= map_element_id <= LANE_BIKE:
        return 4
    if ROAD_LINE_UNKNOWN <= map_element_id <= ROAD_LINE_PASSING_DOUBLE_YELLOW:
        return 5
    if ROAD_EDGE_UNKNOWN <= map_element_id <= ROAD_EDGE_MEDIAN:
        return 6
    if map_element_id == STOP_SIGN:
        return 7
    if map_element_id == CROSSWALK:
        return 8
    if map_element_id == SPEED_BUMP:
        return 9
    if map_element_id == DRIVEWAY:
        return 10
    return map_element_id


def calculate_area(p1, p2, p3):
    """Calculate the area of the triangle using the determinant method."""
    return 0.5 * abs(
        (p1["x"] - p3["x"]) * (p2["y"] - p1["y"])
        - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"])
    )


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)
            if (
                area < polyline_reduction_threshold
                and dist(point1, point3) <= max_segment_length
            ):
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def save_map_binary(map_data, output_file, unique_map_id):
    """Save map data in a binary format readable by C."""
    with open(output_file, "wb") as f:
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))

        # Write objects
        for obj in map_data.get("objects", []):
            trajectory_length = max(
                len(obj.get("position", [])),
                len(obj.get("velocity", [])),
                len(obj.get("heading", [])),
                len(obj.get("valid", [])),
            )
            if trajectory_length == 0:
                trajectory_length = TRAJECTORY_LENGTH
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))
            f.write(struct.pack("i", trajectory_length))

            positions = obj.get("position", [])
            for coord in ["x", "y", "z"]:
                for i in range(trajectory_length):
                    pos = (
                        positions[i]
                        if i < len(positions)
                        else {"x": 0.0, "y": 0.0, "z": 0.0}
                    )
                    f.write(struct.pack("f", float(pos.get(coord, 0.0))))

            velocities = obj.get("velocity", [])
            for coord in ["x", "y", "z"]:
                for i in range(trajectory_length):
                    vel = (
                        velocities[i]
                        if i < len(velocities)
                        else {"x": 0.0, "y": 0.0, "z": 0.0}
                    )
                    f.write(struct.pack("f", float(vel.get(coord, 0.0))))

            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}f",
                    *[
                        float(headings[i]) if i < len(headings) else 0.0
                        for i in range(trajectory_length)
                    ],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}i",
                    *[
                        int(valids[i]) if i < len(valids) else 0
                        for i in range(trajectory_length)
                    ],
                )
            )

            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

        # Write roads
        for road in map_data.get("roads", []):
            geometry = road.get("geometry", [])
            # Class id is the Waymo roadgraph type. Do not rewrite lanes to
            # surface-street or edges to boundary; freeway and median must
            # stay in the same drive class as the other lanes and edges.
            road_type = int(road.get("map_element_id", 0))

            if len(geometry) > 10 and road_type <= ROAD_EDGE_MEDIAN:
                geometry = simplify_polyline(geometry, 0.1, 250)
            size = len(geometry)
            road_type = drive_road_type(road_type)

            f.write(struct.pack("i", road_type))
            f.write(struct.pack("i", size))

            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))

            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))


def _process_single_map(args):
    """Worker function to process a single map file."""
    i, map_path, binary_path = args
    try:
        with open(map_path, "r") as f:
            map_data = json.load(f)
        save_map_binary(map_data, str(binary_path), i)
        return (i, map_path.name, True, None)
    except Exception as e:
        return (i, map_path.name, False, str(e))


def process_all_maps(
    data_folder,
    output_dir,
    max_maps=50_000,
    num_workers=None,
):
    """Process JSON map files into binary format using multiprocessing.

    Args:
        data_folder: Path to the folder containing JSON map files.
        output_dir: Path to the directory where binary files will be written.
        max_maps: Maximum number of maps to process.
        num_workers: Number of parallel workers (defaults to cpu_count()).
    """
    if num_workers is None:
        num_workers = cpu_count()

    data_dir = Path(data_folder)
    binary_dir = Path(output_dir)
    binary_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_path = binary_dir / f"map_{i:03d}.bin"
        tasks.append((i, map_path, binary_path))

    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_process_single_map, tasks),
                total=len(tasks),
                desc="Processing maps",
                unit="map",
            )
        )

    successful = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    print(f"\nProcessed {successful}/{len(results)} maps successfully.")
    if failed > 0:
        print(f"Failed {failed}/{len(results)} files:")
        for i, name, success, error in results:
            if not success:
                print(f"  {name}: {error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON map files to binary format.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to folder containing JSON map files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for binary files.")
    parser.add_argument("--max_maps", type=int, default=50_000, help="Maximum number of maps to process.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers.")
    args = parser.parse_args()

    process_all_maps(
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        max_maps=args.max_maps,
        num_workers=args.num_workers,
    )
