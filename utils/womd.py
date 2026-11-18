"""Download Waymo Open Motion scenes and write PufferDrive logs.

Each scenario JSON becomes binaries/map_%03d.bin in sorted json
order, plus binaries/sdc.bin. Roads are thinned the same way as the
road-only record. A JSON with no objects is written as a road map.
config/womd_short.ini data_dir points at the directory this writes.

After a run, check two agents with:
  python utils/sanity.py --data-dir data/womd_train_1000
The PNG and the GIF are written to data/womd_train_1000/assets.

Step 0: Download the preprocessed JSON scenarios from HuggingFace e.g.,
  https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_train_1000

  uv pip install huggingface_hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='daphne-cornelisse/pufferdrive_womd_train_1000', repo_type='dataset', local_dir='data/womd_train_1000')"

Step 1: Unzip the scenarios into json/
    mkdir -p data/womd_train_1000/json
    tar xzf data/womd_train_1000/pufferdrive_womd_train_1000.tar.gz --strip-components=1 -C data/womd_train_1000/json/

Step 2: Process to map binaries
  python utils/womd.py --data_folder data/womd_train_1000/json --output_dir data/womd_train_1000/binaries
"""

import json
import struct
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.common import (  # noqa: E402
    DRIVE_LOG_MAX,
    ROAD_EDGE_MEDIAN,
    CYCLIST,
    PEDESTRIAN,
    TRUCK,
    TRUCK_MIN_LENGTH_M,
    VEHICLE,
    classify_partner,
    write_log_binary,
    write_road_map,
    write_sdc_index,
)


def calculate_area(p1, p2, p3):
    """Area of the triangle, from the determinant method."""
    return 0.5 * abs(
        (p1["x"] - p3["x"]) * (p2["y"] - p1["y"])
        - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"])
    )


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
    """Drop points whose triangle is small, in the Visvalingam-Whyatt style."""
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


def prepare_road(road):
    """Copy one JSON road and thin a long polyline. The class id stays."""
    geometry = road.get("geometry", [])
    # Do not rewrite lanes to surface-street or edges to boundary.
    # Freeway and median must stay in the same drive class as the
    # other lanes and edges.
    element = int(road.get("map_element_id", 0))
    if len(geometry) > 10 and element <= ROAD_EDGE_MEDIAN:
        geometry = simplify_polyline(geometry, 0.1, 250)
    prepared = dict(road)
    prepared["geometry"] = geometry
    return prepared


def _process_single_map(args):
    """Worker. Returns the self-driving-car index for a log, else None."""
    i, map_path, binary_path = args
    try:
        with open(map_path, "r") as f:
            scenario = json.load(f)
        source_roads = scenario.get("roads") or []
        roads = [prepare_road(road) for road in source_roads]
        objects = scenario.get("objects") or []
        changed = False
        for obj in objects:
            new_type = classify_partner(obj.get("type", ""), float(obj.get("length", 0.0)))
            if obj.get("type") != new_type:
                changed = True
            obj["type"] = new_type
        if objects:
            scenario["roads"] = roads
            write_log_binary(scenario, binary_path)
            if changed:
                scenario["roads"] = source_roads
                with open(map_path, "w") as f:
                    json.dump(scenario, f)
            meta = scenario.get("metadata") or {}
            sdc = int(meta.get("sdc_track_index", 0))
            return (i, map_path.name, True, None, sdc)
        write_road_map(roads, binary_path)
        return (i, map_path.name, True, None, None)
    except Exception as e:
        return (i, map_path.name, False, str(e), None)


def retag_saved_logs(data_dir: Path) -> int:
    """Set truck on saved JSON and log binaries. Roads stay as they are.

    A vehicle whose length is at least TRUCK_MIN_LENGTH_M becomes truck.
    Returns the number of objects retagged.
    """
    data_dir = Path(data_dir)
    changed = 0
    for path in sorted((data_dir / "json").glob("*.json")):
        scenario = json.loads(path.read_text())
        touched = False
        for obj in scenario.get("objects") or []:
            new_type = classify_partner(obj.get("type", ""), float(obj.get("length", 0.0)))
            if obj.get("type") != new_type:
                obj["type"] = new_type
                touched = True
                changed += 1
        if touched:
            path.write_text(json.dumps(scenario))
    binary_dir = data_dir / "binaries"
    for path in sorted(binary_dir.glob("map_*.bin")):
        data = bytearray(path.read_bytes())
        if len(data) < 16:
            continue
        nobj, nroads, typ, traj = struct.unpack_from("<iiii", data, 0)
        agent = typ in (VEHICLE, PEDESTRIAN, CYCLIST, TRUCK)
        if not (2 <= traj <= DRIVE_LOG_MAX and 1 <= nobj <= 512 and nroads > 0 and agent):
            continue
        offset = 8
        patched = False
        for _ in range(nobj):
            obj_type, obj_traj = struct.unpack_from("<ii", data, offset)
            if obj_traj != traj:
                raise ValueError(f"{path.name} mixed trajectory lengths")
            width_at = offset + 8 + (7 * traj * 4) + (traj * 4)
            _width, length = struct.unpack_from("<ff", data, width_at)
            if obj_type == VEHICLE and length >= TRUCK_MIN_LENGTH_M:
                struct.pack_into("<i", data, offset, TRUCK)
                patched = True
            offset = width_at + 6 * 4 + 4
        if patched:
            path.write_bytes(data)
    return changed


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
    from tqdm import tqdm

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

    successful = sum(1 for _, _, success, _, _ in results if success)
    failed = sum(1 for _, _, success, _, _ in results if not success)

    print(f"\nProcessed {successful}/{len(results)} maps successfully.")
    if failed > 0:
        print(f"Failed {failed}/{len(results)} files:")
        for i, name, success, error, _sdc in results:
            if not success:
                print(f"  {name}: {error}")
        return
    sdcs = [sdc for _, _, _, _, sdc in results]
    if any(sdc is not None for sdc in sdcs):
        write_sdc_index(binary_dir / "sdc.bin", [0 if sdc is None else sdc for sdc in sdcs])
        print(f"Wrote {binary_dir / 'sdc.bin'} ({len(sdcs)} maps).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Waymo Open Motion JSON into PufferDrive logs."
    )
    parser.add_argument("--data_dir", type=str, default=None,
        help="Dataset root with json/ and binaries/.")
    parser.add_argument("--data_folder", type=str, default=None,
        help="Folder of JSON scenarios. Overrides --data_dir.")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Directory for map binaries and sdc.bin.")
    parser.add_argument("--max_maps", type=int, default=50_000,
        help="Maximum number of maps to process.")
    parser.add_argument("--num_workers", type=int, default=None,
        help="Number of parallel workers.")
    args = parser.parse_args()
    data_folder = args.data_folder
    output_dir = args.output_dir
    if args.data_dir:
        root = Path(args.data_dir)
        if data_folder is None:
            data_folder = str(root / "json")
        if output_dir is None:
            output_dir = str(root / "binaries")
    if not data_folder or not output_dir:
        parser.error("pass --data_dir, or both --data_folder and --output_dir")

    process_all_maps(
        data_folder=data_folder,
        output_dir=output_dir,
        max_maps=args.max_maps,
        num_workers=args.num_workers,
    )
