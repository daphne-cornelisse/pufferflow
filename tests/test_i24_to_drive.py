"""Synthetic checks for the I24 to PufferDrive converter."""

import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "i24.py"
SPEC = importlib.util.spec_from_file_location("i24_to_drive", MODULE_PATH)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


def read_log_binary(path):
    """Mirror load_map_binary's log-record walk. Returns objects and roads."""
    data = Path(path).read_bytes()
    offset = 0

    def take(fmt):
        nonlocal offset
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        return values

    nobj, nroads, obj_type, traj = take("iiii")
    agent_types = {1, 2, 3, 4}
    if not (2 <= traj <= CONVERTER.MAX_TRAJECTORY_STATES) or obj_type not in agent_types or nroads < 1:
        raise AssertionError((nobj, nroads, obj_type, traj))
    objects = []
    for index in range(nobj):
        series = {}
        for name in ("x", "y", "z", "vx", "vy", "vz", "heading"):
            series[name] = take(f"{traj}f")
        valid = take(f"{traj}i")
        width, length, height, gx, gy, gz = take("ffffff")
        mark = take("i")[0]
        next_type, next_size = take("ii")
        objects.append(
            {
                "type": obj_type,
                "x": series["x"],
                "y": series["y"],
                "heading": series["heading"],
                "vx": series["vx"],
                "valid": valid,
                "width": width,
                "length": length,
                "goal": (gx, gy),
                "mark": mark,
            }
        )
        obj_type = next_type
        if index == nobj - 1:
            road_type, road_size = next_type, next_size
    roads = []
    for index in range(nroads):
        if index > 0:
            road_type, road_size = take("ii")
        xs = take(f"{road_size}f")
        ys = take(f"{road_size}f")
        take(f"{road_size}f")
        take("ffffff")
        take("i")
        roads.append({"type": road_type, "x": xs, "y": ys})
    if offset != len(data):
        raise AssertionError(f"unread bytes {len(data) - offset}")
    return objects, roads


class I24ConversionTest(unittest.TestCase):
    def setUp(self):
        self.road_grid = {
            "source": "synthetic",
            "boundary_y_m": [0.0, 3.6576, 7.3152, 21.9456, 25.6032, 29.2608],
            "x_range_m": [0.0, 500.0],
            "roads": [
                {
                    "id": 0,
                    "type": "lane",
                    "map_element_id": 1,
                    "geometry": [
                        CONVERTER._point(0, 1.8288),
                        CONVERTER._point(500, 1.8288),
                    ],
                },
                {
                    "id": 1,
                    "type": "road_edge",
                    "map_element_id": 15,
                    "geometry": [
                        CONVERTER._point(0, 0.0),
                        CONVERTER._point(500, 0.0),
                    ],
                },
            ],
        }

    def raw_object(self, object_id, y_feet, start=1000.0, seconds=10.0):
        count = round(seconds * 30) + 1
        timestamps = [start + index / 30 for index in range(count)]
        return {
            "id": object_id,
            "class": "sedan",
            "direction": 1,
            "l": 15.0,
            "w": 6.0,
            "h": 5.0,
            "timestamp": timestamps,
            "x_position": [(timestamp - start) * 30 for timestamp in timestamps],
            "y_position": [y_feet] * count,
        }

    def raw_object_with_noisy_stop(self, object_id=4):
        start = 1000.0
        count = 6 * 30 + 1
        timestamps = [start + index / 30 for index in range(count)]
        xs = []
        ys = []
        for index, timestamp in enumerate(timestamps):
            elapsed = timestamp - start
            if elapsed < 2.0:
                x = 10.0 * elapsed
                y = 6.0
            elif elapsed < 4.0:
                x = 20.0 + 0.08 * math.sin(index * 1.7)
                y = 6.0 + 0.08 * math.cos(index * 1.3)
            else:
                x = 20.0 + 10.0 * (elapsed - 4.0)
                y = 6.0
            xs.append(x)
            ys.append(y)
        return {
            "id": object_id,
            "class": "sedan",
            "direction": 1,
            "l": 15.0,
            "w": 6.0,
            "h": 5.0,
            "timestamp": timestamps,
            "x_position": xs,
            "y_position": ys,
        }

    def test_road_edge_moves_away_from_the_lanes(self):
        # A lower edge moves down. An upper edge moves up. The shift is 1.5 ft.
        shift = CONVERTER.ROAD_EDGE_OUTWARD_M
        self.assertAlmostEqual(CONVERTER._edge_y_m(0.0, 10.0), -shift)
        self.assertAlmostEqual(
            CONVERTER._edge_y_m(20.0, 10.0),
            20.0 * CONVERTER.FEET_TO_METERS + shift,
        )

    def test_long_roads_are_sampled_every_five_meters(self):
        line = [
            CONVERTER._point(0.0, 0.0),
            CONVERTER._point(12.0, 0.0),
        ]
        sampled = CONVERTER._sample_polyline(line, CONVERTER.ROAD_SAMPLE_METERS)
        self.assertEqual([(p["x"], p["y"]) for p in sampled], [
            (0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (12.0, 0.0),
        ])

    def test_i24_labels_use_womd_categories(self):
        for name in ("sedan", "midsize", "pickup", "van", "semi", "truck", "cs"):
            raw = self.raw_object(1, 6.0)
            raw["class"] = name
            obj = CONVERTER.normalize_object(raw, self.road_grid)
            self.assertEqual(obj["type"], "vehicle")
            self.assertEqual(obj["source_class"], name)
        long = self.raw_object(1, 6.0)
        long["class"] = "semi"
        long["l"] = 80.0
        truck = CONVERTER.normalize_object(long, self.road_grid)
        self.assertEqual(truck["type"], "truck")
        self.assertGreaterEqual(truck["length_m"], 7.5)
        long["class"] = "sedan"
        self.assertEqual(
            CONVERTER.normalize_object(long, self.road_grid)["type"], "truck"
        )
        cycle = self.raw_object(1, 6.0)
        cycle["class"] = "cyclist"
        cycle["l"] = 80.0
        self.assertEqual(
            CONVERTER.normalize_object(cycle, self.road_grid)["type"], "cyclist"
        )
        raw = self.raw_object(1, 6.0)
        raw["class"] = "motorcycle"
        with self.assertRaises(ValueError):
            CONVERTER.normalize_object(raw, self.road_grid)
        self.assertEqual(CONVERTER.drive_road_type(CONVERTER.LANE_FREEWAY), 5)
        self.assertEqual(
            CONVERTER.drive_road_type(CONVERTER.ROAD_LINE_BROKEN_SINGLE_WHITE), 6
        )
        self.assertEqual(CONVERTER.drive_road_type(CONVERTER.ROAD_EDGE_BOUNDARY), 7)
        self.assertEqual(CONVERTER.drive_road_type(CONVERTER.ROAD_EDGE_MEDIAN), 7)

    def test_normalizes_native_samples_to_10_hz_in_meters(self):
        obj = CONVERTER.normalize_object(self.raw_object(4, 6.0), self.road_grid)
        self.assertEqual(len(obj["position_m"]), 101)
        self.assertAlmostEqual(obj["position_m"][-1][0], 300 * CONVERTER.FEET_TO_METERS)
        self.assertAlmostEqual(obj["velocity_mps"][50][0], 30 * CONVERTER.FEET_TO_METERS)
        self.assertAlmostEqual(obj["heading_rad"][50], 0.0)
        self.assertAlmostEqual(
            obj["length_m"], 15.0 * CONVERTER.FEET_TO_METERS * CONVERTER.BOX_SCALE
        )
        self.assertAlmostEqual(
            obj["width_m"], 6.0 * CONVERTER.FEET_TO_METERS * CONVERTER.BOX_SCALE
        )

    def test_heading_stabilization_is_default_and_only_changes_headings(self):
        raw = self.raw_object_with_noisy_stop()
        default = CONVERTER.normalize_object(raw, self.road_grid)
        explicit_off = CONVERTER.normalize_object(
            raw, self.road_grid, stabilize_headings=False
        )
        stabilized = CONVERTER.normalize_object(
            raw, self.road_grid, stabilize_headings=True
        )
        self.assertEqual(default["heading_rad"], stabilized["heading_rad"])
        self.assertEqual(default["position_m"], explicit_off["position_m"])
        self.assertEqual(default["velocity_mps"], explicit_off["velocity_mps"])
        stop_slice = slice(25, 36)
        self.assertGreater(
            max(abs(value) for value in explicit_off["heading_rad"][stop_slice]), 0.5
        )
        self.assertLess(
            max(abs(value) for value in default["heading_rad"][stop_slice]), 0.1
        )

    def test_builds_native_length_scenario_and_log_binary(self):
        sdc = CONVERTER.normalize_object(self.raw_object(4, 6.0), self.road_grid)
        neighbor = CONVERTER.normalize_object(self.raw_object(9, 18.0), self.road_grid)
        scene = {"scene": 2, "objects": [sdc, neighbor], "map": self.road_grid}
        scenario = CONVERTER.build_track_scenario(scene, sdc_id=4, max_objects=32)
        CONVERTER.validate_scenario(scenario)
        self.assertEqual(scenario["metadata"]["sdc_track_index"], 0)
        self.assertEqual(scenario["metadata"]["frame_count"], 101)
        self.assertEqual(len(scenario["objects"]), 2)
        self.assertEqual(len(scenario["objects"][0]["position"]), 101)
        self.assertTrue(all(scenario["objects"][0]["valid"]))
        self.assertEqual(scenario["objects"][0]["mark_as_expert"], 0)
        self.assertTrue(math.isclose(scenario["objects"][0]["position"][0]["x"], 0.0))
        self.assertIn("goalPosition", scenario["objects"][0])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            written = CONVERTER.write_dataset([scenario], out)
            self.assertEqual(written, [out / "json" / "i24_00000.json"])
            objects, roads = read_log_binary(out / "binaries" / "map_000.bin")
            self.assertEqual(len(objects), 2)
            self.assertEqual(len(objects[0]["x"]), 101)
            self.assertEqual(len(roads), 2)
            self.assertEqual(roads[0]["type"], 5)
            self.assertEqual(roads[1]["type"], 7)
            self.assertAlmostEqual(objects[0]["x"][0], 0.0)
            self.assertEqual(objects[0]["valid"][0], 1)
            self.assertGreater(objects[0]["vx"][10], 0.0)
            raw = (out / "binaries" / "sdc.bin").read_bytes()
            count, sdc_index = struct.unpack("<ii", raw)
            self.assertEqual((count, sdc_index), (1, 0))

    def test_caps_a_track_at_the_90_second_scene(self):
        sdc = CONVERTER.normalize_object(
            self.raw_object(4, 6.0, seconds=100.0), self.road_grid
        )
        self.assertEqual(len(sdc["position_m"]), 1001)
        scene = {"scene": 1, "objects": [sdc], "map": self.road_grid}
        scenario = CONVERTER.build_track_scenario(scene, sdc_id=4, max_objects=8)
        CONVERTER.validate_scenario(scenario)
        self.assertEqual(CONVERTER.MAX_TRAJECTORY_STATES, 901)
        self.assertEqual(scenario["metadata"]["frame_count"], 901)
        self.assertAlmostEqual(scenario["metadata"]["duration_seconds"], 90.0)
        end_x = 90.0 * 30.0 * CONVERTER.FEET_TO_METERS
        self.assertAlmostEqual(scenario["objects"][0]["position"][-1]["x"], end_x)

    def test_box_intersection_area(self):
        box = CONVERTER._box_corners(0.0, 0.0, 0.0, 4.0, 2.0)
        self.assertAlmostEqual(CONVERTER._intersection_area(box, box), 8.0)
        shifted = CONVERTER._box_corners(1.0, 0.0, 0.0, 4.0, 2.0)
        self.assertAlmostEqual(CONVERTER._intersection_area(box, shifted), 6.0)
        far = CONVERTER._box_corners(10.0, 0.0, 0.0, 4.0, 2.0)
        self.assertAlmostEqual(CONVERTER._intersection_area(box, far), 0.0)

    def _placed(self, object_id, x_pos, length, width, frames):
        return {
            "id": object_id,
            "type": "vehicle",
            "length": length,
            "width": width,
            "height": 1.5,
            "position": [{"x": x_pos, "y": 0.0, "z": 0.0} for _ in range(frames)],
            "velocity": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(frames)],
            "heading": [0.0] * frames,
            "valid": [True] * frames,
            "goalPosition": {"x": x_pos, "y": 0.0, "z": 0.0},
            "mark_as_expert": 0,
        }

    def test_deep_overlap_drops_smaller_cars_in_one_pass(self):
        # B sits inside A. C overlaps B and only grazes A.
        # Six frames drops B and C. Five frames drops neither.
        for frames, dropped in ((6, {2, 3}), (5, set())):
            objects = [
                self._placed(1, 0.0, 10.0, 2.0, frames),
                self._placed(2, 4.0, 4.0, 2.0, frames),
                self._placed(3, 5.5, 2.0, 2.0, frames),
            ]
            pairs, focal = CONVERTER._deep_overlap_ids(objects)
            self.assertEqual({item["id"] for item in pairs}, dropped)
            self.assertEqual(focal, [])

    def test_wrong_way_and_deep_overlap_are_removed_from_the_scenario(self):
        sdc = CONVERTER.normalize_object(self.raw_object(4, 6.0), self.road_grid)
        neighbor = CONVERTER.normalize_object(self.raw_object(9, 18.0), self.road_grid)
        westbound = self.raw_object(12, 80.0)
        # Upper carriageway travels toward -x.
        westbound["x_position"] = [-value for value in westbound["x_position"]]
        westbound["direction"] = -1
        westbound = CONVERTER.normalize_object(westbound, self.road_grid)
        ghost = self.raw_object(11, 6.0)
        ghost["x_position"] = [-value for value in ghost["x_position"]]
        ghost["direction"] = -1
        ghost = CONVERTER.normalize_object(ghost, self.road_grid)
        small = self.raw_object(15, 18.0)
        small["l"] = 8.0
        small["w"] = 4.0
        small = CONVERTER.normalize_object(small, self.road_grid)
        scene = {
            "scene": 2,
            "objects": [sdc, neighbor, westbound, ghost, small],
            "map": self.road_grid,
        }
        scenario = CONVERTER.build_track_scenario(scene, sdc_id=4, max_objects=8)
        CONVERTER.validate_scenario(scenario)
        self.assertEqual({obj["id"] for obj in scenario["objects"]}, {4, 9, 12})
        cleanup = scenario["metadata"]["track_cleanup"]
        self.assertIn(11, {item["id"] for item in cleanup["removed_wrong_way"]})
        self.assertIn(15, {item["id"] for item in cleanup["removed_deep_overlap"]})
        self.assertFalse(cleanup["focal_kept_despite_wrong_way"])

    def test_stationary_track_is_removed_and_not_chosen_as_sdc(self):
        moving = CONVERTER.normalize_object(self.raw_object(4, 6.0, seconds=4.0), self.road_grid)
        parked = self.raw_object(7, 18.0, seconds=20.0)
        parked["x_position"] = [100.0] * len(parked["x_position"])
        parked = CONVERTER.normalize_object(parked, self.road_grid)
        jitter = self.raw_object(8, 30.0, seconds=8.0)
        jitter["x_position"] = [200.0 + (index % 2) * 0.5 for index in range(len(jitter["x_position"]))]
        jitter = CONVERTER.normalize_object(jitter, self.road_grid)
        scene = {"scene": 1, "objects": [parked, jitter, moving], "map": self.road_grid}
        self.assertEqual(CONVERTER.choose_sdc(scene)["id"], 4)
        scenario = CONVERTER.build_track_scenario(scene, sdc_id=4, max_objects=8)
        self.assertEqual({obj["id"] for obj in scenario["objects"]}, {4})
        cleanup = scenario["metadata"]["track_cleanup"]
        self.assertEqual(
            {item["id"] for item in cleanup["removed_stationary"]}, {7, 8}
        )
        self.assertFalse(cleanup["focal_kept_despite_stationary"])
        for item in cleanup["removed_stationary"]:
            self.assertLessEqual(item["travel_m"], CONVERTER.STATIONARY_MAX_METERS)
        outside = {
            "id": 2,
            "start_tick": 100,
            "position_m": [[0.0, 0.0], [50.0, 0.0]],
            "velocity_mps": [[0.0, 0.0], [0.0, 0.0]],
            "heading_rad": [0.0, 0.0],
        }
        removed, focal = CONVERTER._stationary_ids([outside], 0, 10, sdc_id=9)
        self.assertEqual(removed, [])
        self.assertFalse(focal)

    def test_position_repair_fixes_spikes_and_keeps_a_stop(self):
        clean = [[float(index), 1.0] for index in range(40)]
        repaired, report = CONVERTER._repair_track_positions(clean)
        self.assertEqual(repaired, clean)
        self.assertEqual(report["smoothed_frames"], 0)
        self.assertEqual(report["copied_huge_frames"], 0)
        self.assertIsNone(report["dropped"])

        spiked = [[float(index), 0.0] for index in range(40)]
        spiked[20] = [20.0, 0.8]
        repaired, report = CONVERTER._repair_track_positions(spiked)
        self.assertAlmostEqual(repaired[20][0], 20.0)
        self.assertAlmostEqual(repaired[20][1], 0.0)
        self.assertGreaterEqual(report["smoothed_frames"], 1)
        self.assertEqual(report["copied_huge_frames"], 0)
        self.assertIsNone(report["dropped"])

        huge = [[float(index), 0.0] for index in range(40)]
        huge[20] = [20.0, 2.0]
        repaired, report = CONVERTER._repair_track_positions(huge)
        self.assertAlmostEqual(repaired[20][0], 20.0)
        self.assertAlmostEqual(repaired[20][1], 0.0)
        self.assertGreaterEqual(report["copied_huge_frames"], 1)
        self.assertIsNone(report["dropped"])

        reversal = [[float(index), 0.0] for index in range(80)]
        for index, x_pos in zip(range(41, 48), (39.0, 38.0, 37.2, 36.8, 38.5, 42.0, 46.0)):
            reversal[index][0] = x_pos
        repaired, report = CONVERTER._repair_track_positions(reversal)
        self.assertIsNone(report["dropped"])
        self.assertEqual(len(repaired), len(reversal))
        for index, point in enumerate(repaired):
            self.assertAlmostEqual(point[0], float(index), delta=0.05)
            self.assertAlmostEqual(point[1], 0.0, delta=0.05)

        turned = [[float(index), 0.0] for index in range(40)]
        x_pos = 39.0
        for _ in range(40):
            x_pos -= 1.0
            turned.append([x_pos, 0.0])
        repaired, report = CONVERTER._repair_track_positions(turned)
        if report["dropped"]:
            self.assertIn(report["dropped"], ("unfixable_reversal", "too_strange"))
        else:
            self.assertGreater(
                report["dropped_prefix_frames"] + report["dropped_suffix_frames"], 10
            )
            self.assertLess(len(repaired), len(turned))
            net = repaired[-1][0] - repaired[0][0]
            wrong = 0.0
            for index in range(len(repaired) - 1):
                step = repaired[index + 1][0] - repaired[index][0]
                if step * net < 0.0:
                    wrong += abs(step)
            self.assertLess(wrong, 1.0)

        stopped = CONVERTER.normalize_object(self.raw_object_with_noisy_stop(), self.road_grid)
        self.assertIsNone(stopped["position_repair"]["dropped"])
        self.assertEqual(len(stopped["position_m"]), 61)
        self.assertAlmostEqual(
            stopped["position_m"][30][0], 20.0 * CONVERTER.FEET_TO_METERS, delta=0.05
        )
        self.assertEqual(stopped["position_repair"]["dropped_prefix_frames"], 0)
        self.assertEqual(stopped["position_repair"]["smoothed_frames"], 0)

    def test_native_length_keeps_a_late_neighbor_in_the_log_binary(self):
        sdc = CONVERTER.normalize_object(
            self.raw_object(4, 6.0, start=1000.0, seconds=20.0), self.road_grid
        )
        late_neighbor = CONVERTER.normalize_object(
            self.raw_object(9, 18.0, start=1015.0, seconds=5.0), self.road_grid
        )
        scene = {"scene": 2, "objects": [sdc, late_neighbor], "map": self.road_grid}
        scenario = CONVERTER.build_track_scenario(scene, sdc_id=4, max_objects=2)
        CONVERTER.validate_scenario(scenario, max_objects=2)
        self.assertEqual(scenario["metadata"]["frame_count"], 201)
        self.assertEqual(scenario["objects"][1]["valid"][:150], [False] * 150)
        self.assertTrue(all(scenario["objects"][1]["valid"][150:]))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            written = CONVERTER.write_dataset([scenario], out)
            self.assertEqual(written, [out / "json" / "i24_00000.json"])
            objects, _roads = read_log_binary(out / "binaries" / "map_000.bin")
            self.assertEqual(len(objects[0]["x"]), 201)
            self.assertEqual(objects[1]["valid"][0], 0)
            self.assertEqual(objects[1]["valid"][150], 1)
            self.assertTrue((out / "binaries" / "sdc.bin").exists())


if __name__ == "__main__":
    unittest.main()
