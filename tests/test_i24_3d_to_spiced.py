import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ocean" / "drive" / "i24_3d_to_spiced.py"
SPEC = importlib.util.spec_from_file_location("i24_3d_to_spiced", MODULE_PATH)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)

DATASET_PATH = Path(__file__).resolve().parents[1] / "ocean" / "drive" / "dataset.py"
DATASET_SPEC = importlib.util.spec_from_file_location("drive_dataset", DATASET_PATH)
DATASET = importlib.util.module_from_spec(DATASET_SPEC)
DATASET_SPEC.loader.exec_module(DATASET)


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
                    "geometry": [CONVERTER._point(0, 1.8288), CONVERTER._point(500, 1.8288)],
                }
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
                # Centimeter-scale annotation noise while the vehicle is stopped.
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

    def test_normalizes_native_samples_to_10_hz_in_meters(self):
        obj = CONVERTER.normalize_object(self.raw_object(4, 6.0), self.road_grid)
        self.assertEqual(len(obj["position_m"]), 101)
        self.assertAlmostEqual(obj["position_m"][-1][0], 300 * CONVERTER.FEET_TO_METERS)
        self.assertAlmostEqual(obj["velocity_mps"][50][0], 30 * CONVERTER.FEET_TO_METERS)
        self.assertAlmostEqual(obj["heading_rad"][50], 0.0)

    def test_fuses_overlapping_per_camera_annotations_at_10_hz(self):
        rows = []
        for camera, start_x, y in (("p1c1", 0.0, 6.0), ("p1c2", 2.0, 8.0)):
            for timestamp, offset in ((10.0, 0.0), (10.2, 2.0)):
                rows.append(
                    {
                        "frame": round((timestamp - 10.0) * 30),
                        "camera": camera,
                        "timestamp": timestamp,
                        "x": start_x + offset,
                        "y": y,
                        "l": 15.0,
                        "w": 6.0,
                        "h": 5.0,
                        "direction": 1,
                        "class": "sedan",
                        "gen": "manual",
                    }
                )

        raw = CONVERTER.annotation_rows_to_raw_object(4, rows)

        self.assertEqual(raw["timestamp"], [10.0, 10.1, 10.2])
        self.assertEqual(raw["x_position"], [1.0, 2.0, 3.0])
        self.assertEqual(raw["y_position"], [7.0, 7.0, 7.0])
        self.assertEqual(raw["source_camera_count"], 2)
        self.assertEqual(raw["annotation_gap_filled_ticks"], 0)
        self.assertEqual(raw["trajectory_source"], "manual_and_interpolated_annotations")

    def test_heading_stabilization_is_opt_in_and_only_changes_headings(self):
        raw = self.raw_object_with_noisy_stop()
        default = CONVERTER.normalize_object(raw, self.road_grid)
        explicit_off = CONVERTER.normalize_object(
            raw, self.road_grid, stabilize_headings=False
        )
        stabilized = CONVERTER.normalize_object(
            raw, self.road_grid, stabilize_headings=True
        )

        self.assertEqual(default["heading_rad"], explicit_off["heading_rad"])
        self.assertEqual(default["position_m"], stabilized["position_m"])
        self.assertEqual(default["velocity_mps"], stabilized["velocity_mps"])

        stop_slice = slice(25, 36)
        self.assertGreater(
            max(abs(value) for value in default["heading_rad"][stop_slice]), 0.5
        )
        self.assertLess(
            max(abs(value) for value in stabilized["heading_rad"][stop_slice]), 0.1
        )

        metadata = CONVERTER._heading_estimation_metadata(True)
        self.assertTrue(metadata["stabilized"])
        self.assertFalse(metadata["positions_modified"])
        self.assertFalse(metadata["velocities_modified"])

    def test_builds_spiced_91_state_scenario(self):
        sdc = CONVERTER.normalize_object(self.raw_object(4, 6.0), self.road_grid)
        neighbor = CONVERTER.normalize_object(self.raw_object(9, 18.0), self.road_grid)
        scene = {"scene": 2, "objects": [sdc, neighbor], "map": self.road_grid}
        scenario = CONVERTER.build_spiced_scenario(
            scene, sdc_id=4, start_tick=sdc["start_tick"], window_index=0, max_objects=32
        )
        CONVERTER.validate_spiced_scenario(scenario)
        self.assertEqual(scenario["metadata"]["sdc_track_index"], 0)
        self.assertEqual(len(scenario["objects"]), 2)
        self.assertEqual(len(scenario["objects"][0]["position"]), 91)
        self.assertTrue(all(scenario["objects"][0]["valid"]))
        self.assertEqual(scenario["objects"][1]["mark_as_expert"], 1)
        self.assertTrue(math.isclose(scenario["objects"][0]["position"][0]["x"], 0.0))
        preview = CONVERTER.build_preview_html(scenario)
        self.assertIn('id="i24-scenario-viz"', preview)
        self.assertIn(scenario["scenario_id"], preview)
        self.assertNotIn("<html", preview.lower())

    def test_plans_all_complete_non_overlapping_windows(self):
        obj = CONVERTER.normalize_object(
            self.raw_object(4, 6.0, seconds=20.0), self.road_grid
        )
        starts = CONVERTER.window_start_ticks(
            obj, sdc_offset=0.0, window_stride=9.0, num_windows=0
        )
        self.assertEqual(starts, [obj["start_tick"], obj["start_tick"] + 90])
        self.assertEqual(
            CONVERTER.window_start_ticks(
                obj, sdc_offset=0.0, window_stride=9.0, num_windows=1
            ),
            starts[:1],
        )

    def test_builds_full_length_scenario_with_late_context(self):
        sdc = CONVERTER.normalize_object(
            self.raw_object(4, 6.0, start=1000.0, seconds=20.0), self.road_grid
        )
        late_neighbor = CONVERTER.normalize_object(
            self.raw_object(9, 18.0, start=1015.0, seconds=5.0), self.road_grid
        )
        scene = {"scene": 2, "objects": [sdc, late_neighbor], "map": self.road_grid}
        scenario = CONVERTER.build_full_length_scenario(scene, sdc_id=4, max_objects=2)
        CONVERTER.validate_spiced_scenario(scenario, max_objects=2)
        self.assertEqual(scenario["scenario_id"], "i24s2v4full")
        self.assertEqual(scenario["metadata"]["frame_count"], 201)
        self.assertEqual(len(scenario["objects"][0]["position"]), 201)
        self.assertEqual(scenario["objects"][1]["valid"][:150], [0] * 150)
        self.assertTrue(all(scenario["objects"][1]["valid"][150:]))

    def test_binary_writer_preserves_full_trajectory_length(self):
        count = 137
        obj = {
            "type": "vehicle",
            "position": [CONVERTER._point(index, 0) for index in range(count)],
            "velocity": [CONVERTER._point(1, 0) for _ in range(count)],
            "heading": [0.0] * count,
            "valid": [1] * count,
        }
        with tempfile.NamedTemporaryFile() as output:
            DATASET.save_map_binary({"objects": [obj], "roads": []}, output.name, 0)
            output.seek(0)
            num_objects, num_roads, obj_type, array_size = struct.unpack("iiii", output.read(16))
        self.assertEqual((num_objects, num_roads, obj_type, array_size), (1, 0, 1, count))


if __name__ == "__main__":
    unittest.main()
