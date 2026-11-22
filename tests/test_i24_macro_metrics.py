"""Synthetic checks for whole-scene I-24 macro evaluation."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.macro_metrics.diagnostic_video import oriented_box_vertices
from eval.macro_metrics.full_scene import build_full_scene, eligible_objects
from eval.macro_metrics.plotting import _common_cell_absolute_errors
from eval.macro_metrics.rollout_archive import (
    TRAJECTORY_KEYS,
    archive_metadata,
    load_rollout_archive,
    write_rollout_archive,
)
from eval.macro_metrics.runtime import I24_POLICY_HIDDEN, I24_POLICY_LAYERS
from eval.macro_metrics.speed_field import (
    FieldGrid,
    SpeedField,
    build_speed_field,
    velocity_error,
)
from eval.macro_metrics.velocity_mape import _rollout_seed, _worker_count


def _track(track_id, start, count, direction=1, flags=()):
    positions = [[float(index + start), 2.0] for index in range(count)]
    track = {
        "id": track_id,
        "type": "vehicle",
        "direction": direction,
        "length_m": 4.5,
        "width_m": 1.8,
        "height_m": 1.5,
        "start_tick": start,
        "position_m": positions,
        "velocity_mps": [[10.0 * direction, 0.0] for _ in positions],
        "heading_rad": [0.0 if direction == 1 else math.pi for _ in positions],
    }
    if flags:
        track["quality_flags"] = list(flags)
    return track


class FullSceneTest(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "scene": 2,
            "objects": [
                _track(10, 100, 12),
                _track(20, 105, 10, direction=-1),
                _track(30, 100, 9),
                _track(40, 100, 12, flags=("speed_above_60_mps",)),
            ],
            "map": {
                "roads": [
                    {
                        "id": 0,
                        "type": "lane",
                        "map_element_id": 1,
                        "geometry": [
                            {"x": 0.0, "y": 2.0, "z": 0.0},
                            {"x": 100.0, "y": 2.0, "z": 0.0},
                        ],
                    }
                ]
            },
        }

    def test_eligibility_is_explicit(self):
        kept, excluded = eligible_objects(self.scene)
        self.assertEqual([item["id"] for item in kept], [10, 20])
        self.assertEqual(
            {item["id"]: item["reason"] for item in excluded},
            {30: "fewer_than_10_states", 40: "major_quality_flag"},
        )

    def test_global_clock_and_valid_intervals(self):
        scenario = build_full_scene(self.scene)
        self.assertEqual(scenario["metadata"]["source_start_tick"], 100)
        self.assertEqual(scenario["metadata"]["source_end_tick"], 114)
        self.assertEqual(scenario["metadata"]["frame_count"], 15)
        first, second = scenario["objects"]
        self.assertEqual(first["valid"], [True] * 12 + [False] * 3)
        self.assertEqual(second["valid"], [False] * 5 + [True] * 10)
        self.assertTrue(scenario["metadata"]["all_active_vehicles_policy_controlled"])


class SpeedFieldTest(unittest.TestCase):
    def setUp(self):
        self.grid = FieldGrid(
            time_edges=np.array([0.0, 1.0, 2.0]),
            position_edges=np.array([0.0, 15.0, 30.0]),
        )

    def test_unsmoothed_is_per_cell_average_and_empty_is_nan(self):
        field = build_speed_field(
            np.array([0.1, 0.2, 1.1]),
            np.array([2.0, 3.0, 20.0]),
            np.array([10.0, 20.0, 5.0]),
            np.array([True, True, True]),
            self.grid,
        )
        self.assertAlmostEqual(field.values[0, 0], 15.0)
        self.assertAlmostEqual(field.values[1, 1], 5.0)
        self.assertTrue(np.isnan(field.values[0, 1]))
        self.assertEqual(field.sample_count[0, 0], 2)

    def test_smoothing_filters_sums_and_counts(self):
        field = build_speed_field(
            np.array([0.1]), np.array([2.0]), np.array([10.0]), np.array([True]),
            self.grid, smooth_time_seconds=1.0, smooth_space_meters=15.0,
        )
        self.assertTrue(np.isfinite(field.values).all())
        np.testing.assert_allclose(field.values, 10.0)

    def test_velocity_error_separates_support(self):
        gt = SpeedField(
            np.array([[10.0, 0.0], [np.nan, 20.0]]), np.ones((2, 2))
        )
        sim = SpeedField(
            np.array([[12.0, 2.0], [30.0, np.nan]]), np.ones((2, 2))
        )
        result = velocity_error(gt, sim)
        self.assertAlmostEqual(result["mape_percent"], 20.0)
        self.assertAlmostEqual(result["stabilized_mape_percent"], 110.0)
        self.assertEqual(result["common_cells"], 2)
        self.assertEqual(result["mape_cells"], 1)
        self.assertEqual(result["zero_gt_cells"], 1)
        self.assertEqual(result["missing_support_cells"], 1)
        self.assertEqual(result["extra_support_cells"], 1)

    def test_cell_error_distribution_uses_only_common_support(self):
        payload = {
            "gt_raw_wb": np.array([[10.0, np.nan], [5.0, 20.0]]),
            "sim_raw_wb": np.array(
                [
                    [[12.0, 30.0], [np.nan, 17.0]],
                    [[9.0, np.nan], [8.0, 20.0]],
                ]
            ),
        }
        errors = _common_cell_absolute_errors(payload, "raw", "wb")
        np.testing.assert_allclose(np.sort(errors), [0.0, 1.0, 2.0, 3.0, 3.0])


class DiagnosticVideoTest(unittest.TestCase):
    def test_oriented_vehicle_box(self):
        vertices = oriented_box_vertices(
            np.array([10.0]),
            np.array([5.0]),
            np.array([0.0]),
            np.array([4.0]),
            np.array([2.0]),
        )
        np.testing.assert_allclose(
            vertices[0],
            [[12.0, 6.0], [12.0, 4.0], [8.0, 4.0], [8.0, 6.0]],
        )


class PolicyConfigurationTest(unittest.TestCase):
    def test_i24_checkpoint_architecture(self):
        self.assertEqual(I24_POLICY_HIDDEN, 128)
        self.assertEqual(I24_POLICY_LAYERS, 1)

    def test_parallel_seed_and_worker_rules(self):
        self.assertEqual([_rollout_seed(101, index) for index in range(3)], [101, 102, 103])
        self.assertEqual(_worker_count(2, 8), 2)
        self.assertEqual(_worker_count(100, 8), 8)


class RolloutArchiveTest(unittest.TestCase):
    def setUp(self):
        self.trajectory = {
            "ids": np.array([10], dtype=np.int64),
            "direction": np.array([-1], dtype=np.int8),
            "x": np.array([[1.0, 2.0]], dtype=np.float32),
            "y": np.array([[3.0, 3.0]], dtype=np.float32),
            "heading": np.array([[math.pi, math.pi]], dtype=np.float32),
            "speed": np.array([[10.0, 10.0]], dtype=np.float32),
            "valid": np.array([[True, True]]),
            "length": np.array([4.5], dtype=np.float32),
            "width": np.array([1.8], dtype=np.float32),
            "collision_state": np.array([[0, 0]], dtype=np.int8),
            "action_accel_bin": np.array([[128.0, np.nan]], dtype=np.float32),
            "action_steer_bin": np.array([[128.0, np.nan]], dtype=np.float32),
        }
        self.metadata = archive_metadata(
            scene=3,
            rollout=0,
            policy_seed=101,
            checkpoint="rl_i24_coef0p068",
            checkpoint_fingerprint="abc123",
            scenario_fingerprint="scenario123",
            implementation_fingerprint="implementation123",
            hidden=128,
            layers=1,
        )

    def test_atomic_archive_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "r000.npz"
            write_rollout_archive(path, self.trajectory, self.metadata)
            loaded = load_rollout_archive(path, self.metadata)
            self.assertEqual(set(loaded), set(TRAJECTORY_KEYS))
            np.testing.assert_allclose(loaded["x"], self.trajectory["x"])
            np.testing.assert_array_equal(loaded["valid"], self.trajectory["valid"])

    def test_archive_rejects_incompatible_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r000.npz"
            write_rollout_archive(path, self.trajectory, self.metadata)
            wrong = dict(self.metadata)
            wrong["policy_seed"] = np.int64(102)
            with self.assertRaisesRegex(ValueError, "incompatible provenance"):
                load_rollout_archive(path, wrong)


if __name__ == "__main__":
    unittest.main()
