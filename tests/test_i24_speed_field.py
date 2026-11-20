import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


DRIVE_DIR = Path(__file__).resolve().parents[1] / "ocean" / "drive"
sys.path.insert(0, str(DRIVE_DIR))
MODULE_PATH = DRIVE_DIR / "plot_i24_speed_field.py"
SPEC = importlib.util.spec_from_file_location("plot_i24_speed_field", MODULE_PATH)
SPEED_FIELD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPEED_FIELD)


class I24SpeedFieldTest(unittest.TestCase):
    def test_regular_edges_are_step_aligned_and_contain_range(self):
        edges = SPEED_FIELD.regular_edges(-3.0, 24.0, 10.0)
        np.testing.assert_allclose(edges, [-10.0, 0.0, 10.0, 20.0, 30.0])

    def test_field_is_sample_weighted_mean_speed(self):
        objects = [
            {
                "start_tick": 100,
                "position_m": [[1.0, 0.0], [1.5, 0.0]],
                "velocity_mps": [[3.0, 0.0], [3.0, 0.0]],
            },
            {
                "start_tick": 100,
                "position_m": [[1.0, 0.0], [1.5, 0.0]],
                "velocity_mps": [[0.0, 5.0], [0.0, 5.0]],
            },
        ]
        field, counts = SPEED_FIELD.build_speed_field(
            objects,
            scene_start_tick=100,
            time_edges=np.asarray([0.0, 0.2]),
            position_edges=np.asarray([0.0, 2.0]),
            min_samples=1,
            smooth_time_seconds=0.0,
            smooth_space_meters=0.0,
        )
        self.assertEqual(field.shape, (1, 1))
        self.assertEqual(counts[0, 0], 4)
        self.assertAlmostEqual(float(field[0, 0]), 4.0)

    def test_empty_cells_remain_masked_without_smoothing(self):
        objects = [
            {
                "start_tick": 0,
                "position_m": [[1.0, 0.0]],
                "velocity_mps": [[2.0, 0.0]],
            }
        ]
        field, _ = SPEED_FIELD.build_speed_field(
            objects,
            scene_start_tick=0,
            time_edges=np.asarray([0.0, 1.0, 2.0]),
            position_edges=np.asarray([0.0, 2.0, 4.0]),
            min_samples=1,
            smooth_time_seconds=0.0,
            smooth_space_meters=0.0,
        )
        self.assertFalse(bool(field.mask[0, 0]))
        self.assertTrue(bool(field.mask[0, 1]))
        self.assertTrue(bool(field.mask[1, 0]))
        self.assertTrue(bool(field.mask[1, 1]))

    def test_weighted_smoothing_does_not_create_out_of_range_speeds(self):
        objects = [
            {
                "start_tick": 0,
                "position_m": [[1.0, 0.0], [3.0, 0.0]],
                "velocity_mps": [[0.0, 0.0], [10.0, 0.0]],
            }
        ]
        field, _ = SPEED_FIELD.build_speed_field(
            objects,
            scene_start_tick=0,
            time_edges=np.asarray([0.0, 1.0]),
            position_edges=np.asarray([0.0, 2.0, 4.0]),
            min_samples=1,
            smooth_time_seconds=0.0,
            smooth_space_meters=2.0,
        )
        self.assertGreaterEqual(float(field.min()), 0.0)
        self.assertLessEqual(float(field.max()), 10.0)


if __name__ == "__main__":
    unittest.main()
