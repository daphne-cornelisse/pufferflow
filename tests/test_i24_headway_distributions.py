import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


DRIVE_DIR = Path(__file__).resolve().parents[1] / "ocean" / "drive"
sys.path.insert(0, str(DRIVE_DIR))
MODULE_PATH = DRIVE_DIR / "plot_i24_headway_distributions.py"
SPEC = importlib.util.spec_from_file_location("plot_i24_headways", MODULE_PATH)
HEADWAYS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HEADWAYS)


class I24HeadwayTest(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "scene": 1,
            "map": {
                "boundary_y_m": [0.0, 4.0, 8.0, 20.0, 24.0, 28.0],
            },
        }

    def object(self, object_id, x, y, direction=1, length=4.0, width=2.0):
        return {
            "id": object_id,
            "direction": direction,
            "start_tick": 0,
            "position_m": [[x, y]],
            "velocity_mps": [[0.0, 0.0]],
            "length_m": length,
            "width_m": width,
            "quality_flags": [],
        }

    def test_lane_state_flags_vehicle_footprint_crossing_boundary(self):
        centered = self.object(1, 0.0, 2.0)
        crossing = self.object(2, 0.0, 3.5)
        self.assertFalse(HEADWAYS.lane_state(self.scene, centered, 0).ambiguous)
        self.assertTrue(HEADWAYS.lane_state(self.scene, crossing, 0).ambiguous)

    def test_short_return_flicker_requires_a_b_a_pattern(self):
        self.assertEqual(HEADWAYS.short_return_flickers([1, 1, 2, 1, 1], 2), 1)
        self.assertEqual(HEADWAYS.short_return_flickers([1, 1, 2, 2, 1], 1), 0)
        self.assertEqual(HEADWAYS.short_return_flickers([1, 1, 2, 2], 3), 0)

    def test_confirmed_switches_ignore_boundary_overlap_samples(self):
        states = [
            HEADWAYS.LaneState(1, False, False),
            HEADWAYS.LaneState(1, True, False),
            HEADWAYS.LaneState(2, True, False),
            HEADWAYS.LaneState(2, False, False),
        ]
        self.assertEqual(HEADWAYS.confirmed_switches(states), 1)

    def test_collects_direction_aligned_bumper_gap(self):
        eastbound = {
            **self.scene,
            "objects": [
                self.object(1, 0.0, 2.0, direction=1),
                self.object(2, 10.0, 2.0, direction=1),
            ],
        }
        westbound = {
            **self.scene,
            "objects": [
                self.object(3, 20.0, 22.0, direction=-1),
                self.object(4, 10.0, 22.0, direction=-1),
            ],
        }
        east_observations = HEADWAYS.collect_headway_observations(
            eastbound, False, 0.0, 1.0
        )
        west_observations = HEADWAYS.collect_headway_observations(
            westbound, False, 0.0, 1.0
        )
        self.assertEqual(len(east_observations), 1)
        self.assertEqual(len(west_observations), 1)
        self.assertAlmostEqual(east_observations[0].gap_m, 6.0)
        self.assertAlmostEqual(west_observations[0].gap_m, 6.0)

    def test_effective_stream_state_follows_curved_centerline(self):
        obj = self.object(1, 10.0, 2.6)
        model = HEADWAYS.EffectiveStreamModel(
            direction=1,
            x_reference_m=0.0,
            x_scale_m=10.0,
            coefficients=((0.5, 2.0), (0.5, 6.0)),
            density_peaks_m=(2.0, 6.0),
        )
        state = HEADWAYS.effective_stream_state(
            self.scene, obj, 0, model, np.asarray([2.6])
        )
        self.assertEqual(state.lane, 1)
        self.assertFalse(state.ambiguous)
        self.assertAlmostEqual(state.residual_m, 0.1)

    def test_effective_stream_collector_requires_residual_overlap(self):
        scene = {
            **self.scene,
            "scene": 2,
            "objects": [
                self.object(1, 0.0, 2.0, direction=1),
                self.object(2, 10.0, 2.1, direction=1),
            ],
        }
        models = {
            1: HEADWAYS.EffectiveStreamModel(
                1, 0.0, 1.0, ((2.0,), (6.0,)), (2.0, 6.0)
            )
        }
        observations, diagnostics = HEADWAYS.collect_effective_headway_observations(
            scene,
            False,
            1.0,
            models,
            {1: np.asarray([2.0]), 2: np.asarray([2.1])},
            1.5,
            0.75,
            0.25,
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(diagnostics["accepted_pair_count"], 1)
        self.assertAlmostEqual(observations[0].gap_m, 6.0)


if __name__ == "__main__":
    unittest.main()
