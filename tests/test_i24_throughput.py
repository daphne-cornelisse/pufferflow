import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


DRIVE_DIR = Path(__file__).resolve().parents[1] / "ocean" / "drive"
sys.path.insert(0, str(DRIVE_DIR))
MODULE_PATH = DRIVE_DIR / "plot_i24_throughput.py"
SPEC = importlib.util.spec_from_file_location("plot_i24_throughput", MODULE_PATH)
THROUGHPUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(THROUGHPUT)


class I24ThroughputTest(unittest.TestCase):
    def object(self, object_id, positions, direction=1, start_tick=0):
        return {
            "id": object_id,
            "direction": direction,
            "start_tick": start_tick,
            "position_m": [[x, 2.0] for x in positions],
            "velocity_mps": [[0.0, 0.0] for _ in positions],
            "length_m": 4.0,
            "width_m": 2.0,
            "quality_flags": [],
        }

    def scene(self, objects=None):
        return {
            "scene": 1,
            "map": {"x_range_m": [0.0, 100.0]},
            "objects": objects or [],
        }

    def test_crossing_time_is_interpolated_for_both_directions(self):
        eastbound = self.object(1, [0.0, 10.0], direction=1, start_tick=10)
        westbound = self.object(2, [10.0, 0.0], direction=-1, start_tick=20)
        self.assertAlmostEqual(
            THROUGHPUT.trajectory_crossing_time_tick(eastbound, 2.5), 10.25
        )
        self.assertAlmostEqual(
            THROUGHPUT.trajectory_crossing_time_tick(westbound, 7.5), 20.25
        )

    def test_crossing_is_counted_only_in_travel_direction(self):
        eastbound_reversal = self.object(1, [10.0, 0.0], direction=1)
        self.assertIsNone(
            THROUGHPUT.trajectory_crossing_time_tick(eastbound_reversal, 5.0)
        )

    def test_ideal_station_roles_follow_travel_direction(self):
        east = THROUGHPUT.ideal_loop_stations(self.scene(), 1)
        west = THROUGHPUT.ideal_loop_stations(self.scene(), -1)
        self.assertEqual([position for _, position in east], [25.0, 50.0, 75.0])
        self.assertEqual([position for _, position in west], [75.0, 50.0, 25.0])

    def test_collect_crossings_uses_scene_relative_seconds(self):
        scene = self.scene([self.object(1, [0.0, 10.0], start_tick=100)])
        events = THROUGHPUT.collect_crossing_events(
            scene, 1, [5.0], False, scene_start_tick=90
        )
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].time_s, 1.05)

    def test_time_bins_preserve_partial_final_interval(self):
        np.testing.assert_allclose(
            THROUGHPUT.time_bin_edges(65.0, 30.0), [0.0, 30.0, 60.0, 65.0]
        )

    def test_complete_time_bins_drop_partial_final_interval(self):
        np.testing.assert_allclose(
            THROUGHPUT.complete_time_bin_edges(65.0, 30.0),
            [0.0, 30.0, 60.0],
        )

    def test_binned_gate_counts(self):
        events = [
            THROUGHPUT.CrossingEvent(1, 1, 25.0, 1, 2.0),
            THROUGHPUT.CrossingEvent(1, 1, 25.0, 2, 12.0),
            THROUGHPUT.CrossingEvent(1, 1, 50.0, 1, 7.0),
        ]
        counts = THROUGHPUT.binned_gate_counts(
            events, np.asarray([25.0, 50.0]), np.asarray([0.0, 10.0, 20.0])
        )
        np.testing.assert_array_equal(counts, [[1, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
