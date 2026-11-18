"""Round-trip checks for the kinematic bicycle and the trailer hitch."""

import math
import unittest

from utils.dynamics import (
    ACCEL_MAX,
    CAR_LIMITS,
    LIMITS,
    MAX_SPEED,
    STEER_MAX,
    TRACTOR_WHEELBASE_M,
    TRUCK_LIMITS,
    bicycle_step,
    discretize_action,
    displacement_errors,
    inverse_action,
    limits_for,
    vehicle_kind,
    wheelbase_for,
)


class DynamicsTest(unittest.TestCase):
    def test_car_inverse_replays_the_logged_step(self):
        start = (0.0, 0.0, 0.2, 12.0, 0.0)
        x, y, heading, vx, vy, _ = bicycle_step(
            *start, wheelbase_for(4.5), 1.5, 0.1, CAR_LIMITS, None,
        )
        accel, steer = inverse_action(
            start[2], start[3], start[4], heading, vx, vy,
            wheelbase_for(4.5), CAR_LIMITS,
        )
        self.assertAlmostEqual(accel, 1.5, places=5)
        self.assertAlmostEqual(steer, 0.1, places=5)
        replay = bicycle_step(
            *start, wheelbase_for(4.5), accel, steer, CAR_LIMITS, None,
        )
        self.assertAlmostEqual(replay[0], x, places=6)
        self.assertAlmostEqual(replay[1], y, places=6)

    def test_shared_bounds_clip_onto_the_grid(self):
        self.assertEqual(vehicle_kind(5.0), "car")
        self.assertEqual(vehicle_kind(12.0), "truck")
        self.assertEqual(vehicle_kind(21.0), "trailer")
        self.assertAlmostEqual(wheelbase_for(21.0), TRACTOR_WHEELBASE_M)
        self.assertIs(limits_for(5.0), limits_for(21.0))
        self.assertEqual(LIMITS.max_speed, MAX_SPEED)
        self.assertEqual(CAR_LIMITS, TRUCK_LIMITS)
        # 100 m/s² and a right-angle steer sit outside the calibrated box.
        heading1 = 1.0
        accel, steer = inverse_action(
            0.0, 30.0, 0.0, heading1, 40.0 * math.cos(heading1),
            40.0 * math.sin(heading1), wheelbase_for(12.0), LIMITS,
        )
        self.assertAlmostEqual(accel, ACCEL_MAX)
        self.assertAlmostEqual(steer, STEER_MAX)
        discrete_accel, discrete_steer = discretize_action(accel, steer)
        self.assertAlmostEqual(discrete_accel, ACCEL_MAX)
        self.assertAlmostEqual(discrete_steer, STEER_MAX)
        self.assertEqual(discretize_action(0.0, 0.0), (0.0, 0.0))

    def test_trailer_heading_follows_a_left_turn(self):
        heading = 0.0
        trailer = 0.0
        x = y = 0.0
        vx, vy = 15.0, 0.0
        wheelbase = wheelbase_for(21.0)
        for _ in range(20):
            x, y, heading, vx, vy, trailer = bicycle_step(
                x, y, heading, vx, vy, wheelbase, 0.0, 0.15,
                TRUCK_LIMITS, trailer,
            )
        self.assertGreater(heading, 0.05)
        self.assertGreater(trailer, 0.0)
        self.assertLess(trailer, heading)

    def test_open_loop_car_error_stays_at_zero(self):
        length = 4.5
        wheelbase = wheelbase_for(length)
        state = (0.0, 0.0, 0.0, 20.0, 0.0)
        poses = [state]
        x, y, heading, vx, vy = state
        for accel, steer in ((0.5, 0.02), (-0.4, -0.03), (0.0, 0.0)):
            x, y, heading, vx, vy, _ = bicycle_step(
                x, y, heading, vx, vy, wheelbase, accel, steer, CAR_LIMITS, None,
            )
            poses.append((x, y, heading, vx, vy))
        errors = displacement_errors(poses, length, discrete=False)
        self.assertTrue(all(error < 1.0e-6 for error in errors))
        self.assertGreater(max(displacement_errors(poses, length, discrete=True)), 0.0)


if __name__ == "__main__":
    unittest.main()
