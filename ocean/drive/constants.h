#pragma once

#include "raylib.h"

// Window playback. pufferenv.h supplies a no-op for other envs.
#define PUF_HAS_SIM_CONTROL

// Entity types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

#define INVALID_POSITION -10000.0f

// One step is 0.1 s. 500 steps is the ini default.
#define SIM_DT 0.1f
// Inclusive log states at 10 Hz. Waymo segments are 91. I24 scenes
// run up to 90 s, so a log is 2..DRIVE_LOG_MAX states.
#define DRIVE_LOG_MAX 901
// Interactive playback only. 20 steps/s is 2x real time so the window is
// easier to follow. Headless eval still runs as fast as it can.
#define PUF_STEPS_PER_SEC 20

// Partners one observation lists. The map may hold more cars than this.
#ifndef MAX_AGENTS
#define MAX_AGENTS 64
#endif
// Cars one map copy may hold. ./build.sh drive sets this from
// config/drive.ini num_agents. Observation width stays MAX_AGENTS.
// A density binary may raise it with -DDRIVE_AGENT_MAX.
#ifndef DRIVE_AGENT_MAX
#define DRIVE_AGENT_MAX 256
#endif

// Collision state. Overlap is reported and left in place.
#define NO_COLLISION 0
#define AGENT_COLLISION 1
#define OFFROAD 2

// Grid map. 5 m cells: the vision spiral is about 50 m and collision
// looks ±2 cells, which covers one bottleneck segment.
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 10
#define SLOTS_PER_CELL (MAX_ENTITIES_PER_CELL * 2 + 1)
#define VISION_RANGE 21

// Observation space
#define MAX_ROAD_SEGMENT_OBSERVATIONS 75

#define PARTNER_FEATURES 7
// Index 6 is the previous step's signed acceleration, clamped to the
// action bounds. Index 7 is the previous step's yaw rate, clamped to
// ±OBS_YAW_ABS. Both are 0 on the first step of a life.
// EGO_TARGET_SPEED is the target in m/s times OBS_SPEED_SCALE, the
// same scale as index 2.
#define EGO_FEATURES 9
#define EGO_TARGET_SPEED (EGO_FEATURES - 1)
// km/h. Training reads target_speed and speed_band from the ini.
// A target of 0 pays nothing. Offline uses this band.
#define SPEED_BAND_KMH 2.0f
// Per second at full lane alignment, times the step. Gigaflow's
// α_velocity. Paid only inside the speed band.
#define SPEED_ALPHA 0.0025f
#define KMH_TO_MPS (1.0f / 3.6f)
// Waymo offline keeps its logged reward, so its target stays off.
#define OFFLINE_TARGET_SPEED_KMH 0.0f
#define ROAD_FEATURES 7

#define OBS_SIZE (EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS \
    + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS)
#define ACT_SIZES {257, 257}
#define NUM_ATNS 2

// Observation normalization
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f

#define OBS_GOAL_SCALE 0.005f
#define OBS_SPEED_SCALE 0.01f
#define OBS_POSITION_SCALE 0.02f
#define OBS_ACCEL_SCALE 0.05f
#define OBS_YAW_SCALE 0.5f
#define OBS_YAW_ABS 5.0f

// Distance thresholds
// Arrival radius. A 5 m disk covers a 5 m centerline segment.
#define MIN_DISTANCE_TO_GOAL 5.0f
#define OBS_DIST_SQ 2500.0f
// Center distance in meters. Farther pairs skip the box test.
#define COLLISION_NEAR 20.0f
#define COLLISION_NEAR_SQ (COLLISION_NEAR * COLLISION_NEAR)
#define COLLISION_BOX_SCALE 0.7f
#define CAR_LENGTH 4.3f
#define CAR_WIDTH 1.8f
#define CAR_HEIGHT 1.5f
// Episode reset draws a speed along the lane, uniform on [0, 50] mph.
#define SPAWN_SPEED_MAX_MPS (50.0f * 1609.344f / 3600.0f)

// Odd counts, so the center bin is 0. move_dynamics steps this table.
#define NUM_ACCEL_BINS 257
#define NUM_STEER_BINS 257
#define ACCEL_MIN -8.0f
#define ACCEL_MAX 8.0f
// Front-wheel angle. tan(δ) = wheelbase / rear-axle radius, and the
// wheelbase is 0.6 of the body length, so atan(0.6) keeps that radius
// at least one body length.
#define STEER_MAX 0.5404195f
#define STEER_MIN -0.5404195f
#define ACCEL_COAST_BIN ((NUM_ACCEL_BINS - 1) / 2)
#define STEER_COAST_BIN ((NUM_STEER_BINS - 1) / 2)

// Right-hand column: playback and follow.
#define SIM_COL_W 132.0f
#define SIM_OBS_W 220.0f
#define SIM_OBS_H 300.0f
#define SIM_OBS_RANGE 12.0f

static float ACCELERATION_VALUES[NUM_ACCEL_BINS];
static float STEERING_VALUES[NUM_STEER_BINS];
static int action_space_ready;

static void init_action_space(void) {
    if (action_space_ready) {
        return;
    }
    // An even count has no center bin, so coast would be a nonzero command.
    assert(NUM_ACCEL_BINS % 2 == 1);
    assert(NUM_STEER_BINS % 2 == 1);
    float accel_step = (ACCEL_MAX - ACCEL_MIN) / (float)(NUM_ACCEL_BINS - 1);
    for (int i = 0; i < NUM_ACCEL_BINS; i++) {
        ACCELERATION_VALUES[i] = ACCEL_MIN + i * accel_step;
    }
    float steer_step = (STEER_MAX - STEER_MIN) / (float)(NUM_STEER_BINS - 1);
    for (int i = 0; i < NUM_STEER_BINS; i++) {
        STEERING_VALUES[i] = STEER_MIN + i * steer_step;
    }
    ACCELERATION_VALUES[ACCEL_COAST_BIN] = 0.0f;
    STEERING_VALUES[STEER_COAST_BIN] = 0.0f;
    action_space_ready = 1;
}

static const float offsets[4][2] = {
    {-1,  1},
    { 1,  1},
    { 1, -1},
    {-1, -1}
};

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
    {-2,  0}, {-1,  0}, {0,  0}, {1,  0}, {2,  0},
    {-2,  1}, {-1,  1}, {0,  1}, {1,  1}, {2,  1},
    {-2,  2}, {-1,  2}, {0,  2}, {1,  2}, {2,  2}
};

// A goal is a later point on this road. Joins are another polyline
// that starts within 1 m and keeps the same direction. The gore is
// wider than that, so it is not a way through.
#define LANE_STITCH 1.0f
#define LANE_HEADING_DOT 0.5f
#define LANE_CAR_RADIUS 2.5f
#define GOAL_FOLLOW_GAP 2.0f
#define LANE_NODE_SNAP 4.0f
#define LANE_SUCC_MAX 32
