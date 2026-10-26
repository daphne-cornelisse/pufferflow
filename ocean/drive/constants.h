#pragma once

#include "raylib.h"

// Window playback. pufferenv.h supplies a no-op for other envs.
#define PUF_HAS_SIM_CONTROL

// Entity types
#define NONE 0
#define VEHICLE 1
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
// Interactive playback only. 20 steps/s is 2x real time so the window is
// easier to follow. Headless eval still runs as fast as it can.
#define PUF_STEPS_PER_SEC 20

// Partners one observation lists. The map may hold more cars than this.
#ifndef MAX_AGENTS
#define MAX_AGENTS 64
#endif
// Cars one map copy may hold. config num_agents must be in 1..this.
// A throughput binary may raise this with -DDRIVE_AGENT_MAX without
// changing the observation width.
#ifndef DRIVE_AGENT_MAX
#define DRIVE_AGENT_MAX 128
#endif

// Collision state
#define NO_COLLISION 0
#define AGENT_COLLISION 1
#define OFFROAD 2

// Agent-agent response. bump separates the cars. through leaves the overlap.
#define COLLISION_BUMP 0
#define COLLISION_THROUGH 1

// Grid map. 5 m cells: the vision spiral is about 50 m and collision
// looks ±2 cells, which covers one bottleneck segment.
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 10
#define SLOTS_PER_CELL (MAX_ENTITIES_PER_CELL * 2 + 1)
#define VISION_RANGE 21

// Observation space
#define MAX_ROAD_SEGMENT_OBSERVATIONS 75

#define PARTNER_FEATURES 7
// Index 6 stays 0 so an old checkpoint keeps that weight. Index 7
// stays 0. It is open.
#define EGO_FEATURES 8
#define ROAD_FEATURES 7

#define OBS_SIZE (EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS \
    + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS)
#define ACT_SIZES {7, 13}
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

// Distance thresholds
// Arrival radius. A 5 m disk covers a 5 m centerline segment.
#define MIN_DISTANCE_TO_GOAL 5.0f
// Closing-speed bounce. 0 stops at the contact, 1 sends it back at full speed.
#define BUMP_BOUNCE 0.4f
// Extra gap so the next step does not start already overlapping.
#define BUMP_SLOP 0.05f
// Cap on one shove, in meters. A bad normal cannot throw a car across the map.
#define BUMP_MAX 4.0f
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

// Action space
#define NUM_ACCEL_BINS 7
#define NUM_STEER_BINS 13

// Right-hand column: playback and follow.
#define SIM_COL_W 132.0f
#define SIM_OBS_W 220.0f
#define SIM_OBS_H 264.0f
#define SIM_OBS_RANGE 12.0f

static const float ACCELERATION_VALUES[NUM_ACCEL_BINS] = {
    -4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f
};

static const float STEERING_VALUES[NUM_STEER_BINS] = {
    -1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f,
     0.167f,  0.333f,  0.500f,  0.667f,  0.833f,  1.000f
};

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
#define GOAL_FOLLOW_GAP 8.0f
#define LANE_NODE_SNAP 4.0f
#define LANE_SUCC_MAX 16
