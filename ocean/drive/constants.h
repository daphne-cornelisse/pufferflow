#pragma once

#include "raylib.h"

// Window playback. pufferenv.h supplies a no-op for other envs.
#define PUF_HAS_SIM_CONTROL
#define PUF_HAS_AGENT_SLIDER

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

// Velocity diagram. One heatmap. Each segment is one lane, listed on y.
// A cell is the mean speed of the cars on that lane. Time is on x.
// Colors match CIRCLES Fig. 7, 10 mph through 80 mph. The panel beside it
// marks a new crash in red and an off-road event in yellow.
#define FLOW_MPS_TO_MPH 2.2369363f
#define FLOW_MPH_LO 10.0f
#define FLOW_MPH_HI 80.0f
#define FLOW_CMAP_N 16
#define FLOW_LANE_MAX 8
#define FLOW_PANEL_HEAD 40.0f
#define FLOW_PANEL_FOOT 36.0f
#define FLOW_PANEL_ROW 40.0f
#define FLOW_PANEL_W 480.0f
#define FLOW_PANEL_GAP 8.0f
#define FLOW_HIT_MAX 1024
// Half the 3.2 m lane spacing. A vehicle is counted on the closest centerline.
#define FLOW_LATERAL 1.6f
// A collision just off the paint still belongs to the nearest lane.
#define FLOW_HIT_REACH 8.0f
// Split a new lane when centerline y jumps by more than this.
#define FLOW_LANE_GAP 1.2f
// |dy| below this is a horizontal lane. The ramp is steeper.
#define FLOW_FLAT_DY 1.0f

// Agent limits
#ifndef MAX_AGENTS
#define MAX_AGENTS 64
#endif
// Cars one map copy may spawn. The viewer slider runs from 1 to this.
#define DRIVE_AGENT_MAX 128
// Mainline demand slider, vehicles per hour. 0 inserts nobody.
#define DRIVE_VPH_MAX 10000
// Train: curriculum goals, fixed cars, respawn. puf_init starts here so the
// vectorized trainer, which only calls puf_init, samples those goals.
// Inference: ./drive sets this before the first reset. Goals sit at the
// route end, and cars enter at the start.
#define DRIVE_MODE_TRAIN 0
#define DRIVE_MODE_INFERENCE 1
// Partners listed in one observation. Fixed even when more cars exist.
#define MAX_OBS_AGENTS 64

// Collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Grid map
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 10
#define SLOTS_PER_CELL (MAX_ENTITIES_PER_CELL * 2 + 1)
#define VISION_RANGE 21

// Observation space
#define MAX_ROAD_SEGMENT_OBSERVATIONS 75

#define PARTNER_FEATURES 7
// Index 6 stays 0 so an old checkpoint keeps that weight. Index 7
// is this episode's max speed divided by MAX_SPEED.
#define EGO_FEATURES 8
#define ROAD_FEATURES 7

#define OBS_SIZE (EGO_FEATURES + PARTNER_FEATURES * MAX_OBS_AGENTS \
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
#define MIN_DISTANCE_TO_GOAL 5.0f
// Steps of collision immunity after a respawn. Used when the route has
// no open point and the fallback still overlaps. Off-road is still penalized.
#define RESPAWN_GRACE 10
// A respawn must sit strictly farther than this from every other agent.
#define RESPAWN_CLEAR 5.0f
// Along-route sample. A gap wider than this is found.
#define RESPAWN_STEP 2.0f
// Past the trip by this much also ends an inference lap.
#define RESPAWN_OVERRUN 40.0f
#define COLLISION_DIST_SQ 225.0f
// Closing-speed bounce. 0 stops at the contact, 1 sends it back at full speed.
#define BUMP_BOUNCE 0.4f
// Extra gap so the next step does not start already overlapping.
#define BUMP_SLOP 0.05f
// Cap on one shove, in meters. A bad normal cannot throw a car across the map.
#define BUMP_MAX 4.0f
#define OBS_DIST_SQ 2500.0f
#define COLLISION_BOX_SCALE 0.7f

// Action space
#define NUM_ACCEL_BINS 7
#define NUM_STEER_BINS 13

// Lowest target speed the viewer slider can stage, in m/s.
#define SIM_SPEED_MIN 1.0f

// Right-hand column: playback, follow, agents.
#define SIM_COL_W 132.0f
#define SIM_AGENT_H 46.0f
#define SIM_DIST_H 78.0f
#define SIM_OBS_W 220.0f
#define SIM_OBS_H 264.0f
#define SIM_OBS_RANGE 12.0f

// Demand tables in flow_init.h. Indices start at 0 and have no gaps.
#define FLOW_MAX 16
#define FLOW_MAX_EDGES 16
#define FLOW_MAX_LANES 16
#define FLOW_NAME 64
#define DEPART_POS_EPS 0.1f
// Center-to-center gap required at the entrance. length 4.3 + minGap 2.8.
#define FLOW_INSERT_GAP 8.0f
// Departures held while the entrance is blocked or every slot is full.
#define FLOW_QUEUE 32

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

static const Color STONE_GRAY = (Color){80, 80, 80, 255};
static const Color PUFF_RED = (Color){187, 0, 0, 255};
static const Color PUFF_CYAN = (Color){0, 187, 187, 255};
static const Color PUFF_WHITE = (Color){241, 241, 241, 241};
static const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
static const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
static const Color ROAD_COLOR = (Color){35, 35, 37, 255};
