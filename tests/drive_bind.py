"""ctypes binding for build/libdrive_offline.so."""

import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libdrive_offline.so"
TRAJ = 91


def load():
    lib = ctypes.CDLL(str(LIB_PATH))
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
    handle = ctypes.c_void_p

    def fn(name, restype, argtypes):
        f = getattr(lib, name)
        f.restype = restype
        f.argtypes = argtypes
        return f

    fn("pf_create_sim", handle, [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_char_p,
    ])
    fn("pf_create_scenario", handle, [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
    ])
    fn("pf_close", None, [handle])
    fn("pf_reset", None, [handle, ctypes.c_uint])
    fn("pf_step", None, [handle])
    fn("pf_integrate", None, [handle, ctypes.c_int])
    fn("pf_recompute", None, [handle])
    fn("pf_place", None, [handle, ctypes.c_int])
    fn("pf_replay", None, [handle, ctypes.c_int])
    fn("pf_spawn_due", None, [handle])
    fn("pf_set_timestep", None, [handle, ctypes.c_int])
    fn("pf_collect", None, [handle, ctypes.c_int, c_float_p, c_int_p, c_uint8_p, c_float_p])
    fn("pf_n", ctypes.c_int, [handle])
    fn("pf_obs_size", ctypes.c_int, [handle])
    fn("pf_traj", ctypes.c_int, [])
    fn("pf_obs", c_float_p, [handle])
    fn("pf_set_action", None, [handle, ctypes.c_int, ctypes.c_float, ctypes.c_float])
    fn("pf_relabel_grid", None, [
        handle, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_int,
    ])
    fn("pf_set_action_value", None, [
        handle, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ])
    fn("pf_reward", ctypes.c_float, [handle, ctypes.c_int])
    fn("pf_timestep", ctypes.c_int, [handle])
    fn("pf_agent", None, [handle, ctypes.c_int, c_float_p])
    fn("pf_set_agent", None, [handle, ctypes.c_int] + [ctypes.c_float] * 9)
    fn("pf_set_goal", None, [handle, ctypes.c_int, ctypes.c_float, ctypes.c_float])
    fn("pf_mean", None, [handle, c_float_p, c_float_p])
    fn("pf_log_n", ctypes.c_int, [handle])
    fn("pf_sdc", ctypes.c_int, [handle])
    fn("pf_base_unfit", ctypes.c_int, [handle, ctypes.c_int])
    fn("pf_log_valid", ctypes.c_int, [handle, ctypes.c_int, ctypes.c_int])
    fn("pf_log_xy", None, [handle, ctypes.c_int, ctypes.c_int, c_float_p, c_float_p])
    fn("pf_log_bins", None, [handle, ctypes.c_int, ctypes.c_int, c_int_p, c_int_p])
    fn("pf_outcome", None, [
        handle, ctypes.c_int, c_float_p, c_float_p, c_float_p,
        c_int_p, c_int_p, c_int_p, c_int_p,
    ])
    fn("pf_n_accel", ctypes.c_int, [])
    fn("pf_n_steer", ctypes.c_int, [])
    fn("pf_coast_accel", ctypes.c_int, [])
    fn("pf_coast_steer", ctypes.c_int, [])
    fn("pf_invalid", ctypes.c_float, [])
    fn("pf_accel_bin", ctypes.c_int, [ctypes.c_float])
    fn("pf_steer_bin", ctypes.c_int, [ctypes.c_float])
    fn("pf_accel_value", ctypes.c_float, [ctypes.c_int])
    fn("pf_steer_value", ctypes.c_float, [ctypes.c_int])
    fn("pf_quantized_ade", None, [
        handle, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_int,
        c_float_p, c_int_p, c_float_p, c_float_p,
    ])
    fn("pf_delta_ade", None, [
        handle, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_float_p, c_int_p, c_float_p, c_float_p, c_float_p,
    ])
    fn("pf_quantized_trace", None, [
        handle, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_int,
        c_float_p, c_float_p, c_uint8_p,
    ])
    fn("pf_delta_trace", None, [
        handle, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_float_p, c_float_p, c_uint8_p,
    ])
    fn("pf_road_counts", None, [handle, c_int_p, c_int_p])
    fn("pf_roads", None, [handle, c_int_p, c_int_p, c_float_p, c_float_p])
    return lib


def create_sim(lib, data_dir, map_id, num_agents, episode_length=500,
        goal_min=5.0, goal_max=30.0, physics="through"):
    h = lib.pf_create_sim(
        str(data_dir).encode(), int(map_id), int(num_agents), int(episode_length),
        float(goal_min), float(goal_max), physics.encode(),
    )
    if not h:
        raise RuntimeError("pf_create_sim failed")
    return h


def create_scenario(lib, data_dir, map_id, json_path, physics="through"):
    h = lib.pf_create_scenario(
        str(data_dir).encode(), int(map_id), str(json_path).encode(), physics.encode(),
    )
    if not h:
        raise RuntimeError("pf_create_scenario failed")
    return h


def observations(lib, h):
    n = lib.pf_n(h)
    obs_n = lib.pf_obs_size(h)
    ptr = lib.pf_obs(h)
    return np.ctypeslib.as_array(ptr, shape=(n, obs_n)).copy()


def agent_state(lib, h, i):
    out = (ctypes.c_float * 11)()
    lib.pf_agent(h, i, out)
    return np.array(out, dtype=np.float32)


def states(lib, h):
    n = lib.pf_n(h)
    return np.stack([agent_state(lib, h, i) for i in range(n)])


def set_actions(lib, h, actions):
    for i, (a, s) in enumerate(actions):
        lib.pf_set_action(h, i, float(a), float(s))


def mean_xy(lib, h):
    x = ctypes.c_float()
    y = ctypes.c_float()
    lib.pf_mean(h, ctypes.byref(x), ctypes.byref(y))
    return float(x.value), float(y.value)


def log_xy(lib, h, i, t):
    x = ctypes.c_float()
    y = ctypes.c_float()
    lib.pf_log_xy(h, i, t, ctypes.byref(x), ctypes.byref(y))
    return float(x.value), float(y.value)


def log_bins(lib, h, i, t):
    a = ctypes.c_int()
    s = ctypes.c_int()
    lib.pf_log_bins(h, i, t, ctypes.byref(a), ctypes.byref(s))
    return int(a.value), int(s.value)


def outcome(lib, h, i):
    ret = ctypes.c_float()
    ade = ctypes.c_float()
    dist = ctypes.c_float()
    collided = ctypes.c_int()
    offroad = ctypes.c_int()
    reached = ctypes.c_int()
    before = ctypes.c_int()
    lib.pf_outcome(
        h, i, ctypes.byref(ret), ctypes.byref(ade), ctypes.byref(dist),
        ctypes.byref(collided), ctypes.byref(offroad),
        ctypes.byref(reached), ctypes.byref(before),
    )
    return {
        "return": float(ret.value),
        "ade": float(ade.value),
        "goal_dist": float(dist.value),
        "collided": int(collided.value),
        "offroad": int(offroad.value),
        "reached": int(reached.value),
        "collided_before": int(before.value),
    }


def collect(lib, h, teleport):
    n = lib.pf_log_n(h)
    obs_n = lib.pf_obs_size(h)
    traj = lib.pf_traj()
    obs = np.zeros((traj, n, obs_n), dtype=np.float32)
    acts = np.zeros((traj, n, 2), dtype=np.int32)
    valid = np.zeros((traj, n), dtype=np.uint8)
    ade = np.zeros(n, dtype=np.float32)
    lib.pf_collect(
        h, int(teleport),
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ade.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return obs, acts, valid, ade


def roads(lib, h):
    nroads = ctypes.c_int()
    npts = ctypes.c_int()
    lib.pf_road_counts(h, ctypes.byref(nroads), ctypes.byref(npts))
    offsets = np.zeros(nroads.value + 1, dtype=np.int32)
    types = np.zeros(nroads.value, dtype=np.int32)
    xs = np.zeros(npts.value, dtype=np.float32)
    ys = np.zeros(npts.value, dtype=np.float32)
    lib.pf_roads(
        h,
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    out = []
    for i in range(nroads.value):
        a = int(offsets[i])
        b = int(offsets[i + 1])
        out.append((int(types[i]), xs[a:b].copy(), ys[a:b].copy()))
    return out
