#!/usr/bin/env python3
"""I24 highway realism, adapted from eval/wosac.

Same window as that scorer: logged frame 10, 81 states (that frame plus
80 futures), 32 rollouts, and the 2024 histograms in dreal.ini. Seven
metrics remain: linear speed, linear acceleration, angular speed, angular
acceleration, distance to the nearest object, time to collision, and
signed distance to the road edge.

Collision indication and off-road indication are not scored. Each agent's
likelihood is divided by the mean ground-truth agent likelihood of that
metric. The ground-truth distribution is the logged window copied 32
times. Every reported score is the mean and the sample standard deviation
across agents, with the total agent count.

A deterministic controller is one closed-loop trajectory copied 32
times, which is how a delta policy fills the WOSAC histogram. Features
are computed on that single trajectory and repeated along the rollout
axis. Repeating identical rollouts does not change a per-rollout
feature, and the histogram still sees 32 * 81 samples.

Every report scores three controllers. Ground truth is the logged
window. The discretized expert replays stored bins and coasts when a
bin is missing. The anchor is the greedy bc_i24.npz policy. A run that
is missing one of them does not write a score. Unfit agents are parked.
Integrate does not freeze a car that enters its goal disk, so the eight
seconds stay on the bicycle model.

An optional sampled policy is a drive checkpoint. It is scored with
those three and written to its own file. Each of its 32 rollouts is a
closed loop from the player's sampled bins, with the MinGRU carry kept
across the window and cleared at the start of the rollout.
"""

import argparse
import configparser
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "wosac"))
sys.path.insert(0, str(ROOT / "tests"))

import drive_bind as drv
import estimators
import metrics
from train_bc_map import load_policy

DATA = ROOT / "data" / "i24"
JSON_DIR = DATA / "json"
CKPT = ROOT / "checkpoints" / "bc_i24.npz"
OUT_PATH = ROOT / "analysis" / "offline" / "i24" / "dreal_highway.json"
RL_OUT = ROOT / "analysis" / "offline" / "i24" / "dreal_rl.json"
INI = Path(__file__).with_name("dreal.ini")
POLICY_ACT = Path("/tmp/i24_policy_act")

INIT_STEPS = 10
SIM_STEPS = 81
ROLLOUTS = 32
VEHICLE = 1
TRUCK = 4
ROAD_EDGE = 7
INVALID = np.float32(-10000.0)

FIELDS = (
    "linear_speed",
    "linear_acceleration",
    "angular_speed",
    "angular_acceleration",
    "distance_to_nearest_object",
    "time_to_collision",
    "distance_to_road_edge",
)
# Every report scores these three.
REQUIRED = ("ground_truth", "discretized_expert", "anchor")


def load_config():
    cfg = configparser.ConfigParser()
    cfg.read(INI)
    return cfg


def hist_params(cfg, name):
    return (
        cfg.getfloat(name, "histogram.min_val"),
        cfg.getfloat(name, "histogram.max_val"),
        cfg.getint(name, "histogram.num_bins"),
        cfg.getfloat(name, "histogram.additive_smoothing_pseudocount"),
        cfg.getboolean(name, "independent_timesteps"),
    )


def vehicle_objects(doc):
    return [
        obj for obj in doc["objects"]
        if obj.get("type") in ("vehicle", "truck")
    ]


def road_edges(lib, h, scenario):
    xs = []
    ys = []
    lengths = []
    for typ, x, y in drv.roads(lib, h):
        if typ != ROAD_EDGE:
            continue
        good = (x != INVALID) & (y != INVALID)
        start = 0
        n = len(x)
        while start < n:
            while start < n and not good[start]:
                start += 1
            end = start
            while end < n and good[end]:
                end += 1
            if end - start >= 2:
                xs.append(np.asarray(x[start:end], dtype=np.float32))
                ys.append(np.asarray(y[start:end], dtype=np.float32))
                lengths.append(end - start)
            start = end
    if not lengths:
        raise RuntimeError("scene has no road-edge polyline")
    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "scenario_id": np.full(len(lengths), scenario),
    }


def _f32_series(values):
    return np.asarray(values, dtype=np.float64).astype(np.float32)


def log_window(obj, mean_x, mean_y, sl):
    pos = obj["position"]
    x = _f32_series([p["x"] for p in pos])[sl] - np.float32(mean_x)
    y = _f32_series([p["y"] for p in pos])[sl] - np.float32(mean_y)
    heading = _f32_series(obj["heading"])[sl]
    valid = np.asarray(obj["valid"], dtype=np.int32)[sl].astype(bool)
    return x, y, heading, valid


def check_log(lib, h, objs, mean_x, mean_y):
    if len(objs) != lib.pf_log_n(h):
        raise RuntimeError(
            f"json vehicles {len(objs)} != simulator log {lib.pf_log_n(h)}"
        )
    traj = lib.pf_log_traj(h)
    probes = [0, INIT_STEPS, traj // 2, traj - 1]
    for i, obj in enumerate(objs):
        for t in probes:
            if t < 0 or t >= traj or t >= len(obj["position"]):
                continue
            got = drv.log_xy(lib, h, i, t)
            want_x = float(_f32_series([obj["position"][t]["x"]])[0] - np.float32(mean_x))
            want_y = float(_f32_series([obj["position"][t]["y"]])[0] - np.float32(mean_y))
            if abs(got[0] - want_x) > 1e-3 or abs(got[1] - want_y) > 1e-3:
                raise RuntimeError(f"log xy mismatch agent {i} t {t}")
            flag = 1 if obj["valid"][t] else 0
            if int(lib.pf_log_valid(h, i, t)) != flag:
                raise RuntimeError(f"log valid mismatch agent {i} t {t}")


def controlled_agents(lib, h):
    agents = []
    for i in range(lib.pf_log_n(h)):
        if lib.pf_base_unfit(h, i):
            continue
        if not lib.pf_log_valid(h, i, INIT_STEPS):
            continue
        agents.append(i)
    if not agents:
        raise RuntimeError("no controllable agent at the init frame")
    return agents


def park_unfit(lib, h):
    invalid = float(lib.pf_invalid())
    for i in range(lib.pf_log_n(h)):
        if lib.pf_base_unfit(h, i):
            lib.pf_set_agent(h, i, invalid, invalid, 0, 0, 0, 1, 1, 0, 0)


def scene_pack(lib, h, doc):
    """Ground-truth window, bodies, and road edges for one scenario."""
    objs = vehicle_objects(doc)
    mean_x, mean_y = drv.mean_xy(lib, h)
    check_log(lib, h, objs, mean_x, mean_y)
    traj = lib.pf_log_traj(h)
    if traj < INIT_STEPS + SIM_STEPS:
        raise RuntimeError(f"log length {traj} is shorter than the eval window")
    agents = controlled_agents(lib, h)
    sl = slice(INIT_STEPS, INIT_STEPS + SIM_STEPS)
    xs, ys, hs, vs = [], [], [], []
    types = []
    for i in agents:
        x, y, heading, valid = log_window(objs[i], mean_x, mean_y, sl)
        if not valid[0]:
            raise RuntimeError(f"agent {i} is invalid at frame {INIT_STEPS}")
        xs.append(x)
        ys.append(y)
        hs.append(heading)
        vs.append(valid)
        types.append(TRUCK if objs[i]["type"] == "truck" else VEHICLE)
    lib.pf_reset(h, 0)
    lib.pf_place(h, INIT_STEPS)
    park_unfit(lib, h)
    length = []
    width = []
    for i in agents:
        state = drv.agent_state(lib, h, i)
        if float(state[10]) < 0.5:
            raise RuntimeError(f"agent {i} was not placed at frame {INIT_STEPS}")
        if abs(float(state[0]) - float(xs[len(length)][0])) > 1e-3:
            raise RuntimeError(f"placed pose disagrees with the log for agent {i}")
        width.append(float(state[7]))
        length.append(float(state[8]))
    scenario = str(doc["scenario_id"])
    gt = {
        "ids": np.asarray(agents, dtype=np.int32),
        "x": np.stack(xs).astype(np.float32),
        "y": np.stack(ys).astype(np.float32),
        "heading": np.stack(hs).astype(np.float32),
        "valid": np.stack(vs),
    }
    sim_meta = {
        "ids": gt["ids"],
        "types": np.asarray(types, dtype=np.int32),
        "length": np.asarray(length, dtype=np.float32),
        "width": np.asarray(width, dtype=np.float32),
    }
    return gt, sim_meta, road_edges(lib, h, scenario), scenario


def _blank(n):
    return {
        "x": np.zeros((n, SIM_STEPS), dtype=np.float32),
        "y": np.zeros((n, SIM_STEPS), dtype=np.float32),
        "heading": np.zeros((n, SIM_STEPS), dtype=np.float32),
        "collision": np.zeros((n, SIM_STEPS), dtype=bool),
        "offroad": np.zeros((n, SIM_STEPS), dtype=bool),
    }


def _contact(lib, h, agent):
    """Vehicle-box overlap and road-edge intersection, from the simulator."""
    collision = ctypes.c_int()
    offroad = ctypes.c_int()
    lib.pf_contact(h, agent, ctypes.byref(collision), ctypes.byref(offroad))
    return int(collision.value) == 1, int(offroad.value) != 0


def _record(lib, h, agents, buf, t):
    for row, agent in enumerate(agents):
        state = drv.agent_state(lib, h, agent)
        if float(state[10]) < 0.5:
            raise RuntimeError(f"agent {agent} left the map at recorded step {t}")
        buf["x"][row, t] = state[0]
        buf["y"][row, t] = state[1]
        buf["heading"][row, t] = state[2]
        hit, outside = _contact(lib, h, agent)
        buf["collision"][row, t] = hit
        buf["offroad"][row, t] = outside


def rollout(lib, h, agents, action_at):
    """Closed loop from frame 10. `action_at(frame)` sets every agent."""
    lib.pf_reset(h, 0)
    lib.pf_place(h, INIT_STEPS)
    park_unfit(lib, h)
    buf = _blank(len(agents))
    lib.pf_check_contacts(h)
    _record(lib, h, agents, buf, 0)
    for k in range(SIM_STEPS - 1):
        frame = INIT_STEPS + k
        lib.pf_set_timestep(h, frame)
        lib.pf_recompute(h)
        action_at(frame)
        lib.pf_integrate(h, 0)
        _record(lib, h, agents, buf, k + 1)
    return buf


def expert_actions(lib, h, agents):
    coast_a = int(lib.pf_coast_accel())
    coast_s = int(lib.pf_coast_steer())

    def action_at(frame):
        for agent in agents:
            accel, steer = drv.log_bins(lib, h, agent, frame)
            if accel < 0:
                accel = coast_a
            if steer < 0:
                steer = coast_s
            lib.pf_set_action(h, agent, float(accel), float(steer))

    return action_at


def anchor_actions(lib, h, agents, pol):
    index = np.asarray(agents, dtype=np.int32)

    def action_at(frame):
        del frame
        obs = drv.observations(lib, h)
        batch = (obs[index] - pol.mu) / pol.sig
        accel, steer, _entropy = pol.act_batch(batch)
        for agent, a_bin, s_bin in zip(agents, accel, steer):
            lib.pf_set_action(h, agent, float(a_bin), float(s_bin))

    return action_at


class PolicySession:
    """Sampled bins from a drive checkpoint. One process, one MinGRU carry."""

    def __init__(self, ckpt, batch, obs_n, hidden=64, layers=4):
        if not POLICY_ACT.is_file():
            raise RuntimeError(f"missing {POLICY_ACT}")
        built = int(subprocess.check_output([str(POLICY_ACT), "--obs"], text=True))
        if built != obs_n:
            raise RuntimeError(f"policy obs {built} != simulator obs {obs_n}")
        self.batch = batch
        self.proc = subprocess.Popen(
            [str(POLICY_ACT), str(ckpt), str(hidden), str(layers), str(batch)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def act(self, obs, terminals):
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        terminals = np.ascontiguousarray(terminals, dtype=np.float32)
        if obs.shape != (self.batch, obs.shape[-1]) or terminals.shape != (self.batch,):
            raise RuntimeError("policy batch does not match the session")
        blob = b"".join((
            np.int32(1).tobytes(),
            np.int32(self.batch).tobytes(),
            obs.tobytes(),
            terminals.tobytes(),
        ))
        self.proc.stdin.write(blob)
        self.proc.stdin.flush()
        raw = self.proc.stdout.read(self.batch * 2 * 4)
        if len(raw) != self.batch * 2 * 4:
            err = self.proc.stderr.read() if self.proc.stderr else b""
            raise RuntimeError(f"policy server closed ({err!r})")
        return np.frombuffer(raw, dtype=np.float32).reshape(self.batch, 2).copy()

    def close(self):
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(np.int32(0).tobytes())
            self.proc.stdin.flush()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=10)


def rollout_sampled(lib, h, agents, session):
    """32 closed loops. The first step of each loop clears the MinGRU carry."""
    index = np.asarray(agents, dtype=np.int32)
    n = len(agents)
    xs = np.zeros((n, ROLLOUTS, SIM_STEPS), dtype=np.float32)
    ys = np.zeros((n, ROLLOUTS, SIM_STEPS), dtype=np.float32)
    hs = np.zeros((n, ROLLOUTS, SIM_STEPS), dtype=np.float32)
    cs = np.zeros((n, ROLLOUTS, SIM_STEPS), dtype=bool)
    os_ = np.zeros((n, ROLLOUTS, SIM_STEPS), dtype=bool)
    for rollout_i in range(ROLLOUTS):
        lib.pf_reset(h, 0)
        lib.pf_place(h, INIT_STEPS)
        park_unfit(lib, h)
        buf = _blank(n)
        lib.pf_check_contacts(h)
        _record(lib, h, agents, buf, 0)
        for k in range(SIM_STEPS - 1):
            frame = INIT_STEPS + k
            lib.pf_set_timestep(h, frame)
            lib.pf_recompute(h)
            obs = drv.observations(lib, h)[index]
            terms = np.zeros(n, dtype=np.float32)
            if k == 0:
                terms[:] = 1.0
            actions = session.act(obs, terms)
            for agent, (accel, steer) in zip(agents, actions):
                lib.pf_set_action(h, agent, float(accel), float(steer))
            lib.pf_integrate(h, 0)
            _record(lib, h, agents, buf, k + 1)
        xs[:, rollout_i, :] = buf["x"]
        ys[:, rollout_i, :] = buf["y"]
        hs[:, rollout_i, :] = buf["heading"]
        cs[:, rollout_i, :] = buf["collision"]
        os_[:, rollout_i, :] = buf["offroad"]
    return {
        "x": xs, "y": ys, "heading": hs,
        "collision": cs, "offroad": os_,
    }


def _repeat(values):
    return np.repeat(values, ROLLOUTS, axis=1)


def scene_likelihoods(gt, sim, edges, scenario, cfg, device, control_valid=None):
    """Per-metric scene likelihood. Values are exp(mean log p), unrounded."""
    n = len(gt["ids"])
    if not np.array_equal(sim["ids"], gt["ids"]):
        raise RuntimeError("simulated agents do not match the ground truth")
    if gt["x"].shape != (n, SIM_STEPS) or sim["x"].shape != (n, SIM_STEPS):
        raise RuntimeError("trajectory window is not 81 states")
    ref_x = gt["x"][:, None, :]
    ref_y = gt["y"][:, None, :]
    ref_h = gt["heading"][:, None, :]
    ref_v = gt["valid"][:, None, :]
    sim_x = sim["x"][:, None, :]
    sim_y = sim["y"][:, None, :]
    sim_h = sim["heading"][:, None, :]
    scenario_ids = np.full((n, 1), scenario)
    length = sim["length"]
    width = sim["width"]
    is_vehicle = (sim["types"] == VEHICLE) | (sim["types"] == TRUCK)
    if not np.all(is_vehicle):
        raise RuntimeError("eval set contains a non-vehicle")

    ref_k = metrics.compute_kinematic_features(ref_x, ref_y, ref_h)
    sim_k = metrics.compute_kinematic_features(sim_x, sim_y, sim_h)
    speed_ok, accel_ok = metrics.compute_kinematic_validity(ref_v)
    sim_d, _sim_c, sim_ttc = metrics.compute_interaction_features(
        sim_x, sim_y, sim_h, scenario_ids, length, width,
        np.ones(n, dtype=bool), device=device,
    )
    ref_d, _ref_c, ref_ttc = metrics.compute_interaction_features(
        ref_x, ref_y, ref_h, scenario_ids, length, width,
        np.ones(n, dtype=bool), device=device, valid=ref_v,
    )
    sim_edge, _sim_off = metrics.compute_map_features(
        sim_x, sim_y, sim_h, scenario_ids, length, width, edges, device=device,
    )
    ref_edge, _ref_off = metrics.compute_map_features(
        ref_x, ref_y, ref_h, scenario_ids, length, width, edges,
        device=device, valid=ref_v,
    )
    ade, min_ade = metrics.compute_displacement_error(
        _repeat(sim_x), _repeat(sim_y), ref_x, ref_y, ref_v,
    )

    logged = {
        "linear_speed": (ref_k[0], _repeat(sim_k[0]), speed_ok[:, 0, :]),
        "linear_acceleration": (ref_k[1], _repeat(sim_k[1]), accel_ok[:, 0, :]),
        "angular_speed": (ref_k[2], _repeat(sim_k[2]), speed_ok[:, 0, :]),
        "angular_acceleration": (ref_k[3], _repeat(sim_k[3]), accel_ok[:, 0, :]),
        "distance_to_nearest_object": (ref_d, _repeat(sim_d), ref_v[:, 0, :]),
        "time_to_collision": (
            ref_ttc, _repeat(sim_ttc), (ref_v & is_vehicle[:, None, None])[:, 0, :],
        ),
        "distance_to_road_edge": (ref_edge, _repeat(sim_edge), ref_v[:, 0, :]),
    }
    row = {"scenario_id": scenario, "num_agents": int(n)}
    for name, (log_values, sim_values, valid) in logged.items():
        lo, hi, bins, smooth, indep = hist_params(cfg, name)
        raw = estimators.log_likelihood_estimate_timeseries(
            log_values=log_values,
            sim_values=sim_values,
            treat_timesteps_independently=indep,
            min_val=lo,
            max_val=hi,
            num_bins=bins,
            additive_smoothing=smooth,
            sanity_check=False,
        )
        reduced = metrics._reduce_average_with_validity(raw, valid, axis=1)
        agent_like = np.exp(np.asarray(reduced, dtype=np.float64))
        if not np.any(np.isfinite(agent_like)):
            raise RuntimeError(f"{scenario} {name} likelihood is not finite")
        row["agent_" + name] = agent_like
        row["likelihood_" + name] = float(np.nanmean(agent_like))
    row["agent_ade"] = np.asarray(ade, dtype=np.float64)
    row["ade"] = float(np.nanmean(ade))
    row["min_ade"] = float(np.nanmean(min_ade))
    row.update(_control_from_sim(sim, control_valid))
    return row


def scene_likelihoods_sampled(gt, sim, edges, scenario, cfg, device):
    """Same seven metrics. sim trajectories are (agents, 32, 81)."""
    n = len(gt["ids"])
    if not np.array_equal(sim["ids"], gt["ids"]):
        raise RuntimeError("simulated agents do not match the ground truth")
    if gt["x"].shape != (n, SIM_STEPS) or sim["x"].shape != (n, ROLLOUTS, SIM_STEPS):
        raise RuntimeError("sampled window is not 32 rollouts of 81 states")
    ref_x = gt["x"][:, None, :]
    ref_y = gt["y"][:, None, :]
    ref_h = gt["heading"][:, None, :]
    ref_v = gt["valid"][:, None, :]
    sim_x = sim["x"]
    sim_y = sim["y"]
    sim_h = sim["heading"]
    scenario_ids = np.full((n, 1), scenario)
    length = sim["length"]
    width = sim["width"]
    is_vehicle = (sim["types"] == VEHICLE) | (sim["types"] == TRUCK)
    if not np.all(is_vehicle):
        raise RuntimeError("eval set contains a non-vehicle")

    ref_k = metrics.compute_kinematic_features(ref_x, ref_y, ref_h)
    sim_k = metrics.compute_kinematic_features(sim_x, sim_y, sim_h)
    speed_ok, accel_ok = metrics.compute_kinematic_validity(ref_v)
    sim_d, _sim_c, sim_ttc = metrics.compute_interaction_features(
        sim_x, sim_y, sim_h, scenario_ids, length, width,
        np.ones(n, dtype=bool), device=device,
    )
    ref_d, _ref_c, ref_ttc = metrics.compute_interaction_features(
        ref_x, ref_y, ref_h, scenario_ids, length, width,
        np.ones(n, dtype=bool), device=device, valid=ref_v,
    )
    sim_edge, _sim_off = metrics.compute_map_features(
        sim_x, sim_y, sim_h, scenario_ids, length, width, edges, device=device,
    )
    ref_edge, _ref_off = metrics.compute_map_features(
        ref_x, ref_y, ref_h, scenario_ids, length, width, edges,
        device=device, valid=ref_v,
    )
    ade, min_ade = metrics.compute_displacement_error(
        sim_x, sim_y, ref_x, ref_y, ref_v,
    )
    logged = {
        "linear_speed": (ref_k[0], sim_k[0], speed_ok[:, 0, :]),
        "linear_acceleration": (ref_k[1], sim_k[1], accel_ok[:, 0, :]),
        "angular_speed": (ref_k[2], sim_k[2], speed_ok[:, 0, :]),
        "angular_acceleration": (ref_k[3], sim_k[3], accel_ok[:, 0, :]),
        "distance_to_nearest_object": (ref_d, sim_d, ref_v[:, 0, :]),
        "time_to_collision": (
            ref_ttc, sim_ttc, (ref_v & is_vehicle[:, None, None])[:, 0, :],
        ),
        "distance_to_road_edge": (ref_edge, sim_edge, ref_v[:, 0, :]),
    }
    row = {"scenario_id": scenario, "num_agents": int(n)}
    for name, (log_values, sim_values, valid) in logged.items():
        if sim_values.shape[1] != ROLLOUTS:
            raise RuntimeError(f"{name} did not keep {ROLLOUTS} rollouts")
        lo, hi, bins, smooth, indep = hist_params(cfg, name)
        raw = estimators.log_likelihood_estimate_timeseries(
            log_values=log_values,
            sim_values=sim_values,
            treat_timesteps_independently=indep,
            min_val=lo,
            max_val=hi,
            num_bins=bins,
            additive_smoothing=smooth,
            sanity_check=False,
        )
        reduced = metrics._reduce_average_with_validity(raw, valid, axis=1)
        agent_like = np.exp(np.asarray(reduced, dtype=np.float64))
        if not np.any(np.isfinite(agent_like)):
            raise RuntimeError(f"{scenario} {name} likelihood is not finite")
        row["agent_" + name] = agent_like
        row["likelihood_" + name] = float(np.nanmean(agent_like))
    row["agent_ade"] = np.asarray(ade, dtype=np.float64)
    row["ade"] = float(np.nanmean(ade))
    row["min_ade"] = float(np.nanmean(min_ade))
    row.update(_control_from_sim(sim, None))
    return row


def _control_from_sim(sim, step_valid):
    if "collision" not in sim or "offroad" not in sim:
        raise RuntimeError("controllability was not recorded")
    collision = np.asarray(sim["collision"], dtype=bool)
    offroad = np.asarray(sim["offroad"], dtype=bool)
    if collision.ndim == 2:
        collision = collision[:, None, :]
        offroad = offroad[:, None, :]
    return _control_row(collision, offroad, step_valid)


def stamp_log_contacts(lib, h, agents, traj, valid):
    """Road-edge and vehicle contacts of the logged window."""
    n = len(agents)
    steps = int(traj["x"].shape[1])
    collision = np.zeros((n, steps), dtype=bool)
    offroad = np.zeros((n, steps), dtype=bool)
    invalid = float(lib.pf_invalid())
    lib.pf_reset(h, 0)
    lib.pf_place(h, INIT_STEPS)
    park_unfit(lib, h)
    for t in range(steps):
        for row, agent in enumerate(agents):
            if not valid[row, t]:
                lib.pf_set_pose(h, agent, invalid, invalid, 0.0)
            else:
                lib.pf_set_pose(
                    h, agent,
                    float(traj["x"][row, t]),
                    float(traj["y"][row, t]),
                    float(traj["heading"][row, t]),
                )
        lib.pf_check_contacts(h)
        for row, agent in enumerate(agents):
            if not valid[row, t]:
                continue
            hit, outside = _contact(lib, h, agent)
            collision[row, t] = hit
            offroad[row, t] = outside
    traj["collision"] = collision
    traj["offroad"] = offroad


def _rising_events(flags):
    """New contacts. A run of overlapping steps counts once."""
    rising = np.empty(flags.shape, dtype=bool)
    rising[:, :, 0] = flags[:, :, 0]
    if flags.shape[2] > 1:
        rising[:, :, 1:] = flags[:, :, 1:] & ~flags[:, :, :-1]
    return rising.sum(axis=2)


def _control_row(collision, offroad, step_valid):
    """Per agent, summed over rollouts. Not part of meta."""
    collided = np.asarray(collision, dtype=bool)
    outside = np.asarray(offroad, dtype=bool)
    if collided.shape != outside.shape or collided.ndim != 3:
        raise RuntimeError("collision and off-road indicators do not match")
    if step_valid is not None:
        valid = np.asarray(step_valid, dtype=bool)
        if valid.shape != (collided.shape[0], collided.shape[2]):
            raise RuntimeError("control validity does not match the window")
        valid = valid[:, None, :]
        collided = collided & valid
        outside = outside & valid
    return {
        "agent_ctrl_collided_rollouts": collided.any(axis=2).sum(axis=1).astype(np.int32),
        "agent_ctrl_collision_events": _rising_events(collided).sum(axis=1).astype(np.int32),
        "agent_ctrl_offroad_rollouts": outside.any(axis=2).sum(axis=1).astype(np.int32),
        "agent_ctrl_offroad_events": _rising_events(outside).sum(axis=1).astype(np.int32),
        "ctrl_rollouts": int(collided.shape[1]),
    }


def require_methods(by_method):
    """Ground truth, discretized expert, and the anchor, on the same scenes."""
    names = tuple(by_method)
    if names[:len(REQUIRED)] != REQUIRED:
        raise RuntimeError(
            "eval scores ground_truth, discretized_expert, and anchor together"
        )
    scenes = None
    for name in names:
        rows = by_method[name]
        if not rows:
            raise RuntimeError(f"eval is missing {name}")
        ids = [row.get("scenario_id") for row in rows]
        if scenes is None:
            scenes = ids
        elif ids != scenes:
            raise RuntimeError(f"{name} does not cover the same scenes")


def reference_trajectories(lib, h, gt, agents, pol):
    """Logged window, stored bins, and the greedy anchor. All three, every run."""
    return {
        "ground_truth": {
            "x": gt["x"], "y": gt["y"], "heading": gt["heading"],
        },
        "discretized_expert": rollout(
            lib, h, agents, expert_actions(lib, h, agents),
        ),
        "anchor": rollout(
            lib, h, agents, anchor_actions(lib, h, agents, pol),
        ),
    }


def _mean_std(values):
    """Sample mean and standard deviation. NaNs are left out of both."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise RuntimeError("metric has no finite agents")
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {"mean": mean, "std": std, "n": int(v.size)}


def _scene_public(row):
    skip = {
        "agent_ade",
        "agent_ctrl_collided_rollouts",
        "agent_ctrl_collision_events",
        "agent_ctrl_offroad_rollouts",
        "agent_ctrl_offroad_events",
        *(f"agent_{name}" for name in FIELDS),
    }
    out = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.floating):
            out[key] = float(value)
        elif isinstance(value, np.integer):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _score_masks(gt_rows):
    """Agents with a finite ground-truth likelihood on all seven metrics."""
    masks = []
    for row in gt_rows:
        mask = np.ones(row["num_agents"], dtype=bool)
        for name in FIELDS:
            values = np.asarray(row["agent_" + name], dtype=np.float64)
            if len(values) != row["num_agents"]:
                raise RuntimeError(f"{name} dropped an agent")
            mask &= np.isfinite(values)
        if not np.any(mask):
            raise RuntimeError(f"{row['scenario_id']} has no fully scored agent")
        masks.append(mask)
    return masks


def _pool_masked(rows, key, masks):
    parts = []
    for row, mask in zip(rows, masks):
        values = np.asarray(row[key], dtype=np.float64)
        chosen = values[mask]
        if chosen.size != int(mask.sum()) or not np.isfinite(chosen).all():
            raise RuntimeError(f"{key} is missing on a scored agent")
        parts.append(chosen)
    return np.concatenate(parts)


def _control_summary(rows, masks):
    """Agents with a contact, and contacts per agent-episode."""
    keys = (
        "agent_ctrl_collided_rollouts",
        "agent_ctrl_collision_events",
        "agent_ctrl_offroad_rollouts",
        "agent_ctrl_offroad_events",
    )
    totals = {key: 0 for key in keys}
    slots = 0
    for row, mask in zip(rows, masks):
        chosen = int(np.asarray(mask).sum())
        rollouts = int(row["ctrl_rollouts"])
        if rollouts < 1:
            raise RuntimeError("control rollouts are missing")
        slots += chosen * rollouts
        for key in keys:
            values = np.asarray(row[key])
            if values.shape != (row["num_agents"],):
                raise RuntimeError(f"{key} does not cover the scene")
            totals[key] += int(values[mask].sum())

    def pack(count):
        if slots < 1:
            raise RuntimeError("controllability has no agents")
        return {
            "count": int(count),
            "slots": int(slots),
            "percent": 100.0 * float(count) / float(slots),
        }

    return {
        "agents_collided": pack(totals["agent_ctrl_collided_rollouts"]),
        "collisions_per_agent": pack(totals["agent_ctrl_collision_events"]),
        "agents_offroad": pack(totals["agent_ctrl_offroad_rollouts"]),
        "offroad_per_agent": pack(totals["agent_ctrl_offroad_events"]),
    }


def aggregate(by_method):
    """Divide each agent by the ground-truth mean, then report mean and std."""
    require_methods(by_method)
    gt_rows = by_method["ground_truth"]
    masks = _score_masks(gt_rows)
    placed = int(sum(row["num_agents"] for row in gt_rows))
    n_agents = int(sum(int(mask.sum()) for mask in masks))
    for method in by_method:
        rows = by_method[method]
        if int(sum(row["num_agents"] for row in rows)) != placed:
            raise RuntimeError(f"{method} does not score the same agents")
    ceiling = {}
    for name in FIELDS:
        stat = _mean_std(_pool_masked(gt_rows, "agent_" + name, masks))
        if stat["mean"] <= 0.0 or stat["n"] != n_agents:
            raise RuntimeError(f"ground-truth ceiling for {name} is unusable")
        ceiling[name] = stat["mean"]
    methods = {}
    for method in by_method:
        rows = by_method[method]
        raw = {}
        normalized = {}
        per_agent = []
        for name in FIELDS:
            values = _pool_masked(rows, "agent_" + name, masks)
            raw[name] = _mean_std(values)
            scaled = values / ceiling[name]
            normalized[name] = _mean_std(scaled)
            per_agent.append(scaled)
        stacked = np.stack(per_agent, axis=0)
        meta_score = _mean_std(stacked.mean(axis=0))
        ade = _mean_std(_pool_masked(rows, "agent_ade", masks))
        for stat in (*normalized.values(), meta_score, ade):
            if stat["n"] != n_agents:
                raise RuntimeError(f"{method} does not use all {n_agents} agents")
        control = _control_summary(rows, masks)
        rollouts = int(rows[0]["ctrl_rollouts"])
        if any(int(row["ctrl_rollouts"]) != rollouts for row in rows):
            raise RuntimeError(f"{method} mixes rollout counts")
        if control["agents_collided"]["slots"] != n_agents * rollouts:
            raise RuntimeError(f"{method} controllability does not cover the scored agents")
        methods[method] = {
            "scenes": len(rows),
            "agents": n_agents,
            "raw": raw,
            "normalized": normalized,
            "meta": meta_score,
            "ade": ade,
            "controllability": control,
            "per_scene": [_scene_public(row) for row in rows],
        }
    return {
        "agents": n_agents,
        "placed": placed,
        "ceiling": ceiling,
        "methods": methods,
    }


def _fmt_pm(stat):
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def _fmt_pct(stat):
    return f"{stat['percent']:6.2f}% ({stat['count']}/{stat['slots']})"


def print_report(summary, accuracy):
    n_agents = summary["agents"]
    print(f"agents {n_agents}", flush=True)
    print(f"anchor joint accuracy {accuracy:.4f}", flush=True)
    print("ground-truth ceiling (mean agent likelihood)", flush=True)
    for name in FIELDS:
        print(f"  {name:28s} {summary['ceiling'][name]:.6f}", flush=True)
    print(f"{'method':22s} {'metric':28s} {'mean ± std':>22s} {'n':>6s}", flush=True)
    for method in summary["methods"]:
        block = summary["methods"][method]
        for name in FIELDS:
            stat = block["normalized"][name]
            print(
                f"{method:22s} {name:28s} {_fmt_pm(stat):>22s} {stat['n']:6d}",
                flush=True,
            )
        print(
            f"{method:22s} {'meta':28s} {_fmt_pm(block['meta']):>22s} {block['meta']['n']:6d}",
            flush=True,
        )
        print(
            f"{method:22s} {'ade_m':28s} {_fmt_pm(block['ade']):>22s} {block['ade']['n']:6d}",
            flush=True,
        )
        if block["agents"] != n_agents:
            raise RuntimeError(f"{method} agent count is {block['agents']}, expected {n_agents}")
    print("Controllability", flush=True)
    print(
        f"{'method':22s} {'agents collided':>22s} {'collisions/agents':>22s} "
        f"{'agents off-road':>22s} {'off-road/agents':>22s}",
        flush=True,
    )
    for method, block in summary["methods"].items():
        control = block["controllability"]
        print(
            f"{method:22s} {_fmt_pct(control['agents_collided']):>22s} "
            f"{_fmt_pct(control['collisions_per_agent']):>22s} "
            f"{_fmt_pct(control['agents_offroad']):>22s} "
            f"{_fmt_pct(control['offroad_per_agent']):>22s}",
            flush=True,
        )
    placed = summary["placed"]
    print(f"total agents {n_agents} of {placed} placed at frame 10", flush=True)


def parse_policies(items):
    policies = []
    seen = set()
    for item in items:
        if "=" not in item:
            raise RuntimeError("--policy needs NAME=checkpoint.bin")
        name, path = item.split("=", 1)
        if not name or name in REQUIRED or name in seen:
            raise RuntimeError(f"policy name {name!r} is not usable")
        ckpt = Path(path)
        if not ckpt.is_file():
            raise RuntimeError(f"missing checkpoint {ckpt}")
        policies.append((name, ckpt))
        seen.add(name)
    return policies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--maps",
        default=None,
        help="comma-separated map ids (default: every scene, in order)",
    )
    args = parser.parse_args()
    policies = parse_policies(args.policy)
    paths = sorted(JSON_DIR.glob("*.json"))
    if len(paths) != 3:
        raise RuntimeError(f"expected 3 I24 scenes, found {len(paths)}")
    selected = list(enumerate(paths))
    if args.maps:
        want = [int(part) for part in args.maps.split(",") if part != ""]
        selected = [(map_id, path) for map_id, path in selected if map_id in want]
        if [map_id for map_id, _path in selected] != want:
            raise RuntimeError(f"--maps {args.maps} does not match the scene files")
    cfg = load_config()
    for name in FIELDS:
        hist_params(cfg, name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pol, accuracy = load_policy(CKPT)
    pol.eval()
    lib = drv.load()
    if int(lib.pf_n_accel()) != int(pol.head_a.out_features):
        raise RuntimeError("BC accel bins do not match the simulator")
    if int(lib.pf_n_steer()) != int(pol.head_s.out_features):
        raise RuntimeError("BC steer bins do not match the simulator")
    by_method = {method: [] for method in REQUIRED}
    for name, _ckpt in policies:
        by_method[name] = []
    for map_id, path in selected:
        doc = json.loads(path.read_text())
        h = drv.create_scenario(lib, DATA, map_id, path)
        try:
            gt, meta, edges, scenario = scene_pack(lib, h, doc)
            agents = [int(i) for i in gt["ids"]]
            trajectories = reference_trajectories(lib, h, gt, agents, pol)
            stamp_log_contacts(
                lib, h, agents, trajectories["ground_truth"], gt["valid"],
            )
            if tuple(trajectories) != REQUIRED:
                raise RuntimeError(
                    "eval scores ground_truth, discretized_expert, and anchor together"
                )
            for method in REQUIRED:
                sim = dict(meta)
                sim.update(trajectories[method])
                logged = gt["valid"] if method == "ground_truth" else None
                row = scene_likelihoods(
                    gt, sim, edges, scenario, cfg, device, control_valid=logged,
                )
                row["map_id"] = map_id
                by_method[method].append(row)
                print(
                    f"map {map_id} {method} agents {row['num_agents']} "
                    f"ade {row['ade']:.3f}",
                    flush=True,
                )
            obs_n = int(lib.pf_obs_size(h))
            for name, ckpt in policies:
                session = PolicySession(
                    ckpt, len(agents), obs_n, args.hidden, args.layers,
                )
                try:
                    traj = rollout_sampled(lib, h, agents, session)
                finally:
                    session.close()
                sim = dict(meta)
                sim.update(traj)
                row = scene_likelihoods_sampled(
                    gt, sim, edges, scenario, cfg, device,
                )
                row["map_id"] = map_id
                by_method[name].append(row)
                print(
                    f"map {map_id} {name} agents {row['num_agents']} "
                    f"ade {row['ade']:.3f}",
                    flush=True,
                )
        finally:
            lib.pf_close(h)
    summary = aggregate(by_method)
    summary["protocol"] = {
        "init_steps": INIT_STEPS,
        "sim_steps": SIM_STEPS,
        "rollouts": ROLLOUTS,
        "metrics": list(FIELDS),
        "meta": "per agent, the unweighted mean of the seven normalized likelihoods",
        "ceiling": "mean ground-truth agent likelihood of each metric",
        "spread": "sample standard deviation across agents",
        "agents": summary["agents"],
        "methods": list(by_method),
        "checkpoint": str(CKPT),
        "anchor_joint_accuracy": accuracy,
        "goal_freeze": False,
        "controllability": (
            "not in meta; agents with at least one event, and events per "
            "agent-episode; a continuous overlap counts once"
        ),
    }
    if policies:
        summary["protocol"]["policies"] = {
            name: str(ckpt) for name, ckpt in policies
        }
        summary["protocol"]["policy_actions"] = (
            "32 sampled closed loops, MinGRU cleared on the first step"
        )
        out = args.out or RL_OUT
        if out.resolve() == OUT_PATH.resolve():
            raise RuntimeError("refusing to overwrite the anchor comparison")
    else:
        out = args.out or OUT_PATH
        if out.resolve() == OUT_PATH.resolve() and args.maps:
            raise RuntimeError("refusing to overwrite the anchor comparison")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print_report(summary, accuracy)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
