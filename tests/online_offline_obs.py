"""Observation and action checks for the offline Waymo env."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_bind as drv

DATA = drv.ROOT / "data" / "womd"
MAP_ID = 0
JSONS = sorted((DATA / "json").glob("*.json"))


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def bicycle(x, y, heading, vx, vy, accel, steer, length, dt=0.1):
    # Same step as bicycle_integrate: new speed, then the circular arc.
    c = math.cos(heading)
    s = math.sin(heading)
    v = vx * c + vy * s + accel * dt
    v = max(-100.0, min(100.0, v))
    wheelbase = 0.6 * length
    if wheelbase < 1.0:
        wheelbase = 1.0
    tsteer = math.tan(steer)
    if tsteer > 1.0e6 or tsteer < -1.0e6:
        tsteer = 0.0
    dth = v * tsteer / wheelbase * dt
    if abs(dth) < 1.0e-4:
        dx_b = v * dt
        dy_b = 0.0
    else:
        radius = v * dt / dth
        dx_b = radius * math.sin(dth)
        dy_b = radius * (1.0 - math.cos(dth))
    x = x + dx_b * c - dy_b * s
    y = y + dx_b * s + dy_b * c
    heading = heading + dth
    if heading > math.pi:
        heading -= 2.0 * math.pi
    if heading < -math.pi:
        heading += 2.0 * math.pi
    return x, y, heading, v * math.cos(heading), v * math.sin(heading)


def test_bins(lib):
    require(lib.pf_n_accel() == 21, "accel bin count")
    require(lib.pf_n_steer() == 31, "steer bin count")
    require(lib.pf_coast_accel() == 10, "accel coast bin")
    require(lib.pf_coast_steer() == 15, "steer coast bin")
    require(lib.pf_accel_bin(-4.0) == 0, "accel -4")
    require(lib.pf_accel_bin(0.0) == lib.pf_coast_accel(), "accel 0")
    require(lib.pf_accel_bin(4.0) == 20, "accel 4")
    require(lib.pf_steer_bin(-1.0) == 0, "steer -1")
    require(lib.pf_steer_bin(0.0) == lib.pf_coast_steer(), "steer 0")
    require(lib.pf_steer_bin(1.0) == 30, "steer 1")
    require(abs(lib.pf_accel_value(lib.pf_coast_accel())) < 1e-6, "coast accel value")
    require(abs(lib.pf_steer_value(lib.pf_coast_steer())) < 1e-6, "coast steer value")
    require(abs(lib.pf_accel_value(0) - (-4.0)) < 1e-5, "accel endpoint")
    require(abs(lib.pf_accel_value(20) - 4.0) < 1e-5, "accel endpoint high")
    require(abs(lib.pf_steer_value(0) - (-1.0)) < 1e-5, "steer endpoint")
    require(abs(lib.pf_steer_value(30) - 1.0) < 1e-5, "steer endpoint high")
    print("pass bins 21 x 31")


def test_dynamics_formula(lib):
    h = drv.create_sim(lib, DATA, MAP_ID, num_agents=1, episode_length=500)
    length = 4.5
    cases = [
        (20, lib.pf_coast_steer(), 10.0, 0.0),
        (lib.pf_coast_accel(), 23, 8.0, 0.2),
    ]
    try:
        for ab, sb, speed, heading in cases:
            accel = lib.pf_accel_value(ab)
            steer = lib.pf_steer_value(sb)
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)
            lib.pf_set_agent(
                h, 0, 0.0, 0.0, heading, vx, vy, 1.8, length, 5000.0, 0.0,
            )
            lib.pf_set_action(h, 0, float(ab), float(sb))
            lib.pf_step(h)
            st = drv.agent_state(lib, h, 0)
            ex = bicycle(0.0, 0.0, heading, vx, vy, accel, steer, length)
            got = (float(st[0]), float(st[1]), float(st[2]), float(st[3]), float(st[4]))
            for name, a, b in zip(("x", "y", "heading", "vx", "vy"), got, ex):
                require(abs(a - b) < 2e-4, f"formula {name} got {a} expected {b}")
        print("pass dynamics formula")
    finally:
        lib.pf_close(h)


def push_goals(lib, handles):
    n = lib.pf_n(handles[0])
    for i in range(n):
        st = drv.agent_state(lib, handles[0], i)
        if st[10] < 0.5:
            continue
        gx = float(st[0]) + math.cos(float(st[2])) * 5000.0
        gy = float(st[1]) + math.sin(float(st[2])) * 5000.0
        for h in handles:
            lib.pf_set_goal(h, i, gx, gy)
    for h in handles:
        lib.pf_recompute(h)


def test_step_matches_integrate(lib):
    drive = drv.create_sim(lib, DATA, MAP_ID, num_agents=4, episode_length=500)
    offline = drv.create_sim(lib, DATA, MAP_ID, num_agents=4, episode_length=500)
    rng = np.random.default_rng(7)
    na = lib.pf_n_accel()
    ns = lib.pf_n_steer()
    try:
        lib.pf_reset(drive, 11)
        lib.pf_reset(offline, 11)
        require(np.array_equal(drv.observations(lib, drive), drv.observations(lib, offline)),
            "reset observations differ")
        push_goals(lib, (drive, offline))
        require(np.array_equal(drv.observations(lib, drive), drv.observations(lib, offline)),
            "far-goal observations differ")
        for step in range(20):
            actions = rng.integers(0, [na, ns], size=(lib.pf_n(drive), 2))
            drv.set_actions(lib, drive, actions)
            drv.set_actions(lib, offline, actions)
            lib.pf_step(drive)
            lib.pf_integrate(offline, 0)
            obs_d = drv.observations(lib, drive)
            obs_o = drv.observations(lib, offline)
            require(np.array_equal(obs_d, obs_o), f"integrate obs differ at step {step}")
            st_d = drv.states(lib, drive)
            st_o = drv.states(lib, offline)
            require(np.array_equal(st_d, st_o), f"integrate state differ at step {step}")
            for i in range(lib.pf_n(drive)):
                rd = lib.pf_reward(drive, i)
                ro = lib.pf_reward(offline, i)
                require(rd == ro, f"reward differ agent {i} step {step}: {rd} {ro}")
        print("pass puf_step matches offline integrate")
    finally:
        lib.pf_close(drive)
        lib.pf_close(offline)


def test_online_offline_obs(lib):
    drive = drv.create_sim(lib, DATA, MAP_ID, num_agents=4, episode_length=500)
    offline = drv.create_sim(lib, DATA, MAP_ID, num_agents=4, episode_length=500)
    rng = np.random.default_rng(1)
    na = lib.pf_n_accel()
    ns = lib.pf_n_steer()
    try:
        lib.pf_reset(drive, 3)
        lib.pf_reset(offline, 3)
        require(np.array_equal(drv.states(lib, drive), drv.states(lib, offline)),
            "reset state differs")
        for step in range(30):
            actions = rng.integers(0, [na, ns], size=(4, 2))
            drv.set_actions(lib, drive, actions)
            drv.set_actions(lib, offline, actions)
            lib.pf_step(drive)
            lib.pf_step(offline)
            require(np.array_equal(
                drv.observations(lib, drive), drv.observations(lib, offline),
            ), f"obs differ at step {step}")
            require(np.array_equal(drv.states(lib, drive), drv.states(lib, offline)),
                f"state differs at step {step}")
        print("pass online and offline observation sequences")
    finally:
        lib.pf_close(drive)
        lib.pf_close(offline)


def test_log_shift(lib):
    import json
    scenario = drv.create_scenario(lib, DATA, MAP_ID, JSONS[MAP_ID])
    try:
        raw = json.loads(JSONS[MAP_ID].read_text())
        vehicles = [o for o in raw["objects"] if o.get("type") == "vehicle"]
        mx, my = drv.mean_xy(lib, scenario)
        pos = vehicles[0]["position"][0]
        expect_x = np.float32(pos["x"]) - np.float32(mx)
        expect_y = np.float32(pos["y"]) - np.float32(my)
        lx, ly = drv.log_xy(lib, scenario, 0, 0)
        require(abs(lx - float(expect_x)) < 1e-3, f"log x {lx} vs {expect_x}")
        require(abs(ly - float(expect_y)) < 1e-3, f"log y {ly} vs {expect_y}")
        st = drv.agent_state(lib, scenario, 0)
        require(abs(float(st[0]) - lx) < 1e-5 and abs(float(st[1]) - ly) < 1e-5,
            "placed pose is not the centered log")
        print(f"pass world-mean shift ({mx:.2f}, {my:.2f})")
    finally:
        lib.pf_close(scenario)


def copy_frame(lib, scenario, sim, frame):
    lib.pf_place(scenario, frame)
    lib.pf_recompute(scenario)
    n = lib.pf_n(scenario)
    invalid = lib.pf_invalid()
    for i in range(n):
        if not lib.pf_log_valid(scenario, i, frame):
            lib.pf_set_agent(sim, i, invalid, invalid, 0, 0, 0, 1.8, 4.5, 0, 0)
            continue
        st = drv.agent_state(lib, scenario, i)
        lib.pf_set_agent(
            sim, i, float(st[0]), float(st[1]), float(st[2]), float(st[3]),
            float(st[4]), float(st[7]), float(st[8]), float(st[5]), float(st[6]),
        )
    lib.pf_recompute(sim)


def test_copied_pose(lib):
    scenario = drv.create_scenario(lib, DATA, MAP_ID, JSONS[MAP_ID])
    n = lib.pf_log_n(scenario)
    sim = drv.create_sim(lib, DATA, MAP_ID, num_agents=n, episode_length=500)
    try:
        checked = 0
        for frame in (0, 10, 40, 80):
            copy_frame(lib, scenario, sim, frame)
            obs_s = drv.observations(lib, scenario)
            obs_d = drv.observations(lib, sim)
            require(np.array_equal(obs_s, obs_d), f"copied pose obs differ at frame {frame}")
            checked += 1
        require(checked == 4, "copied pose frames")
        print("pass logged pose matches drive.h observations")
    finally:
        lib.pf_close(scenario)
        lib.pf_close(sim)


def consecutive_labels(lib, h, i):
    traj = lib.pf_traj()
    count = 0
    for t in range(traj - 1):
        if lib.pf_log_valid(h, i, t) and lib.pf_log_valid(h, i, t + 1):
            count += 1
    return count


def test_collect_and_inverse(lib):
    scenario = drv.create_scenario(lib, DATA, MAP_ID, JSONS[MAP_ID])
    try:
        tele_obs, tele_act, tele_valid, tele_ade = drv.collect(lib, scenario, 1)
        non_obs, non_act, non_valid, non_ade = drv.collect(lib, scenario, 0)
        require(np.array_equal(tele_obs[0], non_obs[0]),
            "non-teleport frame 0 differs from teleport")
        n = lib.pf_log_n(scenario)
        fit = 0
        for i in range(n):
            expect = consecutive_labels(lib, scenario, i)
            if lib.pf_base_unfit(scenario, i):
                require(int(tele_valid[:, i].sum()) == 0, f"unfit teleport mask {i}")
                require(int(non_valid[:, i].sum()) == 0, f"unfit non-teleport mask {i}")
                continue
            fit += 1
            require(int(tele_valid[:, i].sum()) == expect, f"teleport mask {i}")
            require(abs(float(tele_ade[i])) < 1e-6, "teleport ade")
            if float(non_ade[i]) > 2.0:
                require(int(non_valid[:, i].sum()) == 0, f"high ADE kept {i}")
                require(np.all(non_act[:, i] == -1), f"high ADE labels kept {i}")
            else:
                require(int(non_valid[:, i].sum()) == expect, f"non-teleport mask {i}")
                require(np.array_equal(tele_act[:, i], non_act[:, i]), f"labels differ {i}")
        require(fit > 0, "no fit vehicle on map 0")

        for t in (0, 30, 60):
            lib.pf_place(scenario, t)
            lib.pf_recompute(scenario)
            got = drv.observations(lib, scenario)
            require(np.array_equal(got, tele_obs[t]), f"collect obs frame {t}")

        errs = []
        in_range = []
        for i in range(n):
            if lib.pf_base_unfit(scenario, i):
                continue
            for t in range(lib.pf_traj() - 1):
                if not lib.pf_log_valid(scenario, i, t):
                    continue
                if not lib.pf_log_valid(scenario, i, t + 1):
                    continue
                ab, sb = drv.log_bins(lib, scenario, i, t)
                if ab < 0:
                    continue
                lib.pf_place(scenario, t)
                for j in range(n):
                    aj, sj = drv.log_bins(lib, scenario, j, t)
                    if aj < 0:
                        lib.pf_set_action(scenario, j, float(lib.pf_coast_accel()),
                            float(lib.pf_coast_steer()))
                    else:
                        lib.pf_set_action(scenario, j, float(aj), float(sj))
                lib.pf_integrate(scenario, 0)
                st = drv.agent_state(lib, scenario, i)
                x1, y1 = drv.log_xy(lib, scenario, i, t + 1)
                err = math.hypot(float(st[0]) - x1, float(st[1]) - y1)
                errs.append(err)
                # Endpoint bins are the clamp of the labeling grid.
                if abs(lib.pf_steer_value(sb)) < 0.95 and abs(lib.pf_accel_value(ab)) < 3.9:
                    in_range.append(err)
        mean_err = float(np.mean(errs))
        mean_in = float(np.mean(in_range)) if in_range else mean_err
        print(f"one-step inverse mean {mean_err:.3f} m "
            f"in-range {mean_in:.3f} m over {len(errs)} steps")
        require(mean_err < 0.45, f"inverse mean error {mean_err}")
        require(mean_in < 0.2, f"in-range inverse error {mean_in}")
        print(f"pass collect masks, {fit} fit vehicles, "
            f"non-teleport ADE {np.array2string(non_ade, precision=2)}")
    finally:
        lib.pf_close(scenario)


def main():
    require(len(JSONS) == 1000, f"expected 1000 jsons, found {len(JSONS)}")
    require((DATA / "binaries" / "map_000.bin").is_file(), "missing map_000.bin")
    lib = drv.load()
    test_bins(lib)
    test_dynamics_formula(lib)
    test_step_matches_integrate(lib)
    test_online_offline_obs(lib)
    test_log_shift(lib)
    test_copied_pose(lib)
    test_collect_and_inverse(lib)
    print("all observation checks passed")


if __name__ == "__main__":
    main()
