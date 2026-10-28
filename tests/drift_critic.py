"""Closed-loop drift critic on a frozen Waymo BC anchor.

The trunk stays fixed. A linear head on its last hidden layer is the
least-squares fit of discounted future log displacement. Chosen constants,
not tuned: horizon H = 10, gamma = 0.99, alpha = 0.05. The sum is

    E_t = sum_{k=0}^{H} gamma^k * ||x_{t+k}^{sim} - x_{t+k}^{log}||_2

and a sample is kept only when every frame in that window is on the log.
Rollouts drive every controllable agent together, the same closed loop as
the Waymo BC eval. V is clamped at 0 when the trainer builds

    beta = anchor_coef / (1 + alpha * max(V, 0)).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_bind as drv
import train_bc_map as t

DRIFT_H = 10
DRIFT_GAMMA = 0.99
DRIFT_ALPHA = 0.05
TOWN_EPISODES = 32
TOWN_AGENTS = 128
TOWN_GOAL_MIN = 5.0
TOWN_GOAL_MAX = 30.0
TOWN_EPISODE = 200
# puf_step aggregates on the call that reaches episode_length, so a
# 200-length episode moves for 199 calls. That is the trainer's episode.
TOWN_MOVES = TOWN_EPISODE - 1
REPLAY_STEPS = drv.TRAJ - 1


def _window_targets(times, dists, horizon, gamma):
    pos = {int(step): i for i, step in enumerate(times)}
    keep = []
    values = []
    for i, step in enumerate(times):
        acc = 0.0
        ok = True
        for k in range(horizon + 1):
            j = pos.get(int(step) + k)
            if j is None:
                ok = False
                break
            acc += (gamma ** k) * float(dists[j])
        if ok:
            keep.append(i)
            values.append(acc)
    return keep, values


def _check_window():
    keep, values = _window_targets([0, 1, 2], [1.0, 1.0, 1.0], 2, 0.5)
    assert keep == [0]
    assert abs(values[0] - 1.75) < 1e-9


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def _quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    qs = [0.0, 0.1, 0.5, 0.9, 1.0]
    return {f"q{int(q * 100)}": float(np.quantile(x, q)) for q in qs}


def _solve_head(features, target):
    d = features.shape[1]
    gram = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=features.device)
    rhs = torch.zeros(d + 1, dtype=torch.float64, device=features.device)
    for i in range(0, features.shape[0], 65536):
        block = features[i:i + 65536].double()
        y = target[i:i + 65536].double()
        gram[:d, :d] += block.T @ block
        col = block.sum(dim=0)
        gram[:d, d] += col
        gram[d, :d] += col
        gram[d, d] += block.shape[0]
        rhs[:d] += block.T @ y
        rhs[d] += y.sum()
    eye = torch.eye(d, dtype=torch.float64, device=features.device)
    try:
        coef = torch.linalg.solve(gram, rhs)
        ridge = 0.0
    except RuntimeError:
        gram[:d, :d] += 1e-4 * eye
        coef = torch.linalg.solve(gram, rhs)
        ridge = 1e-4
    weight = coef[:d].float().detach().cpu().numpy()
    bias = float(coef[d].detach().cpu())
    return weight, bias, ridge


def _predict(features, weight, bias):
    return features @ weight + bias


def _beta_ratio(pred, alpha):
    v = np.maximum(pred, 0.0)
    return 1.0 / (1.0 + alpha * v)


def _collect_map(lib, h, pol, device):
    agents = t.controllable_agents(lib, h)
    if not agents:
        return None, None
    lib.pf_reset(h, 0)
    t._park_unfit(lib, h)
    times = {a: [] for a in agents}
    dists = {a: [] for a in agents}
    rows = {a: [] for a in agents}
    for step in range(t.HORIZON_STEPS):
        lib.pf_set_timestep(h, step)
        lib.pf_spawn_due(h)
        lib.pf_recompute(h)
        obs = drv.observations(lib, h)
        live = []
        states = {}
        for agent in agents:
            state = drv.agent_state(lib, h, agent)
            if float(state[10]) <= 0.5:
                continue
            live.append(agent)
            states[agent] = state
        if live:
            idx = np.asarray(live)
            batch = (obs[idx] - pol.mu) / pol.sig
            acc, steer, _ent = pol.act_batch(batch)
            for agent, accel, steer_i in zip(live, acc, steer):
                lib.pf_set_action(h, agent, float(accel), float(steer_i))
            for agent, row in zip(live, obs[idx]):
                if step < drv.TRAJ and lib.pf_log_valid(h, agent, step):
                    lx, ly = drv.log_xy(lib, h, agent, step)
                    state = states[agent]
                    dist = float(np.hypot(float(state[0]) - lx, float(state[1]) - ly))
                    times[agent].append(step)
                    dists[agent].append(dist)
                    rows[agent].append(row.astype(np.float32))
        lib.pf_integrate(h, 1)
    kept_obs = []
    kept_e = []
    for agent in agents:
        keep, values = _window_targets(
            times[agent], dists[agent], DRIFT_H, DRIFT_GAMMA,
        )
        for i, value in zip(keep, values):
            kept_obs.append(rows[agent][i])
            kept_e.append(value)
    if not kept_obs:
        return None, None
    obs = np.stack(kept_obs).astype(np.float32)
    target = np.asarray(kept_e, dtype=np.float32)
    norm = (obs - pol.mu) / pol.sig
    with torch.no_grad():
        hidden = pol.trunk(torch.as_tensor(norm, dtype=torch.float32, device=device))
    return hidden.detach().cpu(), target


def _holdout_ids(n_maps):
    n_hold = max(1, n_maps // 10)
    return list(range(n_maps - n_hold, n_maps))


def _plot_drift(held_y, held_p, ratio, path):
    t.apply_figure_style()
    colors = sns.color_palette()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    ax_s, ax_h = axes
    ax_s.scatter(held_y, held_p, s=8, alpha=0.2, linewidths=0, color=colors[0])
    lo = float(min(held_y.min(), held_p.min()))
    hi = float(max(held_y.max(), held_p.max()))
    ax_s.plot([lo, hi], [lo, hi], color="black", lw=1.0)
    ax_s.set_xlabel("discounted displacement E (m)")
    ax_s.set_ylabel("predicted V (m)")
    ax_s.set_title("held-out maps")
    t.style_ax(ax_s)
    ax_h.hist(ratio, bins=40, color=colors[1])
    ax_h.set_xlabel("beta / beta0")
    ax_h.set_ylabel("samples")
    ax_h.set_title(f"alpha {DRIFT_ALPHA:g}, V clamped at 0")
    t.style_ax(ax_h)
    fig.suptitle(
        f"Drift critic  H={DRIFT_H}  gamma={DRIFT_GAMMA:g}  "
        "closed loop, all controllable agents"
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fit_drift_head(n_maps=1000):
    """Fit Vd, bd and write checkpoints/bc_womd_{n}_drift.npz."""
    _check_window()
    paths = t._run_paths(n_maps)
    if not paths["ckpt"].is_file():
        raise SystemExit(f"missing {paths['ckpt']}")
    device = t._device()
    pol, acc = t.load_policy(paths["ckpt"])
    pol.to(device)
    pol.eval()
    hold = set(_holdout_ids(n_maps))
    lib = drv.load()
    train_h = []
    train_y = []
    hold_h = []
    hold_y = []
    for map_id in range(n_maps):
        h = drv.create_scenario(lib, t.DATA, map_id, t.JSONS[map_id])
        try:
            hidden, target = _collect_map(lib, h, pol, device)
        finally:
            lib.pf_close(h)
        if hidden is None:
            print(f"drift map {map_id} no window", flush=True)
            continue
        if map_id in hold:
            hold_h.append(hidden)
            hold_y.append(target)
        else:
            train_h.append(hidden)
            train_y.append(target)
        print(
            f"drift map {map_id} samples {len(target)}",
            flush=True,
        )
    if not train_h or not hold_h:
        raise SystemExit("drift collection produced an empty split")
    x_train = torch.cat(train_h).to(device)
    y_train = torch.as_tensor(np.concatenate(train_y), dtype=torch.float32, device=device)
    x_hold = torch.cat(hold_h).to(device)
    y_hold_np = np.concatenate(hold_y).astype(np.float64)
    weight, bias, ridge = _solve_head(x_train, y_train)
    pred_train = _predict(x_train.detach().cpu().numpy(), weight, bias)
    pred_hold = _predict(x_hold.detach().cpu().numpy(), weight, bias)
    y_train_np = y_train.detach().cpu().numpy().astype(np.float64)
    mean_y = float(y_train_np.mean())
    train_mse = float(np.mean((pred_train - y_train_np) ** 2))
    hold_mse = float(np.mean((pred_hold - y_hold_np) ** 2))
    mean_mse = float(np.mean((y_hold_np - mean_y) ** 2))
    ratio = _beta_ratio(pred_hold, DRIFT_ALPHA)
    out_npz = paths["ckpt"].with_name(paths["ckpt"].stem + "_drift.npz")
    src = np.load(paths["ckpt"])
    payload = {key: np.array(src[key]) for key in src.files}
    src.close()
    payload["Vd"] = weight.astype(np.float32)
    payload["bd"] = np.asarray([bias], dtype=np.float32)
    payload["drift_h"] = np.int32(DRIFT_H)
    payload["drift_gamma"] = np.float32(DRIFT_GAMMA)
    payload["drift_alpha"] = np.float32(DRIFT_ALPHA)
    np.savez(out_npz, **payload)
    summary = {
        "source": str(paths["ckpt"]),
        "checkpoint": str(out_npz),
        "accuracy": acc,
        "horizon": DRIFT_H,
        "gamma": DRIFT_GAMMA,
        "alpha": DRIFT_ALPHA,
        "window": "t through t+H inclusive, log-valid on every frame",
        "rollout": "all controllable agents, greedy, one pass",
        "fit": "unregularized least squares" if ridge == 0.0 else f"ridge {ridge}",
        "n_train": int(len(y_train_np)),
        "n_holdout": int(len(y_hold_np)),
        "holdout_maps": _holdout_ids(n_maps),
        "holdout_note": (
            "Last 10% of map ids. The BC policy was trained on every map, "
            "so this split is new closed-loop labels, not a policy holdout."
        ),
        "train_mse": train_mse,
        "holdout_mse": hold_mse,
        "holdout_mse_predict_train_mean": mean_mse,
        "holdout_pearson": _corr(y_hold_np, pred_hold),
        "train_E": _quantiles(y_train_np),
        "holdout_E": _quantiles(y_hold_np),
        "holdout_V": _quantiles(pred_hold),
        "holdout_beta_over_beta0": _quantiles(ratio),
        "holdout_frac_beta_below_half": float(np.mean(ratio < 0.5)),
    }
    out_json = t.OUT / f"bc_{int(n_maps)}_drift.json"
    out_png = t.OUT / f"bc_{int(n_maps)}_drift.png"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    _plot_drift(y_hold_np, pred_hold, ratio, out_png)
    print(
        f"drift holdout mse {hold_mse:.4f}  mean-predictor {mean_mse:.4f}  "
        f"r {_corr(y_hold_np, pred_hold):.3f}",
        flush=True,
    )
    print(f"wrote {out_npz}", flush=True)
    print(f"wrote {out_json}", flush=True)
    return out_npz


def _rates(flags):
    flags = np.asarray(flags, dtype=np.float64)
    return float(flags.mean()), float(flags.std())


def town01_anchor_baseline(n_maps=1000, episodes=TOWN_EPISODES):
    """Greedy anchor in the Town01 self-play env the RL runs train on."""
    drift_path = drv.ROOT / "checkpoints" / f"bc_womd_{int(n_maps)}_drift.npz"
    plain = t._run_paths(n_maps)["ckpt"]
    path = drift_path if drift_path.is_file() else plain
    pol, acc = t.load_policy(path)
    pol.eval()
    lib = drv.load()
    h = drv.create_sim(
        lib, t.CARLA_DIR, 1, TOWN_AGENTS, episode_length=TOWN_EPISODE,
        goal_min=TOWN_GOAL_MIN, goal_max=TOWN_GOAL_MAX, physics="through",
    )
    scores = []
    collisions = []
    offroads = []
    try:
        for episode in range(episodes):
            lib.pf_reset(h, 1000 + episode)
            n = lib.pf_n(h)
            pose = drv.states(lib, h)
            assert np.all(pose[:, 10] > 0.5)
            scored = np.zeros(n, dtype=bool)
            blocked = np.zeros(n, dtype=bool)
            collided = np.zeros(n, dtype=bool)
            offroad = np.zeros(n, dtype=bool)
            for _step in range(TOWN_MOVES):
                obs = drv.observations(lib, h)
                pose = drv.states(lib, h)
                old_xy = pose[:, 0:2].copy()
                old_goal = pose[:, 5:7].copy()
                batch = (obs - pol.mu) / pol.sig
                accels, steers, _ent = pol.act_batch(batch)
                for i in range(n):
                    lib.pf_set_action(h, i, float(accels[i]), float(steers[i]))
                lib.pf_step(h)
                pose = drv.states(lib, h)
                kind = pose[:, 9].astype(np.int32)
                hit = kind == 1
                road = kind == 2
                collided |= hit
                offroad |= road
                # puf_step sets collided_before_goal before it tests the disk,
                # so a collision on the arrival step does not score.
                blocked |= (hit | road) & ~scored
                dist = np.hypot(pose[:, 0] - old_goal[:, 0], pose[:, 1] - old_goal[:, 1])
                # Arrival respawns inside the step. The car can stay put while
                # the goal moves, or a failed forward goal can teleport the car.
                goal_moved = np.hypot(
                    pose[:, 5] - old_goal[:, 0], pose[:, 6] - old_goal[:, 1],
                ) > 1.0
                jumped = np.hypot(
                    pose[:, 0] - old_xy[:, 0], pose[:, 1] - old_xy[:, 1],
                ) > 12.0
                arrived = (dist < t.GOAL_RADIUS_M) | goal_moved | jumped
                scored |= arrived & ~blocked
            scores.append(float(scored.mean()))
            collisions.append(float(collided.mean()))
            offroads.append(float(offroad.mean()))
            print(
                f"town01 episode {episode} score {scores[-1]:.3f} "
                f"collision {collisions[-1]:.3f} offroad {offroads[-1]:.3f}",
                flush=True,
            )
    finally:
        lib.pf_close(h)
    score_m, score_s = _rates(scores)
    col_m, col_s = _rates(collisions)
    off_m, off_s = _rates(offroads)
    summary = {
        "checkpoint": str(path),
        "accuracy": acc,
        "env": "Town01 map_id 1, data/mixed, physics through",
        "agents": TOWN_AGENTS,
        "episodes": episodes,
        "episode_length": TOWN_EPISODE,
        "movement_steps": TOWN_MOVES,
        "goal_m": [TOWN_GOAL_MIN, TOWN_GOAL_MAX],
        "policy": "greedy argmax of the BC anchor",
        "score": (
            "puf_step first arrival: inside 5 m, or the arrival respawn moved "
            "the goal by more than 1 m or the car by more than 12 m, with no "
            "vehicle collision or off-road on that step or earlier"
        ),
        "score_mean": score_m,
        "score_std": score_s,
        "collision_rate_mean": col_m,
        "collision_rate_std": col_s,
        "offroad_rate_mean": off_m,
        "offroad_rate_std": off_s,
        "episodes_score": scores,
        "episodes_collision_rate": collisions,
        "episodes_offroad_rate": offroads,
    }
    out = t.OUT / "anchor_town01_baseline.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"town01 anchor score {score_m:.3f} ± {score_s:.3f}  "
        f"collision {col_m:.3f}  offroad {off_m:.3f}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return out


def _mean_std(rows, key):
    vals = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std())


def _pack(rows):
    out = {"n": len(rows)}
    for key in ("score", "collision_rate", "offroad_rate", "completion_rate"):
        mean, std = _mean_std(rows, key)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def eval_anchor_replay(n_maps=1000):
    """Paper-style Waymo eval of the greedy anchor: human_replay and self_play."""
    drift_path = drv.ROOT / "checkpoints" / f"bc_womd_{int(n_maps)}_drift.npz"
    plain = t._run_paths(n_maps)["ckpt"]
    path = drift_path if drift_path.is_file() else plain
    pol, acc = t.load_policy(path)
    pol.eval()
    lib = drv.load()
    human = []
    self_play = []
    for map_id in range(n_maps):
        h = drv.create_scenario(lib, t.DATA, map_id, t.JSONS[map_id])
        try:
            agents = t.controllable_agents(lib, h)
            if not agents:
                print(f"replay map {map_id} no agents", flush=True)
                continue
            for agent in agents:
                out = t.rollout(lib, h, pol, agent, steps=REPLAY_STEPS)
                human.append({
                    "map_id": map_id,
                    "agent": agent,
                    "score": float(out["reached"] and not out["collided_before"]),
                    "collision_rate": float(out["collided"]),
                    "offroad_rate": float(out["offroad"]),
                    "completion_rate": float(out["reached"]),
                })
            row = _self_play_map(lib, h, pol, agents)
            row["map_id"] = map_id
            if row["n_agents"] > 0:
                self_play.append(row)
            print(
                f"replay map {map_id} agents {len(agents)} "
                f"self score {row['score']:.2f} "
                f"collision {row['collision_rate']:.2f} "
                f"offroad {row['offroad_rate']:.2f}",
                flush=True,
            )
        finally:
            lib.pf_close(h)
    summary = {
        "checkpoint": str(path),
        "accuracy": acc,
        "steps": REPLAY_STEPS,
        "human_replay": (
            "one controllable agent, partners copied from the log each frame"
        ),
        "self_play": "every controllable agent is greedy BC, nobody is replayed",
        "score": "logged goal inside 5 m before any collision or off-road",
        "rl_checkpoints": (
            "The four minGRU runs are not in this figure. This tree has no "
            "Waymo control_sdc_only path for a trainer checkpoint."
        ),
        "modes": {
            "human_replay": _pack(human),
            "self_play": _pack(self_play),
        },
    }
    out = t.OUT / f"bc_{int(n_maps)}_replay.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    _plot_replay(summary, t.OUT / f"bc_{int(n_maps)}_replay.png")
    human_pack = summary["modes"]["human_replay"]
    self_pack = summary["modes"]["self_play"]
    print(
        f"human_replay n {human_pack['n']} score {human_pack['score_mean']:.3f} "
        f"collision {human_pack['collision_rate_mean']:.3f} "
        f"offroad {human_pack['offroad_rate_mean']:.3f}",
        flush=True,
    )
    print(
        f"self_play n {self_pack['n']} score {self_pack['score_mean']:.3f} "
        f"collision {self_pack['collision_rate_mean']:.3f} "
        f"offroad {self_pack['offroad_rate_mean']:.3f}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return out


def _self_play_map(lib, h, pol, agents):
    ends = {}
    for agent in agents:
        xs, ys = t.log_track(lib, h, agent)
        ends[agent] = (xs[-1], ys[-1])
    lib.pf_reset(h, 0)
    t._park_unfit(lib, h)
    collided = {a: False for a in agents}
    offroad = {a: False for a in agents}
    scored = {a: False for a in agents}
    blocked = {a: False for a in agents}
    completed = {a: False for a in agents}
    seen = {a: False for a in agents}
    for step in range(REPLAY_STEPS):
        lib.pf_set_timestep(h, step)
        lib.pf_spawn_due(h)
        lib.pf_recompute(h)
        obs = drv.observations(lib, h)
        live = []
        for agent in agents:
            state = drv.agent_state(lib, h, agent)
            if float(state[10]) <= 0.5:
                continue
            live.append(agent)
            seen[agent] = True
        if live:
            batch = (obs[np.asarray(live)] - pol.mu) / pol.sig
            accels, steers, _ent = pol.act_batch(batch)
            for agent, accel, steer in zip(live, accels, steers):
                lib.pf_set_action(h, agent, float(accel), float(steer))
        lib.pf_integrate(h, 1)
        for agent in live:
            state = drv.agent_state(lib, h, agent)
            kind = int(state[9])
            if kind == 1:
                collided[agent] = True
            elif kind == 2:
                offroad[agent] = True
            if kind != 0 and not scored[agent]:
                blocked[agent] = True
            ex, ey = ends[agent]
            dist = float(np.hypot(float(state[0]) - ex, float(state[1]) - ey))
            if dist <= t.GOAL_RADIUS_M and not completed[agent]:
                completed[agent] = True
                if not blocked[agent]:
                    scored[agent] = True
    used = [a for a in agents if seen[a]]
    if not used:
        return {
            "n_agents": 0,
            "score": float("nan"),
            "collision_rate": float("nan"),
            "offroad_rate": float("nan"),
            "completion_rate": float("nan"),
        }
    return {
        "n_agents": len(used),
        "score": float(np.mean([scored[a] for a in used])),
        "collision_rate": float(np.mean([collided[a] for a in used])),
        "offroad_rate": float(np.mean([offroad[a] for a in used])),
        "completion_rate": float(np.mean([completed[a] for a in used])),
    }


def _plot_replay(summary, path):
    t.apply_figure_style()
    colors = sns.color_palette()
    modes = ("human_replay", "self_play")
    metrics = ("score", "collision_rate", "offroad_rate")
    labels = ("score", "collision", "off-road")
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.36
    for shift, mode, color in (
        (-width / 2, modes[0], colors[0]),
        (width / 2, modes[1], colors[1]),
    ):
        pack = summary["modes"][mode]
        means = [pack[f"{key}_mean"] for key in metrics]
        stds = [pack[f"{key}_std"] for key in metrics]
        ax.bar(
            x + shift, means, width=width, yerr=stds, color=color, capsize=3,
            label=mode, error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean across cases")
    top = 1.05
    for mode in modes:
        pack = summary["modes"][mode]
        for key in metrics:
            top = max(top, pack[f"{key}_mean"] + pack[f"{key}_std"])
    ax.set_ylim(0.0, 1.05 if top <= 1.05 else top * 1.05)
    ax.legend(frameon=False)
    ax.set_title(
        f"Greedy anchor on Waymo, {summary['steps']} steps. "
        "Std is across agents (human_replay) or maps (self_play)."
    )
    t.style_ax(ax)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}", flush=True)
