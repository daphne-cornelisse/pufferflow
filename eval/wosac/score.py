#!/usr/bin/env python3
"""Waymo Open Sim Agent Challenge scores for this simulator.

Protocol matches PufferDrive 2.0: start at logged frame 10, record 81
states (that frame plus 80 futures), 32 rollouts, control every valid
mover, stop inside a 2 m goal. Likelihoods are the 2024 histograms in
wosac.ini. The map set is data/womd_train_1000, not the 229-scene
clean validation split.

    python eval/wosac/score.py rollout NAME CKPT OUTDIR
    python eval/wosac/score.py score OUTDIR
    python eval/wosac/score.py ground-truth
    python eval/wosac/score.py plot
"""

import configparser
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import estimators
import metrics

ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = ROOT / "data" / "womd_train_1000" / "binaries"
JSON_DIR = ROOT / "data" / "womd_train_1000" / "json"
LOG_DIR = ROOT / "logs" / "womd"
FIG_PATH = LOG_DIR / "wosac.png"
SCORE_PATH = LOG_DIR / "wosac_scores.json"
PUFFER = Path("/tmp/puffer-hr")
INI = ROOT / "config" / "womd_short.ini"
ANCHOR = ROOT / "checkpoints/womd/drive/womd_anchor/0000000499908608.bin"
FREE = ROOT / "checkpoints/womd/drive/womd_free/0000000499908608.bin"
INIT_STEPS = 10
SIM_STEPS = 81
ROLLOUTS = 32
NMAPS = 1000
ROAD_LANE = 4
INVALID = np.float32(-10000.0)
VEHICLE = 1

FIELDS = (
    "linear_speed",
    "linear_acceleration",
    "angular_speed",
    "angular_acceleration",
    "distance_to_nearest_object",
    "time_to_collision",
    "collision_indication",
    "distance_to_road_edge",
    "offroad_indication",
)
KIN = (
    "likelihood_linear_speed",
    "likelihood_linear_acceleration",
    "likelihood_angular_speed",
    "likelihood_angular_acceleration",
)
INTER = (
    "likelihood_collision_indication",
    "likelihood_distance_to_nearest_object",
    "likelihood_time_to_collision",
)
MAP = (
    "likelihood_distance_to_road_edge",
    "likelihood_offroad_indication",
)
METHODS = (
    ("Unregularized", "free", "#111111"),
    ("Regularized", "anchor", "#4C72B0"),
    ("Ground truth", "ground_truth", "#55A868"),
)


class Reader:
    def __init__(self, path):
        self.b = Path(path).read_bytes()
        self.i = 0

    def ints(self, n):
        out = np.frombuffer(self.b, dtype="<i4", count=n, offset=self.i).copy()
        self.i += 4 * n
        return out

    def floats(self, n):
        out = np.frombuffer(self.b, dtype="<f4", count=n, offset=self.i).copy()
        self.i += 4 * n
        return out

    def done(self):
        assert self.i == len(self.b)


def read_log(path):
    r = Reader(path)
    nobj, _nroads, typ, traj = (int(v) for v in r.ints(4))
    assert traj == 91 and 0 <= nobj <= 512
    objs = []
    road_type = 0
    road_size = 0
    for _ in range(nobj):
        x = r.floats(traj)
        y = r.floats(traj)
        r.floats(traj)
        r.floats(traj)
        r.floats(traj)
        r.floats(traj)
        heading = r.floats(traj)
        valid = r.ints(traj)
        width, length, _h, _gx, _gy, _gz = r.floats(6)
        r.ints(1)
        road_type, road_size = (int(v) for v in r.ints(2))
        objs.append({
            "type": int(typ),
            "x": x,
            "y": y,
            "heading": heading,
            "valid": valid,
            "width": float(width),
            "length": float(length),
        })
        typ = road_type
    roads = []
    for i in range(_nroads):
        if i > 0:
            road_type, road_size = (int(v) for v in r.ints(2))
        xs = r.floats(road_size)
        ys = r.floats(road_size)
        r.floats(road_size)
        r.floats(6)
        r.ints(1)
        roads.append((road_type, xs, ys))
    r.done()
    return objs, roads


def world_mean(roads):
    mean_x = np.float32(0.0)
    mean_y = np.float32(0.0)
    count = 0
    for typ, xs, ys in roads:
        if typ < ROAD_LANE:
            continue
        for x, y in zip(xs, ys):
            if x == INVALID or y == INVALID:
                continue
            count += 1
            c = np.float32(count)
            mean_x += (np.float32(x) - mean_x) / c
            mean_y += (np.float32(y) - mean_y) / c
    return float(mean_x), float(mean_y)


def read_meta(path):
    r = Reader(path)
    mean_x, mean_y = (float(v) for v in r.floats(2))
    nedges = int(r.ints(1)[0])
    lengths = []
    xs = []
    ys = []
    for _ in range(nedges):
        n = int(r.ints(1)[0])
        lengths.append(n)
        xs.append(r.floats(n))
        ys.append(r.floats(n))
    r.done()
    return mean_x, mean_y, np.asarray(lengths, dtype=np.int32), (
        np.concatenate(xs) if xs else np.zeros(0, np.float32)
    ), (np.concatenate(ys) if ys else np.zeros(0, np.float32))


def read_rollouts(path):
    r = Reader(path)
    rolls = []
    while r.i < len(r.b):
        roll, n, steps = (int(v) for v in r.ints(3))
        ids = r.ints(n)
        types = r.ints(n)
        length = r.floats(n)
        width = r.floats(n)
        m = n * steps
        x = r.floats(m).reshape(n, steps)
        y = r.floats(m).reshape(n, steps)
        heading = r.floats(m).reshape(n, steps)
        rolls.append({
            "roll": roll,
            "ids": ids,
            "types": types,
            "length": length,
            "width": width,
            "x": x,
            "y": y,
            "heading": heading,
        })
    r.done()
    return rolls


def sdc_table():
    raw = np.fromfile(BIN_DIR / "sdc.bin", dtype="<i4")
    assert int(raw[0]) == NMAPS
    return raw[1:]


def json_paths():
    paths = sorted(JSON_DIR.glob("*.json"))
    assert len(paths) == NMAPS
    return paths


def controlled(objs, sdc):
    cand = [
        i for i, o in enumerate(objs)
        if o["type"] in (1, 2, 3) and int(o["valid"][INIT_STEPS]) == 1
    ]
    out = []
    if sdc in cand:
        out.append(sdc)
    for i in cand:
        if i == sdc:
            continue
        if len(out) >= 256:
            break
        out.append(i)
    return out


def load_config():
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).with_name("wosac.ini"))
    return cfg


def hist_params(cfg, name):
    return (
        cfg.getfloat(name, "histogram.min_val"),
        cfg.getfloat(name, "histogram.max_val"),
        cfg.getint(name, "histogram.num_bins"),
        cfg.getfloat(name, "histogram.additive_smoothing_pseudocount"),
        cfg.getboolean(name, "independent_timesteps"),
    )


def _exp_mean(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or not np.any(np.isfinite(v)):
        return np.nan
    return float(np.exp(np.nanmean(v)))


def scene_row(gt, sim, edges, scenario, tracks, cfg, device):
    ids = sim["ids"]
    assert np.array_equal(ids, gt["ids"])
    n = len(ids)
    steps = sim["x"].shape[-1]
    assert steps == SIM_STEPS and gt["x"].shape[-1] == SIM_STEPS
    ref_x = gt["x"][:, None, :]
    ref_y = gt["y"][:, None, :]
    ref_h = gt["heading"][:, None, :]
    ref_v = gt["valid"][:, None, :]
    sim_x = sim["x"][:, None, :] if sim["x"].ndim == 2 else sim["x"]
    sim_y = sim["y"][:, None, :] if sim["y"].ndim == 2 else sim["y"]
    sim_h = sim["heading"][:, None, :] if sim["heading"].ndim == 2 else sim["heading"]
    assert sim_x.shape[1] == ROLLOUTS
    scenario_ids = np.full((n, 1), scenario)
    eval_mask = np.isin(ids, list(tracks))
    if not np.any(eval_mask):
        return None
    length = sim["length"]
    width = sim["width"]
    is_vehicle = (sim["types"] == VEHICLE)

    sx, sy, sh = sim_x[eval_mask], sim_y[eval_mask], sim_h[eval_mask]
    rx, ry, rh = ref_x[eval_mask], ref_y[eval_mask], ref_h[eval_mask]
    rv = ref_v[eval_mask]
    elen, ewid = length[eval_mask], width[eval_mask]
    eveh = is_vehicle[eval_mask]
    esc = scenario_ids[eval_mask]

    sim_k = metrics.compute_kinematic_features(sx, sy, sh)
    ref_k = metrics.compute_kinematic_features(rx, ry, rh)
    speed_ok, accel_ok = metrics.compute_kinematic_validity(rv)
    sim_d, sim_c, sim_ttc = metrics.compute_interaction_features(
        sim_x, sim_y, sim_h, scenario_ids, length, width, eval_mask, device=device)
    ref_d, ref_c, ref_ttc = metrics.compute_interaction_features(
        ref_x, ref_y, ref_h, scenario_ids, length, width, eval_mask,
        device=device, valid=ref_v)
    sim_edge, sim_off = metrics.compute_map_features(
        sx, sy, sh, esc, elen, ewid, edges, device=device)
    ref_edge, ref_off = metrics.compute_map_features(
        rx, ry, rh, esc, elen, ewid, edges, device=device, valid=rv)
    ade, min_ade = metrics.compute_displacement_error(sx, sy, rx, ry, rv)

    logs = {
        "linear_speed": (ref_k[0], sim_k[0], speed_ok[:, 0, :]),
        "linear_acceleration": (ref_k[1], sim_k[1], accel_ok[:, 0, :]),
        "angular_speed": (ref_k[2], sim_k[2], speed_ok[:, 0, :]),
        "angular_acceleration": (ref_k[3], sim_k[3], accel_ok[:, 0, :]),
        "distance_to_nearest_object": (ref_d, sim_d, rv[:, 0, :]),
        "time_to_collision": (ref_ttc, sim_ttc, (rv & eveh[:, None, None])[:, 0, :]),
        "distance_to_road_edge": (ref_edge, sim_edge, rv[:, 0, :]),
    }
    like = {}
    for name, (logged, sampled, valid) in logs.items():
        lo, hi, bins, smooth, indep = hist_params(cfg, name)
        raw = estimators.log_likelihood_estimate_timeseries(
            log_values=logged, sim_values=sampled,
            treat_timesteps_independently=indep, min_val=lo, max_val=hi,
            num_bins=bins, additive_smoothing=smooth, sanity_check=False)
        like["likelihood_" + name] = metrics._reduce_average_with_validity(
            raw, valid, axis=1)

    sim_hit = np.any(np.where(rv, sim_c, False), axis=2)
    ref_hit = np.any(np.where(rv, ref_c, False), axis=2)
    sim_road = np.any(np.where(rv, sim_off, False), axis=2)
    ref_road = np.any(np.where(rv, ref_off, False), axis=2)
    like["likelihood_collision_indication"] = estimators.log_likelihood_estimate_scenario_level(
        log_values=ref_hit[:, 0], sim_values=sim_hit,
        min_val=0.0, max_val=1.0, num_bins=2, use_bernoulli=True)
    like["likelihood_offroad_indication"] = estimators.log_likelihood_estimate_scenario_level(
        log_values=ref_road[:, 0], sim_values=sim_road,
        min_val=0.0, max_val=1.0, num_bins=2, use_bernoulli=True)

    row = {key: _exp_mean(val) for key, val in like.items()}
    if not np.all(np.isfinite(list(row.values()))):
        return None
    row = {key: round(val, 3) for key, val in row.items()}
    row["ade"] = round(float(np.nanmean(ade)), 3)
    row["min_ade"] = round(float(np.nanmean(min_ade)), 3)
    row["num_agents_per_scene"] = int(np.sum(eval_mask))
    row["scenario_id"] = scenario
    return row


def metametric(row, cfg):
    total = 0.0
    for name in FIELDS:
        total += cfg.getfloat(name, "metametric_weight") * row["likelihood_" + name]
    return total


def aggregate(rows, cfg):
    df = pd.DataFrame(rows)
    assert len(df) > 0
    df["realism_meta_score"] = [metametric(r, cfg) for r in rows]
    df["realism_meta_score"] = df["realism_meta_score"].round(3)
    df["kinematic_metrics"] = df[list(KIN)].mean(axis=1)
    df["interactive_metrics"] = df[list(INTER)].mean(axis=1)
    df["map_based_metrics"] = df[list(MAP)].mean(axis=1)
    out = {
        "scenes": int(len(df)),
        "agents": int(df["num_agents_per_scene"].sum()),
        "realism_meta_score": float(df["realism_meta_score"].mean()),
        "kinematic_metrics": float(df["kinematic_metrics"].mean()),
        "interactive_metrics": float(df["interactive_metrics"].mean()),
        "map_based_metrics": float(df["map_based_metrics"].mean()),
        "ade": float(df["ade"].mean()),
        "min_ade": float(df["min_ade"].mean()),
    }
    return out


def _stack(rolls, key):
    return np.stack([roll[key] for roll in rolls], axis=1)


def score_dir(outdir, device):
    cfg = load_config()
    paths = json_paths()
    sdcs = sdc_table()
    rows = []
    outdir = Path(outdir)
    for map_id, jpath in enumerate(paths):
        rolls = read_rollouts(outdir / f"map_{map_id:04d}.bin")
        assert len(rolls) == ROLLOUTS, (map_id, len(rolls))
        ids = rolls[0]["ids"]
        for roll in rolls:
            assert np.array_equal(roll["ids"], ids)
        meta = read_meta(outdir / f"map_{map_id:04d}.meta")
        mean_x, mean_y, lengths, ex, ey = meta
        objs, _roads = read_log(BIN_DIR / f"map_{map_id:03d}.bin")
        doc = json.loads(jpath.read_text())
        tracks = [int(t["track_index"]) for t in doc["metadata"]["tracks_to_predict"]]
        want = controlled(objs, int(sdcs[map_id]))
        assert list(ids) == want, (map_id, list(ids), want)
        sl = slice(INIT_STEPS, INIT_STEPS + SIM_STEPS)
        gt = {
            "ids": ids,
            "x": np.stack([objs[i]["x"][sl] - np.float32(mean_x) for i in ids]),
            "y": np.stack([objs[i]["y"][sl] - np.float32(mean_y) for i in ids]),
            "heading": np.stack([objs[i]["heading"][sl] for i in ids]),
            "valid": np.stack([objs[i]["valid"][sl] for i in ids]).astype(bool),
        }
        sim = {
            "ids": ids,
            "types": rolls[0]["types"],
            "length": rolls[0]["length"],
            "width": rolls[0]["width"],
            "x": _stack(rolls, "x"),
            "y": _stack(rolls, "y"),
            "heading": _stack(rolls, "heading"),
        }
        scenario = str(doc["scenario_id"])
        edges = {
            "x": ex,
            "y": ey,
            "lengths": lengths,
            "scenario_id": np.full(len(lengths), scenario),
        }
        row = scene_row(gt, sim, edges, scenario, tracks, cfg, device)
        if row is not None:
            rows.append(row)
        if map_id % 25 == 0:
            print(f"scored {map_id + 1}/{NMAPS}", flush=True)
    print(f"scored {NMAPS}/{NMAPS}", flush=True)
    return aggregate(rows, cfg)


def score_ground_truth(device):
    cfg = load_config()
    paths = json_paths()
    sdcs = sdc_table()
    rows = []
    for map_id, jpath in enumerate(paths):
        objs, roads = read_log(BIN_DIR / f"map_{map_id:03d}.bin")
        mean_x, mean_y = world_mean(roads)
        doc = json.loads(jpath.read_text())
        tracks = [int(t["track_index"]) for t in doc["metadata"]["tracks_to_predict"]]
        ids = np.asarray(controlled(objs, int(sdcs[map_id])), dtype=np.int32)
        sl = slice(INIT_STEPS, INIT_STEPS + SIM_STEPS)
        x = np.stack([objs[i]["x"][sl] - np.float32(mean_x) for i in ids])
        y = np.stack([objs[i]["y"][sl] - np.float32(mean_y) for i in ids])
        heading = np.stack([objs[i]["heading"][sl] for i in ids])
        valid = np.stack([objs[i]["valid"][sl] for i in ids]).astype(bool)
        length = np.asarray([objs[i]["length"] for i in ids], dtype=np.float32)
        width = np.asarray([objs[i]["width"] for i in ids], dtype=np.float32)
        types = np.asarray([objs[i]["type"] for i in ids], dtype=np.int32)
        gt = {"ids": ids, "x": x, "y": y, "heading": heading, "valid": valid}
        sim = {
            "ids": ids,
            "types": types,
            "length": length,
            "width": width,
            "x": np.repeat(x[:, None, :], ROLLOUTS, axis=1),
            "y": np.repeat(y[:, None, :], ROLLOUTS, axis=1),
            "heading": np.repeat(heading[:, None, :], ROLLOUTS, axis=1),
        }
        edges_x = []
        edges_y = []
        lengths = []
        for typ, xs, ys in roads:
            if typ != 6:
                continue
            xs = np.array(xs, dtype=np.float32, copy=True)
            ys = np.array(ys, dtype=np.float32, copy=True)
            keep = (xs != INVALID) & (ys != INVALID)
            xs[keep] -= np.float32(mean_x)
            ys[keep] -= np.float32(mean_y)
            lengths.append(len(xs))
            edges_x.append(xs)
            edges_y.append(ys)
        scenario = str(doc["scenario_id"])
        edges = {
            "x": np.concatenate(edges_x) if edges_x else np.zeros(0, np.float32),
            "y": np.concatenate(edges_y) if edges_y else np.zeros(0, np.float32),
            "lengths": np.asarray(lengths, dtype=np.int32),
            "scenario_id": np.full(len(lengths), scenario),
        }
        row = scene_row(gt, sim, edges, scenario, tracks, cfg, device)
        if row is not None:
            rows.append(row)
        if map_id % 25 == 0:
            print(f"ground truth {map_id + 1}/{NMAPS}", flush=True)
    print(f"ground truth {NMAPS}/{NMAPS}", flush=True)
    return aggregate(rows, cfg)


def n_rolls(path):
    if not path.exists():
        return 0
    return len(read_rollouts(path))


def rollout(name, ckpt, outdir, rollouts=ROLLOUTS, hidden=None, layers=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    batch = 32
    agents = 256
    log = LOG_DIR / f"wosac_{name}.out"
    for start in range(0, NMAPS, batch):
        n = min(batch, NMAPS - start)
        ready = all(
            n_rolls(outdir / f"map_{m:04d}.bin") >= rollouts
            for m in range(start, start + n)
        )
        if ready:
            print(f"{name} maps {start}-{start + n - 1} done", flush=True)
            continue
        for m in range(start, start + n):
            p = outdir / f"map_{m:04d}.bin"
            if p.exists():
                p.unlink()
        total = n * agents
        cmd = [
            str(PUFFER), "eval", "--headless",
            f"--config={INI}",
            f"--base.load_model_path={ckpt}",
            "--env.control_mode=control_wosac",
            "--env.init_steps=10",
            "--env.goal_radius=2",
            "--env.goal_stop=1",
            "--env.episode_length=81",
            "--env.num_agents=256",
            f"--env.num_maps={n}",
            f"--env.map_offset={start}",
            f"--env.wosac_dir={outdir}",
            "--vec.num_buffers=1",
            "--vec.num_threads=16",
            "--train.horizon=4",
            "--train.minibatch_size=2048",
            f"--base.eval_agents={total}",
            f"--base.eval_episodes={n * rollouts}",
        ]
        if hidden is not None:
            cmd.append(f"--policy.hidden_size={int(hidden)}")
        if layers is not None:
            cmd.append(f"--policy.num_layers={int(layers)}")
        print(f"{name} maps {start}-{start + n - 1}", flush=True)
        with log.open("a") as fh:
            fh.write(" ".join(cmd) + "\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
        if proc.returncode != 0:
            raise SystemExit(f"{name} batch {start} exited {proc.returncode}")
        for m in range(start, start + n):
            got = n_rolls(outdir / f"map_{m:04d}.bin")
            assert got == rollouts, (m, got)


def plot(scores):
    sns.set_theme(style="ticks")
    order = [label for label, _key, _color in METHODS]
    palette = {label: color for label, _key, color in METHODS}
    meta = pd.DataFrame({
        "method": order,
        "score": [scores[key]["realism_meta_score"] for _l, key, _c in METHODS],
    })
    groups = []
    for label, key, _color in METHODS:
        row = scores[key]
        for metric, title in (
            ("kinematic_metrics", "kinematic"),
            ("interactive_metrics", "interactive"),
            ("map_based_metrics", "map"),
        ):
            groups.append({"method": label, "metric": title, "score": row[metric]})
    group_df = pd.DataFrame(groups)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    sns.barplot(
        data=meta, x="method", y="score", hue="method",
        palette=palette, ax=axes[0], legend=False, order=order)
    axes[0].set_title("Realism meta score")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("score")
    axes[0].tick_params(axis="x", rotation=15)
    sns.barplot(
        data=group_df, x="metric", y="score", hue="method",
        palette=palette, ax=axes[1], hue_order=order,
        order=["kinematic", "interactive", "map"])
    axes[1].set_title("Group metrics")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("score")
    axes[1].legend(frameon=False, title="")
    for ax in axes:
        ax.set_ylim(0, 1)
    sns.despine(fig)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160)
    print(FIG_PATH)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cmd == "rollout":
        _name, ckpt, outdir = sys.argv[2:5]
        rollout(_name, ckpt, outdir)
        return
    if cmd == "score":
        print(json.dumps(score_dir(sys.argv[2], device), indent=2))
        return
    if cmd == "ground-truth":
        print(json.dumps(score_ground_truth(device), indent=2))
        return
    if cmd == "plot":
        plot(json.loads(SCORE_PATH.read_text()))
        return
    if cmd != "all":
        raise SystemExit("use rollout, score, ground-truth, plot, or all")
    rollout("free", FREE, LOG_DIR / "wosac_free")
    rollout("anchor", ANCHOR, LOG_DIR / "wosac_anchor")
    scores = {
        "free": score_dir(LOG_DIR / "wosac_free", device),
        "anchor": score_dir(LOG_DIR / "wosac_anchor", device),
        "ground_truth": score_ground_truth(device),
    }
    SCORE_PATH.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))
    plot(scores)


if __name__ == "__main__":
    main()
