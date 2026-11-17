#!/usr/bin/env python3
"""Human-replay eval on data/womd_train_1000.

Only the logged self-driving car is controlled. Up to 31 other valid
movers follow their logs (PufferDrive's 32-actor scene). The episode
runs for 110 steps. Arrival removes the car; the clock still
runs out so each map is one trainer episode. The goal radius
is 2.5 m, from the spiced self-play config. A car scores 1 only if it
arrives with no collision and no off-road event.

This is the 1000 training maps, not the paper's 10k held-out test set.

    python scripts/human_replay.py rollout NAME CKPT
    python scripts/human_replay.py all
    python scripts/human_replay.py plot
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "womd"
SCORE_PATH = LOG_DIR / "human_replay.json"
FIG_PATH = LOG_DIR / "human_replay.png"
PUFFER = Path("/tmp/puffer-hr")
INI = ROOT / "config" / "womd_short.ini"
ANCHOR = ROOT / "checkpoints/womd/drive/womd_anchor/0000000499908608.bin"
FREE = ROOT / "checkpoints/womd/drive/womd_free/0000000499908608.bin"
NMAPS = 1000
BATCH = 32
ROLLOUTS = 1
GOAL_RADIUS = 2.5
EPISODE = 110

RATE_KEYS = (
    "score",
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "route_progress",
    "completion_rate",
)
ERROR_KEYS = (
    "lateral_error",
    "longitudinal_error",
)
KEYS = RATE_KEYS + ERROR_KEYS
RUNS = (
    ("free", FREE),
    ("anchor", ANCHOR),
)
COLORS = {
    "free": "#111111",
    "anchor": "#4C72B0",
}
LABELS = {
    "free": "Unregularized",
    "anchor": "Regularized",
}


def load():
    if not SCORE_PATH.exists():
        return {
            "maps": NMAPS,
            "rollouts": ROLLOUTS,
            "goal_radius": GOAL_RADIUS,
            "episode_length": EPISODE,
            "actors": 32,
            "runs": {},
        }
    return json.loads(SCORE_PATH.read_text())


def save(doc):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_PATH.write_text(json.dumps(doc, indent=2) + "\n")


def parse_log(text):
    games = re.findall(r"CUDA_EVAL .*games=(\d+)", text)
    assert games, "eval printed no CUDA_EVAL line"
    out = {"n": int(games[-1])}
    for key in KEYS:
        hits = re.findall(
            rf"(?:^|\s){key}\s+(-?\d+(?:\.\d+)?)", text)
        assert hits, f"missing {key}"
        out[key] = float(hits[-1])
    return out


def aggregate(batches):
    total = 0
    acc = {key: 0.0 for key in KEYS}
    for row in batches.values():
        w = row["n"]
        total += w
        for key in KEYS:
            acc[key] += row[key] * w
    assert total > 0
    out = {key: acc[key] / total for key in KEYS}
    out["n"] = total
    return out


def rollout(name, ckpt, hidden=None, layers=None):
    ckpt = Path(ckpt)
    assert ckpt.is_file(), ckpt
    assert PUFFER.is_file(), PUFFER
    doc = load()
    run = doc["runs"].setdefault(name, {"batches": {}})
    batches = run["batches"]
    log_path = LOG_DIR / f"human_{name}.out"
    starts = list(range(0, NMAPS, BATCH))
    for start in starts:
        key = str(start)
        if key in batches and batches[key].get("n"):
            print(f"{name} maps {start} cached", flush=True)
            continue
        n = min(BATCH, NMAPS - start)
        episodes = n * ROLLOUTS
        cmd = [
            str(PUFFER), "eval", "--headless",
            f"--config={INI}",
            f"--base.load_model_path={ckpt}",
            "--env.control_mode=control_sdc_only",
            "--env.num_agents=1",
            f"--env.episode_length={EPISODE}",
            f"--env.goal_radius={GOAL_RADIUS}",
            "--env.goal_stop=0",
            "--env.init_steps=0",
            f"--env.num_maps={n}",
            f"--env.map_offset={start}",
            "--vec.num_buffers=1",
            "--vec.num_threads=8",
            "--train.horizon=4",
            "--train.minibatch_size=4",
            f"--base.eval_agents={n}",
            f"--base.eval_episodes={episodes}",
        ]
        if hidden is not None:
            cmd.append(f"--policy.hidden_size={int(hidden)}")
        if layers is not None:
            cmd.append(f"--policy.num_layers={int(layers)}")
        print(f"{name} maps {start}-{start + n - 1}", flush=True)
        proc = subprocess.run(
            cmd, cwd=ROOT, text=True, capture_output=True)
        with log_path.open("a") as fh:
            fh.write(proc.stdout)
            fh.write(proc.stderr)
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr)[-2000:]
            raise SystemExit(f"eval failed ({proc.returncode})\n{tail}")
        row = parse_log(proc.stdout + proc.stderr)
        assert row["n"] == episodes, (row["n"], episodes)
        batches[key] = row
        run["metrics"] = aggregate(batches)
        save(doc)
        print(
            f"  score {row['score']:.3f} "
            f"collision {row['collision_rate']:.3f} "
            f"at_fault {row['at_fault_collision_rate']:.3f}",
            flush=True)
    run["metrics"] = aggregate(batches)
    save(doc)
    return run["metrics"]


def plot():
    doc = load()
    rows = []
    for name, _ckpt in RUNS:
        metrics = doc["runs"].get(name, {}).get("metrics")
        if not metrics:
            continue
        for key in KEYS:
            rows.append({
                "run": LABELS[name],
                "key": key,
                "value": metrics[key],
                "color": COLORS[name],
            })
    assert rows, "no human-replay metrics"
    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    rate = frame[frame["key"].isin(RATE_KEYS)]
    err = frame[frame["key"].isin(ERROR_KEYS)]
    hue_order = [LABELS[name] for name, _ckpt in RUNS
        if LABELS[name] in set(rate["run"])]
    palette = {LABELS[name]: COLORS[name] for name, _ckpt in RUNS}
    sns.barplot(
        data=rate, x="key", y="value", hue="run",
        hue_order=hue_order, palette=palette, ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("rate")
    axes[0].set_title("Human replay")
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(
        data=err, x="key", y="value", hue="run",
        hue_order=hue_order, palette=palette, ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("meters")
    axes[1].set_title("Path error")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160)
    print(FIG_PATH)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "rollout":
        assert len(sys.argv) == 4
        metrics = rollout(sys.argv[2], sys.argv[3])
        print(json.dumps(metrics, indent=2))
    elif cmd == "all":
        for name, ckpt in RUNS:
            rollout(name, ckpt)
        plot()
    elif cmd == "plot":
        plot()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
