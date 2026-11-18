#!/usr/bin/env python3
"""WOMD sweep eval. Score only the final checkpoint of a finished run.

Human-replay and WOSAC run after training has exited. The plot is the
standard sweep figure without town scores, plus human-likeness.

    python eval/womd_sweep.py mock
"""

import configparser
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval" / "wosac"))
import plot_sweep  # noqa: E402

OUT = ROOT / "logs" / "womd" / "sweep_eval_mockup.png"
PUFFER = Path("/tmp/puffer-sweep")
LIKE_KEYS = (
    "hr_collision",
    "hr_at_fault",
    "wosac_meta",
    "kinematic",
    "map_metric",
    "interactive",
    "ade",
    "episode_length",
)


def final_checkpoint(folder):
    """Last saved bin. Training writes them in step order."""
    bins = list(Path(folder).glob("*.bin"))
    assert bins, folder
    return max(bins, key=lambda path: int(path.stem))


def mock_frames(n=12):
    """Stand-in trials so the figure can be reviewed before a sweep."""
    rng = np.random.default_rng(1)
    coef = np.concatenate([
        [0.0, 0.0217], rng.uniform(0.002, 0.12, n - 2),
    ])
    z = coef / 0.12
    noise = rng.normal(0, 1, n)
    score = np.clip(0.96 - 0.28 * z + 0.02 * noise, 0.55, 0.99)
    summary = []
    returns = []
    losses = []
    failures = []
    for i in range(n):
        hidden = int(rng.choice([32, 64, 128]))
        layers = int(rng.choice([2, 3, 4]))
        row = {
            "run": i,
            "series_id": f"sweep_mock_{i:04d}",
            "order": i,
            "score": float(score[i]),
            "objective": float(score[i]),
            "objective_name": "score",
            "goal": "maximize",
            "completion_rate": float(np.clip(score[i] + 0.03, 0, 1)),
            "episode_return": float(score[i] * 1.4 - 0.2),
            "jerk": float(abs(0.004 + 0.01 * noise[i])),
            "smooth": float(0.2 + 0.05 * noise[i]),
            "wall_min": float(30 + 8 * z[i] + abs(noise[i])),
            "avg_distance_per_infraction": float(18 + 40 * score[i]),
            "distance_before_first_failure": float(25 + 50 * score[i]),
            "parameters": plot_sweep.policy_params(hidden, layers),
            "anchor_coef": float(coef[i]),
            "hr_collision": float(np.clip(0.11 - 0.45 * z[i] + 0.01 * noise[i], 0.02, 0.2)),
            "hr_at_fault": float(np.clip(0.10 - 0.48 * z[i] + 0.008 * noise[i], 0.01, 0.18)),
            "wosac_meta": float(np.clip(0.62 - 0.08 * z[i] + 0.01 * noise[i], 0.5, 0.7)),
            "kinematic": float(np.clip(0.22 + 0.12 * z[i] + 0.01 * noise[i], 0.15, 0.45)),
            "map_metric": float(np.clip(0.75 - 0.08 * z[i] + 0.01 * noise[i], 0.6, 0.85)),
            "interactive": float(np.clip(0.59 + 0.02 * noise[i], 0.5, 0.7)),
            "ade": float(np.clip(10.5 - 6.0 * z[i] + 0.4 * noise[i], 1.5, 14)),
            "episode_length": float(np.clip(22 + 28 * z[i] + 2 * noise[i], 15, 70)),
            "gamma": float(np.clip(0.99 + 0.005 * noise[i], 0.95, 0.999)),
            "gae_lambda": float(np.clip(0.95 + 0.02 * noise[i], 0.8, 0.99)),
            "learning_rate": float(np.clip(0.003 * (1 + 0.4 * noise[i]), 1e-4, 0.02)),
            "ent_coef": float(np.clip(0.001 * (1 + 0.5 * noise[i]), 1e-5, 0.01)),
            "replay_ratio": 1.0,
            "clip_coef": float(np.clip(0.2 + 0.05 * noise[i], 0.05, 0.5)),
            "vf_coef": float(np.clip(2.0 + 0.3 * noise[i], 0.5, 4)),
            "vf_clip_coef": float(np.clip(0.2 + 0.05 * noise[i], 0.05, 1)),
            "momentum": float(np.clip(0.95 + 0.02 * noise[i], 0.8, 0.99)),
            "max_grad_norm": float(np.clip(1.0 + 0.2 * noise[i], 0.3, 2)),
            "hidden_size": hidden,
            "num_layers": layers,
            "minibatch_size": 8192,
            "horizon": int(rng.choice([16, 32, 64])),
            "total_timesteps": float(rng.choice([5e8, 7.5e8, 1e9])),
        }
        summary.append(row)
        final = row["episode_return"]
        kl = 11.5 - 8.0 * z[i]
        dist = row["distance_before_first_failure"]
        for t in range(1, 9):
            step = t * 6.25e7
            frac = t / 8
            returns.append({
                "run": i,
                "series_id": row["series_id"],
                "steps": step,
                "episode_return": -6 + (final + 6) * frac,
            })
            losses.append({
                "run": i,
                "series_id": row["series_id"],
                "steps": step,
                "anchor_loss": 12 - (12 - kl) * frac,
            })
            failures.append({
                "run": i,
                "series_id": row["series_id"],
                "steps": step,
                "distance_before_first_failure": 8 + (dist - 8) * frac,
            })
    return (
        pd.DataFrame(returns),
        pd.DataFrame(losses),
        pd.DataFrame(),
        pd.DataFrame(failures),
        pd.DataFrame(summary),
    )


def read_ini(path):
    # The trainer writes [metrics] twice: the curve, then the final snapshot.
    cfg = configparser.RawConfigParser(strict=False)
    cfg.read(path)
    return cfg


def score_trial(run_id, ckpt, work, hidden, layers):
    import human_replay as hr
    import score

    work.mkdir(parents=True, exist_ok=True)
    hr.PUFFER = PUFFER
    hr.LOG_DIR = work
    hr.SCORE_PATH = work / "human.json"
    hr.KEYS = tuple(dict.fromkeys(hr.KEYS + ("episode_length",)))
    human = hr.rollout(run_id, ckpt, hidden, layers)
    outdir = Path("/tmp/wosac_sweep") / run_id
    score.PUFFER = PUFFER
    score.LOG_DIR = work
    score.rollout(run_id, ckpt, outdir, hidden=hidden, layers=layers)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wosac = score.score_dir(outdir, device)
    shutil.rmtree(outdir, ignore_errors=True)
    return {
        "hr_collision": human["collision_rate"],
        "hr_at_fault": human["at_fault_collision_rate"],
        "episode_length": human["episode_length"],
        "wosac_meta": wosac["realism_meta_score"],
        "kinematic": wosac["kinematic_metrics"],
        "map_metric": wosac["map_based_metrics"],
        "interactive": wosac["interactive_metrics"],
        "ade": wosac["ade"],
    }


def refresh_panel(ini_path):
    metrics_dir = ini_path.parent
    out = metrics_dir.parent / "sweep-panels.png"
    try:
        frames = plot_sweep.load_frames(metrics_dir.parent / "sweep.log", metrics_dir)
    except SystemExit as err:
        print(err)
        return
    returns, losses, counts, failures, summary, _missing = frames
    summary = summary.copy()
    for key in LIKE_KEYS:
        summary[key] = float("nan")
    for path in metrics_dir.glob("*.likeness.json"):
        row = json.loads(path.read_text())
        hit = summary["series_id"] == path.name[: -len(".likeness.json")]
        for key in LIKE_KEYS:
            summary.loc[hit, key] = row.get(key, float("nan"))
    if summary.empty:
        return
    plot_sweep.draw_womd_eval(returns, losses, counts, failures, summary, out)


def score_ini(ini_path):
    cfg = read_ini(ini_path)
    ckpt = Path(cfg.get("base", "load_model_path"))
    run_id = cfg.get("base", "run_id")
    assert ckpt.is_file(), ckpt
    like_path = ini_path.with_suffix(".likeness.json")
    if like_path.exists():
        return
    # C casts these to int. The eval network has to match the checkpoint.
    hidden = int(float(cfg.get("policy", "hidden_size")))
    layers = int(float(cfg.get("policy", "num_layers")))
    metrics = score_trial(run_id, ckpt, ini_path.parent / run_id, hidden, layers)
    like_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(like_path)


def trial(ini_path):
    ini_path = Path(ini_path)
    score_ini(ini_path)
    # An earlier post can fail after the checkpoint is saved. Score those too
    # while this call still holds the GPU.
    for other in sorted(ini_path.parent.glob("sweep_*.ini")):
        if other != ini_path:
            score_ini(other)
    refresh_panel(ini_path)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "mock":
        frames = mock_frames()
        plot_sweep.draw_womd_eval(*frames, OUT)
        print(OUT)
        return
    if len(sys.argv) == 3 and sys.argv[1] == "trial":
        trial(sys.argv[2])
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
