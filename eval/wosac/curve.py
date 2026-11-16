#!/usr/bin/env python3
"""WOSAC scores at every checkpoint of the two 1B runs.

    python eval/wosac/curve.py

Rollouts land in /tmp/wosac_curve and are deleted after each score.
A step already present in logs/womd/wosac_curve.json is skipped.
The ground-truth line is the existing 1000-map baseline.
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D

import score

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "womd" / "drive"
LOG_DIR = ROOT / "logs" / "womd"
OUT_PATH = LOG_DIR / "wosac_curve.json"
FIG_PATH = LOG_DIR / "wosac_curve.png"
DUMP = Path("/tmp/wosac_curve")
GT_PATH = LOG_DIR / "wosac_scores.json"
RUNS = (
    ("free", "womd_free_1b", "Unregularized", "#111111"),
    ("anchor", "womd_anchor_1b", "Regularized", "#4C72B0"),
)
METRICS = (
    ("realism_meta_score", "meta"),
    ("kinematic_metrics", "kinematic"),
    ("interactive_metrics", "interactive"),
    ("map_based_metrics", "map"),
)


def load():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    gt = json.loads(GT_PATH.read_text())["ground_truth"]
    return {
        "ground_truth": {key: gt[key] for key, _title in METRICS},
        "runs": {},
    }


def save(doc):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2) + "\n")


def score_ckpt(name, ckpt):
    step = int(ckpt.stem)
    outdir = DUMP / name / f"{step:016d}"
    score.rollout(f"curve_{name}_{step}", ckpt, outdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = score.score_dir(outdir, device)
    shutil.rmtree(outdir)
    return str(step), metrics


def plot(doc):
    rows = []
    for name, _run_id, label, _color in RUNS:
        for step, metrics in doc["runs"].get(name, {}).items():
            for key, title in METRICS:
                rows.append({
                    "run": label,
                    "step": int(step) / 1e6,
                    "metric": title,
                    "score": metrics[key],
                })
    assert rows, "no checkpoint scores"
    frame = pd.DataFrame(rows)
    hue_order = [label for _n, _r, label, _c in RUNS
        if label in set(frame["run"])]
    palette = {label: color for _n, _r, label, color in RUNS}
    order = [title for _key, title in METRICS]
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.3), sharey=True)
    for ax, title in zip(axes, order):
        part = frame[frame["metric"] == title]
        sns.lineplot(
            data=part, x="step", y="score", hue="run",
            hue_order=hue_order, palette=palette,
            marker="o", ax=ax, legend=(ax is axes[0]))
        key = next(k for k, t in METRICS if t == title)
        ax.axhline(doc["ground_truth"][key], color="#55A868", ls="--", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("steps (millions)")
        ax.set_ylabel("score" if ax is axes[0] else "")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#55A868", ls="--"))
    labels.append("Ground truth")
    axes[0].legend(handles, labels, frameon=False, title="")
    sns.despine(fig)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160)
    print(FIG_PATH)


def main():
    doc = load()
    for name, run_id, _label, _color in RUNS:
        bins = sorted((CKPT / run_id).glob("*.bin"))
        assert bins, run_id
        done = doc["runs"].setdefault(name, {})
        for ckpt in bins:
            step = str(int(ckpt.stem))
            if "realism_meta_score" in done.get(step, {}):
                print(f"{name} {step} cached", flush=True)
                continue
            step, metrics = score_ckpt(name, ckpt)
            done[step] = metrics
            save(doc)
            print(
                f"{name} {step} meta {metrics['realism_meta_score']:.3f}",
                flush=True)
    plot(doc)


if __name__ == "__main__":
    main()
