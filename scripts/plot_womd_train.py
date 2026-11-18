#!/usr/bin/env python3
"""Plot one anchored WOMD run against one unregularized run.

Unregularized (anchor_coef 0) is black. Regularized is colored. Both
keep the anchor file. loss/anchor is the mean distance to it, including
when the coefficient is zero. loss/kl is the PPO approximate KL.

    python scripts/plot_womd_train.py
    python scripts/plot_womd_train.py A.ini B.ini -o logs/womd/curves.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FREE = ROOT / "logs" / "womd" / "drive" / "womd_free.ini"
ANCHORED = ROOT / "logs" / "womd" / "drive" / "womd_anchor.ini"
FREE_COLOR = "#111111"
ANCHOR_COLOR = "#4C72B0"

CURVES = (
    ("env/score", "score"),
    ("env/episode_return", "episode return"),
    ("env/completion_rate", "completion"),
    ("env/offroad_rate", "offroad"),
    ("env/collision_rate", "collision"),
    ("env/episode_length", "episode length"),
    ("loss/policy", "policy loss"),
    ("loss/value", "value loss"),
    ("loss/entropy", "entropy"),
    ("loss/total", "total loss"),
    ("loss/anchor", "KL to anchor"),
    ("loss/kl", "approx KL"),
)


def parse_ini(path):
    sections = {}
    cur = None
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s[0] in "#;":
            continue
        if s[0] == "[" and s[-1] == "]":
            cur = s[1:-1]
            sections.setdefault(cur, {})
            continue
        if cur is None or "=" not in s:
            continue
        key, val = s.split("=", 1)
        sections[cur][key.strip()] = val.strip()
    return sections


def as_floats(text):
    if not text:
        return []
    return [float(part) for part in text.split(",") if part.strip()]


def is_anchored(sections):
    train = sections.get("train", {})
    path = train.get("anchor", "").strip().strip("'\"")
    if not path or path.lower() == "none":
        return False
    return float(train.get("anchor_coef", "0")) != 0.0


def load_run(path):
    sections = parse_ini(path)
    metrics = sections.get("metrics")
    if not metrics or "agent_steps" not in metrics:
        raise SystemExit(f"{path} has no [metrics] yet")
    steps = as_floats(metrics["agent_steps"])
    series = {}
    for key, _label in CURVES:
        vals = as_floats(metrics.get(key))
        n = min(len(vals), len(steps))
        if n:
            series[key] = (steps[:n], vals[:n])
    return {"anchored": is_anchored(sections), "series": series}


def draw(runs, out):
    sns.set_style("ticks")
    fig, axes = plt.subplots(4, 3, figsize=(12, 11), dpi=120)
    axes = axes.ravel()
    ordered = sorted(runs, key=lambda run: run["anchored"])
    for ax, (key, label) in zip(axes, CURVES):
        for run in ordered:
            pair = run["series"].get(key)
            if not pair:
                continue
            steps, vals = pair
            xs = [s / 1e6 for s in steps]
            color = ANCHOR_COLOR if run["anchored"] else FREE_COLOR
            name = "regularized" if run["anchored"] else "unregularized"
            ax.plot(xs, vals, color=color, lw=1.6, label=name)
        ax.set_title(label)
        ax.set_xlabel("steps (M)")
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="*",
        help="run ini files (default: womd_free.ini and womd_anchor.ini)",
    )
    parser.add_argument(
        "-o",
        default=str(ROOT / "logs" / "womd" / "curves.png"),
        help="png path",
    )
    args = parser.parse_args()
    paths = [Path(p) for p in args.logs] if args.logs else [FREE, ANCHORED]
    draw([load_run(p) for p in paths], args.o)


if __name__ == "__main__":
    main()
