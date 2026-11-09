#!/usr/bin/env python3
"""Plot training curves and the advantage distribution.

Reads finished run logs written by ./puffer. [metrics] holds the binned
episode return, PPO losses, mean absolute advantage, and advantage
variance. [advantage] holds a histogram of the raw advantages every few
dozen rollouts. Bin 0 is everything below lo, and the last bin is
everything at or above hi.

    python scripts/plot_train.py logs/drive -o logs/drive/train.png
    python scripts/plot_train.py logs/drive --watch --every 2

--watch redraws after every N newly finished runs. A run counts once
its ini contains [metrics].
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

CURVES = (
    ("env/episode_return", "episode return"),
    ("loss/policy", "policy loss"),
    ("loss/value", "value loss"),
    ("loss/entropy", "entropy"),
    ("loss/total", "total loss"),
    ("loss/kl", "approx kl"),
    ("loss/clipfrac", "clip fraction"),
    ("loss/anchor", "anchor"),
    ("adv/mean_abs", "mean |advantage|"),
)


def parse_ini(path):
    sections = {}
    cur = None
    text = Path(path).read_text(errors="replace")
    for line in text.splitlines():
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


def load_runs(log_dir):
    runs = []
    for path in sorted(Path(log_dir).glob("*.ini")):
        sections = parse_ini(path)
        metrics = sections.get("metrics")
        if not metrics or "agent_steps" not in metrics:
            continue
        steps = as_floats(metrics.get("agent_steps"))
        if not steps:
            continue
        base = sections.get("base", {})
        run_id = base.get("run_id") or path.stem
        series = {}
        for key, _label in CURVES:
            vals = as_floats(metrics.get(key))
            if len(vals) == len(steps):
                series[key] = vals
            elif vals:
                n = min(len(vals), len(steps))
                series[key] = vals[:n]
        adv = sections.get("advantage", {})
        adv_steps = as_floats(adv.get("step"))
        hist = None
        if adv_steps:
            bins = int(float(adv.get("bins", "0") or 0))
            cols = []
            for b in range(bins):
                col = as_floats(adv.get(f"bin_{b}"))
                if len(col) != len(adv_steps):
                    cols = []
                    break
                cols.append(col)
            if cols:
                hist = np.asarray(cols, dtype=float)
        runs.append({
            "id": run_id,
            "path": path,
            "steps": np.asarray(steps, dtype=float),
            "series": series,
            "adv_steps": np.asarray(adv_steps, dtype=float),
            "adv_abs": np.asarray(as_floats(adv.get("mean_abs")), dtype=float),
            "adv_var": np.asarray(as_floats(adv.get("var")), dtype=float),
            "lo": float(adv.get("lo", "-32") or -32),
            "hi": float(adv.get("hi", "32") or 32),
            "hist": hist,
            "mtime": path.stat().st_mtime,
        })
    return runs


def align(steps, values):
    n = min(len(steps), len(values))
    return steps[:n], values[:n]


def draw(runs, out):
    sns.set_style("ticks")
    n = len(runs)
    palette = sns.color_palette("deep", n if n else 1)
    fig = plt.figure(figsize=(12, 13), dpi=100)
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]
    var_ax = fig.add_subplot(gs[3, 0])
    heat_ax = fig.add_subplot(gs[3, 1:])

    for ax, (key, label) in zip(axes, CURVES):
        drew = False
        for run, color in zip(runs, palette):
            if key == "adv/mean_abs" and len(run["adv_abs"]) == len(run["adv_steps"]) and len(run["adv_abs"]):
                xs, ys = run["adv_steps"], run["adv_abs"]
            else:
                vals = run["series"].get(key)
                if vals is None:
                    continue
                xs, ys = align(run["steps"], vals)
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, color=color, lw=1.4)
            drew = True
        ax.set_title(label)
        ax.set_xlabel("steps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        if drew:
            sns.despine(ax=ax)
        else:
            ax.set_axis_off()

    for run, color in zip(runs, palette):
        xs, ys = align(run["adv_steps"], run["adv_var"])
        if len(xs) == 0:
            vals = run["series"].get("adv/var")
            if vals is None:
                continue
            xs, ys = align(run["steps"], vals)
        if len(xs) == 0:
            continue
        var_ax.plot(xs, ys, color=color, lw=1.4)
    var_ax.set_title("advantage variance")
    var_ax.set_xlabel("steps")
    var_ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    var_ax.grid(True, alpha=0.3)
    sns.despine(ax=var_ax)

    latest = None
    for run in runs:
        if run["hist"] is None:
            continue
        if latest is None or run["mtime"] >= latest["mtime"]:
            latest = run
    if latest is None:
        heat_ax.set_axis_off()
        heat_ax.set_title("advantage distribution")
    else:
        hist = latest["hist"]
        frac = hist / np.maximum(hist.sum(axis=0, keepdims=True), 1.0)
        sns.heatmap(
            frac,
            ax=heat_ax,
            cmap="mako",
            cbar_kws={"label": "fraction"},
            xticklabels=False,
            yticklabels=False,
        )
        bins = hist.shape[0]
        lo = latest["lo"]
        hi = latest["hi"]
        width = (hi - lo) / max(bins - 2, 1)
        ticks = [0, bins // 4, bins // 2, 3 * bins // 4, bins - 1]
        labels = []
        for t in ticks:
            if t == 0:
                labels.append(f"< {lo:g}")
            elif t == bins - 1:
                labels.append(f">= {hi:g}")
            else:
                labels.append(f"{lo + (t - 1) * width:g}")
        heat_ax.set_yticks([t + 0.5 for t in ticks])
        heat_ax.set_yticklabels(labels, rotation=0)
        heat_ax.set_ylabel("advantage")
        heat_ax.set_xlabel("rollout window")
        heat_ax.set_title(f"advantage distribution\n{latest['id']}", pad=8)

    fig.suptitle(f"training  ({n} runs)")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} runs={n}", flush=True)


def finished_count(log_dir):
    return len(load_runs(log_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=str(ROOT / "logs" / "drive"),
        help="directory of run ini files",
    )
    parser.add_argument(
        "-o",
        default=None,
        help="png path (default: <log_dir>/train.png)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=2,
        help="with --watch, redraw after this many new finished runs",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep redrawing as runs finish",
    )
    args = parser.parse_args()
    out = args.o or str(Path(args.log_dir) / "train.png")
    if not args.watch:
        runs = load_runs(args.log_dir)
        if not runs:
            print(f"no finished runs in {args.log_dir}", flush=True)
            return
        draw(runs, out)
        return

    plotted = 0
    while True:
        n = finished_count(args.log_dir)
        if n >= plotted + args.every and n > 0:
            draw(load_runs(args.log_dir), out)
            plotted = n
        time.sleep(30)


if __name__ == "__main__":
    main()
