#!/usr/bin/env python3
"""Plot training curves.

Reads finished run logs written by ./puffer. [metrics] holds the binned
episode return, episode length, PPO losses, collision rate,
completion rate, and advantage variance.

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
    ("env/episode_length", "episode length"),
)

# Bottom row, beside advantage variance.
RATES = (
    ("env/collision_rate", "collision rate"),
    ("env/completion_rate", "completion rate"),
)

# One row, used by --row.
ROW = (
    ("env/completion_rate", "completion rate"),
    ("env/collision_rate", "collision rate"),
    ("env/episode_return", "episode return"),
    ("loss/anchor", "anchor loss"),
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
        train = sections.get("train", {})
        run_id = base.get("run_id") or path.stem
        coef_text = train.get("anchor_coef")
        try:
            anchor_coef = float(coef_text) if coef_text not in (None, "") else None
        except ValueError:
            anchor_coef = None
        series = {}
        for key, _label in CURVES + RATES:
            vals = as_floats(metrics.get(key))
            if len(vals) == len(steps):
                series[key] = vals
            elif vals:
                n = min(len(vals), len(steps))
                series[key] = vals[:n]
        adv = sections.get("advantage", {})
        adv_steps = as_floats(adv.get("step"))
        runs.append({
            "id": run_id,
            "anchor_coef": anchor_coef,
            "path": path,
            "steps": np.asarray(steps, dtype=float),
            "series": series,
            "adv_steps": np.asarray(adv_steps, dtype=float),
            "adv_var": np.asarray(as_floats(adv.get("var")), dtype=float),
            "mtime": path.stat().st_mtime,
        })
    return runs


def align(steps, values):
    n = min(len(steps), len(values))
    return steps[:n], values[:n]


def is_unreg(run):
    coef = run.get("anchor_coef")
    return coef is not None and abs(coef) < 1e-12


def run_label(run):
    coef = run.get("anchor_coef")
    if coef is None:
        return run["id"]
    if is_unreg(run):
        return "unreg"
    return f"{coef:g}"


def run_colors(runs):
    n_reg = sum(0 if is_unreg(run) else 1 for run in runs)
    palette = sns.color_palette("deep", n_reg if n_reg else 1)
    colors = []
    reg_i = 0
    for run in runs:
        if is_unreg(run):
            colors.append("black")
            continue
        colors.append(palette[reg_i])
        reg_i += 1
    return colors


def draw(runs, out):
    sns.set_style("ticks")
    n = len(runs)
    colors = run_colors(runs)
    fig = plt.figure(figsize=(12, 13.4), dpi=100)
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]
    var_ax = fig.add_subplot(gs[3, 0])
    rate_axes = [fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])]

    for ax, (key, label) in zip(axes, CURVES):
        drew = False
        for run, color in zip(runs, colors):
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

    for run, color in zip(runs, colors):
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

    for ax, (key, label) in zip(rate_axes, RATES):
        drew = False
        for run, color in zip(runs, colors):
            vals = run["series"].get(key)
            if not vals:
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

    handles = [
        plt.Line2D([0], [0], color=color, lw=1.4, label=run_label(run))
        for run, color in zip(runs, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(n, 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(f"training  ({n} runs)", y=1.03)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} runs={n}", flush=True)


def draw_row(runs, out, panels):
    sns.set_style("ticks")
    n = len(runs)
    colors = run_colors(runs)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.15), dpi=140)
    if len(panels) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, panels):
        drew = False
        for run, color in zip(runs, colors):
            vals = run["series"].get(key)
            if not vals:
                continue
            xs, ys = align(run["steps"], vals)
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, color=color, lw=1.6)
            drew = True
        ax.set_title(label)
        ax.set_xlabel("steps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        if drew:
            sns.despine(ax=ax)
        else:
            ax.set_axis_off()
    handles = [
        plt.Line2D([0], [0], color=color, lw=1.6, label=run_label(run))
        for run, color in zip(runs, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(n, 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout()
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
    parser.add_argument(
        "--row",
        action="store_true",
        help="one row: completion, collision, return, anchor loss",
    )
    args = parser.parse_args()
    out = args.o or str(Path(args.log_dir) / "train.png")
    if not args.watch:
        runs = load_runs(args.log_dir)
        if not runs:
            print(f"no finished runs in {args.log_dir}", flush=True)
            return
        if args.row:
            draw_row(runs, out, ROW)
        else:
            draw(runs, out)
        return

    plotted = 0
    while True:
        n = finished_count(args.log_dir)
        if n >= plotted + args.every and n > 0:
            runs = load_runs(args.log_dir)
            if args.row:
                draw_row(runs, out, ROW)
            else:
                draw(runs, out)
            plotted = n
        time.sleep(30)


if __name__ == "__main__":
    main()
