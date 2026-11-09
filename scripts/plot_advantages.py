#!/usr/bin/env python3
"""Compare advantage histograms at the same rollouts, with episode return.

    python scripts/plot_advantages.py logs/drive/density-advantages.png \
        logs/drive/density64-advantages.jsonl logs/drive/density64.ini 64 \
        logs/drive/density128-advantages.jsonl logs/drive/density128.ini 128 \
        logs/drive/density256-advantages.jsonl logs/drive/density256.ini 256
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import seaborn as sns


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def metric_series(text, key):
    prefix = key + " ="
    for line in text.splitlines():
        if line.startswith(prefix):
            return [
                float(part)
                for part in line.split("=", 1)[1].split(",")
                if part.strip()
            ]
    return None


def load_return(path):
    text = Path(path).read_text(errors="replace")
    steps = metric_series(text, "agent_steps")
    returns = metric_series(text, "env/episode_return")
    if not steps or not returns or len(steps) != len(returns):
        raise SystemExit(f"missing agent_steps or episode return in {path}")
    xs = []
    values = []
    for step, y in zip(steps, returns):
        if xs and step == xs[-1]:
            values[-1] = y
            continue
        xs.append(step)
        values.append(y)
    return np.asarray(xs), np.asarray(values)


# Gigaflow drops |A| < 0.01 * max|A| on the rollout (arXiv:2502.03349, Alg. 1).
# The logged max is this snapshot's max, not the paper's EWMA across iterations.
FILTER_OF_MAX = 0.01


def filter_band(row):
    a_max = max(abs(float(row["min"])), abs(float(row["max"])))
    eta = FILTER_OF_MAX * a_max
    counts = np.asarray(row["counts"], dtype=float)
    lo, hi = float(row["lo"]), float(row["hi"])
    width = (hi - lo) / max(len(counts), 1)
    filtered = 0.0

    def tail(count, left, right):
        span = right - left
        if count <= 0 or span <= 0:
            return 0.0
        inside = min(right, eta) - max(left, -eta)
        if inside <= 0:
            return 0.0
        return count * min(inside, span) / span

    filtered += tail(row["under"], float(row["min"]), lo)
    filtered += tail(row["over"], hi, float(row["max"]))
    for i, count in enumerate(counts):
        left = lo + i * width
        filtered += tail(count, left, left + width)
    return eta, filtered / max(int(row["n"]), 1)


def draw_hist(ax, row, color, label):
    counts = np.asarray(row["counts"], dtype=float)
    lo, hi = float(row["lo"]), float(row["hi"])
    width = (hi - lo) / max(len(counts), 1)
    centers = lo + width * (np.arange(len(counts)) + 0.5)
    frac = counts / max(row["n"], 1)
    ax.bar(
        centers,
        frac,
        width=width * 0.92,
        color=color,
        alpha=0.55,
        linewidth=0,
        label=label,
    )


def interval_abs_mean(left, right):
    """Mean of |x| for x uniform on [left, right]."""
    if right < left:
        left, right = right, left
    span = right - left
    if span == 0.0 or not np.isfinite(span):
        return abs(left)
    if left >= 0.0:
        return 0.5 * (left + right)
    if right <= 0.0:
        return -0.5 * (left + right)
    return (left * left + right * right) / (2.0 * span)


def mean_abs_advantage(row):
    """Mean |advantage| from the logged histogram, including the outer tails."""
    lo, hi = float(row["lo"]), float(row["hi"])
    counts = np.asarray(row["counts"], dtype=float)
    width = (hi - lo) / max(len(counts), 1)
    parts = [(float(row["min"]), lo, float(row["under"]))]
    parts += [
        (lo + i * width, lo + (i + 1) * width, float(count))
        for i, count in enumerate(counts)
    ]
    parts.append((hi, float(row["max"]), float(row["over"])))
    total = 0.0
    weighted = 0.0
    for left, right, count in parts:
        if count <= 0.0 or not np.isfinite(left) or not np.isfinite(right):
            continue
        total += count
        weighted += count * interval_abs_mean(left, right)
    if total <= 0.0:
        return np.nan
    return weighted / total


def window_return_change(steps, values, marks):
    """Δ episode return between consecutive advantage logs.

    Each window starts at a logged rollout and ends at the next log. The last
    window ends at the last logged return, one collection interval later.
    The change is the sample-efficiency score for the snapshot at the start.
    """
    bounds = [float(mark) for mark in marks]
    end = float(steps[-1])
    if end > bounds[-1]:
        bounds.append(end)
    ends = []
    change = []
    for lo, hi in zip(bounds, bounds[1:]):
        if hi <= lo:
            continue
        delta = float(np.interp(hi, steps, values) - np.interp(lo, steps, values))
        ends.append(hi)
        change.append(delta)
    return np.asarray(ends), np.asarray(change)


def parse_series(argv):
    """out.png, then repeating jsonl, ini, label."""
    if len(argv) < 5 or (len(argv) - 2) % 3 != 0:
        raise SystemExit(
            "usage: plot_advantages.py out.png "
            "A.jsonl A.ini Alabel [B.jsonl B.ini Blabel ...]"
        )
    out = argv[1]
    series = []
    rest = argv[2:]
    for i in range(0, len(rest), 3):
        rows = load_jsonl(rest[i])
        steps, values = load_return(rest[i + 1])
        series.append({
            "label": rest[i + 2],
            "by": {int(row["epoch"]): row for row in rows},
            "steps": steps,
            "values": values,
        })
    def order(item):
        try:
            return (0, int(item["label"]))
        except ValueError:
            return (1, item["label"])
    series.sort(key=order)
    return out, series


def series_color(label, index, palette):
    """Keep 128 blue and 256 orange, matching the earlier two-run figure."""
    pinned = {"128": 0, "256": 1, "64": 2}
    if label in pinned:
        return palette[pinned[label]]
    return palette[index % len(palette)]


def snapshot_efficiency(item):
    """Return change over each full advantage-log window this run recorded.

    A run stopped between logs has a short tail after its last snapshot.
    That tail is left off this score. The histogram still shows the snapshot.
    """
    epochs = sorted(item["by"])
    marks = [float(item["by"][epoch]["steps"]) for epoch in epochs]
    ends, change = window_return_change(item["steps"], item["values"], marks)
    n = min(len(epochs), len(change))
    if n > 0 and len(marks) > 1:
        gap = float(np.median(np.diff(marks)))
        if ends[n - 1] - marks[n - 1] < 0.8 * gap:
            n -= 1
    return epochs[:n], ends[:n], change[:n]


def main():
    out, series = parse_series(sys.argv)
    epochs = sorted(set().union(*(item["by"] for item in series)))
    if not epochs:
        raise SystemExit("no rollout epochs")

    sns.set_theme(style="ticks")
    palette = sns.color_palette()
    for index, item in enumerate(series):
        item["color"] = series_color(item["label"], index, palette)
    n = len(epochs)
    fig = plt.figure(figsize=(2.5 * n + 1.6, 7.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.7])
    top = outer[0].subgridspec(1, 3, width_ratios=[1.45, 1.35, 1.05])
    bottom = outer[1].subgridspec(1, n)
    steps_fmt = FuncFormatter(lambda x, _pos: f"{x / 1e9:.1f}B")

    ax = fig.add_subplot(top[0, 0])
    for item in series:
        ax.plot(item["steps"], item["values"], color=item["color"], linewidth=1.8, label=item["label"])
    ax.set_xlabel("steps")
    ax.set_ylabel("episode return")
    ax.set_title("Episode return")
    ax.xaxis.set_major_formatter(steps_fmt)
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)

    # Advantage snapshots are one collection interval apart. Score efficiency
    # on that same interval so each snapshot has one return change.
    gaps = []
    for item in series:
        marks = sorted(float(row["steps"]) for row in item["by"].values())
        if len(marks) > 1:
            gaps.extend(np.diff(marks))
    gap = float(np.median(gaps)) if gaps else 2e8
    gap_m = int(round(gap / 1e6))
    eff_label = f"Δ return / {gap_m}M steps"

    eff = fig.add_subplot(top[0, 2])
    eff.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    scored = {}
    for item in series:
        kept, xs, change = snapshot_efficiency(item)
        scored[item["label"]] = (kept, xs, change)
        eff.plot(xs, change, color=item["color"], linewidth=1.6, marker="o", markersize=3.2, label=item["label"])
    eff.set_xlabel("steps")
    eff.set_ylabel(eff_label)
    eff.set_title("Sample efficiency")
    eff.xaxis.set_major_formatter(steps_fmt)
    eff.grid(alpha=0.3)
    sns.despine(ax=eff)

    # One point per logged rollout: mean |advantage| of that snapshot against
    # the return change over the following collection interval. One regression
    # is fit to every run pooled together.
    corr = fig.add_subplot(top[0, 1], sharey=eff)
    corr.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    pooled_x = []
    pooled_y = []
    for item in series:
        kept, _ends, change = scored[item["label"]]
        mags = np.asarray([mean_abs_advantage(item["by"][epoch]) for epoch in kept])
        y = np.asarray(change[:len(mags)])
        corr.scatter(mags, y, s=36, color=item["color"], label=item["label"], zorder=3)
        pooled_x.append(mags)
        pooled_y.append(y)
    pooled_x = np.concatenate(pooled_x)
    pooled_y = np.concatenate(pooled_y)
    sns.regplot(
        x=pooled_x, y=pooled_y, ax=corr, scatter=False, color="0.2",
        ci=95, truncate=True, line_kws={"linewidth": 1.6, "zorder": 2},
    )
    pearson = float(np.corrcoef(pooled_x, pooled_y)[0, 1])
    corr.text(
        0.04, 0.96, f"r = {pearson:.2f}", transform=corr.transAxes,
        ha="left", va="top", fontsize=11,
    )
    corr.set_xlim(left=0.0)
    corr.set_xlabel("mean |advantage|")
    corr.set_ylabel(eff_label)
    corr.set_title("|advantage| vs efficiency")
    corr.legend(frameon=False, loc="lower right")
    corr.grid(alpha=0.3)
    sns.despine(ax=corr)

    # One advantage scale for every rollout. Each histogram's bins only cover
    # that rollout's 1st–99th percentile, so the shared limits come from the
    # widest of those windows.
    present_rows = [
        item["by"][epoch]
        for epoch in epochs
        for item in series
        if epoch in item["by"]
    ]
    lo = min(float(row["lo"]) for row in present_rows)
    hi = max(float(row["hi"]) for row in present_rows)
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    panels = []
    for i, epoch in enumerate(epochs):
        panel = fig.add_subplot(
            bottom[i],
            sharex=panels[0] if panels else None,
            sharey=panels[0] if panels else None,
        )
        panels.append(panel)
        present = [(item, item["by"][epoch]) for item in series if epoch in item["by"]]
        for item, row in present:
            draw_hist(panel, row, item["color"], item["label"])
        for item, row in present:
            eta, _frac = filter_band(row)
            panel.axvline(eta, color=item["color"], linestyle="--", linewidth=1.0)
            panel.axvline(-eta, color=item["color"], linestyle="--", linewidth=1.0)
        panel.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
        for j, (item, row) in enumerate(present):
            _eta, frac = filter_band(row)
            panel.text(
                0.98, 0.98 - j * 0.13, f"{item['label']}  {100 * frac:.0f}% out",
                transform=panel.transAxes, ha="right", va="top", fontsize=7.5,
                color=item["color"],
            )
        steps = present[0][1]["steps"] / 1e6
        panel.set_title(f"rollout {epoch}\n{steps:.1f}M steps")
        panel.grid(alpha=0.3)
        panel.set_xlabel("advantage")
        if i == 0:
            panel.set_ylabel("fraction of transitions")
            panel.legend(frameon=False)
        else:
            panel.tick_params(labelleft=False)
        sns.despine(ax=panel)
    panels[0].set_xlim(lo - pad, hi + pad)
    panels[0].xaxis.set_major_locator(MultipleLocator(0.5))
    panels[0].set_autoscalex_on(False)

    fig.savefig(out, dpi=200)
    print(f"wrote {out} rollouts={epochs} series={[item['label'] for item in series]}")


if __name__ == "__main__":
    main()
