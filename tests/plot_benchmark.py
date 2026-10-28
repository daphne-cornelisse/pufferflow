"""Human replay, Town01 horizon, and the end of training. One figure.

python tests/plot_benchmark.py CSV PNG [title] [train_ini]
The ini is logs/drive/<run>.ini. Its sibling .out supplies anchor loss.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


REPLAY_KEYS = (
    "replay_collision",
    "replay_offroad",
    "replay_completion",
    "replay_score",
    "replay_ade_lat",
    "replay_ade_lon",
)
# Same names share a palette index across the two rate panels.
RATE_COLORS = {
    "score": 0,
    "collision": 1,
    "off-road": 2,
    "completion": 3,
}
TRAIN_KEYS = (
    ("env/score", "score"),
    ("env/collision_rate", "collision"),
    ("env/offroad_rate", "off-road"),
    ("env/completion_rate", "completion"),
)


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def read_report(csv_path):
    report = csv_path.parent / "report.txt"
    out = {}
    if not report.exists():
        return out
    for line in report.read_text().splitlines():
        bits = line.split()
        if not bits:
            continue
        key = bits[0]
        if key in REPLAY_KEYS or key in ("horizon_collision", "horizon_score"):
            out[key] = float(bits[1])
        if key == "replay" and len(bits) >= 5 and bits[1] == "scenes":
            out["scenes"] = int(bits[2])
            out["skipped"] = int(bits[4])
    return out


def read_horizon(csv_path):
    rows = []
    for line in csv_path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        bits = line.split(",")
        score = float(bits[3]) if len(bits) > 3 and bits[3] else None
        rows.append((
            int(bits[0]), int(bits[1]), float(bits[2]), score,
        ))
    return rows


def series_last(text, key):
    prefix = key + " ="
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        parts = [p for p in line.split("=", 1)[1].split(",") if p.strip()]
        if not parts:
            return None
        return float(parts[-1])
    return None


def last_anchor(ini_path):
    out = ini_path.with_suffix(".out")
    if not out.is_file():
        return None
    found = None
    for line in out.read_text(errors="replace").splitlines():
        match = re.search(r"anchor\s+([0-9]+(?:\.[0-9]+)?)", line)
        if match:
            found = float(match.group(1))
    return found


def read_train(ini_path):
    text = ini_path.read_text(errors="replace")
    metrics = {}
    for key, label in TRAIN_KEYS:
        value = series_last(text, key)
        if value is None:
            raise SystemExit(f"{ini_path} missing {key}")
        metrics[label] = value
    metrics["anchor"] = last_anchor(ini_path)
    return metrics


def rate_bars(ax, labels, values, palette):
    colors = [palette[RATE_COLORS[label]] for label in labels]
    ax.bar(labels, values, color=colors, width=0.72)
    ax.set_ylim(0.0, 1.18)
    ax.set_ylabel("rate")
    for i, val in enumerate(values):
        ax.text(
            i, val + 0.03, f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )


def draw_replay(ax, report, palette):
    labels = ["score", "collision", "off-road", "completion"]
    keys = [
        "replay_score", "replay_collision",
        "replay_offroad", "replay_completion",
    ]
    values = [report.get(key, 0.0) for key in keys]
    rate_bars(ax, labels, values, palette)
    scenes = report.get("scenes")
    if scenes is None:
        ax.set_title("human replay")
    else:
        ax.set_title(f"human replay, {scenes} scenes")
    ade_lat = report.get("replay_ade_lat")
    ade_lon = report.get("replay_ade_lon")
    if ade_lat is not None and ade_lon is not None:
        note = f"ADE  lat {ade_lat:.2f} m    lon {ade_lon:.2f} m"
        skipped = report.get("skipped")
        if skipped:
            note = f"{note}    skipped {skipped}"
        ax.set_xlabel(note)
    style_ax(ax)


def draw_horizon(ax, rows, title, palette):
    horizons = sorted({row[0] for row in rows})
    collision = palette[RATE_COLORS["collision"]]
    score_color = palette[RATE_COLORS["score"]]
    means = []
    score_means = []
    scored = False
    for horizon in horizons:
        counts = [row[2] for row in rows if row[0] == horizon]
        ax.scatter(
            [horizon] * len(counts), counts,
            s=36, color=collision, zorder=3,
        )
        means.append(sum(counts) / len(counts))
        scores = [row[3] for row in rows if row[0] == horizon and row[3] is not None]
        if scores:
            scored = True
            score_means.append(sum(scores) / len(scores))
        else:
            score_means.append(None)
    if horizons:
        ax.plot(
            horizons, means, color=collision, linewidth=1.5,
            label="collisions",
        )
        ax.set_xticks(horizons)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("episode length (steps)")
    ax.set_ylabel("collisions per car")
    ax.set_title(f"{title} horizon")
    if scored:
        score_ax = ax.twinx()
        present = [
            (horizon, value)
            for horizon, value in zip(horizons, score_means)
            if value is not None
        ]
        score_ax.plot(
            [item[0] for item in present],
            [item[1] for item in present],
            color=score_color, linewidth=1.5, marker="o",
            label="steps before collision",
        )
        score_ax.set_ylim(0.0, float(max(horizons)))
        score_ax.set_ylabel("steps before collision")
        handles, labels = ax.get_legend_handles_labels()
        extra_handles, extra_labels = score_ax.get_legend_handles_labels()
        ax.legend(
            handles + extra_handles, labels + extra_labels,
            frameon=False, loc="upper left",
        )
        sns.despine(ax=score_ax, left=True, right=False)
    style_ax(ax)


def draw_train(ax, metrics, palette):
    labels = [label for _key, label in TRAIN_KEYS]
    values = [metrics[label] for label in labels]
    rate_bars(ax, labels, values, palette)
    ax.set_title("end of training")
    if metrics["anchor"] is not None:
        ax.set_xlabel(f"anchor loss {metrics['anchor']:.3f}")
    style_ax(ax)


def main():
    csv_path = Path(sys.argv[1])
    png_path = Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else "Town01"
    train = None
    if len(sys.argv) > 4:
        train = read_train(Path(sys.argv[4]))
    sns.set_style("ticks")
    palette = sns.color_palette()
    rows = read_horizon(csv_path)
    report = read_report(csv_path)
    ncols = 3 if train else 2
    fig, axes = plt.subplots(
        1, ncols, figsize=(4.3 * ncols, 4.2), constrained_layout=True,
    )
    draw_replay(axes[0], report, palette)
    draw_horizon(axes[1], rows, title, palette)
    if train:
        draw_train(axes[2], train, palette)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    print(f"wrote {png_path}", flush=True)


if __name__ == "__main__":
    main()
