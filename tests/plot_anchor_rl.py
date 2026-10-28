"""Compare the four anchor regularizers on Town01 self-play.

Reads logs/drive/<run_id>.ini and analysis/offline/anchor_town01_baseline.json.
The anchor line is the greedy BC policy in that same Town01 env.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_bc_map as t

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("anchor_kl", "KL"),
    ("anchor_rkl", "reverse KL"),
    ("anchor_w1", "Wasserstein"),
    ("anchor_drift", "drift"),
)
METRICS = (
    ("env/score", "score"),
    ("env/collision_rate", "collision rate"),
    ("env/offroad_rate", "off-road rate"),
)
BASELINE_KEYS = {
    "env/score": "score_mean",
    "env/collision_rate": "collision_rate_mean",
    "env/offroad_rate": "offroad_rate_mean",
}


def metric_series(text, key):
    prefix = key + " ="
    for line in text.splitlines():
        if line.startswith(prefix):
            body = line.split("=", 1)[1]
            return [float(part) for part in body.split(",") if part.strip()]
    return None


def load_run(run_id):
    path = ROOT / "logs" / "drive" / f"{run_id}.ini"
    text = path.read_text(errors="replace")
    steps = metric_series(text, "agent_steps")
    if not steps:
        raise SystemExit(f"{path} has no agent_steps")
    series = {}
    for key, _label in METRICS:
        values = metric_series(text, key)
        if not values or len(values) != len(steps):
            raise SystemExit(f"{path} missing {key}")
        series[key] = values
    anchor = None
    for line in text.splitlines():
        if line.startswith("anchor ="):
            anchor = line.split("=", 1)[1].strip()
            break
    return {"steps": steps, "series": series, "anchor": anchor, "path": str(path)}


def plot():
    t.apply_figure_style()
    colors = sns.color_palette(n_colors=len(RUNS))
    loaded = [load_run(run_id) for run_id, _label in RUNS]
    anchors = {row["anchor"] for row in loaded}
    if len(anchors) != 1:
        raise SystemExit(f"runs do not share one anchor: {anchors}")
    baseline_path = t.OUT / "anchor_town01_baseline.json"
    baseline = None
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True)
    for ax, (key, label) in zip(axes, METRICS):
        for color, (_run_id, name), row in zip(colors, RUNS, loaded):
            ax.plot(row["steps"], row["series"][key], color=color, lw=1.4, label=name)
        if baseline is not None:
            ax.axhline(
                baseline[BASELINE_KEYS[key]], color="black", ls=(0, (4, 2)),
                lw=1.2, label="anchor",
            )
        ax.set_xlabel("agent steps")
        ax.set_ylabel(label)
        t.style_ax(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Town01 self-play. Same anchor, four regularizers. "
        "The dashed line is the greedy anchor in that env."
    )
    out = t.OUT / "anchor_rl_selfplay.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    final = []
    for (_run_id, name), row in zip(RUNS, loaded):
        point = {"name": name, "steps": row["steps"][-1]}
        for key, label in METRICS:
            point[label] = row["series"][key][-1]
        final.append(point)
    summary = {
        "anchor": next(iter(anchors)),
        "baseline": baseline,
        "final": final,
    }
    out_json = t.OUT / "anchor_rl_selfplay.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    plot()
