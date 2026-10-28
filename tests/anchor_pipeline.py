"""Wait for the 1000-map figures, fit the drift head, then four RL runs.

Each run is the Town01 self-play trainer for the first batch multiple at
or above 1e9 agent steps. The four runs share checkpoints/bc_womd_1000_drift.npz
and differ only in train.anchor_dist: kl, reverse_kl, wasserstein, drift.
A finished logs/drive/<run_id>.ini is left in place, so a restart skips it.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

FIGURE_PID_FILE = Path("/tmp/bc1000_figures.pid")
FIGURE_LOG = Path("/tmp/bc1000_figures.log")
PLAIN = ROOT / "checkpoints" / "bc_womd_1000.npz"
DRIFT = ROOT / "checkpoints" / "bc_womd_1000_drift.npz"
BATCH = 8192 * 64
EPOCHS = (1_000_000_000 + BATCH - 1) // BATCH
TIMESTEPS = EPOCHS * BATCH
RUNS = (
    ("anchor_kl", "kl"),
    ("anchor_rkl", "reverse_kl"),
    ("anchor_w1", "wasserstein"),
    ("anchor_drift", "drift"),
)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _figure_pid():
    if not FIGURE_PID_FILE.is_file():
        return None
    text = FIGURE_PID_FILE.read_text().strip()
    if not text:
        return None
    return int(text)


def wait_figures():
    """Block until the figure log says FIGURES DONE.

    A dead waiter with a failure in the log stops the pipeline. Replacing
    the pid file with a live waiter lets a restarted figure job continue.
    """
    while True:
        text = FIGURE_LOG.read_text(errors="replace") if FIGURE_LOG.is_file() else ""
        if "FIGURES DONE" in text and PLAIN.is_file():
            print("figures done", flush=True)
            return
        pid = _figure_pid()
        if pid is not None and not _alive(pid) and text and "FIGURES DONE" not in text:
            time.sleep(2)
            text = FIGURE_LOG.read_text(errors="replace") if FIGURE_LOG.is_file() else ""
            if "FIGURES DONE" in text and PLAIN.is_file():
                print("figures done", flush=True)
                return
            raise SystemExit(f"figures did not finish: {text[-500:]}")
        time.sleep(30)


def point_demo():
    readme = ROOT / "README.md"
    text = readme.read_text()
    old = "checkpoints/bc_womd_100.npz"
    new = "checkpoints/bc_womd_1000.npz"
    if old in text:
        readme.write_text(text.replace(old, new))
        print(f"readme anchor path -> {new}", flush=True)


def point_train_anchor():
    path = ROOT / "config" / "drive.ini"
    text = path.read_text()
    old = "anchor = checkpoints/bc_womd_100.npz"
    new = "anchor = checkpoints/bc_womd_1000_drift.npz"
    if old in text:
        path.write_text(text.replace(old, new, 1))
        print(f"drive.ini {new}", flush=True)


def fit():
    report = ROOT / "analysis" / "offline" / "bc_1000_drift.json"
    if DRIFT.is_file() and report.is_file():
        print(f"drift checkpoint exists {DRIFT}", flush=True)
        return
    import drift_critic
    drift_critic.fit_drift_head(1000)
    if not DRIFT.is_file():
        raise SystemExit("fit did not write the drift checkpoint")


def report_drift():
    path = ROOT / "analysis" / "offline" / "bc_1000_drift.json"
    if not path.is_file():
        print("DRIFT HEAD NO REPORT json missing", flush=True)
        return
    summary = json.loads(path.read_text())
    ratio = summary["holdout_beta_over_beta0"]
    med = float(ratio["q50"])
    hold = float(summary["holdout_mse"])
    mean_mse = float(summary["holdout_mse_predict_train_mean"])
    pearson = float(summary["holdout_pearson"])
    print(
        f"drift head holdout mse {hold:.4f} mean-predictor {mean_mse:.4f} "
        f"r {pearson:.3f} median beta/beta0 {med:.4f}",
        flush=True,
    )
    if med < 0.05 or med > 0.95:
        print(
            f"DRIFT BETA DEGENERATE median beta/beta0 {med:.4f} "
            f"q10 {float(ratio['q10']):.4f} q90 {float(ratio['q90']):.4f} "
            f"alpha {summary['alpha']}. Runs continue with this alpha.",
            flush=True,
        )
    if hold >= mean_mse:
        print(
            f"DRIFT HEAD NO BETTER THAN MEAN holdout mse {hold:.4f} "
            f"mean-predictor {mean_mse:.4f}",
            flush=True,
        )


def baseline():
    out = ROOT / "analysis" / "offline" / "anchor_town01_baseline.json"
    if out.is_file():
        print(f"baseline exists {out}", flush=True)
        return
    import drift_critic
    drift_critic.town01_anchor_baseline(1000)


def replay():
    out = ROOT / "analysis" / "offline" / "bc_1000_replay.json"
    if out.is_file():
        print(f"replay exists {out}", flush=True)
        return
    import drift_critic
    drift_critic.eval_anchor_replay(1000)


def build():
    subprocess.check_call(["./build.sh", "drive"], cwd=ROOT)
    print("built ./puffer", flush=True)


def _finished(run_id):
    path = ROOT / "logs" / "drive" / f"{run_id}.ini"
    if not path.is_file():
        return False
    return "\n[metrics]\n" in path.read_text(errors="replace")


def train_one(run_id, dist):
    if _finished(run_id):
        print(f"skip finished {run_id}", flush=True)
        return
    cmd = [
        str(ROOT / "puffer"), "train", "--headless",
        f"--base.run_id={run_id}",
        f"--train.total_timesteps={TIMESTEPS}",
        f"--train.anchor={DRIFT}",
        f"--train.anchor_dist={dist}",
        "--train.anchor_coef=0.075",
        "--train.anchor_drift_alpha=0.05",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)
    if not _finished(run_id):
        raise SystemExit(f"{run_id} exited without a metrics log")


def plot():
    import plot_anchor_rl
    plot_anchor_rl.plot()


def main():
    os.chdir(ROOT)
    wait_figures()
    point_demo()
    fit()
    report_drift()
    point_train_anchor()
    baseline()
    build()
    for run_id, dist in RUNS:
        train_one(run_id, dist)
    plot()
    replay()
    print("PIPELINE DONE", flush=True)
    print(json.dumps({"timesteps": TIMESTEPS, "epochs": EPOCHS}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("PIPELINE FAILED", flush=True)
        raise
