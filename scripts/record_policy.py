#!/usr/bin/env python3
"""Record a short headless video of a drive policy.

The frame is split. The left panel is a bird's-eye view of the scene. The
right panel is one agent's point of view (env.selected_agent_idx, or --agent).
The top right is that agent's observation: road points colored by type,
its blue bounding box and heading arrow, and partner boxes in view.
The clip length is base.record_frames when that key is set, otherwise
one training episode (env.episode_length). Drive recordings are 200
steps. The training episode length is left unchanged. The step size
is 0.1 s.

Uses ./drive, the CPU player, so the recording does not take the training GPU.
The trainer calls this when it saves a checkpoint and base.record_rollout_every
is not 0. On its own it uses the newest checkpoints/**/*.bin.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENV_KEYS = (
    "num_agents",
    "num_bots",
    "selected_agent_idx",
    "map_id",
    "data_dir",
    "reward_vehicle_collision",
    "reward_offroad_collision",
    "reward_goal",
    "reward_jerk",
    "target_speed",
    "speed_band",
    "goal_distance_min",
    "goal_distance_max",
)
POLICY_KEYS = ("hidden_size", "num_layers")


def parse_ini(path):
    sections = {}
    current = None
    with open(path) as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                sections.setdefault(current, {})
                continue
            if current is None or "=" not in line:
                continue
            key, value = line.split("=", 1)
            sections[current][key.strip()] = value.strip()
    return sections


def newest_checkpoint(root):
    best = None
    best_mtime = -1.0
    if not root.is_dir():
        return None
    for path in root.rglob("*.bin"):
        if not path.is_file():
            continue
        mtime = path.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best = path
    return best


def resolve(path):
    if path is None:
        return None
    if path.is_absolute():
        return path
    return ROOT / path


def episode_steps(raw):
    """Ini values are floats. A missing or bad value stays at 500."""
    if raw is None or raw == "":
        return 500
    try:
        steps = int(float(raw))
    except ValueError:
        return 500
    if steps < 1:
        return 500
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ini", type=Path, help="run log ini from the end of training")
    parser.add_argument("--checkpoint", type=Path, help="weights .bin")
    parser.add_argument("--out", type=Path, help="mp4 path; default is next to the ini")
    parser.add_argument("--agent", type=int, default=None, help="selected_agent_idx")
    parser.add_argument(
        "--frames", type=int, default=None,
        help="frames to store; default is one episode",
    )
    parser.add_argument(
        "--episode-length", type=int, default=None,
        help="override env.episode_length from the ini",
    )
    parser.add_argument("--drive", type=Path, default=ROOT / "drive")
    args = parser.parse_args()

    ini_path = resolve(args.ini)
    sections = {}
    if ini_path is not None:
        if not ini_path.is_file():
            sys.stderr.write(f"record: ini not found: {ini_path}\n")
            return 1
        sections = parse_ini(ini_path)

    checkpoint = resolve(args.checkpoint)
    if checkpoint is None:
        load_path = sections.get("base", {}).get("load_model_path", "")
        if load_path and load_path != "None":
            checkpoint = resolve(Path(load_path))
        if checkpoint is None or not checkpoint.is_file():
            checkpoint = newest_checkpoint(ROOT / "checkpoints")
    if checkpoint is None or not checkpoint.is_file():
        sys.stderr.write("record: no checkpoint found\n")
        return 1

    if args.out is not None:
        out = resolve(args.out)
    elif ini_path is not None:
        run_id = sections.get("base", {}).get("run_id") or checkpoint.parent.name
        out = ini_path.parent / f"{run_id}.mp4"
    else:
        out = ROOT / "logs" / "drive" / f"{checkpoint.parent.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    drive = args.drive if args.drive.is_absolute() else ROOT / args.drive
    if not drive.is_file():
        sys.stderr.write(
            f"record: missing {drive}; build with ./build.sh drive ./drive --cpu\n"
        )
        return 1

    env_section = sections.get("env", {})
    episode_length = args.episode_length
    if episode_length is None:
        episode_length = episode_steps(env_section.get("episode_length"))
    # --frames wins. Otherwise base.record_frames, so a clip can be
    # shorter than the training episode.
    raw_frames = sections.get("base", {}).get("record_frames")
    if args.frames is not None:
        frames = args.frames
    elif raw_frames not in (None, ""):
        frames = episode_steps(raw_frames)
    else:
        frames = episode_length
    command = [
        str(drive),
        str(checkpoint),
        "--record",
        str(out),
        "--record-frames",
        str(frames),
    ]
    for key in ENV_KEYS:
        if key in env_section:
            command.append(f"--env.{key}={env_section[key]}")
    if args.agent is not None:
        command.append(f"--env.selected_agent_idx={args.agent}")
    policy = sections.get("policy", {})
    for key in POLICY_KEYS:
        if key in policy:
            command.append(f"--policy.{key}={policy[key]}")
    command.append(f"--env.episode_length={episode_length}")

    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    sys.stderr.write("record: " + " ".join(command) + "\n")
    sys.stderr.flush()
    proc = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True
    )
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        sys.stderr.write((proc.stderr or "")[-4000:])
        sys.stderr.write((proc.stdout or "")[-2000:])
        sys.stderr.write(f"record: failed ({proc.returncode}) {out}\n")
        return proc.returncode or 1
    sys.stderr.write(f"wrote {out} ({out.stat().st_size} bytes)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
