#!/usr/bin/env python3
"""Evaluate whole-scene I-24 checkpoint speed fields across sampled rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROLLOUT_IMPLEMENTATION_FILES = (
    ROOT / "eval" / "macro_metrics" / "build_runtime.sh",
    ROOT / "eval" / "macro_metrics" / "rollout.py",
    ROOT / "eval" / "macro_metrics" / "runtime.py",
    ROOT / "eval" / "dreal_highway" / "policy_act.c",
    ROOT / "ocean" / "drive" / "constants.h",
    ROOT / "ocean" / "drive" / "drive.h",
    ROOT / "ocean" / "drive" / "offline.h",
    ROOT / "ocean" / "drive" / "offline_api.c",
    ROOT / "src" / "puffercpu.c",
)

from eval.macro_metrics.full_scene import generate_full_scene_dataset
from eval.macro_metrics.field_evaluation import (
    MODES,
    SAMPLE_HZ,
    trajectory_fields,
)
from eval.macro_metrics.rollout import ground_truth_arrays, load_scenario
from eval.macro_metrics.runtime import (
    I24_POLICY_HIDDEN,
    I24_POLICY_LAYERS,
)
from eval.macro_metrics.rollout_worker import (
    RolloutJob,
    run_rollout_job,
)
from eval.macro_metrics.speed_field import field_grid


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], int(row["scene"]), row["mode"])].append(
            float(row["mape_percent"])
        )
    output = []
    for (checkpoint, scene, mode), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "checkpoint": checkpoint,
                "scene": scene,
                "mode": mode,
                "rollouts": len(array),
                "mean_mape_percent": float(np.nanmean(array)),
                "std_mape_percent": float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0,
                "median_mape_percent": float(np.nanmedian(array)),
                "p05_mape_percent": float(np.nanquantile(array, 0.05)),
                "p95_mape_percent": float(np.nanquantile(array, 0.95)),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "checkpoint", "scene", "rollout", "mode", "mape_percent",
        "stabilized_mape_percent", "mae_mps", "rmse_mps", "bias_mps",
        "common_cells", "mape_cells", "zero_gt_cells", "gt_support_cells",
        "sim_support_cells", "missing_support_cells", "extra_support_cells",
        "support_recall", "support_precision", "stabilized_floor_mps",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _ensure_runtime(args) -> None:
    if args.rebuild_runtime or not args.library.is_file() or not args.policy_act.is_file():
        subprocess.run([str(ROOT / "eval" / "macro_metrics" / "build_runtime.sh")], check=True)
    if not args.library.is_file() or not args.policy_act.is_file():
        raise FileNotFoundError("Macro runtime build did not produce both required files")


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _worker_count(requested: int, rollouts: int) -> int:
    if requested < 0:
        raise ValueError("workers must be nonnegative")
    available = os.cpu_count() or 1
    return min(rollouts, requested or available)


def _rollout_seed(base_seed: int, rollout: int) -> int:
    if base_seed < 0 or rollout < 0:
        raise ValueError("seed and rollout must be nonnegative")
    return base_seed + rollout


def _run_rollout_jobs(jobs: list[RolloutJob], workers: int) -> list[dict[str, Any]]:
    """Run jobs in isolated processes and return rollout-ordered results."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    results = []
    if workers == 1:
        iterator = ((job, run_rollout_job(job)) for job in jobs)
        for job, result in iterator:
            results.append(result)
            source = "resumed" if result["reused"] else "simulated"
            print(
                f"scene {job.scene} {job.checkpoint_name} r{job.rollout:02d} "
                f"{source} in {result['elapsed_seconds']:.1f} s",
                flush=True,
            )
        return results

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        pending = {executor.submit(run_rollout_job, job): job for job in jobs}
        try:
            for future in as_completed(pending):
                job = pending[future]
                result = future.result()
                results.append(result)
                source = "resumed" if result["reused"] else "simulated"
                print(
                    f"scene {job.scene} {job.checkpoint_name} r{job.rollout:02d} "
                    f"{source} in {result['elapsed_seconds']:.1f} s",
                    flush=True,
                )
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return sorted(results, key=lambda result: int(result["rollout"]))


def evaluate(args) -> dict[str, Any]:
    from eval.macro_metrics.plotting import plot_mape_cdfs, plot_representative_fields

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime(args)
    data_dir = args.output_dir / "full_scene_data"
    scenario_paths = generate_full_scene_dataset(
        args.i24_root,
        data_dir,
        scenes=args.scenes,
        exclude_major_flags=not args.include_major_flags,
    )
    all_rows: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}
    archive_records: list[dict[str, Any]] = []
    scene_metadata = []
    workers = _worker_count(args.workers, args.rollouts)
    implementation_fingerprint = _implementation_fingerprint(
        ROLLOUT_IMPLEMENTATION_FILES
    )

    for map_id, scenario_path in enumerate(scenario_paths):
        scenario = load_scenario(scenario_path)
        scene = int(scenario["metadata"]["source_scene"])
        scenario_fingerprint = _file_fingerprint(scenario_path)
        gt = ground_truth_arrays(scenario)
        duration = float(scenario["metadata"]["duration_seconds"])
        grid = field_grid(
            gt["x"], gt["valid"], duration,
            time_bin_seconds=args.time_bin_seconds,
            space_bin_meters=args.space_bin_meters,
        )
        gt_fields = {mode: trajectory_fields(gt, grid, mode) for mode in MODES}
        scene_metadata.append(scenario["metadata"])

        for checkpoint in args.checkpoints:
            checkpoint_name = checkpoint.stem
            checkpoint_fingerprint = _file_fingerprint(checkpoint)
            archive_dir = (
                args.output_dir / "rollouts" / f"scene{scene}" / checkpoint_name
            )
            payload: dict[str, Any] = {
                "time_edges": grid.time_edges,
                "position_edges": grid.position_edges,
            }
            for mode in MODES:
                for direction, suffix in ((1, "eb"), (-1, "wb")):
                    payload[f"gt_{mode}_{suffix}"] = gt_fields[mode][direction].values
                    payload[f"gt_count_{mode}_{suffix}"] = gt_fields[mode][direction].sample_count
            jobs = [
                RolloutJob(
                    scenario_path=scenario_path,
                    data_dir=data_dir,
                    map_id=map_id,
                    checkpoint=checkpoint,
                    checkpoint_name=checkpoint_name,
                    checkpoint_fingerprint=checkpoint_fingerprint,
                    scenario_fingerprint=scenario_fingerprint,
                    implementation_fingerprint=implementation_fingerprint,
                    scene=scene,
                    rollout=rollout_index,
                    seed=_rollout_seed(args.seed, rollout_index),
                    archive_path=archive_dir / f"r{rollout_index:03d}.npz",
                    library=args.library,
                    policy_act=args.policy_act,
                    hidden=args.hidden,
                    layers=args.layers,
                    time_bin_seconds=args.time_bin_seconds,
                    space_bin_meters=args.space_bin_meters,
                    resume=args.resume,
                )
                for rollout_index in range(args.rollouts)
            ]
            results = _run_rollout_jobs(jobs, workers)
            results.sort(key=lambda result: int(result["rollout"]))
            for result in results:
                all_rows.extend(result["rows"])
                archive_records.append(
                    {
                        "checkpoint": checkpoint_name,
                        "scene": scene,
                        "rollout": int(result["rollout"]),
                        "path": result["archive"],
                        "reused": bool(result["reused"]),
                    }
                )
            field_keys = sorted(results[0]["fields"])
            for key in field_keys:
                payload[key] = np.stack([result["fields"][key] for result in results])
            field_path = args.output_dir / f"scene{scene}_{checkpoint_name}_fields.npz"
            np.savez_compressed(field_path, **payload)
            figure_path = args.output_dir / f"scene{scene}_{checkpoint_name}_representative.png"
            chosen = plot_representative_fields(
                payload, all_rows, checkpoint_name, scene, figure_path,
                speed_max_mps=args.speed_max_mps,
            )
            representative[f"scene{scene}:{checkpoint_name}"] = {
                **chosen,
                "ranking_metric": "smoothed pooled velocity MAPE",
                "figure": str(figure_path),
                "fields": str(field_path),
            }

    _write_csv(args.output_dir / "rollout_metrics.csv", all_rows)
    plot_mape_cdfs(all_rows, args.output_dir / "velocity_mape_cdfs.png")
    result = {
        "protocol": {
            "rollouts_per_checkpoint_scene": args.rollouts,
            "sample_hz": SAMPLE_HZ,
            "time_bin_seconds": args.time_bin_seconds,
            "space_bin_meters": args.space_bin_meters,
            "raw": "per-cell average; no interpolation or smoothing",
            "smoothed": {
                "method": "Gaussian convolution of speed sums and sample counts, then divide",
                "time_sigma_seconds": MODES["smoothed"][0],
                "space_sigma_meters": MODES["smoothed"][1],
            },
            "empty_cells": "excluded from velocity error and reported as support misses/extras",
            "mape_zero_speed": "GT cells at <=1e-6 m/s excluded from conventional MAPE",
            "stabilized_mape_floor_mps": 1.0,
            "selection": "best/median/worst ranked by smoothed pooled MAPE; same rollouts shown raw",
            "scene_windows": "Scene 1 source clock clipped to 90 s; Scenes 2 and 3 complete",
            "all_active_vehicles_policy_controlled": True,
            "background_replay_vehicles": 0,
            "policy_hidden_size": args.hidden,
            "policy_num_layers": args.layers,
            "parallel_workers": workers,
            "rollout_seed_rule": "base seed + rollout index",
            "random_seed": args.seed,
        },
        "scenes": scene_metadata,
        "checkpoints": [str(path) for path in args.checkpoints],
        "summary": _summaries(all_rows),
        "representative": representative,
        "rollout_archives": archive_records,
        "rollout_metrics": all_rows,
    }
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(_jsonable(result), file, indent=2)
        file.write("\n")
    return result


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--i24-root", type=Path,
        default=ROOT / "data" / "i24" / "raw" / "i24_3d_dataset" / "I24-3D",
    )
    argument_parser.add_argument(
        "--checkpoints", type=Path, nargs="+",
        default=[
            ROOT / "checkpoints" / "rl_i24_coef0.bin",
            ROOT / "checkpoints" / "rl_i24_coef0p068.bin",
        ],
    )
    argument_parser.add_argument("--scenes", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    argument_parser.add_argument("--rollouts", type=int, default=32)
    argument_parser.add_argument(
        "--workers", type=int, default=0,
        help="parallel rollout processes; 0 uses min(rollouts, available CPUs)",
    )
    argument_parser.add_argument("--seed", type=int, default=1)
    argument_parser.add_argument("--time-bin-seconds", type=float, default=1.0)
    argument_parser.add_argument("--space-bin-meters", type=float, default=15.0)
    argument_parser.add_argument("--speed-max-mps", type=float, default=35.0)
    argument_parser.add_argument("--hidden", type=int, default=I24_POLICY_HIDDEN)
    argument_parser.add_argument("--layers", type=int, default=I24_POLICY_LAYERS)
    argument_parser.add_argument("--include-major-flags", action="store_true")
    argument_parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="rerun and atomically replace existing per-rollout archives",
    )
    argument_parser.set_defaults(resume=True)
    argument_parser.add_argument("--rebuild-runtime", action="store_true")
    argument_parser.add_argument(
        "--library", type=Path, default=ROOT / "build" / "libdrive_offline.so"
    )
    argument_parser.add_argument(
        "--policy-act", type=Path, default=ROOT / "build" / "i24_policy_act"
    )
    argument_parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis" / "macro_metrics" / "i24_velocity_mape",
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    if args.rollouts < 1:
        raise SystemExit("--rollouts must be at least one")
    if args.workers < 0:
        raise SystemExit("--workers must be nonnegative")
    if args.seed < 0:
        raise SystemExit("--seed must be nonnegative")
    missing = [str(path) for path in args.checkpoints if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing checkpoints: {', '.join(missing)}")
    result = evaluate(args)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
