"""Compare every released I24-3D spline against its authored annotations.

The audit uses only annotation rows marked ``manual`` or ``interpolation``;
rows marked ``spline`` are excluded to avoid comparing a spline with samples
derived from itself. Distances are reported in the source unit (feet).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from i24_3d_to_spiced import (
    FEET_TO_METERS,
    _interpolate,
    load_authored_annotation_rows,
    load_road_grid,
)


DEFAULT_THRESHOLDS = {
    "bad_p95_residual_ft": 25.0,
    "bad_wrong_way_excursion_ft": 50.0,
    "bad_added_carriageway_departure_ft": 6.0,
    "review_p95_residual_ft": 10.0,
    "review_wrong_way_excursion_ft": 25.0,
    "review_added_carriageway_departure_ft": 3.0,
}


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _dedupe_spline(raw: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    samples: dict[float, tuple[float, float]] = {}
    for timestamp, x, y in zip(raw["timestamp"], raw["x_position"], raw["y_position"]):
        samples[float(timestamp)] = (float(x), float(y))
    timestamps = sorted(samples)
    return (
        timestamps,
        [samples[timestamp][0] for timestamp in timestamps],
        [samples[timestamp][1] for timestamp in timestamps],
    )


def _carriageway_bounds_ft(road_grid: dict[str, Any], direction: int) -> tuple[float, float]:
    values = [value / FEET_TO_METERS for value in road_grid["boundary_y_m"]]
    gaps = [right - left for left, right in zip(values, values[1:])]
    median_index = max(range(len(gaps)), key=gaps.__getitem__)
    return (values[0], values[median_index]) if direction >= 0 else (
        values[median_index + 1],
        values[-1],
    )


def _outside_distance(y: float, lower: float, upper: float) -> float:
    return max(lower - y, y - upper, 0.0)


def audit_object(
    scene: int,
    raw_spline: dict[str, Any],
    annotations: list[dict[str, Any]],
    road_grid: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    timestamps, xs, ys = _dedupe_spline(raw_spline)
    ordered_annotations = sorted(annotations, key=lambda row: float(row["timestamp"]))
    residuals: list[float] = []
    source_index = 0
    overlapping_rows = 0
    for row in ordered_annotations:
        timestamp = float(row["timestamp"])
        if timestamp < timestamps[0] or timestamp > timestamps[-1]:
            continue
        spline_x, source_index = _interpolate(timestamps, xs, timestamp, source_index)
        spline_y, source_index = _interpolate(timestamps, ys, timestamp, source_index)
        residuals.append(math.hypot(spline_x - float(row["x"]), spline_y - float(row["y"])))
        overlapping_rows += 1

    direction = int(raw_spline["direction"])
    progress = [direction * x for x in xs]
    best_progress = progress[0]
    wrong_way_excursion = 0.0
    for value in progress[1:]:
        wrong_way_excursion = max(wrong_way_excursion, best_progress - value)
        best_progress = max(best_progress, value)

    lower, upper = _carriageway_bounds_ft(road_grid, direction)
    annotation_outside = max(
        (_outside_distance(float(row["y"]), lower, upper) for row in annotations),
        default=0.0,
    )
    spline_outside = max((_outside_distance(y, lower, upper) for y in ys), default=0.0)
    added_departure = max(0.0, spline_outside - annotation_outside)

    p95 = percentile(residuals, 0.95)
    bad_reasons = []
    if p95 > thresholds["bad_p95_residual_ft"]:
        bad_reasons.append("p95_annotation_residual")
    if wrong_way_excursion > thresholds["bad_wrong_way_excursion_ft"]:
        bad_reasons.append("wrong_way_excursion")
    if added_departure > thresholds["bad_added_carriageway_departure_ft"]:
        bad_reasons.append("added_carriageway_departure")

    review_reasons = []
    if not bad_reasons:
        if p95 > thresholds["review_p95_residual_ft"]:
            review_reasons.append("p95_annotation_residual")
        if wrong_way_excursion > thresholds["review_wrong_way_excursion_ft"]:
            review_reasons.append("wrong_way_excursion")
        if added_departure > thresholds["review_added_carriageway_departure_ft"]:
            review_reasons.append("added_carriageway_departure")

    manual_count = sum(row["gen"] == "manual" for row in annotations)
    return {
        "scene": scene,
        "id": int(raw_spline["id"]),
        "class": str(raw_spline.get("class", "vehicle")),
        "direction": direction,
        "status": "bad" if bad_reasons else ("review" if review_reasons else "good"),
        "bad_reasons": bad_reasons,
        "review_reasons": review_reasons,
        "annotation_count": len(annotations),
        "manual_count": manual_count,
        "interpolation_count": len(annotations) - manual_count,
        "camera_count": len({row["camera"] for row in annotations}),
        "overlapping_annotation_count": overlapping_rows,
        "annotation_time_coverage_fraction": overlapping_rows / len(annotations),
        "median_residual_ft": percentile(residuals, 0.50),
        "p90_residual_ft": percentile(residuals, 0.90),
        "p95_residual_ft": p95,
        "p99_residual_ft": percentile(residuals, 0.99),
        "max_residual_ft": max(residuals, default=float("nan")),
        "wrong_way_excursion_ft": wrong_way_excursion,
        "annotation_carriageway_departure_ft": annotation_outside,
        "spline_carriageway_departure_ft": spline_outside,
        "added_carriageway_departure_ft": added_departure,
        "annotation_start_time": min(float(row["timestamp"]) for row in annotations),
        "annotation_end_time": max(float(row["timestamp"]) for row in annotations),
        "spline_start_time": timestamps[0],
        "spline_end_time": timestamps[-1],
        "annotation_x_min_ft": min(float(row["x"]) for row in annotations),
        "annotation_x_max_ft": max(float(row["x"]) for row in annotations),
        "annotation_y_min_ft": min(float(row["y"]) for row in annotations),
        "annotation_y_max_ft": max(float(row["y"]) for row in annotations),
        "spline_x_min_ft": min(xs),
        "spline_x_max_ft": max(xs),
        "spline_y_min_ft": min(ys),
        "spline_y_max_ft": max(ys),
    }


def audit_dataset(i24_root: Path, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    results = []
    for scene in (1, 2, 3):
        annotations = load_authored_annotation_rows(i24_root, scene)
        spline_path = i24_root / "data" / "spl_obj" / f"scene{scene}_splobj.json"
        with spline_path.open() as file:
            splines = json.load(file)
        road_grid = load_road_grid(i24_root / "data" / "hg" / f"scene{scene}_hg.json")
        for raw_spline in splines:
            object_id = int(raw_spline["id"])
            if object_id not in annotations:
                continue
            results.append(
                audit_object(
                    scene,
                    raw_spline,
                    annotations[object_id],
                    road_grid,
                    thresholds,
                )
            )
    return results


def build_summary(results: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    status_counts = Counter(result["status"] for result in results)
    reason_counts = Counter(
        reason
        for result in results
        for reason in result["bad_reasons"] + result["review_reasons"]
    )
    scene_counts = {
        str(scene): dict(Counter(result["status"] for result in results if result["scene"] == scene))
        for scene in (1, 2, 3)
    }
    return {
        "schema": "i24_3d_spline_annotation_audit_v1",
        "audited_trajectory_count": len(results),
        "thresholds": thresholds,
        "status_counts": dict(status_counts),
        "scene_status_counts": scene_counts,
        "reason_counts": dict(reason_counts),
        "bad_trajectories": [
            {"scene": result["scene"], "id": result["id"], "reasons": result["bad_reasons"]}
            for result in results
            if result["status"] == "bad"
        ],
        "review_trajectories": [
            {"scene": result["scene"], "id": result["id"], "reasons": result["review_reasons"]}
            for result in results
            if result["status"] == "review"
        ],
        "notes": [
            "Spline-derived annotation rows are excluded from the comparison.",
            "A bad trajectory crosses at least one conservative threshold; review is a lower-severity screening tier.",
            "Wrong-way excursion is the largest loss of direction-adjusted longitudinal progress.",
            "Added carriageway departure subtracts any departure already present in the authored annotations.",
        ],
    }


def write_results(output_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "spline_annotation_audit.json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    csv_rows = []
    for result in results:
        row = dict(result)
        row["bad_reasons"] = ";".join(result["bad_reasons"])
        row["review_reasons"] = ";".join(result["review_reasons"])
        csv_rows.append(row)
    with (output_dir / "spline_annotation_audit.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--i24-root", type=Path, default=repo_root.parent / "i24_3d_dataset" / "I24-3D"
    )
    argument_parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "data" / "i24_spline_audit"
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    results = audit_dataset(args.i24_root.resolve(), DEFAULT_THRESHOLDS)
    summary = build_summary(results, DEFAULT_THRESHOLDS)
    write_results(args.output_dir, results, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
