"""Static figures for velocity-MAPE rollout evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


PALETTE = ("#4c72b0", "#dd8452", "#55a868", "#c44e52")


def _style_axis(axis, *, heatmap: bool = False) -> None:
    if heatmap:
        return
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.22, linewidth=0.7)


def plot_mape_cdfs(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Scene-by-field ECDFs; each point is one independently sampled rollout."""
    scenes = sorted({int(row["scene"]) for row in rows})
    modes = ("raw", "smoothed")
    checkpoints = sorted({str(row["checkpoint"]) for row in rows})
    figure, axes = plt.subplots(len(scenes), 2, figsize=(11.5, 3.25 * len(scenes)))
    axes = np.atleast_2d(axes)
    for row_index, scene in enumerate(scenes):
        for column_index, mode in enumerate(modes):
            axis = axes[row_index, column_index]
            for color, checkpoint in zip(PALETTE, checkpoints):
                values = sorted(
                    float(row["mape_percent"])
                    for row in rows
                    if int(row["scene"]) == scene
                    and row["mode"] == mode
                    and row["checkpoint"] == checkpoint
                    and np.isfinite(row["mape_percent"])
                )
                if not values:
                    continue
                y = np.arange(1, len(values) + 1) / len(values)
                axis.step(values, y, where="post", color=color, linewidth=2, label=checkpoint)
            axis.set_title(f"Scene {scene} · {mode}", loc="left", fontweight="semibold")
            axis.set_xlabel("Velocity MAPE (%)")
            axis.set_ylabel("Rollout ECDF")
            axis.set_ylim(0, 1.02)
            _style_axis(axis)
            if row_index == 0 and column_index == 1:
                axis.legend(frameon=False, title="Checkpoint")
    figure.suptitle("Whole-scene velocity MAPE across stochastic rollouts", fontsize=15)
    figure.text(
        0.5,
        0.955,
        "1 s × 15 m cells; raw and sample-weighted Gaussian-smoothed fields scored separately",
        ha="center",
        color="#555555",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _draw_field(axis, values, time_edges, position_edges, norm, title):
    mesh = axis.pcolormesh(
        time_edges,
        position_edges,
        np.ma.masked_invalid(values),
        cmap="viridis",
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    axis.set_title(title, loc="left", fontsize=9.2, fontweight="semibold")
    axis.set_xlim(time_edges[0], time_edges[-1])
    axis.set_ylim(position_edges[0], position_edges[-1])
    _style_axis(axis, heatmap=True)
    return mesh


def _common_cell_absolute_errors(
    payload: dict[str, np.ndarray], mode: str, suffix: str
) -> np.ndarray:
    """Absolute speed errors for every rollout/cell with support in both fields."""
    ground_truth = np.asarray(payload[f"gt_{mode}_{suffix}"], dtype=np.float64)
    simulated = np.asarray(payload[f"sim_{mode}_{suffix}"], dtype=np.float64)
    common = np.isfinite(simulated) & np.isfinite(ground_truth)[None, :, :]
    return np.abs(simulated - ground_truth[None, :, :])[common]


def plot_representative_fields(
    payload: dict[str, np.ndarray],
    metrics: list[dict[str, Any]],
    checkpoint: str,
    scene: int,
    output_path: Path,
    *,
    speed_max_mps: float = 35.0,
) -> dict[str, int]:
    """GT plus best/median/worst rollouts, ranked by smoothed pooled MAPE."""
    scored = sorted(
        (
            (float(row["mape_percent"]), int(row["rollout"]))
            for row in metrics
            if row["checkpoint"] == checkpoint
            and int(row["scene"]) == scene
            and row["mode"] == "smoothed"
            and np.isfinite(row["mape_percent"])
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not scored:
        raise ValueError("No smoothed rollout scores available for representative fields")
    chosen = {
        "best": scored[0][1],
        "median": scored[(len(scored) - 1) // 2][1],
        "worst": scored[-1][1],
    }
    lookup = {
        (row["mode"], int(row["rollout"])): float(row["mape_percent"])
        for row in metrics
        if row["checkpoint"] == checkpoint and int(row["scene"]) == scene
    }
    time_edges = payload["time_edges"]
    position_edges = payload["position_edges"]
    norm = Normalize(0.0, speed_max_mps, clip=True)
    columns = (("raw", "EB", 1), ("raw", "WB", -1), ("smoothed", "EB", 1), ("smoothed", "WB", -1))
    rows = (("Ground truth", None), ("Best", chosen["best"]), ("Median", chosen["median"]), ("Worst", chosen["worst"]))
    figure, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    mesh = None
    for row_index, (row_label, rollout) in enumerate(rows):
        for column_index, (mode, direction_label, direction) in enumerate(columns):
            key_suffix = "eb" if direction == 1 else "wb"
            if rollout is None:
                values = payload[f"gt_{mode}_{key_suffix}"]
                detail = ""
            else:
                values = payload[f"sim_{mode}_{key_suffix}"][rollout]
                detail = f" · r{rollout:02d} · {lookup[(mode, rollout)]:.1f}% pooled"
            mesh = _draw_field(
                axes[row_index, column_index],
                values,
                time_edges,
                position_edges,
                norm,
                f"{row_label} · {mode} {direction_label}{detail}",
            )
            if row_index == 3:
                axes[row_index, column_index].set_xlabel("Time (s)")
            if column_index == 0:
                axes[row_index, column_index].set_ylabel("Longitudinal position (m)")
    assert mesh is not None
    colorbar_axis = figure.add_axes((0.28, 0.035, 0.44, 0.018))
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Mean local speed (m/s)")
    figure.suptitle(
        f"Scene {scene} · {checkpoint} · representative rollout speed fields",
        fontsize=15,
        y=0.985,
    )
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.095, top=0.94, hspace=0.28, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return chosen


def plot_direction_comparison(
    payload: dict[str, np.ndarray],
    metrics: list[dict[str, Any]],
    checkpoint: str,
    scene: int,
    output_path: Path,
    *,
    direction: int,
    speed_max_mps: float = 35.0,
) -> dict[str, int]:
    """Direction-only GT and representative rollout fields.

    With one rollout, show it once. With an ensemble, rank best/median/worst
    using that direction's smoothed MAPE rather than the pooled roadway score.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (westbound) or 1 (eastbound)")
    direction_name = "westbound" if direction == -1 else "eastbound"
    suffix = "wb" if direction == -1 else "eb"
    matching = [
        row
        for row in metrics
        if row["checkpoint"] == checkpoint and int(row["scene"]) == scene
    ]
    scored = sorted(
        (
            (
                float(row["by_direction"][direction_name]["mape_percent"]),
                int(row["rollout"]),
            )
            for row in matching
            if row["mode"] == "smoothed"
            and np.isfinite(row["by_direction"][direction_name]["mape_percent"])
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not scored:
        raise ValueError(f"No {direction_name} smoothed scores are available")
    if len(scored) == 1:
        selected_rows = (("Ground truth", None), ("Rollout 0", scored[0][1]))
        chosen = {"rollout": scored[0][1]}
    else:
        chosen = {
            "best": scored[0][1],
            "median": scored[(len(scored) - 1) // 2][1],
            "worst": scored[-1][1],
        }
        selected_rows = (
            ("Ground truth", None),
            ("Best", chosen["best"]),
            ("Median", chosen["median"]),
            ("Worst", chosen["worst"]),
        )
    lookup = {
        (row["mode"], int(row["rollout"])): float(
            row["by_direction"][direction_name]["mape_percent"]
        )
        for row in matching
    }
    time_edges = payload["time_edges"]
    position_edges = payload["position_edges"]
    norm = Normalize(0.0, speed_max_mps, clip=True)
    figure = plt.figure(figsize=(16.8, 3.35 * len(selected_rows)))
    grid_spec = figure.add_gridspec(
        len(selected_rows), 3, width_ratios=(1.0, 1.0, 0.72)
    )
    axes = np.empty((len(selected_rows), 2), dtype=object)
    for row_index in range(len(selected_rows)):
        for column_index in range(2):
            share_x = axes[0, column_index] if row_index else None
            share_y = axes[row_index, 0] if column_index else None
            axes[row_index, column_index] = figure.add_subplot(
                grid_spec[row_index, column_index], sharex=share_x, sharey=share_y
            )
    diagnostic_spec = grid_spec[:, 2].subgridspec(2, 1, hspace=0.42)
    distribution_axis = figure.add_subplot(diagnostic_spec[0, 0])
    cell_error_axis = figure.add_subplot(diagnostic_spec[1, 0])
    mesh = None
    for row_index, (row_label, rollout) in enumerate(selected_rows):
        for column_index, mode in enumerate(("raw", "smoothed")):
            if rollout is None:
                values = payload[f"gt_{mode}_{suffix}"]
                detail = ""
            else:
                values = payload[f"sim_{mode}_{suffix}"][rollout]
                detail = f" · r{rollout:02d} · {lookup[(mode, rollout)]:.1f}% MAPE"
            mesh = _draw_field(
                axes[row_index, column_index],
                values,
                time_edges,
                position_edges,
                norm,
                f"{row_label} · {mode}{detail}",
            )
            if row_index == len(selected_rows) - 1:
                axes[row_index, column_index].set_xlabel("Time from scene start (s)")
            if column_index == 0:
                axes[row_index, column_index].set_ylabel("Longitudinal position, x (m)")
    assert mesh is not None
    raw_mape = sorted(
        float(row["by_direction"][direction_name]["mape_percent"])
        for row in matching
        if row["mode"] == "raw"
        and np.isfinite(row["by_direction"][direction_name]["mape_percent"])
    )
    smoothed_mape = sorted(
        float(row["by_direction"][direction_name]["mape_percent"])
        for row in matching
        if row["mode"] == "smoothed"
        and np.isfinite(row["by_direction"][direction_name]["mape_percent"])
    )
    for values, label, color in (
        (raw_mape, "Raw", PALETTE[0]),
        (smoothed_mape, "Smoothed", PALETTE[1]),
    ):
        cumulative = np.arange(1, len(values) + 1, dtype=float) / len(values)
        distribution_axis.step(
            values,
            cumulative,
            where="post",
            linewidth=2.2,
            color=color,
            label=f"{label} · median {np.median(values):.1f}%",
        )
        distribution_axis.scatter(
            values,
            cumulative,
            s=18,
            color=color,
            alpha=0.7,
            edgecolors="none",
        )
    distribution_axis.set_title(
        f"{direction_name.capitalize()} MAPE distribution · n={len(smoothed_mape)}",
        loc="left",
        fontsize=9.2,
        fontweight="semibold",
    )
    distribution_axis.set_xlabel("Velocity MAPE (%)")
    distribution_axis.set_ylabel("Rollout ECDF")
    distribution_axis.set_ylim(0.0, 1.02)
    distribution_axis.legend(frameon=False, loc="lower right")
    _style_axis(distribution_axis)

    cell_errors = {
        mode: _common_cell_absolute_errors(payload, mode, suffix)
        for mode in ("raw", "smoothed")
    }
    finite_errors = np.concatenate(
        [values[np.isfinite(values)] for values in cell_errors.values()]
    )
    upper = max(5.0, np.ceil(np.max(finite_errors) / 5.0) * 5.0)
    bins = np.linspace(0.0, upper, 19)
    for mode, label, color in (
        ("raw", "Raw", PALETTE[0]),
        ("smoothed", "Smoothed", PALETTE[1]),
    ):
        values = cell_errors[mode]
        weights = np.full(values.shape, 100.0 / len(values), dtype=np.float64)
        cell_error_axis.hist(
            values,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2.2,
            color=color,
            label=f"{label} · median {np.median(values):.1f} m/s",
        )
    cell_error_axis.set_title(
        "Cell-level absolute speed error",
        loc="left",
        fontsize=9.2,
        fontweight="semibold",
        pad=18,
    )
    cell_error_axis.text(
        0.0,
        1.01,
        (
            f"Common-support cells pooled over {len(smoothed_mape)} rollouts; "
            "time and position discarded"
        ),
        transform=cell_error_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#555555",
    )
    cell_error_axis.set_xlabel(r"$|v_{sim} - v_{GT}|$ (m/s)")
    cell_error_axis.set_ylabel("Cell observations per bin (%)")
    cell_error_axis.set_xlim(bins[0], bins[-1])
    cell_error_axis.legend(frameon=False, loc="upper right")
    _style_axis(cell_error_axis)

    compact = len(selected_rows) == 2
    colorbar_axis = figure.add_axes(
        (0.21, 0.055 if compact else 0.025, 0.36, 0.025 if compact else 0.018)
    )
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Mean local speed (m/s)")
    figure.suptitle(
        f"I-24-3D Scene {scene} {direction_name} speed fields · {checkpoint}",
        fontsize=15,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.18 if compact else 0.135,
        top=0.92,
        hspace=0.28,
        wspace=0.16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return chosen
