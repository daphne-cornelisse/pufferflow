#!/usr/bin/env python3
"""Schematic of return converging to 1 and a policy entering human-compatible equilibria.

Writes figures/training_and_policy_space.png and figures/training_and_policy_space.svg.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Clean seaborn styling at 300 DPI, no bold titles
sns.set_theme(style="ticks", context="notebook")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.weight": "normal",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.85,
})

COLOR_PI = "#1B4F72"      # Navy blue (RL agent / pi*)
COLOR_HUMAN = "#6B1E3A"   # Bordeaux (human-compatible equilibria)
COLOR_INIT = "#7F8C8D"    # Slate gray (initial policy)

OUT_DIR = Path(__file__).resolve().parents[1] / "figures"


def main():
    fig, (ax_lhs, ax_rhs) = plt.subplots(
        1, 2, figsize=(5.15, 2.45), gridspec_kw={"width_ratios": [0.78, 1.15]}, constrained_layout=True
    )
    ax_lhs.set_box_aspect(1)
    fig.patch.set_facecolor("white")
    ax_lhs.set_facecolor("white")
    ax_rhs.set_facecolor("white")

    # Episode return vs time steps, converging to 1.0 at 5B steps.
    rng = np.random.default_rng(42)
    steps = np.linspace(0, 5.0, 120)  # In billions (0 to 5B)
    progress = steps / 5.0

    # Six training seeds: early exploration variance, then asymptotic convergence to 1.0.
    records = []
    for seed in range(6):
        rate = rng.uniform(5.2, 6.4)
        curve = 0.08 + 0.92 * (1.0 - np.exp(-rate * progress))
        # Noise decays as the policy stabilizes near 1.0
        noise = rng.normal(0, 0.045, size=len(steps)) * np.exp(-2.8 * progress)
        noise = np.convolve(noise, np.ones(4) / 4.0, mode="same")
        ret = np.clip(curve + noise, 0.0, 1.02)
        ret[-1] = 1.0 + rng.normal(0, 0.003)
        for t_val, r_val in zip(steps, ret):
            records.append({"steps": t_val, "return": r_val, "seed": seed})

    df_train = pd.DataFrame(records)

    ax_lhs.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
    sns.lineplot(
        data=df_train,
        x="steps",
        y="return",
        errorbar="sd",
        color=COLOR_PI,
        linewidth=1.8,
        ax=ax_lhs,
        zorder=3,
    )

    start = float(df_train.loc[df_train["steps"] == 0.0, "return"].mean())
    ax_lhs.scatter(
        [0.0], [start],
        color=COLOR_INIT, edgecolor="#2C3E50", linewidth=0.8, s=30, zorder=6, clip_on=False,
    )
    ax_lhs.text(0.225, 0.17, r"$\pi_0$", fontsize=12, color="#444444", ha="left", va="center")

    ax_lhs.scatter([5.0], [1.0], color=COLOR_PI, s=36, zorder=5)
    ax_lhs.text(4.875, 1.03, r"$\pi^*$", fontsize=13, color=COLOR_PI, ha="right", va="bottom")

    ax_lhs.set_xlim(0, 5.2)
    ax_lhs.set_ylim(0.0, 1.24)
    ax_lhs.set_xticks([0.0, 2.5, 5.0])
    ax_lhs.set_xticklabels(["0", "2.5B", "5B"], fontsize=11)
    ax_lhs.set_yticks([0.0, 0.5, 1.0])
    ax_lhs.set_yticklabels(["0.0", "0.5", "1.0"], fontsize=11)
    ax_lhs.set_xlabel("Time steps", fontsize=13)
    ax_lhs.set_ylabel("Episode return", fontsize=13)
    ax_lhs.set_axisbelow(True)
    ax_lhs.grid(True, axis="y", color="#888888", linewidth=0.6, alpha=0.3, zorder=0)
    sns.despine(ax=ax_lhs)

    # Solution space: a curved path from pi_0 into the human-compatible set.
    eq_center = (0.64, 0.60)
    eq_width, eq_height, eq_angle = 0.56, 0.30, 26.0

    eq_ellipse_fill = patches.Ellipse(
        xy=eq_center,
        width=eq_width,
        height=eq_height,
        angle=eq_angle,
        facecolor=COLOR_HUMAN,
        edgecolor="none",
        alpha=0.18,
        zorder=1,
    )
    eq_ellipse_edge = patches.Ellipse(
        xy=eq_center,
        width=eq_width,
        height=eq_height,
        angle=eq_angle,
        facecolor="none",
        edgecolor=COLOR_HUMAN,
        linestyle="--",
        linewidth=1.35,
        zorder=2,
    )
    ax_rhs.add_patch(eq_ellipse_fill)
    ax_rhs.add_patch(eq_ellipse_edge)

    theta_rad = np.deg2rad(eq_angle)
    rot_mat = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)],
    ])
    u_pts = rng.normal(0, 0.14, size=(18, 2))
    # Keep points inside the ellipse boundary
    u_pts = u_pts[np.sum((u_pts / np.array([0.24, 0.12])) ** 2, axis=1) < 1.0]
    human_pts = u_pts @ rot_mat.T + np.array(eq_center)
    ax_rhs.scatter(
        human_pts[:, 0], human_pts[:, 1],
        color=COLOR_HUMAN, alpha=0.45, s=16, edgecolors="none", zorder=2.5,
    )

    s_traj = np.linspace(0, 1, 80)
    p0 = np.array([0.16, 0.24])
    p_star = np.array([0.61, 0.58])
    # Curved optimization path with a slight damped wiggle into the equilibrium.
    traj_x = p0[0] + (p_star[0] - p0[0]) * (1 - (1 - s_traj) ** 1.8) + 0.06 * np.sin(np.pi * s_traj)
    traj_y = (
        p0[1]
        + (p_star[1] - p0[1]) * (s_traj ** 0.85)
        - 0.08 * np.sin(1.4 * np.pi * s_traj) * (1 - s_traj)
    )

    ax_rhs.plot(traj_x, traj_y, color=COLOR_PI, linewidth=1.75, zorder=4)

    for idx in (16, 36, 54, 68):
        ax_rhs.annotate(
            "",
            xy=(traj_x[idx + 2], traj_y[idx + 2]),
            xytext=(traj_x[idx], traj_y[idx]),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_PI, lw=1.1, mutation_scale=8.5),
            zorder=4.5,
        )

    ax_rhs.scatter(
        [p0[0]], [p0[1]],
        color=COLOR_INIT, edgecolor="#2C3E50", linewidth=0.8, s=30, zorder=5,
    )
    ax_rhs.text(p0[0] - 0.02, p0[1] - 0.11, r"$\pi_0$", fontsize=12, color="#444444", ha="center")

    ax_rhs.scatter(
        [p_star[0]], [p_star[1]],
        color=COLOR_PI, edgecolor="white", linewidth=0.7, s=46, zorder=6,
    )
    ax_rhs.text(p_star[0] - 0.05, p_star[1] + 0.02, r"$\pi^*$", fontsize=13, color=COLOR_PI, ha="right")

    ax_rhs.text(
        0.56, 0.90, "Human-compatible\nequilibria",
        fontsize=11, color=COLOR_HUMAN, ha="center", va="center", linespacing=1.15,
    )

    ax_rhs.set_xlim(0.0, 1.0)
    ax_rhs.set_ylim(0.0, 1.0)
    ax_rhs.set_xticks([])
    ax_rhs.set_yticks([])
    ax_rhs.set_xlabel("Solution space", fontsize=13)
    ax_rhs.set_ylabel("")
    sns.despine(ax=ax_rhs, left=True)

    # The square return panel sits higher than the solution-space panel.
    # Line the bottom spines up, then drop the right label onto the left one.
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    pos_l = ax_lhs.get_position()
    pos_r = ax_rhs.get_position()
    ax_rhs.set_position([pos_r.x0, pos_l.y0, pos_r.width, pos_r.height])
    fig.canvas.draw()
    gap_px = ax_rhs.xaxis.label.get_window_extent().y0 - ax_lhs.xaxis.label.get_window_extent().y0
    ax_rhs.xaxis.labelpad += gap_px * 72.0 / fig.dpi

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "training_and_policy_space.png"
    svg_path = OUT_DIR / "training_and_policy_space.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
