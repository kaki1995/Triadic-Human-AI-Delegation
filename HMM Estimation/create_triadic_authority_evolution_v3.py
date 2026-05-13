from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
PROFILE_CSV = ANALYSIS_DIR / "state_emission_profile_v3.csv"
TREND_CSV = ANALYSIS_DIR / "state_time_trend_tests_v3.csv"

TEXT = "#111111"
MUTED = "#64748B"
EDGE = "#155E75"
MANAGER_EDGE = "#0F4C81"
MANAGER_FILL = "#E0F2FE"
ACTOR_FILL = "#F8FAFC"
BLUE = "#0057FF"
GREEN = "#00C853"
ORANGE = "#FF6B00"
GRAY = "#94A3B8"

STATE_LABELS = {
    "Aversion": "(a) Aversion state",
    "Neutral": "(b) Neutral state",
    "Appreciation": "(c) Appreciation state",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Times New Roman",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.15,
            "axes.grid": False,
        }
    )


def clean_diagram_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def pct(value: float) -> str:
    return f"{100.0 * value:.0f}%"


def draw_circle_node(ax, center, label: str, *, manager: bool = False) -> None:
    radius = 0.092 if manager else 0.088
    patch = Circle(
        center,
        radius,
        facecolor=MANAGER_FILL if manager else ACTOR_FILL,
        edgecolor=MANAGER_EDGE if manager else EDGE,
        linewidth=1.7,
    )
    ax.add_patch(patch)
    ax.text(center[0], center[1], label, ha="center", va="center", fontsize=11.5, color=TEXT)


def draw_arrow(ax, start, end, *, color: str, linestyle: str = "-", rad: float = 0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.55,
        linestyle=linestyle,
        color=color,
        shrinkA=24,
        shrinkB=24,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def trend_path(trend_tests: pd.DataFrame, metric: str, state: str, periods: np.ndarray) -> np.ndarray:
    row = trend_tests[(trend_tests["metric"] == metric) & (trend_tests["state"] == state)].iloc[0]
    start = float(row["period_1_mean"])
    end = float(row["period_26_mean"])
    progress = (periods - periods.min()) / max(periods.max() - periods.min(), 1.0)
    return start + (end - start) * progress


def draw_state_structure(ax, state: str, ai_share: float, escalation_share: float) -> None:
    clean_diagram_axis(ax)
    employee_share = max(0.0, 1.0 - ai_share)

    ax.set_title(STATE_LABELS[state], loc="left", fontsize=14, weight="bold", color=TEXT, pad=4)

    manager = (0.50, 0.72)
    employee = (0.23, 0.24)
    ai = (0.77, 0.24)

    draw_circle_node(ax, manager, "Manager", manager=True)
    draw_circle_node(ax, employee, "Employee")
    draw_circle_node(ax, ai, "AI")

    draw_arrow(ax, manager, employee, color=GRAY, rad=0.10)
    draw_arrow(ax, manager, ai, color=BLUE, rad=-0.10)
    draw_arrow(ax, employee, manager, color=ORANGE, linestyle="--", rad=-0.18)

    ax.text(0.18, 0.50, f"Employee authority\n{pct(employee_share)}", ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(0.82, 0.50, f"AI authority\n{pct(ai_share)}", ha="center", va="center", fontsize=9.5, color=BLUE)
    ax.text(0.21, 0.75, f"Escalation\n{pct(escalation_share)}", ha="center", va="center", fontsize=9.5, color=ORANGE)


def make_structure_figure(profile: pd.DataFrame) -> Path:
    states = ["Aversion", "Neutral", "Appreciation"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.text(0.04, 0.94, "State-dependent triadic authority structure", fontsize=20, weight="bold", color=TEXT)
    fig.text(
        0.04,
        0.885,
        "Each panel shows the same Manager-Employee-AI triad. Percent labels report fitted HMM state means.",
        fontsize=11,
        color="#374151",
    )

    for ax, state in zip(axes, states):
        row = profile[profile["State"] == state].iloc[0]
        draw_state_structure(
            ax,
            state,
            ai_share=float(row["AI Authority Share"]),
            escalation_share=float(row["Escalation Share"]),
        )

    handles = [
        Line2D([0], [0], color=GRAY, linewidth=1.55, label="Manager to Employee"),
        Line2D([0], [0], color=BLUE, linewidth=1.55, label="Manager to AI"),
        Line2D([0], [0], color=ORANGE, linewidth=1.55, linestyle="--", label="Employee to Manager escalation"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=1.0,
        fontsize=9.5,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.05, right=0.98, wspace=0.18)

    path = ANALYSIS_DIR / "figure_triadic_authority_structure_by_state_v3.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def draw_time_panel(ax, state: str, trend_tests: pd.DataFrame) -> None:
    periods = np.arange(1, 27)
    ai = trend_path(trend_tests, "AI Authority Share", state, periods)
    escalation = trend_path(trend_tests, "Escalation Share", state, periods)

    ax.plot(periods, 100.0 * ai, color=BLUE, linewidth=1.45, label="AI authority")
    ax.plot(periods, 100.0 * escalation, color=ORANGE, linewidth=1.45, linestyle="--", label="Escalation")
    ax.set_title(STATE_LABELS[state], loc="left", fontsize=13, weight="bold", color=TEXT, pad=8)
    ax.set_xlim(1, 26)
    ax.set_ylim(0, 70)
    ax.set_xticks([1, 6, 11, 16, 21, 26])
    ax.set_yticks([0, 20, 40, 60])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9.5, colors=TEXT, width=1.0, length=4)
    ax.set_xlabel("Planning cycle", fontsize=10.5, color=TEXT)
    if state == "Aversion":
        ax.set_ylabel("Predicted share of manager-period decisions (%)", fontsize=10.5, color=TEXT)
    else:
        ax.tick_params(axis="y", labelleft=False)

    ai_direction = "rises" if ai[-1] > ai[0] else "falls"
    esc_direction = "rises" if escalation[-1] > escalation[0] else "falls"
    ax.text(
        0.04,
        0.92,
        f"AI {ai_direction}: {pct(ai[0])} -> {pct(ai[-1])}",
        transform=ax.transAxes,
        fontsize=9.2,
        color=BLUE,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.2},
    )
    ax.text(
        0.04,
        0.82,
        f"Escalation {esc_direction}: {pct(escalation[0])} -> {pct(escalation[-1])}",
        transform=ax.transAxes,
        fontsize=9.2,
        color=ORANGE,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.2},
    )


def make_time_figure(trend_tests: pd.DataFrame) -> Path:
    states = ["Aversion", "Neutral", "Appreciation"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    fig.text(0.04, 0.94, "Authority and escalation over planning cycles", fontsize=20, weight="bold", color=TEXT)
    fig.text(
        0.04,
        0.885,
        "Lines show model-implied changes within each latent state from planning cycle 1 to 26.",
        fontsize=11,
        color="#374151",
    )

    for ax, state in zip(axes, states):
        draw_time_panel(ax, state, trend_tests)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.90),
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=1.0,
        fontsize=10,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.08, right=0.98, wspace=0.14)

    path = ANALYSIS_DIR / "figure_triadic_authority_trends_by_state_v3.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    setup_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    profile = pd.read_csv(PROFILE_CSV)
    trend_tests = pd.read_csv(TREND_CSV)
    for path in [make_structure_figure(profile), make_time_figure(trend_tests)]:
        print(path)


if __name__ == "__main__":
    main()
