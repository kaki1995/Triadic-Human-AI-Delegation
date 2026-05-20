from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
PROFILE_CSV = ANALYSIS_DIR / "state_emission_profile_v3.csv"

TEXT = "#111111"
AXIS = "#222222"
EDGE = "#155E75"
MUTED = "#A7B0B8"
BLUE = "#0057FF"
GREEN = "#00C853"
ORANGE = "#FF6B00"
YELLOW = "#FFC400"
PURPLE = "#A12EAA"
CYAN = "#00B8D9"

STATE_COLORS = {
    "Aversion": ORANGE,
    "Neutral": BLUE,
    "Appreciation": GREEN,
}

ROLE_COLORS = {
    "1": "#4CAF2F",
    "2": CYAN,
    "3": PURPLE,
    "4": ORANGE,
    "5": YELLOW,
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Times New Roman",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.2,
            "axes.grid": False,
        }
    )


def clean_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_node(
    ax,
    xy: tuple[float, float],
    label: str,
    *,
    width: float = 0.24,
    height: float = 0.12,
    edgecolor: str = EDGE,
    facecolor: str = "white",
    fontsize: int = 12,
    sublabel: str | None = None,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=1.7,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x, y + (0.018 if sublabel else 0.0), label, ha="center", va="center", fontsize=fontsize, color=TEXT)
    if sublabel:
        ax.text(x, y - 0.03, sublabel, ha="center", va="center", fontsize=8.5, color="#4B5563")


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 2.0,
    linestyle: str = "-",
    rad: float = 0.0,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    label_size: int = 9,
    alpha: float = 1.0,
    arrowstyle: str = "-|>",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        mutation_scale=12 + lw,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        shrinkA=20,
        shrinkB=20,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=label_size, color=color)


def draw_badge(ax, xy: tuple[float, float], text: str, *, color: str, radius: float = 0.033) -> None:
    fill = color
    text_color = TEXT if color == YELLOW else "white"
    ax.add_patch(Circle(xy, radius, facecolor=fill, edgecolor="white", linewidth=1.5, zorder=6))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=9, weight="bold", color=text_color, zorder=7)


def save(fig, filename: str) -> Path:
    path = ANALYSIS_DIR / filename
    fig.savefig(path, dpi=160, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    return path


def figure_relationship_and_measures() -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    manager = (0.32, 0.74)
    team = (0.14, 0.28)
    ai = (0.52, 0.28)

    draw_node(ax, manager, "Manager", width=0.22, edgecolor=EDGE)
    draw_node(ax, team, "Team", sublabel="Employee side", width=0.22, edgecolor=EDGE)
    draw_node(ax, ai, "AI", width=0.22, edgecolor=EDGE)

    draw_arrow(ax, manager, team, color=ORANGE, lw=3.0, rad=0.08, label="Human delegation", label_offset=(-0.08, 0.04))
    draw_arrow(ax, manager, ai, color=BLUE, lw=3.0, rad=-0.08, label="AI authority", label_offset=(0.07, 0.04))
    draw_arrow(ax, team, manager, color=ORANGE, lw=2.0, linestyle="--", rad=-0.12, label="Escalation", label_offset=(-0.07, -0.03))
    draw_arrow(ax, team, ai, color=GREEN, lw=2.2, rad=0.0, label="Peer / AI benchmark", label_offset=(0.0, -0.065))
    draw_arrow(ax, ai, team, color=GREEN, lw=1.6, linestyle="--", rad=0.0, alpha=0.75)

    ax.text(0.05, 0.92, "Triadic delegation relationship", fontsize=22, color=TEXT, weight="bold")
    ax.text(
        0.05,
        0.86,
        "A manager allocates decision authority between the human side and AI while escalation remains observable.",
        fontsize=11.5,
        color="#374151",
    )

    box_x, box_y, box_w, box_h = 0.66, 0.20, 0.29, 0.61
    ax.add_patch(
        FancyBboxPatch(
            (box_x, box_y),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.012",
            linewidth=1.4,
            edgecolor="#CBD5E1",
            facecolor="#F8FAFC",
        )
    )
    ax.text(box_x + 0.03, box_y + box_h - 0.07, "Observable statistics", fontsize=15, weight="bold", color=TEXT)
    stat_rows = [
        (BLUE, "AI authority share", "Share of decisions where AI holds authority"),
        (ORANGE, "Escalation share", "Share of cases returned to the manager"),
        (GREEN, "Benchmark gap", "Team, peer, and target comparisons"),
    ]
    for i, (color, title, desc) in enumerate(stat_rows):
        y = box_y + box_h - 0.17 - i * 0.18
        ax.plot([box_x + 0.035, box_x + 0.095], [y, y], color=color, linewidth=4.0, solid_capstyle="round")
        ax.text(box_x + 0.12, y + 0.025, title, fontsize=12.5, weight="bold", color=TEXT, va="center")
        ax.text(box_x + 0.12, y - 0.025, desc, fontsize=9.8, color="#4B5563", va="center")
    return save(fig, "figure_triadic_relationship_and_measures_v3.png")


def draw_scenario(ax, caption: str, mode: str) -> None:
    clean_axis(ax)
    manager = (0.50, 0.74)
    team = (0.22, 0.24)
    ai = (0.78, 0.24)
    draw_node(ax, manager, "Manager", width=0.25, height=0.12)
    draw_node(ax, team, "Team", width=0.23, height=0.12)
    draw_node(ax, ai, "AI", width=0.23, height=0.12)

    if mode == "manager":
        draw_arrow(ax, manager, team, color=EDGE, lw=2.2, rad=0.02)
        draw_arrow(ax, manager, ai, color=EDGE, lw=2.2, rad=-0.02)
        badges = [(manager, ["1", "3", "5"]), (team, ["4"]), (ai, ["4"])]
    elif mode == "employee":
        draw_arrow(ax, manager, team, color=EDGE, lw=2.4, rad=0.02)
        draw_arrow(ax, ai, team, color=MUTED, lw=1.7, rad=0.0)
        badges = [(manager, ["1", "3", "5"]), (team, ["2", "4"]), (ai, ["5"])]
    elif mode == "ai":
        draw_arrow(ax, manager, ai, color=EDGE, lw=2.4, rad=-0.02)
        draw_arrow(ax, team, ai, color=MUTED, lw=1.7, rad=0.0)
        badges = [(manager, ["1", "3", "5"]), (team, ["5"]), (ai, ["2", "4"])]
    else:
        draw_arrow(ax, manager, team, color=EDGE, lw=2.2, rad=0.02)
        draw_arrow(ax, manager, ai, color=EDGE, lw=2.2, rad=-0.02)
        draw_arrow(ax, team, ai, color=BLUE, lw=2.0, rad=0.02)
        draw_arrow(ax, ai, team, color=GREEN, lw=1.8, rad=-0.02)
        badges = [(manager, ["1", "3", "5"]), (team, ["2", "4"]), (ai, ["2", "4"])]

    for center, nums in badges:
        x0, y0 = center
        start = x0 - 0.035 * (len(nums) - 1)
        for idx, num in enumerate(nums):
            draw_badge(ax, (start + idx * 0.07, y0 + 0.115), num, color=ROLE_COLORS[num])

    ax.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=11, color=BLUE)


def figure_observable_scenarios() -> Path:
    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95], hspace=0.36, wspace=0.22)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    legend_ax = fig.add_subplot(gs[:, 2])
    clean_axis(legend_ax)

    draw_scenario(axes[0], "Scenario 1: Manager retains authority", "manager")
    draw_scenario(axes[1], "Scenario 2: Manager delegates to team", "employee")
    draw_scenario(axes[2], "Scenario 3: Manager delegates to AI", "ai")
    draw_scenario(axes[3], "Scenario 4: Shared human-AI delegation", "shared")

    fig.text(0.04, 0.96, "Observable triadic delegation scenarios", fontsize=22, weight="bold", color=TEXT)

    legend_ax.add_patch(
        FancyBboxPatch(
            (0.06, 0.12),
            0.88,
            0.76,
            boxstyle="round,pad=0.02,rounding_size=0.012",
            linewidth=1.2,
            edgecolor="#CBD5E1",
            facecolor="#E0F2FE",
        )
    )
    legend_ax.text(0.50, 0.80, "Decision authority roles", ha="center", fontsize=15, weight="bold", color=TEXT)
    role_rows = [
        ("1", "Responsible"),
        ("2", "Execute"),
        ("3", "Approve"),
        ("4", "Contribute"),
        ("5", "Informed"),
    ]
    for idx, (num, label) in enumerate(role_rows):
        y = 0.68 - idx * 0.105
        draw_badge(legend_ax, (0.20, y), num, color=ROLE_COLORS[num], radius=0.04)
        legend_ax.text(0.31, y, label, va="center", fontsize=12.5, color=TEXT)
    legend_ax.plot([0.15, 0.31], [0.20, 0.20], color=EDGE, linewidth=2.2)
    legend_ax.add_patch(
        FancyArrowPatch((0.28, 0.20), (0.32, 0.20), arrowstyle="-|>", mutation_scale=12, color=EDGE, linewidth=2.2)
    )
    legend_ax.text(0.38, 0.20, "Delegation / direction", va="center", fontsize=11, color=TEXT)

    return save(fig, "figure_triadic_observable_scenarios_v3.png")


def figure_state_profiles() -> Path:
    profile = pd.read_csv(PROFILE_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))

    for ax, state in zip(axes, ["Aversion", "Neutral", "Appreciation"]):
        clean_axis(ax)
        row = profile.loc[profile["State"] == state].iloc[0]
        ai_share = float(row["AI Authority Share"])
        escalation = float(row["Escalation Share"])
        human_share = max(0.0, 1.0 - ai_share)
        state_color = STATE_COLORS[state]

        manager = (0.50, 0.75)
        team = (0.22, 0.26)
        ai = (0.78, 0.26)

        ax.text(0.02, 0.96, state, ha="left", va="top", fontsize=15, weight="bold", color=state_color)
        ax.plot([0.02, 0.98], [0.91, 0.91], color=state_color, linewidth=3.0, solid_capstyle="round")
        draw_node(ax, manager, "Manager", width=0.24, height=0.12, edgecolor=state_color)
        draw_node(ax, team, "Team", width=0.22, height=0.12)
        draw_node(ax, ai, "AI", width=0.22, height=0.12)

        draw_arrow(ax, manager, team, color="#334155", lw=1.3 + 4.5 * human_share, rad=0.10)
        draw_arrow(ax, manager, ai, color=BLUE, lw=1.3 + 5.5 * ai_share, rad=-0.10)
        draw_arrow(ax, team, manager, color=ORANGE, lw=1.3 + 5.2 * escalation, linestyle="--", rad=-0.12)
        draw_arrow(ax, team, ai, color=GREEN, lw=1.8, rad=0.0, alpha=0.70)

        ax.text(0.50, 0.07, f"AI authority: {ai_share * 100:.0f}%", ha="center", fontsize=12, color=BLUE, weight="bold")
        ax.text(0.50, 0.01, f"Escalation: {escalation * 100:.0f}%", ha="center", fontsize=12, color=ORANGE, weight="bold")

    fig.text(0.04, 0.92, "State-dependent triadic profiles", fontsize=21, weight="bold", color=TEXT)
    fig.text(
        0.04,
        0.86,
        "Arrow thickness encodes state-level fitted means from the selected three-state HMM.",
        fontsize=11,
        color="#374151",
    )
    return save(fig, "figure_triadic_state_profiles_v3.png")


def main() -> None:
    setup_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_relationship_and_measures(),
        figure_observable_scenarios(),
        figure_state_profiles(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
