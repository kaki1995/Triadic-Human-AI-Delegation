from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_CSV = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"
FLOW_CSV = ANALYSIS_DIR / "triadic_task_flow_by_state_v3.csv"
ACCEPTANCE_CSV = ANALYSIS_DIR / "manager_acceptance_by_ai_confidence_state_v3.csv"
FLOW_FIGURE = ANALYSIS_DIR / "figure_triadic_task_flow_by_state_v3.png"
ACCEPTANCE_FIGURE = ANALYSIS_DIR / "figure_manager_acceptance_by_ai_confidence_v3.png"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
STATE_TITLES = {
    "Aversion": "(a) Aversion state",
    "Neutral": "(b) Neutral state",
    "Appreciation": "(c) Appreciation state",
}

TEXT = "#111111"
MUTED = "#64748B"
GRID = "#E5E7EB"
MANAGER_EDGE = "#0F4C81"
MANAGER_FILL = "#E0F2FE"
ACTOR_EDGE = "#155E75"
ACTOR_FILL = "#F8FAFC"
BLUE = "#0057FF"
GREEN = "#00C853"
ORANGE = "#FF6B00"
CYAN = "#00A6FB"
SLATE = "#334155"
LIGHT_SLATE = "#94A3B8"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Times New Roman",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.0,
            "axes.grid": False,
        }
    )


def pct(value: float) -> str:
    return f"{value:.0f}%"


def read_task_data() -> pd.DataFrame:
    post = pd.read_csv(POSTERIOR_CSV, usecols=["manager_id", "period_id", "state_label"])
    dec = pd.read_excel(
        DATA_PATH,
        sheet_name="decision_episode",
        usecols=[
            "episode_id",
            "manager_id",
            "period_id",
            "manager_action",
            "escalation_flag",
            "ai_confidence",
        ],
    )
    exec_ep = pd.read_excel(
        DATA_PATH,
        sheet_name="execution_episode",
        usecols=["episode_id", "execution_mode", "employee_override_during_execution"],
    )

    tasks = dec.merge(post, on=["manager_id", "period_id"], how="left")
    tasks = tasks.merge(exec_ep, on="episode_id", how="left")
    return tasks[tasks["state_label"].isin(STATE_ORDER)].copy()


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = read_task_data()
    rows: list[dict[str, float | int | str]] = []

    for state in STATE_ORDER:
        g = tasks[tasks["state_label"] == state]
        n = len(g)
        no_escalation = g["escalation_flag"].eq(0)
        fully_approved = g["manager_action"].eq("accept") & no_escalation
        escalated = g["escalation_flag"].eq(1)
        manager_override = g["manager_action"].isin(["modify", "reject"]) & no_escalation

        rows.append(
            {
                "state": state,
                "n_tasks": int(n),
                "fully_approved_no_escalation_n": int(fully_approved.sum()),
                "fully_approved_no_escalation_pct": 100.0 * float(fully_approved.mean()),
                "escalated_to_employee_n": int(escalated.sum()),
                "escalated_to_employee_pct": 100.0 * float(escalated.mean()),
                "manager_override_no_escalation_n": int(manager_override.sum()),
                "manager_override_no_escalation_pct": 100.0 * float(manager_override.mean()),
                "manager_accept_n": int(g["manager_action"].eq("accept").sum()),
                "manager_accept_pct": 100.0 * float(g["manager_action"].eq("accept").mean()),
                "manager_modify_n": int(g["manager_action"].eq("modify").sum()),
                "manager_modify_pct": 100.0 * float(g["manager_action"].eq("modify").mean()),
                "manager_reject_n": int(g["manager_action"].eq("reject").sum()),
                "manager_reject_pct": 100.0 * float(g["manager_action"].eq("reject").mean()),
                "employee_execution_n": int(g["execution_mode"].eq("human").sum()),
                "employee_execution_pct": 100.0 * float(g["execution_mode"].eq("human").mean()),
                "ai_execution_n": int(g["execution_mode"].eq("ai").sum()),
                "ai_execution_pct": 100.0 * float(g["execution_mode"].eq("ai").mean()),
                "joint_execution_n": int(g["execution_mode"].eq("joint").sum()),
                "joint_execution_pct": 100.0 * float(g["execution_mode"].eq("joint").mean()),
                "employee_override_during_execution_n": int(g["employee_override_during_execution"].eq(1).sum()),
                "employee_override_during_execution_pct": 100.0
                * float(g["employee_override_during_execution"].eq(1).mean()),
                "mean_ai_confidence": float(g["ai_confidence"].mean()),
            }
        )

    flow = pd.DataFrame(rows)

    bins = np.linspace(0.0, 1.0, 11)
    tasks["confidence_bin"] = pd.cut(tasks["ai_confidence"], bins=bins, include_lowest=True)
    tasks["confidence_mid_pct"] = tasks["confidence_bin"].apply(lambda interval: 100.0 * interval.mid)
    tasks["fully_approved_no_escalation"] = (
        tasks["manager_action"].eq("accept") & tasks["escalation_flag"].eq(0)
    ).astype(int)
    tasks["manager_accept"] = tasks["manager_action"].eq("accept").astype(int)

    acceptance = (
        tasks.groupby(["state_label", "confidence_mid_pct"], observed=True)
        .agg(
            n_tasks=("episode_id", "size"),
            fully_approved_no_escalation_pct=("fully_approved_no_escalation", lambda x: 100.0 * float(x.mean())),
            manager_accept_pct=("manager_accept", lambda x: 100.0 * float(x.mean())),
            escalated_to_employee_pct=("escalation_flag", lambda x: 100.0 * float(x.mean())),
        )
        .reset_index()
        .rename(columns={"state_label": "state"})
    )
    acceptance["confidence_mid_pct"] = acceptance["confidence_mid_pct"].astype(float)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    flow.to_csv(FLOW_CSV, index=False)
    acceptance.to_csv(ACCEPTANCE_CSV, index=False)
    return flow, acceptance


def load_or_build(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if refresh or not FLOW_CSV.exists() or not ACCEPTANCE_CSV.exists():
        return build_outputs()
    return pd.read_csv(FLOW_CSV), pd.read_csv(ACCEPTANCE_CSV)


def clean_diagram_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_circle_node(ax, center: tuple[float, float], label: str, *, manager: bool = False) -> None:
    patch = Circle(
        center,
        0.095 if manager else 0.088,
        facecolor=MANAGER_FILL if manager else ACTOR_FILL,
        edgecolor=MANAGER_EDGE if manager else ACTOR_EDGE,
        linewidth=1.55,
    )
    ax.add_patch(patch)
    ax.text(center[0], center[1], label, ha="center", va="center", fontsize=10.2, color=TEXT)


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linestyle: str = "-",
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            linestyle=linestyle,
            color=color,
            shrinkA=22,
            shrinkB=22,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_triad(ax, state: str, n_tasks: int) -> None:
    clean_diagram_axis(ax)
    ax.set_title(STATE_TITLES[state], loc="left", fontsize=13.5, weight="bold", color=TEXT, pad=5)
    manager = (0.50, 0.72)
    employee = (0.22, 0.22)
    ai = (0.78, 0.22)

    draw_circle_node(ax, manager, "Manager", manager=True)
    draw_circle_node(ax, employee, "Employee")
    draw_circle_node(ax, ai, "AI")
    draw_arrow(ax, ai, manager, color=GREEN, rad=0.12)
    draw_arrow(ax, manager, employee, color=ORANGE, linestyle="--", rad=0.08)
    draw_arrow(ax, manager, ai, color=BLUE, rad=-0.08)

    ax.text(
        0.50,
        0.02,
        f"Observed tasks: {n_tasks:,}",
        ha="center",
        va="bottom",
        fontsize=9.4,
        color=MUTED,
    )


def format_bar_axis(ax, *, show_xlabel: bool = False) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#222222")
    ax.tick_params(axis="x", labelsize=8.8, colors=TEXT, length=3, width=0.9)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)


def draw_stacked_bar(
    ax,
    values: list[float],
    counts: list[int],
    labels: list[str],
    colors: list[str],
    *,
    row_title: str,
    show_xlabel: bool = False,
) -> None:
    format_bar_axis(ax, show_xlabel=show_xlabel)
    left = 0.0
    for value, count, label, color in zip(values, counts, labels, colors):
        ax.barh(
            0,
            value,
            left=left,
            height=0.36,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )
        if value >= 8.0:
            ax.text(
                left + value / 2,
                0,
                f"{pct(value)}\n{count:,}",
                ha="center",
                va="center",
                fontsize=8.3,
                color="white",
                weight="bold",
            )
        left += value
    ax.text(0, 0.47, row_title, ha="left", va="top", fontsize=10.4, weight="bold", color=TEXT)


def make_flow_figure(flow: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(13.5, 7.3))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.35, 0.92, 0.92],
        hspace=0.32,
        wspace=0.18,
        left=0.055,
        right=0.985,
        top=0.84,
        bottom=0.25,
    )

    fig.text(0.055, 0.935, "State-dependent triadic task flow", fontsize=20, weight="bold", color=TEXT)
    fig.text(
        0.055,
        0.887,
        "Decision bars show full AI approval, employee escalation, and manager override; execution bars show who performed the task.",
        fontsize=11,
        color="#374151",
    )

    for col, state in enumerate(STATE_ORDER):
        row = flow[flow["state"] == state].iloc[0]
        draw_triad(fig.add_subplot(gs[0, col]), state, int(row["n_tasks"]))

        draw_stacked_bar(
            fig.add_subplot(gs[1, col]),
            [
                float(row["fully_approved_no_escalation_pct"]),
                float(row["escalated_to_employee_pct"]),
                float(row["manager_override_no_escalation_pct"]),
            ],
            [
                int(row["fully_approved_no_escalation_n"]),
                int(row["escalated_to_employee_n"]),
                int(row["manager_override_no_escalation_n"]),
            ],
            ["Approved", "Escalated", "Override"],
            [GREEN, ORANGE, SLATE],
            row_title="Manager response to AI recommendation",
        )

        draw_stacked_bar(
            fig.add_subplot(gs[2, col]),
            [
                float(row["employee_execution_pct"]),
                float(row["ai_execution_pct"]),
                float(row["joint_execution_pct"]),
            ],
            [
                int(row["employee_execution_n"]),
                int(row["ai_execution_n"]),
                int(row["joint_execution_n"]),
            ],
            ["Employee", "AI", "Joint"],
            [CYAN, BLUE, GREEN],
            row_title="Task execution mode",
            show_xlabel=True,
        )

    handles = [
        Line2D([0], [0], color=GREEN, linewidth=5, label="Fully approved, no escalation"),
        Line2D([0], [0], color=ORANGE, linewidth=5, label="Escalated to employee"),
        Line2D([0], [0], color=SLATE, linewidth=5, label="Manager modify/reject, no escalation"),
        Line2D([0], [0], color=CYAN, linewidth=5, label="Employee execution"),
        Line2D([0], [0], color=BLUE, linewidth=5, label="AI execution"),
        Line2D([0], [0], color=GREEN, linewidth=5, label="AI + Employee joint execution"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=1.0,
        fontsize=9.2,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.text(0.52, 0.18, "Share of tasks (%)", ha="center", fontsize=10.5, color=TEXT)
    fig.text(
        0.055,
        0.146,
        "Arrow meanings: green = AI recommendation fully approved by manager; orange dashed = employee escalation; blue = AI execution route.",
        fontsize=8.8,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.123,
        "Note: Manager self-execution is not a separate dataset field; manager involvement is observed as accept, modify/reject, and escalation.",
        fontsize=8.8,
        color=MUTED,
    )
    fig.savefig(FLOW_FIGURE, dpi=160, facecolor="white")
    plt.close(fig)
    return FLOW_FIGURE


def make_acceptance_figure(acceptance: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.05, 4.55))
    colors = {"Aversion": ORANGE, "Neutral": GREEN, "Appreciation": BLUE}
    markers = {"Aversion": "o", "Neutral": "s", "Appreciation": "^"}

    for state in STATE_ORDER:
        sub = acceptance[acceptance["state"] == state].sort_values("confidence_mid_pct")
        ax.plot(
            sub["confidence_mid_pct"],
            sub["fully_approved_no_escalation_pct"],
            color=colors[state],
            linewidth=1.45,
            marker=markers[state],
            markersize=4.3,
            label=state,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9.2, colors=TEXT, width=0.9, length=3.5)
    ax.set_xlabel("AI confidence bin (%)", fontsize=10.5, color=TEXT)
    ax.set_ylabel("Fully approved AI recommendations (%)", fontsize=10.5, color=TEXT)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=1.0,
        fontsize=9.4,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(ACCEPTANCE_FIGURE, dpi=160, facecolor="white")
    plt.close(fig)
    return ACCEPTANCE_FIGURE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Recompute summaries from the Excel workbook.")
    args = parser.parse_args()

    setup_style()
    flow, acceptance = load_or_build(refresh=args.refresh)
    for path in [make_flow_figure(flow), make_acceptance_figure(acceptance)]:
        print(path)
    print(FLOW_CSV)
    print(ACCEPTANCE_CSV)


if __name__ == "__main__":
    main()
