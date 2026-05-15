from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_CSV = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"
FLOW_CSV = ANALYSIS_DIR / "triadic_task_flow_by_state_v3.csv"
PATH_CSV = ANALYSIS_DIR / "triadic_task_flow_paths_by_state_v3.csv"
ACCEPTANCE_CSV = ANALYSIS_DIR / "manager_acceptance_by_ai_confidence_state_v3.csv"
FLOW_FIGURE = ANALYSIS_DIR / "figure_triadic_task_flow_by_state_v3.png"
ACCEPTANCE_FIGURE = ANALYSIS_DIR / "figure_manager_acceptance_by_ai_confidence_v3.png"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
DECISION_ORDER = ["Fully approved", "Escalated", "Manager changed"]
EXECUTION_ORDER = ["AI", "Joint", "Employee"]

DECISION_SHORT_LABELS = {
    "Fully approved": "Approved",
    "Escalated": "Escalated",
    "Manager changed": "Changed",
}
EXECUTION_SHORT_LABELS = {
    "AI": "AI",
    "Joint": "Joint",
    "Employee": "Employee",
}

TEXT = "#111111"
MUTED = "#64748B"
GRID = "#E5E7EB"
PANEL_FILL = "#F8FAFC"
PANEL_EDGE = "#E2E8F0"
SOURCE_FILL = "#EEF2F7"
GREEN = "#16A34A"
ORANGE = "#F97316"
SLATE = "#475569"
BLUE = "#2563EB"
TEAL = "#0891B2"
PURPLE = "#7C3AED"

STATE_COLORS = {
    "Aversion": ORANGE,
    "Neutral": BLUE,
    "Appreciation": GREEN,
}
DECISION_COLORS = {
    "Fully approved": GREEN,
    "Escalated": ORANGE,
    "Manager changed": SLATE,
}
EXECUTION_COLORS = {
    "AI": BLUE,
    "Joint": PURPLE,
    "Employee": TEAL,
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
            "axes.linewidth": 1.0,
            "axes.grid": False,
        }
    )


def pct(value: float) -> str:
    return f"{value:.0f}%"


def pct1(value: float) -> str:
    return f"{value:.1f}%"


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


def add_flow_categories(tasks: pd.DataFrame) -> pd.DataFrame:
    tasks = tasks.copy()
    no_escalation = tasks["escalation_flag"].eq(0)
    tasks["decision_path"] = np.select(
        [
            tasks["manager_action"].eq("accept") & no_escalation,
            tasks["escalation_flag"].eq(1),
            tasks["manager_action"].isin(["modify", "reject"]) & no_escalation,
        ],
        DECISION_ORDER,
        default="Other",
    )
    tasks["execution_path"] = tasks["execution_mode"].map(
        {
            "ai": "AI",
            "joint": "Joint",
            "human": "Employee",
        }
    )

    unexpected_decisions = sorted(set(tasks["decision_path"]) - set(DECISION_ORDER))
    if unexpected_decisions:
        raise ValueError(f"Unexpected manager response categories: {unexpected_decisions}")
    if tasks["execution_path"].isna().any():
        unexpected_execution = sorted(tasks.loc[tasks["execution_path"].isna(), "execution_mode"].dropna().unique())
        raise ValueError(f"Unexpected execution modes: {unexpected_execution}")
    return tasks


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = add_flow_categories(read_task_data())
    rows: list[dict[str, float | int | str]] = []

    for state in STATE_ORDER:
        g = tasks[tasks["state_label"] == state]
        n = len(g)
        fully_approved = g["decision_path"].eq("Fully approved")
        escalated = g["decision_path"].eq("Escalated")
        manager_changed = g["decision_path"].eq("Manager changed")

        rows.append(
            {
                "state": state,
                "n_tasks": int(n),
                "fully_approved_no_escalation_n": int(fully_approved.sum()),
                "fully_approved_no_escalation_pct": 100.0 * float(fully_approved.mean()),
                "escalated_to_employee_n": int(escalated.sum()),
                "escalated_to_employee_pct": 100.0 * float(escalated.mean()),
                "manager_override_no_escalation_n": int(manager_changed.sum()),
                "manager_override_no_escalation_pct": 100.0 * float(manager_changed.mean()),
                "manager_accept_n": int(g["manager_action"].eq("accept").sum()),
                "manager_accept_pct": 100.0 * float(g["manager_action"].eq("accept").mean()),
                "manager_modify_n": int(g["manager_action"].eq("modify").sum()),
                "manager_modify_pct": 100.0 * float(g["manager_action"].eq("modify").mean()),
                "manager_reject_n": int(g["manager_action"].eq("reject").sum()),
                "manager_reject_pct": 100.0 * float(g["manager_action"].eq("reject").mean()),
                "employee_execution_n": int(g["execution_path"].eq("Employee").sum()),
                "employee_execution_pct": 100.0 * float(g["execution_path"].eq("Employee").mean()),
                "ai_execution_n": int(g["execution_path"].eq("AI").sum()),
                "ai_execution_pct": 100.0 * float(g["execution_path"].eq("AI").mean()),
                "joint_execution_n": int(g["execution_path"].eq("Joint").sum()),
                "joint_execution_pct": 100.0 * float(g["execution_path"].eq("Joint").mean()),
                "employee_override_during_execution_n": int(g["employee_override_during_execution"].eq(1).sum()),
                "employee_override_during_execution_pct": 100.0
                * float(g["employee_override_during_execution"].eq(1).mean()),
                "mean_ai_confidence": float(g["ai_confidence"].mean()),
            }
        )

    flow = pd.DataFrame(rows)

    path_index = pd.MultiIndex.from_product(
        [STATE_ORDER, DECISION_ORDER, EXECUTION_ORDER],
        names=["state", "decision_path", "execution_path"],
    )
    paths = (
        tasks.groupby(["state_label", "decision_path", "execution_path"], observed=True)
        .agg(n_tasks=("episode_id", "size"))
        .rename_axis(["state", "decision_path", "execution_path"])
        .reindex(path_index, fill_value=0)
        .reset_index()
    )
    state_totals = flow.set_index("state")["n_tasks"]
    paths["state_total"] = paths["state"].map(state_totals).astype(int)
    paths["decision_total"] = paths.groupby(["state", "decision_path"])["n_tasks"].transform("sum").astype(int)
    paths["execution_total"] = paths.groupby(["state", "execution_path"])["n_tasks"].transform("sum").astype(int)
    paths["path_pct_of_state"] = np.where(
        paths["state_total"].gt(0),
        100.0 * paths["n_tasks"] / paths["state_total"],
        0.0,
    )
    paths["path_pct_of_decision"] = np.where(
        paths["decision_total"].gt(0),
        100.0 * paths["n_tasks"] / paths["decision_total"],
        0.0,
    )
    paths["path_pct_of_execution"] = np.where(
        paths["execution_total"].gt(0),
        100.0 * paths["n_tasks"] / paths["execution_total"],
        0.0,
    )

    bins = np.linspace(0.0, 1.0, 11)
    tasks["confidence_bin"] = pd.cut(tasks["ai_confidence"], bins=bins, include_lowest=True)
    tasks["confidence_mid_pct"] = tasks["confidence_bin"].apply(lambda interval: 100.0 * interval.mid)
    tasks["fully_approved_no_escalation"] = tasks["decision_path"].eq("Fully approved").astype(int)
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
    paths.to_csv(PATH_CSV, index=False)
    acceptance.to_csv(ACCEPTANCE_CSV, index=False)
    return flow, acceptance, paths


def load_or_build(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if refresh or not FLOW_CSV.exists() or not ACCEPTANCE_CSV.exists() or not PATH_CSV.exists():
        return build_outputs()
    return pd.read_csv(FLOW_CSV), pd.read_csv(ACCEPTANCE_CSV), pd.read_csv(PATH_CSV)


def clean_alluvial_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_box(
    ax,
    x_center: float,
    y_bottom: float,
    y_top: float,
    *,
    width: float,
    facecolor: str,
    edgecolor: str = "white",
    linewidth: float = 1.0,
    radius: float = 0.006,
    zorder: int = 6,
    alpha: float = 1.0,
) -> None:
    if y_top <= y_bottom:
        return
    ax.add_patch(
        FancyBboxPatch(
            (x_center - width / 2.0, y_bottom),
            width,
            y_top - y_bottom,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            zorder=zorder,
        )
    )


def stack_segments(
    order: list[str],
    values: dict[str, float],
    y_bottom: float,
    y_top: float,
    *,
    gap: float,
) -> dict[str, tuple[float, float]]:
    positive = [label for label in order if values.get(label, 0.0) > 0.0]
    usable_height = max(0.0, y_top - y_bottom - gap * max(0, len(positive) - 1))
    total = sum(max(values.get(label, 0.0), 0.0) for label in order)
    current_top = y_top
    segments: dict[str, tuple[float, float]] = {}

    for label in order:
        value = max(values.get(label, 0.0), 0.0)
        if total <= 0.0 or value <= 0.0:
            segments[label] = (current_top, current_top)
            continue
        height = usable_height * value / total
        y0 = current_top - height
        segments[label] = (y0, current_top)
        current_top = y0 - gap
    return segments


def subdivide_segment(
    order: list[str],
    values: dict[str, float],
    y_bottom: float,
    y_top: float,
) -> dict[str, tuple[float, float]]:
    total = sum(max(values.get(label, 0.0), 0.0) for label in order)
    current_top = y_top
    segments: dict[str, tuple[float, float]] = {}

    for label in order:
        value = max(values.get(label, 0.0), 0.0)
        if total <= 0.0 or value <= 0.0:
            segments[label] = (current_top, current_top)
            continue
        height = (y_top - y_bottom) * value / total
        y0 = current_top - height
        segments[label] = (y0, current_top)
        current_top = y0
    return segments


def add_ribbon(
    ax,
    x0: float,
    y0_bottom: float,
    y0_top: float,
    x1: float,
    y1_bottom: float,
    y1_top: float,
    *,
    color: str,
    alpha: float,
    zorder: int,
) -> None:
    if y0_top - y0_bottom <= 0.0001 or y1_top - y1_bottom <= 0.0001:
        return
    dx = (x1 - x0) * 0.50
    verts = [
        (x0, y0_top),
        (x0 + dx, y0_top),
        (x1 - dx, y1_top),
        (x1, y1_top),
        (x1, y1_bottom),
        (x1 - dx, y1_bottom),
        (x0 + dx, y0_bottom),
        (x0, y0_bottom),
        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )
    )


def draw_node_label(
    ax,
    x_center: float,
    y_bottom: float,
    y_top: float,
    *,
    label: str,
    value_pct: float,
    color: str,
    small_threshold: float = 0.046,
) -> None:
    if y_top <= y_bottom:
        return
    height = y_top - y_bottom
    if height >= small_threshold:
        text = f"{label}\n{pct(value_pct)}"
        fontsize = 8.2
    else:
        text = pct(value_pct)
        fontsize = 8.0
    text_color = TEXT if color == SOURCE_FILL else "white"
    ax.text(
        x_center,
        (y_bottom + y_top) / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight="bold",
        linespacing=0.92,
        zorder=9,
    )


def state_counts(flow_row: pd.Series, paths_state: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    decision_counts = {
        decision: int(paths_state.loc[paths_state["decision_path"].eq(decision), "n_tasks"].sum())
        for decision in DECISION_ORDER
    }
    execution_counts = {
        execution: int(paths_state.loc[paths_state["execution_path"].eq(execution), "n_tasks"].sum())
        for execution in EXECUTION_ORDER
    }
    total = int(flow_row["n_tasks"])
    if sum(decision_counts.values()) != total or sum(execution_counts.values()) != total:
        raise ValueError(f"Task path totals do not reconcile for {flow_row['state']}")
    return decision_counts, execution_counts


def draw_state_flow(
    ax,
    state: str,
    flow_row: pd.Series,
    paths_state: pd.DataFrame,
    *,
    y_center: float,
    band_height: float,
) -> None:
    y_bottom = y_center - band_height / 2.0
    y_top = y_center + band_height / 2.0
    source_x = 0.205
    decision_x = 0.500
    execution_x = 0.795
    node_width = 0.118
    segment_gap = 0.0048

    ax.add_patch(
        FancyBboxPatch(
            (0.010, y_bottom - 0.016),
            0.975,
            band_height + 0.032,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.8,
            edgecolor=PANEL_EDGE,
            facecolor=PANEL_FILL,
            zorder=0,
        )
    )

    n_tasks = int(flow_row["n_tasks"])
    mean_confidence = float(flow_row["mean_ai_confidence"]) * 100.0
    ax.text(0.032, y_center + 0.050, state, ha="left", va="center", fontsize=14.4, weight="bold", color=STATE_COLORS[state])
    ax.text(0.032, y_center + 0.005, f"{n_tasks:,} tasks", ha="left", va="center", fontsize=9.6, color=TEXT)
    ax.text(
        0.032,
        y_center - 0.038,
        f"Mean AI confidence: {pct1(mean_confidence)}",
        ha="left",
        va="center",
        fontsize=8.7,
        color=MUTED,
    )

    decision_counts, execution_counts = state_counts(flow_row, paths_state)
    decision_segments = stack_segments(DECISION_ORDER, decision_counts, y_bottom, y_top, gap=segment_gap)
    execution_segments = stack_segments(EXECUTION_ORDER, execution_counts, y_bottom, y_top, gap=segment_gap)
    source_segments = stack_segments(DECISION_ORDER, decision_counts, y_bottom, y_top, gap=segment_gap)

    source_right = source_x + node_width / 2.0
    decision_left = decision_x - node_width / 2.0
    decision_right = decision_x + node_width / 2.0
    execution_left = execution_x - node_width / 2.0

    for decision in DECISION_ORDER:
        add_ribbon(
            ax,
            source_right,
            source_segments[decision][0],
            source_segments[decision][1],
            decision_left,
            decision_segments[decision][0],
            decision_segments[decision][1],
            color=DECISION_COLORS[decision],
            alpha=0.22,
            zorder=2,
        )

    for decision in DECISION_ORDER:
        d0, d1 = decision_segments[decision]
        values_by_execution = {
            execution: int(
                paths_state.loc[
                    paths_state["decision_path"].eq(decision) & paths_state["execution_path"].eq(execution),
                    "n_tasks",
                ].sum()
            )
            for execution in EXECUTION_ORDER
        }
        decision_subsegments = subdivide_segment(EXECUTION_ORDER, values_by_execution, d0, d1)

        for execution in EXECUTION_ORDER:
            e0, e1 = execution_segments[execution]
            values_by_decision = {
                source_decision: int(
                    paths_state.loc[
                        paths_state["decision_path"].eq(source_decision)
                        & paths_state["execution_path"].eq(execution),
                        "n_tasks",
                    ].sum()
                )
                for source_decision in DECISION_ORDER
            }
            execution_subsegments = subdivide_segment(DECISION_ORDER, values_by_decision, e0, e1)
            add_ribbon(
                ax,
                decision_right,
                decision_subsegments[execution][0],
                decision_subsegments[execution][1],
                execution_left,
                execution_subsegments[decision][0],
                execution_subsegments[decision][1],
                color=DECISION_COLORS[decision],
                alpha=0.18,
                zorder=3,
            )

    draw_box(
        ax,
        source_x,
        y_bottom,
        y_top,
        width=node_width,
        facecolor=SOURCE_FILL,
        edgecolor="#CBD5E1",
        linewidth=0.9,
        zorder=7,
    )
    ax.text(
        source_x,
        y_center,
        f"All tasks\n{n_tasks:,}",
        ha="center",
        va="center",
        fontsize=8.6,
        weight="bold",
        color=TEXT,
        linespacing=0.95,
        zorder=9,
    )

    for decision in DECISION_ORDER:
        y0, y1 = decision_segments[decision]
        value_pct = 100.0 * decision_counts[decision] / n_tasks if n_tasks else 0.0
        draw_box(
            ax,
            decision_x,
            y0,
            y1,
            width=node_width,
            facecolor=DECISION_COLORS[decision],
            linewidth=1.1,
            zorder=8,
        )
        draw_node_label(
            ax,
            decision_x,
            y0,
            y1,
            label=DECISION_SHORT_LABELS[decision],
            value_pct=value_pct,
            color=DECISION_COLORS[decision],
        )

    for execution in EXECUTION_ORDER:
        y0, y1 = execution_segments[execution]
        value_pct = 100.0 * execution_counts[execution] / n_tasks if n_tasks else 0.0
        draw_box(
            ax,
            execution_x,
            y0,
            y1,
            width=node_width,
            facecolor=EXECUTION_COLORS[execution],
            linewidth=1.1,
            zorder=8,
        )
        draw_node_label(
            ax,
            execution_x,
            y0,
            y1,
            label=EXECUTION_SHORT_LABELS[execution],
            value_pct=value_pct,
            color=EXECUTION_COLORS[execution],
        )


def make_flow_figure(flow: pd.DataFrame, paths: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(13.7, 8.6))
    ax = fig.add_axes([0.035, 0.115, 0.930, 0.730])
    clean_alluvial_axis(ax)

    fig.text(0.045, 0.946, "State-dependent triadic task flow", fontsize=21, weight="bold", color=TEXT)
    fig.text(
        0.045,
        0.908,
        "Each row follows tasks from AI recommendation to manager response and final execution mode; ribbon width is proportional to task share.",
        fontsize=11.1,
        color="#374151",
    )

    stage_positions = [(0.205, "AI recommendation"), (0.500, "Manager response"), (0.795, "Execution mode")]
    for x, label in stage_positions:
        ax.text(x, 0.975, label, ha="center", va="center", fontsize=11.2, weight="bold", color=TEXT)
    ax.annotate("", xy=(0.405, 0.975), xytext=(0.295, 0.975), arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": MUTED})
    ax.annotate("", xy=(0.700, 0.975), xytext=(0.590, 0.975), arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": MUTED})

    flow_by_state = flow.set_index("state")
    for state, y_center in zip(STATE_ORDER, [0.790, 0.515, 0.240]):
        paths_state = paths[paths["state"].eq(state)].copy()
        draw_state_flow(
            ax,
            state,
            flow_by_state.loc[state],
            paths_state,
            y_center=y_center,
            band_height=0.218,
        )

    handles = [
        Line2D([0], [0], color=DECISION_COLORS["Fully approved"], linewidth=6, label="Approved, no escalation"),
        Line2D([0], [0], color=DECISION_COLORS["Escalated"], linewidth=6, label="Escalated to employee"),
        Line2D([0], [0], color=DECISION_COLORS["Manager changed"], linewidth=6, label="Manager modify/reject"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.040),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=1.0,
        fontsize=9.4,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.text(
        0.045,
        0.086,
        "Ribbon color follows the manager response. Right-hand node colors identify the final execution mode.",
        fontsize=8.8,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.062,
        "Note: manager self-execution is not a separate dataset field; manager involvement is observed through accept, modify/reject, and escalation.",
        fontsize=8.8,
        color=MUTED,
    )
    fig.savefig(FLOW_FIGURE, dpi=160, facecolor="white")
    plt.close(fig)
    return FLOW_FIGURE


def make_acceptance_figure(acceptance: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.05, 4.55))
    markers = {"Aversion": "o", "Neutral": "s", "Appreciation": "^"}

    for state in STATE_ORDER:
        sub = acceptance[acceptance["state"] == state].sort_values("confidence_mid_pct")
        ax.plot(
            sub["confidence_mid_pct"],
            sub["fully_approved_no_escalation_pct"],
            color=STATE_COLORS[state],
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
    flow, acceptance, paths = load_or_build(refresh=args.refresh)
    for path in [make_flow_figure(flow, paths), make_acceptance_figure(acceptance)]:
        print(path)
    print(FLOW_CSV)
    print(PATH_CSV)
    print(ACCEPTANCE_CSV)


if __name__ == "__main__":
    main()
