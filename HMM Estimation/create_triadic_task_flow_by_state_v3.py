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
FLOW_FIGURE = ANALYSIS_DIR / "figure_triadic_task_flow_by_state_v3_misq.png"
FLOW_FIGURE_PDF = ANALYSIS_DIR / "figure_triadic_task_flow_by_state_v3_misq.pdf"
ACCEPTANCE_FIGURE = ANALYSIS_DIR / "figure_manager_acceptance_by_ai_confidence_v3.png"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
DECISION_ORDER = ["Fully approved", "Escalated", "Manager changed"]
EXECUTION_ORDER = ["AI", "Joint", "Employee"]
TIME_SPECIFIC_PERIODS = [1, 13, 26]

DECISION_SHORT_LABELS = {
    "Fully approved": "Approved",
    "Escalated": "Escalated",
    "Manager changed": "Overridden",
}
EXECUTION_SHORT_LABELS = {
    "AI": "AI",
    "Joint": "Joint",
    "Employee": "Employee",
}

TEXT = "#111111"
MUTED = "#667085"
GRID = "#D9D9D9"
PANEL_FILL = "#FFFFFF"
PANEL_EDGE = "#CBD5E1"
SOURCE_FILL = "#F8FAFC"
SOURCE_EDGE = "#93C5FD"
RULE = "#64748B"
MANAGER_FILL = "#EAF6FF"
BLUE = "#0057FF"
GREEN = "#00C853"
ORANGE = "#FF6B00"
GRAY = "#CBD5E1"
DARK_GRAY = "#94A3B8"
PALE_BLUE = "#BFDBFE"
PALE_GREEN = "#BBF7D0"
PALE_ORANGE = "#FED7AA"

STATE_COLORS = {
    "Aversion": ORANGE,
    "Neutral": BLUE,
    "Appreciation": GREEN,
}
DECISION_COLORS = {
    "Fully approved": PALE_GREEN,
    "Escalated": PALE_ORANGE,
    "Manager changed": GRAY,
}
EXECUTION_COLORS = {
    "AI": PALE_BLUE,
    "Joint": MANAGER_FILL,
    "Employee": "#D7DEE8",
}
STATE_PANEL_LABELS = {
    "Aversion": "Managers in\nAversion state",
    "Neutral": "Managers in\nNeutral state",
    "Appreciation": "Managers in\nAppreciation state",
}
STATE_PANEL_PREFIXES = {
    "Aversion": "(a)",
    "Neutral": "(b)",
    "Appreciation": "(c)",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.0,
            "axes.grid": False,
        }
    )


def pct(value: float) -> str:
    return f"{value:.0f}%"


def pct1(value: float) -> str:
    return f"{value:.1f}%"


def contrast_text(facecolor: str) -> str:
    if not facecolor.startswith("#") or len(facecolor) != 7:
        return TEXT
    red = int(facecolor[1:3], 16) / 255.0
    green = int(facecolor[3:5], 16) / 255.0
    blue = int(facecolor[5:7], 16) / 255.0
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.46 else TEXT


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


def summarize_flow_and_paths(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []

    for state in STATE_ORDER:
        g = tasks[tasks["state_label"] == state]
        n = len(g)
        fully_approved = g["decision_path"].eq("Fully approved")
        escalated = g["decision_path"].eq("Escalated")
        manager_changed = g["decision_path"].eq("Manager changed")

        def mean_pct(series: pd.Series) -> float:
            return 100.0 * float(series.mean()) if n else 0.0

        rows.append(
            {
                "state": state,
                "n_tasks": int(n),
                "fully_approved_no_escalation_n": int(fully_approved.sum()),
                "fully_approved_no_escalation_pct": mean_pct(fully_approved),
                "escalated_to_employee_n": int(escalated.sum()),
                "escalated_to_employee_pct": mean_pct(escalated),
                "manager_override_no_escalation_n": int(manager_changed.sum()),
                "manager_override_no_escalation_pct": mean_pct(manager_changed),
                "manager_accept_n": int(g["manager_action"].eq("accept").sum()),
                "manager_accept_pct": mean_pct(g["manager_action"].eq("accept")),
                "manager_modify_n": int(g["manager_action"].eq("modify").sum()),
                "manager_modify_pct": mean_pct(g["manager_action"].eq("modify")),
                "manager_reject_n": int(g["manager_action"].eq("reject").sum()),
                "manager_reject_pct": mean_pct(g["manager_action"].eq("reject")),
                "employee_execution_n": int(g["execution_path"].eq("Employee").sum()),
                "employee_execution_pct": mean_pct(g["execution_path"].eq("Employee")),
                "ai_execution_n": int(g["execution_path"].eq("AI").sum()),
                "ai_execution_pct": mean_pct(g["execution_path"].eq("AI")),
                "joint_execution_n": int(g["execution_path"].eq("Joint").sum()),
                "joint_execution_pct": mean_pct(g["execution_path"].eq("Joint")),
                "employee_override_during_execution_n": int(g["employee_override_during_execution"].eq(1).sum()),
                "employee_override_during_execution_pct": mean_pct(g["employee_override_during_execution"].eq(1)),
                "mean_ai_confidence": float(g["ai_confidence"].mean()) if n else np.nan,
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
    return flow, paths


def build_time_specific_flow(period_id: int, tasks: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tasks is None:
        tasks = add_flow_categories(read_task_data())
    period_tasks = tasks[tasks["period_id"].eq(period_id)].copy()
    if period_tasks.empty:
        raise ValueError(f"No task episodes found for period_id={period_id}")
    return summarize_flow_and_paths(period_tasks)


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
    radius: float = 0.002,
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
    small_threshold: float = 0.030,
) -> None:
    if y_top <= y_bottom:
        return
    height = y_top - y_bottom
    if height >= small_threshold:
        text = f"{label}\n({pct(value_pct)})"
        fontsize = 6.5
    else:
        text = f"({pct(value_pct)})"
        fontsize = 6.4
    text_color = contrast_text(color)
    ax.text(
        x_center,
        (y_bottom + y_top) / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight="normal",
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
    source_x = 0.248
    decision_x = 0.525
    execution_x = 0.805
    node_width = 0.104
    segment_gap = 0.0048

    ax.add_patch(
        FancyBboxPatch(
            (0.018, y_bottom - 0.013),
            0.958,
            band_height + 0.026,
            boxstyle="round,pad=0,rounding_size=0.002",
            linewidth=0.65,
            edgecolor=PANEL_EDGE,
            facecolor=PANEL_FILL,
            zorder=0,
        )
    )

    n_tasks = int(flow_row["n_tasks"])
    ax.text(
        0.035,
        y_center + 0.012,
        STATE_PANEL_PREFIXES[state],
        ha="left",
        va="center",
        fontsize=6.8,
        weight="normal",
        color=TEXT,
        zorder=10,
    )
    ax.text(
        0.058,
        y_center,
        STATE_PANEL_LABELS[state],
        ha="left",
        va="center",
        fontsize=6.8,
        weight="normal",
        color=TEXT,
        linespacing=1.02,
        zorder=10,
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
            alpha=0.42,
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
                alpha=0.34,
                zorder=3,
            )

    draw_box(
        ax,
        source_x,
        y_bottom,
        y_top,
        width=node_width,
        facecolor=SOURCE_FILL,
        edgecolor=SOURCE_EDGE,
        linewidth=0.65,
        zorder=7,
    )
    ax.text(
        source_x,
        y_center,
        f"Task\n({n_tasks:,})",
        ha="center",
        va="center",
        fontsize=6.4,
        weight="normal",
        color=TEXT,
        linespacing=0.96,
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
            edgecolor="white",
            linewidth=0.8,
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
            edgecolor="white",
            linewidth=0.8,
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


def make_flow_figure(
    flow: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    output_png: Path = FLOW_FIGURE,
    output_pdf: Path = FLOW_FIGURE_PDF,
    title: str = "State-dependent triadic task flow",
    subtitle: str = "Ribbon width shows the share of tasks moving through each response and execution path.",
    note: str = "Note. Counts in parentheses are task episodes pooled across all planning cycles and assigned to each latent state; node labels report within-state task shares.",
) -> Path:
    fig = plt.figure(figsize=(7.40, 5.65))
    ax = fig.add_axes([0.045, 0.185, 0.910, 0.705])
    clean_alluvial_axis(ax)

    fig.text(0.045, 0.957, title, fontsize=10.1, weight="normal", color=TEXT)
    fig.text(
        0.045,
        0.926,
        subtitle,
        fontsize=6.7,
        color=MUTED,
    )

    stage_positions = [(0.248, "AI recommendations"), (0.525, "Managers' decisions"), (0.805, "Task execution")]
    for x, label in stage_positions:
        ax.text(x, 0.972, label, ha="center", va="center", fontsize=6.7, weight="normal", color=TEXT)
    ax.annotate("", xy=(0.425, 0.972), xytext=(0.345, 0.972), arrowprops={"arrowstyle": "-|>", "lw": 0.65, "color": RULE})
    ax.annotate("", xy=(0.710, 0.972), xytext=(0.630, 0.972), arrowprops={"arrowstyle": "-|>", "lw": 0.65, "color": RULE})

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
        Line2D([0], [0], color=DECISION_COLORS["Fully approved"], linewidth=5, label="Approved"),
        Line2D([0], [0], color=DECISION_COLORS["Escalated"], linewidth=5, label="Escalated"),
        Line2D([0], [0], color=DECISION_COLORS["Manager changed"], linewidth=5, label="Overridden"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.095),
        ncol=3,
        frameon=False,
        facecolor="white",
        framealpha=1.0,
        fontsize=6.4,
        handlelength=1.7,
        columnspacing=1.6,
    )
    fig.text(
        0.045,
        0.058,
        note,
        fontsize=5.9,
        color=MUTED,
    )
    fig.savefig(output_png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(output_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_png


def time_specific_figure_paths(period_id: int) -> tuple[Path, Path]:
    stem = f"figure_triadic_task_flow_by_state_v3_misq_t{period_id:02d}"
    return ANALYSIS_DIR / f"{stem}.png", ANALYSIS_DIR / f"{stem}.pdf"


def make_time_specific_flow_figure(period_id: int, tasks: pd.DataFrame | None = None) -> Path:
    flow, paths = build_time_specific_flow(period_id, tasks=tasks)
    output_png, output_pdf = time_specific_figure_paths(period_id)
    return make_flow_figure(
        flow,
        paths,
        output_png=output_png,
        output_pdf=output_pdf,
        title=f"State-dependent triadic task flow at t = {period_id}",
        subtitle="Ribbon width shows the share of task episodes in this planning cycle.",
        note=(
            f"Note. Counts in parentheses are task episodes from planning cycle t = {period_id} "
            "assigned to each latent state; node labels report within-state task shares."
        ),
    )


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
    parser.add_argument(
        "--time-specific",
        action="store_true",
        help="Also render period-specific task-flow snapshots.",
    )
    parser.add_argument(
        "--time-periods",
        nargs="+",
        type=int,
        default=TIME_SPECIFIC_PERIODS,
        help="Planning cycles to render when --time-specific is set.",
    )
    args = parser.parse_args()

    setup_style()
    flow, acceptance, paths = load_or_build(refresh=args.refresh)
    for path in [make_flow_figure(flow, paths), make_acceptance_figure(acceptance)]:
        print(path)
    if args.time_specific:
        tasks = add_flow_categories(read_task_data())
        for period_id in args.time_periods:
            print(make_time_specific_flow_figure(period_id, tasks=tasks))
            print(time_specific_figure_paths(period_id)[1])
    print(FLOW_CSV)
    print(PATH_CSV)
    print(ACCEPTANCE_CSV)


if __name__ == "__main__":
    main()
