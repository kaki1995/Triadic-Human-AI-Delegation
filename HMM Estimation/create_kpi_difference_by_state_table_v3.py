from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_CSV = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"

LONG_CSV = ANALYSIS_DIR / "kpi_differences_by_manager_state_v3.csv"
TABLE_CSV = ANALYSIS_DIR / "table_kpi_differences_by_manager_state_v3.csv"
TABLE_PNG = ANALYSIS_DIR / "table_kpi_differences_by_manager_state_v3.png"

STATE_ORDER = ["Appreciation", "Neutral", "Aversion"]
KPI_COLUMN = "composite_kpi_score"
KPI_LABEL = "Composite KPI Score"

TEXT = "#111111"
MUTED = "#4B5563"
HEADER_FILL = "#E8EEF7"
SECTION_FILL = "#F3F6FA"
GRID = "#CBD5E1"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "font.family": "Times New Roman",
        }
    )


def stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_difference(value: float, p_value: float) -> str:
    if abs(value) < 0.0000005:
        return "0"
    return f"{value * 100:.3f}%{stars(p_value)}"


def load_panel_state() -> pd.DataFrame:
    post = pd.read_csv(POSTERIOR_CSV, usecols=["manager_id", "period_id", "state_label"])
    panel = pd.read_excel(
        DATA_PATH,
        sheet_name="panel_manager_period_outcomes",
        usecols=["manager_id", "period_id", KPI_COLUMN],
    )
    panel_state = panel.merge(post, on=["manager_id", "period_id"], how="inner")
    panel_state = panel_state[panel_state["state_label"].isin(STATE_ORDER)].copy()
    return panel_state


def welch_p_value(left: pd.Series, right: pd.Series) -> float:
    left_values = left.dropna().to_numpy(dtype=float)
    right_values = right.dropna().to_numpy(dtype=float)
    if len(left_values) < 2 or len(right_values) < 2:
        return np.nan
    result = stats.ttest_ind(left_values, right_values, equal_var=False)
    return float(result.pvalue)


def build_tables(panel_state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, float | int | str]] = []
    display_rows: list[dict[str, str]] = []

    grouped = {
        state: panel_state.loc[panel_state["state_label"].eq(state), KPI_COLUMN].dropna()
        for state in STATE_ORDER
    }
    means = {state: float(values.mean()) for state, values in grouped.items()}
    counts = {state: int(values.size) for state, values in grouped.items()}

    for row_state in STATE_ORDER:
        row: dict[str, str] = {"Composite KPI difference": row_state}
        for col_state in STATE_ORDER:
            diff = means[row_state] - means[col_state]
            p_value = np.nan if row_state == col_state else welch_p_value(grouped[row_state], grouped[col_state])
            long_rows.append(
                {
                    "kpi": KPI_COLUMN,
                    "kpi_label": KPI_LABEL,
                    "row_state": row_state,
                    "column_state": col_state,
                    "row_mean": means[row_state],
                    "column_mean": means[col_state],
                    "difference": diff,
                    "difference_pct": diff * 100.0,
                    "row_n": counts[row_state],
                    "column_n": counts[col_state],
                    "p_value": p_value,
                    "stars": stars(p_value),
                    "formatted_difference": format_difference(diff, p_value),
                }
            )
            row[col_state] = format_difference(diff, p_value)
        display_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    display_df = pd.DataFrame(display_rows)[["Composite KPI difference", *STATE_ORDER]]
    return long_df, display_df


def save_table_image(display_df: pd.DataFrame) -> None:
    fig_height = 2.95
    fig, ax = plt.subplots(figsize=(9.6, fig_height))
    ax.axis("off")
    fig.text(
        0.035,
        0.958,
        "Table 5. Difference in Composite KPI Score Conditioned on Manager Willingness State",
        ha="left",
        va="top",
        fontsize=13.5,
        weight="bold",
        color=TEXT,
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
        loc="upper left",
        bbox=[0.035, 0.250, 0.930, 0.530],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    table.scale(1.0, 1.18)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        if row_idx == 0:
            cell.set_facecolor(HEADER_FILL)
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(TEXT)
        else:
            cell.set_facecolor(SECTION_FILL if col_idx == 0 else "white")
            if col_idx == 0:
                cell.get_text().set_ha("left")
                cell.get_text().set_weight("bold")
            cell.get_text().set_color(TEXT)

    fig.text(
        0.035,
        0.074,
        "Note: Entries are row manager-state mean minus column manager-state mean for composite KPI score, "
        "reported in percentage points. "
        "* p < 0.10; ** p < 0.05; *** p < 0.01.",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.045,
        "Significance stars are based on two-sided Welch's t-tests for mean differences across manager-period observations.",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
    )
    fig.savefig(TABLE_PNG, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_style()
    panel_state = load_panel_state()
    long_df, display_df = build_tables(panel_state)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(LONG_CSV, index=False)
    display_df.to_csv(TABLE_CSV, index=False)
    save_table_image(display_df)
    print(LONG_CSV)
    print(TABLE_CSV)
    print(TABLE_PNG)


if __name__ == "__main__":
    main()
