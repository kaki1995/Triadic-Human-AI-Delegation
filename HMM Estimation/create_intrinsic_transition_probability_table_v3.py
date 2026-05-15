from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
ESTIMATION_TABLE = ANALYSIS_DIR / "estimation_results_publication_format_v3.csv"
XI_CSV = ANALYSIS_DIR / "manager_heterogeneity_factor_v3.csv"
BASELINE_TRANSITION_CSV = ANALYSIS_DIR / "baseline_transition_probabilities_v3.csv"

NUMERIC_CSV = ANALYSIS_DIR / "intrinsic_transition_probabilities_by_manager_state_v3.csv"
TABLE_CSV = ANALYSIS_DIR / "table_intrinsic_transition_probabilities_by_manager_state_v3.csv"
TABLE_PNG = ANALYSIS_DIR / "table_intrinsic_transition_probabilities_by_manager_state_v3.png"

RAW_STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
DISPLAY_STATE_LABELS = {
    "Aversion": "Low (Aversion)",
    "Neutral": "Medium (Neutral)",
    "Appreciation": "High (Appreciation)",
}

TEXT = "#111111"
MUTED = "#4B5563"
HEADER_FILL = "#E8EEF7"
ROW_FILL = "#F3F6FA"
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


def softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(centered)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def load_transition_intercepts() -> pd.DataFrame:
    table = pd.read_csv(ESTIMATION_TABLE)
    state_cols = [col for col in table.columns if col.startswith("State ")]
    alpha_start = table.index[table["Variable"].eq("Transition intercept (alpha)")].tolist()
    if not alpha_start:
        raise ValueError("Could not find 'Transition intercept (alpha)' in estimation results table.")

    alpha_rows = table.iloc[alpha_start[0] + 1 : alpha_start[0] + 1 + len(RAW_STATE_ORDER)].copy()
    if len(alpha_rows) != len(RAW_STATE_ORDER):
        raise ValueError("Could not read a complete transition-intercept matrix.")

    alpha_rows["from_state"] = alpha_rows["Variable"].str.replace("From ", "", regex=False)
    alpha_rows = alpha_rows.set_index("from_state")
    alpha_rows = alpha_rows.loc[RAW_STATE_ORDER, state_cols]
    alpha_rows.columns = RAW_STATE_ORDER
    return alpha_rows.astype(float)


def load_xi_value() -> float:
    if not XI_CSV.exists():
        return np.nan
    xi = pd.read_csv(XI_CSV)
    match = xi.loc[xi["factor"].eq("xi_i"), "value"]
    return float(match.iloc[0]) if len(match) else np.nan


def load_intrinsic_probabilities() -> pd.DataFrame:
    probabilities = pd.read_csv(BASELINE_TRANSITION_CSV, index_col=0)
    probabilities = probabilities.loc[RAW_STATE_ORDER, RAW_STATE_ORDER]
    return probabilities.astype(float)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = load_transition_intercepts()
    xi_i = load_xi_value()
    probabilities = load_intrinsic_probabilities()

    numeric_rows: list[dict[str, float | str]] = []
    for row_idx, from_state in enumerate(RAW_STATE_ORDER):
        for col_idx, to_state in enumerate(RAW_STATE_ORDER):
            intrinsic_probability = float(probabilities.loc[from_state, to_state])
            numeric_rows.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "from_state_display": DISPLAY_STATE_LABELS[from_state],
                    "to_state_display": DISPLAY_STATE_LABELS[to_state],
                    "transition_intercept_alpha": float(alpha.loc[from_state, to_state]),
                    "manager_heterogeneity_xi_i": xi_i,
                    "transition_factor_beta_status": "set_to_zero",
                    "intrinsic_probability_source": "baseline_transition_probabilities_v3.csv",
                    "intrinsic_probability": intrinsic_probability,
                    "intrinsic_probability_pct": float(100.0 * intrinsic_probability),
                }
            )

    numeric_df = pd.DataFrame(numeric_rows)
    display_df = pd.DataFrame(
        {
            "t -> t+1": [DISPLAY_STATE_LABELS[state] for state in RAW_STATE_ORDER],
            **{
                DISPLAY_STATE_LABELS[to_state]: [
                    pct(float(probabilities.loc[from_state, to_state]))
                    for from_state in RAW_STATE_ORDER
                ]
                for col_idx, to_state in enumerate(RAW_STATE_ORDER)
            },
        }
    )
    return numeric_df, display_df


def save_table_image(display_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 3.15))
    ax.axis("off")
    fig.text(
        0.035,
        0.956,
        "Table. Intrinsic Transition Probability by Manager Willingness State",
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
        colWidths=[0.25, 0.25, 0.25, 0.25],
        loc="upper left",
        bbox=[0.035, 0.305, 0.930, 0.475],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.1)
    table.scale(1.0, 1.18)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        if row_idx == 0:
            cell.set_facecolor(HEADER_FILL)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(ROW_FILL if col_idx == 0 else "white")
            if col_idx == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_ha("left")
        cell.get_text().set_color(TEXT)

    fig.text(
        0.035,
        0.125,
        "Note: Entries are Pr(S_t+1 = column state | S_t = row state). Transition-factor effects are removed by setting beta = 0.",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.080,
        "Probabilities are computed as the row-wise softmax of the fitted transition intercepts alpha. "
        "The calibrated manager heterogeneity term xi_i is retained in the numeric export.",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
    )
    fig.savefig(TABLE_PNG, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_style()
    numeric_df, display_df = build_tables()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    numeric_df.to_csv(NUMERIC_CSV, index=False)
    display_df.to_csv(TABLE_CSV, index=False)
    save_table_image(display_df)
    print(NUMERIC_CSV)
    print(TABLE_CSV)
    print(TABLE_PNG)


if __name__ == "__main__":
    main()
