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
POSTERIOR_CSV = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"

NUMERIC_CSV = ANALYSIS_DIR / "intrinsic_transition_probabilities_by_manager_state_v3.csv"
TABLE_CSV = ANALYSIS_DIR / "table_intrinsic_transition_probabilities_by_manager_state_v3.csv"
TABLE_PNG = ANALYSIS_DIR / "table_intrinsic_transition_probabilities_by_manager_state_v3.png"
BASELINE_TABLE_CSV = ANALYSIS_DIR / "table_baseline_transition_probabilities_v3.csv"
BASELINE_TABLE_PNG = ANALYSIS_DIR / "table_baseline_transition_probabilities_v3.png"

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


def pct_interval(lower: float, upper: float) -> str:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return "[--]"
    return f"[{pct(lower)}-{pct(upper)}]"


def pct_with_interval(value: float, lower: float, upper: float) -> str:
    return f"{pct(value)} {pct_interval(lower, upper)}"


def image_cell_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    return value.replace(" [", "\n[")


def ci95_for_probability(probability: float, n_origin_transitions: int) -> tuple[float, float]:
    if n_origin_transitions <= 0 or not np.isfinite(probability):
        return np.nan, np.nan
    se = np.sqrt(probability * (1.0 - probability) / n_origin_transitions)
    lower = max(0.0, probability - 1.96 * se)
    upper = min(1.0, probability + 1.96 * se)
    return float(lower), float(upper)


def load_origin_transition_counts() -> dict[str, int]:
    posteriors = pd.read_csv(POSTERIOR_CSV)
    data = posteriors.sort_values(["manager_id", "period_id"]).copy()
    data["from_state"] = data.groupby("manager_id")["state_label"].shift(1)
    transitions = data.dropna(subset=["from_state", "state_label"])
    return transitions["from_state"].value_counts().to_dict()


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


def display_table(
    probabilities: pd.DataFrame,
    numeric_df: pd.DataFrame,
    state_labels: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for from_state in RAW_STATE_ORDER:
        row: dict[str, str] = {"t -> t+1": state_labels[from_state]}
        for to_state in RAW_STATE_ORDER:
            match = numeric_df[
                numeric_df["from_state"].eq(from_state) & numeric_df["to_state"].eq(to_state)
            ].iloc[0]
            row[state_labels[to_state]] = pct_with_interval(
                float(probabilities.loc[from_state, to_state]),
                float(match["intrinsic_probability_ci95_lower"]),
                float(match["intrinsic_probability_ci95_upper"]),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha = load_transition_intercepts()
    xi_i = load_xi_value()
    probabilities = load_intrinsic_probabilities()
    origin_counts = load_origin_transition_counts()

    numeric_rows: list[dict[str, float | str]] = []
    for row_idx, from_state in enumerate(RAW_STATE_ORDER):
        for col_idx, to_state in enumerate(RAW_STATE_ORDER):
            intrinsic_probability = float(probabilities.loc[from_state, to_state])
            origin_count = int(origin_counts.get(from_state, 0))
            ci_lower, ci_upper = ci95_for_probability(intrinsic_probability, origin_count)
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
                    "origin_transition_count_for_ci": origin_count,
                    "intrinsic_probability_ci95_lower": ci_lower,
                    "intrinsic_probability_ci95_upper": ci_upper,
                    "intrinsic_probability_ci95_lower_pct": (
                        float(100.0 * ci_lower) if np.isfinite(ci_lower) else np.nan
                    ),
                    "intrinsic_probability_ci95_upper_pct": (
                        float(100.0 * ci_upper) if np.isfinite(ci_upper) else np.nan
                    ),
                    "ci95_method": (
                        "normal approximation using model baseline probability and "
                        "posterior-decoded origin-state transition count"
                    ),
                }
            )

    numeric_df = pd.DataFrame(numeric_rows)
    intrinsic_display_df = display_table(probabilities, numeric_df, DISPLAY_STATE_LABELS)
    baseline_display_df = display_table(
        probabilities,
        numeric_df,
        {state: state for state in RAW_STATE_ORDER},
    )
    return numeric_df, intrinsic_display_df, baseline_display_df


def save_table_image(display_df: pd.DataFrame) -> None:
    image_df = display_df.copy()
    for col in image_df.columns[1:]:
        image_df[col] = image_df[col].map(image_cell_text)

    fig, ax = plt.subplots(figsize=(10.6, 4.15))
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
        cellText=image_df.values,
        colLabels=image_df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
        loc="upper left",
        bbox=[0.035, 0.315, 0.930, 0.480],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)

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
        "Note: Entries are Pr(S_t+1 = column state | S_t = row state). Brackets are approximate 95% confidence intervals.",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.080,
        "Probabilities are computed as the row-wise softmax of the fitted transition intercepts alpha. "
        "Intervals use the model baseline probability and posterior-decoded origin-state transition counts.",
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
    )
    fig.savefig(TABLE_PNG, dpi=160, facecolor="white")
    plt.close(fig)


def save_baseline_table_image(display_df: pd.DataFrame) -> None:
    image_df = display_df.copy()
    for col in image_df.columns[1:]:
        image_df[col] = image_df[col].map(image_cell_text)

    fig, ax = plt.subplots(figsize=(12.8, 3.8))
    ax.axis("off")
    fig.text(
        0.010,
        0.955,
        "Table 3. Baseline Probability of Intrinsic Propensity to Transition",
        ha="left",
        va="top",
        fontsize=14.5,
        weight="bold",
        color="white",
        bbox={"facecolor": "#111111", "edgecolor": "#111111", "pad": 2.5},
    )

    table = ax.table(
        cellText=image_df.values,
        colLabels=image_df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.28, 0.24, 0.24, 0.24],
        loc="upper left",
        bbox=[0.020, 0.230, 0.960, 0.650],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1.0, 1.45)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#111111")
        cell.set_linewidth(0.7)
        if row_idx == 0:
            cell.set_facecolor("#111111")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white")
            cell.get_text().set_color(TEXT)
            if col_idx == 0:
                cell.get_text().set_ha("left")

    fig.text(
        0.020,
        0.132,
        "Rows are origin states and columns are destination states. Transition covariates are held at their sample means.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=TEXT,
    )
    fig.text(
        0.020,
        0.080,
        "Brackets report approximate 95% confidence intervals using the model baseline probability and posterior-decoded origin-state transition counts.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=TEXT,
    )
    fig.savefig(BASELINE_TABLE_PNG, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_style()
    numeric_df, intrinsic_display_df, baseline_display_df = build_tables()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    numeric_df.to_csv(NUMERIC_CSV, index=False)
    intrinsic_display_df.to_csv(TABLE_CSV, index=False)
    baseline_display_df.to_csv(BASELINE_TABLE_CSV, index=False)
    save_table_image(intrinsic_display_df)
    save_baseline_table_image(baseline_display_df)
    print(NUMERIC_CSV)
    print(TABLE_CSV)
    print(TABLE_PNG)
    print(BASELINE_TABLE_CSV)
    print(BASELINE_TABLE_PNG)


if __name__ == "__main__":
    main()
