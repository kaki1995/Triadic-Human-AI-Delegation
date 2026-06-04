from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
TRANSITION_MATRIX_PATH = ANALYSIS_DIR / "manager_posterior_transition_matrices_v3.csv"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def save_text(text: str, path: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def pct(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{100.0 * value:.1f}%"


def cell_text(row: pd.Series) -> str:
    return f"{pct(row['mean_manager_rate'])} [{pct(row['p025'])}-{pct(row['p975'])}]"


def summarize_transition_distributions(transitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for from_state in STATE_ORDER:
        for to_state in STATE_ORDER:
            subset = transitions[
                transitions["from_state"].eq(from_state)
                & transitions["to_state"].eq(to_state)
                & transitions["from_state_transition_count"].gt(0)
            ].copy()
            rates = pd.to_numeric(subset["transition_rate"], errors="coerce").dropna()
            origin_counts = pd.to_numeric(
                subset["from_state_transition_count"], errors="coerce"
            ).fillna(0)
            transition_counts = pd.to_numeric(
                subset["transition_count"], errors="coerce"
            ).fillna(0)

            if rates.empty:
                row = {
                    "from_state": from_state,
                    "to_state": to_state,
                    "n_managers_with_origin_state": 0,
                    "total_origin_transitions": 0,
                    "total_cell_transitions": 0,
                    "pooled_transition_rate": np.nan,
                    "mean_manager_rate": np.nan,
                    "median_manager_rate": np.nan,
                    "sd_manager_rate": np.nan,
                    "p025": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "p975": np.nan,
                }
            else:
                total_origin = float(origin_counts.sum())
                total_cell = float(transition_counts.sum())
                row = {
                    "from_state": from_state,
                    "to_state": to_state,
                    "n_managers_with_origin_state": int(rates.shape[0]),
                    "total_origin_transitions": int(total_origin),
                    "total_cell_transitions": int(total_cell),
                    "pooled_transition_rate": total_cell / total_origin if total_origin > 0 else np.nan,
                    "mean_manager_rate": float(rates.mean()),
                    "median_manager_rate": float(rates.median()),
                    "sd_manager_rate": float(rates.std(ddof=1)),
                    "p025": float(rates.quantile(0.025)),
                    "p25": float(rates.quantile(0.25)),
                    "p75": float(rates.quantile(0.75)),
                    "p975": float(rates.quantile(0.975)),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def formatted_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for from_state in STATE_ORDER:
        row: dict[str, str] = {"State at t-1": from_state}
        for to_state in STATE_ORDER:
            match = summary[
                summary["from_state"].eq(from_state)
                & summary["to_state"].eq(to_state)
            ].iloc[0]
            row[f"State at t: {to_state}"] = cell_text(match)
        rows.append(row)
    return pd.DataFrame(rows)


def stay_propensity_summary(summary: pd.DataFrame) -> pd.DataFrame:
    diagonal = summary[summary["from_state"].eq(summary["to_state"])].copy()
    diagonal = diagonal.rename(
        columns={
            "from_state": "state",
            "mean_manager_rate": "mean_stay_propensity",
            "median_manager_rate": "median_stay_propensity",
            "sd_manager_rate": "sd_stay_propensity",
            "p025": "p025_stay_propensity",
            "p25": "p25_stay_propensity",
            "p75": "p75_stay_propensity",
            "p975": "p975_stay_propensity",
        }
    )
    columns = [
        "state",
        "n_managers_with_origin_state",
        "total_origin_transitions",
        "total_cell_transitions",
        "pooled_transition_rate",
        "mean_stay_propensity",
        "median_stay_propensity",
        "sd_stay_propensity",
        "p025_stay_propensity",
        "p25_stay_propensity",
        "p75_stay_propensity",
        "p975_stay_propensity",
    ]
    return diagonal[columns].reset_index(drop=True)


def markdown_table(formatted: pd.DataFrame, stay_summary: pd.DataFrame) -> str:
    lines = [
        "# Posterior Transition Distribution Table",
        "",
        "Cells report the mean manager-level decoded posterior transition rate,",
        "with the 2.5th and 97.5th percentiles of the across-manager distribution",
        "in brackets. Managers are included in a row only when they are observed",
        "at least once in the corresponding origin state.",
        "",
        "## Mean Manager-Level Posterior Transitions",
        "",
        formatted.to_markdown(index=False),
        "",
        "## Posterior Propensity to Stay in Each State",
        "",
        stay_summary.assign(
            pooled_transition_rate=lambda df: df["pooled_transition_rate"].map(pct),
            mean_stay_propensity=lambda df: df["mean_stay_propensity"].map(pct),
            median_stay_propensity=lambda df: df["median_stay_propensity"].map(pct),
            sd_stay_propensity=lambda df: df["sd_stay_propensity"].map(pct),
            p025_stay_propensity=lambda df: df["p025_stay_propensity"].map(pct),
            p25_stay_propensity=lambda df: df["p25_stay_propensity"].map(pct),
            p75_stay_propensity=lambda df: df["p75_stay_propensity"].map(pct),
            p975_stay_propensity=lambda df: df["p975_stay_propensity"].map(pct),
        ).to_markdown(index=False),
        "",
        "Note: these are posterior-decoded manager-level transition distributions,",
        "not MCMC posterior draws.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    transitions = pd.read_csv(TRANSITION_MATRIX_PATH)
    summary = summarize_transition_distributions(transitions)
    formatted = formatted_matrix(summary)
    stay_summary = stay_propensity_summary(summary)

    outputs = [
        save_csv(summary, ANALYSIS_DIR / "posterior_transition_distribution_summary_v3.csv"),
        save_csv(formatted, ANALYSIS_DIR / "posterior_transition_distribution_table_v3.csv"),
        save_csv(stay_summary, ANALYSIS_DIR / "posterior_stay_propensity_distribution_v3.csv"),
        save_text(
            markdown_table(formatted, stay_summary),
            ANALYSIS_DIR / "posterior_transition_distribution_table_v3.md",
        ),
    ]

    print("Saved posterior transition distribution outputs:")
    for path in outputs:
        print(f"- {path}")
    print()
    print(formatted.to_string(index=False))


if __name__ == "__main__":
    main()
