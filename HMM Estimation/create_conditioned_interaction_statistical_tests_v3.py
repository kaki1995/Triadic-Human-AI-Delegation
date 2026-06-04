from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from create_conditioned_interaction_effect_figure_v3 import (
    AI_AUTHORITY_DISPLAY_CONFIG,
    ANALYSIS_DIR,
    ESCALATION_DISPLAY_CONFIG,
    STATE_ORDER,
    VARIABLE_LABELS,
    estimate_state_models,
    load_panel_data,
    significance_stars,
)


OUTPUT_STEM = "conditioned_interaction_effect_statistical_tests_v3"

TERM_LABELS = {
    "beta0": "Intercept",
    "beta1_x": "X",
    "beta2_x2": "X^2",
    "beta3_y": "Y",
    "beta4_y2": "Y^2",
    "beta5_xy": "X x Y",
    "beta6_c": "Conditioning Benchmark",
    "beta7_xc": "X x Conditioning Benchmark",
    "beta8_yc": "Y x Conditioning Benchmark",
}

MODEL_ROWS = [
    ("r_squared", "R^2"),
    ("f_statistic", "F-statistic"),
    ("aic", "AIC"),
    ("bic", "BIC"),
    ("n_effective", "Effective n"),
]


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


def fmt_number(value: float, decimals: int = 3) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:.3e}"
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def fmt_estimate(value: float, p_value: float | None = None) -> str:
    return f"{fmt_number(value)}{significance_stars(p_value)}"


def collect_tests() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for config in (AI_AUTHORITY_DISPLAY_CONFIG, ESCALATION_DISPLAY_CONFIG):
        data = load_panel_data(config)
        _, _, estimates = estimate_state_models(data, config)
        if estimates.empty:
            raise RuntimeError(f"No empirical estimates produced for {config.outcome}.")
        estimates = estimates.copy()
        estimates.insert(0, "outcome", config.outcome)
        estimates.insert(1, "outcome_label", config.outcome_label)
        estimates.insert(2, "x_variable", config.x_var)
        estimates.insert(3, "x_label", VARIABLE_LABELS.get(config.x_var, config.x_var))
        estimates.insert(4, "y_variable", config.y_var)
        estimates.insert(5, "y_label", VARIABLE_LABELS.get(config.y_var, config.y_var))
        estimates.insert(6, "conditioning_variable", config.conditioning_var)
        estimates.insert(
            7,
            "conditioning_label",
            VARIABLE_LABELS.get(config.conditioning_var, config.conditioning_var),
        )
        estimates["term_label"] = estimates["term"].map(TERM_LABELS).fillna(estimates["term"])
        estimates["stars"] = estimates["p_value"].map(significance_stars)
        rows.append(estimates)
    return pd.concat(rows, ignore_index=True)


def wide_table(tests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome_label, outcome_group in tests.groupby("outcome_label", sort=False):
        for term, term_label in TERM_LABELS.items():
            row: dict[str, object] = {"outcome": outcome_label, "term": term_label}
            for state in STATE_ORDER:
                match = outcome_group[
                    outcome_group["state"].eq(state) & outcome_group["term"].eq(term)
                ]
                if match.empty:
                    row[state] = ""
                else:
                    item = match.iloc[0]
                    row[state] = fmt_estimate(float(item["estimate"]), float(item["p_value"]))
            rows.append(row)

        model_rows = outcome_group[outcome_group["term"].eq("model")]
        for metric, label in MODEL_ROWS:
            row = {"outcome": outcome_label, "term": label}
            for state in STATE_ORDER:
                match = model_rows[model_rows["state"].eq(state)]
                row[state] = fmt_number(float(match.iloc[0][metric])) if not match.empty else ""
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_report(tests: pd.DataFrame, wide: pd.DataFrame) -> str:
    lines = [
        "# Conditioned Interaction-Effect Statistical Tests",
        "",
        "Posterior-weighted OLS models test whether the response-surface terms are "
        "associated with the outcome within each latent state. Observations are "
        "weighted by the HMM posterior probability of belonging to the state.",
        "",
        "Model specification:",
        "",
        "`outcome = beta0 + beta1*X + beta2*X^2 + beta3*Y + beta4*Y^2 + "
        "beta5*X*Y + beta6*C + beta7*X*C + beta8*Y*C + error`",
        "",
        "- X = Team(t-1) vs. Team(t)",
        "- Y = Team vs. Peer Average",
        "- C = Target Attainment",
        "- Significance: * p < 0.10; ** p < 0.05; *** p < 0.01",
        "",
    ]
    for outcome_label in wide["outcome"].drop_duplicates():
        table = wide[wide["outcome"].eq(outcome_label)].drop(columns=["outcome"])
        lines.extend([f"## {outcome_label}", "", table.to_markdown(index=False), ""])

    long_cols = [
        "outcome_label",
        "state",
        "term_label",
        "estimate",
        "p_value",
        "stars",
        "r_squared",
        "f_statistic",
        "aic",
        "bic",
        "n_effective",
    ]
    lines.extend(
        [
            "## Audit Columns",
            "",
            "The long CSV keeps the numeric estimates and p-values for these columns:",
            "",
            ", ".join(long_cols),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    tests = collect_tests()
    wide = wide_table(tests)
    outputs = [
        save_csv(tests, ANALYSIS_DIR / f"{OUTPUT_STEM}.csv"),
        save_csv(wide, ANALYSIS_DIR / f"{OUTPUT_STEM}_publication.csv"),
        save_text(markdown_report(tests, wide), ANALYSIS_DIR / f"{OUTPUT_STEM}.md"),
    ]

    print("Saved conditioned interaction-effect statistical tests:")
    for path in outputs:
        print(f"- {path}")
    print()
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
