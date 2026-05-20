from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_CSV = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"

MANAGER_SUMMARY_CSV = ANALYSIS_DIR / "regional_context_manager_level_summary_v3.csv"
REGION_SUMMARY_CSV = ANALYSIS_DIR / "regional_context_region_summary_v3.csv"
WELCH_TESTS_CSV = ANALYSIS_DIR / "regional_context_welch_tests_v3.csv"
PAIRWISE_CSV = ANALYSIS_DIR / "regional_context_pairwise_welch_v3.csv"
STATE_COMPOSITION_CSV = ANALYSIS_DIR / "regional_context_state_composition_v3.csv"
TRANSITION_RATES_CSV = ANALYSIS_DIR / "regional_context_transition_rates_v3.csv"
TRANSITION_CHI2_CSV = ANALYSIS_DIR / "regional_context_transition_chi2_v3.csv"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
STATE_SCORE = {"Aversion": 0, "Neutral": 1, "Appreciation": 2}

PANEL_COLUMNS = [
    "manager_id",
    "period_id",
    "region",
    "ai_authority_share",
    "escalation_share",
    "composite_kpi_score",
    "target_attainment",
]

TEST_METRICS = [
    "ai_authority_share_mean",
    "escalation_share_mean",
    "composite_kpi_score_mean",
    "target_attainment_rate",
    "share_aversion",
    "share_neutral",
    "share_appreciation",
    "persistence_rate",
    "upward_transition_rate",
    "downward_transition_rate",
]

PAIRWISE_METRICS = [
    "ai_authority_share_mean",
    "escalation_share_mean",
    "composite_kpi_score_mean",
    "share_aversion",
    "share_appreciation",
    "upward_transition_rate",
    "downward_transition_rate",
]


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def welch_anova(groups: list[np.ndarray]) -> tuple[float, float, float, float]:
    valid = [g[np.isfinite(g)] for g in groups if len(g[np.isfinite(g)]) >= 2]
    k = len(valid)
    if k < 2:
        return np.nan, np.nan, np.nan, np.nan

    n = np.array([len(g) for g in valid], dtype=float)
    means = np.array([float(np.mean(g)) for g in valid])
    variances = np.array([float(np.var(g, ddof=1)) for g in valid])
    variances = np.where(variances <= 0, np.nan, variances)
    if np.isnan(variances).any():
        return np.nan, np.nan, np.nan, np.nan

    weights = n / variances
    weight_sum = weights.sum()
    weighted_mean = float(np.sum(weights * means) / weight_sum)
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2) / (k - 1))
    correction_sum = float(np.sum(((1.0 - weights / weight_sum) ** 2) / (n - 1.0)))
    denominator = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * correction_sum
    f_stat = numerator / denominator
    df1 = float(k - 1)
    df2 = float((k**2 - 1.0) / (3.0 * correction_sum)) if correction_sum > 0 else np.inf
    p_value = float(stats.f.sf(f_stat, df1, df2))
    return f_stat, df1, df2, p_value


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    df = pd.DataFrame({"value": values, "group": groups}).dropna()
    if df.empty:
        return np.nan
    grand_mean = df["value"].mean()
    ss_between = sum(len(g) * (g["value"].mean() - grand_mean) ** 2 for _, g in df.groupby("group"))
    ss_total = float(((df["value"] - grand_mean) ** 2).sum())
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.array(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted.tolist()


def load_panel() -> pd.DataFrame:
    panel = pd.read_excel(DATA_PATH, sheet_name="panel_manager_period", usecols=PANEL_COLUMNS)
    posterior = pd.read_csv(
        POSTERIOR_CSV,
        usecols=[
            "manager_id",
            "period_id",
            "state_label",
            "state_order",
            "gamma_state_1",
            "gamma_state_2",
            "gamma_state_3",
        ],
    )
    merged = panel.merge(posterior, on=["manager_id", "period_id"], how="inner")
    if merged.empty:
        raise ValueError("No rows after merging panel data with posterior state assignments.")
    return merged


def build_manager_summary(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for state in STATE_ORDER:
        panel[f"is_{state.lower()}"] = panel["state_label"].eq(state).astype(float)

    manager_summary = (
        panel.groupby(["manager_id", "region"], observed=True)
        .agg(
            n_periods=("period_id", "size"),
            ai_authority_share_mean=("ai_authority_share", "mean"),
            escalation_share_mean=("escalation_share", "mean"),
            composite_kpi_score_mean=("composite_kpi_score", "mean"),
            target_attainment_rate=("target_attainment", "mean"),
            share_aversion=("is_aversion", "mean"),
            share_neutral=("is_neutral", "mean"),
            share_appreciation=("is_appreciation", "mean"),
        )
        .reset_index()
    )

    transitions = panel.sort_values(["manager_id", "period_id"]).copy()
    transitions["next_state"] = transitions.groupby("manager_id")["state_label"].shift(-1)
    transitions = transitions[transitions["next_state"].notna()].copy()
    transitions["state_score"] = transitions["state_label"].map(STATE_SCORE)
    transitions["next_state_score"] = transitions["next_state"].map(STATE_SCORE)
    transitions["persistence"] = transitions["state_label"].eq(transitions["next_state"]).astype(float)
    transitions["upward_transition"] = (transitions["next_state_score"] > transitions["state_score"]).astype(float)
    transitions["downward_transition"] = (transitions["next_state_score"] < transitions["state_score"]).astype(float)

    transition_summary = (
        transitions.groupby("manager_id", observed=True)
        .agg(
            n_transitions=("period_id", "size"),
            persistence_rate=("persistence", "mean"),
            upward_transition_rate=("upward_transition", "mean"),
            downward_transition_rate=("downward_transition", "mean"),
        )
        .reset_index()
    )

    return manager_summary.merge(transition_summary, on="manager_id", how="left")


def build_region_summary(manager_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in manager_summary.groupby("region", observed=True):
        row: dict[str, float | int | str] = {
            "region": region,
            "n_managers": int(group["manager_id"].nunique()),
        }
        for metric in TEST_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("region")


def build_welch_tests(manager_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TEST_METRICS:
        groups = [g[metric].dropna().to_numpy(dtype=float) for _, g in manager_summary.groupby("region", observed=True)]
        f_stat, df1, df2, p_value = welch_anova(groups)
        rows.append(
            {
                "metric": metric,
                "welch_f": f_stat,
                "df1": df1,
                "df2": df2,
                "p_value": p_value,
                "significance": stars(p_value),
                "eta_squared_region": eta_squared(manager_summary[metric], manager_summary["region"]),
                "unit_of_analysis": "manager",
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_tests(manager_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regions = sorted(manager_summary["region"].dropna().unique())
    for metric in PAIRWISE_METRICS:
        metric_rows = []
        for left, right in combinations(regions, 2):
            left_values = manager_summary.loc[manager_summary["region"].eq(left), metric].dropna().to_numpy(dtype=float)
            right_values = manager_summary.loc[manager_summary["region"].eq(right), metric].dropna().to_numpy(dtype=float)
            result = stats.ttest_ind(left_values, right_values, equal_var=False)
            metric_rows.append(
                {
                    "metric": metric,
                    "region_1": left,
                    "region_2": right,
                    "region_1_mean": float(np.mean(left_values)),
                    "region_2_mean": float(np.mean(right_values)),
                    "difference_region_1_minus_region_2": float(np.mean(left_values) - np.mean(right_values)),
                    "welch_t": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, p_adj in zip(metric_rows, adjusted):
            row["p_value_holm"] = p_adj
            row["significance_holm"] = stars(p_adj)
            rows.append(row)
    return pd.DataFrame(rows)


def build_state_composition(panel: pd.DataFrame) -> pd.DataFrame:
    counts = (
        panel.groupby(["region", "state_label"], observed=True)
        .size()
        .rename("n_manager_periods")
        .reset_index()
    )
    totals = counts.groupby("region")["n_manager_periods"].transform("sum")
    counts["share_within_region"] = counts["n_manager_periods"] / totals
    return counts.sort_values(["region", "state_label"])


def build_transition_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transitions = panel.sort_values(["manager_id", "period_id"]).copy()
    transitions["next_state"] = transitions.groupby("manager_id")["state_label"].shift(-1)
    transitions = transitions[transitions["next_state"].notna()].copy()

    rates = (
        transitions.groupby(["region", "state_label", "next_state"], observed=True)
        .size()
        .rename("n_transitions")
        .reset_index()
        .rename(columns={"state_label": "from_state", "next_state": "to_state"})
    )
    complete = pd.MultiIndex.from_product(
        [sorted(panel["region"].unique()), STATE_ORDER, STATE_ORDER],
        names=["region", "from_state", "to_state"],
    )
    rates = rates.set_index(["region", "from_state", "to_state"]).reindex(complete, fill_value=0).reset_index()
    totals = rates.groupby(["region", "from_state"])["n_transitions"].transform("sum")
    rates["share_from_state_region"] = np.where(totals.gt(0), rates["n_transitions"] / totals, np.nan)

    chi_rows = []
    for from_state in STATE_ORDER:
        sub = transitions[transitions["state_label"].eq(from_state)].copy()
        table = pd.crosstab(sub["region"], sub["next_state"]).reindex(columns=STATE_ORDER, fill_value=0)
        chi2, p_value, dof, _ = stats.chi2_contingency(table)
        n = int(table.to_numpy().sum())
        min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n > 0 and min_dim > 0 else np.nan
        chi_rows.append(
            {
                "from_state": from_state,
                "chi2": float(chi2),
                "df": int(dof),
                "p_value": float(p_value),
                "significance": stars(float(p_value)),
                "cramers_v": cramers_v,
                "n_transitions": n,
                "unit_of_analysis": "manager_period_transition",
            }
        )

    return rates, pd.DataFrame(chi_rows)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    manager_summary = build_manager_summary(panel)
    region_summary = build_region_summary(manager_summary)
    welch_tests = build_welch_tests(manager_summary)
    pairwise_tests = build_pairwise_tests(manager_summary)
    state_composition = build_state_composition(panel)
    transition_rates, transition_chi2 = build_transition_tables(panel)

    manager_summary.to_csv(MANAGER_SUMMARY_CSV, index=False)
    region_summary.to_csv(REGION_SUMMARY_CSV, index=False)
    welch_tests.to_csv(WELCH_TESTS_CSV, index=False)
    pairwise_tests.to_csv(PAIRWISE_CSV, index=False)
    state_composition.to_csv(STATE_COMPOSITION_CSV, index=False)
    transition_rates.to_csv(TRANSITION_RATES_CSV, index=False)
    transition_chi2.to_csv(TRANSITION_CHI2_CSV, index=False)

    for path in [
        MANAGER_SUMMARY_CSV,
        REGION_SUMMARY_CSV,
        WELCH_TESTS_CSV,
        PAIRWISE_CSV,
        STATE_COMPOSITION_CSV,
        TRANSITION_RATES_CSV,
        TRANSITION_CHI2_CSV,
    ]:
        print(path)


if __name__ == "__main__":
    main()
