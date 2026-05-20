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

SITE_MAPPING_CSV = ANALYSIS_DIR / "macro_region_site_mapping_v3.csv"
MANAGER_MAPPING_CSV = ANALYSIS_DIR / "macro_region_manager_mapping_v3.csv"
MANAGER_SUMMARY_CSV = ANALYSIS_DIR / "macro_region_context_manager_level_summary_v3.csv"
REGION_SUMMARY_CSV = ANALYSIS_DIR / "macro_region_context_region_summary_v3.csv"
WELCH_TESTS_CSV = ANALYSIS_DIR / "macro_region_context_welch_tests_v3.csv"
PAIRWISE_CSV = ANALYSIS_DIR / "macro_region_context_pairwise_welch_v3.csv"
STATE_COMPOSITION_CSV = ANALYSIS_DIR / "macro_region_context_state_composition_v3.csv"
TRANSITION_RATES_CSV = ANALYSIS_DIR / "macro_region_context_transition_rates_v3.csv"
TRANSITION_CHI2_CSV = ANALYSIS_DIR / "macro_region_context_transition_chi2_v3.csv"
APPENDIX_TABLE_CSV = ANALYSIS_DIR / "macro_region_context_appendix_table_v3.csv"

GROUP_COL = "macro_region"
STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
STATE_SCORE = {"Aversion": 0, "Neutral": 1, "Appreciation": 2}

REGION_DESIGN = {
    "Europe, Middle East, Africa": {
        "n_managers": 500,
        "country_plants": {
            "Germany": 8,
            "United Kingdom": 4,
            "Austria": 2,
            "Hungary": 1,
            "South Africa": 1,
            "Netherlands": 2,
            "Italy": 2,
        },
    },
    "Asia-Pacific": {
        "n_managers": 220,
        "country_plants": {
            "China": 2,
            "India": 2,
            "Thailand": 1,
            "Indonesia": 1,
            "Malaysia": 1,
        },
    },
    "North, Central, South America": {
        "n_managers": 155,
        "country_plants": {
            "United States": 1,
            "Mexico": 2,
            "Brazil": 2,
        },
    },
}

SCENARIO_SEED = 20260519
SCENARIO_APPRECIATION_WEIGHT = 0.05
SCENARIO_AVERSION_WEIGHT = -0.10

PANEL_COLUMNS = [
    "manager_id",
    "period_id",
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

PAIRWISE_METRICS = TEST_METRICS

APPENDIX_ROWS = [
    ("Delegation outcomes", "AI Authority Share", "ai_authority_share_mean"),
    ("Delegation outcomes", "Escalation Share", "escalation_share_mean"),
    ("Delegation outcomes", "Composite KPI Score", "composite_kpi_score_mean"),
    ("Delegation outcomes", "Target Attainment", "target_attainment_rate"),
    ("Latent-state composition", "Aversion-state share", "share_aversion"),
    ("Latent-state composition", "Neutral-state share", "share_neutral"),
    ("Latent-state composition", "Appreciation-state share", "share_appreciation"),
    ("Transition dynamics", "Persistence rate", "persistence_rate"),
    ("Transition dynamics", "Upward transition rate", "upward_transition_rate"),
    ("Transition dynamics", "Downward transition rate", "downward_transition_rate"),
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


def build_plant_design() -> pd.DataFrame:
    rows = []
    plant_counter = 1
    for macro_region, region_design in REGION_DESIGN.items():
        for country, n_plants in region_design["country_plants"].items():
            for _ in range(n_plants):
                rows.append(
                    {
                        "scenario_plant_id": f"P{plant_counter:03d}",
                        GROUP_COL: macro_region,
                        "country_scenario": country,
                    }
                )
                plant_counter += 1
    plant_design = pd.DataFrame(rows)
    expected_plants = sum(
        sum(region_design["country_plants"].values())
        for region_design in REGION_DESIGN.values()
    )
    if len(plant_design) != expected_plants:
        raise ValueError("Plant design does not match configured plant count.")
    return plant_design


def assign_plants(
    rng: np.random.Generator,
    manager_mapping: pd.DataFrame,
    plant_design: pd.DataFrame,
) -> pd.DataFrame:
    assigned = []
    for macro_region, group in manager_mapping.groupby(GROUP_COL, observed=True):
        plant_subset = plant_design[plant_design[GROUP_COL].eq(macro_region)].reset_index(drop=True)
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)
        splits = np.array_split(shuffled.index.to_numpy(), len(plant_subset))
        for plant, split_index in zip(plant_subset.to_dict("records"), splits):
            block = shuffled.loc[split_index].copy()
            block["scenario_plant_id"] = plant["scenario_plant_id"]
            block["country_scenario"] = plant["country_scenario"]
            assigned.append(block)
    return pd.concat(assigned, ignore_index=True)


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    if sd == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / sd


def build_manager_mapping(
    managers: pd.DataFrame,
    plant_design: pd.DataFrame,
    posterior: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(SCENARIO_SEED)
    managers = managers[["manager_id", "baseline_ai_attitude"]].drop_duplicates().copy()
    if len(managers) != sum(region_design["n_managers"] for region_design in REGION_DESIGN.values()):
        raise ValueError("Configured macro-region manager counts must sum to the number of managers.")

    state_profile = (
        posterior.assign(
            posterior_appreciation_share=posterior["state_label"].eq("Appreciation").astype(float),
            posterior_aversion_share=posterior["state_label"].eq("Aversion").astype(float),
        )
        .groupby("manager_id", observed=True)
        .agg(
            posterior_appreciation_share=("posterior_appreciation_share", "mean"),
            posterior_aversion_share=("posterior_aversion_share", "mean"),
        )
        .reset_index()
    )
    managers = managers.merge(state_profile, on="manager_id", how="left")
    managers["scenario_region_score"] = (
        zscore(managers["baseline_ai_attitude"])
        + SCENARIO_APPRECIATION_WEIGHT * zscore(managers["posterior_appreciation_share"])
        + SCENARIO_AVERSION_WEIGHT * zscore(managers["posterior_aversion_share"])
    )

    ranked = managers.sort_values("scenario_region_score", ascending=False).reset_index(drop=True)
    reference_region = "Europe, Middle East, Africa"
    americas_region = "North, Central, South America"
    asia_target = REGION_DESIGN["Asia-Pacific"]["n_managers"]
    americas_target = REGION_DESIGN[americas_region]["n_managers"]
    europe_target = REGION_DESIGN[reference_region]["n_managers"]
    ai_positive_pool_size = asia_target + americas_target

    ai_positive_pool = ranked.iloc[:ai_positive_pool_size].copy()
    europe_pool = ranked.iloc[ai_positive_pool_size:].copy()
    if len(europe_pool) != europe_target:
        raise ValueError(f"Expected {europe_target} Europe managers, found {len(europe_pool)}.")

    asia_ids = set(rng.choice(ai_positive_pool["manager_id"], size=asia_target, replace=False).tolist())
    ai_positive_pool[GROUP_COL] = np.where(
        ai_positive_pool["manager_id"].isin(asia_ids),
        "Asia-Pacific",
        americas_region,
    )
    europe_pool[GROUP_COL] = reference_region

    manager_mapping = pd.concat([ai_positive_pool, europe_pool], ignore_index=True)
    return assign_plants(rng, manager_mapping, plant_design)


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_excel(DATA_PATH, sheet_name="panel_manager_period", usecols=PANEL_COLUMNS)
    managers = pd.read_excel(DATA_PATH, sheet_name="manager_master", usecols=["manager_id", "baseline_ai_attitude"])
    plant_design = build_plant_design()

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
    manager_mapping = build_manager_mapping(managers, plant_design, posterior)

    merged = (
        panel.merge(
            manager_mapping[["manager_id", GROUP_COL, "country_scenario", "scenario_plant_id"]],
            on="manager_id",
            how="left",
        )
        .merge(posterior, on=["manager_id", "period_id"], how="inner")
    )
    if merged.empty:
        raise ValueError("No rows after merging panel data with posterior state assignments.")
    if merged[GROUP_COL].isna().any():
        missing = sorted(merged.loc[merged[GROUP_COL].isna(), "manager_id"].dropna().unique())
        raise ValueError(f"Missing macro-region values after merge: {missing}")
    return merged, plant_design, manager_mapping


def build_manager_summary(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for state in STATE_ORDER:
        panel[f"is_{state.lower()}"] = panel["state_label"].eq(state).astype(float)

    manager_summary = (
        panel.groupby(["manager_id", GROUP_COL], observed=True)
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
    for region, group in manager_summary.groupby(GROUP_COL, observed=True):
        row: dict[str, float | int | str] = {
            GROUP_COL: region,
            "n_managers": int(group["manager_id"].nunique()),
        }
        for metric in TEST_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_COL)


def build_welch_tests(manager_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TEST_METRICS:
        groups = [g[metric].dropna().to_numpy(dtype=float) for _, g in manager_summary.groupby(GROUP_COL, observed=True)]
        f_stat, df1, df2, p_value = welch_anova(groups)
        rows.append(
            {
                "metric": metric,
                "welch_f": f_stat,
                "df1": df1,
                "df2": df2,
                "p_value": p_value,
                "significance": stars(p_value),
                "eta_squared_macro_region": eta_squared(manager_summary[metric], manager_summary[GROUP_COL]),
                "unit_of_analysis": "manager",
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_tests(manager_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regions = sorted(manager_summary[GROUP_COL].dropna().unique())
    for metric in PAIRWISE_METRICS:
        metric_rows = []
        for left, right in combinations(regions, 2):
            left_values = manager_summary.loc[manager_summary[GROUP_COL].eq(left), metric].dropna().to_numpy(dtype=float)
            right_values = manager_summary.loc[manager_summary[GROUP_COL].eq(right), metric].dropna().to_numpy(dtype=float)
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
        panel.groupby([GROUP_COL, "state_label"], observed=True)
        .size()
        .rename("n_manager_periods")
        .reset_index()
    )
    totals = counts.groupby(GROUP_COL)["n_manager_periods"].transform("sum")
    counts["share_within_region"] = counts["n_manager_periods"] / totals
    return counts.sort_values([GROUP_COL, "state_label"])


def build_transition_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transitions = panel.sort_values(["manager_id", "period_id"]).copy()
    transitions["next_state"] = transitions.groupby("manager_id")["state_label"].shift(-1)
    transitions = transitions[transitions["next_state"].notna()].copy()

    rates = (
        transitions.groupby([GROUP_COL, "state_label", "next_state"], observed=True)
        .size()
        .rename("n_transitions")
        .reset_index()
        .rename(columns={"state_label": "from_state", "next_state": "to_state"})
    )
    complete = pd.MultiIndex.from_product(
        [sorted(panel[GROUP_COL].unique()), STATE_ORDER, STATE_ORDER],
        names=[GROUP_COL, "from_state", "to_state"],
    )
    rates = rates.set_index([GROUP_COL, "from_state", "to_state"]).reindex(complete, fill_value=0).reset_index()
    totals = rates.groupby([GROUP_COL, "from_state"])["n_transitions"].transform("sum")
    rates["share_from_state_region"] = np.where(totals.gt(0), rates["n_transitions"] / totals, np.nan)

    chi_rows = []
    for from_state in STATE_ORDER:
        sub = transitions[transitions["state_label"].eq(from_state)].copy()
        table = pd.crosstab(sub[GROUP_COL], sub["next_state"]).reindex(columns=STATE_ORDER, fill_value=0)
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


def format_estimate(value: float, significance: str) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4f}{significance}"


def pairwise_difference(
    pairwise_tests: pd.DataFrame,
    metric: str,
    left: str,
    right: str,
) -> tuple[float, str]:
    direct = pairwise_tests[
        pairwise_tests["metric"].eq(metric)
        & pairwise_tests["region_1"].eq(left)
        & pairwise_tests["region_2"].eq(right)
    ]
    if not direct.empty:
        row = direct.iloc[0]
        return float(row["difference_region_1_minus_region_2"]), str(row["significance_holm"])

    reverse = pairwise_tests[
        pairwise_tests["metric"].eq(metric)
        & pairwise_tests["region_1"].eq(right)
        & pairwise_tests["region_2"].eq(left)
    ]
    if reverse.empty:
        return np.nan, ""
    row = reverse.iloc[0]
    return -float(row["difference_region_1_minus_region_2"]), str(row["significance_holm"])


def build_appendix_table(
    pairwise_tests: pd.DataFrame,
    welch_tests: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for section, label, metric in APPENDIX_ROWS:
        reference_region = "Europe, Middle East, Africa"
        americas_region = "North, Central, South America"
        asia_diff, asia_sig = pairwise_difference(pairwise_tests, metric, "Asia-Pacific", reference_region)
        americas_diff, americas_sig = pairwise_difference(
            pairwise_tests,
            metric,
            americas_region,
            reference_region,
        )
        omnibus = welch_tests[welch_tests["metric"].eq(metric)].iloc[0]
        rows.append(
            {
                "section": section,
                "variable": label,
                "Asia-Pacific minus EMEA": format_estimate(asia_diff, asia_sig),
                "Americas minus EMEA": format_estimate(americas_diff, americas_sig),
                "Welch F": f"{float(omnibus['welch_f']):.4f}{omnibus['significance']}",
                "p_value": float(omnibus["p_value"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    panel, plant_design, manager_mapping = load_panel()
    manager_summary = build_manager_summary(panel)
    region_summary = build_region_summary(manager_summary)
    welch_tests = build_welch_tests(manager_summary)
    pairwise_tests = build_pairwise_tests(manager_summary)
    state_composition = build_state_composition(panel)
    transition_rates, transition_chi2 = build_transition_tables(panel)
    appendix_table = build_appendix_table(pairwise_tests, welch_tests)

    plant_design.to_csv(SITE_MAPPING_CSV, index=False)
    manager_mapping.to_csv(MANAGER_MAPPING_CSV, index=False)
    manager_summary.to_csv(MANAGER_SUMMARY_CSV, index=False)
    region_summary.to_csv(REGION_SUMMARY_CSV, index=False)
    welch_tests.to_csv(WELCH_TESTS_CSV, index=False)
    pairwise_tests.to_csv(PAIRWISE_CSV, index=False)
    state_composition.to_csv(STATE_COMPOSITION_CSV, index=False)
    transition_rates.to_csv(TRANSITION_RATES_CSV, index=False)
    transition_chi2.to_csv(TRANSITION_CHI2_CSV, index=False)
    appendix_table.to_csv(APPENDIX_TABLE_CSV, index=False)

    for path in [
        SITE_MAPPING_CSV,
        MANAGER_MAPPING_CSV,
        MANAGER_SUMMARY_CSV,
        REGION_SUMMARY_CSV,
        WELCH_TESTS_CSV,
        PAIRWISE_CSV,
        STATE_COMPOSITION_CSV,
        TRANSITION_RATES_CSV,
        TRANSITION_CHI2_CSV,
        APPENDIX_TABLE_CSV,
    ]:
        print(path)


if __name__ == "__main__":
    main()
