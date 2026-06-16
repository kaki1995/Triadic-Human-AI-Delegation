from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_CSV = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"

MANAGER_MAPPING_CSV = ANALYSIS_DIR / "age_locality_manager_mapping_v3.csv"
MANAGER_SUMMARY_CSV = ANALYSIS_DIR / "age_locality_manager_level_summary_v3.csv"
GROUP_SUMMARY_CSV = ANALYSIS_DIR / "age_locality_group_summary_v3.csv"
WELCH_TESTS_CSV = ANALYSIS_DIR / "age_locality_welch_tests_v3.csv"
STATE_COMPOSITION_CSV = ANALYSIS_DIR / "age_locality_state_composition_v3.csv"
FIRST_ORDER_TRANSITION_CSV = ANALYSIS_DIR / "age_locality_first_order_transition_rates_v3.csv"
SECOND_ORDER_TRANSITION_CSV = ANALYSIS_DIR / "age_locality_second_order_transition_rates_v3.csv"
NOTE_PATH = ANALYSIS_DIR / "age_locality_context_analysis_note_v3.md"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
STATE_SCORE = {"Aversion": 0, "Neutral": 1, "Appreciation": 2}

AGE_COLUMNS = [
    "manager_age",
    "age_years",
    "age",
]

LOCALITY_COLUMNS = [
    "manager_locality",
    "locality",
    "local_status",
    "local_nonlocal",
    "local_non_local",
    "is_local",
]

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
    "posterior_aversion_mean",
    "posterior_neutral_mean",
    "posterior_appreciation_mean",
    "persistence_rate",
    "upward_transition_rate",
    "downward_transition_rate",
    "second_order_repeat_rate",
    "second_order_reversal_rate",
]


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
    ss_between = sum(len(g) * (g["value"].mean() - grand_mean) ** 2 for _, g in df.groupby("group", observed=True))
    ss_total = float(((df["value"] - grand_mean) ** 2).sum())
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.array(list(p_values), dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid_mask = np.isfinite(p)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return adjusted.tolist()

    order = valid_idx[np.argsort(p[valid_idx])]
    running_max = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted.tolist()


def write_csv_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def write_text_with_fallback(text: str, path: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_map = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_locality(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, (bool, np.bool_)):
        return "Local" if bool(value) else "Non-local"
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return "Local" if int(value) == 1 else "Non-local"

    text = str(value).strip().lower().replace("_", "-")
    if text in {"local", "home", "host", "domestic", "resident", "1", "true", "yes"}:
        return "Local"
    if text in {"non-local", "nonlocal", "foreign", "expatriate", "international", "0", "false", "no"}:
        return "Non-local"
    return str(value).strip()


def add_age_and_locality_groups(managers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    managers = managers.copy()
    metadata: list[dict[str, str]] = []

    age_col = first_existing(managers.columns, AGE_COLUMNS)
    if age_col is not None:
        managers["age_basis_value"] = pd.to_numeric(managers[age_col], errors="coerce")
        managers["age_group"] = pd.cut(
            managers["age_basis_value"],
            bins=[0, 35, 45, 55, np.inf],
            labels=["Under 35", "35-44", "45-54", "55+"],
            right=False,
        ).astype("object")
        managers["age_group"] = managers["age_group"].fillna("Unknown age")
        age_source = f"observed manager age column: {age_col}"
        age_rule = "Under 35, 35-44, 45-54, 55+"
    elif "seniority_years" in managers.columns:
        managers["age_basis_value"] = pd.to_numeric(managers["seniority_years"], errors="coerce")
        managers["age_group"] = pd.cut(
            managers["age_basis_value"],
            bins=[-np.inf, 4, 8, np.inf],
            labels=["Early-career proxy", "Mid-career proxy", "Senior-career proxy"],
            right=False,
        ).astype("object")
        managers["age_group"] = managers["age_group"].fillna("Unknown seniority")
        age_source = "seniority_years proxy because no manager age column exists in the workbook"
        age_rule = "0-3.99, 4-7.99, and 8+ seniority years"
    else:
        managers["age_basis_value"] = np.nan
        managers["age_group"] = "Unknown age"
        age_source = "unavailable"
        age_rule = "No age or seniority field found"

    managers["age_group_source"] = age_source
    metadata.append(
        {
            "grouping_variable": "age_group",
            "source": age_source,
            "rule": age_rule,
        }
    )

    locality_col = first_existing(managers.columns, LOCALITY_COLUMNS)
    if locality_col is not None:
        managers["locality_group"] = managers[locality_col].map(normalize_locality)
        locality_source = f"observed manager locality column: {locality_col}"
        locality_rule = "Normalized observed locality labels to Local / Non-local where possible"
    elif "region" in managers.columns:
        region = managers["region"].astype(str)
        managers["locality_group"] = np.where(region.str.startswith("DE-"), "Local", "Non-local")
        locality_source = "region proxy because no manager local/non-local column exists"
        locality_rule = "Local = region starts with DE-; Non-local = all other regions"
    else:
        managers["locality_group"] = "Unknown locality"
        locality_source = "unavailable"
        locality_rule = "No locality or region field found"

    managers["locality_group_source"] = locality_source
    metadata.append(
        {
            "grouping_variable": "locality_group",
            "source": locality_source,
            "rule": locality_rule,
        }
    )

    managers["age_x_locality"] = managers["age_group"].astype(str) + " | " + managers["locality_group"].astype(str)
    keep_cols = [
        "manager_id",
        "age_basis_value",
        "age_group",
        "age_group_source",
        "locality_group",
        "locality_group_source",
        "age_x_locality",
    ]
    passthrough = [
        col
        for col in [
            "seniority_years",
            "region",
            "site_id",
            "baseline_ai_attitude",
            "risk_aversion_index",
            "governance_mode",
            "high_pressure",
        ]
        if col in managers.columns
    ]
    return managers[keep_cols + passthrough].copy(), pd.DataFrame(metadata)


def infer_gamma_label_map(posteriors: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for state_idx in sorted(posteriors["most_likely_state"].dropna().astype(int).unique()):
        subset = posteriors[posteriors["most_likely_state"].eq(state_idx)]
        if subset.empty:
            continue
        mapping[f"gamma_state_{state_idx}"] = str(subset["state_label"].mode().iloc[0])
    return mapping


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    managers = pd.read_excel(DATA_PATH, sheet_name="manager_master")
    manager_mapping, metadata = add_age_and_locality_groups(managers)

    panel = pd.read_excel(DATA_PATH, sheet_name="panel_manager_period", usecols=PANEL_COLUMNS)
    posteriors = pd.read_csv(POSTERIOR_CSV)
    gamma_map = infer_gamma_label_map(posteriors)

    posterior_cols = ["manager_id", "period_id", "state_label", "state_order", *gamma_map.keys()]
    posterior = posteriors[posterior_cols].rename(columns={col: f"posterior_{label.lower()}" for col, label in gamma_map.items()})

    merged = (
        panel.merge(posterior, on=["manager_id", "period_id"], how="inner")
        .merge(manager_mapping, on="manager_id", how="left")
        .sort_values(["manager_id", "period_id"])
        .copy()
    )
    if "region_x" in merged.columns:
        merged["region"] = merged["region_x"]
        merged = merged.drop(columns=[col for col in ["region_x", "region_y"] if col in merged.columns])
    elif "region_y" in merged.columns:
        merged["region"] = merged["region_y"]
        merged = merged.drop(columns=["region_y"])
    if merged.empty:
        raise ValueError("No rows after merging panel data with posterior state assignments.")

    for state in STATE_ORDER:
        col = f"posterior_{state.lower()}"
        if col not in merged.columns:
            merged[col] = np.nan

    return merged, metadata


def build_manager_summary(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for state in STATE_ORDER:
        panel[f"is_{state.lower()}"] = panel["state_label"].eq(state).astype(float)

    manager_summary = (
        panel.groupby(
            [
                "manager_id",
                "age_group",
                "locality_group",
                "age_x_locality",
                "age_group_source",
                "locality_group_source",
            ],
            observed=True,
        )
        .agg(
            n_periods=("period_id", "size"),
            ai_authority_share_mean=("ai_authority_share", "mean"),
            escalation_share_mean=("escalation_share", "mean"),
            composite_kpi_score_mean=("composite_kpi_score", "mean"),
            target_attainment_rate=("target_attainment", "mean"),
            share_aversion=("is_aversion", "mean"),
            share_neutral=("is_neutral", "mean"),
            share_appreciation=("is_appreciation", "mean"),
            posterior_aversion_mean=("posterior_aversion", "mean"),
            posterior_neutral_mean=("posterior_neutral", "mean"),
            posterior_appreciation_mean=("posterior_appreciation", "mean"),
        )
        .reset_index()
    )

    dominant = (
        panel.groupby("manager_id", observed=True)["state_label"]
        .agg(lambda x: str(x.value_counts().idxmax()))
        .rename("dominant_state")
        .reset_index()
    )

    transitions = panel.sort_values(["manager_id", "period_id"]).copy()
    transitions["previous_state"] = transitions.groupby("manager_id")["state_label"].shift(1)
    transitions["next_state"] = transitions.groupby("manager_id")["state_label"].shift(-1)
    transitions["state_score"] = transitions["state_label"].map(STATE_SCORE)
    transitions["previous_state_score"] = transitions["previous_state"].map(STATE_SCORE)
    transitions["next_state_score"] = transitions["next_state"].map(STATE_SCORE)

    first_order = transitions[transitions["next_state"].notna()].copy()
    first_order["persistence"] = first_order["state_label"].eq(first_order["next_state"]).astype(float)
    first_order["upward_transition"] = (first_order["next_state_score"] > first_order["state_score"]).astype(float)
    first_order["downward_transition"] = (first_order["next_state_score"] < first_order["state_score"]).astype(float)

    transition_summary = (
        first_order.groupby("manager_id", observed=True)
        .agg(
            n_transitions=("period_id", "size"),
            persistence_rate=("persistence", "mean"),
            upward_transition_rate=("upward_transition", "mean"),
            downward_transition_rate=("downward_transition", "mean"),
        )
        .reset_index()
    )

    second_order = transitions[transitions["previous_state"].notna() & transitions["next_state"].notna()].copy()
    second_order["second_order_repeat"] = second_order["previous_state"].eq(second_order["next_state"]).astype(float)
    second_order["second_order_reversal"] = (
        (second_order["previous_state_score"] < second_order["state_score"])
        & (second_order["next_state_score"] < second_order["state_score"])
    ) | (
        (second_order["previous_state_score"] > second_order["state_score"])
        & (second_order["next_state_score"] > second_order["state_score"])
    )
    second_order_summary = (
        second_order.groupby("manager_id", observed=True)
        .agg(
            n_second_order_windows=("period_id", "size"),
            second_order_repeat_rate=("second_order_repeat", "mean"),
            second_order_reversal_rate=("second_order_reversal", "mean"),
        )
        .reset_index()
    )

    return (
        manager_summary.merge(dominant, on="manager_id", how="left")
        .merge(transition_summary, on="manager_id", how="left")
        .merge(second_order_summary, on="manager_id", how="left")
    )


def group_specs() -> list[tuple[str, list[str]]]:
    return [
        ("age_group", ["age_group"]),
        ("locality_group", ["locality_group"]),
        ("age_x_locality", ["age_x_locality"]),
    ]


def build_group_summary(manager_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, keys in group_specs():
        for key_values, group in manager_summary.groupby(keys, observed=True):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            row: dict[str, float | int | str] = {
                "group_dimension": dimension,
                "group_label": " | ".join(str(value) for value in key_values),
                "n_managers": int(group["manager_id"].nunique()),
            }
            for key, value in zip(keys, key_values):
                row[key] = value
            for metric in TEST_METRICS:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def build_welch_tests(manager_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, keys in group_specs():
        group_col = keys[-1]
        for metric in TEST_METRICS:
            groups = [
                group[metric].dropna().to_numpy(dtype=float)
                for _, group in manager_summary.groupby(group_col, observed=True)
            ]
            f_stat, df1, df2, p_value = welch_anova(groups)
            rows.append(
                {
                    "group_dimension": dimension,
                    "group_column": group_col,
                    "metric": metric,
                    "welch_f": f_stat,
                    "df1": df1,
                    "df2": df2,
                    "p_value": p_value,
                    "eta_squared": eta_squared(manager_summary[metric], manager_summary[group_col]),
                    "unit_of_analysis": "manager",
                }
            )

    tests = pd.DataFrame(rows)
    tests["p_value_holm_within_dimension"] = np.nan
    for dimension, idx in tests.groupby("group_dimension", observed=True).groups.items():
        tests.loc[idx, "p_value_holm_within_dimension"] = holm_adjust(tests.loc[idx, "p_value"])
    tests["significance_holm"] = tests["p_value_holm_within_dimension"].map(stars)
    return tests


def build_state_composition(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, keys in group_specs():
        counts = (
            panel.groupby([*keys, "state_label"], observed=True)
            .size()
            .rename("n_manager_periods")
            .reset_index()
        )
        totals = counts.groupby(keys, observed=True)["n_manager_periods"].transform("sum")
        counts["share_within_group"] = counts["n_manager_periods"] / totals
        counts["group_dimension"] = dimension
        counts["group_label"] = counts[keys].astype(str).agg(" | ".join, axis=1)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True).sort_values(["group_dimension", "group_label", "state_label"])


def build_first_order_transition_rates(panel: pd.DataFrame) -> pd.DataFrame:
    transitions = panel.sort_values(["manager_id", "period_id"]).copy()
    transitions["next_state"] = transitions.groupby("manager_id")["state_label"].shift(-1)
    transitions = transitions[transitions["next_state"].notna()].copy()

    rows = []
    for dimension, keys in group_specs():
        counts = (
            transitions.groupby([*keys, "state_label", "next_state"], observed=True)
            .size()
            .rename("n_transitions")
            .reset_index()
            .rename(columns={"state_label": "from_state", "next_state": "to_state"})
        )
        complete_keys = [sorted(transitions[key].dropna().unique()) for key in keys]
        complete = pd.MultiIndex.from_product(
            [*complete_keys, STATE_ORDER, STATE_ORDER],
            names=[*keys, "from_state", "to_state"],
        )
        counts = counts.set_index([*keys, "from_state", "to_state"]).reindex(complete, fill_value=0).reset_index()
        totals = counts.groupby([*keys, "from_state"], observed=True)["n_transitions"].transform("sum")
        counts["share_from_state_group"] = np.where(totals.gt(0), counts["n_transitions"] / totals, np.nan)
        counts["group_dimension"] = dimension
        counts["group_label"] = counts[keys].astype(str).agg(" | ".join, axis=1)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["group_dimension", "group_label", "from_state", "to_state"]
    )


def build_second_order_transition_rates(panel: pd.DataFrame) -> pd.DataFrame:
    transitions = panel.sort_values(["manager_id", "period_id"]).copy()
    transitions["previous_state"] = transitions.groupby("manager_id")["state_label"].shift(1)
    transitions["next_state"] = transitions.groupby("manager_id")["state_label"].shift(-1)
    transitions = transitions[transitions["previous_state"].notna() & transitions["next_state"].notna()].copy()

    rows = []
    for dimension, keys in group_specs():
        counts = (
            transitions.groupby([*keys, "previous_state", "state_label", "next_state"], observed=True)
            .size()
            .rename("n_windows")
            .reset_index()
            .rename(columns={"state_label": "current_state"})
        )
        complete_keys = [sorted(transitions[key].dropna().unique()) for key in keys]
        complete = pd.MultiIndex.from_product(
            [*complete_keys, STATE_ORDER, STATE_ORDER, STATE_ORDER],
            names=[*keys, "previous_state", "current_state", "next_state"],
        )
        counts = (
            counts.set_index([*keys, "previous_state", "current_state", "next_state"])
            .reindex(complete, fill_value=0)
            .reset_index()
        )
        totals = counts.groupby([*keys, "previous_state", "current_state"], observed=True)["n_windows"].transform("sum")
        counts["share_from_two_state_history_group"] = np.where(totals.gt(0), counts["n_windows"] / totals, np.nan)
        counts["group_dimension"] = dimension
        counts["group_label"] = counts[keys].astype(str).agg(" | ".join, axis=1)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["group_dimension", "group_label", "previous_state", "current_state", "next_state"]
    )


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def best_group(group_summary: pd.DataFrame, dimension: str, metric: str, highest: bool = True) -> pd.Series:
    subset = group_summary[group_summary["group_dimension"].eq(dimension)].copy()
    metric_col = f"{metric}_mean"
    subset = subset[np.isfinite(subset[metric_col])]
    if subset.empty:
        return pd.Series(dtype=object)
    idx = subset[metric_col].idxmax() if highest else subset[metric_col].idxmin()
    return subset.loc[idx]


def make_note(metadata: pd.DataFrame, group_summary: pd.DataFrame, tests: pd.DataFrame) -> str:
    age_source = metadata.loc[metadata["grouping_variable"].eq("age_group"), "source"].iloc[0]
    age_rule = metadata.loc[metadata["grouping_variable"].eq("age_group"), "rule"].iloc[0]
    locality_source = metadata.loc[metadata["grouping_variable"].eq("locality_group"), "source"].iloc[0]
    locality_rule = metadata.loc[metadata["grouping_variable"].eq("locality_group"), "rule"].iloc[0]

    local_app = best_group(group_summary, "locality_group", "share_appreciation", highest=True)
    age_aversion = best_group(group_summary, "age_group", "share_aversion", highest=True)
    age_authority = best_group(group_summary, "age_group", "ai_authority_share_mean", highest=True)

    significant = tests[
        tests["p_value_holm_within_dimension"].notna()
        & tests["p_value_holm_within_dimension"].lt(0.05)
    ].sort_values(["group_dimension", "p_value_holm_within_dimension"])

    lines = [
        "# Age and Locality Context Analysis",
        "",
        "## Link to high-order HMM framing",
        "",
        "The analysis treats each manager's decoded HMM path as a dynamic decision sequence. "
        "In the spirit of high-order HMM decision analysis, it reports both first-order transitions "
        "and second-order transition windows, where the next state is conditioned on the two-state "
        "history rather than only the current state.",
        "",
        "## Group construction",
        "",
        f"- Age grouping: {age_source}. Rule: {age_rule}.",
        f"- Locality grouping: {locality_source}. Rule: {locality_rule}.",
        "",
    ]

    if age_source.startswith("seniority_years proxy"):
        lines.extend(
            [
                "Important limitation: the active workbook does not contain true manager age. "
                "The age-group output is therefore a career-stage proxy, not demographic age.",
                "",
            ]
        )

    lines.extend(["## Descriptive highlights", ""])
    if not local_app.empty:
        lines.append(
            f"- Highest locality-group Appreciation share: {local_app['group_label']} "
            f"({pct(float(local_app['share_appreciation_mean']))})."
        )
    if not age_aversion.empty:
        lines.append(
            f"- Highest age/career group Aversion share: {age_aversion['group_label']} "
            f"({pct(float(age_aversion['share_aversion_mean']))})."
        )
    if not age_authority.empty:
        lines.append(
            f"- Highest age/career group AI authority share: {age_authority['group_label']} "
            f"({pct(float(age_authority['ai_authority_share_mean_mean']))})."
        )

    lines.extend(["", "## Statistical tests", ""])
    if significant.empty:
        lines.append("No Holm-adjusted subgroup differences reached p < .05 across the tested manager-level metrics.")
    else:
        for _, row in significant.head(12).iterrows():
            lines.append(
                f"- {row['group_dimension']} / {row['metric']}: "
                f"Welch F={row['welch_f']:.3f}, adjusted p={row['p_value_holm_within_dimension']:.4g}."
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{MANAGER_MAPPING_CSV.name}`: manager-level age/locality mapping and source metadata.",
            f"- `{MANAGER_SUMMARY_CSV.name}`: manager-level HMM path and outcome summaries.",
            f"- `{GROUP_SUMMARY_CSV.name}`: summaries by age group, locality, and age x locality.",
            f"- `{WELCH_TESTS_CSV.name}`: Welch tests on manager-level metrics.",
            f"- `{STATE_COMPOSITION_CSV.name}`: decoded state composition by subgroup.",
            f"- `{FIRST_ORDER_TRANSITION_CSV.name}`: state-to-state transition rates by subgroup.",
            f"- `{SECOND_ORDER_TRANSITION_CSV.name}`: high-order two-state-history transition rates by subgroup.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    panel, metadata = load_panel()
    manager_mapping = panel[
        [
            "manager_id",
            "age_basis_value",
            "age_group",
            "age_group_source",
            "locality_group",
            "locality_group_source",
            "age_x_locality",
            "seniority_years",
            "region",
            "site_id",
            "baseline_ai_attitude",
            "risk_aversion_index",
            "governance_mode",
            "high_pressure",
        ]
    ].drop_duplicates()

    manager_summary = build_manager_summary(panel)
    group_summary = build_group_summary(manager_summary)
    welch_tests = build_welch_tests(manager_summary)
    state_composition = build_state_composition(panel)
    first_order = build_first_order_transition_rates(panel)
    second_order = build_second_order_transition_rates(panel)
    note = make_note(metadata, group_summary, welch_tests)

    written_paths = [
        write_csv_with_fallback(manager_mapping, MANAGER_MAPPING_CSV),
        write_csv_with_fallback(manager_summary, MANAGER_SUMMARY_CSV),
        write_csv_with_fallback(group_summary, GROUP_SUMMARY_CSV),
        write_csv_with_fallback(welch_tests, WELCH_TESTS_CSV),
        write_csv_with_fallback(state_composition, STATE_COMPOSITION_CSV),
        write_csv_with_fallback(first_order, FIRST_ORDER_TRANSITION_CSV),
        write_csv_with_fallback(second_order, SECOND_ORDER_TRANSITION_CSV),
        write_text_with_fallback(note, NOTE_PATH),
    ]

    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
