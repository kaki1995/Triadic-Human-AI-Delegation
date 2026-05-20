from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_CSV = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"
REGION_MAPPING_CSV = ANALYSIS_DIR / "macro_region_manager_mapping_v3.csv"
OUTPUT_CSV = ANALYSIS_DIR / "macro_region_parameter_comparison_table_v3.csv"
DETAIL_CSV = ANALYSIS_DIR / "macro_region_parameter_comparison_detail_v3.csv"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
COMPARISONS = ["Asia-Pacific", "North, Central, South America"]
REFERENCE_REGION = "Europe, Middle East, Africa"

EMISSION = "escalation_share"
CONTROL_COLS = [
    "decision_latency",
    "demand_volatility",
    "forecast_accuracy",
    "performance_pressure",
    "recent_negative_shock",
    "supply_disruptions",
    "target_difficulty",
    "task_complexity",
]
TRANSITION_COLS = [
    "team_t_minus_1_vs_team_t",
    "team_vs_peer_average",
    "target_attainment",
]
TRANSITION_LABELS = {
    "team_t_minus_1_vs_team_t": "Team (t-1) vs. Team (t)",
    "team_vs_peer_average": "Team vs. Peer",
    "target_attainment": "Target Attainment",
}
CONTROL_LABELS = {
    "decision_latency": "Decision latency",
    "demand_volatility": "Demand volatility",
    "forecast_accuracy": "Forecast accuracy",
    "performance_pressure": "Performance pressure",
    "recent_negative_shock": "Recent negative shock",
    "supply_disruptions": "Supply disruptions",
    "target_difficulty": "Target difficulty",
    "task_complexity": "Task complexity",
}


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return values * 0.0
    return (values - values.mean()) / sd


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


def fmt(value: float, p_value: float | None = None) -> str:
    if not np.isfinite(value):
        return ""
    suffix = "" if p_value is None else stars(float(p_value))
    return f"{value:.4f}{suffix}"


def region_design(data: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=data.index)
    design["intercept"] = 1.0
    for region in COMPARISONS:
        design[f"region_{region}"] = data["macro_region"].eq(region).astype(float)
    return design


def robust_ols(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    model = sm.OLS(y.astype(float), x.astype(float), missing="drop")
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups.loc[x.index]})


def robust_glm_binomial(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    model = sm.GLM(y.astype(float), x.astype(float), family=sm.families.Binomial(), missing="drop")
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups.loc[x.index]}, maxiter=200)


def load_data() -> pd.DataFrame:
    panel = pd.read_excel(
        DATA_PATH,
        sheet_name="panel_manager_period",
        usecols=["manager_id", "period_id", EMISSION, *CONTROL_COLS, *TRANSITION_COLS],
    )
    posterior = pd.read_csv(
        POSTERIOR_CSV,
        usecols=["manager_id", "period_id", "state_label", "state_order"],
    )
    mapping = pd.read_csv(REGION_MAPPING_CSV, usecols=["manager_id", "macro_region"]).drop_duplicates()
    data = (
        panel.merge(posterior, on=["manager_id", "period_id"], how="inner")
        .merge(mapping, on="manager_id", how="inner")
        .sort_values(["manager_id", "period_id"])
        .copy()
    )
    data["state_label"] = pd.Categorical(data["state_label"], categories=STATE_ORDER, ordered=True)
    for col in CONTROL_COLS + TRANSITION_COLS:
        data[f"z_{col}"] = zscore(data[col])
    data["next_state_label"] = data.groupby("manager_id")["state_label"].shift(-1)
    data["next_state_order"] = data.groupby("manager_id")["state_order"].shift(-1)
    return data


def build_emission_design(data: pd.DataFrame) -> pd.DataFrame:
    x = region_design(data)
    for col in CONTROL_COLS:
        z_col = f"z_{col}"
        x[z_col] = data[z_col].astype(float)
        for region in COMPARISONS:
            x[f"region_{region}:{z_col}"] = x[f"region_{region}"] * data[z_col].astype(float)
    return x


def emission_differences(data: pd.DataFrame) -> tuple[dict[str, dict[str, dict[str, tuple[float, float | None]]]], list[dict[str, object]]]:
    out: dict[str, dict[str, dict[str, tuple[float, float | None]]]] = {region: {} for region in COMPARISONS}
    details: list[dict[str, object]] = []
    for state in STATE_ORDER:
        subset = data[data["state_label"].eq(state)].dropna(subset=[EMISSION, *[f"z_{c}" for c in CONTROL_COLS]]).copy()
        x = build_emission_design(subset)
        res = robust_ols(subset[EMISSION], x, subset["manager_id"])

        fitted = res.predict(x)
        subset["resid"] = subset[EMISSION].to_numpy(dtype=float) - fitted
        sigma_by_region = subset.groupby("macro_region", observed=True)["resid"].std(ddof=0)

        for region in COMPARISONS:
            out[region].setdefault("Emission intercept", {})[state] = (
                float(res.params.get(f"region_{region}", np.nan)),
                float(res.pvalues.get(f"region_{region}", np.nan)),
            )
            sigma_diff = float(sigma_by_region.get(region, np.nan) - sigma_by_region.get(REFERENCE_REGION, np.nan))
            out[region].setdefault("Escalation share", {})[state] = (sigma_diff, None)
            details.append(
                {
                    "component": "emission_sigma",
                    "comparison": f"{region} minus {REFERENCE_REGION}",
                    "state": state,
                    "variable": "Escalation share",
                    "estimate": sigma_diff,
                    "p_value": np.nan,
                }
            )

            for col in CONTROL_COLS:
                name = f"region_{region}:z_{col}"
                estimate = float(res.params.get(name, np.nan))
                p_value = float(res.pvalues.get(name, np.nan))
                out[region].setdefault(CONTROL_LABELS[col], {})[state] = (estimate, p_value)
                details.append(
                    {
                        "component": "emission_control",
                        "comparison": f"{region} minus {REFERENCE_REGION}",
                        "state": state,
                        "variable": CONTROL_LABELS[col],
                        "estimate": estimate,
                        "p_value": p_value,
                    }
                )

            details.append(
                {
                    "component": "emission_intercept",
                    "comparison": f"{region} minus {REFERENCE_REGION}",
                    "state": state,
                    "variable": "Emission intercept",
                    "estimate": out[region]["Emission intercept"][state][0],
                    "p_value": out[region]["Emission intercept"][state][1],
                }
            )
    return out, details


def transition_threshold_differences(data: pd.DataFrame) -> tuple[dict[str, dict[str, dict[str, tuple[float, float | None]]]], list[dict[str, object]]]:
    transitions = data.dropna(subset=["next_state_label", *[f"z_{c}" for c in TRANSITION_COLS]]).copy()
    out: dict[str, dict[str, dict[str, tuple[float, float | None]]]] = {region: {} for region in COMPARISONS}
    details: list[dict[str, object]] = []

    for from_state in STATE_ORDER:
        for to_state in STATE_ORDER:
            if to_state == from_state:
                for region in COMPARISONS:
                    out[region].setdefault(f"To {to_state}", {})[from_state] = (np.nan, None)
                continue

            subset = transitions[
                transitions["state_label"].eq(from_state)
                & transitions["next_state_label"].isin([from_state, to_state])
            ].copy()
            if subset.empty or subset["next_state_label"].nunique() < 2:
                continue
            subset["y"] = subset["next_state_label"].eq(to_state).astype(float)
            x = region_design(subset)
            for col in TRANSITION_COLS:
                x[f"z_{col}"] = subset[f"z_{col}"].astype(float)
            try:
                res = robust_glm_binomial(subset["y"], x, subset["manager_id"])
            except Exception:
                continue

            for region in COMPARISONS:
                estimate = float(res.params.get(f"region_{region}", np.nan))
                p_value = float(res.pvalues.get(f"region_{region}", np.nan))
                out[region].setdefault(f"To {to_state}", {})[from_state] = (estimate, p_value)
                details.append(
                    {
                        "component": "transition_threshold",
                        "comparison": f"{region} minus {REFERENCE_REGION}",
                        "state": from_state,
                        "to_state": to_state,
                        "variable": f"To {to_state}",
                        "estimate": estimate,
                        "p_value": p_value,
                    }
                )
    return out, details


def build_transition_beta_design(data: pd.DataFrame) -> pd.DataFrame:
    x = region_design(data)
    for state in STATE_ORDER[1:]:
        x[f"current_{state}"] = data["state_label"].eq(state).astype(float)
    for col in TRANSITION_COLS:
        z_col = f"z_{col}"
        x[z_col] = data[z_col].astype(float)
        for region in COMPARISONS:
            x[f"region_{region}:{z_col}"] = x[f"region_{region}"] * data[z_col].astype(float)
    return x


def transition_beta_differences(data: pd.DataFrame) -> tuple[dict[str, dict[str, dict[str, tuple[float, float | None]]]], list[dict[str, object]]]:
    transitions = data.dropna(subset=["next_state_label", *[f"z_{c}" for c in TRANSITION_COLS]]).copy()
    out: dict[str, dict[str, dict[str, tuple[float, float | None]]]] = {region: {} for region in COMPARISONS}
    details: list[dict[str, object]] = []

    for target_state in STATE_ORDER:
        subset = transitions.copy()
        subset["y"] = subset["next_state_label"].eq(target_state).astype(float)
        x = build_transition_beta_design(subset)
        res = robust_glm_binomial(subset["y"], x, subset["manager_id"])

        for col in TRANSITION_COLS:
            label = TRANSITION_LABELS[col]
            for region in COMPARISONS:
                name = f"region_{region}:z_{col}"
                estimate = float(res.params.get(name, np.nan))
                p_value = float(res.pvalues.get(name, np.nan))
                out[region].setdefault(label, {})[target_state] = (estimate, p_value)
                details.append(
                    {
                        "component": "transition_factor",
                        "comparison": f"{region} minus {REFERENCE_REGION}",
                        "state": target_state,
                        "variable": label,
                        "estimate": estimate,
                        "p_value": p_value,
                    }
                )
    return out, details


def transition_heterogeneity_difference(data: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], list[dict[str, object]]]:
    transitions = data.dropna(subset=["next_state_order", "state_label", *[f"z_{c}" for c in TRANSITION_COLS]]).copy()
    transitions["next_state_order_z"] = zscore(transitions["next_state_order"])
    x = pd.DataFrame(index=transitions.index)
    x["intercept"] = 1.0
    for state in STATE_ORDER[1:]:
        x[f"current_{state}"] = transitions["state_label"].eq(state).astype(float)
    for col in TRANSITION_COLS:
        x[f"z_{col}"] = transitions[f"z_{col}"].astype(float)
    res = robust_ols(transitions["next_state_order_z"], x, transitions["manager_id"])
    transitions["resid"] = transitions["next_state_order_z"].astype(float) - res.predict(x)
    manager_resid = (
        transitions.groupby(["manager_id", "macro_region"], observed=True)["resid"]
        .mean()
        .reset_index()
    )
    delta = manager_resid.groupby("macro_region", observed=True)["resid"].std(ddof=0)

    out: dict[str, tuple[float, float]] = {}
    details: list[dict[str, object]] = []
    europe = manager_resid[manager_resid["macro_region"].eq(REFERENCE_REGION)]["resid"].to_numpy(dtype=float)
    for region in COMPARISONS:
        values = manager_resid[manager_resid["macro_region"].eq(region)]["resid"].to_numpy(dtype=float)
        diff = float(delta.get(region, np.nan) - delta.get(REFERENCE_REGION, np.nan))
        try:
            _, p_value = stats.levene(values, europe, center="median")
            p_value = float(p_value)
        except Exception:
            p_value = np.nan
        out[region] = (diff, p_value)
        details.append(
            {
                "component": "manager_transition_heterogeneity_proxy",
                "comparison": f"{region} minus {REFERENCE_REGION}",
                "variable": "Manager-specific heterogeneity",
                "estimate": diff,
                "p_value": p_value,
            }
        )
    return out, details


def merge_nested(
    target: dict[str, dict[str, dict[str, tuple[float, float | None]]]],
    source: dict[str, dict[str, dict[str, tuple[float, float | None]]]],
) -> None:
    for region, rows in source.items():
        for variable, states in rows.items():
            target[region].setdefault(variable, {}).update(states)


def add_row(
    rows: list[dict[str, object]],
    comparison: str,
    section: str,
    variable: str,
    values: dict[str, tuple[float, float | None]],
) -> None:
    row = {"comparison": comparison, "section": section, "variable": variable}
    for state in STATE_ORDER:
        value, p_value = values.get(state, (np.nan, None))
        row[state] = fmt(value, p_value)
    rows.append(row)


def build_output_table(
    estimates: dict[str, dict[str, dict[str, tuple[float, float | None]]]],
    delta_estimates: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in COMPARISONS:
        comparison = f"{region} minus {REFERENCE_REGION}"
        add_row(rows, comparison, "Emission factors (rho_m, eta)", "Emission intercept", estimates[region].get("Emission intercept", {}))
        add_row(rows, comparison, "Emission standard deviation (sigma)", "Escalation share", estimates[region].get("Escalation share", {}))
        for col in CONTROL_COLS:
            add_row(rows, comparison, "Emission control factors (W)", CONTROL_LABELS[col], estimates[region].get(CONTROL_LABELS[col], {}))
        for to_state in STATE_ORDER:
            add_row(rows, comparison, "Transition threshold (mu)", f"To {to_state}", estimates[region].get(f"To {to_state}", {}))
        delta_value, delta_p = delta_estimates[region]
        add_row(
            rows,
            comparison,
            "Manager-specific heterogeneity",
            "delta",
            {"Aversion": (delta_value, delta_p), "Neutral": (np.nan, None), "Appreciation": (np.nan, None)},
        )
        for col in TRANSITION_COLS:
            label = TRANSITION_LABELS[col]
            add_row(rows, comparison, "Transition factors (beta)", label, estimates[region].get(label, {}))
    return pd.DataFrame(rows)


def main() -> None:
    data = load_data()
    estimates: dict[str, dict[str, dict[str, tuple[float, float | None]]]] = {region: {} for region in COMPARISONS}
    details: list[dict[str, object]] = []

    emission, emission_detail = emission_differences(data)
    thresholds, threshold_detail = transition_threshold_differences(data)
    betas, beta_detail = transition_beta_differences(data)
    delta, delta_detail = transition_heterogeneity_difference(data)

    merge_nested(estimates, emission)
    merge_nested(estimates, thresholds)
    merge_nested(estimates, betas)
    details.extend(emission_detail)
    details.extend(threshold_detail)
    details.extend(beta_detail)
    details.extend(delta_detail)

    output = build_output_table(estimates, delta)
    output.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(details).to_csv(DETAIL_CSV, index=False)
    print(OUTPUT_CSV)
    print(DETAIL_CSV)


if __name__ == "__main__":
    main()
