from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from numpy.linalg import LinAlgError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_PATH = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"
SUMMARY_PATH = ANALYSIS_DIR / "manager_heterogeneity_tests_v3.csv"
DETAIL_PATH = ANALYSIS_DIR / "manager_heterogeneity_tests_detail_v3.csv"

TRANSITION_COLS = [
    "team_t_minus_1_vs_team_t",
    "team_vs_peer_average",
    "target_attainment",
]

EMISSION_COLS = [
    "ai_authority_share",
    "escalation_share",
]

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

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return values * 0.0
    return (values - values.mean()) / sd


def stars_from_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def write_csv_with_fallback(df: pd.DataFrame, path: Path) -> Optional[Path]:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError as exc:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        try:
            df.to_csv(fallback, index=False)
            return fallback
        except PermissionError as fallback_exc:
            print(f"WARNING: Could not write {path}: {exc}")
            print(f"WARNING: Could not write fallback {fallback}: {fallback_exc}")
            return None


def fit_random_intercept(formula: str, data: pd.DataFrame, group_col: str = "manager_id") -> dict[str, float | str | bool]:
    ols = smf.ols(formula, data=data).fit()
    try:
        mixed = smf.mixedlm(formula, data=data, groups=data[group_col]).fit(
            reml=False,
            method="lbfgs",
            maxiter=500,
            disp=False,
        )
    except (LinAlgError, ValueError) as exc:
        fallback = fit_fixed_effect_heterogeneity(formula, data, group_col=group_col)
        fallback["fallback_reason"] = str(exc)
        return fallback

    var_random = float(mixed.cov_re.iloc[0, 0]) if mixed.cov_re.size else 0.0
    var_random = max(var_random, 0.0)
    var_resid = max(float(mixed.scale), 0.0)
    lr_stat = max(0.0, 2.0 * (float(mixed.llf) - float(ols.llf)))

    # The null random-effect variance is on the boundary. A conservative
    # one-variance-component test uses the 50:50 chi-square mixture.
    p_value = min(1.0, 0.5 * float(chi2.sf(lr_stat, df=1)))
    denom = var_random + var_resid

    return {
        "estimate": float(np.sqrt(var_random)),
        "random_intercept_variance": var_random,
        "residual_variance": var_resid,
        "icc": float(var_random / denom) if denom > 0 else np.nan,
        "lr_stat": lr_stat,
        "p_value": max(p_value, 1e-300),
        "significance": stars_from_p(p_value),
        "n_obs": int(mixed.nobs),
        "n_managers": int(data[group_col].nunique()),
        "converged": bool(getattr(mixed, "converged", False)),
        "model_ll": float(mixed.llf),
        "null_ll": float(ols.llf),
        "test_type": "random_intercept_lr_mixture",
        "fallback_reason": "",
    }


def fit_fixed_effect_heterogeneity(
    formula: str,
    data: pd.DataFrame,
    group_col: str = "manager_id",
) -> dict[str, float | str | bool]:
    base = smf.ols(formula, data=data).fit()
    full = smf.ols(f"{formula} + C({group_col})", data=data).fit()
    f_stat, p_value, df_diff = full.compare_f_test(base)
    p_value = float(max(min(p_value, 1.0), 1e-300))

    managers = data[group_col].astype(str)
    reference = sorted(managers.unique())[0]
    effects = pd.Series(0.0, index=pd.Index(sorted(managers.unique()), name=group_col))
    prefix = f"C({group_col})[T."
    for name, value in full.params.items():
        if name.startswith(prefix) and name.endswith("]"):
            manager = name[len(prefix):-1]
            effects.loc[manager] = float(value)
    effects = effects - float(np.average(effects.reindex(managers).to_numpy()))
    weighted_effects = effects.reindex(managers).to_numpy()
    var_manager = float(np.var(weighted_effects, ddof=0))
    var_resid = float(full.mse_resid)
    denom = var_manager + var_resid

    return {
        "estimate": float(np.sqrt(max(var_manager, 0.0))),
        "random_intercept_variance": var_manager,
        "residual_variance": var_resid,
        "icc": float(var_manager / denom) if denom > 0 else np.nan,
        "lr_stat": float(f_stat),
        "p_value": p_value,
        "significance": stars_from_p(p_value),
        "n_obs": int(full.nobs),
        "n_managers": int(data[group_col].nunique()),
        "converged": True,
        "model_ll": float(full.llf),
        "null_ll": float(base.llf),
        "test_type": f"manager_fixed_effect_f_test_df_{int(df_diff)}",
        "fallback_reason": "",
    }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading panel data from {DATA_PATH}")
    panel = pd.read_excel(DATA_PATH, sheet_name="panel_manager_period")
    posteriors = pd.read_csv(POSTERIOR_PATH)

    keep_post = ["manager_id", "period_id", "state_label", "state_order"]
    data = panel.merge(posteriors[keep_post], on=["manager_id", "period_id"], how="inner")
    data = data.sort_values(["manager_id", "period_id"]).copy()
    data["state_label"] = pd.Categorical(data["state_label"], categories=STATE_ORDER, ordered=True)

    for col in TRANSITION_COLS + CONTROL_COLS:
        data[f"z_{col}"] = zscore(data[col])

    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    # Transition heterogeneity: persistent manager differences in state-transition
    # propensity after controlling for the previous posterior state and transition covariates.
    transition = data.copy()
    transition["next_state_order"] = transition.groupby("manager_id")["state_order"].shift(-1)
    transition["next_state_order_z"] = zscore(transition["next_state_order"])
    transition = transition.dropna(subset=["next_state_order_z", "state_label", *TRANSITION_COLS]).copy()
    transition_formula = (
        "next_state_order_z ~ C(state_label) + "
        + " + ".join(f"z_{col}" for col in TRANSITION_COLS)
    )
    print("Fitting transition random-intercept model for delta...")
    delta = fit_random_intercept(transition_formula, transition)
    delta_row = {
        "factor": "delta",
        "label": "Manager-Specific Heterogeneity in Transitions",
        "estimate": delta["estimate"],
        "p_value": delta["p_value"],
        "significance": delta["significance"],
        "icc": delta["icc"],
        "n_obs": delta["n_obs"],
        "n_managers": delta["n_managers"],
        "method": f"post-HMM {delta['test_type']} on standardized next posterior state order",
    }
    rows.append(delta_row)
    detail_rows.append({**delta_row, **{f"fit_{k}": v for k, v in delta.items()}})

    # Emission heterogeneity: persistent manager differences in each standardized
    # emission after controlling for posterior state and emission-side controls.
    emission_estimates: list[float] = []
    emission_p_values: list[float] = []
    emission_iccs: list[float] = []
    emission_n_obs: list[int] = []
    emission_labels: list[str] = []
    control_terms = " + ".join(f"z_{col}" for col in CONTROL_COLS)

    for emission in EMISSION_COLS:
        emission_data = data.dropna(subset=[emission, "state_label", *CONTROL_COLS]).copy()
        emission_data["y_emission_z"] = zscore(emission_data[emission])
        emission_formula = f"y_emission_z ~ C(state_label) + {control_terms}"
        print(f"Fitting emission random-intercept model for xi: {emission}...")
        xi_component = fit_random_intercept(emission_formula, emission_data)
        emission_estimates.append(float(xi_component["estimate"]))
        emission_p_values.append(float(xi_component["p_value"]))
        emission_iccs.append(float(xi_component["icc"]))
        emission_n_obs.append(int(xi_component["n_obs"]))
        emission_labels.append(emission)
        detail_rows.append(
            {
                "factor": "xi_component",
                "label": f"Manager-Specific Heterogeneity in Emissions: {emission}",
                "component": emission,
                "estimate": xi_component["estimate"],
                "p_value": xi_component["p_value"],
                "significance": xi_component["significance"],
                "icc": xi_component["icc"],
                "n_obs": xi_component["n_obs"],
                "n_managers": xi_component["n_managers"],
                "method": "post-HMM random-intercept test on standardized emission",
                **{f"fit_{k}": v for k, v in xi_component.items()},
            }
        )

    xi_estimate = float(np.sqrt(np.mean(np.square(emission_estimates))))
    xi_icc = float(np.mean(emission_iccs))
    fisher_stat = -2.0 * float(np.sum(np.log(np.maximum(emission_p_values, 1e-300))))
    xi_p = max(float(chi2.sf(fisher_stat, df=2 * len(emission_p_values))), 1e-300)
    xi_row = {
        "factor": "xi",
        "label": "Manager-Specific Heterogeneity in Emissions",
        "estimate": xi_estimate,
        "p_value": xi_p,
        "significance": stars_from_p(xi_p),
        "icc": xi_icc,
        "n_obs": int(np.sum(emission_n_obs)),
        "n_managers": int(data["manager_id"].nunique()),
        "method": "RMS random-intercept SD across standardized emissions; p-value combined by Fisher test",
        "components": ", ".join(emission_labels),
    }
    rows.append(xi_row)

    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    print(summary.to_string(index=False))

    summary_path = write_csv_with_fallback(summary, SUMMARY_PATH)
    detail_path = write_csv_with_fallback(detail, DETAIL_PATH)

    if summary_path is not None:
        print(f"Saved summary to {summary_path}")
    if detail_path is not None:
        print(f"Saved details to {detail_path}")


if __name__ == "__main__":
    main()
