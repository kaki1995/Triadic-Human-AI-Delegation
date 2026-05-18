from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.miscmodels.ordinal_model import OrderedModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_PATH = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"
OUTPUT_PATH = ANALYSIS_DIR / "transition_threshold_mu_tests_v3.csv"

TRANSITION_COLS = [
    "team_t_minus_1_vs_team_t",
    "team_vs_peer_average",
    "target_attainment",
]

STATE_LABELS = ["Aversion", "Neutral", "Appreciation"]


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


def main() -> None:
    print(f"Loading panel data from {DATA_PATH}")
    panel = pd.read_excel(DATA_PATH, sheet_name="panel_manager_period")
    posteriors = pd.read_csv(POSTERIOR_PATH)

    data = panel.merge(
        posteriors[["manager_id", "period_id", "state_label", "state_order"]],
        on=["manager_id", "period_id"],
        how="inner",
    ).sort_values(["manager_id", "period_id"])

    data["prev_state_order"] = data.groupby("manager_id")["state_order"].shift(1)
    data = data.dropna(subset=["prev_state_order", "state_order", *TRANSITION_COLS]).copy()
    data["prev_state_label"] = pd.Categorical(
        data["prev_state_order"].map({0: "Aversion", 1: "Neutral", 2: "Appreciation"}),
        categories=STATE_LABELS,
        ordered=True,
    )

    for col in TRANSITION_COLS:
        data[f"z_{col}"] = zscore(data[col])

    exog_parts = []
    prev_dummies = pd.get_dummies(data["prev_state_label"], prefix="from", drop_first=True)
    exog_parts.append(prev_dummies.astype(float))
    exog_parts.append(data[[f"z_{col}" for col in TRANSITION_COLS]].astype(float))
    exog = pd.concat(exog_parts, axis=1)

    endog = data["state_order"].astype(int)
    print("Fitting ordered-logit transition threshold model...")
    model = OrderedModel(endog, exog, distr="logit")
    res = model.fit(method="bfgs", maxiter=1000, disp=False)

    rows: list[dict[str, object]] = []
    threshold_map = [
        ("mu_1", "To Aversion / Neutral cutpoint", "Aversion | Neutral+"),
        ("mu_2", "To Neutral / Appreciation cutpoint", "Aversion+Neutral | Appreciation"),
    ]

    threshold_params = [name for name in res.params.index if "/" in str(name)]
    for idx, (mu_name, label, cutpoint) in enumerate(threshold_map):
        if idx < len(threshold_params):
            param_name = threshold_params[idx]
            estimate = float(res.params[param_name])
            se = float(res.bse[param_name])
            z_stat = estimate / se if se > 0 else np.nan
            p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat)))) if np.isfinite(z_stat) else np.nan
            p_value = max(p_value, 1e-300) if np.isfinite(p_value) else np.nan
            rows.append(
                {
                    "parameter": mu_name,
                    "label": label,
                    "cutpoint": cutpoint,
                    "estimate": estimate,
                    "standard_error": se,
                    "z_stat": z_stat,
                    "p_value": p_value,
                    "significance": stars_from_p(p_value),
                    "n_obs": int(res.nobs),
                    "method": "post-HMM ordered-logit transition threshold test",
                }
            )

    rows.append(
        {
            "parameter": "mu_3",
            "label": "To Appreciation",
            "cutpoint": "upper terminal category; no finite ordered-logit threshold",
            "estimate": np.nan,
            "standard_error": np.nan,
            "z_stat": np.nan,
            "p_value": np.nan,
            "significance": "",
            "n_obs": int(res.nobs),
            "method": "not estimated: three ordered states require two finite thresholds",
        }
    )

    output = pd.DataFrame(rows)
    print(output.to_string(index=False))
    saved_path = write_csv_with_fallback(output, OUTPUT_PATH)
    if saved_path is not None:
        print(f"Saved transition threshold tests to {saved_path}")


if __name__ == "__main__":
    main()
