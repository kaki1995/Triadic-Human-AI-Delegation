from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_PATH = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"
OUTPUT_PATH = ANALYSIS_DIR / "intrinsic_transition_mu_offdiagonal_v3.csv"
DETAIL_PATH = ANALYSIS_DIR / "intrinsic_transition_mu_offdiagonal_detail_v3.csv"

TRANSITION_COLS = [
    "team_t_minus_1_vs_team_t",
    "team_vs_peer_average",
    "target_attainment",
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


def main() -> None:
    print(f"Loading panel data from {DATA_PATH}")
    panel = pd.read_excel(DATA_PATH, sheet_name="panel_manager_period")
    posteriors = pd.read_csv(POSTERIOR_PATH)

    data = panel.merge(
        posteriors[["manager_id", "period_id", "state_order", "state_label"]],
        on=["manager_id", "period_id"],
        how="inner",
    ).sort_values(["manager_id", "period_id"])

    data["from_state_order"] = data.groupby("manager_id")["state_order"].shift(1)
    data["to_state_order"] = data["state_order"]
    data = data.dropna(subset=["from_state_order", "to_state_order", *TRANSITION_COLS]).copy()
    data["from_state_order"] = data["from_state_order"].astype(int)
    data["to_state_order"] = data["to_state_order"].astype(int)

    for col in TRANSITION_COLS:
        data[f"z_{col}"] = zscore(data[col])

    display_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    display_by_to = {to_state: {"to_state": f"To {to_state}"} for to_state in STATE_ORDER}

    for from_idx, from_state in enumerate(STATE_ORDER):
        sub = data[data["from_state_order"].eq(from_idx)].copy()
        if sub.empty:
            continue

        offdiag_states = [idx for idx in range(len(STATE_ORDER)) if idx != from_idx]
        y_code = np.zeros(len(sub), dtype=int)
        code_to_state: dict[int, int] = {}
        for code, to_idx in enumerate(offdiag_states, start=1):
            y_code[sub["to_state_order"].to_numpy() == to_idx] = code
            code_to_state[code] = to_idx

        exog = sm.add_constant(sub[[f"z_{col}" for col in TRANSITION_COLS]].astype(float), has_constant="add")
        model = sm.MNLogit(y_code, exog)
        res = model.fit(method="newton", maxiter=200, disp=False)

        params = res.params
        pvalues = res.pvalues
        if not isinstance(params, pd.DataFrame):
            params = pd.DataFrame(params, index=exog.columns)
            pvalues = pd.DataFrame(pvalues, index=exog.columns)

        # Statsmodels names the non-base equations as zero-based columns. The
        # first equation corresponds to y_code = 1, the second to y_code = 2.
        for equation_col, code in zip(params.columns, sorted(code_to_state)):
            to_idx = code_to_state[code]
            to_state = STATE_ORDER[to_idx]
            mu = float(params.loc["const", equation_col])
            p_value = float(pvalues.loc["const", equation_col])
            sig = stars_from_p(p_value)

            display_by_to[to_state][f"From {from_state}"] = f"{mu:.4f}{sig}"

            detail_rows.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "parameter": "mu",
                    "estimate": mu,
                    "p_value": p_value,
                    "significance": sig,
                    "n_obs_from_state": int(len(sub)),
                    "transition_count": int((sub["to_state_order"] == to_idx).sum()),
                    "reference_category": f"Stay in {from_state}",
                    "method": "post-HMM multinomial logit intercept; destination relative to staying in origin state",
                }
            )

        display_by_to[from_state][f"from_{from_state}"] = ""

    for to_state in STATE_ORDER:
        row = display_by_to[to_state]
        for from_state in STATE_ORDER:
            row.setdefault(f"From {from_state}", "")
        display_rows.append(row)

    display = pd.DataFrame(display_rows)[
        ["to_state", "From Aversion", "From Neutral", "From Appreciation"]
    ]
    detail = pd.DataFrame(detail_rows)
    print(display.to_string(index=False))
    print()
    print(detail.to_string(index=False))

    saved_display = write_csv_with_fallback(display, OUTPUT_PATH)
    saved_detail = write_csv_with_fallback(detail, DETAIL_PATH)
    if saved_display is not None:
        print(f"Saved intrinsic transition mu table to {saved_display}")
    if saved_detail is not None:
        print(f"Saved detail table to {saved_detail}")


if __name__ == "__main__":
    main()
