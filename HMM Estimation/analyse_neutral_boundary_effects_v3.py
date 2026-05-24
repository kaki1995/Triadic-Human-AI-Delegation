from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_PATH = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"
BASELINE_TRANSITION_PATH = ANALYSIS_DIR / "baseline_transition_probabilities_v3.csv"
SENSITIVITY_PATH = ANALYSIS_DIR / "state_dependent_transition_sensitivity_v3.csv"
EMISSION_PROFILE_PATH = ANALYSIS_DIR / "state_emission_profile_v3.csv"
THRESHOLD_PATH = ANALYSIS_DIR / "transition_threshold_mu_tests_v3.csv"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
SIGNAL_VARS = [
    "team_t_minus_1_vs_team_t",
    "team_vs_peer_average",
    "target_attainment",
    "forecast_accuracy",
    "recent_negative_shock",
    "override_rate",
    "composite_kpi_score",
    "ai_authority_share",
    "escalation_share",
]

TRANSITION_PAIRS = [
    ("Aversion", "Neutral"),
    ("Neutral", "Appreciation"),
    ("Appreciation", "Neutral"),
    ("Neutral", "Aversion"),
    ("Neutral", "Neutral"),
]


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * value:.2f}%"


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


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


def transition_subset(df: pd.DataFrame, previous: str, current: str) -> pd.DataFrame:
    return df[(df["prev_state_label"].eq(previous)) & (df["state_label"].eq(current))]


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = ANALYSIS_DIR / filename
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def save_text(text: str, filename: str) -> Path:
    path = ANALYSIS_DIR / filename
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def add_state_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["manager_id", "period_id"]).copy()
    grouped = df.groupby("manager_id", group_keys=False)
    for col in ["state_label", "state_order"]:
        df[f"prev_{col}"] = grouped[col].shift(1)
        df[f"next_{col}"] = grouped[col].shift(-1)
        df[f"prev2_{col}"] = grouped[col].shift(2)
        df[f"prev3_{col}"] = grouped[col].shift(3)
    return df


def posterior_transition_rates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transitions = df.dropna(subset=["prev_state_label"]).copy()
    counts = pd.crosstab(
        transitions["prev_state_label"],
        transitions["state_label"],
    ).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    rates = counts.div(counts.sum(axis=1), axis=0)

    rows = []
    for previous in STATE_ORDER:
        for current in STATE_ORDER:
            rows.append(
                {
                    "from_state": previous,
                    "to_state": current,
                    "n": int(counts.loc[previous, current]),
                    "transition_rate": float(rates.loc[previous, current]),
                    "transition_rate_pct": 100.0 * float(rates.loc[previous, current]),
                }
            )
    return pd.DataFrame(rows), transitions


def neutral_origin_next_state(df: pd.DataFrame) -> pd.DataFrame:
    neutral = df[(df["state_label"].eq("Neutral")) & df["prev_state_label"].notna()].copy()
    origin_counts = neutral["prev_state_label"].value_counts().reindex(STATE_ORDER).fillna(0).astype(int)
    origin_total = int(origin_counts.sum())

    next_ready = neutral[neutral["next_state_label"].notna()].copy()
    next_counts = pd.crosstab(
        next_ready["prev_state_label"],
        next_ready["next_state_label"],
    ).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    next_rates = next_counts.div(next_counts.sum(axis=1), axis=0)

    rows = []
    for origin in STATE_ORDER:
        row = {
            "neutral_origin_state": origin,
            "neutral_origin_n": int(origin_counts.loc[origin]),
            "neutral_origin_share": float(origin_counts.loc[origin] / origin_total),
            "neutral_origin_share_pct": 100.0 * float(origin_counts.loc[origin] / origin_total),
            "next_observed_n": int(next_counts.loc[origin].sum()),
        }
        for destination in STATE_ORDER:
            row[f"next_{destination.lower()}_n"] = int(next_counts.loc[origin, destination])
            row[f"next_{destination.lower()}_rate"] = float(next_rates.loc[origin, destination])
            row[f"next_{destination.lower()}_rate_pct"] = 100.0 * float(
                next_rates.loc[origin, destination]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_signal_profiles(df: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_transitions = len(transitions)
    for previous, current in TRANSITION_PAIRS:
        subset = transition_subset(df, previous, current)
        next_denom = int(subset["next_state_label"].notna().sum())
        row = {
            "transition": f"{previous} -> {current}",
            "from_state": previous,
            "to_state": current,
            "n": int(len(subset)),
            "share_of_all_transitions": float(len(subset) / total_transitions),
            "share_of_all_transitions_pct": 100.0 * float(len(subset) / total_transitions),
            "next_observed_n": next_denom,
        }
        for destination in STATE_ORDER:
            rate = subset["next_state_label"].eq(destination).sum() / next_denom if next_denom else np.nan
            row[f"next_{destination.lower()}_rate"] = float(rate)
            row[f"next_{destination.lower()}_rate_pct"] = 100.0 * float(rate)
        for var in SIGNAL_VARS:
            row[f"mean_{var}"] = float(subset[var].mean()) if var in subset.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_signal_tests(df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        (
            "Aversion -> Neutral vs Appreciation -> Neutral",
            ("Aversion", "Neutral"),
            ("Appreciation", "Neutral"),
        ),
        (
            "Neutral -> Appreciation vs Neutral -> Aversion",
            ("Neutral", "Appreciation"),
            ("Neutral", "Aversion"),
        ),
    ]

    rows = []
    for label, left_pair, right_pair in comparisons:
        left = transition_subset(df, *left_pair)
        right = transition_subset(df, *right_pair)
        for var in SIGNAL_VARS:
            left_values = left[var].dropna()
            right_values = right[var].dropna()
            if left_values.empty or right_values.empty:
                statistic = np.nan
                p_value = np.nan
            else:
                statistic, p_value = ttest_ind(left_values, right_values, equal_var=False)
            rows.append(
                {
                    "comparison": label,
                    "variable": var,
                    "left_transition": f"{left_pair[0]} -> {left_pair[1]}",
                    "right_transition": f"{right_pair[0]} -> {right_pair[1]}",
                    "left_mean": float(left_values.mean()) if not left_values.empty else np.nan,
                    "right_mean": float(right_values.mean()) if not right_values.empty else np.nan,
                    "difference_left_minus_right": (
                        float(left_values.mean() - right_values.mean())
                        if not left_values.empty and not right_values.empty
                        else np.nan
                    ),
                    "welch_t": float(statistic) if np.isfinite(statistic) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "significance": stars_from_p(float(p_value)) if np.isfinite(p_value) else "",
                }
            )
    return pd.DataFrame(rows)


def three_period_paths_into_neutral(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin in STATE_ORDER:
        subset = df[
            df["state_label"].eq("Neutral")
            & df["prev_state_label"].eq(origin)
            & df["prev2_state_label"].eq(origin)
            & df["prev3_state_label"].eq(origin)
        ]
        row = {
            "path_pattern": f"{origin}, {origin}, {origin} -> Neutral",
            "origin_state": origin,
            "destination_state": "Neutral",
            "n": int(len(subset)),
        }
        for var in SIGNAL_VARS:
            row[f"mean_{var}"] = float(subset[var].mean()) if var in subset.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_summary() -> pd.DataFrame:
    sensitivity = pd.read_csv(SENSITIVITY_PATH)
    rows = []
    for (covariate, previous, current), group in sensitivity.groupby(
        ["covariate", "from_state", "to_state"]
    ):
        group = group.sort_values("covariate_value_standardized")
        low = group.iloc[0]
        high = group.iloc[-1]
        mid = group.iloc[(group["covariate_value_standardized"].abs()).argmin()]
        rows.append(
            {
                "covariate": covariate,
                "from_state": previous,
                "to_state": current,
                "low_covariate_value": float(low["covariate_value_original"]),
                "low_transition_probability": float(low["transition_probability"]),
                "mid_covariate_value": float(mid["covariate_value_original"]),
                "mid_transition_probability": float(mid["transition_probability"]),
                "high_covariate_value": float(high["covariate_value_original"]),
                "high_transition_probability": float(high["transition_probability"]),
                "high_minus_low_probability": float(
                    high["transition_probability"] - low["transition_probability"]
                ),
            }
        )
    return pd.DataFrame(rows)


def row_value(df: pd.DataFrame, key_col: str, key: str, value_col: str) -> float:
    return float(df.loc[df[key_col].eq(key), value_col].iloc[0])


def transition_rate(
    df: pd.DataFrame,
    from_state: str,
    to_state: str,
    col: str = "transition_rate",
) -> float:
    mask = df["from_state"].eq(from_state) & df["to_state"].eq(to_state)
    return float(df.loc[mask, col].iloc[0])


def signal_mean(df: pd.DataFrame, transition: str, variable: str) -> float:
    return float(df.loc[df["transition"].eq(transition), f"mean_{variable}"].iloc[0])


def sensitivity_delta(df: pd.DataFrame, covariate: str, from_state: str, to_state: str) -> float:
    mask = (
        df["covariate"].eq(covariate)
        & df["from_state"].eq(from_state)
        & df["to_state"].eq(to_state)
    )
    return float(df.loc[mask, "high_minus_low_probability"].iloc[0])


def sensitivity_low_high(
    df: pd.DataFrame,
    covariate: str,
    from_state: str,
    to_state: str,
) -> tuple[float, float]:
    mask = (
        df["covariate"].eq(covariate)
        & df["from_state"].eq(from_state)
        & df["to_state"].eq(to_state)
    )
    row = df.loc[mask].iloc[0]
    return float(row["low_transition_probability"]), float(row["high_transition_probability"])


def make_markdown_report(
    baseline: pd.DataFrame,
    emissions: pd.DataFrame,
    thresholds: pd.DataFrame,
    posterior_rates: pd.DataFrame,
    neutral_paths: pd.DataFrame,
    signal_profiles: pd.DataFrame,
    signal_tests: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> str:
    baseline_long = (
        baseline.rename_axis("from_state")
        .reset_index()
        .melt(id_vars="from_state", var_name="to_state", value_name="transition_probability")
    )
    mu_1 = row_value(thresholds, "parameter", "mu_1", "estimate")
    mu_2 = row_value(thresholds, "parameter", "mu_2", "estimate")

    a_to_n = float(baseline_long.query("from_state == 'Aversion' and to_state == 'Neutral'")[
        "transition_probability"
    ].iloc[0])
    n_to_n = float(baseline_long.query("from_state == 'Neutral' and to_state == 'Neutral'")[
        "transition_probability"
    ].iloc[0])
    n_to_app = float(
        baseline_long.query("from_state == 'Neutral' and to_state == 'Appreciation'")[
            "transition_probability"
        ].iloc[0]
    )
    app_to_n = float(
        baseline_long.query("from_state == 'Appreciation' and to_state == 'Neutral'")[
            "transition_probability"
        ].iloc[0]
    )
    app_to_app = float(
        baseline_long.query("from_state == 'Appreciation' and to_state == 'Appreciation'")[
            "transition_probability"
        ].iloc[0]
    )

    av_origin = neutral_paths.loc[
        neutral_paths["neutral_origin_state"].eq("Aversion"), "neutral_origin_share"
    ].iloc[0]
    neutral_origin = neutral_paths.loc[
        neutral_paths["neutral_origin_state"].eq("Neutral"), "neutral_origin_share"
    ].iloc[0]
    app_origin = neutral_paths.loc[
        neutral_paths["neutral_origin_state"].eq("Appreciation"), "neutral_origin_share"
    ].iloc[0]

    avn_next_app = neutral_paths.loc[
        neutral_paths["neutral_origin_state"].eq("Aversion"), "next_appreciation_rate"
    ].iloc[0]
    avn_next_av = neutral_paths.loc[
        neutral_paths["neutral_origin_state"].eq("Aversion"), "next_aversion_rate"
    ].iloc[0]
    appn_next_app = neutral_paths.loc[
        neutral_paths["neutral_origin_state"].eq("Appreciation"), "next_appreciation_rate"
    ].iloc[0]
    appn_next_av = neutral_paths.loc[
        neutral_paths["neutral_origin_state"].eq("Appreciation"), "next_aversion_rate"
    ].iloc[0]

    neutral_to_app_kpi = signal_mean(signal_profiles, "Neutral -> Appreciation", "composite_kpi_score")
    neutral_to_av_kpi = signal_mean(signal_profiles, "Neutral -> Aversion", "composite_kpi_score")
    neutral_to_app_override = signal_mean(signal_profiles, "Neutral -> Appreciation", "override_rate")
    neutral_to_av_override = signal_mean(signal_profiles, "Neutral -> Aversion", "override_rate")
    av_to_neutral_authority = signal_mean(signal_profiles, "Aversion -> Neutral", "ai_authority_share")
    app_to_neutral_authority = signal_mean(signal_profiles, "Appreciation -> Neutral", "ai_authority_share")

    peer_low, peer_high = sensitivity_low_high(
        sensitivity, "team_vs_peer_average", "Neutral", "Appreciation"
    )
    peer_av_low, peer_av_high = sensitivity_low_high(
        sensitivity, "team_vs_peer_average", "Neutral", "Aversion"
    )
    app_peer_low, app_peer_high = sensitivity_low_high(
        sensitivity, "team_vs_peer_average", "Appreciation", "Appreciation"
    )

    kpi_test = signal_tests[
        signal_tests["comparison"].eq("Neutral -> Appreciation vs Neutral -> Aversion")
        & signal_tests["variable"].eq("composite_kpi_score")
    ].iloc[0]
    override_test = signal_tests[
        signal_tests["comparison"].eq("Neutral -> Appreciation vs Neutral -> Aversion")
        & signal_tests["variable"].eq("override_rate")
    ].iloc[0]
    authority_test = signal_tests[
        signal_tests["comparison"].eq("Aversion -> Neutral vs Appreciation -> Neutral")
        & signal_tests["variable"].eq("ai_authority_share")
    ].iloc[0]

    lines = [
        "# Neutral Boundary and Joint-Path Analysis",
        "",
        "## Main finding",
        "",
        (
            "The three-state HMM supports a boundary-state interpretation of Neutral. "
            "Neutral is not simply a weak form of Appreciation or a weak form of Aversion; "
            "its substantive meaning depends on the manager's adjacent state path."
        ),
        "",
        "## State ordering",
        "",
        (
            "The emission profile validates the ordered interpretation of the states: "
            f"AI authority rises from {fmt(row_value(emissions, 'State', 'Aversion', 'AI Authority Share'))} "
            f"in Aversion to {fmt(row_value(emissions, 'State', 'Neutral', 'AI Authority Share'))} "
            f"in Neutral and {fmt(row_value(emissions, 'State', 'Appreciation', 'AI Authority Share'))} "
            "in Appreciation. Escalation falls in the opposite direction."
        ),
        "",
        "## Medium/Neutral gap",
        "",
        (
            f"At the intrinsic transition baseline, Aversion moves into Neutral with probability "
            f"{pct(a_to_n)}, but direct Aversion to Appreciation movement is almost absent. "
            f"Neutral is highly persistent ({pct(n_to_n)}) and has only a {pct(n_to_app)} "
            "baseline probability of moving to Appreciation. This means the fitted model does "
            "not support the claim that managers usually move quickly from Neutral to Appreciation "
            "at baseline. A more accurate interpretation is that Neutral is the main boundary or "
            "recalibration state."
        ),
        "",
        (
            f"The ordered threshold test is consistent with this gap: the Aversion/Neutral "
            f"cutpoint is mu_1 = {fmt(mu_1)}, while the Neutral/Appreciation cutpoint is "
            f"mu_2 = {fmt(mu_2)}. The higher second threshold indicates that reaching "
            "Appreciation requires a stronger transition signal than merely leaving Aversion."
        ),
        "",
        "## Appreciation-Neutral and Neutral-Aversion joint paths",
        "",
        (
            f"Among posterior-decoded Neutral observations with a previous state, "
            f"{pct(neutral_origin)} are Neutral persistence, {pct(av_origin)} enter from "
            f"Aversion, and {pct(app_origin)} enter from Appreciation. Therefore, most Neutral "
            "observations represent continued uncertainty, but the boundary entries are asymmetric: "
            "Aversion -> Neutral is much more common than Appreciation -> Neutral."
        ),
        "",
        (
            f"The two boundary meanings differ in the next period. After Aversion -> Neutral, "
            f"the next-period probability of Appreciation is {pct(avn_next_app)} and the probability "
            f"of returning to Aversion is {pct(avn_next_av)}. After Appreciation -> Neutral, the "
            f"next-period probability of returning to Appreciation is {pct(appn_next_app)}, while "
            f"the probability of falling to Aversion is {pct(appn_next_av)}. This suggests that "
            "Neutral after Aversion is a tentative recovery state, whereas Neutral after Appreciation "
            "is a warning or cooling-off state rather than full aversion."
        ),
        "",
        "## Performance-signal interpretation",
        "",
        (
            f"Neutral -> Appreciation is associated with a higher composite KPI score "
            f"({fmt(neutral_to_app_kpi)} vs {fmt(neutral_to_av_kpi)} for Neutral -> Aversion; "
            f"Welch p {fmt_p(kpi_test['p_value'])}) and a lower "
            f"override rate ({fmt(neutral_to_app_override)} vs {fmt(neutral_to_av_override)}; "
            f"Welch p {fmt_p(override_test['p_value'])}). "
            "This supports the interpretation that upward movement out of Neutral is tied to "
            "stronger realized performance and less manual override."
        ),
        "",
        (
            f"Appreciation -> Neutral remains behaviorally closer to the high-confidence side than "
            f"Aversion -> Neutral: AI authority averages {fmt(app_to_neutral_authority)} after "
            f"Appreciation -> Neutral versus {fmt(av_to_neutral_authority)} after Aversion -> Neutral "
            f"(Welch p {fmt_p(authority_test['p_value'])}). "
            "Thus, identical Neutral classifications should not be interpreted as identical managerial "
            "positions."
        ),
        "",
        "## Covariate sensitivity",
        "",
        (
            "The transition-sensitivity analysis shows that peer benchmark performance is the "
            "clearest boundary mechanism. Across the observed range of team-vs-peer-average, "
            f"Neutral -> Appreciation rises from {pct(peer_low)} to {pct(peer_high)}, while "
            f"Neutral -> Aversion falls from {pct(peer_av_low)} to {pct(peer_av_high)}. For "
            f"Appreciation managers, Appreciation persistence rises from {pct(app_peer_low)} to "
            f"{pct(app_peer_high)}. This is the strongest support for the claim that positive "
            "performance signals stabilize Appreciation and protect Neutral managers from sliding "
            "into Aversion."
        ),
        "",
        "## Paper wording",
        "",
        (
            "A defensible paper claim is: The HMM reveals Neutral as a dynamic boundary state. "
            "Managers entering Neutral from Aversion are not equivalent to managers entering Neutral "
            "from Appreciation; the former pattern reflects tentative recovery from low trust, while "
            "the latter reflects a cooling-off or warning state within an otherwise more appreciative "
            "trajectory. The model's sequence-based classification is therefore useful because it "
            "connects current Neutral behavior to adjacent latent-state paths and to performance "
            "signals, rather than treating Neutral as a static middle category."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_excel(DATA_PATH, sheet_name="panel_manager_period")
    posteriors = pd.read_csv(POSTERIOR_PATH)
    baseline = pd.read_csv(BASELINE_TRANSITION_PATH, index_col=0)
    emissions = pd.read_csv(EMISSION_PROFILE_PATH)
    thresholds = pd.read_csv(THRESHOLD_PATH)

    data = (
        posteriors.merge(panel, on=["manager_id", "period_id"], how="left")
        .sort_values(["manager_id", "period_id"])
        .copy()
    )
    data = add_state_lags(data)

    posterior_rates, transitions = posterior_transition_rates(data)
    neutral_paths = neutral_origin_next_state(data)
    signal_profiles = boundary_signal_profiles(data, transitions)
    signal_tests = boundary_signal_tests(data)
    three_period_paths = three_period_paths_into_neutral(data)
    sensitivity = sensitivity_summary()

    outputs = [
        save_csv(posterior_rates, "neutral_boundary_posterior_transition_rates_v3.csv"),
        save_csv(neutral_paths, "neutral_boundary_origin_next_state_v3.csv"),
        save_csv(signal_profiles, "neutral_boundary_signal_profiles_v3.csv"),
        save_csv(signal_tests, "neutral_boundary_signal_tests_v3.csv"),
        save_csv(three_period_paths, "neutral_boundary_three_period_paths_v3.csv"),
        save_csv(sensitivity, "neutral_boundary_transition_sensitivity_summary_v3.csv"),
    ]

    report = make_markdown_report(
        baseline=baseline,
        emissions=emissions,
        thresholds=thresholds,
        posterior_rates=posterior_rates,
        neutral_paths=neutral_paths,
        signal_profiles=signal_profiles,
        signal_tests=signal_tests,
        sensitivity=sensitivity,
    )
    report_path = save_text(report, "neutral_boundary_analysis_note_v3.md")
    outputs.append(report_path)

    print("Saved Neutral boundary analysis outputs:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
