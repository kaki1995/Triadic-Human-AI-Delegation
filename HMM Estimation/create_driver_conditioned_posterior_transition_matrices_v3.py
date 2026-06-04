from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_PATH = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]

DRIVER_LABELS = {
    "team_t_minus_1_vs_team_t": "Team (t-1) vs. Team (t)",
    "team_vs_peer_average": "Team vs. Peer Average",
    "target_attainment": "Target Attainment",
}

DRIVERS = list(DRIVER_LABELS)
LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Observed": 3}
STATE_SORT = {state: idx for idx, state in enumerate(STATE_ORDER)}


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


def load_transition_data() -> pd.DataFrame:
    posterior = pd.read_csv(POSTERIOR_PATH)
    panel = pd.read_excel(
        DATA_PATH,
        sheet_name="panel_manager_period",
        usecols=["manager_id", "period_id", *DRIVERS],
    )
    data = (
        posterior.merge(panel, on=["manager_id", "period_id"], how="inner")
        .sort_values(["manager_id", "period_id"])
        .copy()
    )
    data["from_state"] = data.groupby("manager_id")["state_label"].shift(1)
    data["to_state"] = data["state_label"]
    data["transition_period"] = data["period_id"]
    return data.dropna(subset=["from_state", "to_state"])


def driver_levels(data: pd.DataFrame, driver: str) -> pd.Series:
    values = pd.to_numeric(data[driver], errors="coerce")
    unique = np.sort(values.dropna().unique())

    if unique.size <= 2:
        low_value = unique[0] if unique.size else np.nan
        high_value = unique[-1] if unique.size else np.nan
        labels = np.where(values.eq(high_value), "High", "Low")
        if unique.size == 1:
            labels = np.repeat("Observed", len(values))
        return pd.Series(labels, index=data.index)

    quantiles = values.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    if np.isclose(quantiles[0], quantiles[1]):
        quantiles = values.quantile([0.25, 0.75]).to_numpy(dtype=float)
    if np.isclose(quantiles[0], quantiles[1]):
        return pd.Series(np.repeat("Observed", len(values)), index=data.index)

    return pd.cut(
        values,
        bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)


def summarize_conditioned_transitions(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for driver in DRIVERS:
        scoped = data.dropna(subset=[driver]).copy()
        scoped["driver_level"] = driver_levels(scoped, driver)
        scoped["driver_level_order"] = scoped["driver_level"].map(LEVEL_ORDER).fillna(99)
        for level_name, level_group in scoped.sort_values("driver_level_order").groupby(
            "driver_level", sort=False
        ):
            level_n = int(len(level_group))
            driver_min = float(pd.to_numeric(level_group[driver], errors="coerce").min())
            driver_max = float(pd.to_numeric(level_group[driver], errors="coerce").max())
            driver_mean = float(pd.to_numeric(level_group[driver], errors="coerce").mean())
            for from_state in STATE_ORDER:
                origin = level_group[level_group["from_state"].eq(from_state)]
                origin_count = int(len(origin))
                manager_origin_counts = origin.groupby("manager_id").size()
                for to_state in STATE_ORDER:
                    count = int(origin["to_state"].eq(to_state).sum())
                    manager_transition_counts = (
                        origin[origin["to_state"].eq(to_state)]
                        .groupby("manager_id")
                        .size()
                    )
                    manager_rates = (
                        pd.DataFrame({"origin_count": manager_origin_counts})
                        .join(manager_transition_counts.rename("transition_count"), how="left")
                        .fillna({"transition_count": 0})
                    )
                    manager_rates["transition_rate"] = (
                        manager_rates["transition_count"] / manager_rates["origin_count"]
                    )
                    rows.append(
                        {
                            "driver": driver,
                            "driver_label": DRIVER_LABELS[driver],
                            "driver_level": str(level_name),
                            "driver_min": driver_min,
                            "driver_mean": driver_mean,
                            "driver_max": driver_max,
                            "n_transitions_in_level": level_n,
                            "from_state": from_state,
                            "to_state": to_state,
                            "n_managers_with_origin_state": int(manager_origin_counts.shape[0]),
                            "origin_count": origin_count,
                            "transition_count": count,
                            "pooled_transition_rate": count / origin_count if origin_count else np.nan,
                            "pooled_transition_rate_pct": (
                                100.0 * count / origin_count if origin_count else np.nan
                            ),
                            "mean_manager_transition_rate": (
                                float(manager_rates["transition_rate"].mean())
                                if not manager_rates.empty
                                else np.nan
                            ),
                            "mean_manager_transition_rate_pct": (
                                100.0 * float(manager_rates["transition_rate"].mean())
                                if not manager_rates.empty
                                else np.nan
                            ),
                            "median_manager_transition_rate": (
                                float(manager_rates["transition_rate"].median())
                                if not manager_rates.empty
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def matrices_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (driver, driver_label, level), group in summary.groupby(
        ["driver", "driver_label", "driver_level"], sort=False
    ):
        for from_state in STATE_ORDER:
            row: dict[str, object] = {
                "driver": driver,
                "driver_label": driver_label,
                "driver_level": level,
                "driver_level_order": LEVEL_ORDER.get(str(level), 99),
                "state_at_t": from_state,
                "state_at_t_order": STATE_SORT[from_state],
            }
            for to_state in STATE_ORDER:
                match = group[
                    group["from_state"].eq(from_state)
                    & group["to_state"].eq(to_state)
                ].iloc[0]
                row[to_state] = pct(float(match["mean_manager_transition_rate"]))
                row[f"{to_state}_n"] = int(match["transition_count"])
                row["origin_count"] = int(match["origin_count"])
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_report(matrix: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# Driver-Conditioned Posterior Transition Matrices",
        "",
        "Cells report mean manager-level decoded posterior transition rates. Each matrix",
        "conditions on the observed level of one transition driver at the transition period.",
        "Pooled counts and pooled rates are retained in the summary CSV.",
        "",
    ]
    for driver in DRIVERS:
        driver_label = DRIVER_LABELS[driver]
        levels = (
            matrix.loc[matrix["driver"].eq(driver)]
            .sort_values("driver_level_order")["driver_level"]
            .drop_duplicates()
        )
        lines.extend([f"## {driver_label}", ""])
        for level in levels:
            scoped_matrix = matrix[
                matrix["driver"].eq(driver) & matrix["driver_level"].eq(level)
            ].sort_values("state_at_t_order")[["state_at_t", *STATE_ORDER]].rename(
                columns={"state_at_t": "t -> t+1"}
            )
            scoped_summary = summary[
                summary["driver"].eq(driver) & summary["driver_level"].eq(level)
            ]
            driver_min = scoped_summary["driver_min"].iloc[0]
            driver_mean = scoped_summary["driver_mean"].iloc[0]
            driver_max = scoped_summary["driver_max"].iloc[0]
            n_level = int(scoped_summary["n_transitions_in_level"].iloc[0])
            lines.extend(
                [
                    f"### {level} ({driver_min:.3f} to {driver_max:.3f}; mean={driver_mean:.3f}; n={n_level})",
                    "",
                    scoped_matrix.to_markdown(index=False),
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    data = load_transition_data()
    summary = summarize_conditioned_transitions(data)
    matrix = matrices_from_summary(summary)
    outputs = [
        save_csv(summary, ANALYSIS_DIR / "posterior_transition_by_driver_level_summary_v3.csv"),
        save_csv(matrix, ANALYSIS_DIR / "posterior_transition_by_driver_level_matrices_v3.csv"),
        save_text(
            markdown_report(matrix, summary),
            ANALYSIS_DIR / "posterior_transition_by_driver_level_matrices_v3.md",
        ),
    ]

    print("Saved driver-conditioned posterior transition outputs:")
    for path in outputs:
        print(f"- {path}")
    print()
    for driver in DRIVERS:
        print(DRIVER_LABELS[driver])
        print(
            matrix[matrix["driver"].eq(driver)].sort_values(
                ["driver_level_order", "state_at_t_order"]
            )[
                ["driver_level", "state_at_t", *STATE_ORDER]
            ].to_string(index=False)
        )
        print()


if __name__ == "__main__":
    main()
