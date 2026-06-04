from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_PATH = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"

OUTPUT_STEM = "posterior_transition_kpi_benchmark_matrices_v3"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
STATE_SORT = {state: idx for idx, state in enumerate(STATE_ORDER)}


@dataclass(frozen=True)
class BenchmarkSpec:
    variable: str
    label: str
    definition: str
    threshold: float
    operator: str = ">="

    def mask(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        if self.operator == ">=":
            return numeric.ge(self.threshold)
        if self.operator == "==":
            return numeric.eq(self.threshold)
        raise ValueError(f"Unsupported benchmark operator: {self.operator}")


BENCHMARKS = [
    BenchmarkSpec(
        variable="team_t_minus_1_vs_team_t",
        label="Team(t) >= Team(t-1)",
        definition="current composite KPI is at least the manager's previous-period KPI",
        threshold=0.0,
    ),
    BenchmarkSpec(
        variable="team_vs_peer_average",
        label="Team >= Peer Average",
        definition="current composite KPI is at least the same-region peer average",
        threshold=0.0,
    ),
    BenchmarkSpec(
        variable="target_attainment",
        label="Target Attained",
        definition="current composite KPI meets or exceeds the assigned KPI target",
        threshold=1.0,
        operator="==",
    ),
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


def save_figure(fig: plt.Figure, path: Path, **kwargs) -> Path:
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fig.savefig(fallback, **kwargs)
        return fallback


def load_transition_data() -> pd.DataFrame:
    posterior = pd.read_csv(POSTERIOR_PATH)
    panel = pd.read_excel(
        DATA_PATH,
        sheet_name="panel_manager_period",
        usecols=["manager_id", "period_id", *[spec.variable for spec in BENCHMARKS]],
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


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan
    mean = float(clean.mean())
    if clean.size == 1:
        return mean, np.nan, np.nan
    se = float(clean.std(ddof=1) / np.sqrt(clean.size))
    lower = max(0.0, mean - 1.96 * se)
    upper = min(1.0, mean + 1.96 * se)
    return mean, lower, upper


def pct(value: float, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{100.0 * value:.{decimals}f}%"


def interval_text(lower: float, upper: float) -> str:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return "[--]"
    return f"[{pct(lower)}-{pct(upper)}]"


def summarize_benchmark_transitions(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for benchmark_order, spec in enumerate(BENCHMARKS):
        values = pd.to_numeric(data[spec.variable], errors="coerce")
        scoped = data[spec.mask(values)].dropna(subset=[spec.variable]).copy()
        benchmark_n = int(len(scoped))
        benchmark_manager_n = int(scoped["manager_id"].nunique())
        observed_min = float(pd.to_numeric(scoped[spec.variable], errors="coerce").min())
        observed_mean = float(pd.to_numeric(scoped[spec.variable], errors="coerce").mean())
        observed_max = float(pd.to_numeric(scoped[spec.variable], errors="coerce").max())

        for from_state in STATE_ORDER:
            origin = scoped[scoped["from_state"].eq(from_state)]
            origin_count = int(len(origin))
            manager_origin_counts = origin.groupby("manager_id").size()
            for to_state in STATE_ORDER:
                cell = origin[origin["to_state"].eq(to_state)]
                transition_count = int(len(cell))
                manager_transition_counts = cell.groupby("manager_id").size()
                manager_rates = (
                    pd.DataFrame({"origin_count": manager_origin_counts})
                    .join(manager_transition_counts.rename("transition_count"), how="left")
                    .fillna({"transition_count": 0})
                )
                if not manager_rates.empty:
                    manager_rates["transition_rate"] = (
                        manager_rates["transition_count"] / manager_rates["origin_count"]
                    )
                    mean_rate, ci_lower, ci_upper = mean_ci(manager_rates["transition_rate"])
                else:
                    mean_rate, ci_lower, ci_upper = np.nan, np.nan, np.nan

                rows.append(
                    {
                        "benchmark": spec.variable,
                        "benchmark_label": spec.label,
                        "benchmark_definition": spec.definition,
                        "benchmark_order": benchmark_order,
                        "benchmark_condition": f"{spec.variable} {spec.operator} {spec.threshold:g}",
                        "benchmark_transition_count": benchmark_n,
                        "benchmark_manager_count": benchmark_manager_n,
                        "benchmark_value_min": observed_min,
                        "benchmark_value_mean": observed_mean,
                        "benchmark_value_max": observed_max,
                        "from_state": from_state,
                        "from_state_order": STATE_SORT[from_state],
                        "to_state": to_state,
                        "to_state_order": STATE_SORT[to_state],
                        "n_managers_with_origin_state": int(manager_origin_counts.shape[0]),
                        "origin_count": origin_count,
                        "transition_count": transition_count,
                        "pooled_transition_rate": (
                            transition_count / origin_count if origin_count else np.nan
                        ),
                        "mean_manager_transition_rate": mean_rate,
                        "ci95_lower": ci_lower,
                        "ci95_upper": ci_upper,
                        "mean_manager_transition_rate_pct": (
                            100.0 * mean_rate if np.isfinite(mean_rate) else np.nan
                        ),
                        "ci95_lower_pct": (
                            100.0 * ci_lower if np.isfinite(ci_lower) else np.nan
                        ),
                        "ci95_upper_pct": (
                            100.0 * ci_upper if np.isfinite(ci_upper) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def matrices_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (benchmark, benchmark_label, benchmark_order), group in summary.groupby(
        ["benchmark", "benchmark_label", "benchmark_order"], sort=False
    ):
        for from_state in STATE_ORDER:
            row: dict[str, object] = {
                "benchmark": benchmark,
                "benchmark_label": benchmark_label,
                "benchmark_order": benchmark_order,
                "state_at_t_minus_1": from_state,
                "state_at_t_minus_1_order": STATE_SORT[from_state],
            }
            for to_state in STATE_ORDER:
                match = group[
                    group["from_state"].eq(from_state) & group["to_state"].eq(to_state)
                ].iloc[0]
                mean_value = float(match["mean_manager_transition_rate"])
                lower = float(match["ci95_lower"])
                upper = float(match["ci95_upper"])
                row[to_state] = pct(mean_value)
                row[f"{to_state}_ci95"] = interval_text(lower, upper)
                row[f"{to_state}_transition_count"] = int(match["transition_count"])
                row["origin_count"] = int(match["origin_count"])
            rows.append(row)
    return pd.DataFrame(rows)


def matrix_cell(
    summary: pd.DataFrame,
    benchmark: str,
    from_state: str,
    to_state: str,
) -> tuple[str, str]:
    match = summary[
        summary["benchmark"].eq(benchmark)
        & summary["from_state"].eq(from_state)
        & summary["to_state"].eq(to_state)
    ]
    if match.empty:
        return "--", "[--]"
    row = match.iloc[0]
    return (
        pct(float(row["mean_manager_transition_rate"])),
        interval_text(float(row["ci95_lower"]), float(row["ci95_upper"])),
    )


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def draw_publication_table(summary: pd.DataFrame) -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(18.5, 6.0), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    left = 0.035
    right = 0.985
    title_y = 0.930
    table_top = 0.875
    table_bottom = 0.145
    label_col_w = 0.085
    block_gap = 0.040
    block_w = (right - left - label_col_w - 2 * block_gap) / 3
    col_w = block_w / 3

    block_label_y = 0.810
    t_label_y = 0.735
    column_y = 0.650
    rule_under_cols_y = 0.603
    row_ys = [0.505, 0.382, 0.259]

    ax.text(left, title_y, "Table 4", fontsize=16, weight="bold", ha="left", va="center")
    ax.text(
        left + 0.075,
        title_y,
        "The Mean Posterior Transition Matrices by KPI Benchmark*",
        fontsize=16,
        weight="bold",
        ha="left",
        va="center",
    )
    ax.plot([left, right], [table_top, table_top], color="#111111", lw=0.9)

    ax.text(left, column_y, r"$t-1$", fontsize=12, style="italic", ha="left", va="center")
    ax.plot([left, right], [rule_under_cols_y, rule_under_cols_y], color="#111111", lw=0.65)
    for row_y, state in zip(row_ys, STATE_ORDER):
        ax.text(left, row_y, state, fontsize=11.0, ha="left", va="center")

    for benchmark_idx, spec in enumerate(BENCHMARKS):
        block_left = left + label_col_w + benchmark_idx * (block_w + block_gap)
        block_right = block_left + block_w
        block_center = (block_left + block_right) / 2

        ax.text(
            block_center,
            block_label_y,
            spec.label,
            fontsize=13.0,
            ha="center",
            va="center",
        )
        ax.plot(
            [block_left, block_right],
            [block_label_y - 0.045, block_label_y - 0.045],
            color="#111111",
            lw=0.7,
        )
        ax.text(
            block_center,
            t_label_y,
            r"$t$",
            fontsize=12,
            style="italic",
            ha="center",
            va="center",
        )
        ax.plot(
            [block_left, block_right],
            [t_label_y - 0.048, t_label_y - 0.048],
            color="#444444",
            lw=0.65,
        )

        for col_idx, to_state in enumerate(STATE_ORDER):
            x = block_left + (col_idx + 0.5) * col_w
            ax.text(x, column_y, to_state, fontsize=10.5, ha="center", va="center")

        for row_y, from_state in zip(row_ys, STATE_ORDER):
            for col_idx, to_state in enumerate(STATE_ORDER):
                x = block_left + (col_idx + 0.5) * col_w
                mean_text, ci_text = matrix_cell(summary, spec.variable, from_state, to_state)
                ax.text(
                    x,
                    row_y + 0.027,
                    mean_text,
                    fontsize=11.0,
                    weight="bold" if mean_text != "--" else "normal",
                    ha="center",
                    va="center",
                )
                ax.text(
                    x,
                    row_y - 0.026,
                    ci_text,
                    fontsize=9.4,
                    ha="center",
                    va="center",
                )

    ax.plot([left, right], [table_bottom, table_bottom], color="#111111", lw=0.75)
    note = (
        "*95% confidence intervals in brackets. Matrices condition on observations where the "
        "named KPI benchmark is met at period t."
    )
    ax.text(left, 0.092, note, fontsize=9.4, ha="left", va="center")
    definition = "; ".join(f"{spec.label}: {spec.definition}" for spec in BENCHMARKS)
    ax.text(left, 0.052, definition, fontsize=8.6, ha="left", va="center")
    return fig


def markdown_report(summary: pd.DataFrame, matrix: pd.DataFrame) -> str:
    lines = [
        "# Table 4. The Mean Posterior Transition Matrices by KPI Benchmark",
        "",
        "Cells report mean manager-level decoded posterior transition rates, with 95% "
        "confidence intervals in brackets. Each matrix conditions on observations where "
        "the named KPI benchmark is met at period t.",
        "",
    ]
    rows = []
    for from_state in STATE_ORDER:
        row: dict[str, str] = {"t-1": from_state}
        for spec in BENCHMARKS:
            for to_state in STATE_ORDER:
                mean_text, ci_text = matrix_cell(summary, spec.variable, from_state, to_state)
                row[f"{spec.label} | {to_state}"] = f"{mean_text} {ci_text}"
        rows.append(row)
    lines.extend([pd.DataFrame(rows).to_markdown(index=False), ""])
    lines.extend(["## Benchmark definitions", ""])
    for spec in BENCHMARKS:
        scoped = summary[summary["benchmark"].eq(spec.variable)].iloc[0]
        lines.append(
            f"- {spec.label}: {spec.definition}; "
            f"n={int(scoped['benchmark_transition_count'])} transitions, "
            f"{int(scoped['benchmark_manager_count'])} managers."
        )
    lines.append("")
    lines.append(
        "Note: confidence intervals use a normal approximation around the mean manager-level "
        "transition rate among managers with at least one origin-state transition in the "
        "benchmark-conditioned subset."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    data = load_transition_data()
    summary = summarize_benchmark_transitions(data)
    matrix = matrices_from_summary(summary)
    fig = draw_publication_table(summary)
    outputs = [
        save_csv(summary, ANALYSIS_DIR / f"{OUTPUT_STEM}_summary.csv"),
        save_csv(matrix, ANALYSIS_DIR / f"{OUTPUT_STEM}.csv"),
        save_text(markdown_report(summary, matrix), ANALYSIS_DIR / f"{OUTPUT_STEM}.md"),
        save_figure(
            fig,
            ANALYSIS_DIR / f"{OUTPUT_STEM}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        ),
        save_figure(
            fig,
            ANALYSIS_DIR / f"{OUTPUT_STEM}.pdf",
            bbox_inches="tight",
            facecolor="white",
        ),
    ]
    plt.close(fig)

    print("Saved KPI benchmark posterior transition matrix outputs:")
    for path in outputs:
        print(f"- {path}")
    print()
    print(matrix[["benchmark_label", "state_at_t_minus_1", *STATE_ORDER]].to_string(index=False))


if __name__ == "__main__":
    main()
