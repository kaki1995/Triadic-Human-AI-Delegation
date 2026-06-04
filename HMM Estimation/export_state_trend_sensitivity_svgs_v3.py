from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
SENSITIVITY_CSV = ANALYSIS_DIR / "state_dependent_transition_sensitivity_v3.csv"
TREND_CSV = ANALYSIS_DIR / "state_time_trend_tests_v3.csv"
PROFILE_CSV = ANALYSIS_DIR / "state_emission_profile_v3.csv"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
COVARIATE_LABELS = {
    "team_t_minus_1_vs_team_t": "Team (t-1) vs. Team (t)",
    "team_vs_peer_average": "Peer Average",
    "target_attainment": "Target Attainment",
}
STATE_COLORS = {
    "Aversion": "#111111",
    "Neutral": "#FF6A00",
    "Appreciation": "#0057D9",
}
TO_STATE_STYLES = {
    "Aversion": (STATE_COLORS["Aversion"], "-"),
    "Neutral": (STATE_COLORS["Neutral"], "--"),
    "Appreciation": (STATE_COLORS["Appreciation"], "-."),
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Times New Roman",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
        }
    )


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def save_figure(fig: plt.Figure, output_path: Path) -> list[Path]:
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    return [output_path, png_path]


def trend_path(row: pd.Series, periods: np.ndarray) -> np.ndarray:
    start = float(row["period_1_mean"])
    end = float(row["period_26_mean"])
    progress = (periods - periods.min()) / max(periods.max() - periods.min(), 1.0)
    return start + (end - start) * progress


def export_time_trend(metric: str, filename: str, y_label: str) -> list[Path]:
    trend = pd.read_csv(TREND_CSV)
    periods = np.arange(1, int(trend["n_periods"].max()) + 1)
    fig, ax = plt.subplots(figsize=(8.0, 4.7))

    for state in STATE_ORDER:
        row = trend[(trend["metric"] == metric) & (trend["state"] == state)].iloc[0]
        ax.plot(
            periods,
            trend_path(row, periods) * 100.0,
            color=STATE_COLORS[state],
            linewidth=1.8,
            label=state,
        )

    ax.set_xlabel("Planning cycle", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xlim(periods.min(), periods.max())
    ax.set_xticks([1, 6, 11, 16, 21, 26])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#222222", fontsize=9.5)
    fig.tight_layout()
    out = ANALYSIS_DIR / filename
    paths = save_figure(fig, out)
    plt.close(fig)
    return paths


def export_transition_sensitivity(covariate: str) -> list[Path]:
    data = pd.read_csv(SENSITIVITY_CSV)
    sub = data[data["covariate"] == covariate].copy()
    if sub.empty:
        raise ValueError(f"No sensitivity rows found for {covariate}.")

    covariate_label = COVARIATE_LABELS.get(covariate, str(sub["covariate_label"].iloc[0]))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharey=True)

    for ax, from_state in zip(axes, STATE_ORDER):
        panel = sub[sub["from_state"] == from_state].copy()
        for to_state in STATE_ORDER:
            line = panel[panel["to_state"] == to_state].sort_values("covariate_value_original")
            color, linestyle = TO_STATE_STYLES[to_state]
            ax.plot(
                line["covariate_value_original"],
                line["transition_probability"] * 100.0,
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=f"To {to_state}",
            )
        ax.set_title(f"From {from_state}", fontsize=11.5, weight="bold", loc="left")
        ax.set_xlabel(covariate_label, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Transition probability (%)", fontsize=10)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=[0.02, 0.12, 1.0, 1.0])
    out = ANALYSIS_DIR / f"figure_state_dependent_transition_sensitivity_{covariate}_v3.svg"
    paths = save_figure(fig, out)
    plt.close(fig)
    return paths


def export_state_emission_profile() -> list[Path]:
    profile = pd.read_csv(PROFILE_CSV).set_index("State").loc[STATE_ORDER]
    x = np.arange(len(STATE_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(
        x - width / 2,
        profile["AI Authority Share"] * 100.0,
        width,
        color="#0057FF",
        label="AI authority share",
    )
    ax.bar(
        x + width / 2,
        profile["Escalation Share"] * 100.0,
        width,
        color="#FF6B00",
        label="Escalation share",
    )
    ax.set_xticks(x, STATE_ORDER)
    ax.set_ylabel("State mean (%)", fontsize=11)
    ax.set_ylim(0, 75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#222222", fontsize=9.5)
    fig.tight_layout()
    out = ANALYSIS_DIR / "figure_state_emission_profile_v3.svg"
    paths = save_figure(fig, out)
    plt.close(fig)
    return paths


def main() -> None:
    setup_style()
    paths = []
    paths.extend(
        export_time_trend(
            "AI Authority Share",
            "figure_state_time_trend_ai_authority_v3.svg",
            "AI authority share (%)",
        )
    )
    paths.extend(
        export_time_trend(
            "Escalation Share",
            "figure_state_time_trend_escalation_v3.svg",
            "Escalation share (%)",
        )
    )
    paths.extend(export_state_emission_profile())
    sensitivity = pd.read_csv(SENSITIVITY_CSV, usecols=["covariate"]).drop_duplicates()
    for covariate in sensitivity["covariate"].tolist():
        paths.extend(export_transition_sensitivity(str(covariate)))

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
