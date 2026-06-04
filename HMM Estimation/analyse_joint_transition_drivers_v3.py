from __future__ import annotations

import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy.special import softmax


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts"
ANALYSIS_DIR = ARTIFACT_DIR / "Analysis_v3"
MODEL_PATH = ARTIFACT_DIR / "best_model_artifacts_v3_2emissions.pkl"
PANEL_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
POSTERIOR_PATH = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"

STATE_LABELS = ["Aversion", "Neutral", "Appreciation"]
STATE_ORDER = {state: idx for idx, state in enumerate(STATE_LABELS)}
TRANSITION_VARS = [
    "team_t_minus_1_vs_team_t",
    "team_vs_peer_average",
    "target_attainment",
]
VARIABLE_LABELS = {
    "team_t_minus_1_vs_team_t": "Team (t-1) vs. Team (t)",
    "team_vs_peer_average": "Peer Average",
    "target_attainment": "Target Attainment",
}
SHORT_VARIABLE_LABELS = {
    "team_t_minus_1_vs_team_t": "Team (t-1) vs. Team (t)",
    "team_vs_peer_average": "Peer Average",
    "target_attainment": "Target Attainment",
}
PAIR_SPECS = [
    {
        "key": "peer_target",
        "x_var": "team_vs_peer_average",
        "y_var": "target_attainment",
        "title": "Peer Average x Target Attainment",
    },
    {
        "key": "change_peer",
        "x_var": "team_t_minus_1_vs_team_t",
        "y_var": "team_vs_peer_average",
        "title": "Team (t-1) vs. Team (t) x Peer Average",
    },
    {
        "key": "change_target",
        "x_var": "team_t_minus_1_vs_team_t",
        "y_var": "target_attainment",
        "title": "Team (t-1) vs. Team (t) x Target Attainment",
    },
]
DESTINATION_COLORS = {
    "Aversion": "#111111",
    "Neutral": "#FF6A00",
    "Appreciation": "#0057D9",
}
ORIGIN_COLORS = {
    "Aversion": "#111111",
    "Neutral": "#FF6A00",
    "Appreciation": "#0057D9",
}
TABLE_HEADER_COLORS = {
    "Aversion": "#D9D9D9",
    "Neutral": "#FFD8B8",
    "Appreciation": "#C7DAFF",
}


@dataclass
class Params:
    logit_pi: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    W: np.ndarray
    log_sigma: np.ndarray


setattr(sys.modules["__main__"], "Params", Params)


def patch_pandas_stringarray_pickle() -> None:
    """Allow the project pickle, produced under an older pandas, to load."""
    try:
        from pandas.core.arrays.string_ import StringArray
    except ImportError:
        return

    original = StringArray.__setstate__
    if getattr(original, "_triadic_pickle_patch", False):
        return

    def patched(self: object, state: object) -> object:
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], np.ndarray):
            StringArray.__init__(self, state[1], copy=False)
            return None
        return original(self, state)

    patched._triadic_pickle_patch = True  # type: ignore[attr-defined]
    StringArray.__setstate__ = patched  # type: ignore[method-assign]


def load_inputs() -> tuple[dict[str, object], Params, pd.DataFrame, pd.DataFrame]:
    patch_pandas_stringarray_pickle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MODEL_PATH.open("rb") as stream:
            artifacts = pickle.load(stream)
    panel = pd.read_excel(PANEL_PATH, sheet_name="panel_manager_period")
    posteriors = pd.read_csv(POSTERIOR_PATH)
    return artifacts, artifacts["best_model"], panel, posteriors


def ordered_raw_states(artifacts: dict[str, object], model: Params) -> list[int]:
    emission_cols = list(artifacts["emission_cols"])
    authority_idx = emission_cols.index("ai_authority_share")
    original_means = artifacts["y_scaler"].inverse_transform(model.mu)
    return np.argsort(original_means[:, authority_idx]).tolist()


def variable_grid(panel: pd.DataFrame, variable: str, n_points: int) -> np.ndarray:
    if variable == "target_attainment":
        return np.array([0.0, 1.0])
    return np.linspace(float(panel[variable].min()), float(panel[variable].max()), n_points)


def variable_levels(panel: pd.DataFrame, variable: str) -> dict[str, float]:
    if variable == "target_attainment":
        return {"low": 0.0, "high": 1.0}
    values = panel[variable].astype(float)
    return {
        "low": float(values.quantile(0.10)),
        "high": float(values.quantile(0.90)),
    }


def base_predictor_matrix(
    artifacts: dict[str, object],
    n_rows: int,
) -> np.ndarray:
    x_scaler = artifacts["x_scaler"]
    return np.tile(x_scaler.mean_, (n_rows, 1))


def predict_transition_probabilities(
    artifacts: dict[str, object],
    model: Params,
    from_state: str,
    values: dict[str, float],
) -> dict[str, float]:
    transition_cols = list(artifacts["transition_cols"])
    x_scaler = artifacts["x_scaler"]
    ordered_states = ordered_raw_states(artifacts, model)
    raw_from = ordered_states[STATE_LABELS.index(from_state)]

    predictors = base_predictor_matrix(artifacts, 1)
    for variable, value in values.items():
        predictors[:, transition_cols.index(variable)] = value
    predictors_scaled = x_scaler.transform(predictors)
    logits = model.alpha[raw_from][None, :] + predictors_scaled @ model.beta[raw_from].T
    probabilities = softmax(logits, axis=1)[:, ordered_states].ravel()
    return {state: float(probabilities[idx]) for idx, state in enumerate(STATE_LABELS)}


def build_surface_frame(
    artifacts: dict[str, object],
    model: Params,
    panel: pd.DataFrame,
    n_points: int = 61,
) -> pd.DataFrame:
    transition_cols = list(artifacts["transition_cols"])
    x_scaler = artifacts["x_scaler"]
    ordered_states = ordered_raw_states(artifacts, model)
    rows: list[pd.DataFrame] = []

    for pair in PAIR_SPECS:
        x_var = str(pair["x_var"])
        y_var = str(pair["y_var"])
        held_vars = [col for col in TRANSITION_VARS if col not in {x_var, y_var}]
        held_var = held_vars[0]
        x_values = variable_grid(panel, x_var, n_points)
        y_values = variable_grid(panel, y_var, n_points)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        predictors = base_predictor_matrix(artifacts, x_grid.size)
        predictors[:, transition_cols.index(x_var)] = x_grid.ravel()
        predictors[:, transition_cols.index(y_var)] = y_grid.ravel()
        predictors_scaled = x_scaler.transform(predictors)

        for from_state in STATE_LABELS:
            raw_from = ordered_states[STATE_LABELS.index(from_state)]
            logits = model.alpha[raw_from][None, :] + predictors_scaled @ model.beta[raw_from].T
            probabilities = softmax(logits, axis=1)[:, ordered_states]

            for destination_idx, destination in enumerate(STATE_LABELS):
                rows.append(
                    pd.DataFrame(
                        {
                            "driver_pair": str(pair["key"]),
                            "from_state": from_state,
                            "to_state": destination,
                            "x_variable": x_var,
                            "y_variable": y_var,
                            "held_constant_variable": held_var,
                            "held_constant_value": float(
                                x_scaler.mean_[transition_cols.index(held_var)]
                            ),
                            "x_value": x_grid.ravel(),
                            "y_value": y_grid.ravel(),
                            "transition_probability": probabilities[:, destination_idx],
                        }
                    )
                )

    return pd.concat(rows, ignore_index=True)


def probability_grid(surface: pd.DataFrame, pair_key: str, from_state: str, to_state: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = surface[
        surface["driver_pair"].eq(pair_key)
        & surface["from_state"].eq(from_state)
        & surface["to_state"].eq(to_state)
    ]
    x_values = np.sort(subset["x_value"].unique())
    y_values = np.sort(subset["y_value"].unique())
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    probability = (
        subset.pivot(index="y_value", columns="x_value", values="transition_probability")
        .reindex(index=y_values, columns=x_values)
        .to_numpy()
    )
    return x_grid, y_grid, probability


def style_3d_axis(axis: plt.Axes, pair: dict[str, object]) -> None:
    x_var = str(pair["x_var"])
    y_var = str(pair["y_var"])
    axis.set_xlabel(VARIABLE_LABELS[x_var], labelpad=9)
    axis.set_ylabel(VARIABLE_LABELS[y_var], labelpad=9)
    axis.set_zlabel("Probability (%)", labelpad=8)
    axis.set_zlim(0, 100)
    axis.zaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if x_var == "target_attainment":
        axis.set_xticks([0.0, 1.0], ["No", "Yes"])
    if y_var == "target_attainment":
        axis.set_yticks([0.0, 1.0], ["No", "Yes"])
    axis.view_init(elev=27, azim=-128)
    axis.grid(True, alpha=0.22)
    axis.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
    axis.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
    axis.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))


def plot_origin_transition_overlay(
    surface: pd.DataFrame,
    from_state: str,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.edgecolor": "#292929",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )
    figure = plt.figure(figsize=(18.0, 5.9), dpi=220)

    for panel_idx, pair in enumerate(PAIR_SPECS, start=1):
        axis = figure.add_subplot(1, 3, panel_idx, projection="3d")
        for destination in STATE_LABELS:
            x_grid, y_grid, probability = probability_grid(
                surface, str(pair["key"]), from_state, destination
            )
            alpha = 0.50 if destination != from_state else 0.28
            axis.plot_surface(
                x_grid,
                y_grid,
                100.0 * probability,
                color=DESTINATION_COLORS[destination],
                linewidth=0,
                antialiased=True,
                alpha=alpha,
                shade=True,
            )
        axis.set_title(str(pair["title"]), fontsize=12, pad=10)
        style_3d_axis(axis, pair)

    legend_handles = [
        Patch(facecolor=DESTINATION_COLORS[state], alpha=0.55, label=f"{from_state} -> {state}")
        for state in STATE_LABELS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )
    figure.suptitle(
        f"Joint Transition Drivers from {from_state}",
        fontsize=15,
        y=0.99,
    )
    figure.text(
        0.5,
        0.02,
        (
            "Multiple destination surfaces are overlaid in each panel. "
            "Target Attainment is binary; the connecting plane is a visual guide between endpoints."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.02, right=0.985, bottom=0.13, top=0.82, wspace=0.02)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_neutral_boundary_overlay(surface: pd.DataFrame, output_path: Path) -> None:
    plot_origin_transition_overlay(surface, "Neutral", output_path)


def plot_appreciation_reach_overlay(surface: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.edgecolor": "#292929",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )
    figure = plt.figure(figsize=(18.0, 5.9), dpi=220)

    for panel_idx, pair in enumerate(PAIR_SPECS, start=1):
        axis = figure.add_subplot(1, 3, panel_idx, projection="3d")
        for from_state in STATE_LABELS:
            x_grid, y_grid, probability = probability_grid(
                surface, str(pair["key"]), from_state, "Appreciation"
            )
            axis.plot_surface(
                x_grid,
                y_grid,
                100.0 * probability,
                color=ORIGIN_COLORS[from_state],
                linewidth=0,
                antialiased=True,
                alpha=0.43 if from_state != "Aversion" else 0.30,
                shade=True,
            )
        axis.set_title(str(pair["title"]), fontsize=12, pad=10)
        style_3d_axis(axis, pair)

    legend_handles = [
        Patch(facecolor=ORIGIN_COLORS[state], alpha=0.50, label=f"{state} -> Appreciation")
        for state in STATE_LABELS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )
    figure.suptitle(
        "How Hard Is It to Reach Appreciation?",
        fontsize=15,
        y=0.99,
    )
    figure.text(
        0.5,
        0.02,
        (
            "Surfaces compare the probability of ending in Appreciation from each origin state. "
            "Target Attainment is binary; the connecting plane is a visual guide between endpoints."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.02, right=0.985, bottom=0.13, top=0.82, wspace=0.02)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def scenario_probability_lookup(
    scenarios: pd.DataFrame,
    pair_key: str,
    from_state: str,
    x_level: str,
    y_level: str,
) -> dict[str, float]:
    subset = scenarios[
        scenarios["driver_pair"].eq(pair_key)
        & scenarios["from_state"].eq(from_state)
        & scenarios["x_level"].eq(x_level)
        & scenarios["y_level"].eq(y_level)
    ]
    return {
        str(row["to_state"]): float(row["transition_probability"])
        for _, row in subset.iterrows()
    }


def closeup_limits(scenarios: pd.DataFrame, pair: dict[str, object]) -> tuple[tuple[float, float], tuple[float, float]]:
    pair_scenarios = scenarios[scenarios["driver_pair"].eq(str(pair["key"]))]
    x_var = str(pair["x_var"])
    y_var = str(pair["y_var"])
    if x_var == "target_attainment":
        x_limits = (0.0, 1.0)
    else:
        x_values = pair_scenarios["x_value"].astype(float)
        x_limits = (float(x_values.min()), float(x_values.max()))
    if y_var == "target_attainment":
        y_limits = (0.0, 1.0)
    else:
        y_values = pair_scenarios["y_value"].astype(float)
        y_limits = (float(y_values.min()), float(y_values.max()))
    return x_limits, y_limits


def visible_probability_max(
    surface: pd.DataFrame,
    scenarios: pd.DataFrame,
    from_state: str,
    to_state: str,
) -> float:
    max_probability = 0.0
    for pair in PAIR_SPECS:
        x_limits, y_limits = closeup_limits(scenarios, pair)
        subset = surface[
            surface["driver_pair"].eq(str(pair["key"]))
            & surface["from_state"].eq(from_state)
            & surface["to_state"].eq(to_state)
            & surface["x_value"].between(*x_limits)
            & surface["y_value"].between(*y_limits)
        ]
        if not subset.empty:
            max_probability = max(max_probability, float(subset["transition_probability"].max()))
    return max_probability


def add_scenario_table(axis: plt.Axes, scenarios: pd.DataFrame, pair: dict[str, object]) -> None:
    axis.axis("off")
    pair_key = str(pair["key"])
    rows = [
        ("Low / Low", "low", "low"),
        ("High / Low", "high", "low"),
        ("Low / High", "low", "high"),
        ("High / High", "high", "high"),
    ]
    cell_text = []
    for _, x_level, y_level in rows:
        probabilities = scenario_probability_lookup(scenarios, pair_key, "Neutral", x_level, y_level)
        cell_text.append([f"{100.0 * probabilities[state]:.1f}%" for state in STATE_LABELS])

    table = axis.table(
        cellText=cell_text,
        colLabels=["Aversion", "Neutral", "Appreciation"],
        rowLabels=[row[0] for row in rows],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.42)
    for state_idx, state in enumerate(STATE_LABELS):
        cell = table[(0, state_idx)]
        cell.set_facecolor(TABLE_HEADER_COLORS[state])
        cell.set_edgecolor("#666666")
        cell.set_text_props(weight="bold", color="#111111")
    for row_idx in range(1, len(rows) + 1):
        row_label_cell = table[(row_idx, -1)]
        row_label_cell.set_facecolor("#F3F3F3")
        row_label_cell.set_edgecolor("#777777")
        row_label_cell.set_text_props(weight="bold", color="#222222")
        for col_idx in range(len(STATE_LABELS)):
            table[(row_idx, col_idx)].set_edgecolor("#A0A0A0")

    axis.set_title("Neutral-origin scenario probabilities", fontsize=9, pad=1)


def plot_neutral_appreciation_closeup(
    surface: pd.DataFrame,
    scenarios: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.edgecolor": "#292929",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )
    figure = plt.figure(figsize=(18.0, 8.2), dpi=220)
    grid = GridSpec(2, 3, figure=figure, height_ratios=[3.2, 1.25], hspace=0.08, wspace=0.04)
    zmax = max(10.0, np.ceil(100.0 * visible_probability_max(surface, scenarios, "Neutral", "Appreciation") / 5.0) * 5.0)

    for panel_idx, pair in enumerate(PAIR_SPECS):
        axis = figure.add_subplot(grid[0, panel_idx], projection="3d")
        x_grid, y_grid, probability = probability_grid(
            surface, str(pair["key"]), "Neutral", "Appreciation"
        )
        x_limits, y_limits = closeup_limits(scenarios, pair)
        x_mask = (x_grid[0, :] >= x_limits[0]) & (x_grid[0, :] <= x_limits[1])
        y_mask = (y_grid[:, 0] >= y_limits[0]) & (y_grid[:, 0] <= y_limits[1])
        x_grid = x_grid[np.ix_(y_mask, x_mask)]
        y_grid = y_grid[np.ix_(y_mask, x_mask)]
        probability = probability[np.ix_(y_mask, x_mask)]
        axis.plot_surface(
            x_grid,
            y_grid,
            100.0 * probability,
            color=DESTINATION_COLORS["Appreciation"],
            linewidth=0,
            antialiased=True,
            alpha=0.96,
            shade=True,
        )
        axis.set_title(str(pair["title"]), fontsize=12, pad=10)
        style_3d_axis(axis, pair)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_zlim(0, zmax)
        axis.set_zlabel("P(Appreciation) (%)", labelpad=8)
        axis.zaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.view_init(elev=24, azim=-118)

        table_axis = figure.add_subplot(grid[1, panel_idx])
        add_scenario_table(table_axis, scenarios, pair)

    figure.suptitle(
        "Close-Up: Crossing from Neutral to Appreciation",
        fontsize=15,
        y=0.985,
    )
    figure.text(
        0.5,
        0.02,
        (
            "3D panels zoom to the 10th-90th percentile range for continuous drivers; "
            "tables report competing next-state probabilities at low/high scenarios."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.02, right=0.985, bottom=0.09, top=0.90)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_scenario_summary(
    artifacts: dict[str, object],
    model: Params,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in PAIR_SPECS:
        x_var = str(pair["x_var"])
        y_var = str(pair["y_var"])
        x_levels = variable_levels(panel, x_var)
        y_levels = variable_levels(panel, y_var)

        for from_state in STATE_LABELS:
            for x_label, x_value in x_levels.items():
                for y_label, y_value in y_levels.items():
                    probabilities = predict_transition_probabilities(
                        artifacts,
                        model,
                        from_state,
                        {x_var: x_value, y_var: y_value},
                    )
                    for to_state, probability in probabilities.items():
                        rows.append(
                            {
                                "driver_pair": str(pair["key"]),
                                "from_state": from_state,
                                "to_state": to_state,
                                "x_variable": x_var,
                                "x_level": x_label,
                                "x_value": x_value,
                                "y_variable": y_var,
                                "y_level": y_label,
                                "y_value": y_value,
                                "transition_probability": probability,
                                "transition_probability_pct": 100.0 * probability,
                            }
                        )
    return pd.DataFrame(rows)


def build_interaction_contrasts(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = scenarios.groupby(
        ["driver_pair", "from_state", "to_state", "x_variable", "y_variable"],
        sort=False,
    )
    for keys, group in grouped:
        lookup = {
            (str(row["x_level"]), str(row["y_level"])): float(row["transition_probability"])
            for _, row in group.iterrows()
        }
        contrast = (
            lookup[("high", "high")]
            - lookup[("high", "low")]
            - lookup[("low", "high")]
            + lookup[("low", "low")]
        )
        rows.append(
            {
                "driver_pair": keys[0],
                "from_state": keys[1],
                "to_state": keys[2],
                "x_variable": keys[3],
                "y_variable": keys[4],
                "low_low_probability": lookup[("low", "low")],
                "high_low_probability": lookup[("high", "low")],
                "low_high_probability": lookup[("low", "high")],
                "high_high_probability": lookup[("high", "high")],
                "interaction_contrast": contrast,
                "interaction_contrast_pct_points": 100.0 * contrast,
            }
        )
    return pd.DataFrame(rows)


def infer_gamma_label_map(posteriors: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for state_idx in sorted(posteriors["most_likely_state"].dropna().astype(int).unique()):
        subset = posteriors[posteriors["most_likely_state"].eq(state_idx)]
        if subset.empty:
            continue
        mapping[f"gamma_state_{state_idx}"] = str(subset["state_label"].mode().iloc[0])
    return mapping


def build_manager_posterior_probability_matrix(posteriors: pd.DataFrame) -> pd.DataFrame:
    gamma_map = infer_gamma_label_map(posteriors)
    rows: list[dict[str, object]] = []
    for manager_id, group in posteriors.sort_values(["manager_id", "period_id"]).groupby("manager_id"):
        row: dict[str, object] = {"manager_id": manager_id}
        for _, item in group.iterrows():
            period = int(item["period_id"])
            for gamma_col, label in gamma_map.items():
                row[f"period_{period:02d}_posterior_{label.lower()}"] = float(item[gamma_col])
        rows.append(row)
    return pd.DataFrame(rows)


def build_manager_state_sequence_summary(posteriors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    symbol = {"Aversion": "A", "Neutral": "N", "Appreciation": "P"}
    for manager_id, group in posteriors.sort_values(["manager_id", "period_id"]).groupby("manager_id"):
        states = group["state_label"].tolist()
        orders = group["state_order"].astype(int).to_numpy()
        row: dict[str, object] = {
            "manager_id": manager_id,
            "n_periods": int(len(group)),
            "decoded_state_sequence": " ".join(symbol[state] for state in states),
            "dominant_state": str(group["state_label"].value_counts().idxmax()),
        }
        for state in STATE_LABELS:
            count = int(group["state_label"].eq(state).sum())
            row[f"{state.lower()}_periods"] = count
            row[f"{state.lower()}_share"] = float(count / len(group))
        deltas = np.diff(orders)
        row["upward_transitions"] = int((deltas > 0).sum())
        row["downward_transitions"] = int((deltas < 0).sum())
        row["same_state_transitions"] = int((deltas == 0).sum())
        row["net_state_movement"] = int(orders[-1] - orders[0])
        rows.append(row)
    return pd.DataFrame(rows)


def build_manager_transition_matrices(posteriors: pd.DataFrame) -> pd.DataFrame:
    data = posteriors.sort_values(["manager_id", "period_id"]).copy()
    data["from_state"] = data.groupby("manager_id")["state_label"].shift(1)
    data = data.dropna(subset=["from_state"]).copy()

    rows: list[dict[str, object]] = []
    for manager_id, group in data.groupby("manager_id"):
        counts = (
            pd.crosstab(group["from_state"], group["state_label"])
            .reindex(index=STATE_LABELS, columns=STATE_LABELS, fill_value=0)
            .astype(int)
        )
        denominators = counts.sum(axis=1)
        for from_state in STATE_LABELS:
            for to_state in STATE_LABELS:
                denom = int(denominators.loc[from_state])
                count = int(counts.loc[from_state, to_state])
                rows.append(
                    {
                        "manager_id": manager_id,
                        "from_state": from_state,
                        "to_state": to_state,
                        "transition_count": count,
                        "from_state_transition_count": denom,
                        "transition_rate": float(count / denom) if denom else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_manager_joint_path_profiles(posteriors: pd.DataFrame) -> pd.DataFrame:
    data = posteriors.sort_values(["manager_id", "period_id"]).copy()
    grouped = data.groupby("manager_id", group_keys=False)
    data["previous_state"] = grouped["state_label"].shift(1)
    data["next_state"] = grouped["state_label"].shift(-1)
    data = data.dropna(subset=["previous_state", "next_state"]).copy()
    data["three_period_path"] = (
        data["previous_state"] + " -> " + data["state_label"] + " -> " + data["next_state"]
    )

    rows: list[dict[str, object]] = []
    for manager_id, group in data.groupby("manager_id"):
        counts = group["three_period_path"].value_counts()
        total = int(counts.sum())
        for path, count in counts.items():
            previous, current, following = path.split(" -> ")
            rows.append(
                {
                    "manager_id": manager_id,
                    "three_period_path": path,
                    "previous_state": previous,
                    "current_state": current,
                    "next_state": following,
                    "path_count": int(count),
                    "path_share": float(count / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_manager_transition_summary(transition_matrices: pd.DataFrame) -> pd.DataFrame:
    pivot = transition_matrices.pivot_table(
        index="manager_id",
        columns=["from_state", "to_state"],
        values="transition_rate",
        aggfunc="first",
    )

    rows: list[dict[str, object]] = []
    for manager_id, row in pivot.iterrows():
        def rate(from_state: str, to_state: str) -> float:
            try:
                return float(row[(from_state, to_state)])
            except KeyError:
                return np.nan

        rows.append(
            {
                "manager_id": manager_id,
                "neutral_to_appreciation_rate": rate("Neutral", "Appreciation"),
                "neutral_to_aversion_rate": rate("Neutral", "Aversion"),
                "neutral_persistence_rate": rate("Neutral", "Neutral"),
                "appreciation_persistence_rate": rate("Appreciation", "Appreciation"),
                "appreciation_to_neutral_rate": rate("Appreciation", "Neutral"),
                "aversion_to_neutral_rate": rate("Aversion", "Neutral"),
                "aversion_persistence_rate": rate("Aversion", "Aversion"),
            }
        )
    return pd.DataFrame(rows)


def pct(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{100.0 * value:.2f}%"


def scenario_probability(
    scenarios: pd.DataFrame,
    driver_pair: str,
    from_state: str,
    to_state: str,
    x_level: str,
    y_level: str,
) -> float:
    row = scenarios[
        scenarios["driver_pair"].eq(driver_pair)
        & scenarios["from_state"].eq(from_state)
        & scenarios["to_state"].eq(to_state)
        & scenarios["x_level"].eq(x_level)
        & scenarios["y_level"].eq(y_level)
    ].iloc[0]
    return float(row["transition_probability"])


def make_markdown_note(
    scenarios: pd.DataFrame,
    contrasts: pd.DataFrame,
    manager_summary: pd.DataFrame,
) -> str:
    peer_target_app_hh = scenario_probability(
        scenarios, "peer_target", "Neutral", "Appreciation", "high", "high"
    )
    peer_target_app_hl = scenario_probability(
        scenarios, "peer_target", "Neutral", "Appreciation", "high", "low"
    )
    peer_target_av_hh = scenario_probability(
        scenarios, "peer_target", "Neutral", "Aversion", "high", "high"
    )
    peer_target_av_hl = scenario_probability(
        scenarios, "peer_target", "Neutral", "Aversion", "high", "low"
    )

    change_peer_app_hl = scenario_probability(
        scenarios, "change_peer", "Neutral", "Appreciation", "high", "low"
    )
    change_peer_app_ll = scenario_probability(
        scenarios, "change_peer", "Neutral", "Appreciation", "low", "low"
    )
    change_peer_av_hl = scenario_probability(
        scenarios, "change_peer", "Neutral", "Aversion", "high", "low"
    )
    change_peer_av_ll = scenario_probability(
        scenarios, "change_peer", "Neutral", "Aversion", "low", "low"
    )

    change_target_app_hl = scenario_probability(
        scenarios, "change_target", "Neutral", "Appreciation", "high", "low"
    )
    change_target_app_hh = scenario_probability(
        scenarios, "change_target", "Neutral", "Appreciation", "high", "high"
    )
    change_target_av_hl = scenario_probability(
        scenarios, "change_target", "Neutral", "Aversion", "high", "low"
    )
    change_target_av_hh = scenario_probability(
        scenarios, "change_target", "Neutral", "Aversion", "high", "high"
    )

    manager_rows = len(manager_summary)
    mean_neutral_to_app = manager_summary["neutral_to_appreciation_rate"].mean(skipna=True)
    mean_neutral_to_av = manager_summary["neutral_to_aversion_rate"].mean(skipna=True)
    mean_app_persist = manager_summary["appreciation_persistence_rate"].mean(skipna=True)

    neutral_contrasts = contrasts[
        contrasts["from_state"].eq("Neutral")
        & contrasts["to_state"].isin(["Aversion", "Appreciation"])
    ].copy()
    neutral_contrasts["interaction_contrast_abs"] = neutral_contrasts[
        "interaction_contrast"
    ].abs()
    top_contrast = neutral_contrasts.sort_values(
        "interaction_contrast_abs", ascending=False
    ).iloc[0]

    lines = [
        "# Joint Transition-Driver Analysis",
        "",
        "## Interpretation",
        "",
        (
            "The fitted transition model supports the boundary-state interpretation of Neutral. "
            "The strongest positive moves out of Neutral require social benchmark support, and formal "
            "target attainment is especially important for avoiding a slide into Aversion."
        ),
        "",
        "## Peer Average x Target Attainment",
        "",
        (
            "When a Neutral manager is high on the peer benchmark, predicted movement to Appreciation is "
            f"{pct(peer_target_app_hh)} if the target is attained versus {pct(peer_target_app_hl)} "
            "if the target is not attained. Under the same high peer condition, predicted movement to "
            f"Aversion is {pct(peer_target_av_hh)} with target attainment versus {pct(peer_target_av_hl)} "
            "without target attainment. This is the cleanest threshold-crossing test: peer strength is "
            "the main route toward Appreciation, while target attainment mostly confirms the move and "
            "protects against Aversion."
        ),
        "",
        "## Team (t-1) vs. Team (t) x Peer Average",
        "",
        (
            "For a Neutral manager at the low peer benchmark, strong recent improvement raises predicted "
            f"Appreciation movement only from {pct(change_peer_app_ll)} to {pct(change_peer_app_hl)}, while "
            f"Aversion movement rises from {pct(change_peer_av_ll)} to {pct(change_peer_av_hl)}. Recent "
            "change therefore does not rescue a manager who is still weak relative to peers; the peer "
            "benchmark dominates the path out of Neutral."
        ),
        "",
        "## Team (t-1) vs. Team (t) x Target Attainment",
        "",
        (
            "With strong recent improvement, Neutral -> Appreciation is predicted at "
            f"{pct(change_target_app_hl)} without target attainment and {pct(change_target_app_hh)} "
            "with target attainment. Neutral -> Aversion is predicted at "
            f"{pct(change_target_av_hl)} without target attainment and {pct(change_target_av_hh)} "
            "with target attainment. This says improvement without absolute target success can still "
            "move managers toward Appreciation, but target failure leaves a much larger Aversion risk."
        ),
        "",
        "## Nonlinear joint effect",
        "",
        (
            "The largest Neutral-origin interaction contrast among the requested pairs is "
            f"{top_contrast['driver_pair']} for Neutral -> {top_contrast['to_state']}: "
            f"{top_contrast['interaction_contrast_pct_points']:.2f} percentage points. "
            "The contrast is computed as high-high minus high-low minus low-high plus low-low."
        ),
        "",
        "## Individual manager posterior/path outputs",
        "",
        (
            f"The manager-level files cover {manager_rows} managers. Across managers with observed Neutral "
            f"origin transitions, the mean decoded Neutral -> Appreciation rate is {pct(mean_neutral_to_app)} "
            f"and Neutral -> Aversion is {pct(mean_neutral_to_av)}. The mean Appreciation persistence rate "
            f"is {pct(mean_app_persist)}, reinforcing that reaching Appreciation from Neutral is harder than "
            "remaining there once the manager is already in that state."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts, model, panel, posteriors = load_inputs()

    surface = build_surface_frame(artifacts, model, panel)
    scenarios = build_scenario_summary(artifacts, model, panel)
    contrasts = build_interaction_contrasts(scenarios)

    posterior_matrix = build_manager_posterior_probability_matrix(posteriors)
    state_sequence_summary = build_manager_state_sequence_summary(posteriors)
    transition_matrices = build_manager_transition_matrices(posteriors)
    path_profiles = build_manager_joint_path_profiles(posteriors)
    transition_summary = build_manager_transition_summary(transition_matrices)
    manager_summary = state_sequence_summary.merge(transition_summary, on="manager_id", how="left")

    outputs = {
        "surface": ANALYSIS_DIR / "joint_transition_driver_surface_all_pairs_v3.csv",
        "scenarios": ANALYSIS_DIR / "joint_transition_driver_scenario_summary_v3.csv",
        "contrasts": ANALYSIS_DIR / "joint_transition_driver_interaction_contrasts_v3.csv",
        "posterior_matrix": ANALYSIS_DIR / "manager_posterior_probability_matrix_v3.csv",
        "state_sequence_summary": ANALYSIS_DIR / "manager_state_sequence_summary_v3.csv",
        "transition_matrices": ANALYSIS_DIR / "manager_posterior_transition_matrices_v3.csv",
        "path_profiles": ANALYSIS_DIR / "manager_joint_path_profiles_v3.csv",
        "manager_summary": ANALYSIS_DIR / "manager_transition_path_summary_v3.csv",
        "note": ANALYSIS_DIR / "joint_transition_driver_analysis_note_v3.md",
        "aversion_origin_figure": ANALYSIS_DIR
        / "figure_joint_transition_driver_surfaces_aversion_overlay_v3.png",
        "neutral_figure": ANALYSIS_DIR / "figure_joint_transition_driver_surfaces_neutral_overlay_v3.png",
        "appreciation_origin_figure": ANALYSIS_DIR
        / "figure_joint_transition_driver_surfaces_appreciation_overlay_v3.png",
        "appreciation_figure": ANALYSIS_DIR
        / "figure_joint_transition_driver_surfaces_appreciation_reach_compare_v3.png",
        "neutral_appreciation_closeup": ANALYSIS_DIR
        / "figure_joint_transition_driver_neutral_appreciation_closeup_table_v3.png",
    }

    surface.to_csv(outputs["surface"], index=False)
    scenarios.to_csv(outputs["scenarios"], index=False)
    contrasts.to_csv(outputs["contrasts"], index=False)
    posterior_matrix.to_csv(outputs["posterior_matrix"], index=False)
    state_sequence_summary.to_csv(outputs["state_sequence_summary"], index=False)
    transition_matrices.to_csv(outputs["transition_matrices"], index=False)
    path_profiles.to_csv(outputs["path_profiles"], index=False)
    manager_summary.to_csv(outputs["manager_summary"], index=False)
    outputs["note"].write_text(
        make_markdown_note(scenarios, contrasts, manager_summary),
        encoding="utf-8",
    )

    plot_origin_transition_overlay(surface, "Aversion", outputs["aversion_origin_figure"])
    plot_neutral_boundary_overlay(surface, outputs["neutral_figure"])
    plot_origin_transition_overlay(surface, "Appreciation", outputs["appreciation_origin_figure"])
    plot_appreciation_reach_overlay(surface, outputs["appreciation_figure"])
    plot_neutral_appreciation_closeup(
        surface,
        scenarios,
        outputs["neutral_appreciation_closeup"],
    )

    print("Saved joint transition-driver outputs:")
    for path in outputs.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
