from __future__ import annotations

import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from scipy.special import softmax


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts"
ANALYSIS_DIR = ARTIFACT_DIR / "Analysis_v3"
MODEL_PATH = ARTIFACT_DIR / "best_model_artifacts_v3_2emissions.pkl"
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"

STATE_LABELS = ["Aversion", "Neutral", "Appreciation"]
VARIABLE_LABELS = {
    "team_t_minus_1_vs_team_t": "Team (t-1) vs. Team (t)",
    "team_vs_peer_average": "Peer Average",
    "target_attainment": "Target Attainment",
}
STATE_COLORS = {
    "Aversion": "#111111",
    "Neutral": "#FF6A00",
    "Appreciation": "#0057D9",
}
STATE_CMAPS = {
    state: LinearSegmentedColormap.from_list(f"{state}_surface", ["#ffffff", color])
    for state, color in STATE_COLORS.items()
}
TARGET_BINARY_TICKS = ([0.0, 1.0], ["Not Attained", "Attained"])

FIGURE_SPECS = [
    {
        "key": "neutral_peer_target",
        "from_state": "Neutral",
        "x_var": "team_vs_peer_average",
        "y_var": "target_attainment",
        "destinations": ["Aversion", "Appreciation"],
        "title": "Neutral Boundary: Peer Average and Target Attainment",
    },
    {
        "key": "neutral_history_peer",
        "from_state": "Neutral",
        "x_var": "team_t_minus_1_vs_team_t",
        "y_var": "team_vs_peer_average",
        "destinations": ["Neutral", "Appreciation"],
        "title": "Neutral Boundary: Team (t-1) vs. Team (t) and Peer Average",
    },
    {
        "key": "appreciation_peer_target",
        "from_state": "Appreciation",
        "x_var": "team_vs_peer_average",
        "y_var": "target_attainment",
        "destinations": ["Neutral", "Appreciation"],
        "title": "Appreciation Stability: Peer Average and Target Attainment",
    },
    {
        "key": "appreciation_history_peer",
        "from_state": "Appreciation",
        "x_var": "team_t_minus_1_vs_team_t",
        "y_var": "team_vs_peer_average",
        "destinations": ["Neutral", "Appreciation"],
        "title": "Appreciation Stability: Team (t-1) vs. Team (t) and Peer Average",
    },
]


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


def load_artifacts() -> tuple[dict[str, object], Params, pd.DataFrame]:
    patch_pandas_stringarray_pickle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MODEL_PATH.open("rb") as stream:
            artifacts = pickle.load(stream)
    panel = pd.read_excel(DATA_PATH, sheet_name="panel_manager_period")
    return artifacts, artifacts["best_model"], panel


def get_ordered_raw_states(artifacts: dict[str, object], model: Params) -> list[int]:
    emission_cols = list(artifacts["emission_cols"])
    authority_idx = emission_cols.index("ai_authority_share")
    original_means = artifacts["y_scaler"].inverse_transform(model.mu)
    return np.argsort(original_means[:, authority_idx]).tolist()


def variable_grid(panel: pd.DataFrame, variable: str, n_points: int) -> np.ndarray:
    if variable == "target_attainment":
        return np.linspace(0.0, 1.0, n_points)
    return np.linspace(float(panel[variable].min()), float(panel[variable].max()), n_points)


def response_surface(
    artifacts: dict[str, object],
    model: Params,
    panel: pd.DataFrame,
    spec: dict[str, object],
    n_points: int = 61,
) -> pd.DataFrame:
    transition_cols = list(artifacts["transition_cols"])
    x_scaler = artifacts["x_scaler"]
    ordered_raw = get_ordered_raw_states(artifacts, model)
    from_state = str(spec["from_state"])
    raw_from = ordered_raw[STATE_LABELS.index(from_state)]
    x_var = str(spec["x_var"])
    y_var = str(spec["y_var"])
    x_idx = transition_cols.index(x_var)
    y_idx = transition_cols.index(y_var)

    x_values = variable_grid(panel, x_var, n_points)
    y_values = variable_grid(panel, y_var, n_points)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    predictors = np.tile(x_scaler.mean_, (x_grid.size, 1))
    predictors[:, x_idx] = x_grid.ravel()
    predictors[:, y_idx] = y_grid.ravel()
    predictors_scaled = x_scaler.transform(predictors)

    logits = model.alpha[raw_from][None, :] + predictors_scaled @ model.beta[raw_from].T
    probabilities = softmax(logits, axis=1)[:, ordered_raw]

    rows = []
    for destination_idx, destination in enumerate(STATE_LABELS):
        rows.append(
            pd.DataFrame(
                {
                    "joint_effect": str(spec["key"]),
                    "from_state": from_state,
                    "to_state": destination,
                    "x_variable": x_var,
                    "y_variable": y_var,
                    "x_value": x_grid.ravel(),
                    "y_value": y_grid.ravel(),
                    "transition_probability": probabilities[:, destination_idx],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_response_surface(surface: pd.DataFrame, spec: dict[str, object], output_path: Path) -> None:
    from_state = str(spec["from_state"])
    x_var = str(spec["x_var"])
    y_var = str(spec["y_var"])
    destinations = [str(destination) for destination in spec["destinations"]]
    held_constant = [
        VARIABLE_LABELS[col]
        for col in VARIABLE_LABELS
        if col not in {x_var, y_var}
    ][0]

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
    figure = plt.figure(figsize=(12.5, 5.7), dpi=220)

    for index, destination in enumerate(destinations, start=1):
        axis = figure.add_subplot(1, len(destinations), index, projection="3d")
        subset = surface[surface["to_state"].eq(destination)]
        x_values = np.sort(subset["x_value"].unique())
        y_values = np.sort(subset["y_value"].unique())
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        probability = (
            subset.pivot(index="y_value", columns="x_value", values="transition_probability")
            .reindex(index=y_values, columns=x_values)
            .to_numpy()
        )
        axis.plot_surface(
            x_grid,
            y_grid,
            100.0 * probability,
            cmap=STATE_CMAPS[destination],
            linewidth=0,
            antialiased=True,
            alpha=0.96,
        )
        axis.set_title(f"{from_state} to {destination}", fontsize=13, pad=12)
        axis.set_xlabel(VARIABLE_LABELS[x_var], labelpad=9)
        axis.set_ylabel(VARIABLE_LABELS[y_var], labelpad=9)
        axis.set_zlabel("Probability (%)", labelpad=8)
        axis.set_zlim(0, 100)
        axis.zaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        if y_var == "target_attainment":
            axis.set_yticks(*TARGET_BINARY_TICKS)
        if x_var == "target_attainment":
            axis.set_xticks(*TARGET_BINARY_TICKS)
        axis.view_init(elev=27, azim=-128)
        axis.grid(True, alpha=0.22)
        axis.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))

    figure.suptitle(str(spec["title"]), fontsize=15, y=0.98)
    binary_note = (
        " Target Attainment is binary; interpret endpoints only."
        if "target_attainment" in {x_var, y_var}
        else ""
    )
    figure.text(
        0.5,
        0.02,
        f"Predicted HMM transition probabilities; {held_constant} held at its sample mean.{binary_note}",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.025, right=0.98, bottom=0.14, top=0.87, wspace=0.04)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts, model, panel = load_artifacts()
    surface_outputs = []

    for spec in FIGURE_SPECS:
        key = str(spec["key"])
        surface = response_surface(artifacts, model, panel, spec)
        csv_path = ANALYSIS_DIR / f"joint_effect_response_surface_{key}_v3.csv"
        png_path = ANALYSIS_DIR / f"figure_joint_effect_response_surface_{key}_v3.png"
        surface.to_csv(csv_path, index=False)
        plot_response_surface(surface, spec, png_path)
        surface_outputs.append((csv_path, png_path))

    for csv_path, png_path in surface_outputs:
        print(f"Saved surface data to {csv_path}")
        print(f"Saved figure to {png_path}")


if __name__ == "__main__":
    main()
