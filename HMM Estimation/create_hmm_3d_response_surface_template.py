from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3" / "hmm_3d_response_surfaces"

STATE_ORDER = ["Appreciation", "Neutral", "Aversion"]
STATE_COLORS = {
    "Appreciation": "#0057D9",
    "Neutral": "#FF6A00",
    "Aversion": "#111111",
}
STATE_CMAPS = {
    state: LinearSegmentedColormap.from_list(f"{state}_surface", ["#FFFFFF", color])
    for state, color in STATE_COLORS.items()
}


@dataclass(frozen=True)
class VariableSpec:
    name: str
    label: str
    low: float
    high: float
    fixed: float
    binary: bool = False


@dataclass(frozen=True)
class PredictorPair:
    x_var: str
    y_var: str
    title: str


PredictFn = Callable[[pd.DataFrame], pd.DataFrame | np.ndarray]


VARIABLES = {
    "team_t_minus_1": VariableSpec(
        name="team_t_minus_1",
        label="Team (t-1)",
        low=-1.0,
        high=1.0,
        fixed=0.0,
    ),
    "team_t": VariableSpec(
        name="team_t",
        label="Team (t)",
        low=-1.0,
        high=1.0,
        fixed=0.0,
    ),
    "peer_average": VariableSpec(
        name="peer_average",
        label="Peer Average",
        low=-1.0,
        high=1.0,
        fixed=0.0,
    ),
    "target_attainment": VariableSpec(
        name="target_attainment",
        label="Target Attainment",
        low=0.0,
        high=1.0,
        fixed=0.5,
        binary=True,
    ),
}

PREDICTOR_PAIRS = [
    PredictorPair(
        x_var="team_t_minus_1",
        y_var="team_t",
        title="Team (t-1) x Team (t)",
    ),
    PredictorPair(
        x_var="peer_average",
        y_var="target_attainment",
        title="Peer Average x Target Attainment",
    ),
    PredictorPair(
        x_var="team_t_minus_1",
        y_var="peer_average",
        title="Team (t-1) x Peer Average",
    ),
    PredictorPair(
        x_var="team_t",
        y_var="target_attainment",
        title="Team (t) x Target Attainment",
    ),
]


def setup_academic_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "savefig.facecolor": "white",
        }
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(centered)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def simulated_hmm_predict_fn(grid: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for P(State_t | transition drivers).

    Replace this function with your estimated HMM, multinomial logit, or posterior
    prediction routine. The replacement should accept a DataFrame containing the
    four driver columns and return probabilities for Appreciation, Neutral, and
    Aversion in STATE_ORDER.
    """

    team_lag = grid["team_t_minus_1"].to_numpy(dtype=float)
    team_now = grid["team_t"].to_numpy(dtype=float)
    peer_average = grid["peer_average"].to_numpy(dtype=float)
    target = grid["target_attainment"].to_numpy(dtype=float)

    change = team_now - team_lag
    peer_gap = team_now - peer_average

    appreciation_logit = (
        -0.15
        + 1.25 * team_now
        + 1.00 * change
        + 1.05 * peer_gap
        + 0.95 * target
        + 0.85 * change * target
        + 0.70 * peer_gap * target
        - 0.45 * (team_now - 0.35) ** 2
    )
    neutral_logit = (
        1.05
        - 0.90 * np.abs(team_now)
        - 0.75 * np.abs(change)
        - 0.65 * peer_gap**2
        + 0.35 * target
    )
    aversion_logit = (
        0.10
        - 1.25 * team_now
        - 1.05 * change
        - 1.20 * peer_gap
        - 0.80 * target
        + 0.65 * (peer_average - team_now) * (1.0 - target)
        + 0.55 * np.maximum(-change, 0.0) * (1.0 - target)
    )

    probabilities = softmax(
        np.column_stack([appreciation_logit, neutral_logit, aversion_logit])
    )
    return pd.DataFrame(probabilities, columns=STATE_ORDER, index=grid.index)


def axis_values(
    spec: VariableSpec,
    grid_points: int,
    binary_as_endpoints: bool,
) -> np.ndarray:
    if spec.binary and binary_as_endpoints:
        return np.array([spec.low, spec.high], dtype=float)
    return np.linspace(spec.low, spec.high, grid_points)


def build_prediction_grid(
    pair: PredictorPair,
    variables: Mapping[str, VariableSpec],
    fixed_values: Mapping[str, float] | None = None,
    grid_points: int = 70,
    binary_as_endpoints: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    fixed_values = fixed_values or {}
    x_spec = variables[pair.x_var]
    y_spec = variables[pair.y_var]

    x_values = axis_values(x_spec, grid_points, binary_as_endpoints)
    y_values = axis_values(y_spec, grid_points, binary_as_endpoints)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    grid = {
        name: np.full(x_grid.size, fixed_values.get(name, spec.fixed), dtype=float)
        for name, spec in variables.items()
    }
    grid[pair.x_var] = x_grid.ravel()
    grid[pair.y_var] = y_grid.ravel()
    return pd.DataFrame(grid), x_grid, y_grid


def predict_state_probabilities(
    grid: pd.DataFrame,
    predict_fn: PredictFn,
    states: Sequence[str] = STATE_ORDER,
) -> np.ndarray:
    raw = predict_fn(grid.copy())
    if isinstance(raw, pd.DataFrame):
        missing = [state for state in states if state not in raw.columns]
        if missing:
            raise ValueError(f"Prediction DataFrame is missing columns: {missing}")
        probabilities = raw.loc[:, list(states)].to_numpy(dtype=float)
    else:
        probabilities = np.asarray(raw, dtype=float)

    if probabilities.shape != (len(grid), len(states)):
        raise ValueError(
            "Prediction function must return an array/DataFrame with shape "
            f"({len(grid)}, {len(states)}). Got {probabilities.shape}."
        )

    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Predicted probabilities must have positive row sums.")
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        probabilities = probabilities / row_sums

    return probabilities


def style_3d_axis(
    axis: plt.Axes,
    pair: PredictorPair,
    variables: Mapping[str, VariableSpec],
) -> None:
    x_spec = variables[pair.x_var]
    y_spec = variables[pair.y_var]
    axis.set_xlabel(x_spec.label, labelpad=8)
    axis.set_ylabel(y_spec.label, labelpad=8)
    axis.set_zlabel("Predicted probability", labelpad=8)
    axis.set_zlim(0.0, 1.0)
    axis.zaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    if x_spec.binary:
        axis.set_xticks([x_spec.low, x_spec.high], ["No", "Yes"])
    if y_spec.binary:
        axis.set_yticks([y_spec.low, y_spec.high], ["No", "Yes"])

    axis.view_init(elev=27, azim=-128)
    axis.grid(True, alpha=0.28)
    axis.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    axis.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    axis.zaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))


def add_surface(
    axis: plt.Axes,
    pair: PredictorPair,
    state: str,
    predict_fn: PredictFn,
    variables: Mapping[str, VariableSpec],
    fixed_values: Mapping[str, float] | None,
    grid_points: int,
    binary_as_endpoints: bool,
) -> None:
    grid, x_grid, y_grid = build_prediction_grid(
        pair=pair,
        variables=variables,
        fixed_values=fixed_values,
        grid_points=grid_points,
        binary_as_endpoints=binary_as_endpoints,
    )
    probabilities = predict_state_probabilities(grid, predict_fn)
    state_index = STATE_ORDER.index(state)
    z_grid = probabilities[:, state_index].reshape(x_grid.shape)

    axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap=STATE_CMAPS[state],
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.96,
    )
    style_3d_axis(axis, pair, variables)


def slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("(", "").replace(")", "")


def plot_one_state_figure(
    state: str,
    predict_fn: PredictFn,
    output_dir: Path,
    variables: Mapping[str, VariableSpec] = VARIABLES,
    predictor_pairs: Sequence[PredictorPair] = PREDICTOR_PAIRS,
    fixed_values: Mapping[str, float] | None = None,
    grid_points: int = 70,
    binary_as_endpoints: bool = True,
) -> Path:
    setup_academic_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(13.5, 9.2), dpi=180)
    for index, pair in enumerate(predictor_pairs, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        add_surface(
            axis=axis,
            pair=pair,
            state=state,
            predict_fn=predict_fn,
            variables=variables,
            fixed_values=fixed_values,
            grid_points=grid_points,
            binary_as_endpoints=binary_as_endpoints,
        )
        axis.set_title(pair.title, fontsize=12, pad=10)

    figure.suptitle(
        f"HMM Response Surfaces: P(State_t = {state})",
        fontsize=16,
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        (
            "Non-plotted drivers are held at fixed values. "
            "For binary Target Attainment, the connecting surface is a visual guide."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.09, top=0.90, wspace=0.02, hspace=0.10)

    output_path = output_dir / f"figure_hmm_response_surface_{slug(state)}.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def plot_all_states_figure(
    predict_fn: PredictFn,
    output_dir: Path,
    variables: Mapping[str, VariableSpec] = VARIABLES,
    predictor_pairs: Sequence[PredictorPair] = PREDICTOR_PAIRS,
    fixed_values: Mapping[str, float] | None = None,
    grid_points: int = 70,
    binary_as_endpoints: bool = True,
) -> Path:
    setup_academic_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(20.0, 12.0), dpi=180)
    for row_index, state in enumerate(STATE_ORDER):
        for col_index, pair in enumerate(predictor_pairs):
            axis_index = row_index * len(predictor_pairs) + col_index + 1
            axis = figure.add_subplot(len(STATE_ORDER), len(predictor_pairs), axis_index, projection="3d")
            add_surface(
                axis=axis,
                pair=pair,
                state=state,
                predict_fn=predict_fn,
                variables=variables,
                fixed_values=fixed_values,
                grid_points=grid_points,
                binary_as_endpoints=binary_as_endpoints,
            )
            if row_index == 0:
                axis.set_title(pair.title, fontsize=11, pad=10)
            if col_index == 0:
                axis.text2D(
                    -0.20,
                    0.55,
                    f"P(State_t = {state})",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    weight="bold",
                    color=STATE_COLORS[state],
                )

    figure.suptitle(
        "Joint Transition Drivers and Predicted Latent-State Membership",
        fontsize=17,
        y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        (
            "Rows show focal latent states; columns show predictor combinations. "
            "Non-plotted drivers are held fixed."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.055, top=0.93, wspace=0.01, hspace=0.08)

    output_path = output_dir / "figure_hmm_response_surfaces_all_states.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def generate_hmm_response_surface_figures(
    predict_fn: PredictFn = simulated_hmm_predict_fn,
    output_dir: Path = OUTPUT_DIR,
    variables: Mapping[str, VariableSpec] = VARIABLES,
    predictor_pairs: Sequence[PredictorPair] = PREDICTOR_PAIRS,
    fixed_values: Mapping[str, float] | None = None,
    grid_points: int = 70,
    binary_as_endpoints: bool = True,
    include_combined_figure: bool = True,
) -> list[Path]:
    """Generate journal-style 3D response surface figures.

    To use your estimated model, pass a replacement predict_fn:

        paths = generate_hmm_response_surface_figures(
            predict_fn=my_estimated_hmm_predict_fn,
            fixed_values={
                "team_t_minus_1": panel["team_t_minus_1"].mean(),
                "team_t": panel["team_t"].mean(),
                "peer_average": panel["peer_average"].mean(),
                "target_attainment": panel["target_attainment"].mean(),
            },
        )

    The predict_fn must accept a DataFrame with the four transition-driver columns
    and return either a DataFrame with STATE_ORDER columns or an ndarray with
    columns in STATE_ORDER.
    """

    paths = [
        plot_one_state_figure(
            state=state,
            predict_fn=predict_fn,
            output_dir=output_dir,
            variables=variables,
            predictor_pairs=predictor_pairs,
            fixed_values=fixed_values,
            grid_points=grid_points,
            binary_as_endpoints=binary_as_endpoints,
        )
        for state in STATE_ORDER
    ]
    if include_combined_figure:
        paths.append(
            plot_all_states_figure(
                predict_fn=predict_fn,
                output_dir=output_dir,
                variables=variables,
                predictor_pairs=predictor_pairs,
                fixed_values=fixed_values,
                grid_points=grid_points,
                binary_as_endpoints=binary_as_endpoints,
            )
        )
    return paths


def main() -> None:
    paths = generate_hmm_response_surface_figures()
    print("Saved HMM 3D response surface figures:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
