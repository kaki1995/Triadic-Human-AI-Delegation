from __future__ import annotations

import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts"
ANALYSIS_DIR = ARTIFACT_DIR / "Analysis_v3"
MODEL_PATH = ARTIFACT_DIR / "best_model_artifacts_v3_2emissions.pkl"
FALLBACK_DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
DEFAULT_OUTPUT_STEM = "figure_hmm_aligned_interaction_effect_ai_authority_rate_v3"

STATE_LABELS = ["Aversion", "Neutral", "Appreciation"]
STATE_COLORS = {
    "Aversion": "#111111",
    "Neutral": "#FF6A00",
    "Appreciation": "#0057D9",
}
STATE_CMAPS = {
    state: LinearSegmentedColormap.from_list(f"{state}_surface", ["#FFFFFF", color])
    for state, color in STATE_COLORS.items()
}

VARIABLE_LABELS = {
    "team_t_minus_1_vs_team_t": "Team(t-1) vs. Team(t)",
    "team_vs_peer_average": "Team vs. Peer Average",
    "target_attainment": "Target Attainment",
}


@dataclass
class Params:
    logit_pi: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    W: np.ndarray
    log_sigma: np.ndarray


@dataclass(frozen=True)
class HMMAlignedConfig:
    outcome: str = "ai_authority_share"
    outcome_label: str = "AI Authority Rate"
    x_var: str = "team_t_minus_1_vs_team_t"
    y_var: str = "team_vs_peer_average"
    conditioning_var: str = "target_attainment"
    plotted_conditioning_level: str = "Medium"
    grid_size: int = 75
    surface_alpha: float = 0.92
    z_axis_mode: str = "surface_range"
    minimum_z_span: float = 0.16
    sheet_name: str = "panel_manager_period"
    output_stem: str = DEFAULT_OUTPUT_STEM


AI_AUTHORITY_CONFIG = HMMAlignedConfig()
ESCALATION_CONFIG = HMMAlignedConfig(
    outcome="escalation_share",
    outcome_label="Escalation Rate",
    output_stem="figure_hmm_aligned_interaction_effect_escalation_rate_v3",
)


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


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "savefig.facecolor": "white",
        }
    )


def load_artifacts() -> tuple[dict[str, object], Params]:
    patch_pandas_stringarray_pickle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MODEL_PATH.open("rb") as stream:
            artifacts = pickle.load(stream)
    return artifacts, artifacts["best_model"]


def model_data_path(artifacts: dict[str, object]) -> Path:
    path_value = artifacts.get("dataset_path")
    if path_value:
        path = Path(str(path_value))
        if path.exists():
            return path
    return FALLBACK_DATA_PATH


def load_panel(artifacts: dict[str, object], config: HMMAlignedConfig) -> pd.DataFrame:
    path = model_data_path(artifacts)
    columns = [config.x_var, config.y_var, config.conditioning_var]
    return pd.read_excel(
        path,
        sheet_name=config.sheet_name,
        usecols=lambda col: col in set(columns),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(centered)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def ordered_raw_states(artifacts: dict[str, object], model: Params) -> list[int]:
    emission_cols = list(artifacts["emission_cols"])
    authority_idx = emission_cols.index("ai_authority_share")
    original_means = artifacts["y_scaler"].inverse_transform(model.mu)
    return np.argsort(original_means[:, authority_idx]).tolist()


def emission_means(
    artifacts: dict[str, object],
    model: Params,
    outcome: str,
) -> dict[str, float]:
    emission_cols = list(artifacts["emission_cols"])
    outcome_idx = emission_cols.index(outcome)
    ordered_raw = ordered_raw_states(artifacts, model)
    original_means = artifacts["y_scaler"].inverse_transform(model.mu)
    return {
        state: float(original_means[ordered_raw[idx], outcome_idx])
        for idx, state in enumerate(STATE_LABELS)
    }


def observed_range(data: pd.DataFrame, variable: str) -> tuple[float, float]:
    values = pd.to_numeric(data[variable], errors="coerce").dropna()
    lower = float(values.min())
    upper = float(values.max())
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, 0.05)
        return lower - pad, upper + pad
    return lower, upper


def conditioning_levels(data: pd.DataFrame, variable: str) -> dict[str, float]:
    values = pd.to_numeric(data[variable], errors="coerce").dropna()
    quantiles = values.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
    if np.unique(np.round(quantiles, 12)).size < 3 and not np.isclose(values.min(), values.max()):
        quantiles = np.array([float(values.min()), float(values.mean()), float(values.max())])
    return {"Low": float(quantiles[0]), "Medium": float(quantiles[1]), "High": float(quantiles[2])}


def prediction_grid(
    data: pd.DataFrame,
    config: HMMAlignedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x_limits = observed_range(data, config.x_var)
    y_limits = observed_range(data, config.y_var)
    x_values = np.linspace(x_limits[0], x_limits[1], config.grid_size)
    y_values = np.linspace(y_limits[0], y_limits[1], config.grid_size)
    return np.meshgrid(x_values, y_values)


def transition_probabilities(
    artifacts: dict[str, object],
    model: Params,
    origin_state: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_value: float,
    config: HMMAlignedConfig,
) -> np.ndarray:
    transition_cols = list(artifacts["transition_cols"])
    x_scaler = artifacts["x_scaler"]
    ordered_raw = ordered_raw_states(artifacts, model)
    raw_origin = ordered_raw[STATE_LABELS.index(origin_state)]

    predictors = np.tile(x_scaler.mean_, (x_grid.size, 1))
    predictors[:, transition_cols.index(config.x_var)] = x_grid.ravel()
    predictors[:, transition_cols.index(config.y_var)] = y_grid.ravel()
    predictors[:, transition_cols.index(config.conditioning_var)] = c_value
    predictors_scaled = x_scaler.transform(predictors)

    logits = model.alpha[raw_origin][None, :] + predictors_scaled @ model.beta[raw_origin].T
    return softmax(logits)[:, ordered_raw]


def expected_outcome_surface(
    probabilities: np.ndarray,
    means: dict[str, float],
    x_grid: np.ndarray,
) -> np.ndarray:
    mean_vector = np.array([means[state] for state in STATE_LABELS], dtype=float)
    expected = probabilities @ mean_vector
    return np.clip(expected.reshape(x_grid.shape), 0.0, 1.0)


def build_model_aligned_surfaces(
    artifacts: dict[str, object],
    model: Params,
    data: pd.DataFrame,
    config: HMMAlignedConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    c_levels = conditioning_levels(data, config.conditioning_var)
    c_value = c_levels[config.plotted_conditioning_level]
    x_grid, y_grid = prediction_grid(data, config)
    means = emission_means(artifacts, model, config.outcome)

    surfaces: dict[str, np.ndarray] = {}
    transition_by_origin: dict[str, np.ndarray] = {}
    for origin_state in STATE_LABELS:
        probabilities = transition_probabilities(
            artifacts,
            model,
            origin_state,
            x_grid,
            y_grid,
            c_value,
            config,
        )
        transition_by_origin[origin_state] = probabilities
        surfaces[origin_state] = expected_outcome_surface(probabilities, means, x_grid)
    return x_grid, y_grid, surfaces, transition_by_origin, c_levels


def surface_norms(surfaces: dict[str, np.ndarray]) -> dict[str, Normalize]:
    norms = {}
    for state, surface in surfaces.items():
        lower = float(np.nanmin(surface))
        upper = float(np.nanmax(surface))
        if np.isclose(lower, upper):
            upper = lower + 0.01
        norms[state] = Normalize(vmin=lower, vmax=upper)
    return norms


def surface_z_limits(surfaces: dict[str, np.ndarray], config: HMMAlignedConfig) -> tuple[float, float]:
    if config.z_axis_mode == "full":
        return 0.0, 1.0
    if config.z_axis_mode != "surface_range":
        raise ValueError("z_axis_mode must be 'surface_range' or 'full'.")

    all_values = np.concatenate([surface.ravel() for surface in surfaces.values()])
    lower = float(np.nanmin(all_values))
    upper = float(np.nanmax(all_values))
    span = upper - lower
    padded_span = max(span * 1.35, config.minimum_z_span)
    midpoint = (lower + upper) / 2.0
    z_lower = max(0.0, midpoint - padded_span / 2.0)
    z_upper = min(1.0, midpoint + padded_span / 2.0)
    if z_upper - z_lower < config.minimum_z_span:
        if z_lower == 0.0:
            z_upper = min(1.0, config.minimum_z_span)
        elif z_upper == 1.0:
            z_lower = max(0.0, 1.0 - config.minimum_z_span)
    return z_lower, z_upper


def style_3d_axis(axis: plt.Axes, config: HMMAlignedConfig, z_limits: tuple[float, float]) -> None:
    axis.set_xlabel(VARIABLE_LABELS[config.x_var], labelpad=10)
    axis.set_ylabel(VARIABLE_LABELS[config.y_var], labelpad=10)
    axis.set_zlabel(f"Predicted {config.outcome_label}", labelpad=12)
    axis.set_zlim(*z_limits)
    axis.zaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.view_init(elev=24, azim=-124)
    axis.set_box_aspect((1.35, 1.0, 0.96))
    axis.grid(True, alpha=0.28)
    axis.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    axis.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    axis.zaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))


def plot_surfaces(
    axis: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces: dict[str, np.ndarray],
    norms: dict[str, Normalize],
    config: HMMAlignedConfig,
) -> None:
    for origin_state in STATE_LABELS:
        axis.plot_surface(
            x_grid,
            y_grid,
            surfaces[origin_state],
            cmap=STATE_CMAPS[origin_state],
            norm=norms[origin_state],
            rstride=2,
            cstride=2,
            linewidth=0.10,
            edgecolor=(0.0, 0.0, 0.0, 0.08),
            antialiased=True,
            shade=True,
            alpha=config.surface_alpha,
        )
    style_3d_axis(axis, config, surface_z_limits(surfaces, config))


def add_title_bar(figure: plt.Figure, config: HMMAlignedConfig) -> None:
    title = (
        "Fitted HMM Interaction Effects of Team(t-1) vs. Team(t), "
        f"Team vs. Peer Average, and Target Attainment on {config.outcome_label}"
    )
    figure.patches.append(
        Rectangle(
            (0.025, 0.925),
            0.95,
            0.055,
            transform=figure.transFigure,
            facecolor="#000000",
            edgecolor="#000000",
            linewidth=0,
            zorder=10,
        )
    )
    figure.text(
        0.04,
        0.952,
        title,
        ha="left",
        va="center",
        color="white",
        fontsize=16,
        weight="bold",
        zorder=11,
    )


def add_colorbars(
    figure: plt.Figure,
    colorbar_axes: list[plt.Axes],
    norms: dict[str, Normalize],
    config: HMMAlignedConfig,
) -> None:
    for axis, state in zip(colorbar_axes, STATE_LABELS):
        mappable = cm.ScalarMappable(norm=norms[state], cmap=STATE_CMAPS[state])
        colorbar = figure.colorbar(mappable, cax=axis, orientation="vertical")
        colorbar.ax.set_title(f"Previous\n{state}", fontsize=10, pad=8, weight="bold")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        colorbar.set_label(f"Predicted {config.outcome_label}", fontsize=8, labelpad=8)
        colorbar.outline.set_edgecolor("#555555")
        colorbar.outline.set_linewidth(0.8)


def scenario_values(data: pd.DataFrame, config: HMMAlignedConfig) -> dict[str, tuple[float, float]]:
    x = pd.to_numeric(data[config.x_var], errors="coerce").dropna()
    y = pd.to_numeric(data[config.y_var], errors="coerce").dropna()
    return {
        "Low X / Low Y": (float(x.quantile(0.10)), float(y.quantile(0.10))),
        "High X / Low Y": (float(x.quantile(0.90)), float(y.quantile(0.10))),
        "Low X / High Y": (float(x.quantile(0.10)), float(y.quantile(0.90))),
        "High X / High Y": (float(x.quantile(0.90)), float(y.quantile(0.90))),
    }


def predict_at_point(
    artifacts: dict[str, object],
    model: Params,
    origin_state: str,
    x_value: float,
    y_value: float,
    c_value: float,
    means: dict[str, float],
    config: HMMAlignedConfig,
) -> tuple[float, dict[str, float]]:
    x_grid = np.array([[x_value]], dtype=float)
    y_grid = np.array([[y_value]], dtype=float)
    probabilities = transition_probabilities(
        artifacts,
        model,
        origin_state,
        x_grid,
        y_grid,
        c_value,
        config,
    )[0]
    expected = float(probabilities @ np.array([means[state] for state in STATE_LABELS]))
    return expected, {state: float(probabilities[idx]) for idx, state in enumerate(STATE_LABELS)}


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_num(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.3e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def build_summary_table(
    artifacts: dict[str, object],
    model: Params,
    data: pd.DataFrame,
    surfaces: dict[str, np.ndarray],
    config: HMMAlignedConfig,
    c_levels: dict[str, float],
) -> tuple[list[str], list[list[str]]]:
    means = emission_means(artifacts, model, config.outcome)
    c_value = c_levels[config.plotted_conditioning_level]
    scenarios = scenario_values(data, config)

    rows: list[list[str]] = []
    for scenario_name, (x_value, y_value) in scenarios.items():
        row = [f"Predicted {config.outcome_label}: {scenario_name}"]
        for origin_state in STATE_LABELS:
            expected, _ = predict_at_point(
                artifacts,
                model,
                origin_state,
                x_value,
                y_value,
                c_value,
                means,
                config,
            )
            row.append(fmt_pct(expected))
        rows.append(row)

    for destination_state in STATE_LABELS:
        rows.append(
            [
                f"Emission mean: State_t = {destination_state}",
                *[fmt_pct(means[destination_state]) for _ in STATE_LABELS],
            ]
        )

    rows.append(["Surface minimum", *[fmt_pct(float(surfaces[state].min())) for state in STATE_LABELS]])
    rows.append(["Surface maximum", *[fmt_pct(float(surfaces[state].max())) for state in STATE_LABELS]])
    rows.append(["HMM log likelihood", *[fmt_num(float(artifacts["ll_total"])) for _ in STATE_LABELS]])
    rows.append(["AIC", *[fmt_num(float(artifacts["aic"])) for _ in STATE_LABELS]])
    rows.append(["BIC", *[fmt_num(float(artifacts["bic"])) for _ in STATE_LABELS]])

    columns = ["", *[f"Previous {state}" for state in STATE_LABELS]]
    return columns, rows


def add_summary_table(
    axis: plt.Axes,
    artifacts: dict[str, object],
    model: Params,
    data: pd.DataFrame,
    surfaces: dict[str, np.ndarray],
    config: HMMAlignedConfig,
    c_levels: dict[str, float],
) -> None:
    axis.axis("off")
    columns, rows = build_summary_table(artifacts, model, data, surfaces, config, c_levels)
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.38, 0.206, 0.206, 0.206],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1.0, 1.20)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#111111")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F2F2F2")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")

    note = (
        f"Target Attainment held at {config.plotted_conditioning_level} = "
        f"{c_levels[config.plotted_conditioning_level]:.3f}. "
        f"Each layer is the fitted expected {config.outcome_label} at time t for a previous latent state: "
        "transition probabilities into the three states are multiplied by the fitted state emission means. "
        "The z-axis is zoomed to the fitted surface range to make model-implied variation visible."
    )
    axis.text(
        0.5,
        1.02,
        note,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#333333",
    )


def export_surface_predictions(
    path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces: dict[str, np.ndarray],
    transition_by_origin: dict[str, np.ndarray],
    c_levels: dict[str, float],
    config: HMMAlignedConfig,
) -> None:
    rows = []
    c_value = c_levels[config.plotted_conditioning_level]
    for origin_state in STATE_LABELS:
        frame = pd.DataFrame(
            {
                "previous_state": origin_state,
                config.conditioning_var: c_value,
                config.x_var: x_grid.ravel(),
                config.y_var: y_grid.ravel(),
                f"predicted_{config.outcome}": surfaces[origin_state].ravel(),
            }
        )
        probabilities = transition_by_origin[origin_state]
        for idx, destination_state in enumerate(STATE_LABELS):
            frame[f"p_state_t_{destination_state.lower()}"] = probabilities[:, idx]
        rows.append(frame)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def export_scenario_summary(
    path: Path,
    artifacts: dict[str, object],
    model: Params,
    data: pd.DataFrame,
    config: HMMAlignedConfig,
    c_levels: dict[str, float],
) -> None:
    means = emission_means(artifacts, model, config.outcome)
    c_value = c_levels[config.plotted_conditioning_level]
    rows = []
    for scenario_name, (x_value, y_value) in scenario_values(data, config).items():
        for origin_state in STATE_LABELS:
            expected, probabilities = predict_at_point(
                artifacts,
                model,
                origin_state,
                x_value,
                y_value,
                c_value,
                means,
                config,
            )
            row = {
                "scenario": scenario_name,
                "previous_state": origin_state,
                config.x_var: x_value,
                config.y_var: y_value,
                config.conditioning_var: c_value,
                f"predicted_{config.outcome}": expected,
            }
            for destination_state, probability in probabilities.items():
                row[f"p_state_t_{destination_state.lower()}"] = probability
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def create_model_aligned_figure(config: HMMAlignedConfig = HMMAlignedConfig()) -> list[Path]:
    setup_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts, model = load_artifacts()
    data = load_panel(artifacts, config)
    x_grid, y_grid, surfaces, transition_by_origin, c_levels = build_model_aligned_surfaces(
        artifacts,
        model,
        data,
        config,
    )
    norms = surface_norms(surfaces)

    figure = plt.figure(figsize=(18.5, 10.8), dpi=180)
    add_title_bar(figure, config)
    surface_axis = figure.add_axes([0.055, 0.315, 0.535, 0.565], projection="3d")
    plot_surfaces(surface_axis, x_grid, y_grid, surfaces, norms, config)
    colorbar_axes = [
        figure.add_axes([0.625, 0.385, 0.070, 0.410]),
        figure.add_axes([0.755, 0.385, 0.070, 0.410]),
        figure.add_axes([0.885, 0.385, 0.070, 0.410]),
    ]
    add_colorbars(figure, colorbar_axes, norms, config)

    table_axis = figure.add_axes([0.045, 0.045, 0.910, 0.245])
    add_summary_table(table_axis, artifacts, model, data, surfaces, config, c_levels)

    png_path = ANALYSIS_DIR / f"{config.output_stem}.png"
    pdf_path = ANALYSIS_DIR / f"{config.output_stem}.pdf"
    surface_csv = ANALYSIS_DIR / f"{config.output_stem}_surface_predictions.csv"
    scenario_csv = ANALYSIS_DIR / f"{config.output_stem}_scenario_summary.csv"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    export_surface_predictions(surface_csv, x_grid, y_grid, surfaces, transition_by_origin, c_levels, config)
    export_scenario_summary(scenario_csv, artifacts, model, data, config, c_levels)
    return [png_path, pdf_path, surface_csv, scenario_csv]


def main() -> None:
    paths = []
    for config in [AI_AUTHORITY_CONFIG, ESCALATION_CONFIG]:
        paths.extend(create_model_aligned_figure(config))
    print("Saved fitted-HMM aligned interaction-effect outputs:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
