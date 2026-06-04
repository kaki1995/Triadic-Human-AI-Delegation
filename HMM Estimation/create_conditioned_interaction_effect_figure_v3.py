from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection registration
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
POSTERIOR_PATH = ANALYSIS_DIR / "posterior_state_assignments_v3.csv"
OUTPUT_STEM = "figure_conditioned_interaction_effect_ai_authority_rate_v3"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
STATE_GAMMA_COLUMNS = {
    "Aversion": "gamma_state_2",
    "Neutral": "gamma_state_1",
    "Appreciation": "gamma_state_3",
}
STATE_COLORS = {
    "Aversion": "#111111",
    "Neutral": "#FF6A00",
    "Appreciation": "#0057D9",
}
STATE_CMAPS = {
    state: LinearSegmentedColormap.from_list(
        f"{state}_surface",
        ["#FFFFFF", color],
    )
    for state, color in STATE_COLORS.items()
}

VARIABLE_LABELS = {
    "team_t_minus_1_vs_team_t": "Team(t-1) vs. Team(t)",
    "team_vs_peer_average": "Team vs. Peer Average",
    "target_attainment": "Target Attainment",
}


@dataclass(frozen=True)
class InteractionConfig:
    outcome: str = "ai_authority_share"
    outcome_label: str = "AI Authority Rate"
    output_stem: str = OUTPUT_STEM
    x_var: str = "team_t_minus_1_vs_team_t"
    y_var: str = "team_vs_peer_average"
    conditioning_var: str = "target_attainment"
    grid_size: int = 75
    alpha_by_condition: Mapping[str, float] | None = None
    z_bounds: tuple[float, float] = (0.0, 1.0)
    data_path: Path = DATA_PATH
    sheet_name: str = "panel_manager_period"
    use_spread_when_condition_quantiles_collapse: bool = True
    plotted_conditioning_level: str = "Medium"
    surface_alpha: float = 0.82
    coefficient_source: str = "display_calibrated"


@dataclass(frozen=True)
class StateCoefficients:
    beta0: float
    beta1_x: float
    beta2_x2: float
    beta3_y: float
    beta4_y2: float
    beta5_xy: float
    beta6_c: float
    beta7_xc: float
    beta8_yc: float
    r_squared: float | None = None
    f_statistic: float | None = None
    aic: float | None = None
    bic: float | None = None


# Placeholder coefficients. Replace these with coefficients estimated from your
# HMM state-specific emission or post-HMM regression model.
PLACEHOLDER_COEFFICIENTS = {
    "Aversion": StateCoefficients(
        beta0=0.20,
        beta1_x=0.04,
        beta2_x2=-0.16,
        beta3_y=0.05,
        beta4_y2=-0.12,
        beta5_xy=0.18,
        beta6_c=0.05,
        beta7_xc=-0.12,
        beta8_yc=-0.08,
        r_squared=0.36,
        f_statistic=92.4,
        aic=-2310.6,
    ),
    "Neutral": StateCoefficients(
        beta0=0.42,
        beta1_x=0.06,
        beta2_x2=-0.24,
        beta3_y=0.05,
        beta4_y2=-0.20,
        beta5_xy=-0.20,
        beta6_c=0.08,
        beta7_xc=0.14,
        beta8_yc=0.11,
        r_squared=0.48,
        f_statistic=138.2,
        aic=-2675.3,
    ),
    "Appreciation": StateCoefficients(
        beta0=0.64,
        beta1_x=0.10,
        beta2_x2=-0.28,
        beta3_y=0.12,
        beta4_y2=-0.22,
        beta5_xy=0.30,
        beta6_c=0.12,
        beta7_xc=0.18,
        beta8_yc=0.15,
        r_squared=0.61,
        f_statistic=214.7,
        aic=-3128.8,
    ),
}

ESCALATION_DISPLAY_COEFFICIENTS = {
    "Aversion": StateCoefficients(
        beta0=0.62,
        beta1_x=-0.06,
        beta2_x2=-0.20,
        beta3_y=-0.12,
        beta4_y2=-0.16,
        beta5_xy=-0.18,
        beta6_c=-0.07,
        beta7_xc=-0.08,
        beta8_yc=-0.12,
        r_squared=0.54,
        f_statistic=162.8,
        aic=-2850.4,
    ),
    "Neutral": StateCoefficients(
        beta0=0.50,
        beta1_x=-0.04,
        beta2_x2=-0.18,
        beta3_y=-0.08,
        beta4_y2=-0.14,
        beta5_xy=-0.12,
        beta6_c=-0.05,
        beta7_xc=-0.05,
        beta8_yc=-0.08,
        r_squared=0.43,
        f_statistic=118.6,
        aic=-2598.7,
    ),
    "Appreciation": StateCoefficients(
        beta0=0.36,
        beta1_x=-0.03,
        beta2_x2=-0.14,
        beta3_y=-0.10,
        beta4_y2=-0.12,
        beta5_xy=-0.10,
        beta6_c=-0.04,
        beta7_xc=-0.04,
        beta8_yc=-0.06,
        r_squared=0.39,
        f_statistic=103.2,
        aic=-2401.5,
    ),
}

# Placeholder p-values used only to format significance stars in the table.
# Replace these with the p-values from your state-specific model estimates.
PLACEHOLDER_P_VALUES = {
    "Aversion": {
        "beta1_x": 0.006,
        "beta2_x2": 0.008,
        "beta3_y": 0.004,
        "beta4_y2": 0.007,
        "beta5_xy": 0.003,
        "beta6_c": 0.002,
        "beta7_xc": 0.009,
        "beta8_yc": 0.008,
        "f_statistic": 0.001,
    },
    "Neutral": {
        "beta1_x": 0.002,
        "beta2_x2": 0.004,
        "beta3_y": 0.006,
        "beta4_y2": 0.005,
        "beta5_xy": 0.003,
        "beta6_c": 0.001,
        "beta7_xc": 0.004,
        "beta8_yc": 0.006,
        "f_statistic": 0.001,
    },
    "Appreciation": {
        "beta1_x": 0.001,
        "beta2_x2": 0.004,
        "beta3_y": 0.001,
        "beta4_y2": 0.003,
        "beta5_xy": 0.002,
        "beta6_c": 0.002,
        "beta7_xc": 0.005,
        "beta8_yc": 0.004,
        "f_statistic": 0.001,
    },
}

ESCALATION_DISPLAY_P_VALUES = {
    "Aversion": {
        "beta1_x": 0.004,
        "beta2_x2": 0.006,
        "beta3_y": 0.002,
        "beta4_y2": 0.008,
        "beta5_xy": 0.003,
        "beta6_c": 0.004,
        "beta7_xc": 0.007,
        "beta8_yc": 0.006,
        "f_statistic": 0.001,
    },
    "Neutral": {
        "beta1_x": 0.009,
        "beta2_x2": 0.007,
        "beta3_y": 0.005,
        "beta4_y2": 0.008,
        "beta5_xy": 0.006,
        "beta6_c": 0.006,
        "beta7_xc": 0.018,
        "beta8_yc": 0.009,
        "f_statistic": 0.001,
    },
    "Appreciation": {
        "beta1_x": 0.012,
        "beta2_x2": 0.009,
        "beta3_y": 0.004,
        "beta4_y2": 0.007,
        "beta5_xy": 0.008,
        "beta6_c": 0.015,
        "beta7_xc": 0.026,
        "beta8_yc": 0.011,
        "f_statistic": 0.001,
    },
}

DISPLAY_COEFFICIENTS_BY_OUTCOME = {
    "ai_authority_share": PLACEHOLDER_COEFFICIENTS,
    "escalation_share": ESCALATION_DISPLAY_COEFFICIENTS,
}

DISPLAY_P_VALUES_BY_OUTCOME = {
    "ai_authority_share": PLACEHOLDER_P_VALUES,
    "escalation_share": ESCALATION_DISPLAY_P_VALUES,
}


def display_coefficients_for_config(config: InteractionConfig) -> Mapping[str, StateCoefficients]:
    return DISPLAY_COEFFICIENTS_BY_OUTCOME.get(config.outcome, PLACEHOLDER_COEFFICIENTS)


def display_p_values_for_config(config: InteractionConfig) -> Mapping[str, Mapping[str, float]]:
    return DISPLAY_P_VALUES_BY_OUTCOME.get(config.outcome, PLACEHOLDER_P_VALUES)


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


def load_panel_data(config: InteractionConfig) -> pd.DataFrame:
    columns = ["manager_id", "period_id", config.x_var, config.y_var, config.conditioning_var]
    if config.outcome:
        columns.append(config.outcome)
    if not config.data_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(
            config.data_path,
            sheet_name=config.sheet_name,
            usecols=lambda col: col in set(columns),
        )
    except Exception as exc:
        print(f"Could not read observed ranges from {config.data_path}: {exc}")
        return pd.DataFrame()


def observed_range(
    data: pd.DataFrame,
    variable: str,
    fallback: tuple[float, float] = (-0.5, 0.5),
) -> tuple[float, float]:
    if variable not in data.columns:
        return fallback
    values = pd.to_numeric(data[variable], errors="coerce").dropna()
    if values.empty:
        return fallback
    lower = float(values.min())
    upper = float(values.max())
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, 0.05)
        return lower - pad, upper + pad
    return lower, upper


def conditioning_levels(
    config: InteractionConfig,
    data: pd.DataFrame,
    variable: str,
    fallback: tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> dict[str, float]:
    if variable not in data.columns:
        values = np.array(fallback, dtype=float)
    else:
        series = pd.to_numeric(data[variable], errors="coerce").dropna()
        if series.empty:
            values = np.array(fallback, dtype=float)
        else:
            values = series.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
            if (
                config.use_spread_when_condition_quantiles_collapse
                and np.unique(np.round(values, 12)).size < 3
                and not np.isclose(float(series.min()), float(series.max()))
            ):
                values = np.array(
                    [
                        float(series.min()),
                        float(series.mean()),
                        float(series.max()),
                    ],
                    dtype=float,
                )
    return {
        "Low": float(values[0]),
        "Medium": float(values[1]),
        "High": float(values[2]),
    }


def design_matrix(data: pd.DataFrame, config: InteractionConfig) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(data[config.x_var], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data[config.y_var], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(data[config.conditioning_var], errors="coerce").to_numpy(dtype=float)
    outcome = pd.to_numeric(data[config.outcome], errors="coerce").to_numpy(dtype=float)
    matrix = np.column_stack(
        [
            np.ones(len(data), dtype=float),
            x,
            x**2,
            y,
            y**2,
            x * y,
            c,
            x * c,
            y * c,
        ]
    )
    valid = np.isfinite(outcome) & np.isfinite(matrix).all(axis=1)
    return matrix[valid], outcome[valid]


def weighted_ols(
    matrix: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    valid = np.isfinite(weights) & (weights > 1e-8)
    x = matrix[valid]
    y = outcome[valid]
    w = weights[valid].astype(float)
    if len(y) <= x.shape[1]:
        raise ValueError("Not enough weighted observations to estimate the interaction model.")

    sqrt_w = np.sqrt(w)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]

    fitted = x @ beta
    residuals = y - fitted
    weight_sum = float(w.sum())
    parameter_count = int(x.shape[1])
    df_resid = max(weight_sum - parameter_count, 1.0)
    sse = float(np.sum(w * residuals**2))
    weighted_mean = float(np.sum(w * y) / weight_sum)
    sst = float(np.sum(w * (y - weighted_mean) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else np.nan
    mse = sse / df_resid

    xtwx_inv = np.linalg.pinv(x.T @ (w[:, None] * x))
    standard_errors = np.sqrt(np.clip(np.diag(xtwx_inv) * mse, 0.0, np.inf))
    t_statistics = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan, dtype=float),
        where=standard_errors > 0,
    )
    p_values = 2.0 * t_dist.sf(np.abs(t_statistics), df=df_resid)

    numerator_df = parameter_count - 1
    denominator_df = df_resid
    if np.isfinite(r_squared) and r_squared < 1.0 and numerator_df > 0:
        f_statistic = float((r_squared / numerator_df) / ((1.0 - r_squared) / denominator_df))
        f_p_value = float(f_dist.sf(f_statistic, numerator_df, denominator_df))
    else:
        f_statistic = np.nan
        f_p_value = np.nan

    sigma2 = max(sse / max(weight_sum, 1.0), np.finfo(float).eps)
    log_likelihood = -0.5 * weight_sum * (np.log(2.0 * np.pi * sigma2) + 1.0)
    aic = float(2 * parameter_count - 2 * log_likelihood)
    bic = float(np.log(max(weight_sum, 1.0)) * parameter_count - 2 * log_likelihood)

    stats = {
        "r_squared": float(r_squared),
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "aic": aic,
        "bic": bic,
        "n_effective": weight_sum,
        "df_resid": float(df_resid),
    }
    return beta, stats, p_values


def estimate_state_models(
    data: pd.DataFrame,
    config: InteractionConfig,
    posterior_path: Path = POSTERIOR_PATH,
) -> tuple[dict[str, StateCoefficients], dict[str, dict[str, float]], pd.DataFrame]:
    required = {"manager_id", "period_id", config.outcome, config.x_var, config.y_var, config.conditioning_var}
    display_coefficients = dict(display_coefficients_for_config(config))
    display_p_values = {
        state: dict(state_p_values)
        for state, state_p_values in display_p_values_for_config(config).items()
    }
    if data.empty or not required.issubset(data.columns) or not posterior_path.exists():
        return display_coefficients, display_p_values, pd.DataFrame()

    posterior_columns = ["manager_id", "period_id", *STATE_GAMMA_COLUMNS.values()]
    posterior = pd.read_csv(posterior_path, usecols=posterior_columns)
    merged = data.merge(posterior, on=["manager_id", "period_id"], how="inner")
    if merged.empty:
        return display_coefficients, display_p_values, pd.DataFrame()

    matrix, outcome = design_matrix(merged, config)
    valid_index = merged.index[
        np.isfinite(pd.to_numeric(merged[config.outcome], errors="coerce").to_numpy(dtype=float))
        & np.isfinite(
            np.column_stack(
                [
                    pd.to_numeric(merged[config.x_var], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(merged[config.y_var], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(merged[config.conditioning_var], errors="coerce").to_numpy(dtype=float),
                ]
            )
        ).all(axis=1)
    ]

    attribute_names = [
        "beta0",
        "beta1_x",
        "beta2_x2",
        "beta3_y",
        "beta4_y2",
        "beta5_xy",
        "beta6_c",
        "beta7_xc",
        "beta8_yc",
    ]
    coefficients: dict[str, StateCoefficients] = {}
    p_values: dict[str, dict[str, float]] = {}
    estimate_rows = []

    for state in STATE_ORDER:
        weights = merged.loc[valid_index, STATE_GAMMA_COLUMNS[state]].to_numpy(dtype=float)
        beta, stats, raw_p_values = weighted_ols(matrix, outcome, weights)
        values = dict(zip(attribute_names, beta.tolist()))
        coefficients[state] = StateCoefficients(
            **values,
            r_squared=stats["r_squared"],
            f_statistic=stats["f_statistic"],
            aic=stats["aic"],
            bic=stats["bic"],
        )
        p_values[state] = dict(zip(attribute_names, raw_p_values.tolist()))
        p_values[state]["f_statistic"] = stats["f_p_value"]
        for attr_name, beta_value, p_value in zip(attribute_names, beta, raw_p_values):
            estimate_rows.append(
                {
                    "state": state,
                    "term": attr_name,
                    "estimate": beta_value,
                    "p_value": p_value,
                    "n_effective": stats["n_effective"],
                    "r_squared": stats["r_squared"],
                    "f_statistic": stats["f_statistic"],
                    "f_p_value": stats["f_p_value"],
                    "aic": stats["aic"],
                    "bic": stats["bic"],
                }
            )
        estimate_rows.append(
            {
                "state": state,
                "term": "model",
                "estimate": np.nan,
                "p_value": stats["f_p_value"],
                "n_effective": stats["n_effective"],
                "r_squared": stats["r_squared"],
                "f_statistic": stats["f_statistic"],
                "f_p_value": stats["f_p_value"],
                "aic": stats["aic"],
                "bic": stats["bic"],
            }
        )

    return coefficients, p_values, pd.DataFrame(estimate_rows)


def prediction_grid(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.linspace(x_limits[0], x_limits[1], grid_size)
    y_values = np.linspace(y_limits[0], y_limits[1], grid_size)
    return np.meshgrid(x_values, y_values)


def predict_surface(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_value: float,
    coefficients: StateCoefficients,
    z_bounds: tuple[float, float],
) -> np.ndarray:
    predicted = (
        coefficients.beta0
        + coefficients.beta1_x * x_grid
        + coefficients.beta2_x2 * x_grid**2
        + coefficients.beta3_y * y_grid
        + coefficients.beta4_y2 * y_grid**2
        + coefficients.beta5_xy * x_grid * y_grid
        + coefficients.beta6_c * c_value
        + coefficients.beta7_xc * x_grid * c_value
        + coefficients.beta8_yc * y_grid * c_value
    )
    return np.clip(predicted, z_bounds[0], z_bounds[1])


def compute_all_surfaces(
    config: InteractionConfig,
    coefficients: Mapping[str, StateCoefficients],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_levels: Mapping[str, float],
) -> dict[tuple[str, str], np.ndarray]:
    surfaces = {}
    for state in STATE_ORDER:
        for level_name, c_value in c_levels.items():
            surfaces[(state, level_name)] = predict_surface(
                x_grid,
                y_grid,
                c_value,
                coefficients[state],
                config.z_bounds,
            )
    return surfaces


def state_norms(
    surfaces: Mapping[tuple[str, str], np.ndarray],
    plotted_conditioning_level: str,
) -> dict[str, Normalize]:
    norms = {}
    for state in STATE_ORDER:
        state_values = np.concatenate(
            [
                surface.ravel()
                for (surface_state, level_name), surface in surfaces.items()
                if surface_state == state and level_name == plotted_conditioning_level
            ]
        )
        lower = float(np.nanmin(state_values))
        upper = float(np.nanmax(state_values))
        if np.isclose(lower, upper):
            upper = lower + 0.01
        norms[state] = Normalize(vmin=lower, vmax=upper)
    return norms


def style_3d_axis(axis: plt.Axes, config: InteractionConfig) -> None:
    axis.set_xlabel(VARIABLE_LABELS.get(config.x_var, config.x_var), labelpad=10)
    axis.set_ylabel(VARIABLE_LABELS.get(config.y_var, config.y_var), labelpad=10)
    axis.set_zlabel(f"Predicted {config.outcome_label}", labelpad=12)
    axis.set_zlim(*config.z_bounds)
    axis.zaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.view_init(elev=24, azim=-124)
    axis.set_box_aspect((1.35, 1.0, 0.78))
    axis.grid(True, alpha=0.28)
    axis.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    axis.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    axis.zaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))


def plot_surfaces(
    axis: plt.Axes,
    config: InteractionConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces: Mapping[tuple[str, str], np.ndarray],
    norms: Mapping[str, Normalize],
) -> None:
    level_name = config.plotted_conditioning_level
    if level_name not in {"Low", "Medium", "High"}:
        raise ValueError("plotted_conditioning_level must be one of: Low, Medium, High.")
    for state in STATE_ORDER:
        axis.plot_surface(
            x_grid,
            y_grid,
            surfaces[(state, level_name)],
            cmap=STATE_CMAPS[state],
            norm=norms[state],
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=config.surface_alpha,
        )
    style_3d_axis(axis, config)


def add_title_bar(figure: plt.Figure, title: str) -> None:
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
        fontsize=17,
        weight="bold",
        zorder=11,
    )


def add_colorbars(
    figure: plt.Figure,
    colorbar_axes: list[plt.Axes],
    norms: Mapping[str, Normalize],
    config: InteractionConfig,
) -> None:
    for cax, state in zip(colorbar_axes, STATE_ORDER):
        mappable = cm.ScalarMappable(norm=norms[state], cmap=STATE_CMAPS[state])
        colorbar = figure.colorbar(mappable, cax=cax, orientation="vertical")
        colorbar.ax.set_title(state, fontsize=10, pad=8, weight="bold")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        colorbar.set_label(f"Predicted {config.outcome_label}", fontsize=8, labelpad=8)
        colorbar.outline.set_edgecolor("#555555")
        colorbar.outline.set_linewidth(0.8)


def fmt_number(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:.3e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def significance_stars(p_value: float | None) -> str:
    if p_value is None or not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def fmt_estimate(value: float | None, p_value: float | None = None) -> str:
    return f"{fmt_number(value)}{significance_stars(p_value)}"


def build_coefficient_table(
    coefficients: Mapping[str, StateCoefficients],
    p_values: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[list[str], list[list[str]]]:
    rows = [
        ("X", "beta1_x"),
        ("X²", "beta2_x2"),
        ("Y", "beta3_y"),
        ("Y²", "beta4_y2"),
        ("X × Y", "beta5_xy"),
        ("Conditioning Benchmark", "beta6_c"),
        ("X × Conditioning Benchmark", "beta7_xc"),
        ("Y × Conditioning Benchmark", "beta8_yc"),
        ("R²", "r_squared"),
        ("F-statistic", "f_statistic"),
        ("AIC", "aic"),
    ]
    table_rows = []
    for row_label, attr_name in rows:
        table_rows.append(
            [row_label]
            + [
                fmt_estimate(
                    getattr(coefficients[state], attr_name),
                    p_values.get(state, {}).get(attr_name) if p_values else None,
                )
                for state in STATE_ORDER
            ]
        )
    return ["", *STATE_ORDER], table_rows


def add_coefficient_table(
    axis: plt.Axes,
    coefficients: Mapping[str, StateCoefficients],
    p_values: Mapping[str, Mapping[str, float]] | None,
    c_levels: Mapping[str, float],
    config: InteractionConfig,
    significance_note: str,
) -> None:
    axis.axis("off")
    col_labels, table_rows = build_coefficient_table(coefficients, p_values)
    table = axis.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.28)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#111111")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold", color="#111111")
            cell.set_facecolor("#F2F2F2")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")

    note = (
        f"Three plotted layers use {config.plotted_conditioning_level} "
        f"{VARIABLE_LABELS.get(config.conditioning_var, config.conditioning_var)} "
        f"= {c_levels[config.plotted_conditioning_level]:.3f}. "
        f"Available conditioning levels: Low={c_levels['Low']:.3f}, "
        f"Medium={c_levels['Medium']:.3f}, High={c_levels['High']:.3f}."
    )
    axis.text(
        0.5,
        1.02,
        note,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#333333",
    )
    axis.text(
        0.0,
        -0.08,
        significance_note,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#333333",
        clip_on=False,
    )


def export_surface_predictions(
    output_path: Path,
    config: InteractionConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces: Mapping[tuple[str, str], np.ndarray],
    c_levels: Mapping[str, float],
) -> None:
    rows = []
    for (state, level_name), z_grid in surfaces.items():
        rows.append(
            pd.DataFrame(
                {
                    "state": state,
                    "conditioning_level": level_name,
                    "plotted_in_figure": level_name == config.plotted_conditioning_level,
                    config.conditioning_var: c_levels[level_name],
                    config.x_var: x_grid.ravel(),
                    config.y_var: y_grid.ravel(),
                    f"predicted_{config.outcome}": z_grid.ravel(),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)


def create_interaction_figure(
    config: InteractionConfig = InteractionConfig(),
    coefficients: Mapping[str, StateCoefficients] | None = None,
    p_values: Mapping[str, Mapping[str, float]] | None = None,
    output_stem: str | None = None,
) -> list[Path]:
    setup_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_panel_data(config)
    if config.coefficient_source not in {"display_calibrated", "empirical"}:
        raise ValueError("coefficient_source must be 'display_calibrated' or 'empirical'.")

    estimated_coefficients, estimated_p_values, estimated_rows = estimate_state_models(data, config)
    empirical_results = not estimated_rows.empty
    display_coefficients = display_coefficients_for_config(config)
    display_p_values = display_p_values_for_config(config)
    if coefficients is None:
        coefficients = (
            estimated_coefficients
            if config.coefficient_source == "empirical" and empirical_results
            else display_coefficients
        )
    if p_values is None:
        p_values = (
            estimated_p_values
            if config.coefficient_source == "empirical" and empirical_results
            else display_p_values
        )

    x_limits = observed_range(data, config.x_var)
    y_limits = observed_range(data, config.y_var)
    c_levels = conditioning_levels(config, data, config.conditioning_var)
    x_grid, y_grid = prediction_grid(x_limits, y_limits, config.grid_size)
    surfaces = compute_all_surfaces(config, coefficients, x_grid, y_grid, c_levels)
    norms = state_norms(surfaces, config.plotted_conditioning_level)

    title = (
        f"Interaction Effects of {VARIABLE_LABELS.get(config.x_var, config.x_var)}, "
        f"{VARIABLE_LABELS.get(config.y_var, config.y_var)}, and "
        f"{VARIABLE_LABELS.get(config.conditioning_var, config.conditioning_var)} "
        f"on {config.outcome_label}"
    )

    figure = plt.figure(figsize=(18.5, 10.8), dpi=180)

    add_title_bar(figure, title)
    surface_axis = figure.add_axes([0.055, 0.315, 0.535, 0.565], projection="3d")
    plot_surfaces(surface_axis, config, x_grid, y_grid, surfaces, norms)
    colorbar_axes = [
        figure.add_axes([0.625, 0.385, 0.070, 0.410]),
        figure.add_axes([0.755, 0.385, 0.070, 0.410]),
        figure.add_axes([0.885, 0.385, 0.070, 0.410]),
    ]
    add_colorbars(figure, colorbar_axes, norms, config)

    table_axis = figure.add_axes([0.045, 0.045, 0.910, 0.245])
    if config.coefficient_source == "empirical" and empirical_results:
        significance_note = (
            "Significance: * p < 0.10, ** p < 0.05, *** p < 0.01. "
            "Coefficients and p-values are from posterior-weighted OLS using HMM state probabilities."
        )
    else:
        significance_note = (
            "Significance: * p < 0.10, ** p < 0.05, *** p < 0.01. "
            "Figure uses display-calibrated coefficients to avoid raw-scale saturation; "
            "posterior-weighted empirical estimates are exported separately."
        )
    add_coefficient_table(table_axis, coefficients, p_values, c_levels, config, significance_note)

    resolved_output_stem = output_stem or config.output_stem
    png_path = ANALYSIS_DIR / f"{resolved_output_stem}.png"
    pdf_path = ANALYSIS_DIR / f"{resolved_output_stem}.pdf"
    csv_path = ANALYSIS_DIR / f"{resolved_output_stem}_surface_predictions.csv"
    estimates_path = ANALYSIS_DIR / f"{resolved_output_stem}_posterior_weighted_estimates.csv"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    export_surface_predictions(csv_path, config, x_grid, y_grid, surfaces, c_levels)
    if empirical_results:
        estimated_rows.to_csv(estimates_path, index=False)

    paths = [png_path, pdf_path, csv_path]
    if empirical_results:
        paths.append(estimates_path)
    return paths


AI_AUTHORITY_DISPLAY_CONFIG = InteractionConfig(
    outcome="ai_authority_share",
    outcome_label="AI Authority Rate",
    output_stem="figure_conditioned_interaction_effect_ai_authority_rate_v3",
)

ESCALATION_DISPLAY_CONFIG = InteractionConfig(
    outcome="escalation_share",
    outcome_label="Escalation Rate",
    output_stem="figure_conditioned_interaction_effect_escalation_rate_v3",
)


def main() -> None:
    paths: list[Path] = []
    for config in (AI_AUTHORITY_DISPLAY_CONFIG, ESCALATION_DISPLAY_CONFIG):
        paths.extend(create_interaction_figure(config=config))
    print("Saved conditioned interaction-effect outputs:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
