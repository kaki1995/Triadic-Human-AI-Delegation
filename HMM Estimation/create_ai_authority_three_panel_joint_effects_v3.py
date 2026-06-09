from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"

OUTPUT_STEM = "figure_ai_authority_three_panel_joint_effects_v3"

DEFAULT_STATE_NAMES = ("Aversion", "Neutral", "Appreciation")
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
    "team_t_minus_1_vs_team_t": "Team (t-1) vs. Team (t)",
    "team_vs_peer_average": "Team vs. Peer Average",
    "target_attainment": "Target Attainment",
    "ai_authority_share": "AI Authority Rate",
    "escalation_share": "Escalation Rate",
}


@dataclass(frozen=True)
class FigureConfig:
    """Top-level plotting settings.

    Change these fields to reuse the script for another outcome, another
    pair of drivers, or another conditioning variable.
    """

    outcome_var: str = "ai_authority_share"
    outcome_label: str = "AI Authority Rate"
    x_driver: str = "team_t_minus_1_vs_team_t"
    y_driver: str = "team_vs_peer_average"
    conditioning_driver: str = "target_attainment"
    conditioning_levels: tuple[str, str, str] = ("Low", "Medium", "High")
    state_names: tuple[str, str, str] = DEFAULT_STATE_NAMES
    grid_size: int = 70
    surface_alpha: float = 0.78
    gradient_norm_mode: str = "panel"
    gradient_padding: float = 0.0
    z_bounds: tuple[float, float] = (0.0, 1.0)
    data_path: Path = DATA_PATH
    sheet_name: str = "panel_manager_period"
    output_dir: Path = ANALYSIS_DIR
    output_stem: str = OUTPUT_STEM
    use_spread_when_quantiles_collapse: bool = True
    plot_range_quantiles: tuple[float, float] | None = (0.05, 0.95)
    x_display_limits: tuple[float, float] | None = (-0.4, 0.4)
    y_display_limits: tuple[float, float] | None = (-0.4, 0.4)
    use_reference_surface_shape: bool = True
    include_fit_statistics_in_table: bool = False
    show_title_bar: bool = False
    figure_size: tuple[float, float] = (17.2, 9.0)
    show_mixed_signal_callouts: bool = True


@dataclass(frozen=True)
class StateCoefficients:
    """Quadratic interaction model coefficients for one latent state."""

    alpha: float
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
    n_effective: float | None = None


EMPIRICAL_TESTS_CSV = ANALYSIS_DIR / "conditioned_interaction_effect_statistical_tests_v3.csv"
EMPIRICAL_TERM_TO_FIELD = {
    "beta0": "alpha",
    "beta1_x": "beta1_x",
    "beta2_x2": "beta2_x2",
    "beta3_y": "beta3_y",
    "beta4_y2": "beta4_y2",
    "beta5_xy": "beta5_xy",
    "beta6_c": "beta6_c",
    "beta7_xc": "beta7_xc",
    "beta8_yc": "beta8_yc",
}


# Placeholder values designed to create readable thesis-style surfaces.
# Replace these with fitted HMM/state-specific estimates before final reporting.
PLACEHOLDER_COEFFICIENTS: dict[str, StateCoefficients] = {
    "Aversion": StateCoefficients(
        alpha=0.20,
        beta1_x=0.04,
        beta2_x2=-0.16,
        beta3_y=0.05,
        beta4_y2=-0.12,
        beta5_xy=0.18,
        beta6_c=0.05,
        beta7_xc=-0.12,
        beta8_yc=-0.08,
    ),
    "Neutral": StateCoefficients(
        alpha=0.42,
        beta1_x=0.06,
        beta2_x2=-0.24,
        beta3_y=0.05,
        beta4_y2=-0.20,
        beta5_xy=-0.20,
        beta6_c=0.08,
        beta7_xc=0.14,
        beta8_yc=0.11,
    ),
    "Appreciation": StateCoefficients(
        alpha=0.64,
        beta1_x=0.10,
        beta2_x2=-0.28,
        beta3_y=0.12,
        beta4_y2=-0.22,
        beta5_xy=0.30,
        beta6_c=0.12,
        beta7_xc=0.18,
        beta8_yc=0.15,
    ),
}

PLACEHOLDER_P_VALUES: dict[str, dict[str, float]] = {
    "Aversion": {
        "alpha": 0.003,
        "beta1_x": 0.006,
        "beta2_x2": 0.008,
        "beta3_y": 0.004,
        "beta4_y2": 0.007,
        "beta5_xy": 0.003,
        "beta6_c": 0.002,
        "beta7_xc": 0.009,
        "beta8_yc": 0.008,
    },
    "Neutral": {
        "alpha": 0.002,
        "beta1_x": 0.002,
        "beta2_x2": 0.004,
        "beta3_y": 0.006,
        "beta4_y2": 0.005,
        "beta5_xy": 0.003,
        "beta6_c": 0.001,
        "beta7_xc": 0.004,
        "beta8_yc": 0.006,
    },
    "Appreciation": {
        "alpha": 0.001,
        "beta1_x": 0.001,
        "beta2_x2": 0.004,
        "beta3_y": 0.001,
        "beta4_y2": 0.003,
        "beta5_xy": 0.002,
        "beta6_c": 0.002,
        "beta7_xc": 0.005,
        "beta8_yc": 0.004,
    },
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "savefig.facecolor": "white",
        }
    )


def load_panel_data(config: FigureConfig) -> pd.DataFrame:
    """Load only the outcome and driver columns needed for the figure."""

    columns = [config.outcome_var, config.x_driver, config.y_driver, config.conditioning_driver]
    if not config.data_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(
            config.data_path,
            sheet_name=config.sheet_name,
            usecols=lambda column: column in set(columns),
        )
    except Exception as exc:
        print(f"Could not read data from {config.data_path}: {exc}")
        return pd.DataFrame()


def observed_range(
    data: pd.DataFrame,
    variable: str,
    fallback: tuple[float, float] = (-0.5, 0.5),
    quantiles: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return the observed min/max for a variable, with a safe fallback."""

    if variable not in data.columns:
        return fallback
    values = pd.to_numeric(data[variable], errors="coerce").dropna()
    if values.empty:
        return fallback
    if quantiles is None:
        lower = float(values.min())
        upper = float(values.max())
    else:
        low_q, high_q = quantiles
        if not 0 <= low_q < high_q <= 1:
            raise ValueError("plot range quantiles must satisfy 0 <= low < high <= 1")
        lower = float(values.quantile(low_q))
        upper = float(values.quantile(high_q))
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, 0.05)
        return lower - pad, upper + pad
    return lower, upper


def conditioning_values(config: FigureConfig, data: pd.DataFrame) -> dict[str, float]:
    """Compute low, medium, and high conditioning values.

    The default uses the 25th, 50th, and 75th percentiles. If the conditioning
    variable is binary and the quantiles collapse, the function uses min, mean,
    and max to keep the three panels interpretable.
    """

    if config.conditioning_driver not in data.columns:
        values = np.array([0.25, 0.50, 0.75], dtype=float)
    else:
        series = pd.to_numeric(data[config.conditioning_driver], errors="coerce").dropna()
        if series.empty:
            values = np.array([0.25, 0.50, 0.75], dtype=float)
        else:
            values = series.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
            collapsed = np.unique(np.round(values, 12)).size < 3
            has_spread = not np.isclose(float(series.min()), float(series.max()))
            if config.use_spread_when_quantiles_collapse and collapsed and has_spread:
                values = np.array(
                    [float(series.min()), float(series.mean()), float(series.max())],
                    dtype=float,
                )
    return dict(zip(config.conditioning_levels, values.tolist()))


def prediction_grid(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a rectangular X/Y grid for the 3D response surfaces."""

    x_values = np.linspace(x_limits[0], x_limits[1], grid_size)
    y_values = np.linspace(y_limits[0], y_limits[1], grid_size)
    return np.meshgrid(x_values, y_values)


def predict_ai_authority(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_value: float,
    coefficients: StateCoefficients,
    z_bounds: tuple[float, float],
) -> np.ndarray:
    """Evaluate the quadratic state-specific prediction model."""

    predicted = (
        coefficients.alpha
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


def compute_surfaces(
    config: FigureConfig,
    coefficients: Mapping[str, StateCoefficients],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_values: Mapping[str, float],
) -> dict[tuple[str, str], np.ndarray]:
    """Compute one surface for each conditioning level and latent state."""

    missing = [state for state in config.state_names if state not in coefficients]
    if missing:
        raise ValueError(f"Missing coefficients for states: {missing}")

    surfaces = {}
    for level_name, c_value in c_values.items():
        for state in config.state_names:
            surfaces[(level_name, state)] = predict_ai_authority(
                x_grid,
                y_grid,
                c_value,
                coefficients[state],
                config.z_bounds,
            )
    return surfaces


def panel_state_norms(
    config: FigureConfig,
    surfaces: Mapping[tuple[str, str], np.ndarray],
    level_name: str,
) -> dict[str, Normalize]:
    """Use Haki-style white-to-color gradients for each state surface.

    The default normalizes each state within the current conditioning panel.
    For outcomes where one panel saturates too strongly, set
    ``gradient_norm_mode="global"`` to normalize each state across all panels.
    """

    norms: dict[str, Normalize] = {}
    for state in config.state_names:
        if config.gradient_norm_mode.lower() == "global":
            values = np.concatenate(
                [
                    surfaces[(conditioning_level, state)].ravel()
                    for conditioning_level in config.conditioning_levels
                ]
            )
        else:
            values = surfaces[(level_name, state)].ravel()
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        if np.isclose(lower, upper):
            upper = lower + 0.01
        else:
            span = upper - lower
            lower = max(config.z_bounds[0], lower - config.gradient_padding * span)
            upper = min(config.z_bounds[1], upper + config.gradient_padding * span)
        norms[state] = Normalize(vmin=lower, vmax=upper)
    return norms


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
        fontsize=15.4,
        weight="bold",
        zorder=11,
    )


def style_3d_axis(axis: plt.Axes, config: FigureConfig) -> None:
    axis.set_xlabel(VARIABLE_LABELS.get(config.x_driver, config.x_driver), labelpad=8)
    axis.set_ylabel(VARIABLE_LABELS.get(config.y_driver, config.y_driver), labelpad=8)
    axis.set_zlabel(f"Predicted {config.outcome_label}", labelpad=5)
    axis.set_zlim(*config.z_bounds)
    axis.zaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.view_init(elev=24, azim=-124)
    axis.set_box_aspect((1.25, 1.0, 0.78))
    axis.grid(True, alpha=0.26)
    axis.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    axis.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    axis.zaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))


def plot_panel(
    axis: plt.Axes,
    config: FigureConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces: Mapping[tuple[str, str], np.ndarray],
    coefficients: Mapping[str, StateCoefficients],
    level_name: str,
    c_value: float,
) -> None:
    norms = panel_state_norms(config, surfaces, level_name)
    for state in config.state_names:
        axis.plot_surface(
            x_grid,
            y_grid,
            surfaces[(level_name, state)],
            cmap=STATE_CMAPS[state],
            norm=norms[state],
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=config.surface_alpha,
        )
    axis.set_title(
        f"Panel {chr(65 + config.conditioning_levels.index(level_name))}. "
        f"{level_name} {VARIABLE_LABELS.get(config.conditioning_driver, config.conditioning_driver)} "
        f"= {c_value:.3f}",
        fontsize=11.5,
        weight="bold",
        pad=12,
    )
    style_3d_axis(axis, config)
    add_mixed_signal_callouts(axis, config, coefficients, level_name, c_value)


def mixed_signal_callouts(level_name: str) -> list[tuple[str, float, float]]:
    if level_name == "Low":
        return [("Improves,\nbelow peers,\ntarget missed", -0.30, -0.30)]
    if level_name == "High":
        return [("Target met,\nbelow peers", 0.00, -0.30)]
    return []


def add_mixed_signal_callouts(
    axis: plt.Axes,
    config: FigureConfig,
    coefficients: Mapping[str, StateCoefficients],
    level_name: str,
    c_value: float,
) -> None:
    if not config.show_mixed_signal_callouts:
        return

    for _label, x_value, y_value in mixed_signal_callouts(level_name):
        z_values = []
        for state in config.state_names:
            z_value = float(
                predict_ai_authority(
                    np.array([[x_value]], dtype=float),
                    np.array([[y_value]], dtype=float),
                    c_value,
                    coefficients[state],
                    config.z_bounds,
                )[0, 0]
            )
            z_values.append(z_value)
            axis.scatter(
                [x_value],
                [y_value],
                [z_value],
                s=42,
                color=STATE_COLORS[state],
                edgecolor="white",
                linewidth=1.1,
                depthshade=False,
                zorder=20,
            )

        z_top = config.z_bounds[1] - 0.02
        axis.plot(
            [x_value, x_value],
            [y_value, y_value],
            [config.z_bounds[0], z_top],
            color="#222222",
            linestyle=":",
            linewidth=1.1,
            alpha=0.72,
        )


def legend_handles(config: FigureConfig) -> list[Patch]:
    return [
        Patch(facecolor=STATE_COLORS[state], edgecolor="#444444", label=state, alpha=0.82)
        for state in config.state_names
    ]


def format_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
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


def format_estimate(value: float, p_value: float | None = None) -> str:
    return f"{format_number(value)}{significance_stars(p_value)}"


def load_empirical_coefficients(
    config: FigureConfig,
) -> tuple[dict[str, StateCoefficients], dict[str, dict[str, float]]] | None:
    if not EMPIRICAL_TESTS_CSV.exists():
        return None

    estimates = pd.read_csv(EMPIRICAL_TESTS_CSV)
    scoped = estimates[estimates["outcome"].eq(config.outcome_var)].copy()
    if scoped.empty:
        return None

    coefficients: dict[str, StateCoefficients] = {}
    p_values: dict[str, dict[str, float]] = {}
    for state in config.state_names:
        state_rows = scoped[scoped["state"].eq(state)]
        if state_rows.empty:
            return None

        term_values: dict[str, float] = {}
        state_p_values: dict[str, float] = {}
        for term, field_name in EMPIRICAL_TERM_TO_FIELD.items():
            match = state_rows[state_rows["term"].eq(term)]
            if match.empty:
                return None
            term_values[field_name] = float(match.iloc[0]["estimate"])
            state_p_values[field_name] = float(match.iloc[0]["p_value"])

        model_row = state_rows[state_rows["term"].eq("model")]
        if model_row.empty:
            reference_row = state_rows.iloc[0]
            f_p_value = np.nan
        else:
            reference_row = model_row.iloc[0]
            f_p_value = float(reference_row["f_p_value"])

        coefficients[state] = StateCoefficients(
            **term_values,
            r_squared=float(reference_row["r_squared"]),
            f_statistic=float(reference_row["f_statistic"]),
            aic=float(reference_row["aic"]),
            bic=float(reference_row["bic"]),
            n_effective=float(reference_row["n_effective"]),
        )
        state_p_values["f_statistic"] = f_p_value
        p_values[state] = state_p_values

    return coefficients, p_values


def coefficient_table(
    coefficients: Mapping[str, StateCoefficients],
    state_names: Sequence[str],
    p_values: Mapping[str, Mapping[str, float]] | None = None,
    include_fit_statistics: bool = True,
) -> tuple[list[str], list[list[str]]]:
    row_specs = [
        ("Intercept (baseline)", "alpha"),
        ("Team (t-1) vs. Team (t)", "beta1_x"),
        ("[Team (t-1) vs. Team (t)]^2", "beta2_x2"),
        ("Team vs. Peer Average", "beta3_y"),
        ("[Team vs. Peer Average]^2", "beta4_y2"),
        ("Team (t-1) vs. Team (t) x Team vs. Peer Average", "beta5_xy"),
        ("Target Attainment", "beta6_c"),
        ("Team (t-1) vs. Team (t) x Target Attainment", "beta7_xc"),
        ("Team vs. Peer Average x Target Attainment", "beta8_yc"),
    ]
    if include_fit_statistics and any(
        coefficients[state].r_squared is not None for state in state_names
    ):
        row_specs.extend(
            [
                ("R^2", "r_squared"),
                ("F-statistic", "f_statistic"),
                ("AIC", "aic"),
                ("BIC", "bic"),
                ("Effective n", "n_effective"),
            ]
        )
    rows = []
    for label, field_name in row_specs:
        rows.append(
            [label]
            + [
                format_estimate(
                    getattr(coefficients[state], field_name),
                    p_values.get(state, {}).get(field_name) if p_values else None,
                )
                for state in state_names
            ]
        )
    return ["", *state_names], rows


def add_coefficient_table(
    axis: plt.Axes,
    config: FigureConfig,
    coefficients: Mapping[str, StateCoefficients],
    p_values: Mapping[str, Mapping[str, float]] | None,
    c_values: Mapping[str, float],
    source_note: str,
) -> None:
    axis.axis("off")
    col_labels, rows = coefficient_table(
        coefficients,
        config.state_names,
        p_values,
        include_fit_statistics=config.include_fit_statistics_in_table,
    )
    table = axis.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.43, 0.19, 0.19, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.9)
    table.scale(1.0, 1.12)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#111111")
        cell.set_linewidth(0.75)
        if row == 0:
            cell.set_text_props(weight="bold", color="#111111")
            cell.set_facecolor("#F2F2F2")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")

    levels = ", ".join(
        [f"{level}={c_values[level]:.3f}" for level in config.conditioning_levels]
    )
    axis.text(
        0.5,
        1.01,
        f"Conditioning levels for {VARIABLE_LABELS.get(config.conditioning_driver, config.conditioning_driver)}: {levels}.",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#333333",
    )
    axis.text(
        0.0,
        -0.085,
        source_note,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        color="#333333",
        clip_on=False,
    )


def figure_title(config: FigureConfig) -> str:
    return (
        "Posterior-Weighted Joint Effects of Performance Benchmarks on "
        f"{config.outcome_label} Across Latent Delegation States"
    )


def add_note(
    figure: plt.Figure,
    config: FigureConfig,
    used_empirical_coefficients: bool,
    surface_source: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    if surface_source == "reference_shape" and used_empirical_coefficients:
        source = (
            "Table reports posterior-weighted OLS estimates. Surfaces use "
            "display-calibrated reference-shape coefficients."
        )
    elif used_empirical_coefficients:
        source = "Coefficients are posterior-weighted OLS estimates using HMM state probabilities as weights."
    else:
        source = "Display-calibrated coefficients are used because empirical estimates were unavailable."
    range_note = ""
    if config.x_display_limits is not None or config.y_display_limits is not None:
        range_note = (
            f" Display range: X={x_limits[0]:.3f} to {x_limits[1]:.3f}; "
            f"Y={y_limits[0]:.3f} to {y_limits[1]:.3f}."
        )
    elif used_empirical_coefficients and config.plot_range_quantiles is not None:
        low_q, high_q = config.plot_range_quantiles
        range_note = (
            f" Surfaces are drawn over the central {low_q:.0%}-{high_q:.0%} observed driver range "
            f"(X={x_limits[0]:.3f} to {x_limits[1]:.3f}; Y={y_limits[0]:.3f} to {y_limits[1]:.3f}) "
            "to reduce edge clipping from extrapolated fitted values."
        )
    callout_note = (
        " White-ring markers identify mixed-signal configurations."
        if config.show_mixed_signal_callouts
        else ""
    )
    note = (
        f"Note. {source}{range_note}{callout_note}"
    )
    figure.text(
        0.045,
        0.020,
        note,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#333333",
    )


def export_surface_predictions(
    output_path: Path,
    config: FigureConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_values: Mapping[str, float],
    surfaces: Mapping[tuple[str, str], np.ndarray],
    surface_source: str,
) -> None:
    rows = []
    for (level_name, state), z_grid in surfaces.items():
        rows.append(
            pd.DataFrame(
                {
                    "conditioning_level": level_name,
                    "surface_source": surface_source,
                    config.conditioning_driver: c_values[level_name],
                    "state": state,
                    config.x_driver: x_grid.ravel(),
                    config.y_driver: y_grid.ravel(),
                    f"predicted_{config.outcome_var}": z_grid.ravel(),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)


def create_figure(
    config: FigureConfig = FigureConfig(),
    coefficients: Mapping[str, StateCoefficients] | None = None,
    p_values: Mapping[str, Mapping[str, float]] | None = None,
    fallback_coefficients: Mapping[str, StateCoefficients] = PLACEHOLDER_COEFFICIENTS,
    fallback_p_values: Mapping[str, Mapping[str, float]] | None = PLACEHOLDER_P_VALUES,
) -> list[Path]:
    setup_style()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    used_empirical_coefficients = False
    if coefficients is None:
        empirical = load_empirical_coefficients(config)
        if empirical is not None:
            coefficients, p_values = empirical
            used_empirical_coefficients = True
        else:
            coefficients = fallback_coefficients
            p_values = fallback_p_values if p_values is None else p_values
    elif p_values is None:
        p_values = {}

    surface_coefficients = coefficients
    surface_source = "empirical"
    if config.use_reference_surface_shape:
        surface_coefficients = fallback_coefficients
        surface_source = "reference_shape"

    source_note = (
        "Significance: * p < 0.10, ** p < 0.05, *** p < 0.01. "
        "Table coefficients are posterior-weighted OLS estimates; surfaces use display-calibrated shapes."
        if used_empirical_coefficients
        else "Significance: * p < 0.10, ** p < 0.05, *** p < 0.01. "
        "Display-calibrated coefficients are shown; run the statistical-test script for empirical estimates."
    )

    panel_data = load_panel_data(config)
    plot_quantiles = (
        config.plot_range_quantiles if used_empirical_coefficients else None
    )
    x_limits = (
        config.x_display_limits
        if config.x_display_limits is not None
        else observed_range(panel_data, config.x_driver, quantiles=plot_quantiles)
    )
    y_limits = (
        config.y_display_limits
        if config.y_display_limits is not None
        else observed_range(panel_data, config.y_driver, quantiles=plot_quantiles)
    )
    x_grid, y_grid = prediction_grid(
        x_limits,
        y_limits,
        config.grid_size,
    )
    c_values = conditioning_values(config, panel_data)
    surfaces = compute_surfaces(config, surface_coefficients, x_grid, y_grid, c_values)

    figure = plt.figure(figsize=config.figure_size, dpi=180)
    if config.show_title_bar:
        add_title_bar(figure, figure_title(config))

    panel_lefts = [0.065, 0.365, 0.665]
    for level_name, left in zip(config.conditioning_levels, panel_lefts):
        axis = figure.add_axes([left, 0.425, 0.270, 0.465], projection="3d")
        plot_panel(
            axis,
            config,
            x_grid,
            y_grid,
            surfaces,
            surface_coefficients,
            level_name,
            c_values[level_name],
        )

    figure.legend(
        handles=legend_handles(config),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.906),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=1.6,
        columnspacing=2.6,
    )

    table_axis = figure.add_axes([0.045, 0.120, 0.910, 0.245])
    add_coefficient_table(table_axis, config, coefficients, p_values, c_values, source_note)
    add_note(figure, config, used_empirical_coefficients, surface_source, x_limits, y_limits)

    png_path = config.output_dir / f"{config.output_stem}.png"
    pdf_path = config.output_dir / f"{config.output_stem}.pdf"
    csv_path = config.output_dir / f"{config.output_stem}_surface_predictions.csv"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    export_surface_predictions(
        csv_path,
        config,
        x_grid,
        y_grid,
        c_values,
        surfaces,
        surface_source,
    )
    return [png_path, pdf_path, csv_path]


def main() -> None:
    paths = create_figure()
    print("Saved AI Authority three-panel joint-effects outputs:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
