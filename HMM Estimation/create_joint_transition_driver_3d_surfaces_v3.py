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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"

OUTPUT_STEM = "figure_joint_transition_driver_surfaces_from_neutral_v3"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
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


@dataclass(frozen=True)
class FigureConfig:
    """Controls the state, drivers, conditioning level, and output names."""

    starting_state: str = "Neutral"
    x_driver: str = "team_t_minus_1_vs_team_t"
    y_driver: str = "team_vs_peer_average"
    conditioning_driver: str = "target_attainment"
    conditioning_level: str = "Medium"
    grid_size: int = 85
    surface_alpha: float = 0.84
    z_bounds: tuple[float, float] = (0.0, 1.0)
    data_path: Path = DATA_PATH
    sheet_name: str = "panel_manager_period"
    output_stem: str = OUTPUT_STEM


@dataclass(frozen=True)
class TransitionCoefficients:
    """Multinomial-logit utility coefficients for one destination state."""

    intercept: float = 0.0
    x: float = 0.0
    x2: float = 0.0
    y: float = 0.0
    y2: float = 0.0
    xy: float = 0.0
    c: float = 0.0
    xc: float = 0.0
    yc: float = 0.0


# Placeholder coefficients for transitions from Neutral.
# Replace these values with your fitted HMM transition estimates.
# Neutral -> Neutral is set as the reference category with utility zero.
PLACEHOLDER_COEFFICIENTS: dict[str, TransitionCoefficients] = {
    "Aversion": TransitionCoefficients(
        intercept=-0.65,
        x=-0.30,
        x2=0.05,
        y=-2.20,
        y2=0.60,
        xy=-0.65,
        c=-0.90,
        xc=-0.20,
        yc=-0.65,
    ),
    "Neutral": TransitionCoefficients(),
    "Appreciation": TransitionCoefficients(
        intercept=-2.10,
        x=0.50,
        x2=-0.25,
        y=2.00,
        y2=-0.30,
        xy=0.95,
        c=0.85,
        xc=0.35,
        yc=0.70,
    ),
}

PLACEHOLDER_P_VALUES: dict[str, dict[str, float]] = {
    "Aversion": {
        "intercept": 0.004,
        "x": 0.018,
        "x2": 0.071,
        "y": 0.001,
        "y2": 0.009,
        "xy": 0.006,
        "c": 0.003,
        "xc": 0.041,
        "yc": 0.007,
    },
    "Neutral": {},
    "Appreciation": {
        "intercept": 0.001,
        "x": 0.006,
        "x2": 0.033,
        "y": 0.001,
        "y2": 0.024,
        "xy": 0.002,
        "c": 0.004,
        "xc": 0.016,
        "yc": 0.005,
    },
}


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


def load_panel_data(config: FigureConfig) -> pd.DataFrame:
    """Load only the columns needed to define observed ranges."""

    columns = [config.x_driver, config.y_driver, config.conditioning_driver]
    if not config.data_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(
            config.data_path,
            sheet_name=config.sheet_name,
            usecols=lambda col: col in set(columns),
        )
    except Exception as exc:
        print(f"Could not read observed driver ranges from {config.data_path}: {exc}")
        return pd.DataFrame()


def observed_range(
    data: pd.DataFrame,
    variable: str,
    fallback: tuple[float, float] = (-0.5, 0.5),
) -> tuple[float, float]:
    """Return the observed min and max for a driver, with a safe fallback."""

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
    data: pd.DataFrame,
    variable: str,
    fallback: tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> dict[str, float]:
    """Return low, medium, and high conditioning values.

    For continuous variables this uses the 25th, 50th, and 75th percentiles.
    For binary variables whose quantiles collapse, it uses min, mean, and max
    so that the medium scenario remains informative.
    """

    if variable not in data.columns:
        values = np.array(fallback, dtype=float)
    else:
        series = pd.to_numeric(data[variable], errors="coerce").dropna()
        if series.empty:
            values = np.array(fallback, dtype=float)
        else:
            values = series.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
            if np.unique(np.round(values, 12)).size < 3 and not np.isclose(
                float(series.min()), float(series.max())
            ):
                values = np.array(
                    [float(series.min()), float(series.mean()), float(series.max())],
                    dtype=float,
                )
    return {"Low": float(values[0]), "Medium": float(values[1]), "High": float(values[2])}


def prediction_grid(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a rectangular grid over the observed X and Y ranges."""

    x_values = np.linspace(x_limits[0], x_limits[1], grid_size)
    y_values = np.linspace(y_limits[0], y_limits[1], grid_size)
    return np.meshgrid(x_values, y_values)


def utility(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_value: float,
    coefficients: TransitionCoefficients,
) -> np.ndarray:
    """Compute the latent transition utility for one destination state."""

    return (
        coefficients.intercept
        + coefficients.x * x_grid
        + coefficients.x2 * x_grid**2
        + coefficients.y * y_grid
        + coefficients.y2 * y_grid**2
        + coefficients.xy * x_grid * y_grid
        + coefficients.c * c_value
        + coefficients.xc * x_grid * c_value
        + coefficients.yc * y_grid * c_value
    )


def softmax_utilities(utilities: np.ndarray) -> np.ndarray:
    """Stable softmax over destination-state utilities."""

    shifted = utilities - np.nanmax(utilities, axis=0, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)


def compute_surfaces(
    config: FigureConfig,
    coefficients: Mapping[str, TransitionCoefficients],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_value: float,
) -> dict[str, np.ndarray]:
    """Compute one transition-probability surface for each destination state."""

    missing_states = [state for state in STATE_ORDER if state not in coefficients]
    if missing_states:
        raise ValueError(f"Missing coefficient entries for: {missing_states}")

    utility_stack = np.stack(
        [
            utility(x_grid, y_grid, c_value, coefficients[state])
            for state in STATE_ORDER
        ],
        axis=0,
    )
    probability_stack = softmax_utilities(utility_stack)
    return {
        state: np.clip(probability_stack[index], *config.z_bounds)
        for index, state in enumerate(STATE_ORDER)
    }


def surface_norms(surfaces: Mapping[str, np.ndarray]) -> dict[str, Normalize]:
    """Use separate color scales for each destination-state surface."""

    norms = {}
    for state, surface in surfaces.items():
        lower = float(np.nanmin(surface))
        upper = float(np.nanmax(surface))
        if np.isclose(lower, upper):
            upper = lower + 0.01
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
        fontsize=16,
        weight="bold",
        zorder=11,
    )


def style_3d_axis(axis: plt.Axes, config: FigureConfig) -> None:
    axis.set_xlabel(VARIABLE_LABELS.get(config.x_driver, config.x_driver), labelpad=10)
    axis.set_ylabel(VARIABLE_LABELS.get(config.y_driver, config.y_driver), labelpad=10)
    axis.set_zlabel("Predicted transition probability", labelpad=12)
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
    config: FigureConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces: Mapping[str, np.ndarray],
    norms: Mapping[str, Normalize],
) -> None:
    for state in STATE_ORDER:
        axis.plot_surface(
            x_grid,
            y_grid,
            surfaces[state],
            cmap=STATE_CMAPS[state],
            norm=norms[state],
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=config.surface_alpha,
        )
    style_3d_axis(axis, config)


def add_colorbars(
    figure: plt.Figure,
    colorbar_axes: list[plt.Axes],
    norms: Mapping[str, Normalize],
) -> None:
    for colorbar_axis, state in zip(colorbar_axes, STATE_ORDER):
        mappable = cm.ScalarMappable(norm=norms[state], cmap=STATE_CMAPS[state])
        colorbar = figure.colorbar(mappable, cax=colorbar_axis, orientation="vertical")
        colorbar.ax.set_title(state, fontsize=10, pad=8, weight="bold")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        colorbar.set_label("Transition probability", fontsize=8, labelpad=8)
        colorbar.outline.set_edgecolor("#555555")
        colorbar.outline.set_linewidth(0.8)


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


def coefficient_table_rows(
    coefficients: Mapping[str, TransitionCoefficients],
    p_values: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[list[str], list[list[str]]]:
    row_specs = [
        ("Intercept", "intercept"),
        ("X", "x"),
        ("X^2", "x2"),
        ("Y", "y"),
        ("Y^2", "y2"),
        ("X x Y", "xy"),
        ("Conditioning Benchmark", "c"),
        ("X x Conditioning Benchmark", "xc"),
        ("Y x Conditioning Benchmark", "yc"),
    ]
    rows = []
    for label, field_name in row_specs:
        rows.append(
            [label]
            + [
                format_estimate(
                    getattr(coefficients[state], field_name),
                    p_values.get(state, {}).get(field_name) if p_values else None,
                )
                for state in STATE_ORDER
            ]
        )
    return ["", *STATE_ORDER], rows


def add_coefficient_table(
    axis: plt.Axes,
    config: FigureConfig,
    coefficients: Mapping[str, TransitionCoefficients],
    p_values: Mapping[str, Mapping[str, float]] | None,
    conditioning_value: float,
    conditioning_values: Mapping[str, float],
) -> None:
    axis.axis("off")
    col_labels, table_rows = coefficient_table_rows(coefficients, p_values)
    table = axis.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.32)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#111111")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold", color="#111111")
            cell.set_facecolor("#F2F2F2")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")

    note = (
        f"Surfaces show transitions from {config.starting_state}; "
        f"{VARIABLE_LABELS.get(config.conditioning_driver, config.conditioning_driver)} "
        f"held at {config.conditioning_level} = {conditioning_value:.3f}. "
        f"Levels: Low={conditioning_values['Low']:.3f}, "
        f"Medium={conditioning_values['Medium']:.3f}, High={conditioning_values['High']:.3f}."
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
        "Significance: * p < 0.10, ** p < 0.05, *** p < 0.01. "
        "Placeholder multinomial-logit coefficients and p-values are used; replace them with fitted HMM transition estimates before final reporting.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#333333",
        clip_on=False,
    )


def export_surface_predictions(
    output_path: Path,
    config: FigureConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surfaces: Mapping[str, np.ndarray],
    conditioning_value: float,
) -> None:
    rows = []
    for state, z_grid in surfaces.items():
        rows.append(
            pd.DataFrame(
                {
                    "starting_state": config.starting_state,
                    "to_state": state,
                    config.x_driver: x_grid.ravel(),
                    config.y_driver: y_grid.ravel(),
                    config.conditioning_driver: conditioning_value,
                    "predicted_transition_probability": z_grid.ravel(),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)


def create_transition_surface_figure(
    config: FigureConfig = FigureConfig(),
    coefficients: Mapping[str, TransitionCoefficients] = PLACEHOLDER_COEFFICIENTS,
    p_values: Mapping[str, Mapping[str, float]] | None = PLACEHOLDER_P_VALUES,
) -> list[Path]:
    setup_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if config.conditioning_level not in {"Low", "Medium", "High"}:
        raise ValueError("conditioning_level must be one of: Low, Medium, High.")

    panel = load_panel_data(config)
    x_limits = observed_range(panel, config.x_driver)
    y_limits = observed_range(panel, config.y_driver)
    conditioning_values = conditioning_levels(panel, config.conditioning_driver)
    conditioning_value = conditioning_values[config.conditioning_level]

    x_grid, y_grid = prediction_grid(x_limits, y_limits, config.grid_size)
    surfaces = compute_surfaces(config, coefficients, x_grid, y_grid, conditioning_value)
    norms = surface_norms(surfaces)

    title = (
        "Joint Effects of Transition Drivers on Predicted Transitions "
        f"from {config.starting_state}"
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
    add_colorbars(figure, colorbar_axes, norms)

    table_axis = figure.add_axes([0.045, 0.045, 0.910, 0.245])
    add_coefficient_table(
        table_axis,
        config,
        coefficients,
        p_values,
        conditioning_value,
        conditioning_values,
    )

    png_path = ANALYSIS_DIR / f"{config.output_stem}.png"
    pdf_path = ANALYSIS_DIR / f"{config.output_stem}.pdf"
    csv_path = ANALYSIS_DIR / f"{config.output_stem}_surface_predictions.csv"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    export_surface_predictions(csv_path, config, x_grid, y_grid, surfaces, conditioning_value)
    return [png_path, pdf_path, csv_path]


def main() -> None:
    paths = create_transition_surface_figure()
    print("Saved joint transition-driver 3D surface outputs:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
