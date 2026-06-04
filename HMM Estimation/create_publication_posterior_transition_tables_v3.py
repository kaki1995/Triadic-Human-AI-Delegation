from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts" / "Analysis_v3"
MATRIX_PATH = ANALYSIS_DIR / "posterior_transition_by_driver_level_matrices_v3.csv"

OUTPUT_STEM = "posterior_transition_driver_matrices_publication_v3"

STATE_ORDER = ["Aversion", "Neutral", "Appreciation"]
LEVEL_ORDER = ["Low", "Medium", "High"]
DRIVER_ORDER = [
    "team_t_minus_1_vs_team_t",
    "team_vs_peer_average",
    "target_attainment",
]
DRIVER_LABELS = {
    "team_t_minus_1_vs_team_t": "Team (t-1) vs. Team (t)",
    "team_vs_peer_average": "Team vs. Peer Average",
    "target_attainment": "Target Attainment",
}


def save_figure(fig: plt.Figure, path: Path, **kwargs) -> Path:
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fig.savefig(fallback, **kwargs)
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


def matrix_cell(
    matrix: pd.DataFrame,
    driver: str,
    level: str,
    from_state: str,
    to_state: str,
) -> str:
    match = matrix[
        matrix["driver"].eq(driver)
        & matrix["driver_level"].eq(level)
        & matrix["state_at_t"].eq(from_state)
    ]
    if match.empty:
        return "--"
    value = str(match.iloc[0][to_state])
    return value if value and value != "nan" else "--"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Times New Roman",
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def draw_publication_table(matrix: pd.DataFrame) -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(18.0, 10.6), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    left = 0.035
    right = 0.985
    title_y = 0.965
    table_top = 0.925
    table_bottom = 0.075
    label_col_w = 0.085
    block_gap = 0.030
    block_w = (right - left - label_col_w - 2 * block_gap) / 3
    col_w = block_w / 3

    ax.text(left, title_y, "Table 4", fontsize=16, weight="bold", ha="left", va="center")
    ax.text(
        left + 0.078,
        title_y,
        "The Mean Posterior Transition Matrices by Transition Driver",
        fontsize=16,
        weight="bold",
        ha="left",
        va="center",
    )
    ax.plot([left, right], [table_top, table_top], color="#111111", lw=0.9)

    panel_h = (table_top - table_bottom) / 3
    for panel_idx, level in enumerate(LEVEL_ORDER):
        y_top = table_top - panel_idx * panel_h
        y_bottom = y_top - panel_h + 0.010
        panel_label_y = y_top - 0.018
        block_label_y = y_top - 0.052
        t_label_y = y_top - 0.087
        column_y = y_top - 0.126
        row_ys = [y_top - 0.172, y_top - 0.217, y_top - 0.252]

        ax.text(
            left,
            panel_label_y,
            f"Panel {chr(65 + panel_idx)}. {level} transition-driver level",
            fontsize=11.5,
            weight="bold",
            ha="left",
            va="center",
        )
        ax.plot([left, right], [y_top - 0.033, y_top - 0.033], color="#444444", lw=0.55)

        ax.text(left, column_y, r"$t-1$", fontsize=12, style="italic", ha="left", va="center")
        for row_y, state in zip(row_ys, STATE_ORDER):
            ax.text(left, row_y, state, fontsize=10.5, ha="left", va="center")

        for driver_idx, driver in enumerate(DRIVER_ORDER):
            block_left = left + label_col_w + driver_idx * (block_w + block_gap)
            block_right = block_left + block_w
            block_center = (block_left + block_right) / 2

            ax.text(
                block_center,
                block_label_y,
                DRIVER_LABELS[driver],
                fontsize=12.2,
                weight="bold",
                ha="center",
                va="center",
            )
            ax.plot([block_left, block_right], [block_label_y - 0.021, block_label_y - 0.021], color="#111111", lw=0.7)
            ax.text(block_center, t_label_y, r"$t$", fontsize=12, style="italic", ha="center", va="center")
            ax.plot([block_left, block_right], [t_label_y - 0.024, t_label_y - 0.024], color="#444444", lw=0.6)

            for col_idx, to_state in enumerate(STATE_ORDER):
                x = block_left + (col_idx + 0.5) * col_w
                ax.text(x, column_y, to_state, fontsize=10.2, ha="center", va="center")

            for row_y, from_state in zip(row_ys, STATE_ORDER):
                for col_idx, to_state in enumerate(STATE_ORDER):
                    x = block_left + (col_idx + 0.5) * col_w
                    value = matrix_cell(matrix, driver, level, from_state, to_state)
                    ax.text(
                        x,
                        row_y,
                        value,
                        fontsize=10.4,
                        weight="bold" if value != "--" else "normal",
                        ha="center",
                        va="center",
                    )

        ax.plot([left, right], [y_bottom, y_bottom], color="#111111", lw=0.7)

    note = (
        "Note: cells report mean manager-level decoded posterior transition rates. "
        "Low/Medium/High levels are terciles of each continuous driver; Target Attainment is binary, "
        "so no Medium panel is observed for that driver."
    )
    ax.text(left, 0.040, note, fontsize=9.2, ha="left", va="center")
    return fig


def markdown_report(matrix: pd.DataFrame) -> str:
    lines = [
        "# Table 4. The Mean Posterior Transition Matrices by Transition Driver",
        "",
        "Cells report mean manager-level decoded posterior transition rates.",
        "",
    ]
    for level in LEVEL_ORDER:
        lines.extend([f"## {level} transition-driver level", ""])
        rows = []
        for from_state in STATE_ORDER:
            row: dict[str, str] = {"t-1": from_state}
            for driver in DRIVER_ORDER:
                for to_state in STATE_ORDER:
                    row[f"{DRIVER_LABELS[driver]} | {to_state}"] = matrix_cell(
                        matrix, driver, level, from_state, to_state
                    )
            rows.append(row)
        lines.extend([pd.DataFrame(rows).to_markdown(index=False), ""])
    lines.append(
        "Note: Low/Medium/High levels are terciles of each continuous driver; "
        "Target Attainment is binary, so no Medium panel is observed for that driver."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    matrix = pd.read_csv(MATRIX_PATH)
    fig = draw_publication_table(matrix)
    outputs = [
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
        save_text(markdown_report(matrix), ANALYSIS_DIR / f"{OUTPUT_STEM}.md"),
    ]
    plt.close(fig)

    print("Saved publication-style posterior transition table outputs:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
