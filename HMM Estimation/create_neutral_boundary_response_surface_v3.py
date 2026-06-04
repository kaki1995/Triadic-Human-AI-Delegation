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
PANEL_PATH = PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
OUTPUT_CSV = ANALYSIS_DIR / "neutral_boundary_response_surface_v3.csv"
OUTPUT_PNG = ANALYSIS_DIR / "figure_neutral_boundary_response_surface_v3.png"

PEER_VAR = "team_vs_peer_average"
TARGET_VAR = "target_attainment"
HOLD_AT_MEAN_VAR = "team_t_minus_1_vs_team_t"
HOLD_AT_MEAN_LABEL = "Team (t-1) vs. Team (t)"
ORDERED_LABELS = ["Aversion", "Neutral", "Appreciation"]
SURFACE_COLORS = {
    "Aversion": "#111111",
    "Neutral": "#FF6A00",
    "Appreciation": "#0057D9",
}
SURFACE_CMAPS = {
    state: LinearSegmentedColormap.from_list(f"{state}_surface", ["#ffffff", color])
    for state, color in SURFACE_COLORS.items()
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


def load_model() -> tuple[dict[str, object], Params]:
    patch_pandas_stringarray_pickle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MODEL_PATH.open("rb") as stream:
            artifacts = pickle.load(stream)
    return artifacts, artifacts["best_model"]


def ordered_raw_states(artifacts: dict[str, object], model: Params) -> list[int]:
    emission_cols = list(artifacts["emission_cols"])
    authority_idx = emission_cols.index("ai_authority_share")
    mu_original = artifacts["y_scaler"].inverse_transform(model.mu)
    return np.argsort(mu_original[:, authority_idx]).tolist()


def build_surface_frame(
    artifacts: dict[str, object],
    model: Params,
    panel: pd.DataFrame,
    n_points: int = 61,
) -> pd.DataFrame:
    transition_cols = list(artifacts["transition_cols"])
    x_scaler = artifacts["x_scaler"]
    ordered_states = ordered_raw_states(artifacts, model)
    raw_neutral = ordered_states[ORDERED_LABELS.index("Neutral")]
    peer_idx = transition_cols.index(PEER_VAR)
    target_idx = transition_cols.index(TARGET_VAR)

    peer_values = np.linspace(panel[PEER_VAR].min(), panel[PEER_VAR].max(), n_points)
    target_values = np.linspace(panel[TARGET_VAR].min(), panel[TARGET_VAR].max(), n_points)
    peer_grid, target_grid = np.meshgrid(peer_values, target_values)

    x_original = np.tile(x_scaler.mean_, (peer_grid.size, 1))
    x_original[:, peer_idx] = peer_grid.ravel()
    x_original[:, target_idx] = target_grid.ravel()
    x_standardized = x_scaler.transform(x_original)

    logits = model.alpha[raw_neutral][None, :] + (
        x_standardized @ model.beta[raw_neutral].T
    )
    probabilities = softmax(logits, axis=1)[:, ordered_states]

    rows = []
    for destination_idx, destination_label in enumerate(ORDERED_LABELS):
        rows.append(
            pd.DataFrame(
                {
                    "from_state": "Neutral",
                    "to_state": destination_label,
                    PEER_VAR: peer_grid.ravel(),
                    TARGET_VAR: target_grid.ravel(),
                    "transition_probability": probabilities[:, destination_idx],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_surfaces(surface: pd.DataFrame) -> None:
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
    fig = plt.figure(figsize=(17.0, 5.5), dpi=220)

    for panel_idx, destination in enumerate(ORDERED_LABELS, start=1):
        ax = fig.add_subplot(1, 3, panel_idx, projection="3d")
        sub = surface[surface["to_state"].eq(destination)]
        peer_values = np.sort(sub[PEER_VAR].unique())
        target_values = np.sort(sub[TARGET_VAR].unique())
        x, y = np.meshgrid(peer_values, target_values)
        z = (
            sub.pivot(index=TARGET_VAR, columns=PEER_VAR, values="transition_probability")
            .reindex(index=target_values, columns=peer_values)
            .to_numpy()
        )

        ax.plot_surface(
            x,
            y,
            100.0 * z,
            cmap=SURFACE_CMAPS[destination],
            linewidth=0,
            antialiased=True,
            alpha=0.96,
        )
        ax.set_title(
            f"({chr(96 + panel_idx)}) Neutral to {destination}",
            fontsize=13,
            pad=10,
        )
        ax.set_xlabel("Peer Average", labelpad=8)
        ax.set_ylabel("Target Attainment", labelpad=8)
        ax.set_zlabel("Probability (%)", labelpad=8)
        ax.zaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_zlim(0, 100)
        ax.view_init(elev=27, azim=-128)
        ax.grid(True, alpha=0.22)
        ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        ax.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))

    fig.suptitle(
        "Neutral Boundary Response Surface: Peer Average and Target Attainment",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        (
            f"Predicted transition probabilities from the fitted HMM; "
            f"{HOLD_AT_MEAN_LABEL} held at its sample mean."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.11, top=0.88, wspace=0.04)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts, model = load_model()
    panel = pd.read_excel(PANEL_PATH, sheet_name="panel_manager_period")
    surface = build_surface_frame(artifacts, model, panel)
    surface.to_csv(OUTPUT_CSV, index=False)
    plot_surfaces(surface)
    print(f"Saved surface data to {OUTPUT_CSV}")
    print(f"Saved figure to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
