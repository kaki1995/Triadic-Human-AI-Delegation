# triadic_simulation/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (directory containing this file)."""
    return Path(__file__).resolve().parent


def _data_dir() -> Path:
    """Return the active data directory after the repository folder reorganization."""
    package_data = _project_root() / "data"
    if package_data.exists():
        return package_data
    sibling_data = _project_root().parent / "Datasets" / "data"
    return sibling_data


@dataclass(frozen=True)
class SimConfig:
    """
    Configuration for the Triadic Delegation synthetic data generator.

    Conceptual role:
    - Encodes structural assumptions of the empirical setting.
    - Parameters are fixed design choices (not estimation targets).
    - Behavioral dynamics are analyzed downstream (HMM, panel models).

    Design principles:
    - Reproducibility (fixed seed, immutable config)
    - Parsimony (keep tech features fixed where not theorized)
    - Longitudinal identification (within-manager time series)
    """

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 7

    # ------------------------------------------------------------------
    # Organizational scale and time structure
    # ------------------------------------------------------------------
    n_managers: int = 875
    n_periods: int = 26  # planning cycles

    episodes_per_period_low: int = 8
    episodes_per_period_high: int = 12

    # ------------------------------------------------------------------
    # Sites / plants (context)
    # ------------------------------------------------------------------
    n_sites: int = 12

    # ------------------------------------------------------------------
    # Employee layer (execution panel)
    # ------------------------------------------------------------------
    n_employees: int = 17680

    # ------------------------------------------------------------------
    # Manager governance orientations (ex ante heterogeneity)
    # ------------------------------------------------------------------
    p_fearful: float = 0.35
    p_controlled: float = 0.40
    p_opportunistic: float = 0.25

    # ------------------------------------------------------------------
    # AI system design (contextual; can be held constant or extended)
    # ------------------------------------------------------------------
    ai_version: str = "v1"
    ai_deployment_date: str = "2017-01-01"

    autonomy_level: str = "high"  # {"low","medium","high"}
    confidence_calibration_score: float = 0.75  # 0..1

    # ------------------------------------------------------------------
    # Performance pressure environment
    # ------------------------------------------------------------------
    high_pressure_share_of_managers: float = 0.50

    # ------------------------------------------------------------------
    # Latent state structure (HMM)
    # ------------------------------------------------------------------
    n_states: int = 3

    # ------------------------------------------------------------------
    # Input / output paths
    # ------------------------------------------------------------------
    input_schema_xlsx: str = field(
        default_factory=lambda: str(_data_dir() / "Triadic_Delegation_Dataset.xlsx")
    )
    output_xlsx: str = field(
        default_factory=lambda: str(_data_dir() / "Triadic_Delegation_Dataset_SYNTH.xlsx")
    )
    output_analysis_xlsx: str = field(
        default_factory=lambda: str(_data_dir() / "Triadic_Delegation_Dataset_SYNTH_ANALYSIS.xlsx")
    )

    # Columns to drop from the ANALYSIS export (true latent states only)
    analysis_drop_cols: list[str] = field(default_factory=lambda: [
        "latent_state_true",
        "latent_state_true_next",
    ])

    def __post_init__(self) -> None:
        if self.n_managers <= 0:
            raise ValueError("n_managers must be positive.")
        if self.n_periods <= 0:
            raise ValueError("n_periods must be positive.")
        if self.n_employees < self.n_managers:
            raise ValueError("n_employees must be at least n_managers.")
        if self.episodes_per_period_low <= 0:
            raise ValueError("episodes_per_period_low must be positive.")
        if self.episodes_per_period_low > self.episodes_per_period_high:
            raise ValueError("episodes_per_period_low cannot exceed episodes_per_period_high.")
