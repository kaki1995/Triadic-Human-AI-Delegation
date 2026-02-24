# triadic_sim/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _project_root() -> Path:
    """Return the project root (directory containing this file)."""
    return Path(__file__).resolve().parent


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

    NOTE:
    - This configuration defines the simulated empirical setting.
    - Parameters are not tuned to generate desired results.
    """

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 7

    # ------------------------------------------------------------------
    # Organizational scale and time structure
    # ------------------------------------------------------------------
    n_managers: int = 120
    n_periods: int = 26  # ~3-week cycles -> ~1.5 years

    episodes_per_period_low: int = 30
    episodes_per_period_high: int = 60

    # ------------------------------------------------------------------
    # Sites / plants (context)
    # ------------------------------------------------------------------
    n_sites: int = 12
    # Number of sites/plants used to generate site_master and attach site_id to managers/employees/episodes.

    # ------------------------------------------------------------------
    # Employee layer (execution panel)
    # ------------------------------------------------------------------
    employees_per_manager_low: int = 3
    employees_per_manager_high: int = 8
    # Each manager supervises k employees sampled uniformly in [low, high].

    # ------------------------------------------------------------------
    # Manager governance orientations (ex ante heterogeneity)
    # ------------------------------------------------------------------
    p_fearful: float = 0.35
    p_controlled: float = 0.40
    p_opportunistic: float = 0.25

    # ------------------------------------------------------------------
    # AI transparency intervention (time-varying design feature)
    # ------------------------------------------------------------------
    transparency_shift_period: int = 13
    explanation_capability_pre: str = "none"
    explanation_capability_post: str = "detailed"
    # Allowed values should align with simulator mappings: {"none","basic","detailed"}.

    # ------------------------------------------------------------------
    # AI system design (contextual; can be held constant or extended)
    # ------------------------------------------------------------------
    ai_version: str = "v1"
    ai_deployment_date: str = "2017-01-01"
    # Stored in ai_system_master; ai_version also appears in decision_episode and panel_manager_period.

    autonomy_level: str = "high"
    # Allowed values: {"low","medium","high"}.

    confidence_calibration_score: float = 0.75
    # 0..1; higher = better calibrated confidence.

    # ------------------------------------------------------------------
    # Performance pressure environment
    # ------------------------------------------------------------------
    high_pressure_share_of_managers: float = 0.50
    # Share of managers operating under high KPI pressure.

    # ------------------------------------------------------------------
    # Latent state structure (HMM)
    # ------------------------------------------------------------------
    n_states: int = 3
    # Low / medium / high willingness-to-delegate states.

    # ------------------------------------------------------------------
    # Input / output paths
    # ------------------------------------------------------------------
    input_schema_xlsx: str = field(
        default_factory=lambda: str(_project_root() / "data" / "Triadic_Delegation_Dataset.xlsx")
    )
    output_xlsx: str = field(
        default_factory=lambda: str(_project_root() / "data" / "Triadic_Delegation_Dataset_SYNTH.xlsx")
    )
    output_analysis_xlsx: str = field(
        default_factory=lambda: str(_project_root() / "data" / "Triadic_Delegation_Dataset_SYNTH_ANALYSIS.xlsx")
    )

    # NEW: Columns to drop from the ANALYSIS export (true latent states only)
    analysis_drop_cols: List[str] = field(default_factory=lambda: [
        "latent_state",
        "latent_state_next",
    ])    