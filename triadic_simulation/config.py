# triadic_sim/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (directory containing this file)."""
    return Path(__file__).resolve().parent


@dataclass(frozen=True)
class SimConfig:
    """
    Configuration for the Triadic Delegation synthetic data generator.

    Conceptual role:
    - This file encodes the *structural assumptions* of the empirical setting.
    - Parameters are treated as fixed design choices, not estimation targets.
    - Behavioral dynamics are analyzed downstream (HMM, panel models).

    Design principles:
    - Reproducibility (fixed seed, immutable config)
    - Parsimony (hold technology features constant where not theorized)
    - Longitudinal identification (rich within-manager time series)
    
    NOTE:
    - This configuration defines the simulated empirical setting.
    - Parameters are not tuned to generate results and are held fixed across runs.

    """

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    seed: int = 7
    # Fixed random seed to ensure full reproducibility of the synthetic dataset.
    
    # ------------------------------------------------------------------
    # Organizational scale and time structure
    # ------------------------------------------------------------------

    n_managers: int = 120
    # Number of distinct managers (cross-sectional units).


    n_periods: int = 26
    # Number of decision periods per manager.
    # Each period corresponds to one operational cycle of ~3 weeks,
    # yielding approximately 1.5 years of longitudinal data.

    episodes_per_period_low: int = 30
    episodes_per_period_high: int = 60
    # Number of decision episodes per manager within each period.

    # ------------------------------------------------------------------
    # Manager governance orientations (ex ante heterogeneity)
    # ------------------------------------------------------------------

    p_fearful: float = 0.35
    p_controlled: float = 0.40
    p_opportunistic: float = 0.25
    # Probability distribution over initial governance orientations.
    # These represent *baseline managerial styles*, not outcomes.

    # ------------------------------------------------------------------
    # AI transparency intervention (time-varying design feature)
    # ------------------------------------------------------------------

    transparency_shift_period: int = 13
    # Period at which the AI system's explanation capability changes.

    explanation_capability_pre: str = "none"
    explanation_capability_post: str = "detailed"
    # Level of AI explainability before and after the intervention.


    # ------------------------------------------------------------------
    # AI system design (held constant; not analytically varied)
    # ------------------------------------------------------------------

    autonomy_level: str = "high"
    # AI autonomy is treated as a fixed contextual condition.
    # High autonomy enables meaningful delegation of decision authority,
    # Autonomy is not analyzed as an explanatory variable in this study.

    confidence_calibration_score: float = 0.75
    # Degree to which AI confidence is well calibrated (0–1).
    # Influences error attribution and learning signals received by managers.

    # ------------------------------------------------------------------
    # Performance pressure environment
    # ------------------------------------------------------------------

    high_pressure_share_of_managers: float = 0.50
    # Share of managers operating under high KPI pressure.
    # Pressure affects sensitivity to performance feedback and overrides,

    # ------------------------------------------------------------------
    # Latent state structure (HMM)
    # ------------------------------------------------------------------

    n_states: int = 3
    # Number of latent willingness-to-delegate states.
    # Interpreted as low, medium, and high delegation willingness.

    # ------------------------------------------------------------------
    # Input / output paths
    # ------------------------------------------------------------------

    input_schema_xlsx: str = field(
        default_factory=lambda: str(_project_root() / "data" / "Triadic_Delegation_Dataset.xlsx")
    )
    # Excel file defining the dataset schema (tables, variables, data types).

    output_xlsx: str = field(
        default_factory=lambda: str(_project_root() / "data" / "Triadic_Delegation_Dataset_SYNTH.xlsx")
    )
    # Output file for the generated synthetic dataset.
