# triadic_sim/config.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SimConfig:
    """
    Configuration for the Triadic Delegation synthetic data generator.

    Notes:
    - Keep all tunable simulation parameters here (scale, probabilities, shifts, paths).
    - Treat as immutable (frozen=True) so runs are reproducible & auditable.
    """

    # Reproducibility
    seed: int = 7

    # Scale
    n_managers: int = 120
    n_periods: int = 40
    episodes_per_period_low: int = 30
    episodes_per_period_high: int = 60

    # Governance mode mixture (must sum to 1.0)
    p_fearful: float = 0.35
    p_controlled: float = 0.40
    p_opportunistic: float = 0.25

    # Transparency shift (period index starting at 1)
    transparency_shift_period: int = 20
    explanation_capability_pre: str = "none"       # none/basic/detailed
    explanation_capability_post: str = "detailed"  # none/basic/detailed

    # AI design
    autonomy_level: str = "high"                  # low/medium/high
    confidence_calibration_score: float = 0.75    # 0..1

    # Pressure scenario assignment
    high_pressure_share_of_managers: float = 0.50

    # Latent states (low/med/high willingness)
    n_states: int = 3

    # I/O paths
    input_schema_xlsx: str = "/mnt/data/Triadic_Delegation_Dataset.xlsx"
    output_xlsx: str = "/mnt/data/Triadic_Delegation_Dataset_SYNTH.xlsx"
