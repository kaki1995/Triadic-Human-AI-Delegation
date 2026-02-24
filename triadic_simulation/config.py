# triadic_sim/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _project_root() -> Path:
    """Return the project root (directory containing this file)."""
    return Path(__file__).resolve().parent


def _default_transparency_schedule(n_periods: int) -> List[int]:
    """
    Default 4-level rollout across the horizon: 0 -> 1 -> 2 -> 3.
    Evenly distributes periods across 4 buckets (differences at most 1 period).

    Example for n_periods=26:
      [0]*7 + [1]*7 + [2]*6 + [3]*6
    """
    n = int(n_periods)
    if n <= 0:
        return []
    base = n // 4
    rem = n % 4
    sizes = [base + (1 if i < rem else 0) for i in range(4)]  # sums to n
    sched: List[int] = []
    for lvl, sz in enumerate(sizes):
        sched.extend([lvl] * sz)
    return sched


@dataclass(frozen=True)
class SimConfig:
    """
    Configuration for the Triadic Delegation synthetic data generator.
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

    # ------------------------------------------------------------------
    # Employee layer (execution panel)
    # ------------------------------------------------------------------
    employees_per_manager_low: int = 3
    employees_per_manager_high: int = 8

    # ------------------------------------------------------------------
    # Manager governance orientations (ex ante heterogeneity)
    # ------------------------------------------------------------------
    p_fearful: float = 0.35
    p_controlled: float = 0.40
    p_opportunistic: float = 0.25

    # ------------------------------------------------------------------
    # AI transparency (UPDATED: 4-level time-varying design feature)
    # ------------------------------------------------------------------
    # 0 = none (black box)
    # 1 = basic (inputs + short rationale)
    # 2 = drivers + confidence/uncertainty
    # 3 = process-level details (model type/training basis/constraints/guardrails)
    #
    # Must have length == n_periods and values in {0,1,2,3}.
    # Default is a stepwise rollout 0->1->2->3 across n_periods.
    transparency_schedule: List[int] = field(default_factory=lambda: _default_transparency_schedule(26))
    transparency_shift_period: int = 13          # period (1-indexed) at which explanation capability upgrades
    explanation_capability_pre: str = "none"    # before shift
    explanation_capability_post: str = "detailed" # on/after shift

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
        default_factory=lambda: str(_project_root() / "data" / "Triadic_Delegation_Dataset.xlsx")
    )
    output_xlsx: str = field(
        default_factory=lambda: str(_project_root() / "data" / "Triadic_Delegation_Dataset_SYNTH.xlsx")
    )
    output_analysis_xlsx: str = field(
        default_factory=lambda: str(_project_root() / "data" / "Triadic_Delegation_Dataset_SYNTH_ANALYSIS.xlsx")
    )

    # Columns to drop from the ANALYSIS export (true latent states only)
    analysis_drop_cols: List[str] = field(default_factory=lambda: [
        "latent_state",
        "latent_state_next",
    ])

    def __post_init__(self) -> None:
        # Validate schedule length
        if len(self.transparency_schedule) != self.n_periods:
            raise ValueError(
                f"transparency_schedule must have length n_periods={self.n_periods}, "
                f"but got {len(self.transparency_schedule)}."
            )
        # Validate schedule values
        bad = [x for x in self.transparency_schedule if int(x) < 0 or int(x) > 3]
        if bad:
            raise ValueError(f"transparency_schedule values must be in 0..3, but found: {bad}")