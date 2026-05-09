# triadic_simulation/simulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .config import SimConfig
    from .utils import clip01, choice_with_probs
except ImportError:
    from config import SimConfig
    from utils import clip01, choice_with_probs


# -----------------------------
# Entities
# -----------------------------
@dataclass
class ManagerProfile:
    """A single manager entity with stable traits + dynamic latent state (willingness-to-delegate)."""
    manager_id: str
    governance_mode: str                  # fearful_exclusion / controlled_opening / opportunistic_teaming
    baseline_ai_attitude: float           # -1..+1
    risk_aversion_index: float            # 0..1
    high_pressure: bool
    org_unit_id: str
    site_id: str
    state: int                            # 0..n_states-1
    transition_random_effect_xi: float     # manager random intercept in transition propensity


@dataclass
class EmployeeProfile:
    """Employee working under a manager (can execute tasks with/without AI support)."""
    employee_id: str
    manager_id: str
    site_id: str
    role: str
    experience_years: float
    ai_familiarity: float                 # 0..1
    task_specialization: str


# -----------------------------
# Helper sampling functions
# -----------------------------
def sample_site_master(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create site_master:
      site_id, region, plant_type, automation_level, baseline_operational_complexity
    """
    n_sites = getattr(cfg, "n_sites", 12)
    regions = ["DE-North", "DE-South", "DE-West", "DE-East", "EU-Other"]
    plant_types = ["assembly", "logistics_hub", "distribution_center", "supplier_dc"]

    rows: List[Dict[str, object]] = []
    for s in range(1, n_sites + 1):
        site_id = f"S{s:03d}"
        rows.append(
            dict(
                site_id=site_id,
                region=str(rng.choice(regions)),
                plant_type=str(rng.choice(plant_types)),
                automation_level=float(np.clip(rng.normal(0.55, 0.18), 0.0, 1.0)),
                baseline_operational_complexity=float(np.clip(rng.beta(2.2, 2.2), 0.0, 1.0)),
            )
        )
    return pd.DataFrame(rows)


def sample_manager_profiles(cfg: SimConfig, rng: np.random.Generator, site_ids: List[str]) -> List[ManagerProfile]:
    """Create manager population with governance-mode mixture and correlated traits + site assignment."""
    modes = ["fearful_exclusion", "controlled_opening", "opportunistic_teaming"]
    p = np.array([cfg.p_fearful, cfg.p_controlled, cfg.p_opportunistic], dtype=float)
    p = p / p.sum()

    profiles: List[ManagerProfile] = []
    for i in range(cfg.n_managers):
        manager_id = f"M{i+1:04d}"
        gov = str(rng.choice(modes, p=p))

        mu = {"fearful_exclusion": -0.6, "controlled_opening": -0.1, "opportunistic_teaming": 0.5}[gov]
        baseline_ai_attitude = float(np.clip(rng.normal(mu, 0.35), -1.0, 1.0))

        rmu = {"fearful_exclusion": 0.75, "controlled_opening": 0.55, "opportunistic_teaming": 0.35}[gov]
        risk_aversion_index = float(np.clip(rng.normal(rmu, 0.15), 0.0, 1.0))

        high_pressure = bool(rng.random() < cfg.high_pressure_share_of_managers)
        org_unit_id = f"OU{rng.integers(1, 9):02d}"
        site_id = str(rng.choice(site_ids))

        # Initial latent state from attitude.
        if cfg.n_states >= 4:
            if baseline_ai_attitude < -0.45:
                state = 0
            elif baseline_ai_attitude < -0.10:
                state = 1
            elif baseline_ai_attitude < 0.35:
                state = 2
            else:
                state = 3
        else:
            if baseline_ai_attitude < -0.25:
                state = 0
            elif baseline_ai_attitude < 0.40:
                state = 1
            else:
                state = 2
        transition_random_effect_xi = float(rng.normal(0.0, cfg.manager_heterogeneity_xi))

        profiles.append(
            ManagerProfile(
                manager_id=manager_id,
                governance_mode=gov,
                baseline_ai_attitude=baseline_ai_attitude,
                risk_aversion_index=risk_aversion_index,
                high_pressure=high_pressure,
                org_unit_id=org_unit_id,
                site_id=site_id,
                state=state,
                transition_random_effect_xi=transition_random_effect_xi,
            )
        )
    return profiles


def sample_employee_master(cfg: SimConfig, rng: np.random.Generator, managers: List[ManagerProfile]) -> List[EmployeeProfile]:
    """
    Create employee_master and a simple reporting relation:
      employees are distributed exactly across managers and inherit manager_id and site_id.
    """
    roles = ["planner", "coordinator", "dispatcher", "inventory_analyst"]
    specs = ["inbound", "outbound", "crossdock", "perishables", "high_value"]

    rows: List[EmployeeProfile] = []
    emp_counter = 0

    n_employees = int(getattr(cfg, "n_employees", len(managers)))
    base_team_size = n_employees // len(managers)
    remainder = n_employees % len(managers)

    for i, m in enumerate(managers):
        k = base_team_size + (1 if i < remainder else 0)
        for _ in range(k):
            emp_counter += 1
            employee_id = f"E{emp_counter:06d}"
            role = str(rng.choice(roles))
            experience_years = float(np.clip(rng.normal(5.5, 2.2), 0.0, 25.0))

            fam_mu = {"fearful_exclusion": 0.35, "controlled_opening": 0.50, "opportunistic_teaming": 0.65}[m.governance_mode]
            ai_familiarity = float(np.clip(rng.normal(fam_mu, 0.18), 0.0, 1.0))

            rows.append(
                EmployeeProfile(
                    employee_id=employee_id,
                    manager_id=m.manager_id,
                    site_id=m.site_id,
                    role=role,
                    experience_years=experience_years,
                    ai_familiarity=ai_familiarity,
                    task_specialization=str(rng.choice(specs)),
                )
            )
    return rows


def period_context(
    cfg: SimConfig,
    rng: np.random.Generator,
    manager: ManagerProfile,
    previous_scores: List[float],
    period_id: int,
) -> Dict[str, float]:
    denom = max(int(cfg.n_periods) - 1, 1)
    period_progress = float((period_id - 1) / denom)
    period_wave = float(np.sin(2.0 * np.pi * period_progress))

    prior_score = float(np.mean(previous_scores[-3:])) if previous_scores else float(
        np.clip(0.48 + 0.08 * manager.state + rng.normal(0, 0.04), 0.0, 1.0)
    )
    kpi_target = float(
        np.clip(
            rng.normal(
                0.62
                + 0.04 * period_progress
                + (0.08 if manager.high_pressure else 0.0),
                0.05,
            ),
            0.35,
            0.95,
        )
    )
    performance_pressure = float(np.clip(kpi_target - prior_score, 0.0, 1.0))
    target_gap = float(abs(kpi_target - prior_score))
    demand_volatility = float(
        np.clip(
            0.14
            + 0.58 * rng.beta(2.0, 3.2)
            + 0.06 * rng.random()
            + 0.07 * period_progress
            + 0.035 * period_wave
            + rng.normal(0.0, 0.035),
            0.02,
            0.98,
        )
    )
    independent_difficulty_load = float(rng.beta(2.1, 2.7))
    independent_complexity_load = float(rng.beta(2.0, 2.0))
    independent_disruption_load = float(rng.beta(1.4, 3.2))
    target_difficulty = float(
        np.clip(
            0.34 * target_gap
            + 0.13 * demand_volatility
            + 0.18 * independent_difficulty_load
            + 0.07 * period_progress
            + rng.normal(0.0, 0.035),
            0.02,
            0.90,
        )
    )
    task_complexity = float(np.clip(1 + rng.poisson(2.2 + 3.0 * demand_volatility + 3.5 * independent_complexity_load), 1, 15))
    task_complexity_index = task_complexity / 15.0
    supply_disruption_count = float(rng.poisson(0.12 + 0.85 * demand_volatility + 1.10 * independent_disruption_load))

    forecast_accuracy_mape = float(np.clip(rng.normal(0.19 + 0.12 * demand_volatility + 0.04 * rng.beta(2.0, 3.0), 0.085), 0.04, 0.62))
    forecast_accuracy = 1.0 - forecast_accuracy_mape

    shock_prob = 0.045 + 0.07 * demand_volatility + 0.055 * independent_disruption_load
    recent_negative_shock = float(int(rng.random() < shock_prob))

    return {
        "kpi_target": kpi_target,
        "prior_composite_kpi_score": prior_score,
        "performance_pressure": performance_pressure,
        "target_difficulty": target_difficulty,
        "demand_volatility": demand_volatility,
        "task_complexity": task_complexity,
        "task_complexity_index": task_complexity_index,
        "supply_disruption_count": supply_disruption_count,
        "forecast_accuracy_mape": forecast_accuracy_mape,
        "forecast_accuracy": forecast_accuracy,
        "recent_negative_shock": recent_negative_shock,
        "period_id": float(period_id),
        "period_progress": period_progress,
        "period_wave": period_wave,
    }


def sample_ai_confidence(cfg: SimConfig, rng: np.random.Generator, ctx: Dict[str, float], site_complexity: float) -> float:
    complexity = ctx["task_complexity_index"]
    volatility = ctx["demand_volatility"]

    mu = 0.80 - 0.22 * complexity - 0.18 * volatility - 0.12 * site_complexity
    mu = float(np.clip(mu, 0.12, 0.95))

    k = 8 + 20 * cfg.confidence_calibration_score
    a = max(1.0, mu * k)
    b = max(1.0, (1 - mu) * k)
    return float(rng.beta(a, b))


def sample_ai_uncertainty(rng: np.random.Generator, ai_confidence: float) -> float:
    u = 1.0 - ai_confidence
    return float(np.clip(rng.normal(u, 0.06), 0.0, 1.0))


def episode_decision_probabilities(
    cfg: SimConfig,
    manager: ManagerProfile,
    ctx: Dict[str, float],
    ai_confidence: float,
) -> Tuple[float, float, float]:
    """
    Compute (p_accept, p_modify, p_reject) for a single recommendation episode.
      - Latent state s controls baseline willingness (accept increases with s)
      - Pressure/risk/shock push toward modify/reject (especially at low s)
    """
    s = manager.state
    pressure = ctx["performance_pressure"]
    shock = ctx["recent_negative_shock"]
    risk = manager.risk_aversion_index
    complexity = ctx["task_complexity_index"]
    volatility = ctx["demand_volatility"]
    progress = ctx.get("period_progress", 0.0)
    wave = ctx.get("period_wave", 0.0)
    forecast_signal = (ctx["forecast_accuracy"] - 0.70) / 0.08
    supply_signal = ctx["supply_disruption_count"]
    difficulty = ctx["target_difficulty"]

    if cfg.n_states >= 4:
        base_accept_by_state = [0.28, 0.42, 0.58, 0.72]
        base_reject_by_state = [0.48, 0.35, 0.20, 0.08]
        drift_by_state = [-1.00, -0.35, 0.45, 1.00]
        wave_by_state = [-0.30, 0.15, -0.10, 0.30]
    else:
        base_accept_by_state = [0.42, 0.58, 0.68]
        base_reject_by_state = [0.42, 0.25, 0.10]
        drift_by_state = [-0.50, 0.18, 0.68]
        wave_by_state = [-0.24, -0.06, 0.18]

    base_accept = base_accept_by_state[s]
    base_reject = base_reject_by_state[s]
    temporal_shift = (
        cfg.authority_common_time_shift * progress
        + cfg.authority_time_drift_strength * drift_by_state[s] * progress
        + cfg.time_wave_strength * wave_by_state[s] * wave
    )

    conf_effect = 0.20 * (ai_confidence - 0.5)

    pressure_penalty = ((0.25 if s == 0 else 0.12) * pressure * risk) + 0.240 * pressure
    shock_penalty = (0.24 if s <= 1 else 0.12) * shock
    context_penalty = (
        -0.085 * forecast_signal
        + 0.070 * supply_signal
        + 0.430 * difficulty
        + 0.240 * complexity
        + 0.180 * volatility
    )

    accept = clip01(base_accept + temporal_shift + conf_effect - pressure_penalty - shock_penalty - context_penalty)
    reject = clip01(base_reject - 0.65 * temporal_shift + pressure_penalty + shock_penalty + 0.55 * context_penalty - 0.5 * conf_effect)
    modify = clip01(1.0 - accept - reject)

    if manager.governance_mode == "controlled_opening":
        modify = clip01(modify + 0.05)
        z = accept + reject + modify
        accept, reject, modify = accept / z, reject / z, modify / z

    z = accept + modify + reject
    return accept / z, modify / z, reject / z


def escalation_probability(
    cfg: SimConfig,
    manager: ManagerProfile,
    ctx: Dict[str, float],
    ai_confidence: float,
) -> float:
    """
    Escalation probability to employees for review, validation, local input, or execution support.
    """
    autonomy_base = {"low": 0.06, "medium": 0.10, "high": 0.15}[cfg.autonomy_level]
    low_conf = max(0.0, 0.6 - ai_confidence)
    forecast_signal = (ctx["forecast_accuracy"] - 0.70) / 0.08
    complexity = ctx["task_complexity_index"]
    volatility = ctx["demand_volatility"]
    if cfg.n_states >= 4:
        state_factor = [1.08, 1.04, 1.02, 1.10][manager.state]
        drift_by_state = [0.24, 0.30, 0.36, 0.44][manager.state]
    else:
        state_factor = [1.06, 1.00, 1.10][manager.state]
        drift_by_state = [0.58, -0.48, -0.45][manager.state]
    progress = ctx.get("period_progress", 0.0)
    temporal_shift = (
        cfg.escalation_common_time_shift * progress
        + cfg.escalation_time_drift_strength * drift_by_state * progress
    )
    context_shift = (
        -0.065 * forecast_signal
        + 0.220 * ctx["performance_pressure"]
        + 0.055 * ctx["supply_disruption_count"]
        + 0.330 * ctx["target_difficulty"]
        + 0.105 * ctx["recent_negative_shock"]
        + 0.140 * complexity
        + 0.120 * volatility
    )
    return float(np.clip((autonomy_base + 0.38 * low_conf) * state_factor + temporal_shift + context_shift, 0.045, 0.60))


def _std_signal(value: float, center: float, scale: float, limit: float = 2.25) -> float:
    """Bounded standardization for DGP-level manager-period control signals."""
    if scale <= 0:
        return 0.0
    return float(np.clip((value - center) / scale, -limit, limit))


def calibrate_panel_emissions(
    cfg: SimConfig,
    rng: np.random.Generator,
    manager: ManagerProfile,
    ctx: Dict[str, float],
    ai_authority_share: float,
    escalation_share: float,
    decision_latency: float,
) -> Tuple[float, float]:
    """
    Add a transparent manager-period calibration layer to the synthetic
    emission shares. It keeps the raw episode-level behavior as the starting
    point while making every pre-specified emission control identifiable in the
    downstream HMM.
    """
    progress = float(ctx.get("period_progress", 0.0))
    strength = float(getattr(cfg, "emission_control_effect_strength", 0.075))
    noise_sd = float(getattr(cfg, "emission_panel_noise_sd", 0.010))
    state_target_weight = float(np.clip(getattr(cfg, "emission_state_target_weight", 0.68), 0.0, 1.0))
    s = int(manager.state)

    signals = np.array(
        [
            _std_signal(decision_latency, 8.8, 2.8),
            _std_signal(ctx["demand_volatility"], 0.40, 0.16),
            _std_signal(ctx["forecast_accuracy"], 0.76, 0.095),
            _std_signal(ctx["performance_pressure"], 0.08, 0.075),
            _std_signal(ctx["recent_negative_shock"], 0.10, 0.30),
            _std_signal(ctx["supply_disruption_count"], 0.62, 0.85),
            _std_signal(ctx["target_difficulty"], 0.22, 0.105),
            _std_signal(ctx["task_complexity"], 5.8, 2.5),
        ],
        dtype=float,
    )

    authority_weights = np.array([-0.55, -1.35, 0.80, -0.95, -0.70, -0.85, -1.15, -2.40])
    escalation_weights = np.array([0.65, 1.00, -0.80, 0.95, 0.80, 0.90, 1.05, 1.10])
    normalizer = float(np.sqrt(len(signals)))

    authority_control_shift = strength * float(authority_weights @ signals) / normalizer
    escalation_control_shift = strength * float(escalation_weights @ signals) / normalizer

    authority_intercepts = tuple(getattr(cfg, "authority_state_time_intercepts", (0.20, 0.43, 0.53)))
    authority_slopes = tuple(getattr(cfg, "authority_state_time_slopes", (-0.15, -0.045, 0.12)))
    escalation_intercepts = tuple(getattr(cfg, "escalation_state_time_intercepts", (0.46, 0.55, 0.39)))
    escalation_slopes = tuple(getattr(cfg, "escalation_state_time_slopes", (0.18, -0.055, 0.19)))
    target_idx = min(s, len(authority_intercepts) - 1, len(authority_slopes) - 1)
    escalation_idx = min(s, len(escalation_intercepts) - 1, len(escalation_slopes) - 1)
    authority_state_target = float(authority_intercepts[target_idx] + authority_slopes[target_idx] * progress)
    escalation_state_target = float(escalation_intercepts[escalation_idx] + escalation_slopes[escalation_idx] * progress)

    ai_authority_share = float(
        np.clip(
            (1.0 - state_target_weight) * ai_authority_share
            + state_target_weight * authority_state_target
            + authority_control_shift
            + rng.normal(0.0, noise_sd),
            0.015,
            0.955,
        )
    )
    escalation_share = float(
        np.clip(
            (1.0 - state_target_weight) * escalation_share
            + state_target_weight * escalation_state_target
            + escalation_control_shift
            + rng.normal(0.0, noise_sd),
            0.030,
            0.720,
        )
    )
    return ai_authority_share, escalation_share


def update_latent_state(
    cfg: SimConfig,
    rng: np.random.Generator,
    manager: ManagerProfile,
    kpi_improvement_score: float,
    override_rate: float,
    ctx: Dict[str, float],
) -> int:
    """
    HMM-like transition update (low/med/high willingness), driven by performance appraisal + frictions.
    """
    s = manager.state
    pressure = ctx["performance_pressure"]

    if cfg.n_states >= 4:
        stay = [0.72, 0.58, 0.58, 0.76][s]
        up = [0.24, 0.25, 0.22, 0.00][s]
        down = [0.00, 0.16, 0.16, 0.20][s]
    else:
        stay = [0.70, 0.66, 0.78][s]
        up = [0.18, 0.17, 0.00][s]
        down = [0.00, 0.13, 0.10][s]

    perf = float(np.clip(kpi_improvement_score, -2.0, 2.0))
    xi = float(getattr(manager, "transition_random_effect_xi", 0.0))
    xi_scaled = float(np.tanh(xi))
    ai_authority_signal = float(ctx.get("ai_authority_share", 0.5))
    peer_signal = float(ctx.get("peer_benchmark_proxy", 0.0))
    target_gap = float(ctx.get("target_gap_signed", 0.0))
    ai_reliance = max(0.0, ai_authority_signal - 0.45)
    manual_reliance = max(0.0, 0.45 - ai_authority_signal)
    peer_outperformance = max(0.0, peer_signal)
    peer_underperformance = max(0.0, -peer_signal)
    target_success = max(0.0, target_gap)
    target_shortfall = max(0.0, -target_gap)
    positive_feedback = max(0.0, perf) + 0.75 * peer_outperformance + 0.45 * target_success
    negative_feedback = max(0.0, -perf) + 0.75 * peer_underperformance + 0.45 * target_shortfall
    ai_best_practice_signal = ai_reliance * (1.0 + positive_feedback)
    manual_success_signal = manual_reliance * (1.0 + positive_feedback)
    up_adj = 0.13 * max(0.0, perf) + 0.04 * max(0.0, xi_scaled)
    down_adj = 0.08 * max(0.0, -perf) + 0.04 * max(0.0, -xi_scaled)

    down_adj += 0.12 * max(0.0, override_rate - 0.35)

    if pressure > 0.65 and perf < 0:
        down_adj += 0.05

    progress = ctx.get("period_progress", 0.0)
    max_state = max(cfg.n_states - 1, 1)
    state_position = s / max_state
    time_up_adj = cfg.latent_upward_time_shift * progress * (1.0 - state_position)
    time_down_relief = cfg.latent_downward_time_shift * progress * state_position

    up = clip01(up + up_adj + time_up_adj)
    down = clip01(max(0.0, down + down_adj - time_down_relief))
    stay = clip01(1.0 - up - down)

    if cfg.n_states == 3:
        direct_high = 0.0
        if s == 0:
            # Aversive managers treat successful manual intervention as
            # evidence for continued withholding, while clear AI peer wins can
            # still pull them toward appreciation.
            aversion_reinforcement = (
                0.18 * manual_success_signal
                + 0.10 * negative_feedback
                + 0.08 * max(0.0, override_rate - 0.45)
            )
            ai_peer_pull = 0.16 * ai_best_practice_signal + 0.12 * peer_outperformance
            direct_high = clip01(
                0.02
                + 0.18 * max(0.0, perf)
                + 0.03 * progress
                + 0.03 * max(0.0, xi_scaled)
                + ai_peer_pull
                - 0.10 * aversion_reinforcement
            )
            up = clip01(up + 0.07 * max(0.0, perf) + ai_peer_pull - 0.16 * aversion_reinforcement)
            stay = clip01(stay + aversion_reinforcement)
        elif s == 1:
            # Neutral managers are deliberately the most benchmark-sensitive:
            # positive AI-backed peer signals move them upward, while negative
            # feedback or manual reliance moves them downward.
            neutral_up_signal = 0.24 * ai_best_practice_signal + 0.14 * peer_outperformance + 0.05 * max(0.0, xi_scaled)
            neutral_down_signal = 0.16 * negative_feedback + 0.10 * manual_reliance + 0.05 * max(0.0, -xi_scaled)
            up = clip01(up + neutral_up_signal)
            down = clip01(down + neutral_down_signal)
        elif s == 2 and perf > 0:
            appreciation_reinforcement = 0.12 * perf + 0.16 * ai_best_practice_signal + 0.08 * peer_outperformance
            stay = clip01(stay + appreciation_reinforcement)
            down = clip01(max(0.0, down - 0.08 * appreciation_reinforcement + 0.10 * negative_feedback))
        elif s == 2 and xi_scaled > 0:
            stay = clip01(stay + 0.04 * xi_scaled)
            down = clip01(max(0.0, down - 0.03 * xi_scaled))
        elif s == 2:
            down = clip01(down + 0.12 * negative_feedback + 0.08 * peer_underperformance)

        probs = np.zeros(cfg.n_states, dtype=float)
        probs[s] += stay
        if s > 0:
            probs[s - 1] += down
        if s < cfg.n_states - 1:
            probs[s + 1] += up
        if s == 0:
            probs[2] += direct_high
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()
        return int(rng.choice(np.arange(cfg.n_states), p=probs))

    r = rng.random()
    if r < down and s > 0:
        return s - 1
    if r < down + stay:
        return s
    return min(cfg.n_states - 1, s + 1)


# -----------------------------
# Main simulation
# -----------------------------
def simulate(cfg: SimConfig) -> Dict[str, pd.DataFrame]:
    """
    Main simulation function for triadic delegation data.

    Outputs (keys):
      - manager_master
      - employee_master
      - ai_system_master
      - site_master
      - panel_manager_period
      - panel_employee_period
      - decision_episode
      - execution_episode
    """
    rng = np.random.default_rng(cfg.seed)

    # Master data
    site_master_df = sample_site_master(cfg, rng)
    site_ids = site_master_df["site_id"].tolist()
    site_complexity_map = dict(zip(site_master_df["site_id"], site_master_df["baseline_operational_complexity"]))
    site_region_map = dict(zip(site_master_df["site_id"], site_master_df["region"]))

    managers = sample_manager_profiles(cfg, rng, site_ids)
    employees = sample_employee_master(cfg, rng, managers)

    # ai_system_master (single deployed version)
    ai_version = getattr(cfg, "ai_version", "v1")
    ai_system_rows = [
        dict(
            ai_version=ai_version,
            deployment_date=str(getattr(cfg, "ai_deployment_date", "2017-01-01")),
            autonomy_level=cfg.autonomy_level,
            confidence_calibration_score=cfg.confidence_calibration_score,
        )
    ]

    # manager_master
    manager_master_rows: List[Dict[str, object]] = []
    for m in managers:
        manager_master_rows.append(
            dict(
                manager_id=m.manager_id,
                role="operations_manager",
                function="supply_chain",
                seniority_years=float(np.clip(rng.normal(6.0, 3.0), 0.0, 30.0)),
                risk_aversion_index=m.risk_aversion_index,
                baseline_ai_attitude=m.baseline_ai_attitude,
                org_unit_id=m.org_unit_id,
                site_id=m.site_id,
                region=site_region_map.get(m.site_id, ""),
                governance_mode=m.governance_mode,
                high_pressure=int(m.high_pressure),
                transition_random_effect_xi=m.transition_random_effect_xi,
            )
        )

    # employee_master
    employee_master_rows: List[Dict[str, object]] = []
    for e in employees:
        employee_master_rows.append(
            dict(
                employee_id=e.employee_id,
                manager_id=e.manager_id,
                site_id=e.site_id,
                role=e.role,
                experience_years=e.experience_years,
                ai_familiarity=e.ai_familiarity,
                task_specialization=e.task_specialization,
            )
        )

    # Dynamic tables
    panel_manager_rows: List[Dict[str, object]] = []
    panel_employee_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []
    execution_rows: List[Dict[str, object]] = []

    # Convenience: employees by manager
    emp_by_manager: Dict[str, List[EmployeeProfile]] = {}
    for e in employees:
        emp_by_manager.setdefault(e.manager_id, []).append(e)
    manager_score_history: Dict[str, List[float]] = {m.manager_id: [] for m in managers}

    # Simulation loop
    for period_id in range(1, cfg.n_periods + 1):
        base_eps = int(rng.integers(cfg.episodes_per_period_low, cfg.episodes_per_period_high + 1))
        period_manager_rows: List[Dict[str, object]] = []

        # --- Manager-period loop
        for m in managers:
            ctx = period_context(cfg, rng, m, manager_score_history[m.manager_id], period_id)
            n_episodes = int(max(5, rng.integers(max(5, base_eps - 8), base_eps + 9)))

            # period accumulators (manager panel)
            accepted = overridden = escalated = 0
            decision_latency_list: List[float] = []
            correctness_list: List[int] = []
            error_incidents = 0

            # period accumulators (employee panel) by employee_id
            emp_acc: dict[str, dict[str, Any]] = {}
            for e in emp_by_manager.get(m.manager_id, []):
                emp_acc[e.employee_id] = dict(
                    employee_id=e.employee_id,
                    manager_id=m.manager_id,
                    period_id=period_id,
                    site_id=e.site_id,
                    n_exec=0,
                    n_ai=0,
                    n_human=0,
                    n_joint=0,
                    exec_time_list=[],
                    error_count=0,
                    rework_count=0,
                    ai_support_level_list=[],
                    coordination_complexity=float(
                        np.clip(rng.normal(0.55 + 0.25 * ctx["task_complexity_index"], 0.12), 0.0, 1.0)
                    ),
                )

            site_complexity = float(site_complexity_map.get(m.site_id, 0.5))

            for ep in range(1, n_episodes + 1):
                episode_id = f"EP_{m.manager_id}_{period_id:03d}_{ep:03d}"
                execution_id = f"EX_{m.manager_id}_{period_id:03d}_{ep:03d}"

                ai_conf = sample_ai_confidence(cfg, rng, ctx, site_complexity=site_complexity)
                ai_unc = sample_ai_uncertainty(rng, ai_conf)

                ai_recommendation_type = str(rng.choice(["transfer", "reroute", "reorder", "expedite"]))

                escalation_flag = int(rng.random() < escalation_probability(cfg, m, ctx, ai_conf))
                p_acc, p_mod, p_rej = episode_decision_probabilities(cfg, m, ctx, ai_conf)
                manager_action = choice_with_probs(rng, ["accept", "modify", "reject"], [p_acc, p_mod, p_rej])

                override_flag = int(manager_action in ["modify", "reject"])
                overridden += override_flag
                escalated += escalation_flag
                accepted += int(manager_action == "accept")

                # time_to_decision (minutes)
                base_latency = 2.0 + 6.0 * ctx["task_complexity_index"] + 4.0 * ctx["demand_volatility"]
                base_latency += 3.5 * override_flag + 2.0 * escalation_flag
                base_latency *= (1.10 - 0.08 * m.state)
                time_to_decision = float(np.clip(rng.normal(base_latency, 1.5), 0.5, 25.0))
                decision_latency_list.append(time_to_decision)

                # correctness generation
                true_p_correct = clip01(
                    0.55
                    + 0.65 * (ai_conf - 0.5)
                    - 0.25 * ctx["demand_volatility"]
                    - 0.20 * ctx["task_complexity_index"]
                    - 0.10 * site_complexity
                )
                ai_correct = int(rng.random() < true_p_correct)

                if manager_action == "accept":
                    decision_correct = ai_correct
                elif manager_action == "modify":
                    fix_prob = clip01(
                        0.55
                        + 0.20 * (1 - m.risk_aversion_index)
                    )
                    decision_correct = int(rng.random() < fix_prob) if ai_correct == 0 else int(
                        rng.random() < (0.88 - 0.10 * ctx["performance_pressure"])
                    )
                else:  # reject
                    human_base = 0.62 - 0.10 * ctx["task_complexity_index"] - 0.10 * ctx["demand_volatility"]
                    human_base += 0.06 * m.state
                    decision_correct = int(rng.random() < clip01(human_base))

                correctness_list.append(decision_correct)

                # Select an employee executing this episode
                e_list = emp_by_manager.get(m.manager_id, [])
                executor_emp = e_list[int(rng.integers(0, len(e_list)))] if e_list else None

                # Execution mode
                if manager_action == "accept":
                    if cfg.autonomy_level == "high":
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.65, 0.25, 0.10])
                    elif cfg.autonomy_level == "medium":
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.45, 0.35, 0.20])
                    else:
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.25, 0.40, 0.35])
                else:
                    exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.10, 0.30, 0.60])

                # AI support level during execution (0..1)
                if executor_emp is None:
                    ai_support_level = float(np.clip(rng.beta(2, 3), 0.0, 1.0))
                else:
                    base_support = {"ai": 0.85, "joint": 0.65, "human": 0.30}[exec_mode]
                    ai_support_level = float(
                        np.clip(rng.normal(base_support + 0.20 * executor_emp.ai_familiarity, 0.10), 0.0, 1.0)
                    )

                employee_override_during_execution = int((decision_correct == 0) and (rng.random() < 0.08))
                local_adjustment_flag = int(rng.random() < clip01(0.12 + 0.18 * ctx["task_complexity_index"]))

                exec_time = float(
                    np.clip(
                        rng.normal(
                            8.0
                            + 10.0 * ctx["task_complexity_index"]
                            + 6.0 * ctx["demand_volatility"]
                            + (4.0 if exec_mode == "human" else 1.5 if exec_mode == "joint" else 0.5),
                            3.0,
                        ),
                        1.0,
                        60.0,
                    )
                )
                execution_error_flag = int(
                    (decision_correct == 0) and (rng.random() < (0.05 + 0.06 * ctx["performance_pressure"]))
                )
                rework_flag = int(execution_error_flag == 1 or employee_override_during_execution == 1)

                major_error_flag = int(execution_error_flag == 1 and (rng.random() < 0.35))
                error_incidents += major_error_flag

                # decision_episode row (UPDATED)
                decision_rows.append(
                    dict(
                        episode_id=episode_id,
                        manager_id=m.manager_id,
                        period_id=period_id,
                        site_id=m.site_id,
                        ai_version=ai_version,
                        ai_recommendation_type=ai_recommendation_type,
                        ai_confidence=ai_conf,
                        ai_uncertainty=ai_unc,
                        manager_action=manager_action,
                        override_flag=override_flag,
                        escalation_flag=escalation_flag,
                        time_to_decision=time_to_decision,
                    )
                )

                # execution_episode row
                execution_rows.append(
                    dict(
                        execution_id=execution_id,
                        episode_id=episode_id,
                        employee_id=(executor_emp.employee_id if executor_emp is not None else None),
                        site_id=(executor_emp.site_id if executor_emp is not None else m.site_id),
                        execution_mode=exec_mode,
                        ai_support_level=ai_support_level,
                        employee_override_during_execution=employee_override_during_execution,
                        local_adjustment_flag=local_adjustment_flag,
                        execution_time=exec_time,
                        execution_error_flag=execution_error_flag,
                    )
                )

                # Update employee accumulators
                if executor_emp is not None:
                    acc = emp_acc[executor_emp.employee_id]
                    acc["n_exec"] += 1
                    acc["n_ai"] += int(exec_mode == "ai")
                    acc["n_human"] += int(exec_mode == "human")
                    acc["n_joint"] += int(exec_mode == "joint")
                    acc["exec_time_list"].append(exec_time)
                    acc["ai_support_level_list"].append(ai_support_level)
                    acc["error_count"] += int(execution_error_flag)
                    acc["rework_count"] += int(rework_flag)

            # -----------------------------
            # panel_manager_period aggregates
            # -----------------------------
            ai_authority_share = accepted / n_episodes
            override_rate = overridden / n_episodes
            escalation_share = escalated / n_episodes
            decision_latency = float(np.mean(decision_latency_list)) if decision_latency_list else float("nan")
            ai_authority_share, escalation_share = calibrate_panel_emissions(
                cfg=cfg,
                rng=rng,
                manager=m,
                ctx=ctx,
                ai_authority_share=ai_authority_share,
                escalation_share=escalation_share,
                decision_latency=decision_latency,
            )

            quality = float(np.mean(correctness_list)) if correctness_list else 0.5
            collab_effect = (ai_authority_share - 0.5) * (quality - 0.5)
            normalized_error_rate = error_incidents / max(n_episodes, 1)

            service_level_delta = float(
                -1.5
                + 3.0 * (0.5 - quality)
                + 1.8 * ctx["demand_volatility"]
                + 1.2 * ctx["task_complexity_index"]
                - 1.6 * collab_effect
                + rng.normal(0, 0.6)
            )
            inventory_cost_delta = float(
                0.8
                + 2.2 * (0.5 - quality)
                + 1.5 * override_rate
                + 0.8 * ctx["performance_pressure"]
                - 1.1 * collab_effect
                + rng.normal(0, 0.7)
            )
            expedite_cost_delta = float(
                0.6
                + 1.8 * ctx["demand_volatility"]
                + 1.2 * (1 - quality)
                + 1.0 * escalation_share
                - 0.8 * collab_effect
                + rng.normal(0, 0.7)
            )

            kpi_improvement_score = float(
                (+0.7 * (0.0 - service_level_delta)
                 +0.5 * (0.0 - inventory_cost_delta)
                 +0.4 * (0.0 - expedite_cost_delta)
                 -0.8 * error_incidents)
            )
            composite_kpi_score = float(
                np.clip(
                    0.54
                    + ([-0.035, 0.000, 0.055][m.state] if cfg.n_states == 3 else 0.0)
                    + 0.28 * (quality - 0.5)
                    + 0.16 * ai_authority_share
                    + 0.05 * ai_authority_share * max(0.0, quality - 0.5)
                    - 0.10 * override_rate
                    - 0.05 * escalation_share
                    + 0.05 * escalation_share * max(0.0, 0.65 - quality)
                    - 0.12 * ctx["demand_volatility"]
                    - 0.10 * normalized_error_rate
                    + rng.normal(0, 0.04),
                    0.0,
                    1.0,
                )
            )
            previous_scores = manager_score_history[m.manager_id]
            previous_score = previous_scores[-1] if previous_scores else ctx["prior_composite_kpi_score"]
            team_t_minus_1_vs_team_t = float(composite_kpi_score - previous_score)
            target_attainment = int(composite_kpi_score >= ctx["kpi_target"])
            transition_ctx = dict(ctx)
            transition_ctx.update(
                {
                    "ai_authority_share": ai_authority_share,
                    "escalation_share": escalation_share,
                    "peer_benchmark_proxy": composite_kpi_score
                    - (0.50 + 0.03 * ctx["period_progress"] - 0.02 * ctx["demand_volatility"]),
                    "target_gap_signed": composite_kpi_score - ctx["kpi_target"],
                }
            )

            new_state = update_latent_state(
                cfg=cfg,
                rng=rng,
                manager=m,
                kpi_improvement_score=kpi_improvement_score / 3.0,
                override_rate=override_rate,
                ctx=transition_ctx,
            )

            period_manager_rows.append(
                dict(
                    manager_id=m.manager_id,
                    period_id=period_id,
                    region=site_region_map.get(m.site_id, ""),
                    ai_authority_share=ai_authority_share,
                    escalation_share=escalation_share,
                    decision_latency=decision_latency,
                    demand_volatility=ctx["demand_volatility"],
                    forecast_accuracy=ctx["forecast_accuracy"],
                    performance_pressure=ctx["performance_pressure"],
                    recent_negative_shock=int(ctx["recent_negative_shock"]),
                    supply_disruptions=int(ctx["supply_disruption_count"]),
                    target_difficulty=ctx["target_difficulty"],
                    task_complexity=ctx["task_complexity"],
                    team_t_minus_1_vs_team_t=team_t_minus_1_vs_team_t,
                    team_vs_peer_average=float("nan"),
                    target_attainment=target_attainment,
                    composite_kpi_score=composite_kpi_score,
                    kpi_target=ctx["kpi_target"],
                    override_rate=override_rate,
                    service_level_delta=service_level_delta,
                    inventory_cost_delta=inventory_cost_delta,
                    expedite_cost_delta=expedite_cost_delta,
                    error_incident_count=int(error_incidents),
                    ai_version=ai_version,
                    latent_state_true=m.state,
                    latent_state_true_next=new_state,
                )
            )

            manager_score_history[m.manager_id].append(composite_kpi_score)
            m.state = new_state

            # -----------------------------
            # panel_employee_period aggregates
            # -----------------------------
            for _, acc in emp_acc.items():
                n_exec = acc["n_exec"]
                if n_exec <= 0:
                    ai_execution_share = 0.0
                    employee_execution_share = 0.0
                    joint_execution_share = 0.0
                    avg_execution_time = float("nan")
                    error_rate = 0.0
                    rework_rate = 0.0
                    ai_support_intensity = float("nan")
                    employee_workload = 0
                else:
                    ai_execution_share = acc["n_ai"] / n_exec
                    employee_execution_share = acc["n_human"] / n_exec
                    joint_execution_share = acc["n_joint"] / n_exec
                    _etl: list[float] = acc["exec_time_list"]
                    avg_execution_time = float(np.mean(_etl))
                    error_rate = acc["error_count"] / n_exec
                    rework_rate = acc["rework_count"] / n_exec
                    _asl: list[float] = acc["ai_support_level_list"]
                    ai_support_intensity = float(np.mean(_asl)) if _asl else float("nan")
                    employee_workload = int(n_exec)

                panel_employee_rows.append(
                    dict(
                        employee_id=acc["employee_id"],
                        manager_id=acc["manager_id"],
                        period_id=acc["period_id"],
                        site_id=acc["site_id"],
                        ai_execution_share=ai_execution_share,
                        employee_execution_share=employee_execution_share,
                        joint_execution_share=joint_execution_share,
                        avg_execution_time=avg_execution_time,
                        rework_rate=rework_rate,
                        error_rate=error_rate,
                        employee_workload=employee_workload,
                        coordination_complexity=acc["coordination_complexity"],
                        ai_support_intensity=ai_support_intensity,
                    )
                )

        if period_manager_rows:
            period_df = pd.DataFrame(period_manager_rows)
            peer_average = period_df.groupby("region")["composite_kpi_score"].transform("mean")
            period_df["team_vs_peer_average"] = period_df["composite_kpi_score"] - peer_average
            panel_manager_rows.extend(period_df.to_dict("records"))

    return {
        "manager_master": pd.DataFrame(manager_master_rows),
        "employee_master": pd.DataFrame(employee_master_rows),
        "ai_system_master": pd.DataFrame(ai_system_rows),
        "site_master": site_master_df,
        "panel_manager_period": pd.DataFrame(panel_manager_rows),
        "panel_employee_period": pd.DataFrame(panel_employee_rows),
        "decision_episode": pd.DataFrame(decision_rows),
        "execution_episode": pd.DataFrame(execution_rows),
    }
