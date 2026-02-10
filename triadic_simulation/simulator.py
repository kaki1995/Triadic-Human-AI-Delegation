# triadic_sim/simulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import SimConfig
from .utils import clip01, choice_with_probs


@dataclass
class ManagerProfile:
    """A single manager entity with stable traits + dynamic latent state."""
    manager_id: str
    governance_mode: str                  # fearful_exclusion / controlled_opening / opportunistic_teaming
    baseline_ai_attitude: float           # -1..+1
    risk_aversion_index: float            # 0..1
    high_pressure: bool
    org_unit_id: str
    employee_team_id: str
    state: int                            # 0..n_states-1


def explanation_capability_for_period(cfg: SimConfig, period_id: int) -> str:
    """Return explanation capability given the global transparency shift."""
    return cfg.explanation_capability_post if period_id >= cfg.transparency_shift_period else cfg.explanation_capability_pre


def sample_manager_profiles(cfg: SimConfig, rng: np.random.Generator) -> List[ManagerProfile]:
    """Create manager population with governance-mode mixture and correlated traits."""
    modes = ["fearful_exclusion", "controlled_opening", "opportunistic_teaming"]
    p = np.array([cfg.p_fearful, cfg.p_controlled, cfg.p_opportunistic], dtype=float)
    p = p / p.sum()

    profiles: List[ManagerProfile] = []
    for i in range(cfg.n_managers):
        manager_id = f"M{i+1:04d}"
        gov = str(rng.choice(modes, p=p))

        # Attitude distribution depends on governance mode (domain assumption)
        mu = {"fearful_exclusion": -0.6, "controlled_opening": -0.1, "opportunistic_teaming": 0.5}[gov]
        baseline_ai_attitude = float(np.clip(rng.normal(mu, 0.35), -1.0, 1.0))

        # Risk aversion distribution depends on governance mode (domain assumption)
        rmu = {"fearful_exclusion": 0.75, "controlled_opening": 0.55, "opportunistic_teaming": 0.35}[gov]
        risk_aversion_index = float(np.clip(rng.normal(rmu, 0.15), 0.0, 1.0))

        high_pressure = bool(rng.random() < cfg.high_pressure_share_of_managers)
        org_unit_id = f"OU{rng.integers(1, 9):02d}"
        employee_team_id = f"T{rng.integers(1, 40):03d}"

        # Initial latent state from attitude
        if baseline_ai_attitude < -0.25:
            state = 0
        elif baseline_ai_attitude < 0.25:
            state = 1
        else:
            state = 2

        profiles.append(
            ManagerProfile(
                manager_id=manager_id,
                governance_mode=gov,
                baseline_ai_attitude=baseline_ai_attitude,
                risk_aversion_index=risk_aversion_index,
                high_pressure=high_pressure,
                org_unit_id=org_unit_id,
                employee_team_id=employee_team_id,
                state=state,
            )
        )
    return profiles


def period_context(rng: np.random.Generator, manager: ManagerProfile) -> Dict[str, float]:
    """
    Sample period-specific context variables (pressure, volatility, complexity, shocks).
    """
    base_pressure = 0.75 if manager.high_pressure else 0.45
    performance_pressure_index = float(np.clip(rng.normal(base_pressure, 0.08), 0.0, 1.0))
    target_difficulty = float(np.clip(0.35 + 0.7 * performance_pressure_index + rng.normal(0, 0.1), 0.0, 1.0))
    demand_volatility = float(np.clip(rng.beta(2, 4) + 0.15 * rng.random(), 0.0, 1.0))
    task_complexity_index = float(np.clip(rng.beta(2.2, 2.2), 0.0, 1.0))

    shock_prob = 0.06 + 0.08 * demand_volatility
    recent_negative_shock = float(int(rng.random() < shock_prob))

    return {
        "performance_pressure_index": performance_pressure_index,
        "target_difficulty": target_difficulty,
        "demand_volatility": demand_volatility,
        "task_complexity_index": task_complexity_index,
        "recent_negative_shock": recent_negative_shock,
    }


def sample_ai_confidence(cfg: SimConfig, rng: np.random.Generator, ctx: Dict[str, float]) -> float:
    """
    Sample AI confidence for an episode.
    Higher complexity and volatility reduces expected confidence.
    """
    complexity = ctx["task_complexity_index"]
    volatility = ctx["demand_volatility"]
    mu = 0.78 - 0.25 * complexity - 0.20 * volatility
    mu = float(np.clip(mu, 0.15, 0.95))

    # calibration controls variance: higher calibration => tighter
    k = 8 + 20 * cfg.confidence_calibration_score
    a = max(1.0, mu * k)
    b = max(1.0, (1 - mu) * k)
    return float(rng.beta(a, b))


def episode_decision_probabilities(
    manager: ManagerProfile,
    ctx: Dict[str, float],
    explanation_capability: str,
    ai_confidence: float,
) -> Tuple[float, float, float]:
    """
    Compute (p_accept, p_modify, p_reject) for a single recommendation episode.

    Model intuition:
    - Higher latent state => higher accept, lower reject
    - Transparency/explanations => higher accept, lower reject
    - High pressure + risk + shocks => more reject/modify (esp. in low state)
    """
    s = manager.state
    pressure = ctx["performance_pressure_index"]
    shock = ctx["recent_negative_shock"]
    risk = manager.risk_aversion_index

    base_accept = [0.15, 0.45, 0.75][s]
    base_reject = [0.55, 0.25, 0.10][s]

    tr = {"none": 0.00, "basic": 0.05, "detailed": 0.10}[explanation_capability]
    conf_effect = 0.20 * (ai_confidence - 0.5)

    pressure_penalty = (0.18 if s == 0 else 0.06) * pressure * risk
    shock_penalty = (0.20 if s <= 1 else 0.10) * shock

    accept = clip01(base_accept + tr + conf_effect - pressure_penalty - shock_penalty)
    reject = clip01(base_reject + pressure_penalty + shock_penalty - tr - 0.5 * conf_effect)
    modify = clip01(1.0 - accept - reject)

    if manager.governance_mode == "controlled_opening":
        modify = clip01(modify + 0.05)
        z = accept + reject + modify
        accept, reject, modify = accept / z, reject / z, modify / z

    z = accept + modify + reject
    return accept / z, modify / z, reject / z


def escalation_probability(cfg: SimConfig, manager: ManagerProfile, ai_confidence: float, explanation_capability: str) -> float:
    """
    AI-to-manager escalation probability.
    Higher when confidence is low and explanations are weak; lower for high willingness states.
    """
    autonomy_base = {"low": 0.04, "medium": 0.08, "high": 0.12}[cfg.autonomy_level]
    low_conf = max(0.0, 0.6 - ai_confidence)
    expl_penalty = {"none": 0.06, "basic": 0.03, "detailed": 0.00}[explanation_capability]
    state_factor = [1.15, 1.00, 0.85][manager.state]
    return clip01((autonomy_base + 0.40 * low_conf + expl_penalty) * state_factor)


def update_latent_state(
    cfg: SimConfig,
    rng: np.random.Generator,
    manager: ManagerProfile,
    kpi_improvement_score: float,
    explanation_capability: str,
    override_rate: float,
    ctx: Dict[str, float],
) -> int:
    """
    HMM-like transition update (low/med/high willingness), driven by feedback.
    """
    s = manager.state
    pressure = ctx["performance_pressure_index"]

    stay = [0.78, 0.62, 0.80][s]
    up = [0.18, 0.25, 0.00][s]
    down = [0.00, 0.13, 0.18][s]

    perf = float(np.clip(kpi_improvement_score, -2.0, 2.0))
    up_adj = 0.10 * max(0.0, perf)
    down_adj = 0.08 * max(0.0, -perf)

    tr = {"none": 0.00, "basic": 0.03, "detailed": 0.06}[explanation_capability]
    up_adj += tr
    down_adj -= tr / 2

    down_adj += 0.12 * max(0.0, override_rate - 0.35)

    if pressure > 0.65 and perf < 0:
        down_adj += 0.05

    up = clip01(up + up_adj)
    down = clip01(down + down_adj)
    stay = clip01(1.0 - up - down)

    r = rng.random()
    if r < down and s > 0:
        return s - 1
    if r < down + stay:
        return s
    return min(cfg.n_states - 1, s + 1)


def simulate(cfg: SimConfig) -> Dict[str, pd.DataFrame]:
    """
    Main simulation function. Returns DataFrames keyed by internal table name.

    Output keys:
      - manager_master
      - employee_team_master
      - ai_system_master
      - panel_manager_period
      - decision_episode
      - execution_episode
    """
    rng = np.random.default_rng(cfg.seed)
    managers = sample_manager_profiles(cfg, rng)

    manager_master_rows: List[Dict[str, object]] = []
    employee_team_rows: List[Dict[str, object]] = []
    ai_system_rows: List[Dict[str, object]] = []
    panel_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []
    execution_rows: List[Dict[str, object]] = []

    ai_system_id = "AI001"
    ai_system_rows.append(
        dict(
            ai_system_id=ai_system_id,
            autonomy_level=cfg.autonomy_level,
            confidence_calibration_score=cfg.confidence_calibration_score,
        )
    )

    team_seen: set[str] = set()
    for m in managers:
        manager_master_rows.append(
            dict(
                manager_id=m.manager_id,
                org_unit_id=m.org_unit_id,
                employee_team_id=m.employee_team_id,
                baseline_ai_attitude=m.baseline_ai_attitude,
                risk_aversion_index=m.risk_aversion_index,
                governance_mode=m.governance_mode,
                high_pressure=int(m.high_pressure),
            )
        )
        if m.employee_team_id not in team_seen:
            team_seen.add(m.employee_team_id)
            employee_team_rows.append(
                dict(
                    employee_team_id=m.employee_team_id,
                    team_size=int(rng.integers(3, 10)),
                    avg_experience_years=float(np.clip(rng.normal(5.5, 2.0), 0.5, 25.0)),
                )
            )

    for period_id in range(1, cfg.n_periods + 1):
        explanation_capability = explanation_capability_for_period(cfg, period_id)
        base_eps = int(rng.integers(cfg.episodes_per_period_low, cfg.episodes_per_period_high + 1))

        for m in managers:
            ctx = period_context(rng, m)
            n_episodes = int(max(5, rng.integers(max(5, base_eps - 8), base_eps + 9)))

            accepted = overridden = escalated = 0
            exec_ai = exec_human = exec_joint = 0
            decision_latency_list: List[float] = []
            correctness_list: List[int] = []
            error_incidents = 0

            for e in range(1, n_episodes + 1):
                decision_episode_id = f"D_{m.manager_id}_{period_id:03d}_{e:03d}"
                execution_episode_id = f"X_{m.manager_id}_{period_id:03d}_{e:03d}"

                ai_conf = sample_ai_confidence(cfg, rng, ctx)

                # Explanation actually shown (episode-level)
                if explanation_capability == "none":
                    explanation_provided = 0
                else:
                    base = 0.55 if m.governance_mode == "controlled_opening" else 0.70
                    if m.governance_mode == "fearful_exclusion":
                        base = 0.35
                    explanation_provided = int(rng.random() < base)

                escalation_flag = int(rng.random() < escalation_probability(cfg, m, ai_conf, explanation_capability))

                p_acc, p_mod, p_rej = episode_decision_probabilities(m, ctx, explanation_capability, ai_conf)
                manager_action = choice_with_probs(rng, ["accept", "modify", "reject"], [p_acc, p_mod, p_rej])

                override_flag = int(manager_action in ["modify", "reject"])
                overridden += override_flag
                escalated += escalation_flag
                accepted += int(manager_action == "accept")

                # latency proxy
                base_latency = 2.0 + 6.0 * ctx["task_complexity_index"] + 4.0 * ctx["demand_volatility"]
                base_latency += 3.5 * override_flag + 2.0 * escalation_flag
                base_latency *= (1.10 - 0.08 * m.state)
                decision_latency = float(np.clip(rng.normal(base_latency, 1.5), 0.5, 25.0))
                decision_latency_list.append(decision_latency)

                # correctness generation
                true_p_correct = clip01(
                    0.55
                    + 0.65 * (ai_conf - 0.5)
                    - 0.25 * ctx["demand_volatility"]
                    - 0.20 * ctx["task_complexity_index"]
                )
                ai_correct = int(rng.random() < true_p_correct)

                if manager_action == "accept":
                    decision_correct = ai_correct
                elif manager_action == "modify":
                    fix_prob = clip01(
                        0.55 + 0.20 * (1 - m.risk_aversion_index) + 0.10 * (1 if explanation_provided else 0)
                    )
                    decision_correct = int(rng.random() < fix_prob) if ai_correct == 0 else int(
                        rng.random() < (0.88 - 0.10 * ctx["performance_pressure_index"])
                    )
                else:  # reject
                    human_base = 0.62 - 0.10 * ctx["task_complexity_index"] - 0.10 * ctx["demand_volatility"]
                    human_base += 0.06 * m.state
                    decision_correct = int(rng.random() < clip01(human_base))

                correctness_list.append(decision_correct)

                # execution mode
                if manager_action == "accept":
                    if cfg.autonomy_level == "high":
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.65, 0.25, 0.10])
                    elif cfg.autonomy_level == "medium":
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.45, 0.35, 0.20])
                    else:
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.25, 0.40, 0.35])
                else:
                    exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.10, 0.30, 0.60])

                exec_ai += int(exec_mode == "ai")
                exec_joint += int(exec_mode == "joint")
                exec_human += int(exec_mode == "human")

                employee_override_during_execution = int((decision_correct == 0) and (rng.random() < 0.08))
                major_error = int((decision_correct == 0) and (rng.random() < (0.03 + 0.05 * ctx["performance_pressure_index"])))
                error_incidents += major_error

                decision_rows.append(
                    dict(
                        decision_episode_id=decision_episode_id,
                        manager_id=m.manager_id,
                        period_id=period_id,
                        episode_in_period=e,
                        ai_system_id=ai_system_id,
                        manager_action=manager_action,
                        override_flag=override_flag,
                        escalation_flag=escalation_flag,
                        explanation_provided=explanation_provided,
                        ai_confidence=ai_conf,
                        decision_latency=decision_latency,
                    )
                )

                execution_rows.append(
                    dict(
                        execution_episode_id=execution_episode_id,
                        decision_episode_id=decision_episode_id,
                        manager_id=m.manager_id,
                        period_id=period_id,
                        execution_mode=exec_mode,
                        employee_team_id=m.employee_team_id,
                        employee_override_during_execution=employee_override_during_execution,
                        major_error_flag=major_error,
                    )
                )

            # period-level aggregates
            ai_decision_authority_share = accepted / n_episodes
            override_rate = overridden / n_episodes
            escalation_rate = escalated / n_episodes

            pct_ai = exec_ai / n_episodes
            pct_human = exec_human / n_episodes
            pct_joint = exec_joint / n_episodes

            quality = float(np.mean(correctness_list))
            collab_effect = (ai_decision_authority_share - 0.5) * (quality - 0.5)

            service_level_delta = float(
                -1.5 + 3.0 * (0.5 - quality)
                + 1.8 * ctx["demand_volatility"]
                + 1.2 * ctx["task_complexity_index"]
                - 1.6 * collab_effect
                + rng.normal(0, 0.6)
            )
            inventory_cost_delta = float(
                0.8 + 2.2 * (0.5 - quality)
                + 1.5 * override_rate
                + 0.8 * ctx["performance_pressure_index"]
                - 1.1 * collab_effect
                + rng.normal(0, 0.7)
            )
            expedite_cost_delta = float(
                0.6 + 1.8 * ctx["demand_volatility"]
                + 1.2 * (1 - quality)
                + 1.0 * escalation_rate
                - 0.8 * collab_effect
                + rng.normal(0, 0.7)
            )

            # feedback score used for state update
            kpi_improvement_score = float(
                (+0.7 * (0.0 - service_level_delta)
                 +0.5 * (0.0 - inventory_cost_delta)
                 +0.4 * (0.0 - expedite_cost_delta)
                 -0.8 * error_incidents)
            )

            new_state = update_latent_state(
                cfg, rng, m,
                kpi_improvement_score=kpi_improvement_score / 3.0,
                explanation_capability=explanation_capability,
                override_rate=override_rate,
                ctx=ctx,
            )

            panel_rows.append(
                dict(
                    manager_id=m.manager_id,
                    period_id=period_id,
                    governance_mode=m.governance_mode,
                    explanation_capability=explanation_capability,
                    autonomy_level=cfg.autonomy_level,
                    ai_decision_authority_share=ai_decision_authority_share,
                    manager_retained_authority_share=1.0 - ai_decision_authority_share,
                    pct_tasks_executed_by_ai=pct_ai,
                    pct_tasks_executed_by_humans=pct_human,
                    pct_tasks_joint_execution=pct_joint,
                    override_rate=override_rate,
                    escalation_rate=escalation_rate,
                    avg_decision_latency=float(np.mean(decision_latency_list)),
                    performance_pressure_index=ctx["performance_pressure_index"],
                    target_difficulty=ctx["target_difficulty"],
                    demand_volatility=ctx["demand_volatility"],
                    task_complexity_index=ctx["task_complexity_index"],
                    recent_negative_shock=int(ctx["recent_negative_shock"]),
                    service_level_delta=service_level_delta,
                    inventory_cost_delta=inventory_cost_delta,
                    expedite_cost_delta=expedite_cost_delta,
                    error_incident_count=int(error_incidents),
                    latent_state=m.state,
                    latent_state_next=new_state,
                )
            )

            m.state = new_state

    return {
        "manager_master": pd.DataFrame(manager_master_rows),
        "employee_team_master": pd.DataFrame(employee_team_rows),
        "ai_system_master": pd.DataFrame(ai_system_rows),
        "panel_manager_period": pd.DataFrame(panel_rows),
        "decision_episode": pd.DataFrame(decision_rows),
        "execution_episode": pd.DataFrame(execution_rows),
    }
