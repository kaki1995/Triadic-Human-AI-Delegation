# triadic_sim/simulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import SimConfig
from .utils import clip01, choice_with_probs


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
    site_id: str                          # NEW: for decision_episode.site_id and panel_employee_period.site_id
    state: int                            # 0..n_states-1


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
def explanation_capability_for_period(cfg: SimConfig, period_id: int) -> str:
    """
    Period-level explanation capability (pre/post transparency shift).
    Must map to: ai_system_master.explanation_capability and be used by decision_episode.explanation_provided.
    """
    return cfg.explanation_capability_post if period_id >= cfg.transparency_shift_period else cfg.explanation_capability_pre


def sample_site_master(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create site_master:
      site_id, region, plant_type, automation_level, baseline_operational_complexity
    """
    # Keep small, reusable set of sites
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

        # Attitude distribution depends on governance mode (domain assumption)
        mu = {"fearful_exclusion": -0.6, "controlled_opening": -0.1, "opportunistic_teaming": 0.5}[gov]
        baseline_ai_attitude = float(np.clip(rng.normal(mu, 0.35), -1.0, 1.0))

        # Risk aversion distribution depends on governance mode (domain assumption)
        rmu = {"fearful_exclusion": 0.75, "controlled_opening": 0.55, "opportunistic_teaming": 0.35}[gov]
        risk_aversion_index = float(np.clip(rng.normal(rmu, 0.15), 0.0, 1.0))

        high_pressure = bool(rng.random() < cfg.high_pressure_share_of_managers)
        org_unit_id = f"OU{rng.integers(1, 9):02d}"
        site_id = str(rng.choice(site_ids))

        # Initial latent state from attitude (anchoring strategy; aligns with MISQ-style clarity)
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
                site_id=site_id,
                state=state,
            )
        )
    return profiles


def sample_employee_master(cfg: SimConfig, rng: np.random.Generator, managers: List[ManagerProfile]) -> List[EmployeeProfile]:
    """
    Create employee_master and a simple reporting relation:
      each manager has k employees (k sampled), employees inherit manager_id and site_id.
    """
    roles = ["planner", "coordinator", "dispatcher", "inventory_analyst"]
    specs = ["inbound", "outbound", "crossdock", "perishables", "high_value"]

    rows: List[EmployeeProfile] = []
    emp_counter = 0

    # Configurable average team size; defaults if not present in cfg
    team_low = getattr(cfg, "employees_per_manager_low", 3)
    team_high = getattr(cfg, "employees_per_manager_high", 8)

    for m in managers:
        k = int(rng.integers(team_low, team_high + 1))
        for _ in range(k):
            emp_counter += 1
            employee_id = f"E{emp_counter:06d}"
            role = str(rng.choice(roles))
            experience_years = float(np.clip(rng.normal(5.5, 2.2), 0.0, 25.0))

            # Employees under more "open" governance tend to have slightly higher AI familiarity
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


def period_context(rng: np.random.Generator, manager: ManagerProfile) -> Dict[str, float]:
    """
    Sample period-specific context variables used as:
      - panel_manager_period controls
      - decision probabilities and outcome generation
    """
    base_pressure = 0.75 if manager.high_pressure else 0.45
    performance_pressure_index = float(np.clip(rng.normal(base_pressure, 0.08), 0.0, 1.0))

    # Keep target_difficulty separate; you can fold it into performance_pressure_index later if desired
    target_difficulty = float(np.clip(0.35 + 0.7 * performance_pressure_index + rng.normal(0, 0.1), 0.0, 1.0))

    demand_volatility = float(np.clip(rng.beta(2, 4) + 0.15 * rng.random(), 0.0, 1.0))
    task_complexity_index = float(np.clip(rng.beta(2.2, 2.2), 0.0, 1.0))
    supply_disruption_count = float(int(rng.random() < (0.10 + 0.20 * demand_volatility)))

    # Forecast accuracy as MAPE (higher = worse)
    forecast_accuracy_mape = float(np.clip(rng.normal(0.22 + 0.18 * demand_volatility, 0.07), 0.05, 0.60))

    shock_prob = 0.06 + 0.08 * demand_volatility
    recent_negative_shock = float(int(rng.random() < shock_prob))

    return {
        "performance_pressure_index": performance_pressure_index,
        "target_difficulty": target_difficulty,
        "demand_volatility": demand_volatility,
        "task_complexity_index": task_complexity_index,
        "supply_disruption_count": supply_disruption_count,
        "forecast_accuracy_mape": forecast_accuracy_mape,
        "recent_negative_shock": recent_negative_shock,
    }


def sample_ai_confidence(cfg: SimConfig, rng: np.random.Generator, ctx: Dict[str, float], site_complexity: float) -> float:
    """
    Sample AI confidence for an episode.
    Higher complexity, volatility, and site complexity reduce expected confidence.
    """
    complexity = ctx["task_complexity_index"]
    volatility = ctx["demand_volatility"]

    mu = 0.80 - 0.22 * complexity - 0.18 * volatility - 0.12 * site_complexity
    mu = float(np.clip(mu, 0.12, 0.95))

    # calibration controls variance: higher calibration => tighter Beta distribution
    k = 8 + 20 * cfg.confidence_calibration_score
    a = max(1.0, mu * k)
    b = max(1.0, (1 - mu) * k)
    return float(rng.beta(a, b))


def sample_ai_uncertainty(rng: np.random.Generator, ai_confidence: float) -> float:
    """Simple mapping: uncertainty inversely related to confidence, plus noise."""
    u = 1.0 - ai_confidence
    return float(np.clip(rng.normal(u, 0.06), 0.0, 1.0))


def episode_decision_probabilities(
    manager: ManagerProfile,
    ctx: Dict[str, float],
    explanation_capability: str,
    ai_confidence: float,
) -> Tuple[float, float, float]:
    """
    Compute (p_accept, p_modify, p_reject) for a single recommendation episode.

    Interpretation:
      - Latent state s controls baseline willingness (accept ↑ with s)
      - Explanations improve acceptance, reduce rejection
      - Pressure/risk/shock push toward modify/reject (especially at low s)
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

    # Controlled opening: more partial overrides / adjustments by design
    if manager.governance_mode == "controlled_opening":
        modify = clip01(modify + 0.05)
        z = accept + reject + modify
        accept, reject, modify = accept / z, reject / z, modify / z

    z = accept + modify + reject
    return accept / z, modify / z, reject / z


def escalation_probability(cfg: SimConfig, manager: ManagerProfile, ai_confidence: float, explanation_capability: str) -> float:
    """
    Escalation probability (AI workflow escalates to manager review).
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
    HMM-like transition update (low/med/high willingness), driven by performance appraisal + frictions.
    """
    s = manager.state
    pressure = ctx["performance_pressure_index"]

    stay = [0.78, 0.62, 0.80][s]
    up = [0.18, 0.25, 0.00][s]
    down = [0.00, 0.13, 0.18][s]

    perf = float(np.clip(kpi_improvement_score, -2.0, 2.0))
    up_adj = 0.10 * max(0.0, perf)
    down_adj = 0.08 * max(0.0, -perf)

    # Transparency nudges upward movement (trust calibration)
    tr = {"none": 0.00, "basic": 0.03, "detailed": 0.06}[explanation_capability]
    up_adj += tr
    down_adj -= tr / 2

    # High override rates indicate mismatch → more downward tendency
    down_adj += 0.12 * max(0.0, override_rate - 0.35)

    # Pressure amplifies negative learning
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


# -----------------------------
# Main simulation
# -----------------------------
def simulate(cfg: SimConfig) -> Dict[str, pd.DataFrame]:
    """
    Main simulation function updated to match the NEW data tree.

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

    managers = sample_manager_profiles(cfg, rng, site_ids)
    employees = sample_employee_master(cfg, rng, managers)

    # ai_system_master (single deployed version; you can extend to multiple versions later)
    ai_version = getattr(cfg, "ai_version", "v1")
    ai_system_rows = [
        dict(
            ai_version=ai_version,
            deployment_date=str(getattr(cfg, "ai_deployment_date", "2017-01-01")),
            autonomy_level=cfg.autonomy_level,
            explanation_capability="(varies by period)",
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
                governance_mode=m.governance_mode,
                high_pressure=int(m.high_pressure),
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

    # Simulation loop
    for period_id in range(1, cfg.n_periods + 1):
        explanation_capability = explanation_capability_for_period(cfg, period_id)
        base_eps = int(rng.integers(cfg.episodes_per_period_low, cfg.episodes_per_period_high + 1))

        # --- Manager-period loop
        for m in managers:
            ctx = period_context(rng, m)
            n_episodes = int(max(5, rng.integers(max(5, base_eps - 8), base_eps + 9)))

            # period accumulators (manager panel)
            accepted = overridden = escalated = 0
            decision_latency_list: List[float] = []
            correctness_list: List[int] = []
            error_incidents = 0

            # period accumulators (employee panel) by employee_id
            emp_acc = {}
            for e in emp_by_manager.get(m.manager_id, []):
                emp_acc[e.employee_id] = dict(
                    employee_id=e.employee_id,
                    manager_id=m.manager_id,
                    period_id=period_id,
                    site_id=e.site_id,
                    # execution shares (counts first)
                    n_exec=0,
                    n_ai=0,
                    n_human=0,
                    n_joint=0,
                    # outcomes
                    exec_time_list=[],
                    error_count=0,
                    rework_count=0,
                    # coordination/workload
                    ai_support_level_list=[],
                    coordination_complexity=float(np.clip(rng.normal(0.55 + 0.25 * ctx["task_complexity_index"], 0.12), 0.0, 1.0)),
                )

            site_complexity = float(site_complexity_map.get(m.site_id, 0.5))

            for ep in range(1, n_episodes + 1):
                episode_id = f"EP_{m.manager_id}_{period_id:03d}_{ep:03d}"
                execution_id = f"EX_{m.manager_id}_{period_id:03d}_{ep:03d}"

                ai_conf = sample_ai_confidence(cfg, rng, ctx, site_complexity=site_complexity)
                ai_unc = sample_ai_uncertainty(rng, ai_conf)

                # Recommendation type (context label)
                ai_recommendation_type = str(rng.choice(["transfer", "reroute", "reorder", "expedite"]))

                # Episode-level explanation availability
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

                # time_to_decision (minutes)
                base_latency = 2.0 + 6.0 * ctx["task_complexity_index"] + 4.0 * ctx["demand_volatility"]
                base_latency += 3.5 * override_flag + 2.0 * escalation_flag
                base_latency *= (1.10 - 0.08 * m.state)
                time_to_decision = float(np.clip(rng.normal(base_latency, 1.5), 0.5, 25.0))
                decision_latency_list.append(time_to_decision)

                # correctness generation (latent "ground truth" about whether AI recommendation would be correct)
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
                        + 0.10 * (1 if explanation_provided else 0)
                    )
                    decision_correct = int(rng.random() < fix_prob) if ai_correct == 0 else int(
                        rng.random() < (0.88 - 0.10 * ctx["performance_pressure_index"])
                    )
                else:  # reject
                    human_base = 0.62 - 0.10 * ctx["task_complexity_index"] - 0.10 * ctx["demand_volatility"]
                    human_base += 0.06 * m.state
                    decision_correct = int(rng.random() < clip01(human_base))

                correctness_list.append(decision_correct)

                # Decide who executes + AI support intensity (employee-level behavior)
                # Select an employee executing this episode (simple assignment)
                e_list = emp_by_manager.get(m.manager_id, [])
                if not e_list:
                    # Should not happen, but keep safe
                    executor_emp = None
                else:
                    executor_emp = e_list[int(rng.integers(0, len(e_list)))]

                # Execution mode depends on manager_action and autonomy
                if manager_action == "accept":
                    if cfg.autonomy_level == "high":
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.65, 0.25, 0.10])
                    elif cfg.autonomy_level == "medium":
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.45, 0.35, 0.20])
                    else:
                        exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.25, 0.40, 0.35])
                else:
                    exec_mode = choice_with_probs(rng, ["ai", "joint", "human"], [0.10, 0.30, 0.60])

                # AI support level during execution (0..1), higher when joint, and when employee is familiar
                if executor_emp is None:
                    ai_support_level = float(np.clip(rng.beta(2, 3), 0.0, 1.0))
                else:
                    base_support = {"ai": 0.85, "joint": 0.65, "human": 0.30}[exec_mode]
                    ai_support_level = float(np.clip(rng.normal(base_support + 0.20 * executor_emp.ai_familiarity, 0.10), 0.0, 1.0))

                # Employee interventions
                employee_override_during_execution = int((decision_correct == 0) and (rng.random() < 0.08))
                local_adjustment_flag = int(rng.random() < clip01(0.12 + 0.18 * ctx["task_complexity_index"]))

                # Execution outcomes
                # execution_time increases with complexity/volatility and with human involvement
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
                execution_error_flag = int((decision_correct == 0) and (rng.random() < (0.05 + 0.06 * ctx["performance_pressure_index"])))
                # rework as mild proxy (error or override during execution)
                rework_flag = int(execution_error_flag == 1 or employee_override_during_execution == 1)

                # Manager-level "major" incidents
                major_error_flag = int(execution_error_flag == 1 and (rng.random() < 0.35))
                error_incidents += major_error_flag

                # Write decision_episode row
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
                        explanation_provided=explanation_provided,
                        manager_action=manager_action,
                        override_flag=override_flag,
                        escalation_flag=escalation_flag,
                        time_to_decision=time_to_decision,
                    )
                )

                # Write execution_episode row
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
            # panel_manager_period aggregates (HMM estimation table)
            # -----------------------------
            ai_decision_authority_share = accepted / n_episodes
            override_rate = overridden / n_episodes
            escalation_rate = escalated / n_episodes
            decision_latency_avg = float(np.mean(decision_latency_list)) if decision_latency_list else float("nan")

            quality = float(np.mean(correctness_list)) if correctness_list else 0.5
            collab_effect = (ai_decision_authority_share - 0.5) * (quality - 0.5)

            # KPI deltas (signs are arbitrary; treat as deltas vs target/baseline)
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
                + 0.8 * ctx["performance_pressure_index"]
                - 1.1 * collab_effect
                + rng.normal(0, 0.7)
            )
            expedite_cost_delta = float(
                0.6
                + 1.8 * ctx["demand_volatility"]
                + 1.2 * (1 - quality)
                + 1.0 * escalation_rate
                - 0.8 * collab_effect
                + rng.normal(0, 0.7)
            )

            # Feedback score for transition
            kpi_improvement_score = float(
                (+0.7 * (0.0 - service_level_delta)
                 +0.5 * (0.0 - inventory_cost_delta)
                 +0.4 * (0.0 - expedite_cost_delta)
                 -0.8 * error_incidents)
            )

            new_state = update_latent_state(
                cfg=cfg,
                rng=rng,
                manager=m,
                kpi_improvement_score=kpi_improvement_score / 3.0,
                explanation_capability=explanation_capability,
                override_rate=override_rate,
                ctx=ctx,
            )

            panel_manager_rows.append(
                dict(
                    manager_id=m.manager_id,
                    period_id=period_id,
                    # emissions (main HMM)
                    ai_decision_authority_share=ai_decision_authority_share,
                    override_rate=override_rate,
                    escalation_rate=escalation_rate,
                    decision_latency_avg=decision_latency_avg,
                    # KPI appraisal (transition drivers)
                    service_level_delta=service_level_delta,
                    inventory_cost_delta=inventory_cost_delta,
                    expedite_cost_delta=expedite_cost_delta,
                    error_incident_count=int(error_incidents),
                    # incentives/pressure
                    target_difficulty=ctx["target_difficulty"],
                    performance_pressure_index=ctx["performance_pressure_index"],
                    recent_negative_shock=int(ctx["recent_negative_shock"]),
                    # context controls
                    task_complexity_index=ctx["task_complexity_index"],
                    demand_volatility=ctx["demand_volatility"],
                    supply_disruption_count=int(ctx["supply_disruption_count"]),
                    forecast_accuracy_mape=ctx["forecast_accuracy_mape"],
                    ai_version=ai_version,
                    # (optional) keep for ground-truth validation in synthetic data
                    latent_state=m.state,
                    latent_state_next=new_state,
                )
            )

            # update manager state
            m.state = new_state

            # -----------------------------
            # panel_employee_period aggregates (execution behavior/outcomes)
            # -----------------------------
            for _, acc in emp_acc.items():
                n_exec = acc["n_exec"]
                if n_exec <= 0:
                    # If no tasks sampled for this employee in period, keep zeros but still output (optional)
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
                    avg_execution_time = float(np.mean(acc["exec_time_list"]))
                    error_rate = acc["error_count"] / n_exec
                    rework_rate = acc["rework_count"] / n_exec
                    ai_support_intensity = float(np.mean(acc["ai_support_level_list"])) if acc["ai_support_level_list"] else float("nan")
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
