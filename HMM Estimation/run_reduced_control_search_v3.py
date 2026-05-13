from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


NOTEBOOK_NAME = "model selection & parameter estimation.ipynb"
ANALYSIS_DIR_NAME = "Analysis_v3"
ARTIFACT_NAME = "best_model_artifacts_v3_2emissions.pkl"
FULL_CONTROL_BACKUP_NAME = "best_model_artifacts_v3_2emissions_full_controls_backup.pkl"
NOTEBOOK_CELLS = [2, 4, 6, 10, 12]


def display_plain(obj: Any = None, **_: Any) -> None:
    if obj is None:
        print()
        return
    if hasattr(obj, "to_string"):
        print(obj.to_string())
        return
    print(obj)


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "Datasets").exists() and (candidate / "HMM Estimation").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def run_notebook_bootstrap(notebook_path: Path) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "display": display_plain,
    }
    for cell_idx in NOTEBOOK_CELLS:
        print(f"--- loading notebook cell {cell_idx} ---", flush=True)
        source = "".join(notebook["cells"][cell_idx].get("source", ""))
        exec(compile(source, f"model_selection_cell_{cell_idx}", "exec"), namespace)
        params_class = namespace.get("Params")
        if params_class is not None:
            setattr(sys.modules["__main__"], "Params", params_class)
    return namespace


def parameter_slices(j_states: int, emissions: int, transitions: int, controls: int) -> dict[str, slice]:
    cursor = 0
    slices: dict[str, slice] = {}
    slices["logit_pi"] = slice(cursor, cursor + j_states)
    cursor += j_states
    slices["alpha"] = slice(cursor, cursor + j_states * j_states)
    cursor += j_states * j_states
    slices["beta"] = slice(cursor, cursor + j_states * j_states * transitions)
    cursor += j_states * j_states * transitions
    slices["mu"] = slice(cursor, cursor + j_states * emissions)
    cursor += j_states * emissions
    slices["W"] = slice(cursor, cursor + j_states * emissions * controls)
    cursor += j_states * emissions * controls
    slices["log_sigma"] = slice(cursor, cursor + j_states * emissions)
    return slices


def stars_from_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def make_warm_start(full_model: Any, full_controls: list[str], candidate_controls: list[str], full_transitions: list[str], candidate_transitions: list[str], params_class: type) -> Any:
    control_idx = [full_controls.index(col) for col in candidate_controls]
    transition_idx = [full_transitions.index(col) for col in candidate_transitions]
    return params_class(
        logit_pi=full_model.logit_pi.copy(),
        alpha=full_model.alpha.copy(),
        beta=full_model.beta[:, :, transition_idx].copy(),
        mu=full_model.mu.copy(),
        W=full_model.W[:, :, control_idx].copy(),
        log_sigma=full_model.log_sigma.copy(),
    )


def estimate_significance(model: Any, res: Any, data: Any, emission_cols: list[str], control_cols: list[str], transition_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    j_states, emissions, transitions = model.beta.shape[0], model.mu.shape[1], model.beta.shape[2]
    controls = model.W.shape[2]
    slices = parameter_slices(j_states, emissions, transitions, controls)

    se_vec = np.full(len(res.x), np.nan)
    cov = None
    if hasattr(res, "hess_inv") and hasattr(res.hess_inv, "todense"):
        cov_candidate = np.asarray(res.hess_inv.todense())
        if cov_candidate.shape == (len(res.x), len(res.x)):
            cov = cov_candidate
            se_vec = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    mu_orig = data.y_scaler.inverse_transform(model.mu)
    mu_se = se_vec[slices["mu"]].reshape(j_states, emissions) * data.y_scaler.scale_[None, :]
    w_effect = model.W * data.y_scaler.scale_[None, :, None]
    w_se = se_vec[slices["W"]].reshape(j_states, emissions, controls) * data.y_scaler.scale_[None, :, None]
    beta_se = se_vec[slices["beta"]].reshape(j_states, j_states, transitions)

    rows = []

    def add_row(section: str, variable: str, estimate: float, se: float, **extra: Any) -> None:
        z_stat = float(estimate / se) if np.isfinite(se) and se > 0 else np.nan
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat)))) if np.isfinite(z_stat) else np.nan
        row = {
            "section": section,
            "variable": variable,
            "estimate": float(estimate),
            "standard_error": float(se) if np.isfinite(se) else np.nan,
            "z_stat": z_stat,
            "p_value": p_value,
            "significance": stars_from_p(p_value),
        }
        row.update(extra)
        rows.append(row)

    for state_idx in range(j_states):
        for emission_idx, emission in enumerate(emission_cols):
            add_row(
                "Emission baseline mean",
                emission,
                mu_orig[state_idx, emission_idx],
                mu_se[state_idx, emission_idx],
                state_idx=state_idx,
                emission=emission,
                control="",
                from_state="",
                to_state="",
            )

    for state_idx in range(j_states):
        for emission_idx, emission in enumerate(emission_cols):
            for control_idx, control in enumerate(control_cols):
                add_row(
                    "Emission control effect",
                    f"{emission}|{control}",
                    w_effect[state_idx, emission_idx, control_idx],
                    w_se[state_idx, emission_idx, control_idx],
                    state_idx=state_idx,
                    emission=emission,
                    control=control,
                    from_state="",
                    to_state="",
                )

    for from_idx in range(j_states):
        for to_idx in range(j_states):
            for transition_idx, transition in enumerate(transition_cols):
                add_row(
                    "Transition beta",
                    transition,
                    model.beta[from_idx, to_idx, transition_idx],
                    beta_se[from_idx, to_idx, transition_idx],
                    state_idx="",
                    emission="",
                    control="",
                    from_state=from_idx,
                    to_state=to_idx,
                )

    detail = pd.DataFrame(rows)

    joint_rows = []
    if cov is not None and controls:
        w_slice = slices["W"]
        w_cov = cov[w_slice, w_slice]
        w_vec = model.W.ravel()
        for control_idx, control in enumerate(control_cols):
            local_idx = [
                (state_idx * emissions + emission_idx) * controls + control_idx
                for state_idx in range(j_states)
                for emission_idx in range(emissions)
            ]
            coef = w_vec[local_idx]
            coef_cov = w_cov[np.ix_(local_idx, local_idx)]
            stat = float(coef @ np.linalg.pinv(coef_cov) @ coef)
            df = len(local_idx)
            p_value = float(chi2.sf(stat, df))
            joint_rows.append({
                "control": control,
                "joint_wald_chi2": stat,
                "df": df,
                "p_value": p_value,
                "significance": stars_from_p(p_value),
            })
    joint = pd.DataFrame(joint_rows)
    return detail, joint


def candidate_sets(full_controls: list[str]) -> list[tuple[str, list[str]]]:
    priority = [
        "decision_latency",
        "task_complexity",
        "target_difficulty",
        "demand_volatility",
        "forecast_accuracy",
        "performance_pressure",
        "supply_disruptions",
        "recent_negative_shock",
    ]
    priority = [col for col in priority if col in full_controls]

    candidates: list[tuple[str, list[str]]] = [("k0__no_emission_controls", [])]
    for control in priority:
        candidates.append((f"k1__{control}", [control]))

    pair_controls = [
        ("decision_latency", "task_complexity"),
        ("decision_latency", "target_difficulty"),
        ("decision_latency", "demand_volatility"),
        ("task_complexity", "target_difficulty"),
        ("forecast_accuracy", "supply_disruptions"),
        ("performance_pressure", "recent_negative_shock"),
    ]
    for pair in pair_controls:
        controls = [col for col in pair if col in full_controls]
        if len(controls) == 2:
            candidates.append((f"k2__{controls[0]}__{controls[1]}", controls))

    triple_controls = [
        ("decision_latency", "task_complexity", "target_difficulty"),
        ("decision_latency", "demand_volatility", "task_complexity"),
        ("forecast_accuracy", "supply_disruptions", "target_difficulty"),
    ]
    for triple in triple_controls:
        controls = [col for col in triple if col in full_controls]
        if len(controls) == 3:
            candidates.append((f"k3__{controls[0]}__{controls[1]}__{controls[2]}", controls))

    seen = set()
    unique_candidates = []
    for name, controls in candidates:
        key = tuple(controls)
        if key not in seen:
            unique_candidates.append((name, controls))
            seen.add(key)
    return unique_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiter", type=int, default=260)
    parser.add_argument("--maxfun", type=int, default=250_000)
    parser.add_argument("--time-cap-min", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N candidates; 0 means all.")
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    notebook_path = repo_root / "HMM Estimation" / NOTEBOOK_NAME
    artifact_dir = repo_root / "HMM Estimation" / "Artefacts"
    analysis_dir = artifact_dir / ANALYSIS_DIR_NAME
    analysis_dir.mkdir(parents=True, exist_ok=True)

    namespace = run_notebook_bootstrap(notebook_path)
    params_class = namespace["Params"]
    setattr(sys.modules["__main__"], "Params", params_class)

    artifact_path = artifact_dir / ARTIFACT_NAME
    with artifact_path.open("rb") as f:
        full_artifact = pickle.load(f)
    if not full_artifact.get("control_cols"):
        backup_path = artifact_dir / FULL_CONTROL_BACKUP_NAME
        if not backup_path.exists():
            raise FileNotFoundError(
                f"Active artifact has no emission controls and full-control backup is missing: {backup_path}"
            )
        with backup_path.open("rb") as f:
            full_artifact = pickle.load(f)

    full_model = full_artifact["best_model"]
    full_controls = list(full_artifact["control_cols"])
    transition_cols = list(full_artifact["transition_cols"])
    emission_cols = list(full_artifact["emission_cols"])
    candidate_plan = candidate_sets(full_controls)
    if args.limit > 0:
        candidate_plan = candidate_plan[: args.limit]

    summary_path = analysis_dir / "reduced_control_search_v3.csv"
    detail_path = analysis_dir / "reduced_control_search_significance_v3.csv"
    joint_path = analysis_dir / "reduced_control_search_joint_wald_v3.csv"
    candidate_artifact_dir = artifact_dir / "Reduced_Control_Candidates_v3"
    candidate_artifact_dir.mkdir(parents=True, exist_ok=True)

    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    completed = set(existing_summary["candidate"].tolist()) if not existing_summary.empty and "candidate" in existing_summary else set()
    summary_rows = existing_summary.to_dict("records") if not existing_summary.empty else []
    detail_frames = [pd.read_csv(detail_path)] if detail_path.exists() else []
    joint_frames = [pd.read_csv(joint_path)] if joint_path.exists() else []

    for candidate_name, controls in candidate_plan:
        if candidate_name in completed:
            print(f"Skipping completed candidate {candidate_name}", flush=True)
            continue

        print()
        print(f"=== Candidate {candidate_name}: controls={controls}; transitions={transition_cols} ===", flush=True)
        namespace["CONTROL_COLS"] = list(controls)
        data = namespace["load_sequences"](namespace["DATA_PATH"], transition_cols_override=transition_cols)
        y_stack = np.stack(data.Y)
        x_stack = np.stack(data.X)
        z_stack = np.stack(data.Z)
        warm_start = make_warm_start(full_model, full_controls, controls, transition_cols, transition_cols, params_class)

        started = time.time()
        model, res, is_conv = namespace["fit_model_batched"](
            J=int(full_artifact["best_J"]),
            Y_stack=y_stack,
            X_stack=x_stack,
            Z_stack=z_stack,
            maxiter=args.maxiter,
            maxfun=args.maxfun,
            n_starts=1,
            seed=20260508 + len(summary_rows),
            sigma_min=0.1,
            sigma_max=3.5,
            print_every=100,
            time_cap_min=args.time_cap_min,
            l2=0.01,
            ftol=1e-8,
            gtol=2e-6,
            warm_starts=[warm_start],
            use_subset=False,
            do_emission_only_warmstart=False,
            emission_only_maxiter=0,
            jitter_warm_starts=False,
        )
        runtime_min = (time.time() - started) / 60.0

        detail, joint = estimate_significance(model, res, data, emission_cols, controls, transition_cols)
        detail.insert(0, "candidate", candidate_name)
        detail.insert(1, "controls", ", ".join(controls))
        joint.insert(0, "candidate", candidate_name)
        joint.insert(1, "controls", ", ".join(controls))

        emission_detail = detail[detail["section"].eq("Emission control effect")].copy()
        individual_sig10 = int((emission_detail["p_value"] < 0.10).sum())
        individual_sig05 = int((emission_detail["p_value"] < 0.05).sum())
        individual_sig01 = int((emission_detail["p_value"] < 0.01).sum())
        joint_sig10 = int((joint["p_value"] < 0.10).sum()) if not joint.empty else 0
        joint_sig05 = int((joint["p_value"] < 0.05).sum()) if not joint.empty else 0
        joint_sig01 = int((joint["p_value"] < 0.01).sum()) if not joint.empty else 0
        all_joint_sig05 = bool(len(joint) > 0 and (joint["p_value"] < 0.05).all())
        all_joint_sig10 = bool(len(joint) > 0 and (joint["p_value"] < 0.10).all())

        n_obs = int(y_stack.shape[0] * y_stack.shape[1])
        ll_total = float(getattr(res, "true_ll", np.nan))
        k_params = int(len(getattr(res, "x", [])))
        aic = 2.0 * k_params - 2.0 * ll_total
        bic = np.log(n_obs) * k_params - 2.0 * ll_total
        summary_row = {
            "candidate": candidate_name,
            "controls": ", ".join(controls),
            "K": len(controls),
            "transition_cols": ", ".join(transition_cols),
            "P": len(transition_cols),
            "LL": ll_total,
            "AIC": aic,
            "BIC": bic,
            "k_params": k_params,
            "n_obs": n_obs,
            "soft_converged": bool(is_conv),
            "scipy_success": bool(getattr(res, "success", False)),
            "iterations": int(getattr(res, "nit", -1)) if getattr(res, "nit", None) is not None else np.nan,
            "runtime_min": runtime_min,
            "individual_control_cells_p_lt_0_10": individual_sig10,
            "individual_control_cells_p_lt_0_05": individual_sig05,
            "individual_control_cells_p_lt_0_01": individual_sig01,
            "joint_controls_p_lt_0_10": joint_sig10,
            "joint_controls_p_lt_0_05": joint_sig05,
            "joint_controls_p_lt_0_01": joint_sig01,
            "all_retained_controls_joint_p_lt_0_10": all_joint_sig10,
            "all_retained_controls_joint_p_lt_0_05": all_joint_sig05,
            "message": str(getattr(res, "message", "")),
        }
        summary_rows.append(summary_row)
        detail_frames.append(detail)
        joint_frames.append(joint)

        candidate_artifact = {
            **full_artifact,
            "best_model": model,
            "best_res_final": res,
            "best_is_conv_final": bool(is_conv and getattr(res, "success", False)),
            "control_cols": list(controls),
            "transition_cols": list(transition_cols),
            "best_spec": f"{full_artifact.get('best_spec', 'single_period')}_reduced_controls",
            "ll_total": ll_total,
            "aic": aic,
            "bic": bic,
            "k_params": k_params,
            "n_obs_total": n_obs,
            "control_selection": summary_row,
            "control_selection_source": "reduced_control_search_v3",
            "z_scaler": data.z_scaler,
            "x_scaler": data.x_scaler,
            "y_scaler": data.y_scaler,
        }
        with (candidate_artifact_dir / f"{candidate_name}.pkl").open("wb") as f:
            pickle.dump(candidate_artifact, f)

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        pd.concat(detail_frames, ignore_index=True).to_csv(detail_path, index=False)
        pd.concat(joint_frames, ignore_index=True).to_csv(joint_path, index=False)

        print(
            f"Candidate {candidate_name}: joint_sig05={joint_sig05}/{len(controls)}, "
            f"individual_sig05={individual_sig05}/{len(emission_detail)}, "
            f"LL={ll_total:.2f}, BIC={bic:.2f}, runtime={runtime_min:.1f} min",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        ranked = summary.sort_values(
            [
                "all_retained_controls_joint_p_lt_0_05",
                "joint_controls_p_lt_0_05",
                "K",
                "individual_control_cells_p_lt_0_05",
                "BIC",
            ],
            ascending=[False, False, False, False, True],
        )
        print()
        print("Top reduced-control candidates:")
        print(ranked.head(10).to_string(index=False))
        print(f"Saved summary to {summary_path}")
        print(f"Saved coefficient detail to {detail_path}")
        print(f"Saved joint Wald detail to {joint_path}")


if __name__ == "__main__":
    main()
