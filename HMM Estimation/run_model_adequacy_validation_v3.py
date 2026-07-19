from __future__ import annotations

import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "HMM Estimation" / "Artefacts"
ANALYSIS_DIR = ARTIFACT_DIR / "Analysis_v3"
MODEL_PATH = ARTIFACT_DIR / "best_model_artifacts_v3_2emissions.pkl"
RESULT_PATH = ANALYSIS_DIR / "model_adequacy_validation_v3.csv"
REPORT_PATH = ANALYSIS_DIR / "model_adequacy_validation_v3.md"


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
    """Load the artifact created with the project's older pandas version."""
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


def load_artifact() -> dict[str, object]:
    patch_pandas_stringarray_pickle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MODEL_PATH.open("rb") as stream:
            return pickle.load(stream)


def add(
    rows: list[dict[str, object]],
    dimension: str,
    test: str,
    status: str,
    value: object,
    criterion: str,
    interpretation: str,
) -> None:
    rows.append(
        {
            "dimension": dimension,
            "test": test,
            "status": status,
            "value": value,
            "criterion": criterion,
            "interpretation": interpretation,
        }
    )


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_artifact()
    model = artifact["best_model"]
    result = artifact["best_res_final"]
    comparison = artifact["model_comparison_results"].copy()
    rows: list[dict[str, object]] = []

    # Estimation and numerical adequacy.
    scipy_success = bool(getattr(result, "success", False))
    strict = bool(artifact.get("best_is_conv_final", False))
    add(
        rows,
        "Statistical fit",
        "Selected-fit optimizer exit",
        "PASS" if scipy_success and strict else "FAIL",
        f"SciPy success={scipy_success}; strict/soft convergence={strict}",
        "Both optimizer success and the project's convergence gate must hold",
        "The selected three-state, single-period fit passed the stored convergence gate.",
    )

    jac = np.asarray(getattr(result, "jac", []), dtype=float)
    jac_inf = float(np.max(np.abs(jac))) if jac.size else np.nan
    add(
        rows,
        "Statistical fit",
        "First-order stationarity",
        "PASS" if np.isfinite(jac_inf) and jac_inf <= 0.1 else "CAUTION",
        f"max |gradient|={jac_inf:.4g}",
        "Diagnostic tolerance: max absolute numerical gradient <= 0.1",
        "The relative-function convergence exit is not accompanied by a small stored numerical gradient.",
    )

    sigma = np.exp(model.log_sigma)
    cfg = artifact["final_cfg"]["common_fit_config"]
    sigma_min, sigma_max = float(cfg["sigma_min"]), float(cfg["sigma_max"])
    near_bound = bool(np.any(sigma <= sigma_min * 1.01) or np.any(sigma >= sigma_max / 1.01))
    add(
        rows,
        "Statistical fit",
        "Emission variance boundary",
        "CAUTION" if near_bound else "PASS",
        f"sigma range={sigma.min():.4f} to {sigma.max():.4f}",
        f"No fitted sigma within 1% of bounds [{sigma_min}, {sigma_max}]",
        "Boundary variances can indicate a degenerate state or an unstable likelihood optimum.",
    )

    # State-count and transition-window selection, using converged candidates only.
    converged = comparison[comparison["strict_converged"].astype(bool)].copy()
    selected_spec, selected_j = str(artifact["best_spec"]), int(artifact["best_J"])
    selected_bic = float(artifact["bic"])
    best_converged = converged.sort_values("BIC").iloc[0]
    selection_pass = str(best_converged["spec"]) == selected_spec and int(best_converged["J"]) == selected_j
    add(
        rows,
        "Statistical fit",
        "AIC/BIC selection among core converged candidates",
        "PASS" if selection_pass else "FAIL",
        f"selected BIC={selected_bic:.2f}; best converged={best_converged['spec']}, J={int(best_converged['J'])}, BIC={float(best_converged['BIC']):.2f}",
        "Lowest BIC among strictly converged core candidates",
        "The converged core comparison supports three rather than two states and the single-period transition window.",
    )

    extra = pd.read_csv(ARTIFACT_DIR / "model_selection_v3_extra_states.csv")
    extra_conv = extra[extra["soft_converged"].astype(str).str.lower().eq("true")]
    extra_best = extra_conv.sort_values("BIC").iloc[0]
    add(
        rows,
        "Statistical fit",
        "Additional state-count search",
        "PASS" if selected_bic < float(extra_best["BIC"]) else "CAUTION",
        f"selected J=3 BIC={selected_bic:.2f}; best converged extra J={int(extra_best['J'])} BIC={float(extra_best['BIC']):.2f}",
        "Selected model beats every converged J=1,4,5 candidate on BIC",
        "J=5 did not converge. The converged J=4 solution also has a worse LL than J=3, indicating a local optimum rather than a definitive rejection of four states.",
    )

    higher = pd.read_csv(ARTIFACT_DIR / "model_selection_v3_higher_order.csv")
    provisional = higher.sort_values("BIC").iloc[0]
    add(
        rows,
        "Robustness",
        "Higher-order transition comparison",
        "CAUTION",
        f"provisional best: order=2, J={int(provisional['J'])}, BIC={float(provisional['BIC']):.2f}, converged={bool(provisional['strict_converged'])}",
        "A lower-BIC higher-order model must converge before it can challenge the selected model",
        "The second-order four-state screening fit has a lower provisional BIC but hit its EM iteration limit.",
    )

    # Posterior state quality.
    selected_row = comparison[(comparison["spec"] == selected_spec) & (comparison["J"] == selected_j)].iloc[0]
    occupancy = float(selected_row["occupancy_min"])
    certainty = float(selected_row["certainty_mean"])
    add(
        rows,
        "Structural adequacy",
        "Minimum state occupancy",
        "PASS" if occupancy >= 0.05 else "FAIL",
        f"minimum posterior occupancy={occupancy:.3f}",
        "Each state has at least 5% posterior occupancy",
        "No selected state is empirically empty or vanishingly small.",
    )
    add(
        rows,
        "Structural adequacy",
        "Posterior classification certainty",
        "PASS" if certainty >= 0.80 else "CAUTION",
        f"mean maximum posterior probability={certainty:.3f}",
        "Mean maximum posterior probability >= 0.80",
        "The fitted states are sharply classified in sample.",
    )

    original_mu = artifact["y_scaler"].inverse_transform(model.mu)
    ordered = np.argsort(original_mu[:, 0])
    authority = original_mu[ordered, 0]
    escalation = original_mu[ordered, 1]
    monotone = bool(np.all(np.diff(authority) > 0) and np.all(np.diff(escalation) < 0))
    standardized_mu = model.mu[ordered]
    pair_dist = [float(np.linalg.norm(standardized_mu[j] - standardized_mu[i])) for i in range(selected_j) for j in range(i + 1, selected_j)]
    min_distance = min(pair_dist)
    add(
        rows,
        "Structural adequacy",
        "State separation and interpretation",
        "PASS" if monotone and min_distance >= 1.0 else "CAUTION",
        f"authority means={np.round(authority, 3).tolist()}; escalation means={np.round(escalation, 3).tolist()}; min standardized centroid distance={min_distance:.2f}",
        "Theory-consistent monotonic profiles and minimum standardized centroid distance >= 1",
        "The profiles are theory-consistent, but the closest pair of state centroids is only moderately separated under the stated diagnostic threshold.",
    )

    q0 = softmax(model.alpha, axis=1)
    row_error = float(np.max(np.abs(q0.sum(axis=1) - 1.0)))
    add(
        rows,
        "Structural adequacy",
        "Transition matrix validity at mean covariates",
        "PASS" if np.isfinite(q0).all() and row_error < 1e-10 and np.all(q0 > 0) else "FAIL",
        f"max row-sum error={row_error:.2e}; probability range={q0.min():.4f} to {q0.max():.4f}",
        "Finite, positive probabilities with rows summing to one",
        "The baseline transition matrix is numerically valid.",
    )

    # Heterogeneity and evidence gaps.
    heterogeneity = pd.read_csv(ANALYSIS_DIR / "manager_heterogeneity_tests_v3.csv")
    transition_het = heterogeneity.loc[heterogeneity["factor"].eq("delta")].iloc[0]
    emission_het = heterogeneity.loc[heterogeneity["factor"].eq("xi")].iloc[0]
    has_random_effect = hasattr(model, "xi") or hasattr(model, "delta")
    add(
        rows,
        "Structural adequacy",
        "Manager heterogeneity accounted for in likelihood",
        "PASS" if has_random_effect else "FAIL",
        f"transition p={float(transition_het['p_value']):.3g}, ICC={float(transition_het['icc']):.3f}; emission p={float(emission_het['p_value']):.3g}, ICC={float(emission_het['icc']):.3f}; random-effect parameter present={has_random_effect}",
        "Material detected manager heterogeneity is modeled inside the HMM likelihood",
        "The repository detects strong manager heterogeneity post-HMM, but the fitted Params object contains no manager random effect; spurious state dependence remains a live risk.",
    )

    trained_path = Path(str(artifact["dataset_path"]))
    active_path = ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx"
    add(
        rows,
        "Reproducibility",
        "Fitted dataset is present",
        "PASS" if trained_path.exists() else "FAIL",
        str(trained_path),
        "The exact workbook named by the model artifact remains available",
        "The fitted artifact can be tied to its stated calibration workbook.",
    )
    add(
        rows,
        "Reproducibility",
        "Post-estimation scripts use fitted dataset",
        "CAUTION" if trained_path.resolve() != active_path.resolve() else "PASS",
        f"fitted={trained_path.name}; analysis-script default={active_path.name}",
        "Post-estimation scripts and model artifact use the same workbook",
        "Several analysis scripts default to a different active workbook, so downstream tables can mix a calibrated fit with non-calibrated panel values.",
    )

    for test, interpretation in [
        ("Manager-level or temporal holdout log-likelihood", "No model refit on a training split and evaluation on untouched managers/periods is stored."),
        ("Out-of-sample RMSPE / predictive density", "No untouched holdout predictions are stored."),
        ("Parameter and latent-state recovery", "The available archived truth workbook is a different simulation and cannot validate this fit."),
        ("Sensitivity to initial state probabilities and random starts", "Continuation checkpoints do not constitute a documented multi-start stability distribution."),
        ("Static latent-class / non-dynamic benchmark", "J=1 is available, but no same-state-count static latent-class benchmark is stored."),
    ]:
        add(rows, "Predictive/robustness evidence", test, "NOT TESTED", "not available", "Required for a complete literature-aligned adequacy claim", interpretation)

    results = pd.DataFrame(rows)
    results.to_csv(RESULT_PATH, index=False)

    counts = results["status"].value_counts().to_dict()
    verdict = "NOT YET ADEQUATELY VALIDATED"
    report = f"""# HMM Model Adequacy Validation (V3)

## Verdict

**{verdict}.** The selected first-order, three-state model has strong in-sample separation, valid transition probabilities, high posterior certainty, non-trivial state occupancy, and the lowest BIC among the converged core and extra-state fits. Those results establish a credible in-sample descriptive HMM.

They do not establish full model adequacy. The decisive gaps are the absence of genuine out-of-sample refitting/evaluation, unavailable parameter/state recovery for this exact simulation, untested initialization sensitivity, and a missing same-state-count static latent-class benchmark. In addition, significant manager heterogeneity is detected after estimation but is not included in the HMM likelihood, leaving a risk of spurious state dependence. The stored optimizer gradient also warrants a tighter continuation check.

Status totals: PASS={counts.get('PASS', 0)}, CAUTION={counts.get('CAUTION', 0)}, FAIL={counts.get('FAIL', 0)}, NOT TESTED={counts.get('NOT TESTED', 0)}.

## Test results

{markdown_table(results[['dimension', 'test', 'status', 'value', 'interpretation']])}

## Interpretation of model selection

- The converged core fits favor `single_period, J=3` (BIC {selected_bic:.2f}) over `J=2` and over the three-period alternatives.
- Converged J=1 and J=4 extra-state fits have substantially worse BIC. J=5 did not converge.
- The lower provisional BIC from the second-order J=4 screening model is not admissible selection evidence because that fit stopped at the EM iteration limit.
- LMD and MSC are not reported because this is a frequentist maximum-likelihood HMM and the repository does not implement those criteria.

## Required next validation stage

1. Split by manager (not by individual rows), refit every candidate on the training managers, and compare untouched-manager log predictive density and emission RMSPE.
2. Fit a same-state-count static latent-class model and compare holdout predictive density against the HMM.
3. Refit the selected HMM across multiple dispersed initializations and report likelihood, parameter, and decoded-state stability after label alignment.
4. Add manager random effects (or a correlated-random-effects approximation) inside the likelihood and re-check state persistence.
5. Regenerate a known-truth dataset with the current generator and run parameter/state recovery on that exact dataset.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved: {RESULT_PATH}")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
