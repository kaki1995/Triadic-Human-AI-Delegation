# HMM Model Adequacy Validation (V3)

## Verdict

**NOT YET ADEQUATELY VALIDATED.** The selected first-order, three-state model has strong in-sample separation, valid transition probabilities, high posterior certainty, non-trivial state occupancy, and the lowest BIC among the converged core and extra-state fits. Those results establish a credible in-sample descriptive HMM.

They do not establish full model adequacy. The decisive gaps are the absence of genuine out-of-sample refitting/evaluation, unavailable parameter/state recovery for this exact simulation, untested initialization sensitivity, and a missing same-state-count static latent-class benchmark. In addition, significant manager heterogeneity is detected after estimation but is not included in the HMM likelihood, leaving a risk of spurious state dependence. The stored optimizer gradient also warrants a tighter continuation check.

Status totals: PASS=7, CAUTION=5, FAIL=1, NOT TESTED=5.

## Test results

| dimension | test | status | value | interpretation |
|---|---|---|---|---|
| Statistical fit | Selected-fit optimizer exit | PASS | SciPy success=True; strict/soft convergence=True | The selected three-state, single-period fit passed the stored convergence gate. |
| Statistical fit | First-order stationarity | CAUTION | max \|gradient\|=4536 | The relative-function convergence exit is not accompanied by a small stored numerical gradient. |
| Statistical fit | Emission variance boundary | CAUTION | sigma range=0.1000 to 0.4474 | Boundary variances can indicate a degenerate state or an unstable likelihood optimum. |
| Statistical fit | AIC/BIC selection among core converged candidates | PASS | selected BIC=19672.98; best converged=single_period, J=3, BIC=19672.98 | The converged core comparison supports three rather than two states and the single-period transition window. |
| Statistical fit | Additional state-count search | PASS | selected J=3 BIC=19672.98; best converged extra J=4 BIC=41397.04 | J=5 did not converge. The converged J=4 solution also has a worse LL than J=3, indicating a local optimum rather than a definitive rejection of four states. |
| Robustness | Higher-order transition comparison | CAUTION | provisional best: order=2, J=4, BIC=18232.72, converged=False | The second-order four-state screening fit has a lower provisional BIC but hit its EM iteration limit. |
| Structural adequacy | Minimum state occupancy | PASS | minimum posterior occupancy=0.171 | No selected state is empirically empty or vanishingly small. |
| Structural adequacy | Posterior classification certainty | PASS | mean maximum posterior probability=0.944 | The fitted states are sharply classified in sample. |
| Structural adequacy | State separation and interpretation | CAUTION | authority means=[0.038, 0.224, 0.441]; escalation means=[0.593, 0.569, 0.506]; min standardized centroid distance=0.76 | The profiles are theory-consistent, but the closest pair of state centroids is only moderately separated under the stated diagnostic threshold. |
| Structural adequacy | Transition matrix validity at mean covariates | PASS | max row-sum error=1.11e-16; probability range=0.0012 to 0.9058 | The baseline transition matrix is numerically valid. |
| Structural adequacy | Manager heterogeneity accounted for in likelihood | FAIL | transition p=1.27e-37, ICC=0.083; emission p=1.53e-10, ICC=0.046; random-effect parameter present=False | The repository detects strong manager heterogeneity post-HMM, but the fitted Params object contains no manager random effect; spurious state dependence remains a live risk. |
| Reproducibility | Fitted dataset is present | PASS | C:\Users\Admin\OneDrive\Desktop\Algorithm-Appreciation-and-Aversion-in-Triadic-Delegation-Settings\Datasets\Triadic_Delegation_Analysis_Dataset_v3_state_trend_calibrated.xlsx | The fitted artifact can be tied to its stated calibration workbook. |
| Reproducibility | Post-estimation scripts use fitted dataset | CAUTION | fitted=Triadic_Delegation_Analysis_Dataset_v3_state_trend_calibrated.xlsx; analysis-script default=Triadic_Delegation_Analysis_Dataset_v3.xlsx | Several analysis scripts default to a different active workbook, so downstream tables can mix a calibrated fit with non-calibrated panel values. |
| Predictive/robustness evidence | Manager-level or temporal holdout log-likelihood | NOT TESTED | not available | No model refit on a training split and evaluation on untouched managers/periods is stored. |
| Predictive/robustness evidence | Out-of-sample RMSPE / predictive density | NOT TESTED | not available | No untouched holdout predictions are stored. |
| Predictive/robustness evidence | Parameter and latent-state recovery | NOT TESTED | not available | The available archived truth workbook is a different simulation and cannot validate this fit. |
| Predictive/robustness evidence | Sensitivity to initial state probabilities and random starts | NOT TESTED | not available | Continuation checkpoints do not constitute a documented multi-start stability distribution. |
| Predictive/robustness evidence | Static latent-class / non-dynamic benchmark | NOT TESTED | not available | J=1 is available, but no same-state-count static latent-class benchmark is stored. |

## Interpretation of model selection

- The converged core fits favor `single_period, J=3` (BIC 19672.98) over `J=2` and over the three-period alternatives.
- Converged J=1 and J=4 extra-state fits have substantially worse BIC. J=5 did not converge.
- The lower provisional BIC from the second-order J=4 screening model is not admissible selection evidence because that fit stopped at the EM iteration limit.
- LMD and MSC are not reported because this is a frequentist maximum-likelihood HMM and the repository does not implement those criteria.

## Required next validation stage

1. Split by manager (not by individual rows), refit every candidate on the training managers, and compare untouched-manager log predictive density and emission RMSPE.
2. Fit a same-state-count static latent-class model and compare holdout predictive density against the HMM.
3. Refit the selected HMM across multiple dispersed initializations and report likelihood, parameter, and decoded-state stability after label alignment.
4. Add manager random effects (or a correlated-random-effects approximation) inside the likelihood and re-check state persistence.
5. Regenerate a known-truth dataset with the current generator and run parameter/state recovery on that exact dataset.
