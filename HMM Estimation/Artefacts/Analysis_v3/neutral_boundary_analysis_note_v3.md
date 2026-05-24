# Neutral Boundary and Joint-Path Analysis

## Main finding

The three-state HMM supports a boundary-state interpretation of Neutral. Neutral is not simply a weak form of Appreciation or a weak form of Aversion; its substantive meaning depends on the manager's adjacent state path.

## State ordering

The emission profile validates the ordered interpretation of the states: AI authority rises from 0.022 in Aversion to 0.286 in Neutral and 0.615 in Appreciation. Escalation falls in the opposite direction.

## Medium/Neutral gap

At the intrinsic transition baseline, Aversion moves into Neutral with probability 66.61%, but direct Aversion to Appreciation movement is almost absent. Neutral is highly persistent (90.58%) and has only a 2.56% baseline probability of moving to Appreciation. This means the fitted model does not support the claim that managers usually move quickly from Neutral to Appreciation at baseline. A more accurate interpretation is that Neutral is the main boundary or recalibration state.

The ordered threshold test is consistent with this gap: the Aversion/Neutral cutpoint is mu_1 = 0.776, while the Neutral/Appreciation cutpoint is mu_2 = 1.331. The higher second threshold indicates that reaching Appreciation requires a stronger transition signal than merely leaving Aversion.

## Appreciation-Neutral and Neutral-Aversion joint paths

Among posterior-decoded Neutral observations with a previous state, 73.65% are Neutral persistence, 22.11% enter from Aversion, and 4.24% enter from Appreciation. Therefore, most Neutral observations represent continued uncertainty, but the boundary entries are asymmetric: Aversion -> Neutral is much more common than Appreciation -> Neutral.

The two boundary meanings differ in the next period. After Aversion -> Neutral, the next-period probability of Appreciation is 6.78% and the probability of returning to Aversion is 41.20%. After Appreciation -> Neutral, the next-period probability of returning to Appreciation is 17.22%, while the probability of falling to Aversion is 26.08%. This suggests that Neutral after Aversion is a tentative recovery state, whereas Neutral after Appreciation is a warning or cooling-off state rather than full aversion.

## Performance-signal interpretation

Neutral -> Appreciation is associated with a higher composite KPI score (0.507 vs 0.470 for Neutral -> Aversion; Welch p < 0.001) and a lower override rate (0.601 vs 0.691; Welch p < 0.001). This supports the interpretation that upward movement out of Neutral is tied to stronger realized performance and less manual override.

Appreciation -> Neutral remains behaviorally closer to the high-confidence side than Aversion -> Neutral: AI authority averages 0.437 after Appreciation -> Neutral versus 0.308 after Aversion -> Neutral (Welch p < 0.001). Thus, identical Neutral classifications should not be interpreted as identical managerial positions.

## Covariate sensitivity

The transition-sensitivity analysis shows that peer benchmark performance is the clearest boundary mechanism. Across the observed range of team-vs-peer-average, Neutral -> Appreciation rises from 0.03% to 10.55%, while Neutral -> Aversion falls from 96.44% to 0.01%. For Appreciation managers, Appreciation persistence rises from 2.78% to 99.46%. This is the strongest support for the claim that positive performance signals stabilize Appreciation and protect Neutral managers from sliding into Aversion.

## Paper wording

A defensible paper claim is: The HMM reveals Neutral as a dynamic boundary state. Managers entering Neutral from Aversion are not equivalent to managers entering Neutral from Appreciation; the former pattern reflects tentative recovery from low trust, while the latter reflects a cooling-off or warning state within an otherwise more appreciative trajectory. The model's sequence-based classification is therefore useful because it connects current Neutral behavior to adjacent latent-state paths and to performance signals, rather than treating Neutral as a static middle category.
