# Higher-Order HMM Comparison

The higher-order model is a second-order nonhomogeneous HMM: transitions use the two-state latent history,
`P(s_t | s_{t-1}, s_{t-2}, x_t)`, while emissions remain tied to the current latent state.

## Overall Selection

- Best AIC: order=2, spec=single_period, J=4, AIC=15373.22, BIC=18232.72.
- Best BIC: order=2, spec=single_period, J=4, AIC=15373.22, BIC=18232.72.

## Paired Results

- single_period J=2: delta LL=-47.95, delta AIC=135.91, delta BIC=296.55; BIC prefers first_order.
- single_period J=3: delta LL=118.25, delta AIC=-74.50, delta BIC=576.12; BIC prefers first_order.
- single_period J=4: delta LL=12625.52, delta AIC=-24835.04, delta BIC=-23164.32; BIC prefers higher_order.

## Convergence Caution

The higher-order rows are EM screening fits. In the current run they reached the configured EM iteration limit, so the information criteria should be treated as provisional until continued to a tighter EM tolerance.

Lower AIC/BIC is better. Positive delta LL means the higher-order model fits the observed sequences better before penalizing extra parameters.
