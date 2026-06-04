# Conditioned Interaction-Effect Statistical Tests

Posterior-weighted OLS models test whether the response-surface terms are associated with the outcome within each latent state. Observations are weighted by the HMM posterior probability of belonging to the state.

Model specification:

`outcome = beta0 + beta1*X + beta2*X^2 + beta3*Y + beta4*Y^2 + beta5*X*Y + beta6*C + beta7*X*C + beta8*Y*C + error`

- X = Team(t-1) vs. Team(t)
- Y = Team vs. Peer Average
- C = Target Attainment
- Significance: * p < 0.10; ** p < 0.05; *** p < 0.01

## AI Authority Rate

| term                       | Aversion   | Neutral    | Appreciation   |
|:---------------------------|:-----------|:-----------|:---------------|
| Intercept                  | 0.302***   | 0.255***   | 0.403***       |
| X                          | -0.292***  | -0.197***  | -0.295***      |
| X^2                        | 0.463***   | 0.119      | -0.001         |
| Y                          | 2.255***   | 1.972***   | 2.46***        |
| Y^2                        | 3.614***   | 3.479***   | 2.254***       |
| X x Y                      | -1.089***  | -0.507**   | -0.215         |
| Conditioning Benchmark     | 0.424***   | 0.408***   | 0.338***       |
| X x Conditioning Benchmark | 0.115      | 0.071      | 0.166          |
| Y x Conditioning Benchmark | -2.406***  | -2.101***  | -2.2***        |
| R^2                        | 0.609      | 0.581      | 0.675          |
| F-statistic                | 1.459e+03  | 1.963e+03  | 1.008e+03      |
| AIC                        | -3.603e+03 | -6.070e+03 | -2.240e+03     |
| BIC                        | -3.541e+03 | -6.004e+03 | -2.184e+03     |
| Effective n                | 7.509e+03  | 1.135e+04  | 3.893e+03      |

## Escalation Rate

| term                       | Aversion   | Neutral    | Appreciation   |
|:---------------------------|:-----------|:-----------|:---------------|
| Intercept                  | 0.578***   | 0.541***   | 0.607***       |
| X                          | -0.058***  | 0.092***   | -0.008         |
| X^2                        | -0.112     | 0.285**    | -0.084         |
| Y                          | -0.544***  | -0.753***  | -0.461***      |
| Y^2                        | -1.023***  | -0.939***  | -1.388***      |
| X x Y                      | 0.246      | 0.047      | 0.556*         |
| Conditioning Benchmark     | -0.035     | -0.059***  | -0.059**       |
| X x Conditioning Benchmark | 0.172**    | 0.067      | -0.019         |
| Y x Conditioning Benchmark | -0.199     | 0.014      | -0.04          |
| R^2                        | 0.18       | 0.2        | 0.181          |
| F-statistic                | 205.098    | 354.847    | 107.451        |
| AIC                        | -5.849e+03 | -7.091e+03 | -3.490e+03     |
| BIC                        | -5.787e+03 | -7.025e+03 | -3.434e+03     |
| Effective n                | 7.509e+03  | 1.135e+04  | 3.893e+03      |

## Audit Columns

The long CSV keeps the numeric estimates and p-values for these columns:

outcome_label, state, term_label, estimate, p_value, stars, r_squared, f_statistic, aic, bic, n_effective
