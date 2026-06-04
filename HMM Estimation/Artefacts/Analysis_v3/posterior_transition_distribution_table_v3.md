# Posterior Transition Distribution Table

Cells report the mean manager-level decoded posterior transition rate,
with the 2.5th and 97.5th percentiles of the across-manager distribution
in brackets. Managers are included in a row only when they are observed
at least once in the corresponding origin state.

## Mean Manager-Level Posterior Transitions

| State at t-1   | State at t: Aversion   | State at t: Neutral   | State at t: Appreciation   |
|:---------------|:-----------------------|:----------------------|:---------------------------|
| Aversion       | 60.4% [0.0%-93.8%]     | 38.0% [0.0%-100.0%]   | 1.6% [0.0%-14.3%]          |
| Neutral        | 28.2% [0.0%-71.4%]     | 65.0% [22.2%-92.3%]   | 6.8% [0.0%-20.2%]          |
| Appreciation   | 0.3% [0.0%-0.0%]       | 21.1% [0.0%-100.0%]   | 78.5% [0.0%-100.0%]        |

## Posterior Propensity to Stay in Each State

| state        |   n_managers_with_origin_state |   total_origin_transitions |   total_cell_transitions | pooled_transition_rate   | mean_stay_propensity   | median_stay_propensity   | sd_stay_propensity   | p025_stay_propensity   | p25_stay_propensity   | p75_stay_propensity   | p975_stay_propensity   |
|:-------------|-------------------------------:|---------------------------:|-------------------------:|:-------------------------|:-----------------------|:-------------------------|:---------------------|:-----------------------|:----------------------|:----------------------|:-----------------------|
| Aversion     |                            826 |                       7473 |                     5128 | 68.6%                    | 60.4%                  | 66.7%                    | 23.9%                | 0.0%                   | 50.0%                 | 77.6%                 | 93.8%                  |
| Neutral      |                            875 |                      10856 |                     7499 | 69.1%                    | 65.0%                  | 66.7%                    | 18.5%                | 22.2%                  | 55.6%                 | 78.9%                 | 92.3%                  |
| Appreciation |                            598 |                       3546 |                     3107 | 87.6%                    | 78.5%                  | 87.5%                    | 28.3%                | 0.0%                   | 71.4%                 | 100.0%                | 100.0%                 |

Note: these are posterior-decoded manager-level transition distributions,
not MCMC posterior draws.
