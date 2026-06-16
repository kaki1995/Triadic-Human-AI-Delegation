# Age and Locality Context Analysis

## Link to high-order HMM framing

The analysis treats each manager's decoded HMM path as a dynamic decision sequence. In the spirit of high-order HMM decision analysis, it reports both first-order transitions and second-order transition windows, where the next state is conditioned on the two-state history rather than only the current state.

## Group construction

- Age grouping: seniority_years proxy because no manager age column exists in the workbook. Rule: 0-3.99, 4-7.99, and 8+ seniority years.
- Locality grouping: region proxy because no manager local/non-local column exists. Rule: Local = region starts with DE-; Non-local = all other regions.

Important limitation: the active workbook does not contain true manager age. The age-group output is therefore a career-stage proxy, not demographic age.

## Descriptive highlights

- Highest locality-group Appreciation share: Local (17.25%).
- Highest age/career group Aversion share: Mid-career proxy (34.66%).
- Highest age/career group AI authority share: Early-career proxy (34.62%).

## Statistical tests

No Holm-adjusted subgroup differences reached p < .05 across the tested manager-level metrics.

## Outputs

- `age_locality_manager_mapping_v3.csv`: manager-level age/locality mapping and source metadata.
- `age_locality_manager_level_summary_v3.csv`: manager-level HMM path and outcome summaries.
- `age_locality_group_summary_v3.csv`: summaries by age group, locality, and age x locality.
- `age_locality_welch_tests_v3.csv`: Welch tests on manager-level metrics.
- `age_locality_state_composition_v3.csv`: decoded state composition by subgroup.
- `age_locality_first_order_transition_rates_v3.csv`: state-to-state transition rates by subgroup.
- `age_locality_second_order_transition_rates_v3.csv`: high-order two-state-history transition rates by subgroup.
