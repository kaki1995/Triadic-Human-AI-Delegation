# Algorithm Appreciation and Aversion in Triadic Delegation Settings

This repository contains the synthetic data generator, Hidden Markov Model
(HMM) estimation workflow, and analysis artefacts for the triadic delegation
study.

## Current Layout

```text
Data Generation Engine/     Active synthetic data generation code
Datasets/                   Current schema and analysis workbooks
HMM Estimation/             Active HMM notebooks, runners, figures, and tables
Archived/                   Historical code, notebooks, datasets, and model artefacts
docs/                       Reproduction notes and artefact inventory
```

The active folders are intentionally left in their current locations so existing
notebooks and runner scripts can still be used for model reruns. Historical
material has been consolidated under `Archived/` instead of being deleted.

## Active Workflow

1. Generate or refresh the synthetic dataset:

```powershell
python "Data Generation Engine/main.py"
```

2. Run HMM model selection:

```powershell
python "HMM Estimation/run_model_selection_v3_all_converged.py"
```

3. Run additional model-selection or robustness scripts as needed:

```powershell
python "HMM Estimation/run_model_selection_extra_states_v3.py"
python "HMM Estimation/run_reduced_control_search_v3.py"
python "HMM Estimation/run_transition_threshold_tests_v3.py"
python "HMM Estimation/run_manager_heterogeneity_tests_v3.py"
python "HMM Estimation/run_regional_context_tests_v3.py"
python "HMM Estimation/run_macro_region_context_tests_v3.py"
python "HMM Estimation/run_macro_region_parameter_comparison_v3.py"
```

4. Recreate final tables and figures:

```powershell
python "HMM Estimation/create_kpi_difference_by_state_table_v3.py"
python "HMM Estimation/create_intrinsic_transition_probability_table_v3.py"
python "HMM Estimation/create_triadic_task_flow_by_state_v3.py"
python "HMM Estimation/create_triadic_wordcloud.py"
```

See `docs/reproduction.md` for the fuller run notes.
