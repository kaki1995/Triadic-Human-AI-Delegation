# Reproduction Notes

These notes preserve the current runnable layout. No active script paths were
changed during the archive cleanup.

## Environment

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Generation

The generator reads the schema workbook from `Datasets/` and writes the current
synthetic and analysis workbooks back to `Datasets/`.

```powershell
python "Data Generation Engine/main.py"
```

Expected current outputs:

```text
Datasets/Triadic_Delegation_Dataset_SYNTH.xlsx
Datasets/Triadic_Delegation_Analysis_Dataset_v3.xlsx
```

## HMM Model Selection

The active model-selection runner executes selected cells from
`HMM Estimation/model selection & parameter estimation.ipynb`.

```powershell
python "HMM Estimation/run_model_selection_v3_all_converged.py"
```

Additional runners are kept in `HMM Estimation/` for continuation searches,
extra-state checks, reduced-control checks, and regional robustness tests.

## Analysis Exports

Analysis CSVs, tables, and figures are currently written to:

```text
HMM Estimation/Artefacts/Analysis_v3/
```

Main figure/table regeneration scripts:

```powershell
python "HMM Estimation/create_kpi_difference_by_state_table_v3.py"
python "HMM Estimation/create_intrinsic_transition_probability_table_v3.py"
python "HMM Estimation/create_triadic_authority_evolution_v3.py"
python "HMM Estimation/create_triadic_relationship_figures_v3.py"
python "HMM Estimation/create_triadic_task_flow_by_state_v3.py"
python "HMM Estimation/create_triadic_wordcloud.py"
```

## Archive Policy

Historical code, notebooks, datasets, and model artefacts were moved to
`Archived/`. They are intentionally preserved for auditability and for possible
reuse when rerunning older model versions.
