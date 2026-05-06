# Triadic-Human-AI-Delegation

This repository provides the simulation framework and synthetic data generator
supporting the Master's thesis titled:

**Revisiting Delegation Theory in the Age of AI: Dynamic Algorithm Appreciation
and Aversion in Triadic Organizational Relationships.**

The project investigates how managers dynamically delegate task execution and
decision authority to AI systems within triadic organizational settings
involving managers, AI systems, and human employees. It focuses on how algorithm
appreciation and aversion evolve as managers receive performance feedback under
uncertainty and performance pressure.

## Research Motivation

Modern organizations increasingly rely on AI-based decision support systems
while retaining human accountability. Delegation is no longer purely dyadic
(manager-employee), but triadic, involving:

- Managers retain ultimate authority and accountability.
- AI systems generate recommendations and may execute tasks.
- Human employees execute, coordinate, or override decisions.

This project simulates such settings to study:

- Dynamic willingness to delegate to AI.
- Algorithm appreciation vs. aversion over time.
- Effects of performance feedback and pressure.
- Delegation authority vs. execution responsibility.

The simulation is designed to support Hidden Markov Model (HMM) and panel-based
analyses.

## Repository Structure

```text
triadic_simulation/
  config.py        # Global simulation configuration
  simulator.py     # Core simulation logic
  schema.py        # Dataset and sheet mappings
  utils.py         # Helper functions
  io_excel.py      # Excel read/write utilities
  main.py          # Entry point

notebooks/
  01_generate_dataset.ipynb
  02_validate_dataset.ipynb
  03_analysis_figures.ipynb
```

## Dataset Overview

The simulation generates a longitudinal triadic delegation dataset with the
following default scale:

- 875 managers.
- 17,680 employees.
- 26 planning cycles.

The generated workbook contains the following core tables:

| Table | Description |
| --- | --- |
| `manager_master` | Manager characteristics and governance orientation |
| `employee_master` | Human employee characteristics |
| `ai_system_master` | AI system properties, autonomy, and calibration |
| `site_master` | Site-level operational context |
| `panel_manager_period` | Period-level delegation, controls, transitions, and latent states |
| `panel_manager_period_outcomes` | Derived manager-period outcome measures |
| `panel_employee_period` | Period-level employee execution outcomes |
| `decision_episode` | Episode-level AI recommendation and manager action data |
| `execution_episode` | Episode-level execution mode and error data |

The primary analysis table is `panel_manager_period`, structured at the
manager-period level.

The manager-period table includes these HMM emission variables:

- `ai_authority_share`
- `escalation_share`
- `decision_latency`
- `demand_volatility`
- `forecast_accuracy`
- `performance_pressure`
- `recent_negative_shock`
- `supply_disruptions`
- `target_difficulty`
- `task_complexity`

It also includes these transition variables:

- `team_t_minus_1_vs_team_t`
- `team_vs_peer_average`
- `target_attainment`

## Simulation Logic

- Managers have latent willingness-to-delegate states: low, medium, or high.
- AI generates recommendations with varying confidence.
- Managers accept, modify, or reject AI recommendations.
- Execution is delegated to AI, humans, or jointly.
- Performance feedback and operating conditions drive state transitions over time.
- Latent states evolve via an HMM-like transition process.

All parameters are explicitly configurable via `triadic_simulation/config.py`.

## How to Run

1. Install dependencies.

```text
pip install numpy pandas openpyxl
```

2. Configure simulation parameters in `triadic_simulation/config.py`.

3. Generate the dataset.

```text
python -m triadic_simulation.main
```

This produces:

```text
triadic_simulation/data/Triadic_Delegation_Dataset_SYNTH.xlsx
triadic_simulation/data/Triadic_Delegation_Dataset_SYNTH_ANALYSIS.xlsx
```

Both outputs are aligned to the provided schema workbook.

## Reproducibility

- All randomness is controlled via a global seed.
- Configuration is centralized and immutable.
- Simulation logic is fully deterministic given the same configuration.
- The project is designed for replication, robustness checks, and scenario
  comparison.

## Intended Use

This repository is intended for:

- Academic research and thesis work.
- Methodological illustration of human-AI delegation dynamics.
- Synthetic data generation for model development, including HMM and panel
  models.

It is not intended as a production decision-support system.
