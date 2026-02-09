# Triadic-Human-AI-Delegation

This repository provides the simulation framework and synthetic data generator supporting the Master’s thesis titled "**_Revisiting Delegation Theory in the Age of AI: Dynamic Algorithm Appreciation and Aversion in Triadic Organizational Relationships._**"

The project investigates how managers dynamically delegate task execution and decision authority to AI systems within triadic organizational settings involving managers, AI systems, and human employees. It focuses on how algorithm appreciation and aversion evolve as managers receive performance feedback under uncertainty and performance pressure.

## Research Motivation

Modern organizations increasingly rely on AI-based decision support systems while retaining human accountability. Delegation is no longer purely dyadic (manager–employee), but triadic, involving:

- Managers – retain ultimate authority and accountability

- AI systems – generate recommendations and may execute tasks

- Human employees – execute, coordinate, or override decisions

This project simulates such settings to study:

- Dynamic willingness to delegate to AI

- Algorithm appreciation vs. aversion over time

- Effects of transparency, performance feedback, and pressure

- Delegation authority vs. execution responsibility

The simulation is designed to support Hidden Markov Model (HMM) and panel-based analyses.

## Repository Structure

```text
triadic_sim/
  config.py        # Global simulation configuration
  simulator.py     # Core simulation logic
  schema.py        # Dataset & sheet mappings
  utils.py         # Helper functions
  io_excel.py      # Excel read/write utilities
  main.py          # Entry point

notebooks/
  01_generate_dataset.ipynb
  02_validate_dataset.ipynb
  03_analysis_figures.ipynb
```

## Dataset Overview

The simulation generates a **longitudinal triadic delegation dataset** with the following core tables:

| Table | Description |
|------|------------|
| `manager_master` | Manager characteristics and governance orientation |
| `employee_team_master` | Human execution teams |
| `ai_system_master` | AI system properties (autonomy, calibration) |
| `panel_manager_period` | Period-level delegation, KPIs, and latent states |

The primary analysis table is panel_manager_period, structured at the manager–period level.

## Simulation Logic (High-Level)

- Managers have latent willingness-to-delegate states (low / medium / high).

- AI generates recommendations with varying confidence.

- Managers accept, modify, or reject AI recommendations.

- Execution is delegated to AI, humans, or jointly.

- Performance feedback and transparency drive state transitions over time.

- Latent states evolve via an HMM-like transition process.

All parameters are explicitly configurable via config.py.

## How to Run

1. Install dependencies

```text
pip install numpy pandas openpyxl
```

3. Configure simulation

Edit parameters in:

```text
triadic_sim/config.py
```

3. Generate dataset

```text
python -m triadic_sim.main
```

This will produce:

```text
Triadic_Delegation_Dataset_SYNTH.xlsx
```

aligned to the provided schema workbook.

## Reproducibility

- All randomness is controlled via a global seed.

- Configuration is centralized and immutable.

- Simulation logic is fully deterministic given the same configuration.

- Designed for replication, robustness checks, and scenario comparison.

## Intended Use

This repository is intended for:

- Academic research and thesis work

- Methodological illustration of human–AI delegation dynamics

- Synthetic data generation for model development (HMM, panel models)

It is not intended as a production decision-support system.
