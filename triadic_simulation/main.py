# triadic_sim/main.py
from __future__ import annotations

from typing import Dict
import pandas as pd

from .config import SimConfig
from .schema import SheetMap
from .simulator import simulate
from .io_excel import write_to_schema_workbook


def _drop_validation_columns(
    dfs: Dict[str, pd.DataFrame],
    drop_cols: list[str],
) -> Dict[str, pd.DataFrame]:
    """
    Remove validation-only columns (e.g., true latent states) from all tables.
    This produces an "analysis" dataset that mimics real-world observables.
    """
    cleaned: Dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        if df is None:
            cleaned[name] = df
            continue
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        cleaned[name] = df.drop(columns=cols_to_drop, errors="ignore")
    return cleaned


def run() -> None:
    cfg = SimConfig()
    dfs = simulate(cfg)

    sm = SheetMap()
    sheet_map = {
        # L3 master
        "manager_master": sm.manager_master,
        "employee_master": sm.employee_master,
        "ai_system_master": sm.ai_system_master,
        "site_master": sm.site_master,

        # L1 panels
        "panel_manager_period": sm.panel_manager_period,
        "panel_employee_period": sm.panel_employee_period,

        # L2 episodes
        "decision_episode": sm.decision_episode,
        "execution_episode": sm.execution_episode,
    }

    # ------------------------------------------------------------
    # 1) FULL synthetic dataset (includes latent_state_true*)
    # ------------------------------------------------------------
    write_to_schema_workbook(
        input_schema_xlsx=cfg.input_schema_xlsx,
        output_xlsx=cfg.output_xlsx,
        dfs=dfs,
        sheet_map=sheet_map,
    )
    print(f"✅ Wrote FULL synthetic dataset: {cfg.output_xlsx}")

    # ------------------------------------------------------------
    # 2) ANALYSIS dataset (drops validation-only columns)
    # ------------------------------------------------------------
    dfs_analysis = _drop_validation_columns(dfs, cfg.analysis_drop_cols)

    write_to_schema_workbook(
        input_schema_xlsx=cfg.input_schema_xlsx,
        output_xlsx=cfg.output_analysis_xlsx,
        dfs=dfs_analysis,
        sheet_map=sheet_map,
    )
    print(f"✅ Wrote ANALYSIS dataset (no true latent states): {cfg.output_analysis_xlsx}")

    # Diagnostics
    for k, df in dfs.items():
        print(f"{k}: {df.shape}")


if __name__ == "__main__":
    run()
