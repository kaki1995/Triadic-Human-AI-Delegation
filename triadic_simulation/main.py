# triadic_sim/main.py
from __future__ import annotations

from .config import SimConfig
from .schema import SheetMap
from .simulator import simulate
from .io_excel import write_to_schema_workbook


def run() -> None:
    cfg = SimConfig()
    dfs = simulate(cfg)

    sm = SheetMap()
    sheet_map = {
        # L3 master
        "manager_master": sm.manager_master,
        "employee_master": sm.employee_master,          # NEW
        "ai_system_master": sm.ai_system_master,
        "site_master": sm.site_master,                  # NEW

        # L1 panels
        "panel_manager_period": sm.panel_manager_period,
        "panel_employee_period": sm.panel_employee_period,  # NEW

        # L2 episodes
        "decision_episode": sm.decision_episode,
        "execution_episode": sm.execution_episode,
    }

    write_to_schema_workbook(
        input_schema_xlsx=cfg.input_schema_xlsx,
        output_xlsx=cfg.output_xlsx,
        dfs=dfs,
        sheet_map=sheet_map,
    )

    print(f"✅ Wrote: {cfg.output_xlsx}")
    for k, df in dfs.items():
        print(f"{k}: {df.shape}")


if __name__ == "__main__":
    run()
