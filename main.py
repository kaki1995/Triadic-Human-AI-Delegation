# triadic_sim/main.py
from __future__ import annotations

from .config import SimConfig
from .schema import SheetMap
from .simulator import simulate
from .io_excel import write_to_schema_workbook


def run() -> None:
    cfg = SimConfig()
    dfs = simulate(cfg)

    sheet_map = {
        "manager_master": SheetMap().manager_master,
        "employee_team_master": SheetMap().employee_team_master,
        "ai_system_master": SheetMap().ai_system_master,
        "panel_manager_period": SheetMap().panel_manager_period,
        "decision_episode": SheetMap().decision_episode,
        "execution_episode": SheetMap().execution_episode,
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
