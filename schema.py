# triadic_sim/schema.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SheetMap:
    """
    Maps internal DataFrame keys -> Excel sheet names.
    Update here if your workbook uses different sheet names.
    """
    manager_master: str = "manager_master"
    employee_team_master: str = "employee_team_master"
    ai_system_master: str = "ai_system_master"
    panel_manager_period: str = "panel_manager_period"
    decision_episode: str = "decision_episode"
    execution_episode: str = "execution_episode"
