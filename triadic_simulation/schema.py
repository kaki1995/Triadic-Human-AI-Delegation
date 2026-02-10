# triadic_sim/schema.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SheetMap:
    """
    Maps internal DataFrame keys -> Excel sheet names.

    Conceptual role:
    - Centralizes the mapping between simulator outputs and Excel schema.
    - Ensures reproducibility and prevents silent schema drift.
    - Acts as the authoritative contract between code and data documentation.
    """

    # ------------------------------------------------------------------
    # LEVEL 3 - Master data (stable attributes)
    # ------------------------------------------------------------------
    manager_master: str = "manager_master"
    employee_master: str = "employee_master"
    ai_system_master: str = "ai_system_master"
    site_master: str = "site_master"

    # ------------------------------------------------------------------
    # LEVEL 1 - Dynamic panels
    # ------------------------------------------------------------------
    panel_manager_period: str = "panel_manager_period"
    panel_employee_period: str = "panel_employee_period"

    # ------------------------------------------------------------------
    # LEVEL 2 - Episode-level data
    # ------------------------------------------------------------------
    decision_episode: str = "decision_episode"
    execution_episode: str = "execution_episode"
