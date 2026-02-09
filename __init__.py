"""
Triadic Human–AI Delegation Simulation Package

This package provides tools to simulate dynamic delegation
between managers, AI systems, and human employees in
triadic organizational settings.

Core entry points:
- simulate(): generate synthetic delegation datasets
- SimConfig: configure simulation scenarios
"""

from .config import SimConfig
from .simulator import simulate

__all__ = [
    "SimConfig",
    "simulate",
]
