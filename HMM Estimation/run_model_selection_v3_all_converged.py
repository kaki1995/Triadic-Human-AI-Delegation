from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from types import ModuleType
from typing import Any


NOTEBOOK_NAME = "model selection & parameter estimation.ipynb"
NOTEBOOK_CELLS = [2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21]


def display_plain(obj: Any = None, **_: Any) -> None:
    """Small replacement for Jupyter display() when running notebook cells."""
    if obj is None:
        print()
        return

    if hasattr(obj, "to_string"):
        print(obj.to_string())
        return

    print(obj)


def find_repo_root(start: Path) -> Path:
    """Find the project root without depending on this script's folder depth."""
    for candidate in (start, *start.parents):
        if (candidate / "Datasets").exists() and (candidate / "HMM Estimation").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from runner path.")


def register_pickle_classes(namespace: dict[str, Any], main_module: ModuleType) -> None:
    """Expose notebook dataclasses on __main__ so saved pickle artifacts load."""
    params_class = namespace.get("Params")
    if params_class is not None:
        setattr(main_module, "Params", params_class)


def run_notebook_cells(notebook_path: Path, cell_indices: list[int]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "display": display_plain,
    }

    for cell_idx in cell_indices:
        print(f"\n--- executing notebook cell {cell_idx} ---", flush=True)
        source = "".join(cells[cell_idx].get("source", ""))
        exec(compile(source, f"notebook_cell_{cell_idx}", "exec"), namespace)
        register_pickle_classes(namespace, sys.modules["__main__"])


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve())
    os.chdir(repo_root)

    notebook_path = repo_root / "HMM Estimation" / NOTEBOOK_NAME
    run_notebook_cells(notebook_path, NOTEBOOK_CELLS)


if __name__ == "__main__":
    main()
