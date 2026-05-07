from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat


def _display(obj=None, **_kwargs):
    if obj is None:
        print()
        return
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def main() -> None:
    runner_path = Path(__file__).resolve()
    repo_root = runner_path.parents[2]
    os.chdir(repo_root)

    notebook_path = repo_root / "HMM Estimation" / "model selection & parameter estimation.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)

    namespace = {
        "__name__": "__main__",
        "display": _display,
    }

    # Imports/helpers, benchmark construction, loading, stacks, HMM core,
    # estimator, converged model selection, diagnostics, and final artifact save.
    for cell_idx in [2, 4, 6, 8, 10, 12, 14, 15, 16, 18, 19]:
        print(f"\n--- executing notebook cell {cell_idx} ---", flush=True)
        source = nb.cells[cell_idx].get("source", "")
        exec(compile(source, f"notebook_cell_{cell_idx}", "exec"), namespace)
        if "Params" in namespace:
            setattr(sys.modules["__main__"], "Params", namespace["Params"])


if __name__ == "__main__":
    main()
