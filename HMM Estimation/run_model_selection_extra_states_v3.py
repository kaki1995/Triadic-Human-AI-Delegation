from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


NOTEBOOK_NAME = "model selection & parameter estimation.ipynb"
BOOTSTRAP_CELLS = [2, 4, 6, 10, 12]
MODEL_SELECTION_CELL = 14


def display_plain(obj: Any = None, **_: Any) -> None:
    if obj is None:
        print()
        return
    if hasattr(obj, "to_string"):
        print(obj.to_string())
        return
    print(obj)


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "Datasets").exists() and (candidate / "HMM Estimation").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def register_pickle_classes(namespace: dict[str, Any], main_module: ModuleType) -> None:
    params_class = namespace.get("Params")
    if params_class is not None:
        setattr(main_module, "Params", params_class)


def patch_bootstrap_source(source: str, data_path: Path) -> str:
    old = "MANUAL_XLSX_PATH = None"
    new = f"MANUAL_XLSX_PATH = Path({str(data_path)!r})"
    if old not in source:
        raise ValueError("Could not patch MANUAL_XLSX_PATH in notebook bootstrap cell.")
    return source.replace(old, new)


def patch_model_selection_source(source: str, states: list[int]) -> str:
    patched = source
    patched = patched.replace("J_CANDIDATES = [2, 3]", f"J_CANDIDATES = {states!r}")
    patched = patched.replace(
        "MAX_CONTINUATION_ROUNDS_BY_J = {2: 4, 3: 5, 4: 6}",
        "\n".join(
            [
                "MAX_CONTINUATION_ROUNDS_BY_J = {2: 4, 3: 5, 4: 6}",
                "SCREEN_CONFIG_BY_J.update({",
                "    1: dict(maxiter=70, maxfun=60_000, diag_bias=2.2, subset_size=96, n_starts=3),",
                "    5: dict(maxiter=180, maxfun=220_000, diag_bias=3.0, subset_size=96, n_starts=3),",
                "})",
                "FULL_CONFIG_BY_J.update({",
                "    1: dict(maxiter=300, maxfun=250_000, diag_bias=2.2),",
                "    5: dict(maxiter=900, maxfun=850_000, diag_bias=3.0),",
                "})",
                "MAX_CONTINUATION_ROUNDS_BY_J.update({1: 3, 5: 6})",
            ]
        ),
    )
    patched = patched.replace(
        '_checkpoint_root = _artifact_dir / "Convergence_Checkpoints"',
        '_checkpoint_root = _artifact_dir / "Convergence_Checkpoints_extra_states_v3"',
    )
    patched = patched.replace(
        '_progress_path = _artifact_dir / "model_selection_v3_all_converged_progress.csv"',
        '_progress_path = _artifact_dir / "model_selection_v3_extra_states_progress.csv"',
    )
    patched = patched.replace(
        '_final_comparison_path = _artifact_dir / "model_selection_v3_single_vs_3period.csv"',
        '_final_comparison_path = _artifact_dir / "model_selection_v3_extra_states.csv"',
    )
    return patched


def combine_results(repo_root: Path) -> None:
    import pandas as pd

    artifact_dir = repo_root / "HMM Estimation" / "Artefacts"
    base_path = artifact_dir / "model_selection_v3_single_vs_3period.csv"
    extra_path = artifact_dir / "model_selection_v3_extra_states.csv"
    combined_path = artifact_dir / "model_selection_v3_single_vs_3period_J1_J5.csv"

    if not base_path.exists() or not extra_path.exists():
        return

    base = pd.read_csv(base_path)
    extra = pd.read_csv(extra_path)
    combined = (
        pd.concat([base, extra], ignore_index=True)
        .drop_duplicates(subset=["spec", "J"], keep="last")
        .sort_values(["spec", "J"])
        .reset_index(drop=True)
    )
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined model comparison to {combined_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3 model-selection candidates for additional latent-state counts.")
    parser.add_argument("--states", nargs="+", type=int, default=[1, 4, 5])
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Datasets") / "Triadic_Delegation_Analysis_Dataset_v3.xlsx",
        help="Dataset path to use for comparability with existing V3 model-selection CSV.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    notebook_path = repo_root / "HMM Estimation" / NOTEBOOK_NAME
    data_path = args.data
    if not data_path.is_absolute():
        data_path = repo_root / data_path
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "display": display_plain,
    }

    for cell_idx in BOOTSTRAP_CELLS:
        print(f"\n--- executing notebook cell {cell_idx} ---", flush=True)
        source = "".join(notebook["cells"][cell_idx].get("source", ""))
        if cell_idx == 2:
            source = patch_bootstrap_source(source, data_path)
        exec(compile(source, f"notebook_cell_{cell_idx}", "exec"), namespace)
        register_pickle_classes(namespace, sys.modules["__main__"])

    print(f"\n--- executing extra-state model selection for J={args.states} ---", flush=True)
    source = "".join(notebook["cells"][MODEL_SELECTION_CELL].get("source", ""))
    source = patch_model_selection_source(source, args.states)
    exec(compile(source, "extra_state_model_selection", "exec"), namespace)

    combine_results(repo_root)


if __name__ == "__main__":
    main()
