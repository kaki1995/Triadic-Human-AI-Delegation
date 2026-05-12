from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


NOTEBOOK_NAME = "model selection & parameter estimation.ipynb"
BOOTSTRAP_CELLS = [2, 4, 6, 10, 12]


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


def occupancy_metrics(p_hat: Any, data_m: Any, forward_backward: Any) -> tuple[float, float]:
    gammas = []
    for y_i, x_i, z_i in zip(data_m.Y, data_m.X, data_m.Z):
        _, log_g = forward_backward(p_hat, y_i, x_i, z_i)
        gammas.append(np.exp(log_g))
    gamma_all = np.concatenate(gammas, axis=0)
    return float(gamma_all.mean(axis=0).min()), float(gamma_all.max(axis=1).mean())


def score_row(
    spec_name: str,
    transition_cols: list[str],
    j_states: int,
    p_hat: Any,
    res: Any,
    is_conv: bool,
    data_m: Any,
    forward_backward: Any,
    runtime_min: float,
    round_idx: int,
    source: str,
    screen_converged: bool,
) -> dict[str, Any]:
    y_stack = np.stack(data_m.Y)
    n_obs = int(y_stack.shape[0] * y_stack.shape[1])
    ll_total = float(getattr(res, "true_ll", np.nan))
    k_params = int(len(getattr(res, "x", [])))
    aic = 2.0 * k_params - 2.0 * ll_total
    bic = np.log(n_obs) * k_params - 2.0 * ll_total
    occupancy_min, certainty_mean = occupancy_metrics(p_hat, data_m, forward_backward)
    return {
        "spec": spec_name,
        "J": int(j_states),
        "P": len(transition_cols),
        "transition_cols": ", ".join(transition_cols),
        "LL": ll_total,
        "AIC": aic,
        "BIC": bic,
        "k_params": k_params,
        "n_obs": n_obs,
        "soft_converged": bool(is_conv),
        "scipy_success": bool(getattr(res, "success", False)),
        "strict_converged": bool(is_conv and getattr(res, "success", False)),
        "screen_converged": bool(screen_converged),
        "continuation_round": int(round_idx),
        "iterations": int(getattr(res, "nit", -1)) if getattr(res, "nit", None) is not None else np.nan,
        "occupancy_min": occupancy_min,
        "certainty_mean": certainty_mean,
        "runtime_min": runtime_min,
        "source": source,
        "message": str(getattr(res, "message", "")),
    }


def write_updated_rows(path: Path, row: dict[str, Any]) -> None:
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df = (
        new_df.drop_duplicates(subset=["spec", "J"], keep="last")
        .sort_values(["spec", "J"])
        .reset_index(drop=True)
    )
    new_df.to_csv(path, index=False)


def update_combined_csv(artifact_dir: Path, row: dict[str, Any]) -> Path:
    combined_path = artifact_dir / "model_selection_v3_single_vs_3period_J1_J5.csv"
    base_path = artifact_dir / "model_selection_v3_single_vs_3period.csv"
    frames = []
    if base_path.exists():
        frames.append(pd.read_csv(base_path))
    if combined_path.exists():
        frames.append(pd.read_csv(combined_path))
    frames.append(pd.DataFrame([row]))
    combined = pd.concat(frames, ignore_index=True)
    combined["strict_converged"] = combined["strict_converged"].astype(str).str.lower().isin(["true", "1", "yes"])
    combined["LL"] = pd.to_numeric(combined["LL"], errors="coerce")
    combined["_strict_rank"] = combined["strict_converged"].astype(int)
    combined = (
        combined.sort_values(["spec", "J", "LL", "_strict_rank"])
        .drop_duplicates(subset=["spec", "J"], keep="last")
        .drop(columns=["_strict_rank"])
        .sort_values(["spec", "J"])
        .reset_index(drop=True)
    )
    combined.to_csv(combined_path, index=False)
    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue one V3 HMM model-selection candidate.")
    parser.add_argument("--spec", choices=["single_period", "three_period"], required=True)
    parser.add_argument("--states", type=int, required=True)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--maxfun", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-cap-min", type=int, default=360)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Datasets") / "Triadic_Delegation_Analysis_Dataset_v3.xlsx",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    notebook_path = repo_root / "HMM Estimation" / NOTEBOOK_NAME
    data_path = args.data if args.data.is_absolute() else repo_root / args.data
    artifact_dir = repo_root / "HMM Estimation" / "Artefacts"
    output_path = artifact_dir / "model_selection_v3_candidate_continuations.csv"

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

    transition_specs = namespace["TRANSITION_SPECS"]
    transition_cols = list(transition_specs[args.spec])
    load_sequences = namespace["load_sequences"]
    fit_model_batched = namespace["fit_model_batched"]
    forward_backward = namespace["forward_backward"]

    data_m = load_sequences(data_path, transition_cols_override=transition_cols)
    y_stack = np.stack(data_m.Y)
    x_stack = np.stack(data_m.X)
    z_stack = np.stack(data_m.Z)

    data_signature = f"{data_path.resolve()}::{data_path.stat().st_size}::{data_path.stat().st_mtime_ns}"
    checkpoint_dir = (
        artifact_dir
        / "Convergence_Checkpoints_candidate_continuations_v3"
        / hashlib.sha1(data_signature.encode("utf-8")).hexdigest()[:16]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"candidate_v3_{args.spec}_J{args.states}.pkl"

    ckpt = None
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as f:
            ckpt = pickle.load(f)
        if ckpt.get("spec") != args.spec or int(ckpt.get("J", -1)) != args.states:
            ckpt = None
        elif list(ckpt.get("transition_cols", [])) != transition_cols:
            ckpt = None

    seed = args.seed
    if seed is None:
        spec_idx = list(transition_specs).index(args.spec)
        seed = 2026 + 100 * spec_idx + args.states

    if ckpt is not None:
        p_current = ckpt["params"]
        round_history = list(ckpt.get("round_history", []))
        start_round = int(ckpt.get("row", {}).get("continuation_round", len(round_history))) + 1
        screen_converged = bool(ckpt.get("row", {}).get("screen_converged", False))
        print(f"Resuming {checkpoint_path} at continuation round {start_round}.", flush=True)
        if bool(ckpt.get("is_conv")) and bool(getattr(ckpt.get("res"), "success", False)):
            row = ckpt["row"]
            write_updated_rows(output_path, row)
            combined_path = update_combined_csv(artifact_dir, row)
            print("Candidate already converged.", flush=True)
            print(pd.DataFrame([row]).to_string(index=False), flush=True)
            print(f"Updated {combined_path.resolve()}", flush=True)
            return
    else:
        print("Screening warm starts on a manager subset.", flush=True)
        screen_cfg = {
            "maxiter": 120,
            "maxfun": 120_000,
            "diag_bias": 2.6,
            "subset_size": 96,
            "sigma_min": 0.1,
            "sigma_max": 3.5,
            "print_every": 75,
            "time_cap_min": args.time_cap_min,
            "l2": 0.01,
            "ftol": 1e-8,
            "gtol": 2e-6,
        }
        p_current, _res_screen, screen_converged = fit_model_batched(
            J=args.states,
            Y_stack=y_stack,
            X_stack=x_stack,
            Z_stack=z_stack,
            seed=seed,
            n_starts=3,
            use_subset=True,
            do_emission_only_warmstart=True,
            emission_only_maxiter=60,
            emission_only_maxfun=40_000,
            jitter_warm_starts=True,
            **screen_cfg,
        )
        round_history = []
        start_round = 1

    final_row = None
    for round_idx in range(start_round, args.max_rounds + 1):
        print(
            f"\nFull-data continuation round {round_idx}/{args.max_rounds} "
            f"(maxiter={args.maxiter}, maxfun={args.maxfun})",
            flush=True,
        )
        t0 = time.time()
        p_hat, res, is_conv = fit_model_batched(
            J=args.states,
            Y_stack=y_stack,
            X_stack=x_stack,
            Z_stack=z_stack,
            seed=seed + 10_000 + round_idx,
            n_starts=1,
            warm_starts=[p_current],
            use_subset=False,
            do_emission_only_warmstart=False,
            emission_only_maxiter=0,
            jitter_warm_starts=False,
            maxiter=args.maxiter,
            maxfun=args.maxfun,
            diag_bias=2.6,
            sigma_min=0.1,
            sigma_max=3.5,
            print_every=75,
            time_cap_min=args.time_cap_min,
            l2=0.01,
            ftol=1e-8,
            gtol=2e-6,
        )
        row = score_row(
            args.spec,
            transition_cols,
            args.states,
            p_hat,
            res,
            is_conv,
            data_m,
            forward_backward,
            runtime_min=(time.time() - t0) / 60.0,
            round_idx=round_idx,
            source="candidate_continuation",
            screen_converged=screen_converged,
        )
        round_history.append(row)
        with checkpoint_path.open("wb") as f:
            pickle.dump(
                {
                    "spec": args.spec,
                    "J": int(args.states),
                    "transition_cols": transition_cols,
                    "params": p_hat,
                    "res": res,
                    "is_conv": bool(is_conv),
                    "row": row,
                    "round_history": round_history,
                },
                f,
            )
        write_updated_rows(output_path, row)
        combined_path = update_combined_csv(artifact_dir, row)
        print(
            f"Round {round_idx}: LL={row['LL']:.2f}, AIC={row['AIC']:.2f}, "
            f"BIC={row['BIC']:.2f}, converged={row['strict_converged']}, "
            f"iterations={row['iterations']}, runtime={row['runtime_min']:.1f} min",
            flush=True,
        )
        print(f"Saved checkpoint to {checkpoint_path.resolve()}", flush=True)
        print(f"Updated {combined_path.resolve()}", flush=True)
        final_row = row
        p_current = p_hat
        if row["strict_converged"]:
            break

    if final_row is not None:
        print()
        print(pd.DataFrame([final_row]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
