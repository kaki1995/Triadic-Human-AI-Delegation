from __future__ import annotations

"""
Updated HMM loader/specification for the v2 workbook.
This keeps transparency as a main effect only and uses BIC over J=2..4.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "Triadic_Delegation_Dataset_SYNTH_ANALYSIS_v2.xlsx"


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(z, axis=-1):
    return z - logsumexp(z, axis=axis, keepdims=True)


def build_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["manager_id", "period_id"]).copy()
    df["kpi_operational_gap_index"] = (
        (-1.0) * df["service_level_delta"]
        + (-0.6) * df["inventory_cost_delta"]
        + (-0.4) * df["expedite_cost_delta"]
        + (-1.2) * df["error_incident_count"]
    )
    df["within_unit_temporal_benchmark"] = df["team_vs_tminus1"].astype(float)
    df["horizontal_peer_benchmark"] = df["team_vs_peer"].astype(float)
    df["ai_team_benchmark"] = df["team_ai_vs_without_ai"].astype(float)
    df["threshold_benchmark"] = df["threshold_trigger"].astype(float)
    df["transparency_level_norm"] = df["transparency_level"].astype(float) / 3.0
    return df


@dataclass
class HMMData:
    Y: list
    X: list
    Z: list
    ids: list
    periods: list
    y_scaler: StandardScaler
    x_scaler: StandardScaler
    z_scaler: StandardScaler


def load_sequences(xlsx_path: Path) -> HMMData:
    df = pd.read_excel(xlsx_path, sheet_name="panel_manager_period")
    df = build_benchmarks(df)

    dec_ep = pd.read_excel(xlsx_path, sheet_name="decision_episode")
    esc_agg = (
        dec_ep.groupby(["manager_id", "period_id"])["escalation_flag"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "n_tasks", "sum": "n_escalated"})
    )
    esc_agg["share_authority_esc"] = esc_agg["n_escalated"] / esc_agg["n_tasks"]
    df = df.merge(
        esc_agg[["manager_id", "period_id", "share_authority_esc"]],
        on=["manager_id", "period_id"],
        how="left",
    )
    df["share_authority_esc"] = df["share_authority_esc"].fillna(0.0)

    emission_cols = ["ai_decision_authority_share", "share_authority_esc"]
    transition_cols = [
        "within_unit_temporal_benchmark",
        "horizontal_peer_benchmark",
        "ai_team_benchmark",
        "threshold_benchmark",
        "transparency_level_norm",
    ]
    control_cols = [
        "ai_implementation_age",
        "task_complexity_index",
        "task_stakes",
        "ai_accuracy",
    ]

    ids = sorted(df["manager_id"].unique())
    y_scaler, x_scaler, z_scaler = StandardScaler(), StandardScaler(), StandardScaler()
    y_scaler.fit(df[emission_cols].values)
    x_scaler.fit(df[transition_cols].values)
    z_scaler.fit(df[control_cols].values)

    Y, X, Z, periods = [], [], [], []
    for mid in ids:
        g = df[df["manager_id"] == mid].sort_values("period_id")
        Y.append(y_scaler.transform(g[emission_cols].values))
        X.append(x_scaler.transform(g[transition_cols].values))
        Z.append(z_scaler.transform(g[control_cols].values))
        periods.append(g["period_id"].to_numpy())

    return HMMData(Y=Y, X=X, Z=Z, ids=ids, periods=periods, y_scaler=y_scaler, x_scaler=x_scaler, z_scaler=z_scaler)


if __name__ == "__main__":
    data = load_sequences(DATA_PATH)
    print(f"Loaded N={len(data.Y)} managers | T={data.Y[0].shape[0]} | D={data.Y[0].shape[1]}")
    print("Transition covariates: within_unit_temporal_benchmark, horizontal_peer_benchmark, ai_team_benchmark, threshold_benchmark, transparency_level_norm")
    print("Emission controls: ai_implementation_age, task_complexity_index, task_stakes, ai_accuracy")
