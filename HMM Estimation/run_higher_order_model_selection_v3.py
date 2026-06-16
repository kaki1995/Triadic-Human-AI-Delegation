from __future__ import annotations

import argparse
import hashlib
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_DATA_PATHS = [
    PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3_state_trend_calibrated.xlsx",
    PROJECT_ROOT / "Datasets" / "Triadic_Delegation_Analysis_Dataset_v3.xlsx",
]
DATA_PATH = next((path for path in PREFERRED_DATA_PATHS if path.exists()), PREFERRED_DATA_PATHS[-1])
ARTIFACT_DIR = PROJECT_ROOT / "HMM Estimation" / "Artefacts"

HIGHER_ORDER_CSV = ARTIFACT_DIR / "model_selection_v3_higher_order.csv"
COMPARISON_CSV = ARTIFACT_DIR / "model_selection_v3_first_vs_higher_order.csv"
NOTE_PATH = ARTIFACT_DIR / "higher_order_model_comparison_note_v3.md"

FIRST_ORDER_CANDIDATES = [
    ARTIFACT_DIR / "model_selection_v3_single_vs_3period_J1_J5.csv",
    ARTIFACT_DIR / "model_selection_v3_single_vs_3period.csv",
]

EMISSION_COLS = [
    "ai_authority_share",
    "escalation_share",
]

CONTROL_COLS = [
    "decision_latency",
    "demand_volatility",
    "forecast_accuracy",
    "performance_pressure",
    "recent_negative_shock",
    "supply_disruptions",
    "target_difficulty",
    "task_complexity",
]

SINGLE_PERIOD_TRANSITION_COLS = [
    "team_t_minus_1_vs_team_t",
    "team_vs_peer_average",
    "target_attainment",
]

THREE_PERIOD_TRANSITION_COLS = [
    "team_prev3_avg_vs_team_t",
    "team_vs_peer_average_3period",
    "target_attainment_3period",
]

TRANSITION_SPECS = {
    "single_period": list(SINGLE_PERIOD_TRANSITION_COLS),
    "three_period": list(THREE_PERIOD_TRANSITION_COLS),
}


@dataclass
class HMMData:
    Y: list[np.ndarray]
    X: list[np.ndarray]
    Z: list[np.ndarray]
    ids: list[str]
    periods: list[np.ndarray]
    y_scaler: StandardScaler
    x_scaler: StandardScaler
    z_scaler: StandardScaler


@dataclass
class HigherOrderParams:
    logit_pi: np.ndarray
    init_alpha: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    W: np.ndarray
    log_sigma: np.ndarray


@dataclass
class Params:
    logit_pi: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    W: np.ndarray
    log_sigma: np.ndarray


def build_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["manager_id", "period_id"]).copy()
    required_cols = [
        "manager_id",
        "period_id",
        "composite_kpi_score",
        *EMISSION_COLS,
        *CONTROL_COLS,
        *SINGLE_PERIOD_TRANSITION_COLS,
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required V3 columns: {missing}")

    numeric_cols = [
        "period_id",
        "composite_kpi_score",
        *EMISSION_COLS,
        *CONTROL_COLS,
        *SINGLE_PERIOD_TRANSITION_COLS,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("manager_id", group_keys=False)
    prev3_kpi = grouped["composite_kpi_score"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["team_prev3_avg_vs_team_t"] = (df["composite_kpi_score"] - prev3_kpi).fillna(0.0)
    df["team_vs_peer_average_3period"] = grouped["team_vs_peer_average"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["target_attainment_3period"] = grouped["target_attainment"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    return df


def load_sequences(xlsx_path: Path, transition_cols: list[str]) -> HMMData:
    df = pd.read_excel(xlsx_path, sheet_name="panel_manager_period")
    df = build_benchmarks(df)

    missing = [col for col in [*EMISSION_COLS, *transition_cols, *CONTROL_COLS] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required model columns: {missing}")

    df = df.dropna(subset=[*EMISSION_COLS, *transition_cols, *CONTROL_COLS])
    Y_list: list[np.ndarray] = []
    X_list: list[np.ndarray] = []
    Z_list: list[np.ndarray] = []
    ids: list[str] = []
    periods: list[np.ndarray] = []

    for manager_id, group in df.groupby("manager_id"):
        group = group.sort_values("period_id")
        if len(group) < 4:
            continue
        Y_list.append(group[EMISSION_COLS].to_numpy(float))
        X_list.append(group[transition_cols].to_numpy(float))
        Z_list.append(group[CONTROL_COLS].to_numpy(float))
        ids.append(str(manager_id))
        periods.append(group["period_id"].to_numpy())

    if not Y_list:
        raise ValueError("No valid manager sequences after filtering missing values.")

    y_scaler = StandardScaler().fit(np.vstack(Y_list))
    x_scaler = StandardScaler().fit(np.vstack(X_list))
    z_scaler = StandardScaler().fit(np.vstack(Z_list))
    Y_list = [y_scaler.transform(y) for y in Y_list]
    X_list = [x_scaler.transform(x) for x in X_list]
    Z_list = [z_scaler.transform(z) for z in Z_list]
    return HMMData(Y_list, X_list, Z_list, ids, periods, y_scaler, x_scaler, z_scaler)


def pack_params(p: HigherOrderParams) -> np.ndarray:
    return np.concatenate(
        [
            p.logit_pi.ravel(),
            p.init_alpha.ravel(),
            p.alpha.ravel(),
            p.beta.ravel(),
            p.mu.ravel(),
            p.W.ravel(),
            p.log_sigma.ravel(),
        ]
    )


def unpack_params(theta: np.ndarray, J: int, D: int, P: int, K: int) -> HigherOrderParams:
    idx = 0

    def take(n: int) -> np.ndarray:
        nonlocal idx
        value = theta[idx : idx + n]
        idx += n
        return value

    return HigherOrderParams(
        logit_pi=take(J),
        init_alpha=take(J * J).reshape(J, J),
        alpha=take(J * J * J).reshape(J, J, J),
        beta=take(J * J * J * P).reshape(J, J, J, P),
        mu=take(J * D).reshape(J, D),
        W=take(J * D * K).reshape(J, D, K),
        log_sigma=take(J * D).reshape(J, D),
    )


def smart_init_params(
    rng: np.random.Generator,
    J: int,
    D: int,
    P: int,
    K: int,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    log_sigma_low: float,
    log_sigma_high: float,
    diag_bias: float,
) -> HigherOrderParams:
    mu0 = y_mean[None, :] + rng.normal(0.0, 1.0, (J, D)) * y_std[None, :]
    log_sigma0 = np.log(np.clip(y_std, np.exp(log_sigma_low), np.exp(log_sigma_high)))[None, :]
    log_sigma0 = np.repeat(log_sigma0, J, axis=0)
    log_sigma0 = np.clip(log_sigma0 + rng.normal(0.0, 0.12, (J, D)), log_sigma_low, log_sigma_high)

    logit_pi0 = rng.normal(0.0, 0.20, J)
    init_alpha0 = rng.normal(0.0, 0.20, (J, J)) + np.eye(J) * diag_bias
    alpha0 = rng.normal(0.0, 0.18, (J, J, J))
    for previous in range(J):
        for current in range(J):
            alpha0[previous, current, current] += diag_bias
    beta0 = rng.normal(0.0, 0.02, (J, J, J, P))
    W0 = rng.normal(0.0, 0.03, (J, D, K))
    return HigherOrderParams(
        logit_pi=logit_pi0,
        init_alpha=init_alpha0,
        alpha=alpha0,
        beta=beta0,
        mu=mu0,
        W=W0,
        log_sigma=log_sigma0,
    )


def compute_log_emissions(p: HigherOrderParams, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    means = p.mu[None, None, :, :] + np.einsum("jdk,ntk->ntjd", p.W, Z)
    residuals = Y[:, :, None, :] - means
    sigma2 = np.maximum(np.exp(2.0 * p.log_sigma), 1e-6)
    log_norm = np.sum(np.log(2.0 * np.pi * sigma2), axis=1)
    return -0.5 * (
        log_norm[None, None, :] + np.sum(residuals**2 / sigma2[None, None, :, :], axis=3)
    )


def batched_second_order_ll(p: HigherOrderParams, Y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
    _, T, _ = Y.shape
    logB = compute_log_emissions(p, Y, Z)
    log_pi = log_softmax(p.logit_pi, axis=0)
    log_init = log_softmax(p.init_alpha, axis=1)
    logQ = log_softmax(
        p.alpha[None, None, :, :, :] + np.einsum("abcp,ntp->ntabc", p.beta, X),
        axis=4,
    )

    pair_alpha = (
        log_pi[None, :, None]
        + logB[:, 0, :, None]
        + log_init[None, :, :]
        + logB[:, 1, None, :]
    )
    for t in range(2, T):
        pair_alpha = logB[:, t, None, :] + logsumexp(
            pair_alpha[:, :, :, None] + logQ[:, t, :, :, :],
            axis=1,
        )
    ll = np.sum(logsumexp(pair_alpha, axis=(1, 2)))
    return float(ll)


def second_order_forward_backward(
    p: HigherOrderParams,
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
) -> tuple[float, np.ndarray]:
    T, _ = Y.shape
    Y_n = Y[None, :, :]
    X_n = X[None, :, :]
    Z_n = Z[None, :, :]
    logB = compute_log_emissions(p, Y_n, Z_n)[0]
    log_pi = log_softmax(p.logit_pi, axis=0)
    log_init = log_softmax(p.init_alpha, axis=1)
    logQ = log_softmax(
        p.alpha[None, :, :, :] + np.einsum("abcp,tp->tabc", p.beta, X),
        axis=3,
    )

    pair_alpha_by_t: list[np.ndarray] = []
    pair_alpha = log_pi[:, None] + logB[0, :, None] + log_init + logB[1, None, :]
    pair_alpha_by_t.append(pair_alpha)
    for t in range(2, T):
        pair_alpha = logB[t, None, :] + logsumexp(
            pair_alpha[:, :, None] + logQ[t, :, :, :],
            axis=0,
        )
        pair_alpha_by_t.append(pair_alpha)

    ll = float(logsumexp(pair_alpha_by_t[-1]))
    pair_beta_by_t: list[np.ndarray] = [np.zeros_like(pair_alpha_by_t[-1])]
    pair_beta = pair_beta_by_t[0]
    for t in reversed(range(1, T - 1)):
        pair_beta = logsumexp(
            logQ[t + 1, :, :, :] + logB[t + 1, None, None, :] + pair_beta[None, :, :],
            axis=2,
        )
        pair_beta_by_t.insert(0, pair_beta)

    log_gamma = np.empty((T, p.mu.shape[0]))
    first_pair = pair_alpha_by_t[0] + pair_beta_by_t[0]
    log_gamma[0] = logsumexp(first_pair, axis=1) - ll
    for t in range(1, T):
        pair_index = t - 1
        pair_joint = pair_alpha_by_t[pair_index] + pair_beta_by_t[pair_index]
        log_gamma[t] = logsumexp(pair_joint, axis=0) - ll
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    return ll, log_gamma


def second_order_expectations(
    p: HigherOrderParams,
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    T, _ = Y.shape
    logB = compute_log_emissions(p, Y[None, :, :], Z[None, :, :])[0]
    log_pi = log_softmax(p.logit_pi, axis=0)
    log_init = log_softmax(p.init_alpha, axis=1)
    logQ = log_softmax(
        p.alpha[None, :, :, :] + np.einsum("abcp,tp->tabc", p.beta, X),
        axis=3,
    )

    pair_alpha_by_t: list[np.ndarray] = []
    pair_alpha = log_pi[:, None] + logB[0, :, None] + log_init + logB[1, None, :]
    pair_alpha_by_t.append(pair_alpha)
    for t in range(2, T):
        pair_alpha = logB[t, None, :] + logsumexp(
            pair_alpha[:, :, None] + logQ[t, :, :, :],
            axis=0,
        )
        pair_alpha_by_t.append(pair_alpha)

    ll = float(logsumexp(pair_alpha_by_t[-1]))

    pair_beta_by_t: list[np.ndarray] = [np.zeros_like(pair_alpha_by_t[-1])]
    pair_beta = pair_beta_by_t[0]
    for t in reversed(range(1, T - 1)):
        pair_beta = logsumexp(
            logQ[t + 1, :, :, :] + logB[t + 1, None, None, :] + pair_beta[None, :, :],
            axis=2,
        )
        pair_beta_by_t.insert(0, pair_beta)

    log_gamma = np.empty((T, p.mu.shape[0]))
    first_pair = pair_alpha_by_t[0] + pair_beta_by_t[0]
    log_gamma[0] = logsumexp(first_pair, axis=1) - ll
    for t in range(1, T):
        pair_index = t - 1
        pair_joint = pair_alpha_by_t[pair_index] + pair_beta_by_t[pair_index]
        log_gamma[t] = logsumexp(pair_joint, axis=0) - ll
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi_init = np.exp(pair_alpha_by_t[0] + pair_beta_by_t[0] - ll)
    xi_init /= max(float(xi_init.sum()), 1e-300)

    xi_trans = np.empty((T - 2, p.mu.shape[0], p.mu.shape[0], p.mu.shape[0]))
    for out_idx, t in enumerate(range(2, T)):
        log_xi = (
            pair_alpha_by_t[t - 2][:, :, None]
            + logQ[t, :, :, :]
            + logB[t, None, None, :]
            + pair_beta_by_t[t - 1][None, :, :]
            - ll
        )
        xi = np.exp(log_xi)
        total = float(xi.sum())
        if total > 0:
            xi /= total
        xi_trans[out_idx] = xi
    return ll, gamma, xi_init, xi_trans


def kmeans_initial_params(
    Y_stack: np.ndarray,
    X_stack: np.ndarray,
    Z_stack: np.ndarray,
    J: int,
    seed: int,
    diag_bias: float,
    sigma_min: float,
    sigma_max: float,
    jitter: float = 0.0,
) -> HigherOrderParams:
    _, _, D = Y_stack.shape
    P = X_stack.shape[2]
    K = Z_stack.shape[2]
    y_flat = Y_stack.reshape(-1, D)
    rng = np.random.default_rng(seed)
    labels = KMeans(n_clusters=J, n_init=10, random_state=seed).fit_predict(y_flat)
    centers = np.vstack([y_flat[labels == j].mean(axis=0) for j in range(J)])
    order = np.argsort(centers[:, 0])
    inverse = np.empty(J, dtype=int)
    inverse[order] = np.arange(J)
    labels = inverse[labels]
    centers = centers[order]

    mu = centers + rng.normal(0.0, jitter, centers.shape)
    log_sigma = np.empty((J, D))
    for j in range(J):
        subset = y_flat[labels == j]
        if len(subset) < 2:
            subset = y_flat
        log_sigma[j] = np.log(np.clip(subset.std(axis=0), sigma_min, sigma_max))

    logit_pi = np.zeros(J)
    init_alpha = np.eye(J) * diag_bias
    alpha = np.zeros((J, J, J))
    for previous in range(J):
        for current in range(J):
            alpha[previous, current, current] = diag_bias
    beta = np.zeros((J, J, J, P))
    W = rng.normal(0.0, 0.01 + jitter, (J, D, K))
    return HigherOrderParams(
        logit_pi=logit_pi,
        init_alpha=init_alpha,
        alpha=alpha,
        beta=beta,
        mu=mu,
        W=W,
        log_sigma=log_sigma,
    )


def weighted_linear_emissions(
    Y_all: np.ndarray,
    Z_all: np.ndarray,
    gamma_all: np.ndarray,
    sigma_min: float,
    sigma_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, D = Y_all.shape
    J = gamma_all.shape[1]
    K = Z_all.shape[1]
    design = np.column_stack([np.ones(len(Z_all)), Z_all])
    ridge = 1e-6 * np.eye(K + 1)
    ridge[0, 0] = 0.0
    mu = np.zeros((J, D))
    W = np.zeros((J, D, K))
    log_sigma = np.zeros((J, D))

    for j in range(J):
        weights = np.asarray(gamma_all[:, j], dtype=float)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-8:
            weights = np.ones_like(weights)
            weight_sum = float(weights.sum())
        weighted_design = design * weights[:, None]
        lhs = design.T @ weighted_design + ridge
        for d in range(D):
            rhs = design.T @ (weights * Y_all[:, d])
            try:
                coef = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            fitted = design @ coef
            resid = Y_all[:, d] - fitted
            sigma = np.sqrt(max(float(np.sum(weights * resid**2) / weight_sum), sigma_min**2))
            mu[j, d] = coef[0]
            W[j, d, :] = coef[1:]
            log_sigma[j, d] = np.log(np.clip(sigma, sigma_min, sigma_max))
    return mu, W, log_sigma


def fit_weighted_multinomial(
    x_design: np.ndarray,
    class_weights: np.ndarray,
    theta0: np.ndarray,
    *,
    maxiter: int,
    l2: float,
) -> np.ndarray:
    row_weight = class_weights.sum(axis=1)
    mask = row_weight > 1e-10
    if int(mask.sum()) < 2 or float(row_weight[mask].sum()) <= 1e-8:
        return theta0

    X = x_design[mask]
    W = class_weights[mask]
    row_weight = row_weight[mask]
    J, Q = theta0.shape

    def objective(theta_flat: np.ndarray) -> tuple[float, np.ndarray]:
        theta = theta_flat.reshape(J, Q)
        logits = X @ theta.T
        log_probs = log_softmax(logits, axis=1)
        probs = np.exp(log_probs)
        value = -float(np.sum(W * log_probs))
        if l2 > 0:
            value += 0.5 * l2 * float(np.sum(theta[:, 1:] ** 2))
        grad = (probs * row_weight[:, None] - W).T @ X
        if l2 > 0:
            grad[:, 1:] += l2 * theta[:, 1:]
        return value, grad.ravel()

    res = minimize(
        lambda theta: objective(theta),
        theta0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "maxfun": maxiter * 25, "ftol": 1e-8, "gtol": 1e-5},
    )
    return res.x.reshape(J, Q)


def m_step_second_order(
    current: HigherOrderParams,
    Y_all: np.ndarray,
    Z_all: np.ndarray,
    gamma_all: np.ndarray,
    gamma0_sum: np.ndarray,
    xi_init_sum: np.ndarray,
    X_trans: np.ndarray,
    xi_trans: np.ndarray,
    *,
    sigma_min: float,
    sigma_max: float,
    transition_maxiter: int,
    transition_l2: float,
) -> HigherOrderParams:
    J = gamma_all.shape[1]
    P = X_trans.shape[1]
    eps = 1e-10

    pi = gamma0_sum + eps
    logit_pi = np.log(pi / pi.sum())

    init_probs = xi_init_sum + eps
    init_probs = init_probs / init_probs.sum(axis=1, keepdims=True)
    init_alpha = np.log(init_probs)

    mu, W, log_sigma = weighted_linear_emissions(Y_all, Z_all, gamma_all, sigma_min, sigma_max)

    x_design = np.column_stack([np.ones(len(X_trans)), X_trans])
    alpha = np.zeros_like(current.alpha)
    beta = np.zeros_like(current.beta)
    for previous in range(J):
        for current_state in range(J):
            theta0 = np.column_stack(
                [
                    current.alpha[previous, current_state, :],
                    current.beta[previous, current_state, :, :],
                ]
            )
            theta = fit_weighted_multinomial(
                x_design,
                xi_trans[:, previous, current_state, :],
                theta0,
                maxiter=transition_maxiter,
                l2=transition_l2,
            )
            alpha[previous, current_state, :] = theta[:, 0]
            beta[previous, current_state, :, :] = theta[:, 1:]

    return HigherOrderParams(
        logit_pi=logit_pi,
        init_alpha=init_alpha,
        alpha=alpha,
        beta=beta,
        mu=mu,
        W=W,
        log_sigma=log_sigma,
    )


def fit_second_order_model_em(
    J: int,
    Y_stack: np.ndarray,
    X_stack: np.ndarray,
    Z_stack: np.ndarray,
    *,
    seed: int,
    n_starts: int,
    em_iters: int,
    em_tol: float,
    diag_bias: float,
    sigma_min: float,
    sigma_max: float,
    transition_maxiter: int,
    transition_l2: float,
    print_every: int,
    warm_start: HigherOrderParams | None = None,
) -> tuple[HigherOrderParams, object, bool]:
    N, T, D = Y_stack.shape
    P = X_stack.shape[2]
    K = Z_stack.shape[2]
    Y_all = Y_stack.reshape(N * T, D)
    Z_all = Z_stack.reshape(N * T, K)
    X_trans = X_stack[:, 2:, :].reshape(N * (T - 2), P)

    best_params: HigherOrderParams | None = None
    best_ll = -np.inf
    best_iter = 0
    best_converged = False

    for start_idx in range(n_starts):
        if start_idx == 0 and warm_start is not None:
            params = warm_start
        else:
            params = kmeans_initial_params(
                Y_stack,
                X_stack,
                Z_stack,
                J,
                seed + start_idx,
                diag_bias,
                sigma_min,
                sigma_max,
                jitter=0.02 * start_idx,
            )
        previous_ll = -np.inf
        converged = False
        for iteration in range(1, em_iters + 1):
            gamma_blocks = []
            xi_init_sum = np.zeros((J, J))
            gamma0_sum = np.zeros(J)
            xi_trans_blocks = []
            ll_total = 0.0
            for n in range(N):
                ll, gamma, xi_init, xi_trans = second_order_expectations(
                    params,
                    Y_stack[n],
                    X_stack[n],
                    Z_stack[n],
                )
                ll_total += ll
                gamma_blocks.append(gamma)
                gamma0_sum += gamma[0]
                xi_init_sum += xi_init
                xi_trans_blocks.append(xi_trans)

            gamma_all = np.vstack(gamma_blocks)
            xi_trans_all = np.vstack(xi_trans_blocks)
            if iteration % max(print_every, 1) == 0 or iteration == 1:
                delta = ll_total - previous_ll if np.isfinite(previous_ll) else np.nan
                print(
                    f"    EM order=2 J={J} start {start_idx + 1}/{n_starts} "
                    f"iter={iteration} LL={ll_total:.2f} delta={delta:.4f}",
                    flush=True,
                )
            if np.isfinite(previous_ll) and abs(ll_total - previous_ll) < em_tol:
                converged = True
                previous_ll = ll_total
                break
            previous_ll = ll_total
            params = m_step_second_order(
                params,
                Y_all,
                Z_all,
                gamma_all,
                gamma0_sum,
                xi_init_sum,
                X_trans,
                xi_trans_all,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                transition_maxiter=transition_maxiter,
                transition_l2=transition_l2,
            )

        final_ll = batched_second_order_ll(params, Y_stack, X_stack, Z_stack)
        if final_ll > best_ll:
            best_ll = final_ll
            best_params = params
            best_iter = iteration
            best_converged = converged
        print(
            f"    EM done: order=2 J={J} start {start_idx + 1}/{n_starts} "
            f"LL={final_ll:.2f} converged={converged} iter={iteration}",
            flush=True,
        )

    if best_params is None:
        raise RuntimeError("EM did not produce a candidate.")

    res = SimpleNamespace(
        true_ll=float(best_ll),
        true_negll=-float(best_ll),
        k_params=len(pack_params(best_params)),
        x=pack_params(best_params),
        success=bool(best_converged),
        nit=int(best_iter),
        message="EM converged" if best_converged else "EM iteration limit reached",
    )
    return best_params, res, bool(best_converged)


def fit_second_order_model(
    J: int,
    Y_stack: np.ndarray,
    X_stack: np.ndarray,
    Z_stack: np.ndarray,
    *,
    seed: int,
    maxiter: int,
    maxfun: int,
    n_starts: int,
    l2: float,
    diag_bias: float,
    sigma_min: float,
    sigma_max: float,
    ftol: float,
    gtol: float,
    print_every: int,
    warm_start: HigherOrderParams | None = None,
) -> tuple[HigherOrderParams, object, bool]:
    N, _, D = Y_stack.shape
    P = X_stack.shape[2]
    K = Z_stack.shape[2]
    log_sigma_low = float(np.log(sigma_min))
    log_sigma_high = float(np.log(sigma_max))

    y_flat = Y_stack.reshape(-1, D)
    y_mean = y_flat.mean(axis=0)
    y_std = np.maximum(y_flat.std(axis=0), 1e-3)

    n_logit_pi = J
    n_init_alpha = J * J
    n_alpha = J * J * J
    n_beta = J * J * J * P
    n_mu = J * D
    n_w = J * D * K
    n_log_sigma = J * D
    log_sigma_start = n_logit_pi + n_init_alpha + n_alpha + n_beta + n_mu + n_w
    total_params = log_sigma_start + n_log_sigma
    bounds = [(None, None)] * total_params
    for i in range(log_sigma_start, total_params):
        bounds[i] = (log_sigma_low, log_sigma_high)

    def neg_ll(theta: np.ndarray) -> float:
        p = unpack_params(theta, J, D, P, K)
        ll = batched_second_order_ll(p, Y_stack, X_stack, Z_stack)
        return -ll if np.isfinite(ll) else 1e40

    def objective(theta: np.ndarray) -> float:
        base = neg_ll(theta)
        if not np.isfinite(base) or l2 <= 0:
            return base if np.isfinite(base) else 1e40
        p = unpack_params(theta, J, D, P, K)
        penalty = (
            np.sum(p.init_alpha**2)
            + np.sum(p.alpha**2)
            + np.sum(p.beta**2)
            + np.sum(p.W**2)
            + 0.10 * np.sum(p.mu**2)
        )
        return float(base + l2 * penalty)

    runs = []
    for start_idx in range(n_starts):
        rng = np.random.default_rng(seed + start_idx)
        if start_idx == 0 and warm_start is not None:
            p0 = warm_start
        else:
            p0 = smart_init_params(
                rng,
                J,
                D,
                P,
                K,
                y_mean,
                y_std,
                log_sigma_low,
                log_sigma_high,
                diag_bias,
            )
        theta0 = pack_params(p0)
        start_time = time.time()
        counter = {"i": 0}

        def callback(_: np.ndarray) -> None:
            counter["i"] += 1
            if counter["i"] % print_every == 0:
                elapsed_min = (time.time() - start_time) / 60.0
                print(
                    f"    order=2 J={J} start {start_idx + 1}/{n_starts} "
                    f"iter={counter['i']} elapsed={elapsed_min:.1f} min",
                    flush=True,
                )

        res = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={"maxiter": maxiter, "maxfun": maxfun, "ftol": ftol, "gtol": gtol},
        )
        true_negll = neg_ll(res.x)
        res.true_negll = true_negll
        res.true_ll = -true_negll
        res.k_params = len(res.x)
        p_hat = unpack_params(res.x, J, D, P, K)
        runs.append((p_hat, res, true_negll))
        print(
            f"    done: order=2 J={J} start {start_idx + 1}/{n_starts} "
            f"success={res.success} nit={getattr(res, 'nit', None)} "
            f"true_negLL={true_negll:.2f} msg={res.message}",
            flush=True,
        )

    converged = [(p, r, tnl) for (p, r, tnl) in runs if bool(r.success)]
    if converged:
        best_p, best_res, _ = min(converged, key=lambda item: item[2])
        return best_p, best_res, True
    best_p, best_res, _ = min(runs, key=lambda item: item[2])
    return best_p, best_res, False


def occupancy_metrics(p: HigherOrderParams, data: HMMData) -> tuple[float, float]:
    gammas = []
    for y_i, x_i, z_i in zip(data.Y, data.X, data.Z):
        _, log_gamma = second_order_forward_backward(p, y_i, x_i, z_i)
        gammas.append(np.exp(log_gamma))
    gamma_all = np.concatenate(gammas, axis=0)
    return float(gamma_all.mean(axis=0).min()), float(gamma_all.max(axis=1).mean())


def score_row(
    spec_name: str,
    transition_cols: list[str],
    J: int,
    p_hat: HigherOrderParams,
    res: object,
    strict_converged: bool,
    data: HMMData,
    runtime_min: float,
    source: str,
) -> dict[str, object]:
    n_obs = int(sum(len(y) for y in data.Y))
    ll_total = float(getattr(res, "true_ll", np.nan))
    k_params = int(getattr(res, "k_params", len(getattr(res, "x", []))))
    aic = 2.0 * k_params - 2.0 * ll_total
    bic = np.log(n_obs) * k_params - 2.0 * ll_total
    occupancy_min, certainty_mean = occupancy_metrics(p_hat, data)
    return {
        "model_order": 2,
        "spec": spec_name,
        "J": int(J),
        "P": len(transition_cols),
        "transition_cols": ", ".join(transition_cols),
        "LL": ll_total,
        "AIC": aic,
        "BIC": bic,
        "k_params": k_params,
        "n_obs": n_obs,
        "strict_converged": bool(strict_converged and getattr(res, "success", False)),
        "scipy_success": bool(getattr(res, "success", False)),
        "iterations": int(getattr(res, "nit", -1)) if getattr(res, "nit", None) is not None else np.nan,
        "occupancy_min": occupancy_min,
        "certainty_mean": certainty_mean,
        "runtime_min": runtime_min,
        "source": source,
        "message": str(getattr(res, "message", "")),
    }


def data_signature(path: Path) -> str:
    return f"{path.resolve()}::{path.stat().st_size}::{path.stat().st_mtime_ns}"


def checkpoint_path(signature: str, spec_name: str, J: int) -> Path:
    checkpoint_dir = ARTIFACT_DIR / "Convergence_Checkpoints_higher_order_v3" / hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"candidate_v3_order2_{spec_name}_J{J}.pkl"


def first_order_checkpoint_candidates(spec_name: str, J: int) -> list[Path]:
    filename = f"candidate_v3_{spec_name}_J{J}.pkl"
    roots = [
        ARTIFACT_DIR / "Convergence_Checkpoints",
        ARTIFACT_DIR / "Convergence_Checkpoints_extra_states_v3",
        ARTIFACT_DIR / "Convergence_Checkpoints_candidate_continuations_v3",
    ]
    candidates: list[Path] = []
    for root in roots:
        direct = root / filename
        if direct.exists():
            candidates.append(direct)
        if root.exists():
            candidates.extend(path for path in root.rglob(filename) if path not in candidates)
    return candidates


def load_first_order_warm_start(
    spec_name: str,
    J: int,
    transition_cols: list[str],
) -> HigherOrderParams | None:
    expected_ll = expected_first_order_ll(spec_name, J)
    for path in first_order_checkpoint_candidates(spec_name, J):
        try:
            with path.open("rb") as f:
                checkpoint = pickle.load(f)
        except Exception as exc:
            print(f"Could not load first-order checkpoint {path}: {exc}", flush=True)
            continue
        if checkpoint.get("spec") != spec_name or int(checkpoint.get("J", -1)) != int(J):
            continue
        if list(checkpoint.get("transition_cols", [])) != list(transition_cols):
            continue
        row_ll = checkpoint.get("row", {}).get("LL")
        if expected_ll is not None and row_ll is not None:
            row_ll = float(row_ll)
            if row_ll < expected_ll - 1.0:
                print(
                    f"Skipping stale first-order checkpoint {path}: "
                    f"checkpoint LL={row_ll:.2f}, comparison LL={expected_ll:.2f}",
                    flush=True,
                )
                continue
        p = checkpoint.get("params")
        if p is None:
            continue
        if p.beta.shape[2] != len(transition_cols):
            continue
        alpha = np.empty((J, J, J))
        beta = np.empty((J, J, J, len(transition_cols)))
        for previous in range(J):
            alpha[previous, :, :] = p.alpha
            beta[previous, :, :, :] = p.beta
        print(f"Using first-order warm start from {path}", flush=True)
        return HigherOrderParams(
            logit_pi=np.array(p.logit_pi, copy=True),
            init_alpha=np.array(p.alpha, copy=True),
            alpha=alpha,
            beta=beta,
            mu=np.array(p.mu, copy=True),
            W=np.array(p.W, copy=True),
            log_sigma=np.array(p.log_sigma, copy=True),
        )
    print(f"No first-order warm start found for spec={spec_name} J={J}", flush=True)
    return None


def write_csv_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def write_text_with_fallback(text: str, path: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def append_or_replace(path: Path, row: dict[str, object]) -> Path:
    new_df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df = (
        new_df.drop_duplicates(subset=["model_order", "spec", "J"], keep="last")
        .sort_values(["spec", "J", "model_order"])
        .reset_index(drop=True)
    )
    return write_csv_with_fallback(new_df, path)


def load_first_order_results() -> pd.DataFrame:
    for path in FIRST_ORDER_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            df["model_order"] = 1
            return df
    return pd.DataFrame()


def expected_first_order_ll(spec_name: str, J: int) -> float | None:
    first = load_first_order_results()
    if first.empty:
        return None
    subset = first[first["spec"].eq(spec_name) & first["J"].eq(J)].copy()
    if subset.empty:
        return None
    value = pd.to_numeric(pd.Series([subset.iloc[0]["LL"]]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return None
    return float(value)


def build_comparison(higher_order: pd.DataFrame, specs: list[str], states: list[int]) -> pd.DataFrame:
    first = load_first_order_results()
    frames = []
    if not first.empty:
        first = first[first["spec"].isin(specs) & first["J"].isin(states)].copy()
        frames.append(first)
    if not higher_order.empty:
        higher = higher_order[higher_order["spec"].isin(specs) & higher_order["J"].isin(states)].copy()
        frames.append(higher)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["LL"] = pd.to_numeric(combined["LL"], errors="coerce")
    combined["AIC"] = pd.to_numeric(combined["AIC"], errors="coerce")
    combined["BIC"] = pd.to_numeric(combined["BIC"], errors="coerce")
    combined["strict_converged"] = combined["strict_converged"].astype(str).str.lower().isin(["true", "1", "yes"])
    combined = combined.sort_values(["spec", "J", "model_order"]).reset_index(drop=True)

    deltas = []
    for (spec_name, J), group in combined.groupby(["spec", "J"], observed=True):
        first_row = group[group["model_order"].eq(1)]
        higher_row = group[group["model_order"].eq(2)]
        if first_row.empty or higher_row.empty:
            continue
        f = first_row.iloc[0]
        h = higher_row.iloc[0]
        deltas.append(
            {
                "spec": spec_name,
                "J": int(J),
                "first_order_LL": float(f["LL"]),
                "higher_order_LL": float(h["LL"]),
                "delta_LL_higher_minus_first": float(h["LL"] - f["LL"]),
                "first_order_AIC": float(f["AIC"]),
                "higher_order_AIC": float(h["AIC"]),
                "delta_AIC_higher_minus_first": float(h["AIC"] - f["AIC"]),
                "first_order_BIC": float(f["BIC"]),
                "higher_order_BIC": float(h["BIC"]),
                "delta_BIC_higher_minus_first": float(h["BIC"] - f["BIC"]),
                "first_order_k_params": int(f["k_params"]),
                "higher_order_k_params": int(h["k_params"]),
                "first_order_strict_converged": bool(f["strict_converged"]),
                "higher_order_strict_converged": bool(h["strict_converged"]),
                "bic_prefers": "higher_order" if float(h["BIC"]) < float(f["BIC"]) else "first_order",
                "aic_prefers": "higher_order" if float(h["AIC"]) < float(f["AIC"]) else "first_order",
            }
        )
    delta_df = pd.DataFrame(deltas)
    if delta_df.empty:
        return combined
    return combined.merge(delta_df, on=["spec", "J"], how="left")


def make_note(comparison: pd.DataFrame) -> str:
    lines = [
        "# Higher-Order HMM Comparison",
        "",
        "The higher-order model is a second-order nonhomogeneous HMM: transitions use the two-state latent history,",
        "`P(s_t | s_{t-1}, s_{t-2}, x_t)`, while emissions remain tied to the current latent state.",
        "",
    ]
    if comparison.empty or "delta_BIC_higher_minus_first" not in comparison.columns:
        lines.append("No complete first-order vs higher-order pairs were available for comparison.")
        return "\n".join(lines) + "\n"

    best_aic = comparison.loc[comparison["AIC"].idxmin()]
    best_bic = comparison.loc[comparison["BIC"].idxmin()]
    lines.extend(
        [
            "## Overall Selection",
            "",
            f"- Best AIC: order={int(best_aic['model_order'])}, spec={best_aic['spec']}, "
            f"J={int(best_aic['J'])}, AIC={float(best_aic['AIC']):.2f}, BIC={float(best_aic['BIC']):.2f}.",
            f"- Best BIC: order={int(best_bic['model_order'])}, spec={best_bic['spec']}, "
            f"J={int(best_bic['J'])}, AIC={float(best_bic['AIC']):.2f}, BIC={float(best_bic['BIC']):.2f}.",
            "",
        ]
    )

    delta = (
        comparison[["spec", "J", "delta_LL_higher_minus_first", "delta_AIC_higher_minus_first", "delta_BIC_higher_minus_first", "bic_prefers", "aic_prefers"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["spec", "J"])
    )
    if delta.empty:
        lines.append("No complete first-order vs higher-order pairs were available for comparison.")
    else:
        lines.extend(["## Paired Results", ""])
        for _, row in delta.iterrows():
            lines.append(
                f"- {row['spec']} J={int(row['J'])}: "
                f"delta LL={row['delta_LL_higher_minus_first']:.2f}, "
                f"delta AIC={row['delta_AIC_higher_minus_first']:.2f}, "
                f"delta BIC={row['delta_BIC_higher_minus_first']:.2f}; "
                f"BIC prefers {row['bic_prefers']}."
            )
    lines.extend(
        [
            "",
            "## Convergence Caution",
            "",
            "The higher-order rows are EM screening fits. In the current run they reached the configured EM iteration limit, "
            "so the information criteria should be treated as provisional until continued to a tighter EM tolerance.",
            "",
            "Lower AIC/BIC is better. Positive delta LL means the higher-order model fits the observed sequences better before penalizing extra parameters.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_candidate(
    spec_name: str,
    J: int,
    args: argparse.Namespace,
    signature: str,
) -> dict[str, object]:
    transition_cols = TRANSITION_SPECS[spec_name]
    ckpt_path = checkpoint_path(signature, spec_name, J)
    checkpoint_warm_start = None

    if ckpt_path.exists() and not args.refit:
        with ckpt_path.open("rb") as f:
            checkpoint = pickle.load(f)
        row = checkpoint.get("row")
        if row is not None:
            print(f"Reusing checkpoint for order=2 spec={spec_name} J={J}", flush=True)
            return row
    elif ckpt_path.exists() and args.continue_fit:
        with ckpt_path.open("rb") as f:
            checkpoint = pickle.load(f)
        checkpoint_warm_start = checkpoint.get("params")
        if checkpoint_warm_start is not None:
            print(f"Continuing from higher-order checkpoint {ckpt_path}", flush=True)

    data = load_sequences(args.data, transition_cols)
    Y_stack = np.stack(data.Y)
    X_stack = np.stack(data.X)
    Z_stack = np.stack(data.Z)

    print(f"=== Higher-order candidate: order=2 spec={spec_name} J={J} ===", flush=True)
    if checkpoint_warm_start is not None:
        warm_start = checkpoint_warm_start
    else:
        warm_start = load_first_order_warm_start(spec_name, J, transition_cols) if args.first_order_warm_start else None
    t0 = time.time()
    p_hat, res, is_conv = fit_second_order_model_em(
        J,
        Y_stack,
        X_stack,
        Z_stack,
        seed=args.seed + 1000 * list(TRANSITION_SPECS).index(spec_name) + J,
        n_starts=args.n_starts,
        em_iters=args.em_iters,
        em_tol=args.em_tol,
        diag_bias=args.diag_bias,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        transition_maxiter=args.transition_maxiter,
        transition_l2=args.transition_l2,
        print_every=args.print_every,
        warm_start=warm_start,
    )
    row = score_row(
        spec_name,
        transition_cols,
        J,
        p_hat,
        res,
        is_conv,
        data,
        runtime_min=(time.time() - t0) / 60.0,
        source="second_order_full_data",
    )
    row["dataset_path"] = str(args.data.resolve())
    row["data_signature"] = signature
    with ckpt_path.open("wb") as f:
        pickle.dump(
            {
                "model_order": 2,
                "spec": spec_name,
                "J": int(J),
                "transition_cols": transition_cols,
                "params": p_hat,
                "res": res,
                "row": row,
                "data_signature": signature,
            },
            f,
        )
    append_or_replace(HIGHER_ORDER_CSV, row)
    print(
        f"Finished order=2 spec={spec_name} J={J}: "
        f"LL={row['LL']:.2f}, AIC={row['AIC']:.2f}, BIC={row['BIC']:.2f}, "
        f"converged={row['strict_converged']}",
        flush=True,
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit second-order V3 HMM candidates and compare with first-order HMMs.")
    parser.add_argument("--states", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--specs", nargs="+", choices=sorted(TRANSITION_SPECS), default=["single_period"])
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--n-starts", type=int, default=2)
    parser.add_argument("--em-iters", type=int, default=35)
    parser.add_argument("--em-tol", type=float, default=1e-3)
    parser.add_argument("--transition-maxiter", type=int, default=80)
    parser.add_argument("--transition-l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--diag-bias", type=float, default=2.5)
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=3.5)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--continue-fit", action="store_true")
    parser.add_argument("--no-first-order-warm-start", dest="first_order_warm_start", action="store_false")
    parser.set_defaults(first_order_warm_start=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    signature = data_signature(args.data)
    rows = []
    for spec_name in args.specs:
        for J in args.states:
            rows.append(run_candidate(spec_name, J, args, signature))

    if HIGHER_ORDER_CSV.exists():
        higher = pd.read_csv(HIGHER_ORDER_CSV)
    else:
        higher = pd.DataFrame(rows)
    comparison = build_comparison(higher, list(args.specs), list(args.states))
    comparison_path = write_csv_with_fallback(comparison, COMPARISON_CSV)
    note_path = write_text_with_fallback(make_note(comparison), NOTE_PATH)

    print(f"Saved higher-order rows to {HIGHER_ORDER_CSV.resolve()}", flush=True)
    print(f"Saved comparison to {comparison_path.resolve()}", flush=True)
    print(f"Saved note to {note_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
