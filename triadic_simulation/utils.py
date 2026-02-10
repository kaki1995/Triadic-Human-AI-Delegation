# triadic_sim/utils.py
from __future__ import annotations

import math
from typing import List, Sequence
import numpy as np


# ------------------------------------------------------------------
# Numerical helpers
# ------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """
    Standard logistic (sigmoid) function.

    Used for mapping latent utilities or scores to probabilities.
    """
    # guard against overflow
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def clip01(x: float) -> float:
    """
    Clip numeric value to the closed interval [0, 1].

    Ensures all probabilities and shares remain valid.
    """
    if x != x:  # NaN check
        return 0.0
    return float(min(1.0, max(0.0, x)))


def safe_divide(num: float, den: float | None, default: float = 0.0) -> float:
    """
    Safe division helper.

    Returns `default` if denominator is zero or invalid.
    """
    if den is None or den == 0 or den != den:
        return default
    return float(num / den)


def zscore(x: float, mean: float, std: float | None) -> float:
    """
    Compute a z-score with numerical safety.

    Used when constructing standardized KPI indices.
    """
    if std is None or std <= 1e-9 or x != x:
        return 0.0
    return float((x - mean) / std)


# ------------------------------------------------------------------
# Sampling helpers
# ------------------------------------------------------------------

def normalize_probs(probs: Sequence[float]) -> np.ndarray:
    """
    Normalize a sequence of non-negative weights into probabilities.

    Falls back to uniform distribution if sum is zero or invalid.
    """
    p = np.array(probs, dtype=float)

    # Replace NaNs or negatives defensively
    p = np.where(np.isnan(p), 0.0, p)
    p = np.where(p < 0.0, 0.0, p)

    s = p.sum()
    if s <= 0.0:
        return np.ones_like(p) / len(p)
    return p / s


def choice_with_probs(
    rng: np.random.Generator,
    options: List[str],
    probs: Sequence[float],
) -> str:
    """
    Sample one item from `options` given (possibly unnormalized) probabilities.

    Robustness:
    - Handles NaNs, negatives, and zero-sum probabilities
    - Falls back to uniform choice if needed

    Args:
        rng: numpy random generator
        options: list of categorical labels
        probs: list of weights (need not sum to 1)

    Returns:
        Chosen option (str)
    """
    if len(options) != len(probs):
        raise ValueError("options and probs must have the same length")

    p = normalize_probs(probs)
    return str(rng.choice(options, p=p))
