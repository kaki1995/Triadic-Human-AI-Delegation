# triadic_sim/utils.py
from __future__ import annotations
import math
from typing import List
import numpy as np

def sigmoid(x: float) -> float:
    """Standard logistic function."""
    return 1.0 / (1.0 + math.exp(-x))

def clip01(x: float) -> float:
    """Clip numeric value to [0, 1]."""
    return float(min(1.0, max(0.0, x)))

def choice_with_probs(rng: np.random.Generator, options: List[str], probs: List[float]) -> str:
    """
    Sample one item from options given unnormalized probabilities.

    Args:
        rng: Numpy Generator
        options: list of labels
        probs: list of weights (need not sum to 1)

    Returns:
        Chosen option.
    """
    p = np.array(probs, dtype=float)
    p = p / p.sum()
    return str(rng.choice(options, p=p))
