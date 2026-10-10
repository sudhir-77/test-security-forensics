"""Leaked-item residual for a known compromised item set.

Ability is estimated from secure items only. The index is the standardized
difference between the observed leaked-item score and the score expected from
that ability. The leaked set is an input from the program, not something
searched for on the evaluation cohort.
"""

from __future__ import annotations

import numpy as np

from indices.mathutil import sigmoid
from indices.person_fit import estimate_theta_mle


def preknowledge_from_theta(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    leaked_mask: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    leaked = np.asarray(leaked_mask, dtype=bool)
    if int(leaked.sum()) == 0:
        raise ValueError("Leaked mask is empty.")
    responses = np.asarray(X, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    logits = np.clip(a[leaked] * (theta[:, None] - b[leaked][None, :]), -30.0, 30.0)
    prob = sigmoid(logits)
    observed = responses[:, leaked].sum(axis=1)
    expected = prob.sum(axis=1)
    variance = np.clip(np.sum(prob * (1.0 - prob), axis=1), 1e-12, None)
    return (observed - expected) / np.sqrt(variance)


def preknowledge_diff(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    leaked_mask: np.ndarray,
) -> np.ndarray:
    leaked = np.asarray(leaked_mask, dtype=bool)
    secure = ~leaked
    if int(secure.sum()) == 0:
        raise ValueError("Preknowledge index needs at least one secure item.")
    responses = np.asarray(X, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    theta = estimate_theta_mle(responses[:, secure], a[secure], b[secure])
    return preknowledge_from_theta(responses, a, b, leaked, theta)
