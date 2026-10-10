"""Snijders-corrected lz* person-fit statistic for the 2PL."""

from __future__ import annotations

import numpy as np

from indices.mathutil import sigmoid

THETA_BOUND = 4.5


def estimate_theta_mle(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_iter: int = 25,
) -> np.ndarray:
    """Bounded Newton-Raphson MLE of ability. Theta is clipped to +/- 4.5.

    Responses that are all correct or all wrong have no finite MLE. The bound
    keeps lz* defined for those patterns.
    """
    responses = np.asarray(X, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if responses.ndim != 2:
        raise ValueError("X must be (n_persons, n_items).")
    theta = np.zeros(responses.shape[0], dtype=np.float64)
    a2 = a * a
    for _ in range(n_iter):
        logits = np.clip(a * (theta[:, None] - b[None, :]), -30.0, 30.0)
        prob = sigmoid(logits)
        resid = responses - prob
        score = resid @ a
        info = (prob * (1.0 - prob)) @ a2
        step = np.zeros_like(theta)
        ok = info > 1e-8
        step[ok] = score[ok] / info[ok]
        step = np.clip(step, -1.0, 1.0)
        theta = np.clip(theta + step, -THETA_BOUND, THETA_BOUND)
    return theta


def lz_star(theta: np.ndarray, a: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Snijders-corrected standardized log-likelihood person fit.

    With item weight w_i = logit_i = a_i (theta - b_i),

        c = -sum(w_i a_i P_i Q_i) / sum(a_i^2 P_i Q_i)
        w*_i = w_i + c a_i
        lz* = sum(w*_i (X_i - P_i)) / sqrt(sum(w*_i^2 P_i Q_i))

    The correction is the one that makes the weighted residual orthogonal to
    the 2PL score function when ability is estimated. Lower values are more
    aberrant. A one-item test has a zero statistic because no residual
    degree of freedom remains after estimating theta.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    responses = np.asarray(X, dtype=np.float64)
    logits = np.clip(a * (theta[:, None] - b[None, :]), -30.0, 30.0)
    prob = sigmoid(logits)
    weights = logits
    resid = responses - prob
    pq = prob * (1.0 - prob)
    info = pq @ (a * a)
    cov = (weights * pq) @ a
    correction = np.zeros(theta.shape[0], dtype=np.float64)
    ok = info > 1e-12
    correction[ok] = -cov[ok] / info[ok]
    weights_star = weights + correction[:, None] * a[None, :]
    numerator = np.sum(weights_star * resid, axis=1)
    denominator = np.sqrt(np.sum((weights_star**2) * pq, axis=1))
    out = np.zeros(theta.shape[0], dtype=np.float64)
    ok_den = denominator > 1e-12
    out[ok_den] = numerator[ok_den] / denominator[ok_den]
    return out
