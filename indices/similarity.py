"""Wollack omega answer-similarity index for dichotomous items.

omega treats one response vector as the source and asks whether the copier
matches it more often than the copier's own item probabilities allow.
The statistic is not symmetric. `similarity_max` is the larger of (1) omega
against a fixed-size random set of other candidates on the same form and
split and (2) omega against the seized answer key. The key is not a scored
candidate.
"""

from __future__ import annotations

import numpy as np


def _as_float(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)


def omega_vs_vector(X: np.ndarray, P: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Omega of each row of X against one source response vector.

    P is the copier's model probability of a correct answer, shape (n, m).
    """
    responses = _as_float(X)
    prob = np.clip(_as_float(P), 1e-6, 1.0 - 1e-6)
    src = _as_float(source).reshape(-1)
    if src.shape[0] != responses.shape[1]:
        raise ValueError("Source length must equal the number of items.")
    matches = responses @ src + (1.0 - responses) @ (1.0 - src)
    expected_item = (1.0 - prob) + src[None, :] * (2.0 * prob - 1.0)
    variance = np.clip(np.sum(expected_item * (1.0 - expected_item), axis=1), 1e-12, None)
    return (matches - expected_item.sum(axis=1)) / np.sqrt(variance)


def pairwise_omega(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Return omega[j, s] with s as the source and j as the copier.

    The diagonal is -inf so a candidate is never their own source.
    """
    responses = _as_float(X)
    prob = np.clip(_as_float(P), 1e-6, 1.0 - 1e-6)
    n_persons, _ = responses.shape
    transposed = responses.T
    matches = responses @ transposed + (1.0 - responses) @ (1.0 - transposed)
    one_minus = 1.0 - prob
    base = one_minus.sum(axis=1)
    coef = 2.0 * prob - 1.0
    expected = base[:, None] + coef @ transposed
    sum_sq = np.sum(one_minus * one_minus, axis=1)
    coef2 = 2.0 * one_minus * coef + coef * coef
    sum_expected_sq = sum_sq[:, None] + coef2 @ transposed
    variance = np.clip(expected - sum_expected_sq, 1e-12, None)
    omega = (matches - expected) / np.sqrt(variance)
    if n_persons:
        np.fill_diagonal(omega, -np.inf)
    return omega


def similarity_max(
    X: np.ndarray,
    P: np.ndarray,
    key: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Max omega versus k other candidates and versus the seized key."""
    n_persons = X.shape[0]
    k_use = min(int(k), max(n_persons - 1, 0))
    if n_persons >= 2 and k_use >= 1:
        omega = pairwise_omega(X, P)
        noise = rng.random((n_persons, n_persons))
        np.fill_diagonal(noise, np.inf)
        refs = np.argpartition(noise, k_use, axis=1)[:, :k_use]
        sampled = np.take_along_axis(omega, refs, axis=1)
        best_peer = sampled.max(axis=1)
    else:
        best_peer = np.full(n_persons, -np.inf, dtype=np.float64)
    best_key = omega_vs_vector(X, P, key)
    return np.maximum(best_peer, best_key)
