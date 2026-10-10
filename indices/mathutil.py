"""Stable 2PL helpers shared by the simulator and the indices."""

from __future__ import annotations

import numpy as np

LOGIT_CLIP = 30.0


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(z, dtype=np.float64), -LOGIT_CLIP, LOGIT_CLIP)
    return 1.0 / (1.0 + np.exp(-clipped))


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -np.asarray(z, dtype=np.float64))


def log_one_minus_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, np.asarray(z, dtype=np.float64))


def item_logits(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a (n_persons, n_items) logit matrix a*(theta - b)."""
    theta = np.asarray(theta, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a[None, :] * (theta[:, None] - b[None, :])


def irt_probabilities(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return sigmoid(item_logits(theta, a, b))
