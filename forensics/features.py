"""Feature construction. Item parameters and time intensities come from train."""

from __future__ import annotations

import numpy as np

from indices.mathutil import irt_probabilities
from indices.person_fit import estimate_theta_mle, lz_star
from indices.preknowledge import preknowledge_diff
from indices.response_time import fit_time_intensity, rt_residual
from indices.similarity import similarity_max

FEATURE_NAMES = ("lz_star", "similarity_max", "preknowledge_diff", "rt_residual")


def feature_matrix(
    X: np.ndarray,
    RT: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    beta: np.ndarray,
    leaked_mask: np.ndarray,
    key: np.ndarray,
    ref_k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, total_score).

    Features columns follow FEATURE_NAMES. Ability for lz* and for the
    similarity probabilities uses every item. The preknowledge residual
    re-estimates ability on secure items only.
    """
    responses = np.asarray(X, dtype=np.float64)
    theta = estimate_theta_mle(responses, a, b)
    lz = lz_star(theta, a, b, responses)
    prob = irt_probabilities(theta, a, b)
    sim = similarity_max(responses, prob, key, ref_k, rng)
    pre = preknowledge_diff(responses, a, b, leaked_mask)
    rt = rt_residual(RT, beta)
    features = np.column_stack([lz, sim, pre, rt]).astype(np.float64)
    total = responses.sum(axis=1).astype(np.float64)
    return features, total


def train_time_intensity(RT: np.ndarray) -> np.ndarray:
    return fit_time_intensity(RT)
