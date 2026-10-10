"""Person-level residual from an item-only log response-time model."""

from __future__ import annotations

import numpy as np

RT_FLOOR_SECONDS = 0.5


def log_response_time(RT: np.ndarray) -> np.ndarray:
    times = np.asarray(RT, dtype=np.float64)
    return np.log(np.clip(times, RT_FLOOR_SECONDS, None))


def fit_time_intensity(RT: np.ndarray) -> np.ndarray:
    """Item time intensity: mean log response time on a clean training cohort."""
    return log_response_time(RT).mean(axis=0)


def rt_residual(RT: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Mean log-time residual. Negative values are faster than the item norm.

    Beta is fit on the training cohort only. The person speed parameter is
    not partialled out, because a candidate who rushes every item would
    otherwise have a mean residual of zero by construction.
    """
    beta = np.asarray(beta, dtype=np.float64)
    return (log_response_time(RT) - beta[None, :]).mean(axis=1)
