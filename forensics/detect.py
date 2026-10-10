"""Isolation forest plus a max-index score. Thresholds are fit on validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# High values are more suspicious: low lz*, high similarity, high preknowledge
# residual, fast (negative) response-time residual.
ORIENTATION = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)


@dataclass
class Detector:
    means: np.ndarray
    stds: np.ndarray
    scaler: StandardScaler
    forest: IsolationForest
    blend_mean_extreme: float
    blend_sd_extreme: float
    blend_mean_iforest: float
    blend_sd_iforest: float
    threshold_fpr01: float
    threshold_fpr001: float


def oriented_matrix(features: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    z = (np.asarray(features, dtype=np.float64) - means) / stds
    return z * ORIENTATION


def blend_scores(
    extreme: np.ndarray,
    iforest_score: np.ndarray,
    mean_extreme: float,
    sd_extreme: float,
    mean_iforest: float,
    sd_iforest: float,
) -> np.ndarray:
    z_extreme = (extreme - mean_extreme) / sd_extreme
    z_iforest = (iforest_score - mean_iforest) / sd_iforest
    return 0.5 * z_extreme + 0.5 * z_iforest


def _components(
    features: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    scaler: StandardScaler,
    forest: IsolationForest,
) -> tuple[np.ndarray, np.ndarray]:
    oriented = oriented_matrix(features, means, stds)
    extreme = oriented.max(axis=1)
    scaled = scaler.transform(np.asarray(features, dtype=np.float64))
    # score_samples is higher for inliers. Negate so high means anomalous.
    iforest_score = -forest.score_samples(scaled)
    return extreme, iforest_score


def fit_detector(
    train_features: np.ndarray,
    val_features: np.ndarray,
    seed: int,
    n_estimators: int,
) -> Detector:
    """Fit scaler and forest on train. Set 1% and 0.1% thresholds on val."""
    train = np.asarray(train_features, dtype=np.float64)
    val = np.asarray(val_features, dtype=np.float64)
    if not np.isfinite(train).all() or not np.isfinite(val).all():
        raise ValueError("Feature matrix contains non-finite values.")
    means = train.mean(axis=0)
    stds = np.maximum(train.std(axis=0, ddof=0), 1e-6)
    scaler = StandardScaler()
    scaler.fit(train)
    forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=int(min(256, train.shape[0])),
        contamination=0.01,
        random_state=seed,
        n_jobs=1,
        bootstrap=False,
    )
    forest.fit(scaler.transform(train))
    extreme_val, iforest_val = _components(val, means, stds, scaler, forest)
    blend_mean_extreme = float(extreme_val.mean())
    blend_sd_extreme = float(max(extreme_val.std(ddof=0), 1e-6))
    blend_mean_iforest = float(iforest_val.mean())
    blend_sd_iforest = float(max(iforest_val.std(ddof=0), 1e-6))
    combined_val = blend_scores(
        extreme_val,
        iforest_val,
        blend_mean_extreme,
        blend_sd_extreme,
        blend_mean_iforest,
        blend_sd_iforest,
    )
    return Detector(
        means=means,
        stds=stds,
        scaler=scaler,
        forest=forest,
        blend_mean_extreme=blend_mean_extreme,
        blend_sd_extreme=blend_sd_extreme,
        blend_mean_iforest=blend_mean_iforest,
        blend_sd_iforest=blend_sd_iforest,
        threshold_fpr01=float(np.quantile(combined_val, 0.99)),
        threshold_fpr001=float(np.quantile(combined_val, 0.999)),
    )


def score_features(detector: Detector, features: np.ndarray) -> dict[str, np.ndarray]:
    extreme, iforest_score = _components(
        features, detector.means, detector.stds, detector.scaler, detector.forest
    )
    combined = blend_scores(
        extreme,
        iforest_score,
        detector.blend_mean_extreme,
        detector.blend_sd_extreme,
        detector.blend_mean_iforest,
        detector.blend_sd_iforest,
    )
    flagged = (combined > detector.threshold_fpr01).astype(np.int64)
    flagged_001 = (combined > detector.threshold_fpr001).astype(np.int64)
    return {
        "extreme": extreme,
        "iforest_score": iforest_score,
        "combined_score": combined,
        "flagged": flagged,
        "flagged_fpr001": flagged_001,
    }
