"""Detection metrics. Higher scores are more suspicious."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC. Ties use average ranks. Positives are y == 1."""
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes.")
    ranks = rankdata(s, method="average")
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def recall_for_label(flagged: np.ndarray, labels: np.ndarray, label: str) -> float:
    mask = np.asarray(labels) == label
    if int(mask.sum()) == 0:
        raise ValueError(f"No rows with label {label}.")
    return float(np.mean(np.asarray(flagged)[mask] == 1))


def observed_fpr(flagged: np.ndarray, labels: np.ndarray, clean_label: str = "clean") -> float:
    mask = np.asarray(labels) == clean_label
    if int(mask.sum()) == 0:
        raise ValueError("No clean rows.")
    return float(np.mean(np.asarray(flagged)[mask] == 1))


def within_form_z(
    scores: np.ndarray,
    forms: np.ndarray,
    ref_scores: np.ndarray,
    ref_forms: np.ndarray,
) -> np.ndarray:
    """Z-score each form using the reference cohort mean and population sd."""
    scores = np.asarray(scores, dtype=np.float64)
    forms = np.asarray(forms)
    ref_scores = np.asarray(ref_scores, dtype=np.float64)
    ref_forms = np.asarray(ref_forms)
    out = np.zeros_like(scores)
    for form in np.unique(forms):
        ref = ref_scores[ref_forms == form]
        mu = float(ref.mean())
        sd = float(ref.std(ddof=0))
        if sd < 1e-8:
            sd = 1.0
        out[forms == form] = (scores[forms == form] - mu) / sd
    return out


def criterion_passed(value: float, op: str, threshold: float, threshold_high: float | None) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    if op == "between":
        if threshold_high is None:
            raise ValueError("between requires threshold_high.")
        return threshold <= value <= threshold_high
    raise ValueError(f"Unknown operator {op}.")


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FPR and TPR on a descending score sweep, including (0, 0) and (1, 1)."""
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    return fpr, tpr
