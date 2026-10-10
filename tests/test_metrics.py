"""Hand-computed detection metrics and the validation threshold."""

from __future__ import annotations

import numpy as np

from forensics.detect import blend_scores, fit_detector, oriented_matrix, score_features
from forensics.metrics_eval import (
    criterion_passed,
    observed_fpr,
    recall_for_label,
    roc_auc,
    within_form_z,
)
from forensics.reporting import build_success_criteria, render_resume, resume_figures


def test_roc_auc_hand_computed_with_one_discordant_pair():
    # Scores 0, 2, 1, 3 and labels 0, 0, 1, 1.
    # Positive ranks are 2 and 4. AUC = (6 - 3) / 4 = 0.75.
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 2.0, 1.0, 3.0])
    assert abs(roc_auc(labels, scores) - 0.75) < 1e-12


def test_roc_auc_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    assert abs(roc_auc(labels, scores) - 1.0) < 1e-12


def test_recall_and_fpr_hand_computed():
    flagged = np.array([1, 0, 1, 0])
    labels = np.array(["copy", "copy", "clean", "clean"])
    assert abs(recall_for_label(flagged, labels, "copy") - 0.5) < 1e-12
    assert abs(observed_fpr(flagged, labels) - 0.5) < 1e-12


def test_criterion_interval_is_inclusive():
    assert criterion_passed(0.005, "between", 0.005, 0.015)
    assert criterion_passed(0.015, "between", 0.005, 0.015)
    assert not criterion_passed(0.004, "between", 0.005, 0.015)
    assert criterion_passed(0.80, ">=", 0.80, None)
    assert not criterion_passed(0.79, ">=", 0.80, None)


def test_within_form_z_uses_the_reference_form_only():
    ref_scores = np.array([1.0, 2.0, 3.0, 10.0])
    ref_forms = np.array(["A", "A", "A", "B"])
    scores = np.array([5.0, 10.0])
    forms = np.array(["A", "B"])
    got = within_form_z(scores, forms, ref_scores, ref_forms)
    mean_a = 2.0
    sd_a = float(np.std(np.array([1.0, 2.0, 3.0]), ddof=0))
    assert abs(float(got[0]) - ((5.0 - mean_a) / sd_a)) < 1e-12
    assert abs(float(got[1]) - 0.0) < 1e-12


def test_oriented_extreme_score_hand_computed():
    features = np.array([[-2.0, 5.0, 0.0, -1.0]])
    means = np.array([0.0, 3.0, 0.0, 0.0])
    stds = np.ones(4)
    oriented = oriented_matrix(features, means, stds)
    # Low lz and fast RT are flipped so that large means suspicious.
    assert np.allclose(oriented[0], np.array([2.0, 2.0, 0.0, 1.0]))
    assert abs(float(oriented.max()) - 2.0) < 1e-12


def test_blend_is_the_average_of_standardized_components():
    blended = blend_scores(
        extreme=np.array([4.0]),
        iforest_score=np.array([2.0]),
        mean_extreme=0.0,
        sd_extreme=1.0,
        mean_iforest=0.0,
        sd_iforest=1.0,
    )
    assert abs(float(blended[0]) - 3.0) < 1e-12


def test_validation_threshold_matches_the_held_out_quantile():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(180, 4))
    val = rng.normal(size=(220, 4))
    detector = fit_detector(train, val, seed=7, n_estimators=30)
    scored = score_features(detector, val)
    assert abs(detector.threshold_fpr01 - float(np.quantile(scored["combined_score"], 0.99))) < 1e-8
    rate = float(scored["flagged"].mean())
    assert 0.0 <= rate <= 0.05
    assert int(scored["flagged"].sum()) < len(val)


def test_resume_numbers_are_the_stored_rounded_figures():
    metrics = {
        "power_copy_fpr01": 0.9854,
        "power_preknowledge_fpr01": 0.742,
        "power_rapidguess_fpr01": 0.997,
        "observed_fpr": 0.01024,
        "auc_combined": 0.9638,
        "auc_baseline": 0.5482,
        "auc_lift": 0.4156,
        "n_evaluation": 8000,
    }
    figures = resume_figures(metrics)
    payload = {"metrics": {**metrics, "resume_figures": figures}}
    text = render_resume(payload)
    assert "98.5%" in text
    assert "74.2%" in text
    assert "99.7%" in text
    assert "0.964" in text
    assert "0.548" in text
    assert "8000" in text
    assert "1.02%" in text
    criteria = build_success_criteria(metrics, coverage=0.81)
    assert criteria[-1]["passed"] is True
    assert criteria[0]["passed"] is True
