"""Item-calibration gradients and a small parameter-recovery check."""

from __future__ import annotations

import numpy as np

from forensics.calibrate import calibrate_2pl, e_step, loglik_gradients, normal_quadrature
from indices.mathutil import log_one_minus_sigmoid, log_sigmoid, sigmoid


def test_quadrature_weights_sum_to_one():
    _theta, log_weights = normal_quadrature(21)
    assert abs(float(np.exp(log_weights).sum()) - 1.0) < 1e-8


def test_e_step_posteriors_sum_to_one():
    rng = np.random.default_rng(4)
    responses = (rng.random((25, 6)) < 0.6).astype(np.float64)
    a = np.ones(6)
    b = np.linspace(-1.0, 1.0, 6)
    theta, log_weights = normal_quadrature(11)
    post = e_step(responses, a, b, theta, log_weights)
    assert np.allclose(post.sum(axis=1), 1.0)


def test_loglik_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    n_persons, n_items, n_quad = 20, 2, 9
    responses = (rng.random((n_persons, n_items)) < 0.55).astype(np.float64)
    post = rng.random((n_persons, n_quad))
    post /= post.sum(axis=1, keepdims=True)
    theta, _log_w = normal_quadrature(n_quad)
    a = np.array([1.1, 0.8])
    b = np.array([-0.3, 0.4])

    def expected_loglik(a_value, b_value):
        u = theta[:, None] - b_value[None, :]
        logits = a_value[None, :] * u
        log_p = log_sigmoid(logits)
        log_q = log_one_minus_sigmoid(logits)
        total = 0.0
        for item in range(n_items):
            ll = responses[:, item][:, None] * log_p[None, :, item]
            ll = ll + (1.0 - responses[:, item][:, None]) * log_q[None, :, item]
            total += float(np.sum(post * ll))
        return total

    analytic_a, analytic_b = loglik_gradients(responses, post, theta, a, b)
    eps = 1e-5
    for item in range(n_items):
        a_plus = a.copy()
        a_minus = a.copy()
        a_plus[item] += eps
        a_minus[item] -= eps
        grad_a = (expected_loglik(a_plus, b) - expected_loglik(a_minus, b)) / (2 * eps)
        b_plus = b.copy()
        b_minus = b.copy()
        b_plus[item] += eps
        b_minus[item] -= eps
        grad_b = (expected_loglik(a, b_plus) - expected_loglik(a, b_minus)) / (2 * eps)
        assert abs(float(analytic_a[item]) - grad_a) < 1e-4
        assert abs(float(analytic_b[item]) - grad_b) < 1e-4


def test_calibrated_difficulties_track_the_true_difficulties():
    rng = np.random.default_rng(6)
    n_persons, n_items = 700, 12
    a_true = np.clip(rng.lognormal(0.0, 0.15, size=n_items), 0.7, 1.8)
    b_true = np.linspace(-1.8, 1.8, n_items)
    theta = rng.normal(0.0, 1.0, size=n_persons)
    logits = a_true[None, :] * (theta[:, None] - b_true[None, :])
    responses = (rng.random((n_persons, n_items)) < sigmoid(logits)).astype(np.float64)
    a_hat, b_hat = calibrate_2pl(responses, n_quad=15, n_iter=12, n_newton=6)
    assert float(np.corrcoef(b_true, b_hat)[0, 1]) > 0.90
    assert float(np.corrcoef(a_true, a_hat)[0, 1]) > 0.70
