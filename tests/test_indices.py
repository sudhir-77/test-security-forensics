"""Hand-computed checks for the four test-security indices."""

from __future__ import annotations

import math

import numpy as np

from indices.mathutil import sigmoid
from indices.person_fit import estimate_theta_mle, lz_star
from indices.preknowledge import preknowledge_from_theta
from indices.response_time import fit_time_intensity, rt_residual
from indices.similarity import omega_vs_vector, pairwise_omega, similarity_max


def test_lz_star_matches_expanded_three_item_arithmetic():
    theta = np.array([0.0])
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-1.0, 0.0, 1.0])
    x = np.array([[1.0, 0.0, 1.0]])
    s1 = 1.0 / (1.0 + math.exp(-1.0))
    prob = np.array([s1, 0.5, 1.0 - s1])
    weights = np.array([1.0, 0.0, -1.0])
    resid = x[0] - prob
    pq = prob * (1.0 - prob)
    info = float(np.sum(pq))
    cov = float(np.sum(weights * pq))
    correction = -cov / info
    weights_star = weights + correction
    expected = float(np.sum(weights_star * resid) / math.sqrt(np.sum(weights_star**2 * pq)))
    got = float(lz_star(theta, a, b, x)[0])
    assert abs(got - expected) < 1e-12


def test_lz_star_is_zero_for_a_one_item_test():
    # After estimating ability, one item leaves no residual degree of freedom.
    got = lz_star(np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([[1.0]]))
    assert abs(float(got[0])) < 1e-12


def test_reversed_pattern_has_worse_person_fit_than_guttman():
    a = np.ones(5)
    b = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ordered = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]])
    reversed_pattern = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    theta_ordered = estimate_theta_mle(ordered, a, b)
    theta_reversed = estimate_theta_mle(reversed_pattern, a, b)
    lz_ordered = float(lz_star(theta_ordered, a, b, ordered)[0])
    lz_reversed = float(lz_star(theta_reversed, a, b, reversed_pattern)[0])
    assert lz_reversed < lz_ordered - 0.5


def test_all_correct_ability_hits_the_upper_bound():
    a = np.ones(8)
    b = np.zeros(8)
    theta = estimate_theta_mle(np.ones((1, 8)), a, b)
    assert float(theta[0]) > 3.0


def test_pairwise_omega_hand_computed_two_by_two():
    responses = np.array([[1.0, 0.0], [1.0, 0.0]])
    prob = np.array([[0.5, 0.5], [0.8, 0.2]])
    omega = pairwise_omega(responses, prob)
    assert omega[0, 0] == -np.inf
    assert omega[1, 1] == -np.inf
    assert abs(float(omega[0, 1]) - math.sqrt(2.0)) < 1e-10
    assert abs(float(omega[1, 0]) - (1.0 / math.sqrt(2.0))) < 1e-10


def test_pairwise_omega_matches_item_loop():
    rng = np.random.default_rng(0)
    responses = (rng.random((6, 5)) < 0.5).astype(np.float64)
    prob = np.clip(rng.random((6, 5)), 0.05, 0.95)
    got = pairwise_omega(responses, prob)
    expected = np.zeros((6, 6))
    for j in range(6):
        for s in range(6):
            if j == s:
                expected[j, s] = -np.inf
                continue
            source = responses[s]
            item_expected = np.where(source == 1.0, prob[j], 1.0 - prob[j])
            matches = float(np.sum(responses[j] == source))
            variance = float(np.sum(item_expected * (1.0 - item_expected)))
            expected[j, s] = (matches - float(item_expected.sum())) / math.sqrt(variance)
    assert np.allclose(got, expected, equal_nan=True)


def test_similarity_max_includes_the_seized_key():
    responses = np.array([[1.0, 0.0], [0.0, 1.0]])
    prob = np.full((2, 2), 0.5)
    key = np.array([1.0, 0.0])
    rng = np.random.default_rng(1)
    got = similarity_max(responses, prob, key, k=1, rng=rng)
    peer = pairwise_omega(responses, prob)
    key_omega = omega_vs_vector(responses, prob, key)
    expected = np.maximum(np.array([peer[0, 1], peer[1, 0]]), key_omega)
    assert np.allclose(got, expected)


def test_copier_matches_the_key_more_than_a_clean_candidate():
    rng = np.random.default_rng(2)
    n_items = 40
    a = np.ones(n_items)
    b = np.zeros(n_items)
    key = (rng.random(n_items) < 0.6).astype(np.float64)
    clean = (rng.random(n_items) < 0.5).astype(np.float64)
    copier = key.copy()
    copier[0] = 1.0 - copier[0]
    responses = np.vstack([clean, copier])
    prob = sigmoid(a * (np.array([0.0, 0.0])[:, None] - b[None, :]))
    omega = omega_vs_vector(responses, prob, key)
    assert float(omega[1]) > float(omega[0]) + 2.0


def test_preknowledge_residual_hand_computed():
    got = preknowledge_from_theta(
        np.array([[1.0, 1.0, 0.0]]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([True, True, False]),
        np.array([0.0]),
    )
    assert abs(float(got[0]) - math.sqrt(2.0)) < 1e-12


def test_rt_residual_hand_computed():
    times = np.array([[math.e**2, math.e**2]])
    beta = np.array([1.0, 1.0])
    assert abs(float(rt_residual(times, beta)[0]) - 1.0) < 1e-12
    fitted = fit_time_intensity(np.full((4, 3), math.e))
    assert np.allclose(fitted, np.ones(3))
