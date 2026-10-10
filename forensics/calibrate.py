"""Marginal maximum likelihood calibration of the 2PL with a normal prior."""

from __future__ import annotations

import numpy as np

from indices.mathutil import log_one_minus_sigmoid, log_sigmoid, sigmoid


def normal_quadrature(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and log weights for a standard normal prior."""
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    theta = nodes * np.sqrt(2.0)
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)
    return theta.astype(np.float64), log_weights.astype(np.float64)


def e_step(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    theta_q: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    """Posterior mass of each person on each quadrature node. Rows sum to 1."""
    responses = np.asarray(X, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    logits = a[None, :] * (theta_q[:, None] - b[None, :])
    log_p = log_sigmoid(logits)
    log_q = log_one_minus_sigmoid(logits)
    delta = (log_p - log_q).T
    loglik = responses @ delta + log_q.sum(axis=1)[None, :]
    log_post = loglik + log_weights[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    return post


def loglik_gradients(
    X: np.ndarray,
    post: np.ndarray,
    theta_q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of the expected complete-data log-likelihood for each item.

    `post` is person-by-node posterior mass. Returns (dL/da, dL/db).
    """
    responses = np.asarray(X, dtype=np.float64)
    post = np.asarray(post, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = theta_q[:, None] - b[None, :]
    logits = np.clip(a[None, :] * u, -30.0, 30.0)
    prob = sigmoid(logits)
    row_mass = post.sum(axis=1)
    node_mass = post.sum(axis=0)
    score = responses.T @ row_mass - prob.T @ node_mass
    weighted_u = post @ u
    grad_a = np.sum(responses * weighted_u, axis=0) - (prob * u).T @ node_mass
    grad_b = -a * score
    return grad_a, grad_b


def _m_step(
    X: np.ndarray,
    post: np.ndarray,
    theta_q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_newton: int,
) -> tuple[np.ndarray, np.ndarray]:
    responses = np.asarray(X, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    for _ in range(n_newton):
        u = theta_q[:, None] - b[None, :]
        logits = np.clip(a[None, :] * u, -30.0, 30.0)
        prob = sigmoid(logits)
        row_mass = post.sum(axis=1)
        node_mass = post.sum(axis=0)
        score = responses.T @ row_mass - prob.T @ node_mass
        weighted_u = post @ u
        grad_a = np.sum(responses * weighted_u, axis=0) - (prob * u).T @ node_mass
        grad_b = -a * score
        pq = prob * (1.0 - prob)
        hess_aa = -((pq * (u**2)).T @ node_mass) - 1e-4
        hess_bb = -((a**2) * (pq.T @ node_mass)) - 1e-4
        hess_ab = -score + a * ((pq * u).T @ node_mass)
        det = hess_aa * hess_bb - hess_ab**2
        safe = np.abs(det) > 1e-10
        solved_a = np.zeros_like(a)
        solved_b = np.zeros_like(b)
        solved_a[safe] = (hess_bb[safe] * grad_a[safe] - hess_ab[safe] * grad_b[safe]) / det[safe]
        solved_b[safe] = (-hess_ab[safe] * grad_a[safe] + hess_aa[safe] * grad_b[safe]) / det[safe]
        # Newton step for a maximum is -H^{-1} g, and `solved` is H^{-1} g.
        step_a = np.clip(-solved_a, -0.5, 0.5)
        step_b = np.clip(-solved_b, -0.5, 0.5)
        a = np.clip(a + step_a, 0.30, 3.0)
        b = np.clip(b + step_b, -4.5, 4.5)
    return a, b


def calibrate_2pl(
    X: np.ndarray,
    n_quad: int = 21,
    n_iter: int = 15,
    n_newton: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 2PL item parameters by marginal MLE. Returns (a, b)."""
    responses = np.asarray(X, dtype=np.float64)
    n_items = responses.shape[1]
    theta_q, log_weights = normal_quadrature(n_quad)
    p_hat = np.clip(responses.mean(axis=0), 0.02, 0.98)
    b = -np.log(p_hat / (1.0 - p_hat))
    a = np.ones(n_items, dtype=np.float64)
    for _ in range(n_iter):
        post = e_step(responses, a, b, theta_q, log_weights)
        a, b = _m_step(responses, post, theta_q, a, b, n_newton)
    return a, b
