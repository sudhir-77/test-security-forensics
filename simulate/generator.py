"""2PL response simulator with three injected misconduct types.

The calibration cohort (train and validation) is entirely clean and is drawn
from `seed_calibration`. The evaluation cohort is drawn from `seed_evaluation`.
Misconduct is injected only into the evaluation cohort. A seized answer key is
stored per form and is not enrolled as a candidate, so the person who produced
the key is not scored and cannot become a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from indices.mathutil import sigmoid
from simulate.design import (
    A_LOG_MEAN,
    A_LOG_SD,
    A_MAX,
    A_MIN,
    B_MAX,
    B_MIN,
    B_SD,
    BETA_MEAN,
    BETA_SD,
    KEY_THETA,
    PREKNOWLEDGE_RT_FACTOR,
    RAPID_P_CORRECT,
    RAPID_RT_HIGH,
    RAPID_RT_LOW,
    RT_ERROR_SD,
    RT_FLOOR_SECONDS,
    TAU_SD,
    Design,
)

FORM_NAMES = ("A", "B", "C", "D", "E", "F")


@dataclass
class CohortBlock:
    candidate_id: np.ndarray
    site: np.ndarray
    week: np.ndarray
    label: np.ndarray
    theta: np.ndarray
    tau: np.ndarray
    X: np.ndarray
    RT: np.ndarray


@dataclass
class FormSim:
    form_id: str
    a_true: np.ndarray
    b_true: np.ndarray
    beta_true: np.ndarray
    leaked_mask: np.ndarray
    key: np.ndarray
    train: CohortBlock
    val: CohortBlock
    test: CohortBlock


@dataclass
class SimulatedExam:
    design_name: str
    forms: list[FormSim]
    seeds: list[int]

    def iter_blocks(self):
        for form in self.forms:
            yield form, "train", form.train
            yield form, "val", form.val
            yield form, "test", form.test


def _draw_clean(
    rng: np.random.Generator,
    n: int,
    a: np.ndarray,
    b: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = rng.normal(0.0, 1.0, size=n)
    tau = rng.normal(0.0, TAU_SD, size=n)
    logits = a[None, :] * (theta[:, None] - b[None, :])
    prob = sigmoid(logits)
    X = (rng.random(prob.shape) < prob).astype(np.uint8)
    log_rt = beta[None, :] - tau[:, None] + rng.normal(0.0, RT_ERROR_SD, size=prob.shape)
    RT = np.clip(np.exp(log_rt), RT_FLOOR_SECONDS, None).astype(np.float32)
    return theta, tau, X, RT


def _enroll(
    rng: np.random.Generator,
    start_id: int,
    n: int,
    n_sites: int,
    n_weeks: int,
    a: np.ndarray,
    b: np.ndarray,
    beta: np.ndarray,
    label: str = "clean",
) -> tuple[CohortBlock, int]:
    theta, tau, X, RT = _draw_clean(rng, n, a, b, beta)
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    site_ix = rng.integers(1, n_sites + 1, size=n)
    sites = np.array([f"S{int(i):02d}" for i in site_ix], dtype=object)
    weeks = rng.integers(1, n_weeks + 1, size=n).astype(np.int32)
    labels = np.full(n, label, dtype=object)
    block = CohortBlock(
        candidate_id=ids,
        site=sites,
        week=weeks,
        label=labels,
        theta=theta.astype(np.float64),
        tau=tau.astype(np.float64),
        X=X,
        RT=RT,
    )
    return block, start_id + n


def _inject_copy(
    rng: np.random.Generator,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    key: np.ndarray,
    copy_rate: float,
) -> np.ndarray:
    logits = a[None, :] * (theta[:, None] - b[None, :])
    own = (rng.random((theta.shape[0], a.shape[0])) < sigmoid(logits)).astype(np.uint8)
    take = rng.random(own.shape) < copy_rate
    copied = np.where(take, key.astype(np.uint8)[None, :], own)
    return copied.astype(np.uint8)


def _inject_preknowledge(
    rng: np.random.Generator,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    RT: np.ndarray,
    leaked: np.ndarray,
    prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    logits = a[None, :] * (theta[:, None] - b[None, :])
    p = sigmoid(logits)
    p = p.copy()
    p[:, leaked] = np.maximum(p[:, leaked], prob)
    X = (rng.random(p.shape) < p).astype(np.uint8)
    RT2 = RT.copy()
    RT2[:, leaked] = np.clip(RT2[:, leaked] * np.float32(PREKNOWLEDGE_RT_FACTOR), RT_FLOOR_SECONDS, None)
    return X, RT2.astype(np.float32)


def _inject_rapid(rng: np.random.Generator, n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    X = (rng.random((n, m)) < RAPID_P_CORRECT).astype(np.uint8)
    RT = rng.uniform(RAPID_RT_LOW, RAPID_RT_HIGH, size=(n, m)).astype(np.float32)
    return X, RT


def _apply_misconduct(
    rng: np.random.Generator,
    block: CohortBlock,
    a: np.ndarray,
    b: np.ndarray,
    leaked: np.ndarray,
    key: np.ndarray,
    design: Design,
) -> None:
    n = block.X.shape[0]
    order = rng.permutation(n)
    n_copy = design.n_copy
    n_pre = design.n_preknowledge
    n_rapid = design.n_rapid
    if n_copy + n_pre + n_rapid > n:
        raise ValueError("Misconduct counts exceed the evaluation cohort.")
    copy_ix = order[:n_copy]
    pre_ix = order[n_copy : n_copy + n_pre]
    rapid_ix = order[n_copy + n_pre : n_copy + n_pre + n_rapid]
    if n_copy:
        block.X[copy_ix] = _inject_copy(rng, block.theta[copy_ix], a, b, key, design.copy_rate)
        block.label[copy_ix] = "copy"
    if n_pre:
        X_pre, RT_pre = _inject_preknowledge(
            rng,
            block.theta[pre_ix],
            a,
            b,
            block.RT[pre_ix],
            leaked,
            design.preknowledge_prob,
        )
        block.X[pre_ix] = X_pre
        block.RT[pre_ix] = RT_pre
        block.label[pre_ix] = "preknowledge"
    if n_rapid:
        X_r, RT_r = _inject_rapid(rng, n_rapid, a.shape[0])
        block.X[rapid_ix] = X_r
        block.RT[rapid_ix] = RT_r
        block.label[rapid_ix] = "rapid_guess"


def _draw_form_items(rng: np.random.Generator, design: Design) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = design.n_items
    a = np.clip(rng.lognormal(A_LOG_MEAN, A_LOG_SD, size=m), A_MIN, A_MAX)
    b = np.clip(rng.normal(0.0, B_SD, size=m), B_MIN, B_MAX)
    beta = rng.normal(BETA_MEAN, BETA_SD, size=m)
    leaked = np.zeros(m, dtype=bool)
    leaked_ix = rng.choice(m, size=design.n_leaked, replace=False)
    leaked[leaked_ix] = True
    key_p = sigmoid(a * (KEY_THETA - b))
    key = (rng.random(m) < key_p).astype(np.uint8)
    return a, b, beta, leaked, key


def simulate(design: Design) -> SimulatedExam:
    if design.n_forms > len(FORM_NAMES):
        raise ValueError("n_forms exceeds the form-name table.")
    item_rng = np.random.default_rng(design.seed_items)
    cal_rng = np.random.default_rng(design.seed_calibration)
    eval_rng = np.random.default_rng(design.seed_evaluation)
    next_id = 1
    forms: list[FormSim] = []
    for f in range(design.n_forms):
        a, b, beta, leaked, key = _draw_form_items(item_rng, design)
        train, next_id = _enroll(
            cal_rng, next_id, design.n_train, design.n_sites, design.n_weeks, a, b, beta
        )
        val, next_id = _enroll(
            cal_rng, next_id, design.n_val, design.n_sites, design.n_weeks, a, b, beta
        )
        test, next_id = _enroll(
            eval_rng, next_id, design.n_test, design.n_sites, design.n_weeks, a, b, beta
        )
        _apply_misconduct(eval_rng, test, a, b, leaked, key, design)
        forms.append(
            FormSim(
                form_id=FORM_NAMES[f],
                a_true=a.astype(np.float64),
                b_true=b.astype(np.float64),
                beta_true=beta.astype(np.float64),
                leaked_mask=leaked,
                key=key,
                train=train,
                val=val,
                test=test,
            )
        )
    return SimulatedExam(design_name=design.name, forms=forms, seeds=design.seeds)
