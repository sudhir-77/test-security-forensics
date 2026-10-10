"""Locked simulation and model settings.

Calibration candidates and evaluation candidates are drawn from different
Generator seeds. Item parameters use a third seed so the form structure does
not depend on either cohort.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Design:
    name: str
    n_forms: int
    n_items: int
    n_train: int
    n_val: int
    n_test: int
    n_copy: int
    n_preknowledge: int
    n_rapid: int
    n_leaked: int
    n_sites: int
    n_weeks: int
    ref_k: int
    n_quad: int
    em_iter: int
    n_newton: int
    n_estimators: int
    copy_rate: float
    preknowledge_prob: float
    seed_items: int
    seed_calibration: int
    seed_evaluation: int
    seed_model: int
    seed_references: int

    @property
    def n_per_form(self) -> int:
        return self.n_train + self.n_val + self.n_test

    @property
    def n_candidates(self) -> int:
        return self.n_forms * self.n_per_form

    @property
    def n_misconduct(self) -> int:
        return self.n_forms * (self.n_copy + self.n_preknowledge + self.n_rapid)

    @property
    def seeds(self) -> list[int]:
        return [
            self.seed_items,
            self.seed_calibration,
            self.seed_evaluation,
            self.seed_model,
            self.seed_references,
        ]


# Shared measurement model. Documented again in simulate/PARAMETERS.md.
A_LOG_MEAN = 0.0
A_LOG_SD = 0.25
A_MIN = 0.5
A_MAX = 2.5
B_SD = 1.0
B_MIN = -2.5
B_MAX = 2.5
BETA_MEAN = 4.007333185232471  # log(55 seconds)
BETA_SD = 0.30
TAU_SD = 0.25
RT_ERROR_SD = 0.35
RT_FLOOR_SECONDS = 0.5
KEY_THETA = 0.25
RAPID_P_CORRECT = 0.5
RAPID_RT_LOW = 1.5
RAPID_RT_HIGH = 5.0
PREKNOWLEDGE_RT_FACTOR = 0.50

_COMMON = dict(
    copy_rate=0.95,
    preknowledge_prob=0.995,
    seed_items=20260111,
    seed_calibration=20260315,
    seed_evaluation=20260901,
    seed_model=20260501,
    seed_references=20260707,
)

FULL = Design(
    name="full",
    n_forms=4,
    n_items=150,
    n_train=1500,
    n_val=1500,
    n_test=2000,
    n_copy=50,
    n_preknowledge=50,
    n_rapid=50,
    n_leaked=50,
    n_sites=8,
    n_weeks=8,
    ref_k=400,
    n_quad=21,
    em_iter=15,
    n_newton=6,
    n_estimators=200,
    **_COMMON,
)

QUICK = Design(
    name="quick",
    n_forms=2,
    n_items=20,
    n_train=60,
    n_val=60,
    n_test=80,
    n_copy=6,
    n_preknowledge=6,
    n_rapid=6,
    n_leaked=6,
    n_sites=4,
    n_weeks=4,
    ref_k=20,
    n_quad=11,
    em_iter=6,
    n_newton=4,
    n_estimators=40,
    **_COMMON,
)
