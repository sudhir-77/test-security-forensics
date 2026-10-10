"""Split integrity: clean calibration, held-out evaluation, disjoint seeds."""

from __future__ import annotations

import numpy as np

from simulate.design import FULL, QUICK
from simulate.generator import simulate


def _ids(exam, split: str) -> set[int]:
    found = set()
    for _form, name, block in exam.iter_blocks():
        if name == split:
            found.update(int(v) for v in block.candidate_id)
    return found


def test_full_design_is_twenty_thousand_candidates_with_three_percent_misconduct():
    assert FULL.n_candidates == 20000
    assert FULL.n_items == 150
    assert FULL.n_forms == 4
    assert FULL.n_misconduct == 600
    assert abs(FULL.n_misconduct / FULL.n_candidates - 0.03) < 1e-12
    assert FULL.seed_calibration != FULL.seed_evaluation


def test_splits_are_disjoint_and_misconduct_is_evaluation_only():
    exam = simulate(QUICK)
    train_ids = _ids(exam, "train")
    val_ids = _ids(exam, "val")
    test_ids = _ids(exam, "test")
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) == QUICK.n_forms * QUICK.n_train
    assert len(val_ids) == QUICK.n_forms * QUICK.n_val
    assert len(test_ids) == QUICK.n_forms * QUICK.n_test
    labels = {split: set() for split in ("train", "val", "test")}
    for _form, split, block in exam.iter_blocks():
        labels[split].update(block.label.tolist())
    assert labels["train"] == {"clean"}
    assert labels["val"] == {"clean"}
    assert labels["test"] == {"clean", "copy", "preknowledge", "rapid_guess"}


def test_regeneration_is_identical_and_evaluation_seed_does_not_change_train():
    first = simulate(QUICK)
    second = simulate(QUICK)
    assert np.array_equal(first.forms[0].train.X, second.forms[0].train.X)
    assert np.array_equal(first.forms[0].test.X, second.forms[0].test.X)
    from dataclasses import replace

    shifted = replace(QUICK, seed_evaluation=QUICK.seed_evaluation + 17)
    other = simulate(shifted)
    assert np.array_equal(first.forms[0].train.X, other.forms[0].train.X)
    assert np.array_equal(first.forms[0].val.X, other.forms[0].val.X)
    assert not np.array_equal(first.forms[0].test.X, other.forms[0].test.X)
