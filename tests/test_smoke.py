"""CLI smoke test and an in-process quick run used for coverage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from forensics.pipeline import SCORE_COLUMNS, run_pipeline
from simulate.design import QUICK

ROOT = Path(__file__).resolve().parents[1]


def test_quick_pipeline_writes_evaluation_scores_and_keeps_splits_apart(tmp_path: Path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    payload = run_pipeline(QUICK, data_dir, results_dir)
    scores = pd.read_csv(results_dir / "candidate_scores.csv")
    assert list(scores.columns) == SCORE_COLUMNS
    assert set(scores["true_label"]) == {"clean", "copy", "preknowledge", "rapid_guess"}
    assert scores["candidate_id"].is_unique
    splits = pd.read_csv(data_dir / "splits.csv")
    train_ids = set(splits.loc[splits["split"] == "train", "candidate_id"])
    val_ids = set(splits.loc[splits["split"] == "val", "candidate_id"])
    test_ids = set(splits.loc[splits["split"] == "test", "candidate_id"])
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert set(scores["candidate_id"]) == test_ids
    assert set(splits.loc[splits["split"] == "train", "true_label"]) == {"clean"}
    assert set(splits.loc[splits["split"] == "val", "true_label"]) == {"clean"}
    audit = json.loads((data_dir / "run_audit.json").read_text(encoding="utf-8"))
    assert audit["item_calibration_on"] == "train"
    assert audit["forest_fit_on"] == "train"
    assert audit["threshold_fit_on"] == "validation"
    assert audit["evaluation_on"] == "test"
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["response_rows"] == QUICK.n_candidates * QUICK.n_items
    assert manifest["incomplete_candidates"] == 0
    for key in (
        "power_copy_fpr01",
        "power_preknowledge_fpr01",
        "power_rapidguess_fpr01",
        "observed_fpr",
        "auc_combined",
        "auc_baseline",
    ):
        assert key in payload["metrics"]
        assert key in payload
    assert (results_dir / "roc_curves.png").exists()
    assert (results_dir / "power_bars.png").exists()
    assert (results_dir / "dashboard.html").exists()
    assert (results_dir / "dashboard_flags.png").exists()
    assert (results_dir / "kpi_by_form.csv").exists()
    assert payload["splits"]["exam"]["test"] == QUICK.n_forms * QUICK.n_test
    assert payload["seeds"][1] != payload["seeds"][2]


def test_cli_quick_smoke(tmp_path: Path):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    env["FORENSICS_SKIP_PYTEST"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--quick",
            "--skip-pytest",
            "--data-dir",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
        ],
        cwd=str(ROOT),
        env=env,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0
    payload = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["project"] == "test-security-forensics"
    assert isinstance(payload["success_criteria"], list)
    assert payload["splits"]["exam"]["train"] == QUICK.n_forms * QUICK.n_train
    assert "peak_rss_gb" in payload["hardware"]
    scores = pd.read_csv(tmp_path / "results" / "candidate_scores.csv")
    assert "candidate_id" in scores.columns
    assert len(scores) == QUICK.n_forms * QUICK.n_test
