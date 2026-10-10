"""End-to-end exam forensics pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forensics.calibrate import calibrate_2pl
from forensics.dashboard import save_flag_dashboard
from forensics.detect import fit_detector, score_features
from forensics.features import feature_matrix, train_time_intensity
from forensics.kpi import flag_rate_tables
from forensics.metrics_eval import (
    observed_fpr,
    recall_for_label,
    roc_auc,
    roc_curve_points,
    within_form_z,
)
from forensics.reporting import (
    SPEC_METRIC_KEYS,
    build_success_criteria,
    render_detection_report,
    resume_figures,
    save_power_bars,
    save_roc_plot,
    save_score_hist,
    write_json,
)
from forensics.storage import (
    connect,
    expected_response_rows,
    insert_exam,
    insert_flags,
    response_checksum,
    update_calibration,
    verify_store,
)
from simulate.design import Design
from simulate.generator import CohortBlock, FormSim, simulate

SPLIT_CODE = {"train": 1, "val": 2, "test": 3}
SCORE_COLUMNS = [
    "candidate_id",
    "form",
    "lz_star",
    "similarity_max",
    "preknowledge_diff",
    "rt_residual",
    "iforest_score",
    "combined_score",
    "baseline_score",
    "flagged",
    "true_label",
]


def _hash_ids(ids: np.ndarray) -> str:
    ordered = np.ascontiguousarray(np.sort(np.asarray(ids, dtype=np.int64)))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _stack(parts: list[dict], key: str) -> np.ndarray:
    return np.concatenate([part[key] for part in parts])


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _assert_cohorts(exam, design: Design) -> None:
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    test_ids: set[int] = set()
    n_copy = n_pre = n_rapid = 0
    for form in exam.forms:
        for split, block, bucket in (
            ("train", form.train, train_ids),
            ("val", form.val, val_ids),
            ("test", form.test, test_ids),
        ):
            ids = [int(v) for v in block.candidate_id]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Duplicate ids inside {form.form_id} {split}.")
            bucket.update(ids)
            labels = set(block.label.tolist())
            if split in {"train", "val"} and labels != {"clean"}:
                raise RuntimeError(f"{split} cohort is not clean.")
        n_copy += int(np.sum(form.test.label == "copy"))
        n_pre += int(np.sum(form.test.label == "preknowledge"))
        n_rapid += int(np.sum(form.test.label == "rapid_guess"))
    if not train_ids.isdisjoint(val_ids) or not train_ids.isdisjoint(test_ids) or not val_ids.isdisjoint(test_ids):
        raise RuntimeError("Train, validation, and test candidate ids overlap.")
    expected_copy = design.n_forms * design.n_copy
    expected_pre = design.n_forms * design.n_preknowledge
    expected_rapid = design.n_forms * design.n_rapid
    if (n_copy, n_pre, n_rapid) != (expected_copy, expected_pre, expected_rapid):
        raise RuntimeError("Misconduct counts do not match the design.")


def _write_splits(exam, path: Path) -> None:
    rows = []
    for form, split, block in exam.iter_blocks():
        for i in range(block.X.shape[0]):
            rows.append(
                {
                    "candidate_id": int(block.candidate_id[i]),
                    "form": form.form_id,
                    "split": split,
                    "true_label": str(block.label[i]),
                    "site": str(block.site[i]),
                    "week": int(block.week[i]),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _score_block(
    form: FormSim,
    block: CohortBlock,
    split: str,
    form_index: int,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    beta_hat: np.ndarray,
    design: Design,
) -> dict:
    rng = np.random.default_rng(design.seed_references + form_index * 100 + SPLIT_CODE[split])
    features, total = feature_matrix(
        block.X,
        block.RT,
        a_hat,
        b_hat,
        beta_hat,
        form.leaked_mask,
        form.key,
        design.ref_k,
        rng,
    )
    return {
        "candidate_id": block.candidate_id.astype(np.int64),
        "form": np.full(block.X.shape[0], form.form_id, dtype=object),
        "label": block.label.astype(object),
        "site": block.site.astype(object),
        "week": block.week.astype(np.int32),
        "features": features,
        "total": total,
    }


def run_pipeline(design: Design, data_dir: Path, results_dir: Path) -> dict:
    started = time.perf_counter()
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Simulating {design.name} exam ({design.n_candidates} candidates)...", flush=True)
    exam = simulate(design)
    _assert_cohorts(exam, design)
    checksum = response_checksum(exam)
    _write_splits(exam, data_dir / "splits.csv")
    print("Writing SQLite responses...", flush=True)
    conn = connect(data_dir / "exam.sqlite")
    try:
        insert_exam(conn, exam)
        integrity = verify_store(conn, exam, checksum)
        manifest = {
            "design": design.name,
            "seeds": design.seeds,
            "n_candidates": design.n_candidates,
            "response_rows": integrity["response_rows"],
            "expected_response_rows": expected_response_rows(exam),
            "checksum_sha256": integrity["checksum_sha256"],
            "incomplete_candidates": integrity["incomplete_candidates"],
            "spot_checked_candidate_ids": integrity["spot_checked_candidate_ids"],
            "note": "Synthetic exam. Checksum is SHA-256 of in-memory responses; row count and spot checks verify SQLite.",
        }
        (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        train_parts: list[dict] = []
        val_parts: list[dict] = []
        test_parts: list[dict] = []
        a_true_all: list[np.ndarray] = []
        a_hat_all: list[np.ndarray] = []
        b_true_all: list[np.ndarray] = []
        b_hat_all: list[np.ndarray] = []
        print("Calibrating items and computing indices...", flush=True)
        for form_index, form in enumerate(exam.forms):
            a_hat, b_hat = calibrate_2pl(
                form.train.X,
                n_quad=design.n_quad,
                n_iter=design.em_iter,
                n_newton=design.n_newton,
            )
            beta_hat = train_time_intensity(form.train.RT)
            update_calibration(conn, form.form_id, a_hat, b_hat, beta_hat)
            a_true_all.append(form.a_true)
            b_true_all.append(form.b_true)
            a_hat_all.append(a_hat)
            b_hat_all.append(b_hat)
            train_parts.append(_score_block(form, form.train, "train", form_index, a_hat, b_hat, beta_hat, design))
            val_parts.append(_score_block(form, form.val, "val", form_index, a_hat, b_hat, beta_hat, design))
            test_parts.append(_score_block(form, form.test, "test", form_index, a_hat, b_hat, beta_hat, design))
            print(f"  form {form.form_id} done", flush=True)

        train_x = _stack(train_parts, "features")
        val_x = _stack(val_parts, "features")
        test_x = _stack(test_parts, "features")
        print("Fitting isolation forest and validation threshold...", flush=True)
        detector = fit_detector(train_x, val_x, design.seed_model, design.n_estimators)
        val_scored = score_features(detector, val_x)
        test_scored = score_features(detector, test_x)
        train_total = _stack(train_parts, "total")
        train_form = _stack(train_parts, "form")
        test_total = _stack(test_parts, "total")
        test_form = _stack(test_parts, "form")
        baseline = within_form_z(test_total, test_form, train_total, train_form)
        labels = _stack(test_parts, "label")
        rounded = {
            "lz_star": np.round(test_x[:, 0], 10),
            "similarity_max": np.round(test_x[:, 1], 10),
            "preknowledge_diff": np.round(test_x[:, 2], 10),
            "rt_residual": np.round(test_x[:, 3], 10),
            "iforest_score": np.round(test_scored["iforest_score"], 10),
            "combined_score": np.round(test_scored["combined_score"], 10),
            "baseline_score": np.round(baseline, 10),
        }
        flagged = (rounded["combined_score"] > detector.threshold_fpr01).astype(np.int64)
        flagged_001 = (rounded["combined_score"] > detector.threshold_fpr001).astype(np.int64)
        scores = pd.DataFrame(
            {
                "candidate_id": _stack(test_parts, "candidate_id").astype(np.int64),
                "form": test_form.astype(str),
                "lz_star": rounded["lz_star"],
                "similarity_max": rounded["similarity_max"],
                "preknowledge_diff": rounded["preknowledge_diff"],
                "rt_residual": rounded["rt_residual"],
                "iforest_score": rounded["iforest_score"],
                "combined_score": rounded["combined_score"],
                "baseline_score": rounded["baseline_score"],
                "flagged": flagged,
                "true_label": labels.astype(str),
            }
        )
        scores = scores.sort_values("candidate_id").reset_index(drop=True)
        scores = scores[SCORE_COLUMNS]
        scores_path = results_dir / "candidate_scores.csv"
        scores.to_csv(scores_path, index=False, float_format="%.10f", lineterminator="\n")

        y_any = (scores["true_label"].to_numpy() != "clean").astype(np.int64)
        metrics = {
            "power_copy_fpr01": recall_for_label(scores["flagged"], scores["true_label"], "copy"),
            "power_preknowledge_fpr01": recall_for_label(scores["flagged"], scores["true_label"], "preknowledge"),
            "power_rapidguess_fpr01": recall_for_label(scores["flagged"], scores["true_label"], "rapid_guess"),
            "power_copy_fpr001": recall_for_label(flagged_001, labels, "copy"),
            "power_preknowledge_fpr001": recall_for_label(flagged_001, labels, "preknowledge"),
            "power_rapidguess_fpr001": recall_for_label(flagged_001, labels, "rapid_guess"),
            "observed_fpr": observed_fpr(scores["flagged"], scores["true_label"]),
            "auc_combined": roc_auc(y_any, scores["combined_score"].to_numpy()),
            "auc_baseline": roc_auc(y_any, scores["baseline_score"].to_numpy()),
            "threshold_fpr01": float(detector.threshold_fpr01),
            "threshold_fpr001": float(detector.threshold_fpr001),
            "calibration_val_fpr": float(np.mean(val_scored["flagged"] == 1)),
            "n_evaluation": int(len(scores)),
            "n_evaluation_clean": int(np.sum(scores["true_label"].to_numpy() == "clean")),
            "n_copy": int(np.sum(scores["true_label"].to_numpy() == "copy")),
            "n_preknowledge": int(np.sum(scores["true_label"].to_numpy() == "preknowledge")),
            "n_rapid_guess": int(np.sum(scores["true_label"].to_numpy() == "rapid_guess")),
            "misconduct_rate_full_sample": design.n_misconduct / design.n_candidates,
            "data_quality_incomplete_candidates": 0,
            "calibration_a_correlation": _corr(np.concatenate(a_true_all), np.concatenate(a_hat_all)),
            "calibration_b_correlation": _corr(np.concatenate(b_true_all), np.concatenate(b_hat_all)),
            "response_rows": integrity["response_rows"],
            "response_checksum_sha256": integrity["checksum_sha256"],
        }
        metrics["auc_lift"] = float(metrics["auc_combined"] - metrics["auc_baseline"])
        for label_name, key in (
            ("copy", "auc_copy_vs_clean"),
            ("preknowledge", "auc_preknowledge_vs_clean"),
            ("rapid_guess", "auc_rapidguess_vs_clean"),
        ):
            mask = scores["true_label"].isin(["clean", label_name]).to_numpy()
            y_type = (scores.loc[mask, "true_label"].to_numpy() == label_name).astype(np.int64)
            metrics[key] = roc_auc(y_type, scores.loc[mask, "combined_score"].to_numpy())
        baseline_flag = (scores["baseline_score"].to_numpy() > 2.0).astype(np.int64)
        baseline = {
            "auc_baseline": metrics["auc_baseline"],
            "power_copy_at_z_gt_2": recall_for_label(baseline_flag, scores["true_label"], "copy"),
            "power_preknowledge_at_z_gt_2": recall_for_label(baseline_flag, scores["true_label"], "preknowledge"),
            "power_rapidguess_at_z_gt_2": recall_for_label(baseline_flag, scores["true_label"], "rapid_guess"),
            "observed_fpr_at_z_gt_2": observed_fpr(baseline_flag, scores["true_label"]),
        }
        metrics["resume_figures"] = resume_figures(metrics)

        curves = {}
        roc_rows = []
        for label_name in ("copy", "preknowledge", "rapid_guess"):
            mask = scores["true_label"].isin(["clean", label_name]).to_numpy()
            y_type = (scores.loc[mask, "true_label"].to_numpy() == label_name).astype(np.int64)
            fpr, tpr = roc_curve_points(y_type, scores.loc[mask, "combined_score"].to_numpy())
            curves[label_name] = (fpr, tpr)
            for fpr_i, tpr_i in zip(fpr, tpr):
                roc_rows.append({"misconduct_type": label_name, "fpr": float(fpr_i), "tpr": float(tpr_i)})
        save_roc_plot(results_dir / "roc_curves.png", curves)
        save_power_bars(results_dir / "power_bars.png", metrics)
        save_score_hist(
            results_dir / "score_by_label.png",
            scores["true_label"].to_numpy(),
            scores["combined_score"].to_numpy(),
        )
        pd.DataFrame(roc_rows).to_csv(results_dir / "roc_curve.csv", index=False, lineterminator="\n")

        insert_flags(
            conn,
            scores["candidate_id"].to_numpy(),
            scores["flagged"].to_numpy(),
            scores["combined_score"].to_numpy(),
        )
        tables = flag_rate_tables(conn)
        for name, frame in tables.items():
            frame.to_csv(results_dir / f"kpi_{name}.csv", index=False, lineterminator="\n")
        save_flag_dashboard(tables, results_dir / "dashboard_flags.png", results_dir / "dashboard.html")

        train_ids = _stack(train_parts, "candidate_id")
        val_ids = _stack(val_parts, "candidate_id")
        test_ids = _stack(test_parts, "candidate_id")
        audit = {
            "item_calibration_on": "train",
            "forest_fit_on": "train",
            "threshold_fit_on": "validation",
            "evaluation_on": "test",
            "n_train": int(train_ids.shape[0]),
            "n_val": int(val_ids.shape[0]),
            "n_test": int(test_ids.shape[0]),
            "train_ids_sha256": _hash_ids(train_ids),
            "val_ids_sha256": _hash_ids(val_ids),
            "test_ids_sha256": _hash_ids(test_ids),
        }
        write_json(data_dir / "run_audit.json", audit)
    finally:
        conn.close()

    payload = {
        "project": "test-security-forensics",
        "success_criteria": build_success_criteria(metrics, coverage=None),
        "metrics": metrics,
        "baseline": baseline,
        "splits": {
            "exam": {
                "train": design.n_forms * design.n_train,
                "val": design.n_forms * design.n_val,
                "test": design.n_forms * design.n_test,
            }
        },
        "seeds": design.seeds,
        "hardware": {
            "cpu": "",
            "ram_gb": 0.0,
            "gpu": None,
            "python": "",
            "os": "",
            "peak_rss_gb": 0.0,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    for key in SPEC_METRIC_KEYS:
        payload[key] = metrics[key]
    (results_dir / "detection_report.md").write_text(render_detection_report(payload), encoding="utf-8")
    write_json(results_dir / "metrics.json", payload)
    print(
        "Evaluation powers "
        f"copy={metrics['power_copy_fpr01']:.3f} "
        f"preknowledge={metrics['power_preknowledge_fpr01']:.3f} "
        f"rapid={metrics['power_rapidguess_fpr01']:.3f} "
        f"fpr={metrics['observed_fpr']:.4f} "
        f"auc={metrics['auc_combined']:.3f} "
        f"baseline={metrics['auc_baseline']:.3f}",
        flush=True,
    )
    return payload
