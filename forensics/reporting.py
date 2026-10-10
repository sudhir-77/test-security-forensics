"""Plots, metrics JSON, detection report, resume notes, and README table."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forensics.metrics_eval import criterion_passed

RESULTS_START = "<!-- RESULTS_TABLE_START -->"
RESULTS_END = "<!-- RESULTS_TABLE_END -->"

SPEC_METRIC_KEYS = (
    "power_copy_fpr01",
    "power_preknowledge_fpr01",
    "power_rapidguess_fpr01",
    "observed_fpr",
    "auc_combined",
    "auc_baseline",
)


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n", encoding="utf-8")


def save_roc_plot(path: Path, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.axvline(0.01, color="black", linestyle="--", linewidth=0.8, label="1% FPR")
    ax.axvline(0.001, color="gray", linestyle=":", linewidth=0.8, label="0.1% FPR")
    ax.set_xlabel("False-positive rate (versus clean candidates)")
    ax.set_ylabel("Detection power (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC by misconduct type")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_power_bars(path: Path, metrics: dict) -> None:
    labels = ["Answer copying", "Preknowledge", "Rapid guessing"]
    at_01 = [
        metrics["power_copy_fpr01"],
        metrics["power_preknowledge_fpr01"],
        metrics["power_rapidguess_fpr01"],
    ]
    at_001 = [
        metrics["power_copy_fpr001"],
        metrics["power_preknowledge_fpr001"],
        metrics["power_rapidguess_fpr001"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.18, at_01, width=0.36, label="1% FPR", color="#1f4e79")
    ax.bar(x + 0.18, at_001, width=0.36, label="0.1% FPR", color="#7aa0c4")
    ax.axhline(0.80, color="#1f4e79", linestyle="--", linewidth=0.7)
    ax.axhline(0.60, color="#7aa0c4", linestyle=":", linewidth=0.7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Detection power at the validation thresholds")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_score_hist(path: Path, labels: np.ndarray, scores: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    order = ["clean", "copy", "preknowledge", "rapid_guess"]
    for name in order:
        mask = labels == name
        if int(mask.sum()) == 0:
            continue
        ax.hist(scores[mask], bins=40, density=True, histtype="step", linewidth=1.4, label=name)
    ax.set_xlabel("Combined score")
    ax.set_ylabel("Density")
    ax.set_title("Combined score by true label (evaluation cohort)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def resume_figures(metrics: dict) -> dict:
    return {
        "copy_recall_percent": round(100.0 * float(metrics["power_copy_fpr01"]), 1),
        "preknowledge_recall_percent": round(100.0 * float(metrics["power_preknowledge_fpr01"]), 1),
        "rapid_guess_recall_percent": round(100.0 * float(metrics["power_rapidguess_fpr01"]), 1),
        "auc_combined": round(float(metrics["auc_combined"]), 3),
        "auc_baseline": round(float(metrics["auc_baseline"]), 3),
        "n_evaluation": int(metrics["n_evaluation"]),
        "observed_fpr_percent": round(100.0 * float(metrics["observed_fpr"]), 2),
    }


def render_resume(payload: dict) -> str:
    figs = payload["metrics"]["resume_figures"]
    copy_pct = figs["copy_recall_percent"]
    pre_pct = figs["preknowledge_recall_percent"]
    rapid_pct = figs["rapid_guess_recall_percent"]
    auc = figs["auc_combined"]
    base = figs["auc_baseline"]
    n_eval = figs["n_evaluation"]
    fpr_pct = figs["observed_fpr_percent"]
    return f"""# Resume notes

## Draft bullets

- Built exam-forensics indices (person-fit, answer similarity, preknowledge, response-time anomalies) and detected simulated answer copying at {copy_pct}% recall at a 1% false-positive rate.
- Combined psychometric indices with an isolation forest, raising AUC from {base} (score-only) to {auc} on {n_eval} simulated candidates.
- Wrote SQL KPI queries and a dashboard tracking flag rates by form, site, and week, with an observed false-positive rate of {fpr_pct}% on clean simulated candidates.
- Flagged simulated item preknowledge at {pre_pct}% recall and rapid guessing at {rapid_pct}% recall at the same 1% false-positive threshold.

## Skills and tools

Python, NumPy, SciPy, pandas, scikit-learn IsolationForest, SQLite, matplotlib, pytest, 2PL IRT, lz* person-fit, Wollack omega.

## Project summary

This project simulates a four-form certification exam with a two-parameter logistic model and lognormal response times, then injects answer copying, item preknowledge, and rapid guessing. It estimates item parameters on a clean cohort, scores person-fit, answer similarity, leaked-item residuals, and response-time residuals, and combines those indices with an isolation forest to flag candidates at a 1% validation threshold.

## GitHub description

Simulated certification-exam forensics using person-fit, answer similarity, preknowledge, and response-time indices.
"""


def render_detection_report(payload: dict) -> str:
    m = payload["metrics"]
    lines = [
        "# Detection report",
        "",
        "Thresholds are the 99th and 99.9th percentiles of the combined score on the clean validation slice. Power is recall on the evaluation cohort. The false-positive rate is the flag rate on clean evaluation candidates.",
        "",
        "| Misconduct | Recall at 1% FPR | Recall at 0.1% FPR |",
        "|---|---:|---:|",
        f"| Answer copying | {m['power_copy_fpr01']:.6f} | {m['power_copy_fpr001']:.6f} |",
        f"| Item preknowledge | {m['power_preknowledge_fpr01']:.6f} | {m['power_preknowledge_fpr001']:.6f} |",
        f"| Rapid guessing | {m['power_rapidguess_fpr01']:.6f} | {m['power_rapidguess_fpr001']:.6f} |",
        "",
        f"Observed false-positive rate at the 1% threshold: {m['observed_fpr']:.6f}",
        f"Combined AUC: {m['auc_combined']:.6f}",
        f"Score-only baseline AUC: {m['auc_baseline']:.6f}",
        f"AUC lift: {m['auc_lift']:.6f}",
        "",
        "Plots: `roc_curves.png`, `power_bars.png`, `score_by_label.png`.",
        "",
    ]
    return "\n".join(lines)


def render_results_table(payload: dict) -> str:
    header = [
        "| Criterion | Value | Rule | Passed |",
        "|---|---:|---|---|",
    ]
    for row in payload["success_criteria"]:
        high = row["threshold_high"]
        if row["op"] == "between" and high is not None:
            rule = f"{row['threshold']} to {high}"
        else:
            rule = f"{row['op']} {row['threshold']}"
        passed = "yes" if row["passed"] else "no"
        header.append(f"| {row['name']} | {row['value']:.6f} | {rule} | {passed} |")
    header.extend(
        [
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key in SPEC_METRIC_KEYS:
        header.append(f"| {key} | {payload['metrics'][key]:.6f} |")
    header.append(f"| runtime_seconds | {payload['runtime_seconds']:.3f} |")
    header.append(f"| peak_rss_gb | {payload['hardware']['peak_rss_gb']:.4f} |")
    return "\n".join(header)


def update_readme(readme_path: Path, payload: dict) -> None:
    if not readme_path.exists():
        return
    text = readme_path.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        return
    table = render_results_table(payload)
    before, rest = text.split(RESULTS_START, 1)
    _, after = rest.split(RESULTS_END, 1)
    updated = f"{before}{RESULTS_START}\n{table}\n{RESULTS_END}{after}"
    readme_path.write_text(updated, encoding="utf-8")


def build_success_criteria(metrics: dict, coverage: float | None) -> list[dict]:
    rows = [
        {
            "name": "Detection power (recall) for answer copying at 1% false-positive rate",
            "value": float(metrics["power_copy_fpr01"]),
            "op": ">=",
            "threshold": 0.80,
            "threshold_high": None,
        },
        {
            "name": "Detection power for item preknowledge at 1% FPR",
            "value": float(metrics["power_preknowledge_fpr01"]),
            "op": ">=",
            "threshold": 0.60,
            "threshold_high": None,
        },
        {
            "name": "False-positive control (observed FPR within 0.5 percentage points of 1%)",
            "value": float(metrics["observed_fpr"]),
            "op": "between",
            "threshold": 0.005,
            "threshold_high": 0.015,
        },
        {
            "name": "combined model AUC at least 0.10 higher than a raw-score z-score baseline",
            "value": float(metrics["auc_lift"]),
            "op": ">=",
            "threshold": 0.10,
            "threshold_high": None,
        },
    ]
    if coverage is not None:
        rows.append(
            {
                "name": "Test coverage",
                "value": float(coverage),
                "op": ">=",
                "threshold": 0.75,
                "threshold_high": None,
            }
        )
    for row in rows:
        row["passed"] = bool(
            criterion_passed(row["value"], row["op"], row["threshold"], row["threshold_high"])
        )
    return rows
