"""Reproduce the exam-forensics study.

Thread caps are set before NumPy is imported. The evaluation cohort is not used
to calibrate items, fit the forest, or choose the false-positive threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import psutil

ROOT = Path(__file__).resolve().parent


class PeakRSS:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._sample()

    def _loop(self) -> None:
        while not self._stop.wait(0.25):
            self._sample()

    def _sample(self) -> None:
        try:
            proc = psutil.Process()
            processes = [proc, *proc.children(recursive=True)]
        except psutil.Error:
            return
        for process in processes:
            try:
                info = process.memory_info()
                self.peak = max(self.peak, int(info.rss), int(getattr(info, "peak_wset", 0)))
            except psutil.Error:
                continue

    @property
    def gb(self) -> float:
        return self.peak / (1024**3)


def measure_coverage(results_dir: Path) -> float | None:
    cov_path = results_dir / "coverage.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=simulate",
        "--cov=indices",
        "--cov=forensics",
        f"--cov-report=json:{cov_path}",
        "--cov-report=term-missing",
        "-q",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    env["FORENSICS_SKIP_PYTEST"] = "1"
    completed = subprocess.run(command, cwd=str(ROOT), env=env)
    print(f"pytest exit code: {completed.returncode}", flush=True)
    if not cov_path.exists():
        return None
    payload = json.loads(cov_path.read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"]) / 100.0


def hardware_snapshot(peak_gb: float) -> dict:
    cpu = platform.processor() or platform.machine()
    return {
        "cpu": cpu,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": None,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": round(peak_gb, 4),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate an exam and score test-security indices.")
    parser.add_argument("--quick", action="store_true", help="Run the same pipeline on a small exam.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip the coverage run at the end.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda":
        print("CUDA was requested. This project scores the exam on CPU.", flush=True)
    started = time.perf_counter()
    sampler = PeakRSS()
    sampler.start()
    from forensics.pipeline import run_pipeline
    from forensics.reporting import (
        SPEC_METRIC_KEYS,
        build_success_criteria,
        render_detection_report,
        render_resume,
        resume_figures,
        update_readme,
        write_json,
    )
    from simulate.design import FULL, QUICK

    design = QUICK if args.quick else FULL
    payload = run_pipeline(design, args.data_dir, args.results_dir)
    coverage = None
    if not args.skip_pytest and os.environ.get("FORENSICS_SKIP_PYTEST") != "1":
        print("Measuring test coverage...", flush=True)
        coverage = measure_coverage(args.results_dir)
        if coverage is None:
            coverage = 0.0
    sampler.stop()
    metrics_path = args.results_dir / "metrics.json"
    pipeline_runtime = float(payload["runtime_seconds"])
    payload["hardware"] = hardware_snapshot(sampler.gb)
    payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
    payload["metrics"]["requested_device"] = args.device
    payload["metrics"]["pipeline_runtime_seconds"] = pipeline_runtime
    if coverage is not None:
        payload["metrics"]["test_coverage"] = float(coverage)
    payload["metrics"]["resume_figures"] = resume_figures(payload["metrics"])
    payload["success_criteria"] = build_success_criteria(payload["metrics"], coverage)
    for key in SPEC_METRIC_KEYS:
        payload[key] = payload["metrics"][key]
    write_json(metrics_path, payload)
    (args.results_dir / "detection_report.md").write_text(render_detection_report(payload), encoding="utf-8")
    default_dirs = args.data_dir.resolve() == (ROOT / "data").resolve() and args.results_dir.resolve() == (
        ROOT / "results"
    ).resolve()
    if default_dirs and not args.quick:
        (ROOT / "RESUME_NOTES.md").write_text(render_resume(payload), encoding="utf-8")
        update_readme(ROOT / "README.md", payload)
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Runtime seconds: {payload['runtime_seconds']}", flush=True)
    print(f"Peak RSS GiB: {payload['hardware']['peak_rss_gb']}", flush=True)
    for row in payload["success_criteria"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status} {row['name']}: {row['value']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
