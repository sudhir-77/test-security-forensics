# Resume notes

## Draft bullets

- Built exam-forensics indices (person-fit, answer similarity, preknowledge, response-time anomalies) and detected simulated answer copying at 100.0% recall at a 1% false-positive rate.
- Combined psychometric indices with an isolation forest, raising AUC from 0.624 (score-only) to 0.999 on 8000 simulated candidates.
- Wrote SQL KPI queries and a dashboard tracking flag rates by form, site, and week, with an observed false-positive rate of 1.11% on clean simulated candidates.
- Flagged simulated item preknowledge at 95.5% recall and rapid guessing at 100.0% recall at the same 1% false-positive threshold.

## Skills and tools

Python, NumPy, SciPy, pandas, scikit-learn IsolationForest, SQLite, matplotlib, pytest, 2PL IRT, lz* person-fit, Wollack omega.

## Project summary

This project simulates a four-form certification exam with a two-parameter logistic model and lognormal response times, then injects answer copying, item preknowledge, and rapid guessing. It estimates item parameters on a clean cohort, scores person-fit, answer similarity, leaked-item residuals, and response-time residuals, and combines those indices with an isolation forest to flag candidates at a 1% validation threshold.

## GitHub description

Simulated certification-exam forensics using person-fit, answer similarity, preknowledge, and response-time indices.
