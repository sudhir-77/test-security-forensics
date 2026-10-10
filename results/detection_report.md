# Detection report

Thresholds are the 99th and 99.9th percentiles of the combined score on the clean validation slice. Power is recall on the evaluation cohort. The false-positive rate is the flag rate on clean evaluation candidates.

| Misconduct | Recall at 1% FPR | Recall at 0.1% FPR |
|---|---:|---:|
| Answer copying | 1.000000 | 1.000000 |
| Item preknowledge | 0.955000 | 0.905000 |
| Rapid guessing | 1.000000 | 1.000000 |

Observed false-positive rate at the 1% threshold: 0.011081
Combined AUC: 0.999452
Score-only baseline AUC: 0.624383
AUC lift: 0.375070

Plots: `roc_curves.png`, `power_bars.png`, `score_by_label.png`.
