# Simulation parameters

All exam data are synthetic. No candidate file is downloaded and no real examinee records are used. `python run.py` draws the exam from the seeds below and writes SQLite plus a SHA-256 manifest under `data/`.

## Cohorts and seeds

| Stream | Seed | What it draws |
|---|---|---|
| Items, leaked-item masks, seized answer keys | 20260111 | Shared form structure |
| Calibration cohort | 20260315 | Clean train and validation candidates |
| Evaluation cohort | 20260901 | Test candidates and injected misconduct |
| Isolation forest | 20260501 | Tree randomness only |
| Similarity reference samples | 20260707 | Which other candidates enter each omega max |

The calibration seed and the evaluation seed are disjoint. Train is the first `n_train` clean draws on each form. Validation is the next `n_val` clean draws. The test cohort is the evaluation draw. Splits are saved to `data/splits.csv` before item calibration, the forest, or the threshold are fit.

## Full run size

| Quantity | Value |
|---|---|
| Forms | 4 (A-D), each with its own 150 items |
| Train / validation / test per form | 1500 / 1500 / 2000 |
| Candidates | 20,000 |
| Responses | 20,000 x 150 = 3,000,000 |
| Misconduct in the test cohort | 50 copy + 50 preknowledge + 50 rapid guessing per form |
| Misconduct rate on the full sample | 600 / 20,000 = 3% |
| Sites | 8 (`S01`-`S08`) |
| Weeks | 8 |

Train and validation are clean. Misconduct labels exist only on the evaluation cohort and are not inputs to calibration, the isolation forest, or the threshold.

## Response model

Two-parameter logistic:

`P(correct) = 1 / (1 + exp(-a * (theta - b)))`

| Parameter | Distribution |
|---|---|
| theta (ability) | Normal(0, 1) |
| a (discrimination) | lognormal(0, 0.25), clipped to [0.5, 2.5] |
| b (difficulty) | Normal(0, 1), clipped to [-2.5, 2.5] |

## Response times

Lognormal model, in seconds:

`log(T) = beta_item - tau_person + Normal(0, 0.35)`

| Parameter | Value |
|---|---|
| beta_item | Normal(log(55), 0.30) |
| tau_person | Normal(0, 0.25) |
| Floor | 0.5 seconds |

The scored residual is the person's mean of `log(T) - beta_hat`, where `beta_hat` is the training-cohort mean log time. Person speed is left inside that residual. A speed parameter fit on the same responses would absorb a candidate who is fast on every item.

## Injected misconduct (evaluation cohort only)

Answer copying. Each form has one seized answer key, generated as a 2PL response at theta = 0.25. The key is a reference vector, not a candidate. Each copier copies each item from that key with probability 0.95 and otherwise answers from their own ability. Response times stay on the honest lognormal model.

Item preknowledge. Each form has 50 known leaked items (chosen with the item seed and treated as a breach the program has already identified). On leaked items the response probability is at least 0.995. Response times on those items are multiplied by 0.50. Secure items stay on the honest model. The index estimates ability from secure items only.

Rapid guessing. Every item is answered with probability 0.5, independently. Response times are uniform between 1.5 and 5.0 seconds.

## Quick run

`python run.py --quick` uses 2 forms, 20 items, 60/60/80 candidates per form, 6 of each misconduct type per form, 6 leaked items, 11 quadrature points, 6 EM iterations, and 40 isolation-forest trees. The statistical procedure is the same as the full run.
