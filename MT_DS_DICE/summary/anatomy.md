# Boundary-error anatomy of the V0 Teachers (Results_anatomy)

Teacher of each V0 run at iteration 15000, unlabeled training cases with their hidden GT; mode `train` (pseudo-labels as made in training) unless marked `sliding`. Mean ± sample std over the seeds. Definitions and rules: README.md, "Boundary-error anatomy".

## LA_8l (seeds 1337, 42, 500; 72 cases each)

### Error types (contribution to Dice)

|  | block | shift | jitter |
|---|---|---|---|
| share of ΔDice | 0.515 ± 0.027 | 0.408 ± 0.022 | 0.077 ± 0.005 |
| ΔDice (points) | 5.52 ± 0.46 | 4.37 ± 0.10 | 0.83 ± 0.03 |
| ΔHD95 (voxels, reported only) | 7.02 ± 1.77 | 0.29 ± 0.04 | 0.06 ± 0.01 |
| share of error voxels | 0.538 ± 0.023 | 0.390 ± 0.019 | 0.072 ± 0.004 |
| sliding reference: share of ΔDice | 0.537 ± 0.020 | 0.394 ± 0.017 | 0.069 ± 0.003 |

- Dice of the pseudo-labels (mode `train`): 89.17 ± 0.43; sliding reference: 89.06 ± 0.36; HD95 9.31 ± 1.77.
- Shift direction consistency: 0.565 ± 0.057 (majority: under (FN), over (FP), over (FP)); share of cases over-segmented: 0.639 ± 0.077.
- Oracle threshold (upper bound of calibration): best single threshold 0.50, 0.80, 0.65 → +0.13 ± 0.19 Dice points; best per case → +0.67 ± 0.23.
- Reachable by U→L (errors with |d̃| ≤ 6, share of ΔDice): 0.941 ± 0.006.
- Check against training diagnostics: Teacher error in the GT band (radius 2) 0.223 ± 0.002, conf > 0.95 among those errors 0.750 ± 0.004.

### By pseudo layer |d̃| (pooled over cases and seeds)

| |d̃| | error rate | share of errors | FP share | error rate, conf > 0.95 | block | shift | jitter |
|---|---|---|---|---|---|---|---|
| 1 | 0.321 | 0.523 | 0.593 | 0.267 | 0.285 | 0.590 | 0.126 |
| 2 | 0.126 | 0.201 | 0.570 | 0.123 | 0.606 | 0.366 | 0.028 |
| 3 | 0.062 | 0.096 | 0.528 | 0.060 | 0.920 | 0.077 | 0.003 |
| 4-6 | 0.024 | 0.111 | 0.443 | 0.024 | 1.000 | 0.000 | 0.000 |
| >6 | 0.002 | 0.069 | 0.265 | 0.002 | 1.000 | 0.000 | 0.000 |

### Separability of confident errors (conf > 0.95; AUROC, mean over seeds)

| signal | |d̃| = 1 | |d̃| = 2 | |d̃| = 3 | A (weighted) | unstratified 1–3 (not used) | AP 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| entropy | 0.610 ± 0.019 | 0.606 ± 0.029 | 0.572 ± 0.029 | 0.603 ± 0.023 | 0.675 ± 0.021 | 0.341 / 0.177 / 0.084 |
| ds_disagreement | 0.557 ± 0.014 | 0.585 ± 0.025 | 0.614 ± 0.028 | 0.575 ± 0.020 | 0.638 ± 0.019 | 0.305 / 0.164 / 0.100 |
| flip_eval | 0.690 ± 0.013 | 0.729 ± 0.015 | 0.745 ± 0.016 | 0.711 ± 0.013 | 0.755 ± 0.011 | 0.442 / 0.289 / 0.196 |
| tta_train | 0.696 ± 0.011 | 0.721 ± 0.011 | 0.726 ± 0.016 | 0.709 ± 0.009 | 0.750 ± 0.007 | 0.453 / 0.288 / 0.193 |

Error base rate among confident voxels, |d̃| = 1 / 2 / 3: 0.267 / 0.123 / 0.060. `entropy` ranks exactly as the confidence (two classes).

### Decision (README rules 1–3)

- Dominant error type: **block** → new information needed (shape / position priors); U→L reaches at most 0.94.
- Signals separating confident errors (A ≥ 0.70 and ≥ A_entropy + 0.05): **flip_eval, tta_train** → reliability weighting with it is viable.
- Non-block share: 0.485.

Runs: seed_1337: /aita_data/home/aita_duanxuelong/Baseline/MT_DS_DICE/Results_baseline_pl/seed_1337/LA_8l (Teacher iteration 15000, anatomy commit 1c7173e, 19 min); seed_42: /aita_data/home/aita_duanxuelong/Baseline/MT_DS_DICE/Results_baseline_pl/seed_42/LA_8l (Teacher iteration 15000, anatomy commit 1c7173e, 17 min); seed_500: /aita_data/home/aita_duanxuelong/Baseline/MT_DS_DICE/Results_baseline_pl/seed_500/LA_8l (Teacher iteration 15000, anatomy commit 1c7173e, 25 min)

## Pancreas_6l (seeds 1337, 42, 500; 56 cases each)

### Error types (contribution to Dice)

|  | block | shift | jitter |
|---|---|---|---|
| share of ΔDice | 0.658 ± 0.023 | 0.218 ± 0.014 | 0.125 ± 0.009 |
| ΔDice (points) | 16.96 ± 1.22 | 5.60 ± 0.18 | 3.21 ± 0.18 |
| ΔHD95 (voxels, reported only) | 32.93 ± 9.72 | 1.57 ± 0.37 | 0.87 ± 0.21 |
| share of error voxels | 0.676 ± 0.023 | 0.213 ± 0.013 | 0.110 ± 0.011 |
| sliding reference: share of ΔDice | 0.569 ± 0.042 | 0.284 ± 0.021 | 0.148 ± 0.022 |

- Dice of the pseudo-labels (mode `train`): 72.61 ± 1.18; sliding reference: 77.51 ± 1.39; HD95 35.13 ± 9.72.
- Shift direction consistency: 0.686 ± 0.041 (majority: over (FP), over (FP), over (FP)); share of cases over-segmented: 0.792 ± 0.037.
- Oracle threshold (upper bound of calibration): best single threshold 0.90, 0.90, 0.75 → +0.95 ± 0.41 Dice points; best per case → +1.91 ± 0.22.
- Reachable by U→L (errors with |d̃| ≤ 6, share of ΔDice): 0.954 ± 0.007.
- Check against training diagnostics: Teacher error in the GT band (radius 2) 0.226 ± 0.002, conf > 0.95 among those errors 0.695 ± 0.021.

### By pseudo layer |d̃| (pooled over cases and seeds)

| |d̃| | error rate | share of errors | FP share | error rate, conf > 0.95 | block | shift | jitter |
|---|---|---|---|---|---|---|---|
| 1 | 0.351 | 0.508 | 0.756 | 0.271 | 0.487 | 0.331 | 0.183 |
| 2 | 0.166 | 0.220 | 0.788 | 0.145 | 0.746 | 0.184 | 0.071 |
| 3 | 0.089 | 0.113 | 0.794 | 0.073 | 0.948 | 0.039 | 0.013 |
| 4-6 | 0.033 | 0.122 | 0.734 | 0.025 | 1.000 | 0.000 | 0.000 |
| >6 | 0.000 | 0.037 | 0.288 | 0.000 | 1.000 | 0.000 | 0.000 |

### Separability of confident errors (conf > 0.95; AUROC, mean over seeds)

| signal | |d̃| = 1 | |d̃| = 2 | |d̃| = 3 | A (weighted) | unstratified 1–3 (not used) | AP 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| entropy | 0.677 ± 0.031 | 0.736 ± 0.048 | 0.737 ± 0.063 | 0.707 ± 0.042 | 0.752 ± 0.041 | 0.405 / 0.290 / 0.153 |
| ds_disagreement | 0.643 ± 0.021 | 0.710 ± 0.036 | 0.756 ± 0.039 | 0.686 ± 0.028 | 0.731 ± 0.030 | 0.388 / 0.321 / 0.255 |
| flip_eval | 0.777 ± 0.020 | 0.831 ± 0.025 | 0.867 ± 0.021 | 0.811 ± 0.021 | 0.834 ± 0.022 | 0.581 / 0.507 / 0.424 |
| tta_train | 0.773 ± 0.020 | 0.819 ± 0.025 | 0.848 ± 0.020 | 0.802 ± 0.021 | 0.819 ± 0.022 | 0.576 / 0.491 / 0.392 |

Error base rate among confident voxels, |d̃| = 1 / 2 / 3: 0.270 / 0.145 / 0.073. `entropy` ranks exactly as the confidence (two classes).

### Decision (README rules 1–3)

- Dominant error type: **block** → new information needed (shape / position priors); U→L reaches at most 0.95.
- Signals separating confident errors (A ≥ 0.70 and ≥ A_entropy + 0.05): **entropy, flip_eval, tta_train** → reliability weighting with it is viable.
- Non-block share: 0.342.

Runs: seed_1337: /aita_data/home/aita_duanxuelong/Baseline/MT_DS_DICE/Results_baseline_pl/seed_1337/Pancreas_6l (Teacher iteration 15000, anatomy commit 1c7173e, 37 min); seed_42: /aita_data/home/aita_duanxuelong/Baseline/MT_DS_DICE/Results_baseline_pl/seed_42/Pancreas_6l (Teacher iteration 15000, anatomy commit 1c7173e, 26 min); seed_500: /aita_data/home/aita_duanxuelong/Baseline/MT_DS_DICE/Results_baseline_pl/seed_500/Pancreas_6l (Teacher iteration 15000, anatomy commit 1c7173e, 43 min)

## Together with V1 (README rule 4)

Non-block share ≥ 0.5 on both datasets: **no** (LA_8l 0.49, Pancreas_6l 0.34). V1 go → discuss (block errors dominate on a dataset); V1 stop → discuss.
