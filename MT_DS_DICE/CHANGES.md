# CHANGES — MT_DS_PL

MT_DS_DICE is now the single baseline MT_DS_PL. Earlier variants (MSE consistency, LossNorm, Plan A)
are at tags `archive/plan-a-mse` (= `ae3ea95`) and `archive/step1-step2`.

## V1 probe (logging only; README.md, "V1 probe")

| Area | Change |
|---|---|
| `code/v1_probe.py` (new) | distance buckets, sampling (own Generator), grouped attention grid, label- and gradient-level reverse correction, output sensitivity, bucket profiles → `v1_probe.csv` |
| `networks/vnet_ds.py` | `forward(..., features=True)` also returns the same `x1`, `x7_up`; `decode_from_x7(x7_up, x1, dropout=False)`. Construction and computation order unchanged |
| `train.py` | `--v1_probe` (off by default, needs `--diag_interval`); on probe steps the forward also returns its features; probe after the diagnostics, before `backward()`; `loss_true` kept until after it; the ".grad is None" check repeated; resume of an older checkpoint defaults `v1_probe=False` |
| Tests | `tests/verify_v1.py`, `tests/zero_interference_v1.py` |

The extra decodes must not update the decoder's BN running buffers in place even if the values are copied
back: the training graph saved those tensors for `native_batch_norm_backward`, and the version-counter check
makes `loss.backward()` fail (found by `verify_v1.py`). They run on copies of the buffers instead.

Acceptance (`tests/outputs/v1/`; all PASS):

1. `verify_v1_cpu.txt`: buckets = clipped scipy chessboard distance (13 masks, patch borders touched);
   hand example for contamination / effective keys / swap / correction / harm; a full probe step leaves .grad
   None, the state_dict (BN buffers included) and the global RNG unchanged and `backward()` usable; one
   step's parameter gradients with and without the probe are bitwise equal; signed correction rate 0;
   deterministic for a fixed seed; query block size has no effect (≤ 1.1e-6).
2. `zero_interference_v1.txt` (LA, 100 iterations, warmup 50, GPU): committed code vs this code with the probe
   off / every 50 / every iteration: training metrics bitwise (5 columns) and ≤ 1.33e-7 (CE-derived, the
   off-vs-off noise floor), the 31 diagnostics columns at 50 and 100 bitwise, final Student / Teacher
   state_dict (87 BN buffers each) and optimizer momentum bitwise; the probe rows at 50 and 100 are bitwise
   equal between every-50 and every-iteration runs. Pancreas off vs probe (every 10): PASS.
3. Overhead (A100, GPU shared with other jobs): LA 2.1 s per probe step (median of 100; first step 3.6 s),
   Pancreas 1.0 s (median of 10); peak allocated +15 MiB (LA) / +18 MiB (Pancreas); every 50 iterations
   this is about +12 % (LA) / +5 % (Pancreas) training time, ≈ 11 / 5 min over 15000 iterations.

## What changed and why

| Area | Change | Why |
|---|---|---|
| Objective (`objectives.py`, `train.py`) | removed `consistency_loss`, `normalize_loss`, `--consistency_type`, `--loss_norm`; `L = L_sup + 0.3·L_pl` from iteration 2000, `L_pl` = per-scale (CE + Dice)/2 to that scale's Teacher argmax, no mask, no LossNorm | one baseline |
| Names | CSV `loss_consistency` → `loss_pl`; probes `gnorm_cons_*` → `gnorm_pl_*` | |
| Evaluation (`evaluate.py`) | DuCiSC protocol: `label[label < 0] = 0`; empty prediction → dice = jc = hd95 = asd = 0, counted in the mean, `empty_prediction_cases`; `nms1` via `largest_component` (scipy 26-connectivity = skimage default; empty prediction left unchanged instead of asserting); per-metric mean and `np.std`. Sliding window unchanged | results comparable with DuCiSC's tables |
| `test_metrics.json` | main entry nms=0, plus `nms1` | DuCiSC's default is nms=0 |
| `code/test.py` (new) | `--run_dir <run> [--nms]` re-tests `best_model.pth` → `retest_metrics.json` (cuDNN deterministic, as in training) | DuCiSC's `test_3d.py` |
| Diagnostics (`diagnostics.py`) | new: `teacher_err_gt_band`, `conf_wrong_095`, `conf_right_095`, `ce_lab_in/out` (`labeled_band_ce`); oracle gradient space with `L_true` (hidden-GT deep CE + Dice, `autograd.grad` only): `cos_pl_true_*`, `cos_sup_true_*`, `frac_ch_pl_true_neg_*`, `frac_ch_pl_sup_neg_*`, `ch_rank_corr_*`, `hit_rate_*` | whether labeled gradients can tell, per channel, if a pseudo-label update is trustworthy |
| Resume | `best_dice = max(checkpoint, best_metrics.json)` | best_model.pth can be newer than the last periodic checkpoint |
| Data (`data.py`) | training volumes cached at init (float32 image, uint8 label); `__getitem__` takes `label.astype(np.int64)`, augmentation untouched; `--no-cache_volumes` reads from disk. `--persistent_workers` removed (no gain earlier) | Pancreas loop ×2.97 faster |
| `scripts/launch.py` | refuses a dirty tree (ignored `Results*` excepted) | every run's `git_commit` names a clean commit |
| `scripts/report.py` | jsonl / csv de-duplicated by iteration (last wins); Dice / JC in % and HD95 / ASD with 2 decimals; columns `empty pred.`, `nms1 Dice`, `nms1 HD95` | |
| Tests | `verify_core.py` without LossNorm / MSE; `verify_pl.py` + new-column and oracle-gradient checks and the inverted-GT test; `verify_ducisc_test.py` (new); `ducisc_reference.py` + verbatim `test_patch.py` functions (`.cuda()` → `.to(device)` only); `compare_csv.py` `OLD:NEW` renames and `--near`; `bench_loader.py` now measures the cache | |
| Removed outputs | `tests/outputs/commit1/*`, `tests/outputs/commit2/zero_interference_LA_100.txt` (MSE / LossNorm) | `tests/outputs/commit2/` otherwise holds the previous commit's outputs |

## Acceptance (`tests/outputs/mt_ds_pl/`; all PASS)

The CUDA CE forward is not deterministic (`ce_nondeterminism.txt`: 5 distinct values over 200
identical calls); the CE gradient does not depend on that value. Running the **same frozen code
twice** (ref0 vs ref0b) shows it: `ce`, `pl_ce` and the sums containing them (`loss_supervised`,
`loss_pl`, `loss_total`) differ by ≤ 2.2e-7 relative, while `dice`, `pl_dice`, `gradient_norm`,
`lr` and every diagnostics column are bitwise identical. Comparisons below therefore check those
columns bitwise and the CE-derived ones within 4 float32 ULP.

1. **Regression** (`regression_E1.txt`, CSVs in `runs/`): step-0 reference from `ae3ea95`
   (`--consistency_type pseudo_ce_dice --warmup_iterations 100 --max_iterations 300 --val_interval 0`,
   seed 42) vs this code with the same settings, LA and Pancreas: training metrics bitwise / ≤ 2.2e-7,
   the 14 original diagnostics columns bitwise; the 17 new columns are all finite (102/102 per dataset).
2. **Zero interference** (`zero_interference_pl_100.txt`): LA, 100 iterations, warmup 50; diagnostics
   off (×2) / every 50 / every iteration give identical trajectories (≤ 1.3e-7 on CE-derived columns,
   same as off vs off); the rows at 50 / 100 agree bitwise in all 31 diagnostic columns between the
   every-50 and every-iteration runs. `hit_rate_b7` is NaN in 1 of 100 rows (no channel with
   cos(pl, sup) < 0).
3. **Tests**: `verify_core.txt`, `verify_pl.txt` (L_pl vs DuCiSC `get_loss`: value diff ≤ 6e-8, gradient
   diff 0; inverted hidden GT → parameter gradients bitwise unchanged while the oracle columns change;
   new columns vs hand / scipy computations), `verify_ducisc_test.txt` (October MT_DS LA_8l
   `best_model.pth`; 2 LA + 2 Pancreas test cases: masks bitwise equal, per-case metrics equal for nms 0
   and 1, `var_all_case` averages equal to `evaluate`, `test_all_case` summary lines equal; synthetic
   empty prediction: zeros, counted, nms skipped).
4. **End to end** (`end_to_end.txt`, `runs/e2e_LA/`): validation → best Student → test with nms0 / nms1;
   `code/test.py` reproduces `test_metrics.json` exactly; resume takes the larger best Dice;
   `report.py` de-duplicates rows (`report_check_e2e.md`).
5. **Cache** (`bench_loader.txt`, 100 iterations × 2): loop time LA ×1.20, Pancreas ×2.97; batch-wait
   share 32 % → 19 % (LA), 76 % → 25 % (Pancreas); peak memory (PSS, main + workers) 1.75 → 2.66 GiB
   (LA), 1.79 → 3.40 GiB (Pancreas).

## Runs

- V0 baseline: `scripts/launch.py --datasets LA Pancreas --seeds 42 500 1337 --output_root Results_baseline_pl`,
  run from the separate checkout `Baseline/MT_DS_DICE` at 49758b3; all 6 runs finished 2026-11-06.
  Test Dice (nms0, mean ± std, n = 3): LA_8l 89.38 ± 0.20, Pancreas_6l 80.64 ± 0.16.
- V1 probe: LA_8l and Pancreas_6l, seed 42, `Results_V1/` (commands in README.md, "V1 probe").
