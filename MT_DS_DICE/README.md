# MT_DS_DICE — baseline MT_DS_PL

Mean Teacher with four-scale deep supervision on DuCiSC's batch-norm VNet (parameter names and
seeded initialization identical to the original), trained with DuCiSC's pseudo-label loss:

```text
L_sup = mean_4_scales((CE + foreground Dice) / 2)   to the labeled GT (nearest-neighbour per scale)
L_pl  = mean_4_scales((CE + foreground Dice) / 2)   to argmax(Teacher) of the same scale; no mask, no LossNorm
L     = L_sup                  iteration < 2000
L     = L_sup + 0.3 * L_pl     afterwards (constant weight)
```

15000 iterations, constant lr 0.01 (SGD, momentum 0.9, wd 1e-4), EMA 0.99, batch 2 labeled + 2
unlabeled, Teacher in train mode (independent Dropout3d p=0.5). Training volumes are validated
once and kept in memory (`--no-cache_volumes` reads them from disk; sampling is identical).

## Evaluation (DuCiSC's `utils/test_patch.py`)

Sliding window with the averaged softmax foreground, threshold 0.5; LA patch 112×112×80, stride
18/4; Pancreas 96³, stride 16/16; BraTS19 stride 16. `label[label < 0] = 0`; an empty prediction scores
Dice = JC = HD95 = ASD = 0 and counts in the mean (`empty_prediction_cases` records how often).
Validation every 200 iterations from 800 on test.list (BraTS19: val.list); the Student is saved
when Dice strictly improves; at the end the best Student is tested on test.list.
`test_metrics.json` holds the nms=0 result (DuCiSC's default) and `nms1` (largest 26-connected
component, getLargestCC), each with per-case values, means and population std (`np.std`).

```bash
PY=~/.conda/envs/SSL29/bin/python
CUDA_VISIBLE_DEVICES=0 $PY code/train.py --dataset LA --output_dir Results_tmp/seed_42/LA_8l
CUDA_VISIBLE_DEVICES=0 $PY code/test.py --run_dir Results_tmp/seed_42/LA_8l --nms   # re-test, like test_3d.py
$PY scripts/launch.py --datasets LA Pancreas --seeds 42 500 1337 --output_root Results_baseline_pl
$PY scripts/report.py --root Results_baseline_pl --output summary/baseline_seeds.md
```

`launch.py` refuses to start from a dirty working tree (ignored `Results*` excepted), so
`config.json` → `git_commit` always names a clean commit.

## Outputs per run

`config.json` (arguments, git commit, labeled / unlabeled case IDs), `training_metrics.csv`,
`validation_metrics.jsonl`, `best_metrics.json`, `test_metrics.json`, `diagnostics.csv`, `log.txt`,
`best_model.pth`, `last_checkpoint.pth`. `--resume <last_checkpoint.pth>` restores the stored
configuration and RNG states (DataLoader workers are not restored bit-exactly); the best Dice is the
larger of the checkpoint's value and `best_metrics.json`.

## Diagnostics (`diagnostics.csv`, every 50 iterations, logging only)

Computed before `backward()`; they never enter the objective (all parameter `.grad` stay `None`,
checked at run time; trajectories are identical with diagnostics off, every 50 or every iteration).

- Gradient probes on `block_seven.conv[-3].weight` (b7) and `block_three.conv[-3].weight` (b3):
  `gnorm_sup_*`, `gnorm_pl_*` (of `0.3·L_pl`), `cos_*`.
- Oracle gradient space: `L_true` = the same deep CE + Dice of the unlabeled outputs against their
  hidden GT, differentiated by `autograd.grad` only. `cos_pl_true_*`, `cos_sup_true_*`; per output
  channel cosines cos(g_pl[c], g_sup[c]) and cos(g_pl[c], g_true[c]) → `frac_ch_pl_true_neg_*`,
  `frac_ch_pl_sup_neg_*`, `ch_rank_corr_*` (Spearman), `hit_rate_*` (share of channels with
  cos(pl, sup) < 0 that also have cos(pl, true) < 0; NaN if none).
- Oracle boundary, full-resolution head, bands = dilation − erosion (radius 2): student CE to the
  pseudo-label / hidden GT inside and outside the GT band (`ce_pl_in/out`, `ce_gt_in/out`),
  `band_iou` (pseudo vs GT band), Teacher confidence in the GT / pseudo band, `gt_band_fraction`,
  `teacher_err_gt_band`, `conf_wrong_095` / `conf_right_095` (share of wrong / right Teacher voxels in
  the GT band with confidence > 0.95).
- Labeled: `ce_lab_in`, `ce_lab_out` (student CE to the labeled GT inside / outside its band).

Columns are NaN where a band or subset is empty (e.g. a patch without foreground).

## Tests

`tests/verify_core.py` (backbone vs DuCiSC's VNet, initialization, gradients, EMA),
`tests/verify_pl.py` (L_pl vs DuCiSC's `get_loss`, diagnostics), `tests/verify_ducisc_test.py`
(evaluation vs DuCiSC's `test_patch.py` with fixed weights), `tests/compare_csv.py`,
`tests/check_ce_nondeterminism.py`, `tests/bench_loader.py`. Verbatim DuCiSC reference code is in
`tests/ducisc_reference.py`. Results: `tests/outputs/mt_ds_pl/`; see `CHANGES.md`.

Earlier variants (MSE consistency, LossNorm, Plan A) are kept at tags `archive/plan-a-mse` and
`archive/step1-step2`.
