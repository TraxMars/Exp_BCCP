# CHANGES

History: `a7b645f` (original MT_DS upload) → `archive/step1-step2` (Plan A working tree, archived as is)
→ commit 1 (simplification, no change to training) → commit 2 (MT_DS_PL + diagnostics).
Self-check outputs are in `tests/outputs/`.

## Commit 1 — simplification only

### What changed and why

| File | Change | Why |
|---|---|---|
| `code/cbc.py`, `code/cbc_gain.py`, `code/perturbation.py`, `tests/verify_cbc*.py`, `tests/verify_perturbation.py` | deleted | Plan A (BCCP) stopped; kept at tag `archive/step1-step2` |
| `scripts/step*_report.py`, `scripts/step1b_crosscheck.py`, `scripts/run_six.py`, `scripts/summarize.py`, `MT_DS_LossNorm/scripts/launch_three.py` | replaced by `scripts/launch.py` + `scripts/report.py` | one launcher (datasets × seeds, extra train args after `--`, queued by free GPU memory) and one multi-seed report |
| `MT_DS_LossNorm/` (code, tests, verification-run code copies) | deleted; `--loss_norm` in the single `code/` | the two code bases differed only in the LossNorm lines; `--loss_norm` computes `L_sup/sg(L_sup) + w·L_mt/sg(L_mt)` exactly as before (same `clamp_min(1e-8)`) |
| `code/train.py` (351 → 190 lines) | top-level imports, GPU via `CUDA_VISIBLE_DEVICES`; only `labeled_bs < batch_size` and labelnum-range checks; no TensorBoard, `run_status.json`, `latest_validation.json`, `teacher_at_best_student.pth`, code snapshot or `data_split_manifest.json`; `config.json` = args + git commit + labeled / unlabeled IDs; `--resume` restores the configuration stored in the checkpoint (still restores RNG states) | less bookkeeping, same training step |
| `code/networks/vnet_ds.py` (218 → 78) | batch-norm only, `conv_bn_relu` builder, no `taps` / `decode_from` | parameter names and module construction order unchanged (checked: same seeded initial weights) |
| `code/objectives.py` | `deep_ce_dice(outputs, targets)`; `supervised_loss` passes nearest-neighbour GT per scale; `normalize_loss` moved here | shared by the supervised and (commit 2) pseudo-label losses |
| `code/data.py` | numeric checks once at dataset init; train/val/test overlap check kept there; sha256 manifest removed | DataLoader and sampling unchanged |
| `provenance.json`, `launch_manifest.json`, `supervisor.json`, `verification_report.json`, `summary/*` | deleted | records of the October six-run group; they stay in `../MT_DS` and at the archive tag |
| `requirements.txt` | tensorboard removed | no longer used |
| `.gitignore` | `Results*/` keeps `*.csv`, `*.jsonl`, `*.json`; `*.pth` excluded | |

Old result directories (`../MT_DS/Results`, `../../Exp_BCCP/...`) are untouched.

Python line counts (tracked files): archive tag 6879 (incl. 2082 in LossNorm verification-run
code copies) → commit 1 848.

| file | before | after |
|---|---|---|
| code/train.py | 351 (+355 LossNorm copy) | 190 |
| code/networks/vnet_ds.py | 218 (+218) | 78 |
| code/objectives.py | 35 (+40) | 42 |
| code/data.py | 121 (+121) | 105 |
| code/evaluate.py | 66 (+66) | 66 |
| code/cbc.py, cbc_gain.py, perturbation.py | 605 (+605) | 0 |
| scripts | 913 (+73) | 186 (launch 89, report 97) |
| tests | 499 (+509) | 180 (verify_core 114, compare_csv 39, check_ce_nondeterminism 27) |

### Acceptance

- `tests/verify_core.py`: PASS — bitwise identical predictions to DuCiSC's original VNet at all four
  scales; identical parameter names, order and seeded initial weights to the legacy VNetDS (seeds 42 /
  500 / 1337); LossNorm gradient semantics (`tests/outputs/commit1/verify_core.txt`).
- MT_DS (default), seed 42, LA, 200 iterations vs `Exp_BCCP/PlanA_BCCP/MT_DS/Results_step1b/seed_42/LA_8l/training_metrics.csv`
  (`tests/outputs/commit1/bitwise_LA_200.txt`):
  - `loss_supervised`, `dice`, `loss_consistency`, `gradient_norm`: **bitwise identical** at all 21
    logged iterations, in two runs on two GPUs.
  - `ce`: 1 (resp. 2) of 21 logged values differ by one float32 ULP (e.g. it. 80: 0.2097110003 vs
    0.2097110301). Cause: CUDA `F.cross_entropy` (3D, mean) has no deterministic forward
    (`nll_loss2d_forward_out_cuda_template`; 4 distinct values over 200 identical calls,
    `tests/outputs/commit1/ce_nondeterminism.txt`). The CE gradient does not depend on that value,
    so the trajectory stays bitwise identical. Control: the **old code re-run** against the same
    reference shows 8 such mismatches (5 in `ce`, 3 in `loss_supervised`), so the strict criterion
    on `ce` cannot be met by any code, including the original.
- `--loss_norm` vs `MT_DS_LossNorm/Results_step1b/seed_42/LA_8l`: iteration 1 bitwise identical;
  from iteration 10–20 on, runs diverge — **including the old MT_DS_LossNorm code re-run as a
  control**, and two runs of the new code from each other. Under LossNorm the gradient is ∇L / L, so
  the ULP noise of the CE forward value enters every update and is amplified. LossNorm LA runs are
  therefore not bitwise reproducible with either code; equivalence rests on the identical formula
  and the unit test in `verify_core.py`.

## Commit 2 — MT_DS_PL baseline and lightweight diagnostics

### What changed and why

| File | Change | Why |
|---|---|---|
| `code/objectives.py` | `pseudo_label_consistency = deep_ce_dice(student, [t.detach().argmax(1) for t in teacher])` | DuCiSC's pseudo-label loss (`get_loss(..., if_pseudo=True)`) without the dynamic-threshold mask and without LossNorm; every scale uses its own Teacher output |
| `code/train.py` | `--consistency_type {mse, pseudo_ce_dice}` (default `mse`, unchanged behaviour); `L = L_sup + 0.3·L_cons` from iteration 2000 (constant); CSV adds `pl_ce`, `pl_dice` (empty under mse) and cumulative `data_wait_seconds` | new baseline MT_DS_PL; Teacher train mode, EMA 0.99, lr, batch, augmentation and evaluation untouched |
| `code/diagnostics.py`, `code/train.py` | `--diag_interval` (default 50, 0 = off) → `diagnostics.csv` | logging only, see below |
| `code/train.py` | `--persistent_workers / --no-persistent_workers`, default **off** | benchmark below: no gain, so the random stream is left as before |
| `tests/ducisc_reference.py` | verbatim `get_loss`, `convert_to_soft_label`, `custom_cross_entropy` (train_DuCiSC.py 100–158) and `Binary_dice_loss` (losses.py 4–8), with the original `dice_loss` binding | reference for the equivalence test |
| `tests/verify_pl.py`, `tests/bench_loader.py` | equivalence / diagnostics tests; DataLoader timing | |
| `scripts/report.py` | per-phase medians of `diagnostics.csv` (0–2k, 2k–6k, 6k–10k, 10k–15k, pooled over seeds) | |

Diagnostics (every 50 iterations, before `loss.backward()`):
- Gradient probes: `torch.autograd.grad(..., retain_graph=True)` of `L_sup` and of `0.3·L_cons` w.r.t.
  `block_seven.conv[-3].weight` (b7) and `block_three.conv[-3].weight` (b3); norms and cosine. The
  weight is the constant 0.3 also before warmup. After the probes the code raises if any parameter
  `.grad` is not None. Under `--loss_norm` the probes use the raw (unnormalized) losses.
- Oracle boundary statistics on the full-resolution head of the unlabeled half: GT band and pseudo
  band = dilation − erosion (cube, radius 2, patch borders ignored). `ce_pl_in/out`, `ce_gt_in/out`
  are the student CE to the Teacher argmax and to the hidden GT, inside / outside the **GT band**
  (same voxels for both); `band_iou` (pseudo band vs GT band); Teacher mean max-probability inside the
  GT band and inside the pseudo band; `gt_band_fraction`. The hidden GT is sliced from the batch only
  after the objective is built and is passed only to a `no_grad` function that returns floats (checked);
  `tests/verify_pl.py` checks that changing it leaves all parameter gradients bitwise unchanged.

### Acceptance (`tests/outputs/commit2/`)

1. **DuCiSC equivalence** (`verify_pl.txt`): four scales with VNetDS LA shapes, batch 2, coarsest
   scale's pseudo-label all background, 3 random trials on CPU and on CUDA: max |value diff| of
   `(ce + dice)/2` (and of ce, dice separately) 0 on CPU, 6.0e-8 on CUDA; max |gradient diff| w.r.t.
   the student outputs 0. Both ≤ 1e-6. The Teacher outputs receive no gradient.
2. **Diagnostics do not interfere** (`zero_interference_LA_100.txt`, MT_DS mse, LA, seed 42, 100
   iterations): diagnostics every 50 and every iteration vs off — `dice`, `loss_consistency`,
   `gradient_norm` bitwise identical at every logged iteration, and the diagnostic rows at 50/100 are
   identical between the every-50 and every-iteration runs. `ce` (and `loss_supervised = loss_total`
   before warmup, which contain it) show the same one-ULP CUDA CE-forward noise **between two
   diagnostics-off runs** (6 mismatches) as between off and on (4–5), so they are not caused by the
   diagnostics (see commit 1). vs the step1b log: identical except the same CE noise.
3. **Smoke run** (`smoke_pl_LA/`, `smoke_pl_LA_summary.txt`; pseudo_ce_dice, LA, warmup 100, 300 it.):
   - all losses and diagnostics finite; up to iteration 100 the trajectory equals the mse run
     (consistency not yet in the objective).
   - after warmup L_sup 0.13–0.31, L_pl 0.10–0.29 (median L_pl/L_sup 0.93; weighted 0.3·L_pl ≈ 0.28·L_sup).
   - total gradient norm (logged rows): median 0.96 for (50, 100] → 1.25 for (100, 150] → 1.19 for
     (150, 300]; single-step values are noisy (0.61–2.76).
   - probe ratio |∇0.3·L_pl| / |∇L_sup|: b7 0.28–1.50, b3 0.20–2.86 (median ≈ 0.6 / 0.47); cosine
     b7 −0.78 … +0.39, b3 −0.30 … +0.53. For comparison, at iterations 50 / 100 the mse consistency
     gives |∇0.3·L_mse| = 0.0035 / 0.0051 on b7 and 0.018 on b3, i.e. 4–8× smaller than the
     pseudo-label term.
   - boundary: band IoU 0.11–0.29 and Teacher confidence in the band 0.62–0.80 this early; after
     warmup the student CE to the hidden GT inside the band (0.69–1.18) is well above its CE to the
     pseudo-label (0.25–0.85).
4. **report.py** (`report_check_*.md`): validation / test part checked on the October MT_DS results
   (LA_8l test Dice 0.8775 = its `test_metrics.json`), diagnostics part on the smoke run. With one seed
   the mean ± sd column shows the value and n=1.
5. **DataLoader** (`bench_loader.txt`, 100 iterations, 2 repeats each): persistent workers change the
   elapsed time by −3.5 % (LA, slower) and +2.6 % (Pancreas) — far below the 15 % threshold, so the
   default stays off. Waiting for batches takes ~25–30 % of LA and ~70 % of Pancreas time with
   `num_workers=2` regardless of persistence (each sample reads a whole volume); raising `num_workers`
   would help Pancreas but changes the random stream, so it is not changed here.
6. **.gitignore**: unchanged from commit 1 (`Results*/` keeps csv / jsonl / json, `*.pth` excluded).

### Not started (waiting for review)

```bash
PY=~/.conda/envs/SSL29/bin/python
$PY scripts/launch.py --datasets LA Pancreas --seeds 42 500 1337 --output_root Results_baseline_pl \
    -- --consistency_type pseudo_ce_dice
$PY scripts/report.py --root Results_baseline_pl --output summary/baseline_seeds.md
```
