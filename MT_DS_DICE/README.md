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

## V1 probe (`--v1_probe`, `v1_probe.csv`, logging only)

Observes, during MT_DS_PL training, a labeled ← unlabeled grouped channel cross-attention at D2 (`x7_up`,
32 channels, 1/2 resolution) whose keys are retrieved by boundary distance. Off by default; with it, the
training trajectory is unchanged (`tests/outputs/v1/`). It runs on the diagnostics steps, after the
diagnostics and before `backward()`, on the same tensors (`code/v1_probe.py`).

**Definitions (all at D2).** Labeled GT and hidden unlabeled GT are downsampled nearest-neighbour (as
`supervised_loss`); the unlabeled pseudo-label is `teacher_outputs[2].argmax`, its confidence the softmax
maximum. Signed distance bucket d ∈ {−3, −2, −1, +1, +2, +3}: background voxels first covered by the k-th
3×3×3 max-pool (stride 1, padding 1) of the foreground get d = +k, foreground voxels covered by that of the
background d = −k (= Chebyshev distance, checked against scipy); |d| ≤ 3 is the band. The −inf padding of
max-pool keeps patch borders from counting as object borders. Query buckets come from the labeled GT, key
buckets b̃ from the pseudo-label; the true side of a key (hidden GT) is used for evaluation only.
Sampling uses its own `torch.Generator`, seeded with `(seed + 12345) · 1000003 + iteration`, so the
samples of a step depend on (seed, iteration) only and do not depend on how often the probe runs; queries:
≤ 512 labeled band tokens per GT bucket (both labeled cases pooled); keys: ≤ 1024 unlabeled band tokens per
pseudo bucket plus 2048 random unlabeled tokens outside the pseudo band (both unlabeled cases pooled).
Attention per channel group g (G groups): w^g = softmax_j(cos(f^g, u^g_j)/τ + mask), f'^g = Σ_j w^g_j u^g_j,
keys detached. Masks: `signed` b̃_j = b(x); `unsigned` |b̃_j| = |b(x)|; `band` any in-band key; `none`
every key. Grid: 4 masks × G ∈ {1, 4} × τ ∈ {0.05, 0.1, 0.2}, plus per mask a uniform-weight version
(`*_uniform`, every allowed key equally weighted; G = τ = NA). Queries without an allowed key are left out.

**Rows** (`iteration, family, constraint, G, tau, subset, metric, value`; NA where a dimension does not apply):

- `attention` / `attention_uniform`: `contamination` = Σ_j w_j·1[true side(u_j) ≠ GT side(x)], `swap` =
  1 − cos(f, f') (32-d, groups concatenated), `eff_keys` = exp(attention entropy); averaged over groups and
  queries; subsets `all`, `d1`, `d2`, `d3` (query |d|); `n_valid_queries`.
- `reverse_label` / `reverse_label_uniform` (every grid point): with w averaged over groups, m_j = Σ_x w_j(x)
  and r_j = Σ_x w_j(x)·GT side(x) / m_j; majority side of r_j (> 0.5 foreground, < 0.5 background; a tie is
  none). `correction_rate`: on wrong-pseudo-side keys, m-weighted share whose majority is the true side;
  `harm_rate`: on right-pseudo-side keys, m-weighted share whose majority is the other side;
  `correction_mass` / `harm_mass`: the same masses as a share of the total attention mass; `wrong_mass_frac`:
  share of the mass on wrong-pseudo-side keys (coverage). Subsets `all` and `conf095` (keys with Teacher
  confidence > 0.95). By construction the signed correction rate is 0; the probe stops with an error otherwise.
- `reverse_grad` (signed / unsigned, G = 4, τ = 0.1): L_UL = CE(out_conv_2(f'), labeled GT) over the sampled
  queries, with the values u_j taken from the training graph's unlabeled `x7_up` (not detached; weights
  from detached keys); g_UL = ∂L_UL/∂x7_up_U, g_pl = ∂(0.3·L_pl)/∂x7_up_U, g_true = ∂L_true/∂x7_up_U by
  `autograd.grad` only. On key tokens with m_j > 1e-3 the per-token cosines `cos_ul_true` = cos(g_UL, g_true)
  and `cos_pl_true` = cos(g_pl, g_true), medians over subsets `wrong`, `wrong_conf095`, `right` (pseudo
  side), with `n_tokens`; `loss_ul`. g_UL passes the D2 head only: it approximates the method's reverse path
  (in the method the replaced feature would continue through the decoder).
- `sensitivity` (signed / unsigned, G = 4, τ = 0.1): every labeled D2 band token is replaced by its attention
  output (no_grad, query blocks); the decoder is run twice from `x7_up` (`decode_from_x7`: block_eight →
  block_eight_up → +x1 → block_nine → out_conv, no Dropout, train-mode BN on copies of the running buffers,
  inside `fork_rng`). In the full-resolution GT band (radius 2, `diagnostics.boundary_band`):
  `dp_final_band` mean |Δ p_fg|, `dce_final_band` mean CE change to the GT (after − before),
  `flip_final_band` share of changed argmax; `dp_d2` mean |Δ p_fg| of the D2 head on the replaced tokens;
  `bypass_ratio` = dp_final_band / dp_d2. Also `dp_final_replaced` / `bypass_ratio_replaced` (full-resolution
  voxels under replaced D2 tokens) and `replaced_fraction` (band tokens with at least one allowed key).
- `profile`: per bucket mean feature (32-d, all tokens): `cos_unl_pseudo_vs_true` (unlabeled by pseudo vs
  true bucket), `cos_lab_vs_unl_true` (labeled by GT bucket vs unlabeled by true bucket),
  `reldiff_lab_vs_unl_true` = ‖μ_L − μ_U‖ / mean(‖μ_L‖, ‖μ_U‖); subset `global`: the same relative
  difference of the global channel means; `bucket_mean`: mean over buckets.
- `sample`: `n_query` / `n_key` per bucket, `n_key` outside the band; `meta`: `probe_seconds`.

**Zero interference.** Everything runs under no_grad except the `reverse_grad` probe, which calls
`autograd.grad` only (the existing ".grad must be None" check runs after it). The extra decodes use
copies of the decoder's BN running buffers: those buffers are saved by the training graph for its
backward, so even an in-place update that is copied back would break `backward()`.

**Criteria (fixed before the V1 runs).** For each dataset (LA_8l, Pancreas_6l, seed 42), take for every
quantity the median over the probe steps with 6000 ≤ iteration ≤ 15000 (`scripts/report_v1.py` →
`summary/V1.md`; NaN values, i.e. empty subsets or buckets on that step, are left out); the criteria must hold
on both datasets.

- Forward (premise of U→L), for each of `signed` and `unsigned` at G = 4: τ* = the τ with the lowest median
  `contamination` (subset `all`). Holds when contamination(τ*) ≤ the same mask's uniform-weight
  contamination, and ≤ contamination(`none`, G = 4, τ*) − 0.05, and median `swap`(τ*) ≥ 0.1. "Forward
  holds" requires both `signed` and `unsigned`.
- Reverse (premise of the two-way scheme), `unsigned`, G = 4, τ = 0.1: `correction_mass` > `harm_mass`
  (subset `all`); `correction_rate` (subset `conf095`) ≥ 0.5; median `cos_ul_true` (subset `wrong`) > 0
  and > median `cos_pl_true` (subset `wrong`).
- Output sensitivity (risk flag, not a stopping criterion): `bypass_ratio` < 0.2 → "x1 skip bypass is strong".
- Decision: forward and reverse hold → unsigned buckets, two-way, go to V2; forward holds and reverse does
  not → signed buckets, one-way U→L, go to V2; forward does not hold → stop and discuss (fallback: bucket
  profiles only, ablation level 2).

**Runs.** Same settings as the V0 baseline (`Results_baseline_pl`, commit 49758b3) plus the probe; GPU
chosen with `nvidia-smi`:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -u code/train.py --dataset LA --labelnum 8 --seed 42 \
    --diag_interval 50 --v1_probe --output_dir Results_V1/seed_42/LA_8l
CUDA_VISIBLE_DEVICES=<gpu> python -u code/train.py --dataset Pancreas --labelnum 6 --seed 42 \
    --diag_interval 50 --v1_probe --output_dir Results_V1/seed_42/Pancreas_6l
```

Acceptance: `training_metrics.csv` of each run vs the V0 seed-42 run, as in `CHANGES.md` (`tests/compare_csv.py`,
5 columns bitwise, the CE-derived ones `--near`).

## Boundary-error anatomy (offline, V0 Teachers; `scripts/anatomy.py`)

What kind of errors do the Teacher's pseudo-labels make, and can any signal tell its confident errors from
its confident correct voxels? Inputs: the Teacher of each V0 run (`last_checkpoint.pth`, iteration 15000; the
earlier Teachers were not kept), the run's unlabeled training cases (`config.json`) and their hidden GT.
Training code is not touched.

**Inference.** Main mode `train`, as pseudo-labels are made in training: Teacher in train mode (BN batch
statistics, Dropout on), batch = the tile + a random crop of another unlabeled case (cropped and augmented as
in `data.py`; its output is discarded). The volume is tiled with patch-size tiles on a grid with a random
offset (starts clamped into the volume); every voxel takes the output of exactly one tile (no averaging).
- Pass 0: LA tiles get the training rot90 / flip, outputs are turned back. It gives the pseudo-label
  P = argmax of the full-resolution head, its confidence, entropy and the three lower DS heads.
- Passes 1–8 (`tta_train`): the 8 flips over the 3 axes, each with a new grid offset, partner and Dropout mask.
- Passes 9–16 (`flip_eval`): eval mode (running BN statistics, no Dropout), the grid of pass 0, the 8 flips.
- Reference `sliding`: `evaluate.predict_volume` (the test protocol); taxonomy and Dice contributions only.

**Distances.** Signed Chebyshev distance d to the GT boundary and d̃ to the pseudo boundary (V1's convention:
background +k, foreground −k, not clipped). Pseudo layers |d̃| = 1, 2, 3, 4–6, > 6 (V1's D2 band |d̃| ≤ 3 is
about |d̃| ≤ 6 at full resolution).

**Error taxonomy** (full resolution, per case; errors E = {P ≠ GT}, FP = P∖GT, FN = GT∖P):
- `block`: deep errors, max(|d|, |d̃|) > 3, i.e. more than 3 voxels from both boundaries (a missed or added
  region, a missed thin part, a distant island), plus the errors of the same type (FP or FN) connected to them
  within 3 steps (26-connectivity; the region's own rim). Not whole connected components: the thin error
  shell along the surface connects nearly every error of a case. A uniform shift of ≤ 3 voxels is `shift`, of
  ≥ 6 voxels `block`, 4–5 voxels in between (tests/verify_anatomy.py).
- The other (boundary) errors, each in the 17³ window centred on it: with ≥ 20 boundary errors in the window,
  local bias b = (FP − FN) / (FP + FN); |b| ≥ 0.4 (≥ 70 % one sign) → `shift`, otherwise `jitter`; < 20 →
  `jitter`. (Centred windows, not a fixed grid: a grid cell cut obliquely by the surface holds the FP shell
  without the FN shell next to it and would call random jitter a shift.)
- Direction consistency: share of the shift voxels (pooled over cases) with the majority sign.
- Contribution: ΔDice_k = Dice(P with the category-k errors set to GT) − Dice(P), mean over cases;
  share_k = ΔDice_k / Σ_j ΔDice_j. ΔHD95_k likewise (reported, not used). Reachable share: ΔDice of the errors
  with |d̃| ≤ 6 / ΔDice of all errors. Oracle thresholds (upper bounds of calibration): Dice gain of the best
  single foreground threshold per run (thresholds 0.05, 0.10, …, 0.95) and of the best per case, vs 0.5.
- Per pseudo layer: error rate, share of the errors, FP share, composition by category.

**Separability.** Confident voxels (pass-0 confidence > 0.95) in each layer |d̃| = 1, 2, 3; label: error.
All from the logit difference z = logit_fg − logit_bg (float32 probabilities saturate on confident voxels).
Scores: `entropy` (pass 0; with two classes entropy is a decreasing function of |z|, so it ranks voxels exactly
as the confidence does and is computed as −|z|); `ds_disagreement` = mean over the three lower heads (z
trilinear to full resolution) of |q_head − q_full|, q = probability of the class other than the pseudo class;
`flip_eval` and `tta_train` = mean q over their passes (= 1 − mean probability of the pseudo class). AUROC and AP pooled over the run's voxels of the layer, and the median of
per-case AUROCs (cases with ≥ 10 errors and ≥ 10 correct voxels in the layer). A = error-count-weighted mean
of the three layer AUROCs. The unstratified AUROC over layers 1–3 is reported to show the confound with
boundary distance, not used.

**Decision rules (fixed before the run).** Per dataset, quantities averaged over the 3 seeds, mode `train`.
1. Dominant error type: the category with share_k > 0.5 (by ΔDice); otherwise mixed (top two reported).
2. `shift` dominant and direction consistency ≥ 0.7 → logit bias / threshold calibration (oracle gain = its
   upper bound); `shift` dominant and consistency < 0.7 → shifts vary per case or region, global calibration
   does not apply, discuss; `jitter` dominant → boundary tolerance or distance-softened pseudo-labels; `block`
   dominant → new information is needed (e.g. shape / position priors), U→L reaches at most the reachable
   share; mixed → discuss.
3. A signal separates confident errors when A ≥ 0.70 and, for signals other than entropy, A ≥ A_entropy + 0.05.
   Some signal separates → reliability weighting of pseudo-labels with it is viable; none → confident errors
   are not detectable with these signals.
4. Together with V1: go to V2 (U→L, scheme as V1 decides) only if V1 says go and the non-block share is ≥ 0.5
   on both datasets; V1 says go but the block share is > 0.5 on a dataset → discuss; V1 says stop → discuss.

```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/anatomy.py --run ../../Baseline/MT_DS_DICE/Results_baseline_pl/seed_42/LA_8l \
    --output Results_anatomy/seed_42/LA_8l
python scripts/report_anatomy.py --root Results_anatomy --output summary/anatomy.md
```

## Tests

`tests/verify_core.py` (backbone vs DuCiSC's VNet, initialization, gradients, EMA),
`tests/verify_pl.py` (L_pl vs DuCiSC's `get_loss`, diagnostics), `tests/verify_ducisc_test.py`
(evaluation vs DuCiSC's `test_patch.py` with fixed weights), `tests/compare_csv.py`,
`tests/check_ce_nondeterminism.py`, `tests/bench_loader.py`. Verbatim DuCiSC reference code is in
`tests/ducisc_reference.py`. Results: `tests/outputs/mt_ds_pl/`; see `CHANGES.md`. V1 probe:
`tests/verify_v1.py` (buckets vs scipy, hand-computed statistics, probe side effects) and
`tests/zero_interference_v1.py`; results in `tests/outputs/v1/`.

Earlier variants (MSE consistency, LossNorm, Plan A) are kept at tags `archive/plan-a-mse` and
`archive/step1-step2`.
