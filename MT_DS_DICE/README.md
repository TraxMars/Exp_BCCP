# MT_DS_DICE

Mean Teacher + four-scale deep supervision on DuCiSC's batch-norm VNet (parameter names and
initialization identical to the original). One code base covers both former baselines:

```text
L_sup = mean_4_scales((CE + foreground Dice) / 2)                 # nearest-neighbour GT per scale
L_mt  = mean_4_scales(MSE(softmax(student), stopgrad(softmax(teacher))))
L     = L_sup                      (iteration < warmup, 2000)
L     = L_sup + 0.3 * L_mt         (afterwards)                    # MT_DS (default)
L     = L_sup/sg(L_sup) + 0.3 * L_mt/sg(L_mt)                      # --loss_norm (MT_DS_LossNorm)

# --consistency_type pseudo_ce_dice (MT_DS_PL): L_mt is replaced by
L_pl  = mean_4_scales((CE + foreground Dice) / 2)  to  argmax(teacher) of the same scale
```

`--diag_interval 50` (default) writes logging-only `diagnostics.csv`: gradient norms / cosine of
`L_sup` and `0.3·L_cons` on two decoder/encoder convolutions, and oracle boundary statistics of the
unlabeled images against their hidden GT (never part of the objective).

Defaults: 15000 iterations, constant lr 0.01 (SGD, momentum 0.9, wd 1e-4), EMA 0.99, batch 2+2,
Teacher in train mode (independent Dropout3d p=0.5), validation every 200 iterations from 800,
final test of the best Student. LA / Pancreas select on test.list; BraTS19 on val.list.

```bash
PY=~/.conda/envs/SSL29/bin/python
$PY tests/verify_core.py                                   # backbone / init / loss checks
CUDA_VISIBLE_DEVICES=0 $PY code/train.py --dataset LA --output_dir Results_tmp/LA_8l
$PY scripts/launch.py --datasets LA Pancreas --seeds 42 500 1337 --output_root Results_x [-- train args]
$PY scripts/report.py --root Results_x --output summary/x.md
```

Each run writes `config.json` (arguments, git commit, labeled / unlabeled case IDs),
`training_metrics.csv`, `validation_metrics.jsonl`, `best_metrics.json`, `test_metrics.json`,
`diagnostics.csv`, `log.txt`, `best_model.pth` and `last_checkpoint.pth` (`--resume` restores its stored configuration
and RNG states; DataLoader workers are not restored bit-exactly).

The Plan A (BCCP) code — CBC observers, channel perturbation, the separate MT_DS_LossNorm copy and
the step reports — is preserved at tag `archive/step1-step2`; its results remain in `../../Exp_BCCP`.
See `CHANGES.md` for what changed and how it was verified.
