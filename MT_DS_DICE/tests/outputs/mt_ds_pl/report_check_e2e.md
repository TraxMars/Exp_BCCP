# Multi-seed summary: `<scratch>/e2e_LA`

Dice / JC in %, HD95 / ASD in voxels. Test metrics follow DuCiSC (an empty prediction scores 0 and
counts in the mean); nms1 = largest connected component. Validation medians use 10000 ≤ iteration ≤
15000; worst-case Dice is the lowest per-case test Dice; sd is the sample standard deviation across seeds.

## LA_8l

| seed | iters | best val Dice | @iter | last-5 val Dice | val Dice med 10k-15k | val HD95 med | val ASD med | test Dice | test JC | test HD95 | test ASD | worst-case Dice | empty pred. | nms1 Dice | nms1 HD95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 200 | 63.47 | 200 | 60.61 | — | — | — | 63.47 | 50.20 | 25.33 | 8.07 | 0.00 | 0 | 63.21 | 30.82 |
| mean ± sd | | 63.47 (n=1) | 200 (n=1) | 60.61 (n=1) | — | — | — | 63.47 (n=1) | 50.20 (n=1) | 25.33 (n=1) | 8.07 (n=1) | 0.00 (n=1) | 0 (n=1) | 63.21 (n=1) | 30.82 (n=1) |

### LA_8l: diagnostics, median per phase

| diagnostic | 0k–2k | 2k–6k | 6k–10k | 10k–15k |
|---|---|---|---|---|
| gnorm_sup_b7 | 0.0563 | — | — | — |
| gnorm_pl_b7 | 0.0375 | — | — | — |
| cos_b7 | -0.3465 | — | — | — |
| cos_pl_true_b7 | -0.3928 | — | — | — |
| cos_sup_true_b7 | 0.6486 | — | — | — |
| frac_ch_pl_true_neg_b7 | 0.7969 | — | — | — |
| frac_ch_pl_sup_neg_b7 | 0.8438 | — | — | — |
| ch_rank_corr_b7 | 0.5939 | — | — | — |
| hit_rate_b7 | 0.9143 | — | — | — |
| gnorm_sup_b3 | 0.2605 | — | — | — |
| gnorm_pl_b3 | 0.1058 | — | — | — |
| cos_b3 | 0.2141 | — | — | — |
| cos_pl_true_b3 | -0.2759 | — | — | — |
| cos_sup_true_b3 | 0.0594 | — | — | — |
| frac_ch_pl_true_neg_b3 | 0.8359 | — | — | — |
| frac_ch_pl_sup_neg_b3 | 0.3906 | — | — | — |
| ch_rank_corr_b3 | 0.0216 | — | — | — |
| hit_rate_b3 | 0.9091 | — | — | — |
| ce_pl_in | 0.5219 | — | — | — |
| ce_gt_in | 0.8069 | — | — | — |
| ce_pl_out | 0.3802 | — | — | — |
| ce_gt_out | 0.2692 | — | — | — |
| band_iou | 0.1112 | — | — | — |
| teacher_conf_gt_band | 0.6592 | — | — | — |
| teacher_conf_pl_band | 0.6657 | — | — | — |
| gt_band_fraction | 0.0950 | — | — | — |
| teacher_err_gt_band | 0.3997 | — | — | — |
| conf_wrong_095 | 0.0059 | — | — | — |
| conf_right_095 | 0.0106 | — | — | — |
| ce_lab_in | 0.7033 | — | — | — |
| ce_lab_out | 0.2702 | — | — | — |

4 diagnostic rows from 1 seed(s); phases are (start, end] iterations.
