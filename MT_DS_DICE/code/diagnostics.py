"""Logging-only diagnostics; nothing here enters the optimized objective.

Boundary statistics run under no_grad on the full-resolution outputs the training step already
produced (no extra forward). Gradient probes use torch.autograd.grad on the retained graph, which
returns gradients without writing any parameter's .grad.
"""
import math

import torch
import torch.nn.functional as F


def boundary_band(mask, radius=2):
    """Cubic dilation minus erosion; patch borders are not treated as object borders."""
    mask = mask[:, None].float()
    size = 2 * radius + 1
    dilated = F.max_pool3d(mask, size, stride=1, padding=radius)
    eroded = -F.max_pool3d(-mask, size, stride=1, padding=radius)
    return (dilated - eroded)[:, 0].bool()


def masked_mean(values, mask):
    return values[mask].float().mean().item() if mask.any() else math.nan


@torch.no_grad()
def boundary_diagnostics(student_logits, teacher_logits, hidden_labels, radius=2):
    """Oracle statistics of the unlabeled batch on the full-resolution head.

    "in" / "out" are inside / outside the GT band, so both CEs are measured on the same voxels;
    Teacher confidence is reported in both the GT band and the Teacher's own (pseudo) band. The
    error and confidence-split statistics are taken inside the GT band.
    """
    pseudo = teacher_logits.argmax(dim=1)
    confidence = teacher_logits.softmax(dim=1).amax(dim=1)
    ce_pseudo = F.cross_entropy(student_logits, pseudo, reduction="none")
    ce_gt = F.cross_entropy(student_logits, hidden_labels, reduction="none")
    gt_band, pseudo_band = boundary_band(hidden_labels, radius), boundary_band(pseudo, radius)
    union = (gt_band | pseudo_band).sum().item()
    wrong = pseudo != hidden_labels
    return dict(ce_pl_in=masked_mean(ce_pseudo, gt_band), ce_gt_in=masked_mean(ce_gt, gt_band),
                ce_pl_out=masked_mean(ce_pseudo, ~gt_band), ce_gt_out=masked_mean(ce_gt, ~gt_band),
                band_iou=(gt_band & pseudo_band).sum().item() / union if union else math.nan,
                teacher_conf_gt_band=masked_mean(confidence, gt_band),
                teacher_conf_pl_band=masked_mean(confidence, pseudo_band),
                gt_band_fraction=gt_band.float().mean().item(),
                teacher_err_gt_band=masked_mean(wrong.float(), gt_band),
                conf_wrong_095=masked_mean((confidence > 0.95).float(), wrong & gt_band),
                conf_right_095=masked_mean((confidence > 0.95).float(), ~wrong & gt_band))


@torch.no_grad()
def labeled_band_ce(student_logits, labels, radius=2):
    """Student CE to the labeled GT inside / outside its boundary band (full-resolution head)."""
    band = boundary_band(labels, radius)
    ce = F.cross_entropy(student_logits, labels, reduction="none")
    return dict(ce_lab_in=masked_mean(ce, band), ce_lab_out=masked_mean(ce, ~band))


def ranks(values):
    return values.argsort().argsort().float()


def spearman(left, right):
    left, right = ranks(left), ranks(right)
    return F.cosine_similarity(left - left.mean(), right - right.mean(), dim=0, eps=1e-12).item()


def gradient_diagnostics(loss_sup, loss_pl, parameters, loss_true=None):
    """Gradient norms of L_sup and of the weighted pseudo-label term on each named parameter, and their cosine.

    With loss_true (the same deep CE + Dice on the unlabeled outputs against the hidden GT; an oracle,
    differentiated here only), also: whole-layer cosines of g_pl and g_sup with g_true, and per output
    channel c the cosines cos(g_pl[c], g_sup[c]) and cos(g_pl[c], g_true[c]) summarized as the share of
    negative channels, their Spearman correlation, and how often a negative pl-sup channel is also a
    negative pl-true channel (hit_rate; NaN when no channel has cos(pl, sup) < 0).
    """
    names, tensors = list(parameters), list(parameters.values())
    sup = torch.autograd.grad(loss_sup, tensors, retain_graph=True)
    pl = torch.autograd.grad(loss_pl, tensors, retain_graph=True)
    true = torch.autograd.grad(loss_true, tensors, retain_graph=True) if loss_true is not None else [None] * len(names)
    row = {}
    for name, left, right, oracle in zip(names, sup, pl, true):
        row[f"gnorm_sup_{name}"] = left.norm().item()
        row[f"gnorm_pl_{name}"] = right.norm().item()
        row[f"cos_{name}"] = F.cosine_similarity(left.flatten(), right.flatten(), dim=0, eps=1e-12).item()
        if oracle is None:
            continue
        row[f"cos_pl_true_{name}"] = F.cosine_similarity(right.flatten(), oracle.flatten(), dim=0, eps=1e-12).item()
        row[f"cos_sup_true_{name}"] = F.cosine_similarity(left.flatten(), oracle.flatten(), dim=0, eps=1e-12).item()
        pl_sup = F.cosine_similarity(right.flatten(1), left.flatten(1), dim=1, eps=1e-12)
        pl_true = F.cosine_similarity(right.flatten(1), oracle.flatten(1), dim=1, eps=1e-12)
        row[f"frac_ch_pl_true_neg_{name}"] = (pl_true < 0).float().mean().item()
        row[f"frac_ch_pl_sup_neg_{name}"] = (pl_sup < 0).float().mean().item()
        row[f"ch_rank_corr_{name}"] = spearman(pl_sup, pl_true)
        flagged = pl_sup < 0
        row[f"hit_rate_{name}"] = (pl_true[flagged] < 0).float().mean().item() if flagged.any() else math.nan
    return row
