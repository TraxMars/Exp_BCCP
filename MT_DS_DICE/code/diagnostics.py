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
    Teacher confidence is reported in both the GT band and the Teacher's own (pseudo) band.
    """
    pseudo = teacher_logits.argmax(dim=1)
    confidence = teacher_logits.softmax(dim=1).amax(dim=1)
    ce_pseudo = F.cross_entropy(student_logits, pseudo, reduction="none")
    ce_gt = F.cross_entropy(student_logits, hidden_labels, reduction="none")
    gt_band, pseudo_band = boundary_band(hidden_labels, radius), boundary_band(pseudo, radius)
    union = (gt_band | pseudo_band).sum().item()
    return dict(ce_pl_in=masked_mean(ce_pseudo, gt_band), ce_gt_in=masked_mean(ce_gt, gt_band),
                ce_pl_out=masked_mean(ce_pseudo, ~gt_band), ce_gt_out=masked_mean(ce_gt, ~gt_band),
                band_iou=(gt_band & pseudo_band).sum().item() / union if union else math.nan,
                teacher_conf_gt_band=masked_mean(confidence, gt_band),
                teacher_conf_pl_band=masked_mean(confidence, pseudo_band),
                gt_band_fraction=gt_band.float().mean().item())


def gradient_diagnostics(loss_sup, loss_cons, parameters):
    """Gradient norms of L_sup and of the weighted consistency term on each named parameter, and their cosine."""
    names, tensors = list(parameters), list(parameters.values())
    sup = torch.autograd.grad(loss_sup, tensors, retain_graph=True)
    cons = torch.autograd.grad(loss_cons, tensors, retain_graph=True)
    row = {}
    for name, left, right in zip(names, sup, cons):
        row[f"gnorm_sup_{name}"] = left.norm().item()
        row[f"gnorm_cons_{name}"] = right.norm().item()
        row[f"cos_{name}"] = F.cosine_similarity(left.flatten(), right.flatten(), dim=0, eps=1e-12).item()
    return row
