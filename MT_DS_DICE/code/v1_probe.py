"""V1 probe (logging only): boundary-distance-bucketed labeled ← unlabeled grouped channel cross-attention at D2.

D2 is VNetDS's x7_up (32 channels, 1/2 resolution). Queries are labeled tokens, keys / values unlabeled
tokens, all taken from the training step's own Student forward. Nothing here enters the objective:
everything runs under no_grad except the gradient-level probe, which calls torch.autograd.grad only
(no parameter .grad is written); sampling uses its own torch.Generator; the two extra decodes run in
torch.random.fork_rng with the decoder's BatchNorm buffers restored afterwards. Definitions: README.md, "V1 probe".
"""
import contextlib
import math
import time

import torch
import torch.nn.functional as F
from torch import nn

from diagnostics import boundary_band


BUCKETS = (-3, -2, -1, 1, 2, 3)
CONSTRAINTS = ("signed", "unsigned", "band", "none")
GROUPS = (1, 4)
TAUS = (0.05, 0.1, 0.2)
FOCUS_GROUPS, FOCUS_TAU = 4, 0.1   # gradient-level reverse correction and output sensitivity
QUERIES, KEYS, KEYS_OUTSIDE = 512, 1024, 2048   # per GT bucket / per pseudo bucket / outside the pseudo band
CONFIDENT = 0.95
MIN_MASS = 1e-3
CHUNK = 1024   # queries per attention block
FIELDS = ["iteration", "family", "constraint", "G", "tau", "subset", "metric", "value"]
NA = "NA"


def nearest(labels, shape):
    """Nearest-neighbour downsampling, as in objectives.supervised_loss."""
    return F.interpolate(labels[:, None].float(), size=tuple(shape), mode="nearest")[:, 0].long()


def distance_buckets(mask, depth=3):
    """+k for background voxels first covered by the k-th 3×3×3 max-pool of the foreground, −k for foreground
    voxels covered by that of the background, 0 beyond depth (outside the band).

    That is the Chebyshev distance to the other side, clipped at depth. max_pool3d pads with −inf, so the
    patch border is never taken for an object border.
    """
    buckets = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)
    foreground = mask.bool()[:, None].float()
    for covered, sign in ((foreground, 1), (1 - foreground, -1)):
        for k in range(1, depth + 1):
            grown = F.max_pool3d(covered, 3, stride=1, padding=1)
            buckets[(grown > covered)[:, 0]] = sign * k
            covered = grown
    return buckets


def tokens(features):
    """[B, C, D, H, W] → [B·D·H·W, C], in the order of labels.flatten()."""
    return features.movedim(1, -1).reshape(-1, features.shape[1])


def sample(mask, limit, generator):
    index = mask.nonzero().squeeze(1)
    if len(index) > limit:
        index = index[torch.randperm(len(index), generator=generator)[:limit].to(index.device)]
    return index


def allowed_keys(constraint, query_buckets, key_buckets):
    if constraint == "signed":
        return query_buckets[:, None] == key_buckets[None]
    if constraint == "unsigned":
        return query_buckets.abs()[:, None] == key_buckets.abs()[None]
    if constraint == "band":
        return (key_buckets != 0)[None].expand(len(query_buckets), -1)
    if constraint == "none":
        return torch.ones(len(query_buckets), len(key_buckets), dtype=torch.bool, device=key_buckets.device)
    raise ValueError(constraint)


def attention_weights(queries, keys, allowed, groups, tau):
    """[G, Nq, Nk]: per channel group softmax_j(cos(f^g, u^g_j) / tau) over the allowed keys.

    tau=None gives uniform weights over the allowed keys (one group). Rows without an allowed key are zero.
    """
    if tau is None:
        return (allowed.float() / allowed.sum(1, keepdim=True).clamp_min(1))[None]
    q = F.normalize(queries.unflatten(1, (groups, -1)).transpose(0, 1), dim=-1)
    k = F.normalize(keys.unflatten(1, (groups, -1)).transpose(0, 1), dim=-1)
    logits = (q @ k.transpose(1, 2) / tau).masked_fill(~allowed, -math.inf)
    return logits.softmax(-1).masked_fill(~allowed.any(1)[None, :, None], 0.0)


def mix(weights, values):
    """f'^g = Σ_j w^g_j u^g_j, with the groups concatenated back to [Nq, C]."""
    grouped = values.unflatten(1, (weights.shape[0], -1)).transpose(0, 1)
    return (weights @ grouped).transpose(0, 1).flatten(1)


def query_statistics(weights, queries, values, query_side, key_side):
    """Per query: contamination (attention on keys whose true side differs from the query's GT side),
    swap magnitude 1 − cos(f, f'), effective keys exp(entropy); group-dependent terms averaged over groups."""
    on_foreground, on_background = weights @ key_side.float(), weights @ (1 - key_side).float()
    contamination = torch.where(query_side.bool(), on_background, on_foreground).mean(0)
    swap = 1 - F.cosine_similarity(queries, mix(weights, values), dim=1, eps=1e-12)
    effective = (-torch.special.xlogy(weights, weights).sum(-1)).exp().mean(0)
    return contamination, swap, effective


def attention_pass(queries, keys, query_buckets, query_side, key_buckets, key_side, constraint, groups, tau):
    """Query statistics, and per key the received mass m_j = Σ_x w_j(x) and its foreground part
    Σ_x w_j(x)·GT side(x), with w averaged over groups; in query blocks of CHUNK."""
    stats, mass, foreground = [], keys.new_zeros(len(keys)), keys.new_zeros(len(keys))
    for chunk in torch.arange(len(queries), device=queries.device).split(CHUNK):
        allowed = allowed_keys(constraint, query_buckets[chunk], key_buckets)
        weights = attention_weights(queries[chunk], keys, allowed, groups, tau)
        stats.append((*query_statistics(weights, queries[chunk], keys, query_side[chunk], key_side), allowed.any(1)))
        averaged = weights.mean(0)
        mass += averaged.sum(0)
        foreground += query_side[chunk].float() @ averaged
    contamination, swap, effective, valid = (torch.cat(column) for column in zip(*stats))
    return dict(contamination=contamination, swap=swap, effective=effective, valid=valid,
                mass=mass, foreground=foreground)


def ratio(numerator, denominator):
    numerator, denominator = float(numerator), float(denominator)
    return numerator / denominator if denominator > 0 else math.nan


def reverse_label_statistics(mass, foreground, key_true, key_pseudo, key_confidence):
    """Label-level reverse correction. r_j = foreground / mass is the GT side a key receives; its majority
    side (r_j > 0.5 foreground, < 0.5 background, a tie or zero mass none) is compared with the key's true
    side. Correction: wrong-pseudo-side keys whose majority is the true side; harm: right-pseudo-side keys
    whose majority is the other side; rates are mass-weighted within the set, masses are shares of the
    total attention mass. Again on keys with Teacher confidence > 0.95."""
    received = foreground / mass
    majority = torch.full_like(key_true, -1)
    majority[received > 0.5], majority[received < 0.5] = 1, 0
    corrected = majority == key_true
    harmed = (majority >= 0) & (majority != key_true)
    wrong = key_pseudo != key_true
    total = mass.sum()
    result = {}
    for subset, keep in (("all", torch.ones_like(wrong)), ("conf095", key_confidence > CONFIDENT)):
        wrong_mass, right_mass = mass[wrong & keep].sum(), mass[~wrong & keep].sum()
        correction, harm = mass[wrong & keep & corrected].sum(), mass[~wrong & keep & harmed].sum()
        result[subset] = dict(correction_rate=ratio(correction, wrong_mass), harm_rate=ratio(harm, right_mass),
                              correction_mass=ratio(correction, total), harm_mass=ratio(harm, total),
                              wrong_mass_frac=ratio(wrong_mass, total))
    return result


def median(values):
    return values.double().quantile(0.5).item() if values.numel() else math.nan


def mean(values):
    return values.double().mean().item() if values.numel() else math.nan


BATCHNORM_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


@contextlib.contextmanager
def batchnorm_restored(*modules):
    """Run train-mode BatchNorm in modules on copies of running_mean / running_var / num_batches_tracked and
    put the original tensors back on exit.

    Copying values back would not do: the training graph saved the running buffers for its backward, and an
    in-place update bumps their version counter, which makes loss.backward() fail. The originals are never written.
    """
    layers = [layer for module in modules for layer in module.modules()
              if isinstance(layer, nn.modules.batchnorm._BatchNorm)]
    saved = [(layer, [getattr(layer, name) for name in BATCHNORM_BUFFERS]) for layer in layers]
    for layer, buffers in saved:
        for name, buffer in zip(BATCHNORM_BUFFERS, buffers):
            setattr(layer, name, buffer.clone())
    try:
        yield
    finally:
        for layer, buffers in saved:
            for name, buffer in zip(BATCHNORM_BUFFERS, buffers):
                setattr(layer, name, buffer)


@torch.no_grad()
def output_sensitivity(model, x7_up, x1, labels, buckets, keys, key_buckets, constraint):
    """Replace every labeled D2 band token by its attention output (G=4, tau=0.1) and decode twice from x7_up."""
    flat, band = tokens(x7_up), (buckets.flatten() != 0).nonzero().squeeze(1)
    swapped, replaced = flat.clone(), torch.zeros(len(flat), dtype=torch.bool, device=flat.device)
    for chunk in band.split(CHUNK):
        allowed = allowed_keys(constraint, buckets.flatten()[chunk], key_buckets)
        valid = allowed.any(1)
        swapped[chunk[valid]] = mix(attention_weights(flat[chunk], keys, allowed, FOCUS_GROUPS, FOCUS_TAU), keys)[valid]
        replaced[chunk[valid]] = True
    swapped = swapped.view(*buckets.shape, -1).movedim(-1, 1)
    devices = [x7_up.device] if x7_up.is_cuda else []
    with torch.random.fork_rng(devices=devices), \
            batchnorm_restored(model.block_eight, model.block_eight_up, model.block_nine):
        before = model.decode_from_x7(x7_up, x1, dropout=False)
        after = model.decode_from_x7(swapped, x1, dropout=False)
    band_full = boundary_band(labels, 2)
    replaced_full = nearest(replaced.view(buckets.shape), labels.shape[1:]).bool()
    change = (after.softmax(1)[:, 1] - before.softmax(1)[:, 1]).abs()
    ce_change = F.cross_entropy(after, labels, reduction="none") - F.cross_entropy(before, labels, reduction="none")
    head_change = (model.out_conv_2(swapped).softmax(1)[:, 1] - model.out_conv_2(x7_up).softmax(1)[:, 1]).abs()
    dp_d2 = mean(head_change[replaced.view(buckets.shape)])
    dp_final, dp_final_replaced = mean(change[band_full]), mean(change[replaced_full])
    return dict(dp_final_band=dp_final, dce_final_band=mean(ce_change[band_full]),
                flip_final_band=mean((after.argmax(1) != before.argmax(1))[band_full].float()),
                dp_d2=dp_d2, bypass_ratio=ratio(dp_final, dp_d2),
                dp_final_replaced=dp_final_replaced, bypass_ratio_replaced=ratio(dp_final_replaced, dp_d2),
                replaced_fraction=ratio(replaced.sum(), len(band)))


@torch.no_grad()
def bucket_profiles(labeled, unlabeled, labeled_buckets, pseudo_buckets, true_buckets):
    """Mean D2 feature per bucket: unlabeled by pseudo vs by true bucket, labeled (GT) vs unlabeled (true)."""
    def means(flat, buckets):
        return {b: flat[buckets == b].double().mean(0) if (buckets == b).any() else None for b in BUCKETS}

    def relative(left, right):
        return ((left - right).norm() / ((left.norm() + right.norm()) / 2)).item()

    lab = means(labeled, labeled_buckets.flatten())
    by_pseudo, by_true = means(unlabeled, pseudo_buckets.flatten()), means(unlabeled, true_buckets.flatten())
    rows, differences = [], []
    for b in BUCKETS:
        subset = f"d{b:+d}"
        if by_pseudo[b] is not None and by_true[b] is not None:
            rows.append((subset, "cos_unl_pseudo_vs_true", F.cosine_similarity(by_pseudo[b], by_true[b], dim=0).item()))
        if lab[b] is not None and by_true[b] is not None:
            rows.append((subset, "cos_lab_vs_unl_true", F.cosine_similarity(lab[b], by_true[b], dim=0).item()))
            differences.append(relative(lab[b], by_true[b]))
            rows.append((subset, "reldiff_lab_vs_unl_true", differences[-1]))
    rows.append(("global", "reldiff_lab_vs_unl", relative(labeled.double().mean(0), unlabeled.double().mean(0))))
    rows.append(("bucket_mean", "reldiff_lab_vs_unl_true", sum(differences) / len(differences) if differences else math.nan))
    return rows


def v1_probe(model, iteration, labeled, unlabeled, labels, hidden, teacher_d2, loss_pl, loss_true, seed):
    """One probe step; returns long-format rows (FIELDS).

    labeled / unlabeled: dict(x1, x7_up) of the training forward; labels / hidden: full-resolution labeled GT
    and hidden unlabeled GT; teacher_d2 = teacher_outputs[2]; loss_pl is the weighted pseudo-label term
    (0.3·L_pl) and loss_true the oracle of diagnostics, both on the retained training graph.
    """
    synchronize = torch.cuda.synchronize if teacher_d2.is_cuda else (lambda: None)
    synchronize()
    start = time.monotonic()
    rows = []

    def add(family, metric, value, constraint=NA, groups=NA, tau=NA, subset=NA):
        rows.append(dict(iteration=iteration, family=family, constraint=constraint, G=groups, tau=tau,
                         subset=subset, metric=metric, value=float(value)))

    x7_lab, x7_unl = labeled["x7_up"], unlabeled["x7_up"]
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        shape = x7_lab.shape[2:]
        lab, true = nearest(labels, shape), nearest(hidden, shape)
        pseudo, confidence = teacher_d2.argmax(1), teacher_d2.softmax(1).amax(1)
        lab_b, pseudo_b, true_b = distance_buckets(lab), distance_buckets(pseudo), distance_buckets(true)
        flat_lab, flat_unl = tokens(x7_lab.detach()), tokens(x7_unl.detach())
        query_index = torch.cat([sample(lab_b.flatten() == b, QUERIES, generator) for b in BUCKETS])
        key_index = torch.cat([sample(pseudo_b.flatten() == b, KEYS, generator) for b in BUCKETS]
                              + [sample(pseudo_b.flatten() == 0, KEYS_OUTSIDE, generator)])
        queries, query_b, query_side = flat_lab[query_index], lab_b.flatten()[query_index], lab.flatten()[query_index]
        keys, key_b = flat_unl[key_index], pseudo_b.flatten()[key_index]
        key_true, key_pseudo = true.flatten()[key_index], pseudo.flatten()[key_index]
        key_confidence = confidence.flatten()[key_index]
        for b in BUCKETS:
            add("sample", "n_query", (query_b == b).sum(), subset=f"d{b:+d}")
            add("sample", "n_key", (key_b == b).sum(), subset=f"d{b:+d}")
        add("sample", "n_key", (key_b == 0).sum(), subset="outside")

        settings = [(groups, tau) for groups in GROUPS for tau in TAUS] + [(None, None)]
        for constraint in CONSTRAINTS:
            for groups, tau in settings:
                result = attention_pass(queries, keys, query_b, query_side, key_b, key_true, constraint, groups or 1, tau)
                suffix = "" if tau is not None else "_uniform"
                grid = dict(constraint=constraint, groups=groups if groups else NA, tau=tau if tau else NA)
                valid = result["valid"]
                add(f"attention{suffix}", "n_valid_queries", valid.sum(), **grid)
                for subset, keep in [("all", valid)] + [(f"d{d}", valid & (query_b.abs() == d)) for d in (1, 2, 3)]:
                    for metric, key in (("contamination", "contamination"), ("swap", "swap"), ("eff_keys", "effective")):
                        add(f"attention{suffix}", metric, mean(result[key][keep]), subset=subset, **grid)
                reverse = reverse_label_statistics(result["mass"], result["foreground"], key_true, key_pseudo, key_confidence)
                for subset, values in reverse.items():
                    for metric, value in values.items():
                        add(f"reverse_label{suffix}", metric, value, subset=subset, **grid)
                if constraint == "signed" and any(values["correction_rate"] > 0 for values in reverse.values()):
                    raise RuntimeError("V1 probe: the signed constraint cannot correct a pseudo side (implementation error)")

    # Gradient level (signed / unsigned, G=4, tau=0.1): autograd.grad only, no parameter .grad.
    g_pl = tokens(torch.autograd.grad(loss_pl, x7_unl, retain_graph=True)[0])[key_index]
    g_true = tokens(torch.autograd.grad(loss_true, x7_unl, retain_graph=True)[0])[key_index]
    values = tokens(x7_unl)[key_index]   # in the training graph: not detached
    head_weight, head_bias = model.out_conv_2.weight.detach().flatten(1), model.out_conv_2.bias.detach()
    wrong, confident = key_pseudo != key_true, key_confidence > CONFIDENT
    for constraint in ("signed", "unsigned"):
        grid = dict(constraint=constraint, groups=FOCUS_GROUPS, tau=FOCUS_TAU)
        g_ul, mass, loss_ul = torch.zeros_like(values), keys.new_zeros(len(keys)), 0.0
        blocks = []
        with torch.no_grad():
            for chunk in torch.arange(len(queries), device=queries.device).split(CHUNK):
                allowed = allowed_keys(constraint, query_b[chunk], key_b)
                weights = attention_weights(queries[chunk], keys, allowed, FOCUS_GROUPS, FOCUS_TAU)
                mass += weights.mean(0).sum(0)
                blocks.append((chunk, weights, allowed.any(1)))
        count = sum(int(valid.sum()) for _, _, valid in blocks)
        for chunk, weights, valid in blocks:
            if not valid.any():
                continue
            logits = F.linear(mix(weights[:, valid], values), head_weight, head_bias)
            block_loss = F.cross_entropy(logits, query_side[chunk][valid], reduction="none").sum() / count
            g_ul += torch.autograd.grad(block_loss, values)[0]
            loss_ul += block_loss.item()
        del blocks
        add("reverse_grad", "loss_ul", loss_ul if count else math.nan, **grid)
        with torch.no_grad():
            cos_ul = F.cosine_similarity(g_ul, g_true, dim=1, eps=1e-12)
            cos_pl = F.cosine_similarity(g_pl, g_true, dim=1, eps=1e-12)
            received = mass > MIN_MASS
            for subset, keep in (("wrong", received & wrong), ("wrong_conf095", received & wrong & confident),
                                 ("right", received & ~wrong)):
                add("reverse_grad", "cos_ul_true", median(cos_ul[keep]), subset=subset, **grid)
                add("reverse_grad", "cos_pl_true", median(cos_pl[keep]), subset=subset, **grid)
                add("reverse_grad", "n_tokens", keep.sum(), subset=subset, **grid)
    del g_pl, g_true, values

    for constraint in ("signed", "unsigned"):
        sensitivity = output_sensitivity(model, x7_lab, labeled["x1"], labels, lab_b, keys, key_b, constraint)
        for metric, value in sensitivity.items():
            add("sensitivity", metric, value, constraint=constraint, groups=FOCUS_GROUPS, tau=FOCUS_TAU)
    for subset, metric, value in bucket_profiles(flat_lab, flat_unl, lab_b, pseudo_b, true_b):
        add("profile", metric, value, subset=subset)
    synchronize()
    add("meta", "probe_seconds", time.monotonic() - start)
    return rows
