"""Observation-only channel boundary contribution (CBC) at the D3/D2 decoder taps.

Probe losses are differentiated with autograd.grad on the graph the training step has
already built. They never enter the optimized objective, so parameter gradients, BN
statistics and random streams are exactly those of the baseline.

For a tap feature F and probe loss L, psi_c = mean_x dL/dF_c(x) * F_c(x). Removing
channel c changes L by about -sum_x dL/dF_c * F_c, so psi_c < 0 means the channel
currently helps the probe, and psi_c > 0 means it hurts. A channel group's score is the
sum of its members' scores. Each stream (labeled / unlabeled) has a boundary probe (bd)
and a region probe on the complementary voxels (rg).

Boundary specificity is measured across channels, not across voxels: the two probes cover
disjoint voxels, so their spatial gradients are nearly orthogonal by construction. The
residual res = bd - proj_rg(bd) over the channel vector keeps what the boundary ranking
does not share with the region ranking.
"""
from collections import deque
import csv
import json
import math

import torch
import torch.nn.functional as F


TAPS = ("d3", "d2")        # x6_up at 1/4 and x7_up at 1/2 resolution
PROBE_HEADS = (1, 2, 3)    # deep-supervision outputs at 1/4, 1/2 and full resolution
STREAMS = ("lab", "unl")
KINDS = ("bd", "rg")
TINY = 1e-30


def boundary_band(mask, radius):
    """Cubic dilation minus erosion; patch borders are not treated as object borders."""
    mask = mask[:, None].float()
    size = 2 * radius + 1
    dilated = F.max_pool3d(mask, size, stride=1, padding=radius)
    eroded = -F.max_pool3d(-mask, size, stride=1, padding=radius)
    return (dilated - eroded)[:, 0]


def pool_to(volume, shape):
    """Average a full-resolution (B, D, H, W) map onto a coarser deep-supervision grid."""
    factor = volume.shape[-1] // shape[-1]
    if factor < 1 or tuple(volume.shape[1:]) != tuple(size * factor for size in shape):
        raise ValueError(f"Cannot pool {tuple(volume.shape[1:])} onto {tuple(shape)}")
    return volume if factor == 1 else F.avg_pool3d(volume[:, None], factor)[:, 0]


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum()


def degenerate(band):
    # Without both band and non-band voxels one of the two probes is 0/0.
    return not band.any() or bool(band.all())


def labeled_probes(outputs, labels, radius):
    """Soft-target CE inside / outside the GT band, averaged over the probe heads."""
    band = boundary_band(labels, radius)
    if degenerate(band):
        return None
    boundary, region = [], []
    for index in PROBE_HEADS:
        logits = outputs[index]
        # Average-pooled targets and bands avoid nearest-neighbour boundary aliasing.
        target = pool_to(labels.float(), logits.shape[2:])
        weight = pool_to(band, logits.shape[2:])
        log_probability = logits.log_softmax(dim=1)
        ce = -(target * log_probability[:, 1] + (1 - target) * log_probability[:, 0])
        boundary.append(weighted_mean(ce, weight))
        region.append(weighted_mean(ce, 1 - weight))
    stats = dict(band_fraction=band.mean().item())
    return torch.stack(boundary).mean(), torch.stack(region).mean(), stats


def unlabeled_probes(outputs, teacher_outputs, radius):
    """Confidence-weighted student/teacher MSE inside / outside the teacher's band."""
    teacher_probability = [output.detach().softmax(dim=1) for output in teacher_outputs]
    band = boundary_band(teacher_probability[-1].argmax(dim=1), radius)
    if degenerate(band):
        return None
    boundary, region = [], []
    for index in PROBE_HEADS:
        student = outputs[index].softmax(dim=1)
        teacher = teacher_probability[index]
        confidence = teacher.max(dim=1).values
        weight = pool_to(band, student.shape[2:])
        error = (student - teacher).pow(2).mean(dim=1)
        boundary.append(weighted_mean(error, weight * confidence))
        region.append(weighted_mean(error, (1 - weight) * confidence))
    stats = dict(band_fraction=band.mean().item(),
                 confidence=teacher_probability[-1].max(dim=1).values.mean().item())
    return torch.stack(boundary).mean(), torch.stack(region).mean(), stats


def contributions(loss_boundary, loss_region, taps):
    features = [taps[name] for name in TAPS]
    result = {name: {} for name in TAPS}
    for kind, loss in (("bd", loss_boundary), ("rg", loss_region)):
        grads = torch.autograd.grad(loss, features, retain_graph=True)
        for name, feature, grad in zip(TAPS, features, grads):
            # Double precision keeps the products of tiny probe gradients away from underflow.
            result[name][kind] = (grad.double() * feature.detach().double()).mean(dim=(0, 2, 3, 4))
    return result


def residual(boundary, region):
    """Boundary contributions with their least-squares region component removed."""
    if boundary is None or region is None:
        return None
    return boundary - (boundary @ region) / (region @ region).clamp_min(TINY) * region


def cosine(left, right):
    if left is None or right is None:
        return math.nan
    return float(left @ right / (left.norm() * right.norm()).clamp_min(TINY))


def spearman(left, right):
    if left is None or right is None:
        return math.nan
    ranks = [vector.argsort().argsort().double() for vector in (left, right)]
    ranks = [rank - rank.mean() for rank in ranks]
    return float((ranks[0] * ranks[1]).sum() / (ranks[0].norm() * ranks[1].norm()).clamp_min(TINY))


def top_jaccard(left, right, fraction):
    """Overlap of the most helpful (most negative) channels."""
    if left is None or right is None:
        return math.nan
    k = max(1, round(fraction * left.numel()))
    first, second = set((-left).topk(k).indices.tolist()), set((-right).topk(k).indices.tolist())
    return len(first & second) / len(first | second)


class CBCTracker:
    """EMA of L1-normalized CBC vectors plus windowed diagnostics.

    Each observation is L1-normalized before entering the EMA, so iterations contribute
    equally regardless of the probe-loss scale. The EMA momentum follows CHAP's cosine
    schedule from 0.5 to 0.99 over training.
    """

    def __init__(self, max_iterations, band_radius=2, top_fraction=0.25,
                 momentum_low=0.5, momentum_high=0.99, lag_logs=10, dump_interval=500):
        self.max_iterations, self.band_radius = max_iterations, band_radius
        self.top_fraction, self.dump_interval = top_fraction, dump_interval
        self.momentum_low, self.momentum_high = momentum_low, momentum_high
        self.ema, self.previous, self.window = {}, {}, {}
        self.snapshots = deque(maxlen=lag_logs + 1)

    def beta(self, iteration):
        progress = min(iteration / self.max_iterations, 1.0)
        return self.momentum_low + 0.5 * (self.momentum_high - self.momentum_low) * (
            1 - math.cos(math.pi * progress))

    def _accumulate(self, key, value):
        total, count = self.window.get(key, (0.0, 0))
        self.window[key] = (total + value, count + 1)

    @torch.no_grad()
    def _update(self, iteration, stream, tap, values):
        beta = self.beta(iteration)
        vectors = {kind: values[kind].cpu() for kind in KINDS}
        vectors = {kind: vector / vector.abs().sum().clamp_min(TINY)
                   for kind, vector in vectors.items()}
        self._accumulate(f"{tap}/{stream}_raw_cos_bd_rg", cosine(vectors["bd"], vectors["rg"]))
        key = f"{stream}/{tap}/bd"
        if key in self.previous:
            # Raw batch-to-batch agreement, before any temporal smoothing.
            self._accumulate(f"{tap}/{stream}_raw_rho", spearman(vectors["bd"], self.previous[key]))
            self._accumulate(f"{tap}/{stream}_raw_jaccard",
                             top_jaccard(vectors["bd"], self.previous[key], self.top_fraction))
        self.previous[key] = vectors["bd"]
        for kind, vector in vectors.items():
            key = f"{stream}/{tap}/{kind}"
            self.ema[key] = vector.clone() if key not in self.ema else (
                beta * self.ema[key] + (1 - beta) * vector)

    def observe(self, iteration, labeled_outputs, labels, labeled_taps,
                unlabeled_outputs, teacher_outputs, unlabeled_taps):
        streams = dict(
            lab=(labeled_probes(labeled_outputs, labels, self.band_radius), labeled_taps),
            unl=(unlabeled_probes(unlabeled_outputs, teacher_outputs, self.band_radius), unlabeled_taps))
        for stream, (probe, taps) in streams.items():
            if probe is not None and not torch.isfinite(torch.stack(probe[:2])).all():
                probe = None
            self._accumulate(f"{stream}_skipped", float(probe is None))
            if probe is None:
                continue
            loss_boundary, loss_region, stats = probe
            self._accumulate(f"{stream}_loss_bd", loss_boundary.item())
            self._accumulate(f"{stream}_loss_rg", loss_region.item())
            for key, value in stats.items():
                self._accumulate(f"{stream}_{key}", value)
            for tap, values in contributions(loss_boundary, loss_region, taps).items():
                self._update(iteration, stream, tap, values)

    def rows(self, iteration):
        ema = {key: value.clone() for key, value in self.ema.items()}
        lagged = self.snapshots[0] if len(self.snapshots) == self.snapshots.maxlen else {}
        self.snapshots.append(ema)
        window = {key: total / count for key, (total, count) in self.window.items()}
        self.window = {}
        rows = []
        for tap in TAPS:
            row = dict(iteration=iteration, tap=tap, beta=self.beta(iteration))
            res = {}
            for stream in STREAMS:
                for name in ("skipped", "loss_bd", "loss_rg", "band_fraction", "confidence"):
                    row[f"{stream}_{name}"] = window.get(f"{stream}_{name}", math.nan)
                for name in ("raw_cos_bd_rg", "raw_rho", "raw_jaccard"):
                    row[f"{stream}_{name}"] = window.get(f"{tap}/{stream}_{name}", math.nan)
                bd, rg = (ema.get(f"{stream}/{tap}/{kind}") for kind in KINDS)
                res[stream] = residual(bd, rg)
                # Channel-space specificity: 1 - cos^2 is the share of the boundary vector
                # that the region vector cannot explain.
                row[f"{stream}_cos_bd_rg"] = cosine(bd, rg)
                row[f"{stream}_residual_fraction"] = 1 - row[f"{stream}_cos_bd_rg"] ** 2
                row[f"{stream}_rho_bd_rg"] = spearman(bd, rg)
                row[f"{stream}_helpful_fraction"] = (
                    math.nan if bd is None else float((bd < 0).double().mean()))
                old_bd, old_rg = (lagged.get(f"{stream}/{tap}/{kind}") for kind in KINDS)
                row[f"{stream}_lag_rho"] = spearman(bd, old_bd)
                row[f"{stream}_lag_jaccard"] = top_jaccard(bd, old_bd, self.top_fraction)
                old_res = residual(old_bd, old_rg)
                row[f"{stream}_lag_rho_res"] = spearman(res[stream], old_res)
                row[f"{stream}_lag_jaccard_res"] = top_jaccard(res[stream], old_res, self.top_fraction)
            left, right = ema.get(f"lab/{tap}/bd"), ema.get(f"unl/{tap}/bd")
            row["cross_rho_bd"] = spearman(left, right)
            row["cross_jaccard_bd"] = top_jaccard(left, right, self.top_fraction)
            row["cross_rho_res"] = spearman(res["lab"], res["unl"])
            row["cross_jaccard_res"] = top_jaccard(res["lab"], res["unl"], self.top_fraction)
            rows.append(row)
        return rows

    def log(self, iteration, output_dir):
        rows = self.rows(iteration)
        path = output_dir / "channel_diagnostics.csv"
        new = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)
        if iteration % self.dump_interval == 0 and self.ema:
            with (output_dir / "cbc_scores.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(iteration=iteration, ema={
                    key: value.tolist() for key, value in sorted(self.ema.items())})) + "\n")
        return rows

    def state_dict(self):
        return dict(ema=self.ema, previous=self.previous, window=self.window,
                    snapshots=list(self.snapshots))

    def load_state_dict(self, state):
        self.ema, self.previous, self.window = state["ema"], state["previous"], state["window"]
        self.snapshots.clear()
        self.snapshots.extend(state["snapshots"])
