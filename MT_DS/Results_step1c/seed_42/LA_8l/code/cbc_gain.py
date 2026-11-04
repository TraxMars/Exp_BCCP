"""Step-1c observation: gain-free channel contributions at the D3/D2 taps (logging only).

Step 1b found boundary and region contributions nearly collinear on the unlabeled ground
truth, with 69-94% of channels at psi > 0: removing almost any channel lowers the band CE
to first order, the signature of a margin-magnitude (gain) direction shared by every probe.
This tracker separates that direction from localization.

  hard  CE on every DS head (1/4, 1/2, 1) against the nearest-neighbour target that
        supervised_loss trains on, inside (bd) / outside (rg) a band rebuilt from that
        target at the head's own resolution
  full  the same on the full-resolution head only, with no resampling at all
  gain  psi of the mean |margin| inside (gbd) and outside (grg) the band, same heads.
        bd and rg are projected off span{gbd, grg} (conservative: neither keeps a gain
        component the other lacks); the boundary-specific residual is what remains of the
        gain-free bd after its component along the gain-free rg.
  k     closed form at the D2 head, no gradients: K_c = (mean_inner a_c F_c -
        mean_outer a_c F_c) / std(margin), a_c = W_1c - W_0c, inner / outer = foreground /
        background side of the band (bd) or of the far field (rg). K_c > 0: channel c
        separates the classes. It reads the auxiliary D2 head only, not the final output.

Per-head CE of the old soft-target probe (avg-pooled target and band, as in the LA step-1
and 1b runs) and the entropy floor of that soft target are logged forward-only.
Like step 1b, every probe uses autograd.grad on the retained training graph or plain
tensor algebra: parameter gradients, BN statistics and random streams stay untouched.
"""
import math

import torch

from cbc import (PAIRS, PROBE_HEADS, STREAMS, TAPS, TINY, CBCTracker, band_iou, boundary_band,
                 cosine, head_target, pool_to, residual, spearman, top_jaccard, weighted_mean)


FAMILIES = {"hard": PROBE_HEADS, "full": (3,)}
CE_KINDS = ("bd", "rg")
GAIN_KINDS = ("gbd", "grg")
ROWS = [(family, tap) for family in FAMILIES for tap in TAPS] + [("k", "d2")]
HEAD_LOSSES = tuple(f"{name}_h{index}" for name in ("hard_bd", "hard_rg", "soft_bd", "soft_floor")
                    for index in PROBE_HEADS)


def head_maps(outputs, target, confidence, radius, index):
    """Nearest target, rebuilt band and confidence weights on one head's grid.

    The band radius shrinks with the head's stride (at least one voxel); images whose band
    is empty or covers the whole head are given zero weight.
    """
    shape = outputs[index].shape[2:]
    factor = target.shape[-1] // shape[-1]
    head = head_target(target, shape, "nearest")
    band = boundary_band(head, max(1, round(radius / factor)))
    flat = band.flatten(1) > 0
    valid = (flat.any(dim=1) & ~flat.all(dim=1)).float().view(-1, 1, 1, 1)
    return head, band, pool_to(confidence, shape) * valid


def family_probes(outputs, target, confidence, radius, heads):
    """CE and |margin| inside / outside the rebuilt band, averaged over the usable heads."""
    terms = {kind: [] for kind in CE_KINDS + GAIN_KINDS}
    for index in heads:
        head, band, trust = head_maps(outputs, target, confidence, radius, index)
        weights = dict(bd=band * trust, rg=(1 - band) * trust)
        if min(weight.sum() for weight in weights.values()) <= 0:
            continue
        logits = outputs[index]
        log_probability = logits.log_softmax(dim=1)
        ce = -(head * log_probability[:, 1] + (1 - head) * log_probability[:, 0])
        margin = (logits[:, 1] - logits[:, 0]).abs()
        for kind, weight in weights.items():
            terms[kind].append(weighted_mean(ce, weight))
            terms["g" + kind].append(weighted_mean(margin, weight))
    if not terms["bd"]:
        return None
    return {kind: torch.stack(values).mean() for kind, values in terms.items()}


@torch.no_grad()
def head_losses(outputs, target, confidence, radius):
    """Per-head band / region CE of the hard probe, and band CE and entropy floor of the soft one."""
    full_band = boundary_band(target, radius)
    flat = full_band.flatten(1) > 0
    valid = (flat.any(dim=1) & ~flat.all(dim=1)).float().view(-1, 1, 1, 1)
    stats = {}
    for index in PROBE_HEADS:
        log_probability = outputs[index].float().log_softmax(dim=1)
        shape = log_probability.shape[2:]
        head, band, trust = head_maps(outputs, target, confidence, radius, index)
        ce = -(head * log_probability[:, 1] + (1 - head) * log_probability[:, 0])
        for kind, weight in (("bd", band * trust), ("rg", (1 - band) * trust)):
            stats[f"hard_{kind}_h{index}"] = weighted_mean(ce, weight).item() if weight.sum() > 0 else math.nan
        soft = pool_to(target.float(), shape)
        weight = pool_to(full_band, shape) * pool_to(confidence * valid, shape)
        ce = -(soft * log_probability[:, 1] + (1 - soft) * log_probability[:, 0])
        entropy = -(soft * soft.clamp_min(TINY).log() + (1 - soft) * (1 - soft).clamp_min(TINY).log())
        usable = weight.sum() > 0
        stats[f"soft_bd_h{index}"] = weighted_mean(ce, weight).item() if usable else math.nan
        stats[f"soft_floor_h{index}"] = weighted_mean(entropy, weight).item() if usable else math.nan
    return stats


def taylor(losses, taps):
    """psi_c = mean_x dL/dF_c * F_c of every probe loss at both taps."""
    features = [taps[name] for name in TAPS]
    result = {name: {} for name in TAPS}
    for kind, loss in losses.items():
        grads = torch.autograd.grad(loss, features, retain_graph=True)
        for name, feature, grad in zip(TAPS, features, grads):
            result[name][kind] = (grad.double() * feature.detach().double()).mean(dim=(0, 2, 3, 4))
    return result


@torch.no_grad()
def contrast_scores(head_weight, feature, outputs, target, confidence, radius):
    """Closed-form class contrast of each channel's share a_c F_c of the D2-head margin."""
    weight = head_weight.detach().flatten(1).double()
    parts = (weight[1] - weight[0]).view(1, -1, 1, 1, 1) * feature.detach().double()
    # The bias shifts the margin uniformly and changes neither the contrast nor the spread.
    margin = parts.sum(dim=1)
    head, band, trust = head_maps(outputs, target, confidence, radius, 2)
    head, band, trust = head.double(), band.double(), trust.double()
    scores = {}
    for kind, region in (("bd", band * trust), ("rg", (1 - band) * trust)):
        inner, outer = region * head, region * (1 - head)
        if inner.sum() <= 0 or outer.sum() <= 0:
            return None
        side = lambda mask: (parts * mask[:, None]).sum(dim=(0, 2, 3, 4)) / mask.sum()
        center = (margin * region).sum() / region.sum()
        spread = (((margin - center) ** 2 * region).sum() / region.sum()).sqrt()
        scores[kind] = (side(inner) - side(outer)) / spread.clamp_min(TINY)
    return scores


def off_span(vector, basis):
    """Remove the least-squares component of vector along the span of the basis vectors."""
    if not basis:
        return vector
    matrix = torch.stack(basis, dim=1)
    return vector - matrix @ (torch.linalg.pinv(matrix) @ vector)


def decompose(store, stream, family, tap):
    """Raw, gain-free and boundary-specific vectors of one stream from an EMA store."""
    get = lambda kind: store.get(f"{stream}/{family}/{tap}/{kind}")
    bd, rg = get("bd"), get("rg")
    if bd is None or rg is None:
        return None
    gains = [vector for vector in (get(kind) for kind in GAIN_KINDS) if vector is not None]
    free_bd, free_rg = off_span(bd, gains), off_span(rg, gains)
    return dict(bd=bd, rg=rg, free=free_bd, free_rg=free_rg, res=residual(free_bd, free_rg))


def norm_ratio(part, whole):
    return float(part.norm() / whole.norm().clamp_min(TINY))


class GainFreeTracker(CBCTracker):
    """EMA bookkeeping of CBCTracker (L1-normalized observations, CHAP cosine momentum,
    split halves, lagged snapshots) applied to the hard / full / k probes of step 1c."""

    def __init__(self, max_iterations, d2_head, band_radius=2, top_fraction=0.25, **kwargs):
        super().__init__(max_iterations, band_radius, top_fraction, "nearest", **kwargs)
        self.d2_head = d2_head

    @torch.no_grad()
    def _store(self, iteration, stream, family, tap, values):
        beta = self.beta(iteration)
        half = self.halves[self.count.get(stream, 0) % 2]
        for kind, vector in values.items():
            vector = vector.cpu()
            vector = vector / vector.abs().sum().clamp_min(TINY)
            key = f"{stream}/{family}/{tap}/{kind}"
            self._smooth(self.ema, key, vector, beta)
            self._smooth(half, key, vector, beta)

    def observe(self, iteration, labeled_outputs, labels, labeled_taps,
                unlabeled_outputs, teacher_outputs, unlabeled_taps, oracle_labels=None):
        """oracle_labels: hidden GT of the unlabeled images, used for logging only."""
        radius = self.band_radius
        confidence, pseudo = teacher_outputs[-1].detach().softmax(dim=1).max(dim=1)
        unit = lambda target: torch.ones_like(target, dtype=confidence.dtype)
        streams = dict(lab=(labeled_outputs, labels, unit(labels), labeled_taps),
                       unl=(unlabeled_outputs, pseudo, confidence, unlabeled_taps))
        pseudo_band = boundary_band(pseudo, radius)
        if pseudo_band.any():
            self._accumulate("unl_band_confidence", weighted_mean(confidence, pseudo_band).item())
        if oracle_labels is not None:
            streams["orc"] = (unlabeled_outputs, oracle_labels, unit(oracle_labels), unlabeled_taps)
            self._accumulate("unl_band_iou_gt", band_iou(pseudo_band, boundary_band(oracle_labels, radius)))
        for stream, (outputs, target, trust, taps) in streams.items():
            for key, value in head_losses(outputs, target, trust, radius).items():
                if math.isfinite(value):
                    self._accumulate(f"{stream}_{key}", value)
            observed = False
            for family, heads in FAMILIES.items():
                losses = family_probes(outputs, target, trust, radius, heads)
                usable = losses is not None and bool(torch.isfinite(torch.stack(list(losses.values()))).all())
                self._accumulate(f"{stream}_{family}_skipped", float(not usable))
                if usable:
                    for tap, values in taylor(losses, taps).items():
                        self._store(iteration, stream, family, tap, values)
                    observed = True
            scores = contrast_scores(self.d2_head.weight, taps["d2"], outputs, target, trust, radius)
            usable = scores is not None and all(bool(torch.isfinite(v).all()) for v in scores.values())
            self._accumulate(f"{stream}_k_skipped", float(not usable))
            if usable:
                self._store(iteration, stream, "k", "d2", scores)
                observed = True
            if observed:
                self.count[stream] = self.count.get(stream, 0) + 1

    def rows(self, iteration):
        ema = {key: value.clone() for key, value in self.ema.items()}
        lagged = self.snapshots[0] if len(self.snapshots) == self.snapshots.maxlen else {}
        self.snapshots.append(ema)
        window = {key: total / count for key, (total, count) in self.window.items()}
        self.window = {}
        rows = []
        for family, tap in ROWS:
            # Most helpful channels: most negative psi, but most positive contrast K.
            sign = -1.0 if family == "k" else 1.0
            row = dict(iteration=iteration, family=family, tap=tap, beta=self.beta(iteration),
                       unl_band_iou_gt=window.get("unl_band_iou_gt", math.nan),
                       unl_band_confidence=window.get("unl_band_confidence", math.nan))
            parts = {}
            for stream in STREAMS:
                now = decompose(ema, stream, family, tap)
                halves = [decompose(half, stream, family, tap) for half in self.halves]
                old = decompose(lagged, stream, family, tap)
                parts[stream] = now
                row[f"{stream}_skipped"] = window.get(f"{stream}_{family}_skipped", math.nan)
                values = dict.fromkeys(("cos_bd_rg", "spec", "gain_share_bd", "gain_share_rg", "keep_bd",
                                        "cos_free", "spec_free", "pos_fraction"), math.nan)
                if now is not None:
                    values.update(cos_bd_rg=cosine(now["bd"], now["rg"]), cos_free=cosine(now["free"], now["free_rg"]),
                                  keep_bd=norm_ratio(now["free"], now["bd"]),
                                  gain_share_rg=1 - norm_ratio(now["free_rg"], now["rg"]) ** 2,
                                  pos_fraction=float((now["bd"] > 0).double().mean()))
                    values.update(spec=1 - values["cos_bd_rg"] ** 2, spec_free=1 - values["cos_free"] ** 2,
                                  gain_share_bd=1 - values["keep_bd"] ** 2)
                for name in ("bd", "free", "res"):
                    pick = lambda parts_, name=name: None if parts_ is None else parts_[name]
                    values[f"split_{name}"] = spearman(*(pick(half) for half in halves))
                    values[f"lag_{name}"] = spearman(pick(now), pick(old))
                for name, value in values.items():
                    row[f"{stream}_{name}"] = value
                for name in HEAD_LOSSES:
                    row[f"{stream}_{name}"] = window.get(f"{stream}_{name}", math.nan)
            for left, right in PAIRS:
                first, second = parts[left], parts[right]
                for name in ("bd", "free", "res"):
                    row[f"rho_{left}_{right}_{name}"] = (math.nan if first is None or second is None
                                                         else spearman(first[name], second[name]))
                row[f"jaccard_{left}_{right}_free"] = (math.nan if first is None or second is None else
                                                       top_jaccard(sign * first["free"], sign * second["free"],
                                                                   self.top_fraction))
            rows.append(row)
        return rows
