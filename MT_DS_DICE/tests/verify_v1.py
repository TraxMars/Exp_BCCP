"""Unit checks of the V1 probe (code/v1_probe.py) and the VNetDS feature / decode additions.

  1. distance_buckets equals the clipped scipy Chebyshev (chessboard) distance on synthetic masks.
  2. Contamination, effective keys, swap and the label-level correction / harm statistics on a hand example.
  3. A full probe step on a small VNetDS: signed correction rate is 0; no parameter .grad, no change to the
     model state (BatchNorm buffers included) or to the global RNG; deterministic; query-block size has no
     effect; the training graph is still usable for loss.backward() afterwards.
  4. forward(features=True) returns the same outputs; decode_from_x7 reproduces the full-resolution head.
"""
import copy
import math
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
import v1_probe as probe_module
from networks import VNetDS
from objectives import pseudo_label_consistency, supervised_loss
from v1_probe import distance_buckets, query_statistics, reverse_label_statistics, v1_probe


def blobs(shape, seed, threshold):
    rng = np.random.default_rng(seed)
    smooth = ndimage.gaussian_filter(rng.normal(size=shape), sigma=2)
    return smooth > threshold


def scipy_buckets(mask, depth=3):
    inside = ndimage.distance_transform_cdt(mask, metric="chessboard")       # fg: distance to nearest bg
    outside = ndimage.distance_transform_cdt(~mask, metric="chessboard")     # bg: distance to nearest fg
    expected = np.where(mask, -inside, outside)
    expected[np.abs(expected) > depth] = 0
    return expected


def check_buckets():
    masks = [blobs((24, 20, 16), seed, threshold) for seed in range(4) for threshold in (-0.05, 0.0, 0.1)]
    cube = np.zeros((20, 20, 20), dtype=bool)
    cube[0:12, 4:15, 9:20] = True          # touches three patch borders
    masks.append(cube)
    count = 0
    for mask in masks:
        assert mask.any() and (~mask).any()
        ours = distance_buckets(torch.from_numpy(mask)[None].long())[0].numpy()
        np.testing.assert_array_equal(ours, scipy_buckets(mask))
        count += mask.size
    # The patch border is not an object border: a cube touching it has no band along that face.
    ours = distance_buckets(torch.from_numpy(cube)[None].long())[0]
    # (0, 9, 14) and (5, 9, 19) lie on patch faces but ≥ 6 voxels from any background: no band there.
    assert ours[0, 9, 14] == 0 and ours[5, 9, 19] == 0 and ours[0, 4, 14] == -1 and ours[0, 3, 14] == 1
    for constant in (np.zeros((8, 8, 8), bool), np.ones((8, 8, 8), bool)):
        assert not distance_buckets(torch.from_numpy(constant)[None].long()).any()
    print(f"PASS: distance buckets equal the clipped scipy chessboard distance on {len(masks)} masks "
          f"({count} voxels, 3 borders touched); constant masks have no band")


def check_hand_example():
    # 3 queries (GT side fg, bg, bg), 4 keys; one group.
    weights = torch.tensor([[0.6, 0.2, 0.2, 0.0],
                            [0.1, 0.5, 0.0, 0.4],
                            [0.3, 0.0, 0.3, 0.4]])[None]
    query_side = torch.tensor([1, 0, 0])
    key_true = torch.tensor([1, 0, 1, 1])
    key_pseudo = torch.tensor([0, 0, 1, 0])      # k0 and k3 have a wrong pseudo side
    key_confidence = torch.tensor([0.99, 0.5, 0.97, 0.99])
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, 2.0]])
    contamination, swap, effective = query_statistics(weights, queries, values, query_side, key_true)
    # q0 (fg): mass on true-bg keys {k1} = 0.2; q1 (bg): on true-fg keys {k0,k2,k3} = 0.5; q2: 0.3+0.3+0.4 = 1.0
    torch.testing.assert_close(contamination, torch.tensor([0.2, 0.5, 1.0]))
    entropy = [-sum(w * math.log(w) for w in row if w > 0) for row in weights[0].tolist()]
    torch.testing.assert_close(effective, torch.tensor([math.exp(h) for h in entropy]))
    mixed = weights[0] @ values
    torch.testing.assert_close(swap, 1 - F.cosine_similarity(queries, mixed, dim=1))
    # Received mass m = [1.0, 0.7, 0.5, 0.8]; from the fg query [0.6, 0.2, 0.2, 0.0]
    # → r = [0.6, 0.286, 0.4, 0.0] → majority [fg, bg, bg, bg].
    # Wrong keys k0 (true fg: corrected, m 1.0), k3 (true fg: not corrected, m 0.8) → correction rate 1.0/1.8.
    # Right keys k1 (true bg: fine), k2 (true fg, majority bg: harmed, m 0.5) → harm rate 0.5/1.2.
    mass, foreground = weights[0].sum(0), query_side.float() @ weights[0]
    result = reverse_label_statistics(mass, foreground, key_true, key_pseudo, key_confidence)
    expected = dict(all=dict(correction_rate=1.0 / 1.8, harm_rate=0.5 / 1.2, correction_mass=1.0 / 3.0,
                             harm_mass=0.5 / 3.0, wrong_mass_frac=1.8 / 3.0),
                    # conf > 0.95 keeps k0, k2, k3: wrong {k0, k3}, right {k2} (harmed)
                    conf095=dict(correction_rate=1.0 / 1.8, harm_rate=1.0, correction_mass=1.0 / 3.0,
                                 harm_mass=0.5 / 3.0, wrong_mass_frac=1.8 / 3.0))
    for subset, values_ in expected.items():
        for key, value in values_.items():
            assert abs(result[subset][key] - value) < 1e-6, (subset, key, result[subset][key], value)
    # A tie (r = 0.5) is no majority: neither corrected nor harmed.
    tie = reverse_label_statistics(torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5]), torch.tensor([1, 0]),
                                   torch.tensor([0, 0]), torch.tensor([1.0, 1.0]))
    assert tie["all"]["correction_rate"] == 0 and tie["all"]["harm_rate"] == 0
    print("PASS: contamination [0.2, 0.5, 1.0], effective keys, swap; correction rate 1.0/1.8, harm rate 0.5/1.2 "
          "(conf>0.95: 1.0), masses /3.0, coverage 0.6 as hand-computed; a tie counts as neither")


def training_step(model, teacher, images, labels, hidden):
    """As in train.py on a diagnostics step: outputs, losses and the features of the same forward."""
    supervised, labeled = model(images[:2], deepsup=True, features=True)
    unlabeled_outputs, unlabeled = model(images[2:], deepsup=True, features=True)
    loss_sup = supervised_loss(supervised, labels)[0]
    with torch.no_grad():
        teacher_outputs = teacher(images[2:], deepsup=True)
    loss_pl = pseudo_label_consistency(unlabeled_outputs, teacher_outputs)[0]
    loss_true = supervised_loss(unlabeled_outputs, hidden)[0]
    return loss_sup + 0.3 * loss_pl, labeled, unlabeled, teacher_outputs, 0.3 * loss_pl, loss_true


def small_case():
    torch.manual_seed(0)
    images = torch.randn(4, 1, 48, 48, 32)
    masks = np.stack([blobs((48, 48, 32), seed, 0.0) for seed in range(4)])
    full = torch.from_numpy(masks).long()
    return images, full[:2], full[2:]


def run_probe(model, teacher, case, seed=7, backward=False):
    images, labels, hidden = case
    torch.manual_seed(123)   # same Dropout masks in the training forward as in check_probe
    loss, labeled, unlabeled, teacher_outputs, loss_pl, loss_true = training_step(model, teacher, images, labels, hidden)
    rows = v1_probe(model, 50, labeled, unlabeled, labels, hidden, teacher_outputs[2], loss_pl, loss_true, seed)
    if backward:
        loss.backward()
    return rows


def keyed(rows):
    return {(r["family"], r["constraint"], r["G"], r["tau"], r["subset"], r["metric"]): r["value"]
            for r in rows if r["family"] != "meta"}


def check_probe():
    case = small_case()
    model = VNetDS(n_channels=1, n_classes=2, has_dropout=True).train()
    teacher = copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))   # Teacher ≠ Student, so pseudo sides can be wrong
    state = copy.deepcopy(model.state_dict())
    torch.manual_seed(123)
    rng_before = torch.get_rng_state()
    images, labels, hidden = case
    loss, labeled, unlabeled, teacher_outputs, loss_pl, loss_true = training_step(model, teacher, images, labels, hidden)
    rng_after_forward = torch.get_rng_state()   # Dropout in the forward consumes the global RNG
    after_forward = copy.deepcopy(model.state_dict())   # the training forward itself updates BN buffers
    rows = v1_probe(model, 50, labeled, unlabeled, labels, hidden, teacher_outputs[2], loss_pl, loss_true, 7)
    assert torch.equal(torch.get_rng_state(), rng_after_forward), "probe consumed the global RNG"
    assert all(parameter.grad is None for parameter in model.parameters()), "probe wrote .grad"
    for key, value in model.state_dict().items():
        assert torch.equal(value, after_forward[key]), f"probe changed {key}"
    loss.backward()   # the retained training graph is intact
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    del rng_before

    values = keyed(rows)
    families = sorted({key[0] for key in values})
    assert all(isinstance(r["value"], float) for r in rows)
    signed = [value for key, value in values.items() if key[0].startswith("reverse_label") and key[1] == "signed"
              and key[5] == "correction_rate"]
    assert signed and all(value == 0 or math.isnan(value) for value in signed)
    assert any(value == 0 for value in signed)
    wrong_mass = values[("reverse_label", "unsigned", 4, 0.1, "all", "wrong_mass_frac")]
    unsigned = values[("reverse_label", "unsigned", 4, 0.1, "all", "correction_rate")]
    nonfinite = sorted(key for key, value in values.items() if not math.isfinite(value))

    model.zero_grad(set_to_none=True)
    model.load_state_dict(state)
    again = keyed(run_probe(model, teacher, case))
    model.load_state_dict(state)
    original_chunk = probe_module.CHUNK
    probe_module.CHUNK = 37
    try:
        chunked = keyed(run_probe(model, teacher, case))
    finally:
        probe_module.CHUNK = original_chunk
    model.load_state_dict(state)
    other_seed = keyed(run_probe(model, teacher, case, seed=8))
    assert again.keys() == values.keys() == chunked.keys()
    same = all(again[key] == values[key] or (math.isnan(again[key]) and math.isnan(values[key])) for key in values)
    gap = max(abs(chunked[key] - values[key]) for key in values if math.isfinite(values[key]))
    assert same, "probe not deterministic for a fixed seed"
    assert gap < 1e-4, gap
    assert any(other_seed[key] != values[key] for key in values if math.isfinite(values[key]))

    # Same step with and without the probe: parameter gradients and the full state bitwise identical.
    results = []
    for with_probe in (False, True):
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        loss, labeled, unlabeled, teacher_outputs, loss_pl, loss_true = training_step(model, teacher, *case)
        if with_probe:
            v1_probe(model, 50, labeled, unlabeled, case[1], case[2], teacher_outputs[2], loss_pl, loss_true, 7)
        loss.backward()
        results.append(([parameter.grad.clone() for parameter in model.parameters()],
                         copy.deepcopy(model.state_dict()), torch.get_rng_state()))
    (grads_off, state_off, rng_off), (grads_on, state_on, rng_on) = results
    assert all(torch.equal(left, right) for left, right in zip(grads_off, grads_on))
    assert all(torch.equal(state_off[key], state_on[key]) for key in state_off) and torch.equal(rng_off, rng_on)
    print(f"PASS: probe step on a 48x48x32 batch: {len(rows)} rows, families {families}; signed correction rate "
          f"0 (or NaN without wrong keys) in all {len(signed)} (setting, subset) pairs (unsigned G=4 tau=0.1: "
          f"{unsigned:.3f}, wrong-key mass share {wrong_mass:.3f})")
    print("PASS: no parameter .grad, state_dict incl. BN buffers bitwise unchanged, global RNG untouched, "
          "loss.backward() still works; same seed → identical rows; query blocks of 37 vs 1024: max |diff| "
          f"{gap:.1e}; another seed → different samples")
    print("PASS: one training step with vs without the probe: all parameter gradients, the state_dict (BN buffers "
          "included) and the global RNG state after backward are bitwise identical")
    print(f"  non-finite values (empty subsets in this tiny case): {len(nonfinite)} of {len(values)}"
          + (f", e.g. {nonfinite[:3]}" if nonfinite else ""))


def check_network():
    torch.manual_seed(42)
    reference = VNetDS(n_channels=1, n_classes=2, has_dropout=True).state_dict()
    torch.manual_seed(42)
    current = VNetDS(n_channels=1, n_classes=2, has_dropout=True)
    assert list(reference) == list(current.state_dict())
    assert all(torch.equal(reference[key], value) for key, value in current.state_dict().items())
    model = VNetDS(n_channels=1, n_classes=2, has_dropout=False)
    inputs = torch.randn(2, 1, 32, 32, 32)
    for mode in ("train", "eval"):
        getattr(model, mode)()
        with torch.no_grad():
            plain = model(inputs, deepsup=True)
            outputs, features = model(inputs, deepsup=True, features=True)
            decoded = model.decode_from_x7(features["x7_up"], features["x1"])
        assert all(torch.equal(left, right) for left, right in zip(plain, outputs))
        assert torch.equal(decoded, outputs[-1]), mode
        assert features["x7_up"].shape[1:] == (32, 16, 16, 16)
    print("PASS: same seeded initial weights and parameter order; forward(features=True) returns identical outputs; "
          "decode_from_x7(x7_up, x1) reproduces the full-resolution head bitwise (train- and eval-mode BN)")


def main():
    torch.set_num_threads(8)
    check_buckets()
    check_hand_example()
    check_network()
    check_probe()


if __name__ == "__main__":
    main()
