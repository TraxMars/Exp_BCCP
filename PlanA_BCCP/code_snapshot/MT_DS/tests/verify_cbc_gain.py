"""Check the step-1c gain-free probes: band geometry, gain removal, closed-form contrast and
zero training side effects."""
import copy
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from cbc import TAPS, boundary_band
from cbc_gain import (ROWS, GainFreeTracker, contrast_scores, decompose, family_probes, head_losses,
                      head_maps, off_span)
from networks import VNetDS
from verify_cbc import training_step


def box_labels():
    labels = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    labels[:, 8:24, 10:22, 6:26] = 1
    return labels


def check_geometry_and_probes():
    labels = box_labels()
    shapes = [None, (8, 8, 8), (16, 16, 16), (32, 32, 32)]
    outputs = [None] + [torch.zeros(2, 2, *shape) for shape in shapes[1:]]
    unit = torch.ones(2, 32, 32, 32)
    # The full-resolution head keeps radius 2; coarse heads rebuild a radius-1 band.
    head, band, _ = head_maps(outputs, labels, unit, 2, 3)
    assert torch.equal(band, boundary_band(labels, 2)) and torch.equal(head, labels.float())
    target = F.interpolate(labels[:, None].float(), size=(8, 8, 8), mode="nearest")[:, 0]
    assert torch.equal(head_maps(outputs, labels, unit, 2, 1)[1], boundary_band(target, 1))
    # An object that vanishes at 1/4 resolution leaves that image without weight there.
    tiny = labels.clone()
    tiny[1] = 0
    tiny[1, 9, 9, 9] = 1
    assert head_maps(outputs, tiny, unit, 2, 1)[2][1].sum() == 0
    assert head_maps(outputs, tiny, unit, 2, 3)[2][1].sum() > 0

    # A prediction equal to the training targets has zero CE on every head; |margin| is 40.
    fitted = [None]
    for shape in shapes[1:]:
        target = F.interpolate(labels[:, None].float(), size=shape, mode="nearest")[:, 0]
        fitted.append(torch.stack([-20 * (2 * target - 1), 20 * (2 * target - 1)], 1))
    losses = family_probes(fitted, labels, unit, 2, (1, 2, 3))
    assert losses["bd"] < 1e-6 and losses["rg"] < 1e-6
    torch.testing.assert_close(torch.stack([losses["gbd"], losses["grg"]]), torch.full((2,), 40.0))
    stats = head_losses(fitted, labels, unit, 2)
    assert max(stats[f"hard_{kind}_h{i}"] for kind in ("bd", "rg") for i in (1, 2, 3)) < 1e-6
    # Full resolution: soft and hard probes coincide and the soft target has no entropy.
    assert abs(stats["soft_bd_h3"] - stats["hard_bd_h3"]) < 1e-6 and stats["soft_floor_h3"] < 1e-6
    # Coarse heads: the soft target has an entropy floor the fitted prediction cannot reach.
    assert stats["soft_floor_h1"] > 0.1 and stats["soft_bd_h1"] > stats["soft_floor_h1"]
    print("PASS: rebuilt per-head bands, hard / full probes, gain probes and soft-target floor")


def check_gain_removal():
    torch.manual_seed(0)
    basis = [torch.randn(16, dtype=torch.float64) for _ in range(2)]
    vector = torch.randn(16, dtype=torch.float64)
    rest = off_span(vector, basis)
    assert max(abs(float(rest @ b)) for b in basis) < 1e-10
    assert off_span(2 * basis[0] - basis[1], basis).abs().max() < 1e-10
    assert torch.equal(off_span(vector, []), vector)
    store = {"lab/hard/d2/bd": basis[0] + 0.1 * vector, "lab/hard/d2/rg": basis[1] + vector,
             "lab/hard/d2/gbd": basis[0], "lab/hard/d2/grg": basis[1]}
    parts = decompose(store, "lab", "hard", "d2")
    torch.testing.assert_close(parts["free"], 0.1 * rest)
    torch.testing.assert_close(parts["free_rg"], rest)
    assert parts["res"].abs().max() < 1e-10   # gain-free bd is a copy of the gain-free rg
    print("PASS: projection off span{gbd, grg} and the boundary-specific residual")


def check_contrast():
    # Real D2 head: the margin is sum_c a_c F_c plus the bias difference.
    model = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).eval()
    taps = {}
    with torch.no_grad():
        outputs = model(torch.randn(2, 1, 32, 32, 32), deepsup=True, taps=taps)
    weight, bias = model.out_conv_2.weight.flatten(1), model.out_conv_2.bias
    margin = ((weight[1] - weight[0]).view(1, -1, 1, 1, 1) * taps["d2"]).sum(1) + bias[1] - bias[0]
    torch.testing.assert_close(margin, outputs[2][:, 1] - outputs[2][:, 0])

    # Synthetic channels at 1/2 resolution: 0 separates everywhere, 1 only inside the band,
    # 2 is constant, 3 has the wrong sign everywhere.
    labels = box_labels()
    outputs = [None, None, torch.zeros(2, 2, 16, 16, 16)]
    unit = torch.ones(2, 32, 32, 32)
    head, band, _ = head_maps(outputs, labels, unit, 2, 2)
    signed = 2 * head - 1
    feature = torch.stack([signed, signed * band, torch.ones_like(head), -signed], 1)
    head_weight = torch.stack([torch.zeros(4), torch.ones(4)]).view(2, 4, 1, 1, 1)
    scores = contrast_scores(head_weight, feature, outputs, labels, unit, 2)
    for kind in ("bd", "rg"):
        assert scores[kind][0] > 0 and scores[kind][3] < 0 and abs(scores[kind][2]) < 1e-12
    assert scores["bd"][1] > 0 and abs(scores["rg"][1]) < 1e-12
    print("PASS: D2 margin decomposition and closed-form class contrast")


def check_no_side_effects():
    torch.manual_seed(0)
    model = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).train()
    teacher = copy.deepcopy(model).requires_grad_(False)
    images = torch.randn(4, 1, 32, 32, 32)
    labels = box_labels()
    oracle = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    oracle[:, 4:20, 6:28, 10:22] = 1
    results, trackers = [], []
    for observe in (False, True):
        torch.manual_seed(123)
        student = copy.deepcopy(model)
        tracker = GainFreeTracker(10, student.out_conv_2) if observe else None
        trackers.append(tracker)
        results.append(training_step(student, copy.deepcopy(teacher), images, labels, tracker, oracle=oracle))
    for left, right in zip(results[0][0] + results[0][1], results[1][0] + results[1][1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(results[0][2], results[1][2])
    tracker = trackers[1]
    for stream in ("lab", "orc"):
        for family in ("hard", "full"):
            for tap in TAPS:
                assert all(f"{stream}/{family}/{tap}/{kind}" in tracker.ema for kind in ("bd", "rg", "gbd", "grg"))
        assert f"{stream}/k/d2/bd" in tracker.ema and tracker.count[stream] == 1
    for key, vector in tracker.ema.items():
        assert torch.isfinite(vector).all() and abs(vector.abs().sum() - 1) < 1e-9, key
    print("PASS: 1c observation and oracle leave gradients, BN buffers and RNG bit-identical")

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for iteration in (50, 100):
            rows = tracker.log(iteration, directory)
        assert [(row["family"], row["tap"]) for row in rows] == ROWS
        lines = (directory / "channel_diagnostics.csv").read_text().splitlines()
        assert len(lines) == 1 + 2 * len(ROWS)
        header = lines[0].split(",")
        for name in ("orc_spec_free", "orc_keep_bd", "orc_split_res", "lab_split_res", "rho_lab_orc_res",
                     "rho_lab_orc_free", "orc_pos_fraction", "lab_soft_floor_h1", "orc_hard_rg_h3"):
            assert name in header, name
        restored = GainFreeTracker(10, None)
        restored.load_state_dict(copy.deepcopy(tracker.state_dict()))
        assert all(torch.equal(restored.ema[key], tracker.ema[key]) for key in tracker.ema)
        assert restored.count == tracker.count and len(restored.snapshots) == len(tracker.snapshots)
    print("PASS: diagnostics CSV and resume state round trip")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_geometry_and_probes()
    check_gain_removal()
    check_contrast()
    check_no_side_effects()
    print("All step-1c verification checks passed.")
