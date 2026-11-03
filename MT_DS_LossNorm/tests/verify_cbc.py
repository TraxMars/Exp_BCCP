"""Check the observation-only CBC probes: geometry, algebra and zero training side effects."""
import copy
from pathlib import Path
import sys
import tempfile

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from cbc import (CBCTracker, STREAMS, TAPS, band_iou, boundary_band, contributions, cosine,
                 labeled_probes, pool_to, pseudo_label_probes, residual, spearman, top_jaccard)
from networks import VNetDS
from objectives import consistency_loss, supervised_loss


def check_geometry():
    mask = torch.zeros(1, 12, 12, 12)
    mask[0, 3:9, 3:9, 3:9] = 1
    band = boundary_band(mask, 1)
    assert band[0, 6, 6, 6] == 0 and band[0, 0, 0, 0] == 0
    assert band[0, 3, 6, 6] == 1 and band[0, 2, 6, 6] == 1 and band[0, 4, 6, 6] == 0
    assert set(band.unique().tolist()) <= {0.0, 1.0}
    # An object touching the patch border is not eroded there.
    edge = torch.zeros(1, 8, 8, 8)
    edge[0, :4] = 1
    assert boundary_band(edge, 1)[0, 0, 4, 4] == 0
    pooled = pool_to(band, (3, 3, 3))
    torch.testing.assert_close(pooled.mean(), band.mean())
    try:
        pool_to(band, (5, 5, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-integer pooling must fail")
    print("PASS: morphological band, border handling and mean-preserving pooling")


def check_algebra():
    features = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
    other = torch.randn(2, 5, 2, 2, 2, requires_grad=True)
    a, b = torch.randn(2, 3, 4, 4, 4), torch.randn(2, 3, 4, 4, 4)
    taps = {TAPS[0]: features, TAPS[1]: other}
    loss_bd = (a * features).sum() + other.sum()
    loss_rg = (b * features).sum() + 2 * other.sum()
    values = contributions(loss_bd, loss_rg, taps)
    f, a, b = features.detach().double().flatten(2), a.double().flatten(2), b.double().flatten(2)
    torch.testing.assert_close(values[TAPS[0]]["bd"], (a * f).mean(dim=(0, 2)))
    torch.testing.assert_close(values[TAPS[0]]["rg"], (b * f).mean(dim=(0, 2)))
    torch.testing.assert_close(values[TAPS[1]]["rg"], 2 * other.detach().double().mean(dim=(0, 2, 3, 4)))
    bd, rg = torch.randn(16, dtype=torch.float64), torch.randn(16, dtype=torch.float64)
    res = residual(bd, rg)
    assert abs(res @ rg) < 1e-10 and residual(3 * rg, rg).abs().max() < 1e-12
    assert abs(cosine(bd, 2 * bd) - 1) < 1e-12 and residual(bd, None) is None
    print("PASS: Taylor contributions and channel-space residual / cosine")


def check_rank_metrics():
    x = torch.tensor([0.3, -0.1, 0.2, -0.5], dtype=torch.float64)
    assert abs(spearman(x, x) - 1) < 1e-12 and abs(spearman(x, -x) + 1) < 1e-12
    assert top_jaccard(x, x, 0.5) == 1.0 and top_jaccard(x, -x, 0.5) == 0.0
    tracker = CBCTracker(max_iterations=100)
    assert abs(tracker.beta(0) - 0.5) < 1e-12 and abs(tracker.beta(100) - 0.99) < 1e-12
    print("PASS: rank metrics and CHAP cosine momentum schedule")


def training_step(model, teacher, images, labels, tracker=None, iteration=1, oracle=None):
    """One baseline step; returns parameter gradients, BN buffers and RNG state."""
    labeled_taps, unlabeled_taps = {}, {}
    outputs = model(images[:2], deepsup=True, taps=labeled_taps)
    unlabeled = model(images[2:], deepsup=True, taps=unlabeled_taps)
    with torch.no_grad():
        targets = teacher(images[2:], deepsup=True)
    loss = supervised_loss(outputs, labels)[0] + 0.3 * consistency_loss(unlabeled, targets)
    model.zero_grad(set_to_none=True)
    if tracker is not None:
        tracker.observe(iteration, outputs, labels, labeled_taps, unlabeled, targets, unlabeled_taps,
                        oracle)
        # The probes, including the oracle on hidden GT, never reach a parameter gradient.
        assert all(parameter.grad is None for parameter in model.parameters())
    loss.backward()
    grads = [parameter.grad.clone() for parameter in model.parameters()]
    buffers = [buffer.clone() for buffer in model.buffers()]
    return grads, buffers, torch.get_rng_state()


def check_no_side_effects():
    torch.manual_seed(0)
    model = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).train()
    teacher = copy.deepcopy(model).requires_grad_(False)
    images = torch.randn(4, 1, 32, 32, 32)
    labels = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    labels[:, 8:24, 10:22, 6:26] = 1
    oracle = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    oracle[:, 4:20, 6:28, 10:22] = 1
    tracker = CBCTracker(max_iterations=10)
    results = []
    for observer in (None, tracker):
        torch.manual_seed(123)
        results.append(training_step(copy.deepcopy(model), copy.deepcopy(teacher),
                                     images, labels, observer, oracle=oracle))
    for left, right in zip(results[0][0] + results[0][1], results[1][0] + results[1][1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(results[0][2], results[1][2])
    # The random-init teacher may predict a single class, which skips the pseudo-label probe.
    assert all(f"{stream}/{tap}/{kind}" in tracker.ema
               for stream in ("lab", "orc") for tap in TAPS for kind in ("bd", "rg"))
    assert tracker.count["lab"] == tracker.count["orc"] == 1
    for key, vector in tracker.ema.items():
        assert torch.isfinite(vector).all() and abs(vector.abs().sum() - 1) < 1e-9, key
    print("PASS: observation and oracle leave gradients, BN buffers and RNG bit-identical")

    # Taps are exactly the tensors read by the D3/D2 heads.
    model.eval()
    taps = {}
    with torch.no_grad():
        outputs = model(images, deepsup=True, taps=taps)
        torch.testing.assert_close(model.out_conv_3(taps["d3"]), outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(model.out_conv_2(taps["d2"]), outputs[2], rtol=0, atol=0)
        for left, right in zip(outputs, model(images, deepsup=True)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert taps["d3"].shape[1:] == (64, 8, 8, 8) and taps["d2"].shape[1:] == (32, 16, 16, 16)
    print("PASS: D3/D2 taps feed the deep-supervision heads; outputs unchanged")

    # Empty bands are skipped instead of producing 0/0.
    zero = torch.zeros_like(labels)
    model.train()
    outputs = model(images[:2], deepsup=True)
    assert labeled_probes(outputs, zero, 2) is None
    teacher_logits = [torch.stack([torch.ones_like(o[:, 0]), -torch.ones_like(o[:, 0])], 1)
                      for o in outputs]
    assert pseudo_label_probes(outputs, teacher_logits, 2)[0] is None
    # Only images with a usable band count; the other image's weights are zeroed.
    mixed = labels.clone()
    mixed[1] = 0
    probe = labeled_probes(outputs, mixed, 2)
    assert probe is not None and probe[2]["valid_images"] == 1
    # A pseudo-label equal to the GT reproduces the GT probe exactly and its band IoU is 1.
    confident = [torch.stack([-20 * (2 * labels - 1).float(), 20 * (2 * labels - 1).float()], 1)]
    confident = [None, None, None, confident[0]]
    pseudo, band = pseudo_label_probes(outputs, confident, 2)
    exact = labeled_probes(outputs, labels, 2)
    torch.testing.assert_close(torch.stack(pseudo[:2]), torch.stack(exact[:2]))
    assert band_iou(band, boundary_band(labels, 2)) == 1.0
    # A band covering the whole patch leaves the region probe undefined.
    grid = torch.stack(torch.meshgrid(*[torch.arange(32)] * 3, indexing="ij")).sum(0) % 2
    assert labeled_probes(outputs, grid.expand(2, -1, -1, -1), 1) is None
    print("PASS: empty or all-covering labeled / pseudo-label bands are skipped")

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for iteration in (50, 100):
            rows = tracker.log(iteration, directory)
        assert [row["tap"] for row in rows] == list(TAPS)
        lines = (directory / "channel_diagnostics.csv").read_text().splitlines()
        assert len(lines) == 1 + 2 * len(TAPS)
        header = lines[0].split(",")
        for name in ("lab_split_rho", "unl_lag_rho", "orc_lag_rho_rg", "rho_unl_orc_bd",
                     "rho_lab_orc_bd", "unl_band_iou_gt", "unl_band_confidence", "lab_valid_images"):
            assert name in header, name
        restored = CBCTracker(max_iterations=10)
        restored.load_state_dict(copy.deepcopy(tracker.state_dict()))
        for key in tracker.ema:
            assert torch.equal(restored.ema[key], tracker.ema[key])
        assert len(restored.snapshots) == len(tracker.snapshots)
        assert restored.count == tracker.count and restored.halves[0].keys() == tracker.halves[0].keys()
    print("PASS: diagnostics CSV and resume state round trip")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_geometry()
    check_algebra()
    check_rank_metrics()
    check_no_side_effects()
    print("All CBC verification checks passed.")
