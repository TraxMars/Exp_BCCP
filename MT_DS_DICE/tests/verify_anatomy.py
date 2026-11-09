"""Checks of scripts/anatomy.py (CPU): distances, error taxonomy on synthetic shapes, contributions, tiled inference.

  python tests/verify_anatomy.py   -> tests/outputs/anatomy/verify_anatomy.txt
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import anatomy as A   # noqa: E402

LINES = []


def log(text):
    print(text, flush=True)
    LINES.append(text)


def ball(shape, center, radius):
    grid = np.indices(shape).astype(float)
    return sum((g - c) ** 2 for g, c in zip(grid, center)) <= radius ** 2


def shares(pred, label):
    d_label, d_pred = A.signed_distance(label), A.signed_distance(pred)
    category = A.classify_errors(pred, label, d_label, d_pred)
    n = max((category > 0).sum(), 1)
    return {name: (category == code).sum() / n for code, name in enumerate(A.CATEGORIES, 1)}, category, d_pred


def check_distance():
    rng = np.random.default_rng(0)
    mask = rng.random((9, 8, 7)) < 0.3
    d = A.signed_distance(mask)
    points = np.argwhere(np.ones(mask.shape, bool))
    for p in points:
        other = np.argwhere(mask != mask[tuple(p)])
        expected = np.abs(other - p).max(1).min()
        assert d[tuple(p)] == (-expected if mask[tuple(p)] else expected), p
    assert (A.signed_distance(np.zeros((3, 3, 3), bool)) == A.FAR).all()
    log("PASS distance: signed Chebyshev distance = brute force on a random 9x8x7 mask (bg +k, fg -k); empty mask -> FAR")


def check_taxonomy():
    shape, center = (48, 48, 48), (24, 24, 24)
    label = ball(shape, center, 12)
    results = {}

    s, _, _ = shares(ball(shape, center, 13), label)                       # 1-voxel over-segmentation
    assert s["block"] == 0 and s["shift"] >= 0.9, s
    results["over-segmented by 1 voxel"] = s

    rng = np.random.default_rng(1)
    d = A.signed_distance(label)
    pred = label.copy()
    pred[(d == -1) & (rng.random(shape) < 0.5)] = False                    # random +-1 voxel jitter
    pred[(d == 1) & (rng.random(shape) < 0.5)] = True
    s, _, _ = shares(pred, label)
    assert s["block"] == 0 and s["jitter"] >= 0.9, s
    results["random +-1 jitter"] = s

    pred = label.copy()
    pred[30:] = False                                                      # a missed cap, 6 voxels deep
    s, _, _ = shares(pred, label)
    assert s["block"] >= 0.99, s
    results["missed cap"] = s

    pred = label.copy()
    pred[2:6, 2:6, 2:6] = True                                             # distant island
    s, category, _ = shares(pred, label)
    assert (category[2:6, 2:6, 2:6] == 1).all() and s["block"] == 1, s
    results["distant island"] = s

    thin = ball(shape, center, 8)
    thin[24:42, 23:26, 23:26] = True                                       # GT: ball + 3x3 tube
    pred = ball(shape, center, 8)                                          # the tube is missed
    s, category, _ = shares(pred, thin)
    assert (category[33:42, 23:26, 23:26] == 1).all(), "missed thin part must be block"
    results["missed thin tube (|d| <= 2, |d~| large)"] = s

    s, _, _ = shares(ball(shape, center, 15), label)                       # uniform 3-voxel shift
    assert s["block"] == 0 and s["shift"] >= 0.99, s
    results["uniform 3-voxel over-segmentation"] = s
    s, _, _ = shares(ball(shape, center, 18), label)                       # uniform 6-voxel shift
    assert s["block"] >= 0.99, s
    results["uniform 6-voxel over-segmentation (block by definition)"] = s
    for name, s in results.items():
        log(f"PASS taxonomy, {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in s.items()))


def check_contributions():
    shape, center = (48, 48, 48), (24, 24, 24)
    label = ball(shape, center, 12)
    pred = ball(shape, center, 13)
    pred[1:9, 1:9, 1:9] = True                          # island with ~12 % of the surface: moves HD95
    _, category, d_pred = shares(pred, label)
    row = A.anatomy_rows(pred, label, category, d_pred, with_hd95=True)
    fixed_all = A.fixed(pred, label, category > 0)
    assert A.dice(fixed_all, label) == 1.0 and np.isclose(row["ddice_all"], 1 - row["dice"])
    assert row["ddice_block"] > 0 and row["ddice_shift"] > 0 and row["ddice_jitter"] >= 0
    assert row["dhd95_block"] > 0, "removing the distant island must lower HD95"
    assert row["n_shift_fp"] == row["n_shift"] and row["n_shift_fn"] == 0
    assert np.isclose(row["ddice_reach"], row["ddice_all"]), "shell and island lie within |d~| <= 6"
    deep = label.copy()
    deep[26:] = False                                   # missed cap 10 voxels deep: |d~| up to 10
    _, category, d_pred = shares(deep, label)
    far = A.anatomy_rows(deep, label, category, d_pred, with_hd95=False)
    assert 0 < far["ddice_reach"] < far["ddice_all"]
    log(f"PASS contributions: dice {row['dice']:.4f}, ddice block {row['ddice_block']:.4f} / shift "
        f"{row['ddice_shift']:.4f} / jitter {row['ddice_jitter']:.4f}, dhd95 block {row['dhd95_block']:.2f}; "
        f"fixing all errors gives Dice 1; reachable share of a 10-voxel-deep missed cap "
        f"{far['ddice_reach'] / far['ddice_all']:.3f} (< 1)")


class PerVoxel(torch.nn.Module):
    """logits [-x, x] per voxel (equivariant to any flip / rotation); lower heads by average pooling."""

    def forward(self, x, deepsup=False):
        full = torch.cat([-x, x], 1)
        heads = [F.avg_pool3d(full, k) for k in (8, 4, 2)] + [full]
        return heads if deepsup else full


def check_tiling():
    rng = np.random.default_rng(3)
    model = PerVoxel()
    for shape, patch in (((37, 41, 29), (16, 16, 12)), ((37, 41, 10), (16, 16, 12))):
        image = rng.standard_normal(shape).astype(np.float32)
        expected = 2 * image   # logit difference of [-x, x]
        partner = lambda r: rng.standard_normal(patch).astype(np.float32)
        augment = lambda r: A.augment_params(r, "LA")
        for mode in ("train", "eval"):
            for flip in ((), (2,), (2, 3, 4)):
                (full, *lower), _ = A.tiled_pass(model, image, patch, rng, "cpu", mode, flip=flip,
                                                 augment=augment, partner=partner, deep=True)
                assert full.shape == shape and np.allclose(full, expected, atol=1e-6), (mode, flip)
                assert len(lower) == 3 and all(p.shape == shape for p in lower)
    constant = np.full((37, 41, 29), 0.7, np.float32)
    (full, *lower), _ = A.tiled_pass(model, constant, (16, 16, 12), rng, "cpu", "eval", deep=True)
    assert all(np.allclose(p, full, atol=1e-6) for p in lower)
    log("PASS tiling: every voxel covered once, LA rot90/flip and the 8 flips inverted exactly (per-voxel model), "
        "volume smaller than the patch padded and cropped back, lower heads resized to full resolution")


def check_auroc():
    labels = np.array([0, 0, 1, 1], bool)
    assert A.auroc(labels, np.array([0.1, 0.2, 0.8, 0.9]))[0] == 1.0
    assert A.auroc(labels, np.array([0.9, 0.8, 0.2, 0.1]))[0] == 0.0
    assert np.isnan(A.auroc(np.ones(3, bool), np.arange(3.0))[0])
    log("PASS auroc: perfect / inverted / single-class cases")


if __name__ == "__main__":
    check_distance()
    check_taxonomy()
    check_contributions()
    check_tiling()
    check_auroc()
    output = PROJECT / "tests/outputs/anatomy"
    output.mkdir(parents=True, exist_ok=True)
    (output / "verify_anatomy.txt").write_text("\n".join(LINES) + "\n")
