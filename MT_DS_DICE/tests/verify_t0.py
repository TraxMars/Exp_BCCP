"""Checks of scripts/t0.py (CPU): mode settings, comparability of P0-P3, sliding windows vs evaluate.predict_volume.

  python tests/verify_t0.py   -> tests/outputs/t0/verify_t0.txt
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import t0 as T   # noqa: E402
import anatomy as A   # noqa: E402
from evaluate import predict_volume   # noqa: E402
from networks.vnet_ds import VNetDS   # noqa: E402

LINES = []


def log(text):
    print(text, flush=True)
    LINES.append(text)


def models():
    torch.manual_seed(0)
    state = VNetDS(n_channels=1, n_classes=2, has_dropout=True).state_dict()
    return T.load_models(state, "cpu")


def check_settings(m):
    for mode, (batch_statistics, dropout) in T.SETTINGS.items():
        bn = [module.training for module in m[mode].modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        assert all(flag == batch_statistics for flag in bn) and m[mode].dropout.training == dropout, mode
    assert len({id(model) for model in m.values()}) == len(T.MODES)
    log("PASS settings: P0 batch BN + Dropout, P1 batch BN only, P2 running BN + Dropout, P3/P4 eval, P5/P6 as P0; "
        "one model copy per mode")


def check_partner_does_not_leak(m):
    x, other = torch.randn(1, 1, 32, 32, 32), torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        alone, paired = m["P3"](x), m["P3"](torch.cat([x, other]))[:1]
    difference = (alone - paired).abs().max().item()
    assert difference < 1e-4, difference
    with torch.no_grad():
        alone_bn, paired_bn = m["P1"](x), m["P1"](torch.cat([x, other]))[:1]
    assert (alone_bn - paired_bn).abs().max().item() > 1e-3, "batch-statistics BN must couple the samples"
    log(f"PASS partner: with running BN the partner leaves the tile's output unchanged (max diff {difference:.1e}); "
        f"with batch statistics it changes it (control)")


def check_comparability(m):
    rng0 = np.random.default_rng(1)
    volumes = [(rng0.standard_normal((40, 36, 34)).astype(np.float32), None) for _ in range(3)]
    augment = lambda r: A.augment_params(r, "LA")
    patch = (32, 32, 32)
    draws, recorders, offsets = {}, {mode: T.MaskRecorder(m[mode].dropout) for mode in ("P0", "P2")}, {}
    for mode in ("P0", "P1", "P2", "P3"):
        rng = T.reset(42, 0)
        sampler = T.partner_sampler(volumes, 0, patch, augment)
        record = []
        partner = lambda r: record.append(sampler(r)) or record[-1]
        _, offsets[mode] = A.tiled_pass(m[mode], volumes[0][0], patch, rng, "cpu", "train", augment=augment,
                                        partner=partner)
        draws[mode] = record
    assert all(offsets[k] == offsets["P0"] for k in offsets)
    assert all(len(draws[k]) == len(draws["P0"]) and all(np.array_equal(a, b) for a, b in zip(draws[k], draws["P0"]))
               for k in draws)
    assert recorders["P0"].calls and T.same_masks(recorders["P0"], recorders["P2"])
    control = T.MaskRecorder(m["P5"].dropout)
    rng = np.random.default_rng(7)
    torch.manual_seed(7)
    A.tiled_pass(m["P5"], volumes[0][0], patch, rng, "cpu", "train", augment=augment,
                 partner=T.partner_sampler(volumes, 0, patch, augment), offsets=offsets["P0"])
    assert not T.same_masks(recorders["P0"], control), "another seed must give other masks"
    log(f"PASS comparability: P0-P3 share grid offsets {offsets['P0']}, tile order, LA augmentation and "
        f"{len(draws['P0'])} partner crops; P0 / P2 Dropout masks identical over {len(recorders['P0'].calls)} "
        f"Dropout calls (another seed: different)")


def check_sliding(m):
    rng = np.random.default_rng(2)
    image = rng.standard_normal((40, 36, 30)).astype(np.float32)   # z smaller than the patch: padded
    patch, stride = (32, 32, 32), (6, 6, 4)
    p = T.sliding(m["P4"], image, patch, stride, "cpu")
    reference = predict_volume(m["P4"], image, patch, stride, "cpu")
    assert np.array_equal(p > 0.5, reference)

    class PerVoxel(torch.nn.Module):
        def forward(self, x, deepsup=False):
            return torch.cat([-x, x], 1)
    sampler = T.partner_sampler([(image, None), (image[::-1].copy(), None)], 0, patch,
                                lambda r: A.augment_params(r, "LA"))
    p6 = T.sliding(PerVoxel(), image, patch, stride, "cpu", rng=np.random.default_rng(3),
                   augment=lambda r: A.augment_params(r, "LA"), partner=sampler)
    assert np.allclose(p6, torch.sigmoid(2 * torch.from_numpy(image)).numpy(), atol=1e-6)
    log("PASS sliding: eval windows = evaluate.predict_volume exactly (padding included); with partner and LA "
        "augmentation every window is turned back exactly (per-voxel model)")


if __name__ == "__main__":
    m = models()
    check_settings(m)
    check_partner_does_not_leak(m)
    check_comparability(m)
    check_sliding(m)
    output = PROJECT / "tests/outputs/t0"
    output.mkdir(parents=True, exist_ok=True)
    (output / "verify_t0.txt").write_text("\n".join(LINES) + "\n")
