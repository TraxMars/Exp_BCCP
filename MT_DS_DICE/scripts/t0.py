"""Offline T0: the Teacher's inference mode vs pseudo-label quality (README.md, "T0").

  CUDA_VISIBLE_DEVICES=<gpu> python scripts/t0.py --run <V0 run dir> --output Results_T0/seed_<s>/<run>
Runs P0-P6 on every unlabeled case of one V0 run; writes cases.csv (one row per case and mode) and summary.json.
Reuses scripts/anatomy.py (tiles, partner crops, augmentation, taxonomy) without changing it.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.special import expit
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import anatomy as A   # noqa: E402  (also puts code/ on sys.path)
from data import SPECS, case_path, load_case   # noqa: E402
from networks.vnet_ds import VNetDS   # noqa: E402

MODES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6")
# (BN: batch statistics?, Dropout on?) per mode; P4 is eval throughout
SETTINGS = dict(P0=(True, True), P1=(True, False), P2=(False, True), P3=(False, False), P4=(False, False),
                P5=(True, True), P6=(True, True))
REPEATS = 8


def configure(model, batch_statistics, dropout):
    """BN mode from model.train / eval; the one Dropout3d module (used at x5 and x9) set separately."""
    model.train(batch_statistics)
    model.dropout.train(dropout)
    return model


def load_models(state, device):
    models = {}
    for mode in MODES:
        model = VNetDS(n_channels=1, n_classes=2, has_dropout=True).to(device)
        model.load_state_dict(state)
        models[mode] = configure(model.requires_grad_(False), *SETTINGS[mode])
    return models


def partner_sampler(volumes, index, patch, augment):
    """anatomy's partner: a random crop of another unlabeled case, augmented as in data.py."""
    others = [i for i in range(len(volumes)) if i != index]

    def partner(rng):
        image = A.random_crop(volumes[others[int(rng.integers(len(others)))]][0], patch, rng)
        return A.apply_augment(torch.from_numpy(np.ascontiguousarray(image))[None, None], augment(rng))[0, 0].numpy()
    return partner


def reset(seed, index):
    torch.manual_seed(seed * 1000 + index)
    return np.random.default_rng(seed * 1000 + index)


@torch.no_grad()
def sliding(model, image, patch, stride, device, rng=None, augment=None, partner=None):
    """evaluate.predict_volume's padding, windows and averaging; returns the averaged foreground probability.
    With a partner (P6) every window is forwarded as in P0: LA augmentation and the partner in the batch."""
    padding = []
    for size, p in zip(image.shape, patch):
        missing = max(p - size, 0)
        padding.append((missing // 2, missing - missing // 2))
    padded = np.pad(image, padding)
    scores = np.zeros(padded.shape, dtype=np.float32)
    counts = np.zeros(padded.shape, dtype=np.float32)
    starts = [[min(step * delta, size - p) for step in range(math.ceil((size - p) / delta) + 1)]
              for size, p, delta in zip(padded.shape, patch, stride)]
    for position in itertools.product(*starts):
        crop = tuple(slice(start, start + p) for start, p in zip(position, patch))
        x = torch.from_numpy(np.ascontiguousarray(padded[crop][None, None])).to(device)
        params = augment(rng) if augment else (0, None)
        x = A.apply_augment(x, params)
        if partner is not None:
            x = torch.cat([x, torch.from_numpy(np.ascontiguousarray(partner(rng)))[None, None].to(device)])
        foreground = model(x)[:1].softmax(dim=1)[:, 1:]
        scores[crop] += A.invert_augment(foreground, params)[0, 0].cpu().numpy()
        counts[crop] += 1
    original = tuple(slice(left, left + size) for (left, _), size in zip(padding, image.shape))
    return (scores / counts)[original]


class MaskRecorder:
    """Channel Dropout masks of the first sample: dropped = output channel all zero, unknown where the input
    channel is all zero (dropped or not, the output is zero)."""

    def __init__(self, module):
        self.calls, self.handle = [], module.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        x, y = inputs[0][0].flatten(1), output[0].flatten(1)
        known = (x != 0).any(1)
        self.calls.append((((y == 0).all(1) & known).cpu(), known.cpu()))

    def remove(self):
        self.handle.remove()


def same_masks(a, b):
    if len(a.calls) != len(b.calls):
        return False
    return all(torch.equal(da[ka & kb], db[ka & kb]) for (da, ka), (db, kb) in zip(a.calls, b.calls))


def mode_row(pred, confidence, label, d_label, reference):
    d_pred = A.signed_distance(pred)
    category = A.classify_errors(pred, label, d_label, d_pred)
    row = A.anatomy_rows(pred, label, category, d_pred, with_hd95=False)
    row["hd95"] = A.hd95(pred, label)
    error, band = pred != label, np.abs(d_label) <= 2
    row["err_gt_band"] = float(error[band].mean())
    row["conf_wrong_095"] = float((confidence[error & band] > A.CONFIDENT).mean()) if (error & band).any() else math.nan
    row["agree_p0"] = A.dice(pred, reference) if reference is not None else 1.0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case_limit", type=int)
    args = parser.parse_args()
    started = time.time()
    config = json.loads((args.run / "config.json").read_text())
    run_args = config["args"]
    dataset, seed = run_args["dataset"], int(run_args["seed"])
    patch, stride = tuple(run_args["patch_size"]), SPECS[dataset]["stride"]
    root = Path(run_args["data_root"]) / SPECS[dataset]["directory"]   # as train.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True   # as the anatomy

    checkpoint = torch.load(args.run / "last_checkpoint.pth", map_location="cpu", weights_only=False)
    models = load_models(checkpoint["teacher"], device)
    ids = config["unlabeled_ids"]
    volumes = [(image, label.astype(bool)) for image, label in (load_case(case_path(root, dataset, c)) for c in ids)]
    augment = lambda r: A.augment_params(r, dataset)

    rows, mode_seconds, checks = [], {mode: 0.0 for mode in MODES}, {}
    for index, case in enumerate(ids[:args.case_limit]):
        tick = time.time()
        image, label = volumes[index]
        partner = partner_sampler(volumes, index, patch, augment)
        outputs, offsets = {}, {}
        recorders = {m: MaskRecorder(models[m].dropout) for m in ("P0", "P2")} if index == 0 else {}
        for mode in ("P0", "P1", "P2", "P3"):   # same grid, tile order, augmentation, partners; P0 / P2 same masks
            start = time.time()
            rng = reset(seed, index)
            (z,), offsets[mode] = A.tiled_pass(models[mode], image, patch, rng, device, "train",
                                               augment=augment, partner=partner)
            outputs[mode] = (z > 0, expit(np.abs(z.astype(np.float64))))
            mode_seconds[mode] += time.time() - start
        assert all(offsets[m] == offsets["P0"] for m in offsets), offsets
        if recorders:
            checks["dropout_masks_p0_p2_identical_case0"] = same_masks(recorders["P0"], recorders["P2"])
            checks["dropout_calls_case0"] = len(recorders["P0"].calls)
            for recorder in recorders.values():
                recorder.remove()

        start = time.time()
        p = sliding(models["P4"], image, patch, stride, device)
        outputs["P4"] = (p > 0.5, np.maximum(p, 1 - p))
        mode_seconds["P4"] += time.time() - start

        start = time.time()
        total = np.zeros(image.shape, np.float64)
        for repeat in range(REPEATS):
            torch.manual_seed((seed * 1000 + index) * 100 + repeat + 1)
            rng = np.random.default_rng([seed * 1000 + index, repeat + 1])
            (z,), _ = A.tiled_pass(models["P5"], image, patch, rng, device, "train", augment=augment,
                                   partner=partner, offsets=offsets["P0"])
            total += z
        z = total / REPEATS
        outputs["P5"] = (z > 0, expit(np.abs(z)))
        mode_seconds["P5"] += time.time() - start

        start = time.time()
        torch.manual_seed((seed * 1000 + index) * 100 + 99)
        p = sliding(models["P6"], image, patch, stride, device, rng=np.random.default_rng([seed * 1000 + index, 99]),
                    augment=augment, partner=partner)
        outputs["P6"] = (p > 0.5, np.maximum(p, 1 - p))
        mode_seconds["P6"] += time.time() - start
        gpu_seconds = time.time() - tick

        d_label = A.signed_distance(label)
        reference = outputs["P0"][0]
        for mode in MODES:
            pred, confidence = outputs[mode]
            rows.append(dict(case=case, mode=mode, **mode_row(pred, confidence, label, d_label,
                                                               None if mode == "P0" else reference)))
        summary = "  ".join(f"{m} {r['dice'] * 100:.2f}" for m, r in zip(MODES, rows[-len(MODES):]))
        print(f"[{index + 1}/{len(ids[:args.case_limit])}] {case}: {summary}  "
              f"(gpu {gpu_seconds:.0f} s, total {time.time() - tick:.0f} s)", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    git = subprocess.run(["git", "-C", str(PROJECT), "describe", "--always", "--dirty", "--abbrev=40"],
                         capture_output=True, text=True).stdout.strip()
    summary = dict(run=str(args.run.resolve()), dataset=dataset, seed=seed, iteration=checkpoint["iteration"],
                   run_commit=config["git_commit"], t0_commit=git, cases=len(rows) // len(MODES),
                   seconds=time.time() - started, mode_seconds=mode_seconds, device=str(device),
                   gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None, **checks)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
