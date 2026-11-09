"""Offline boundary-error anatomy of one V0 Teacher (README.md, "Boundary-error anatomy").

  CUDA_VISIBLE_DEVICES=<gpu> python scripts/anatomy.py --run <V0 run dir> --output Results_anatomy/seed_<s>/<run>
Writes cases.csv (per case), layers.csv (per pseudo layer, pooled over cases), separability.csv and
summary.json. Nothing here changes training.
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
from scipy import ndimage
from scipy.special import expit, logit
from medpy import metric
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from data import SPECS, case_path, load_case   # noqa: E402
from evaluate import predict_volume   # noqa: E402
from networks.vnet_ds import VNetDS   # noqa: E402


CATEGORIES = ("block", "shift", "jitter")   # codes 1, 2, 3 in the category map
DEPTH = 3            # block: a voxel farther than this from both boundaries
WINDOW, MIN_ERRORS, BIAS = 17, 20, 0.4   # shift: |FP - FN| >= 0.4 (FP + FN) in the window
LAYERS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-6", 4, 6), (">6", 7, math.inf))
REACH = 6            # V1's D2 band |d̃| <= 3 at full resolution
CONFIDENT = 0.95
SIGNALS = ("entropy", "ds_disagreement", "flip_eval", "tta_train")
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
FLIPS = [dims for k in range(4) for dims in itertools.combinations((2, 3, 4), k)]   # 8 flips
FAR = 10 ** 6
LAYER_COUNTS = ("n_voxels", "n_err", "n_err_fp", "n_conf095", "n_err_conf095", *[f"n_{c}" for c in CATEGORIES])


# ---------------------------------------------------------------- distances and taxonomy

def signed_distance(mask):
    """Chebyshev distance to the other side: background +k, foreground −k; volume borders are not boundaries."""
    if not mask.any():
        return np.full(mask.shape, FAR, np.int32)
    if mask.all():
        return np.full(mask.shape, -FAR, np.int32)
    outside = ndimage.distance_transform_cdt(~mask, metric="chessboard")
    inside = ndimage.distance_transform_cdt(mask, metric="chessboard")
    return np.where(mask, -inside, outside).astype(np.int32)


def window_counts(mask, size):
    """Number of mask voxels in the size³ window centred on each voxel (zeros outside the volume)."""
    return np.rint(ndimage.uniform_filter(mask.astype(np.float64), size=size, mode="constant") * size ** 3)


def classify_errors(pred, label, d_label, d_pred):
    """Category map: 0 correct, 1 block, 2 shift, 3 jitter."""
    fp, fn = pred & ~label, ~pred & label
    depth = np.maximum(np.abs(d_label), np.abs(d_pred))
    block = np.zeros(pred.shape, bool)
    for errors in (fp, fn):
        # deep errors plus the errors of the same type within DEPTH steps of them (the region's own rim); whole
        # components would not do: the thin shell along the surface connects every error of a case
        deep = errors & (depth > DEPTH)
        if deep.any():
            block |= ndimage.binary_dilation(deep, structure=np.ones((3, 3, 3)), iterations=DEPTH, mask=errors)
    boundary = (fp | fn) & ~block
    n_fp, n_fn = window_counts(boundary & fp, WINDOW), window_counts(boundary & fn, WINDOW)
    total = n_fp + n_fn
    shift = boundary & (total >= MIN_ERRORS) & (np.abs(n_fp - n_fn) >= BIAS * total)
    category = np.zeros(pred.shape, np.uint8)
    category[block], category[shift], category[boundary & ~shift] = 1, 2, 3
    return category


def dice(pred, label):
    total = pred.sum() + label.sum()
    return 2.0 * (pred & label).sum() / total if total else 1.0


def hd95(pred, label):
    if not pred.any() or not label.any():
        return math.nan
    if np.array_equal(pred, label):
        return 0.0
    return float(metric.binary.hd95(pred, label))


def fixed(pred, label, mask):
    return np.where(mask, label, pred)


def anatomy_rows(pred, label, category, d_pred, with_hd95):
    """Counts and metric contributions of one case's categories."""
    row = dict(dice=dice(pred, label), n_err=int((category > 0).sum()),
               n_fp=int((pred & ~label).sum()), n_fn=int((~pred & label).sum()))
    if with_hd95:
        row["hd95"] = hd95(pred, label)
    for code, name in enumerate(CATEGORIES, 1):
        mask = category == code
        row[f"n_{name}"] = int(mask.sum())
        row[f"ddice_{name}"] = dice(fixed(pred, label, mask), label) - row["dice"]
        if with_hd95:
            row[f"dhd95_{name}"] = row["hd95"] - hd95(fixed(pred, label, mask), label)
    shift = category == 2
    row["n_shift_fp"], row["n_shift_fn"] = int((shift & pred).sum()), int((shift & ~pred).sum())
    reach = (category > 0) & (np.abs(d_pred) <= REACH)
    row["ddice_reach"] = dice(fixed(pred, label, reach), label) - row["dice"]
    row["ddice_all"] = 1.0 - row["dice"]
    return row


def layer_index(d_pred):
    absolute = np.abs(d_pred)
    index = np.full(d_pred.shape, -1, np.int8)
    for i, (_, low, high) in enumerate(LAYERS):
        index[(absolute >= low) & (absolute <= high)] = i
    return index


# ---------------------------------------------------------------- inference

def augment_params(rng, dataset):
    if SPECS[dataset]["rotation"]:
        return int(rng.integers(0, 4)), int(rng.integers(0, 2))
    return 0, None


def apply_augment(x, params):
    """data.py: flip(rot90(image, k), axis) on tensors [B, C, X, Y, Z]."""
    k, axis = params
    x = torch.rot90(x, k, dims=(2, 3))
    return x if axis is None else torch.flip(x, dims=[2 + axis])


def invert_augment(x, params):
    k, axis = params
    if axis is not None:
        x = torch.flip(x, dims=[2 + axis])
    return torch.rot90(x, -k, dims=(2, 3))


def random_crop(image, patch, rng):
    """data.py's padding and uniform crop, numpy only."""
    if any(size <= p for size, p in zip(image.shape, patch)):
        pads = [max((p - size) // 2 + 3, 0) for size, p in zip(image.shape, patch)]
        image = np.pad(image, [(pad, pad) for pad in pads])
    starts = [int(rng.integers(0, size - p + 1)) for size, p in zip(image.shape, patch)]
    return image[tuple(slice(s, s + p) for s, p in zip(starts, patch))]


def tile_starts(size, patch, offset):
    return sorted({min(max(start, 0), size - patch) for start in range(offset - patch, size, patch)})


def tiles(shape, patch, offsets, rng):
    """Tile crops in random order; with first-come assignment every voxel gets exactly one tile."""
    grid = list(itertools.product(*[tile_starts(s, p, o) for s, p, o in zip(shape, patch, offsets)]))
    order = rng.permutation(len(grid))
    return [tuple(slice(start, start + p) for start, p in zip(grid[i], patch)) for i in order]


def pad_to_patch(image, patch):
    padding = [((max(p - s, 0)) // 2, max(p - s, 0) - max(p - s, 0) // 2) for s, p in zip(image.shape, patch)]
    original = tuple(slice(left, left + s) for (left, _), s in zip(padding, image.shape))
    return np.pad(image, padding), original


@torch.no_grad()
def tiled_pass(model, image, patch, rng, device, mode, flip=(), augment=None, partner=None, deep=False,
               offsets=None):
    """One forward per voxel. mode 'train': batch = tile + partner crop (train-mode BN couples them);
    mode 'eval': tiles alone. Returns the full-resolution logit difference z = logit_fg - logit_bg (and the three
    lower heads' z, trilinear to full resolution, when deep). z instead of probabilities: confident voxels
    saturate float32 probabilities."""
    padded, original = pad_to_patch(image, patch)
    if offsets is None:
        offsets = [int(rng.integers(0, p)) for p in patch]
    outputs = [np.zeros(padded.shape, np.float32) for _ in range(4 if deep else 1)]
    filled = np.zeros(padded.shape, bool)
    for crop in tiles(padded.shape, patch, offsets, rng):
        x = torch.from_numpy(np.ascontiguousarray(padded[crop]))[None, None].to(device)
        params = augment(rng) if augment else (0, None)
        x = apply_augment(x, params)
        if flip:
            x = torch.flip(x, dims=list(flip))
        if mode == "train":
            other = torch.from_numpy(np.ascontiguousarray(partner(rng)))[None, None].to(device)
            x = torch.cat([x, other])
        heads = model(x, deepsup=True)
        heads = heads if deep else heads[-1:]
        differences = []
        for head in heads[::-1]:   # full resolution first
            z = head[:1, 1:] - head[:1, :1]
            if z.shape[2:] != x.shape[2:]:
                z = F.interpolate(z, size=x.shape[2:], mode="trilinear", align_corners=False)
            if flip:
                z = torch.flip(z, dims=list(flip))
            differences.append(invert_augment(z, params)[0, 0].cpu().numpy())
        new = ~filled[crop]
        for output, z in zip(outputs, differences):
            output[crop][new] = z[new]
        filled[crop] = True
    assert filled.all()
    return [output[original] for output in outputs], offsets


# ---------------------------------------------------------------- separability

def auroc(labels, scores):
    if labels.all() or not labels.any():
        return math.nan, math.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case_limit", type=int)
    parser.add_argument("--no_reference", action="store_true", help="skip the sliding-window reference")
    args = parser.parse_args()
    started = time.time()
    config = json.loads((args.run / "config.json").read_text())
    run_args = config["args"]
    dataset, seed = run_args["dataset"], int(run_args["seed"])
    patch = tuple(run_args["patch_size"])
    root = Path(run_args["data_root"]) / SPECS[dataset]["directory"]   # as train.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(args.run / "last_checkpoint.pth", map_location="cpu", weights_only=False)
    models = {}
    for mode in ("train", "eval"):   # separate copies: train-mode forwards move the BN running buffers
        model = VNetDS(n_channels=1, n_classes=2, has_dropout=True).to(device)
        model.load_state_dict(checkpoint["teacher"])
        models[mode] = model.requires_grad_(False).train(mode == "train")

    ids = config["unlabeled_ids"]
    volumes = [load_case(case_path(root, dataset, case)) for case in ids]
    volumes = [(image, label.astype(bool)) for image, label in volumes]
    ids, cases = ids[:args.case_limit], volumes[:args.case_limit]

    rng = np.random.default_rng(seed + 2026)
    augment = lambda r: augment_params(r, dataset)

    def partner_for(index):
        others = [i for i in range(len(volumes)) if i != index]

        def partner(r):
            image = random_crop(volumes[others[int(r.integers(len(others)))]][0], patch, r)
            return apply_augment(torch.from_numpy(np.ascontiguousarray(image))[None, None], augment(r))[0, 0].numpy()
        return partner

    case_rows, layer_counts = [], np.zeros((len(LAYERS), len(LAYER_COUNTS)), np.int64)
    pooled = {name: [] for name in ("layer", "error", "case", *SIGNALS)}
    for index, (case, (image, label)) in enumerate(zip(ids, cases)):
        tick = time.time()
        torch.manual_seed(seed * 1000 + index)
        partner = partner_for(index)
        (z0, *lower), offsets = tiled_pass(models["train"], image, patch, rng, device, "train",
                                           augment=augment, partner=partner, deep=True)
        pred = z0 > 0   # argmax (a tie goes to background)
        sign = np.where(pred, 1.0, -1.0)
        other = lambda z: expit(-sign * z)   # probability of the class other than the pseudo class, unsaturated
        tta = np.mean([other(tiled_pass(models["train"], image, patch, rng, device, "train", flip=flip,
                                        partner=partner)[0][0]) for flip in FLIPS], axis=0)
        flip_eval = np.mean([other(tiled_pass(models["eval"], image, patch, rng, device, "eval", flip=flip,
                                              offsets=offsets)[0][0]) for flip in FLIPS], axis=0)
        reference = None if args.no_reference else predict_volume(
            models["eval"], image, patch, SPECS[dataset]["stride"], device)
        gpu_seconds = time.time() - tick

        confidence = expit(np.abs(z0.astype(np.float64)))
        # two classes: entropy is a decreasing function of |z|, so -|z| ranks voxels exactly as entropy does
        scores = dict(entropy=-np.abs(z0), ds_disagreement=np.mean([np.abs(other(z) - other(z0)) for z in lower], axis=0),
                      flip_eval=flip_eval, tta_train=tta)
        d_label, d_pred = signed_distance(label), signed_distance(pred)
        category = classify_errors(pred, label, d_label, d_pred)
        row = dict(case=case, **anatomy_rows(pred, label, category, d_pred, with_hd95=True))
        error = category > 0
        band = np.abs(d_label) <= 2   # diagnostics.boundary_band(radius=2)
        row["err_gt_band"] = float(error[band].mean())
        row["conf_wrong_095"] = float((confidence[error & band] > CONFIDENT).mean()) if (error & band).any() else math.nan
        for t in THRESHOLDS:   # p_fg >= t  <=>  z >= logit(t)
            row[f"dice_t{t:.2f}"] = dice(z0 >= logit(t), label)
        if reference is not None:
            d_ref = signed_distance(reference)
            ref = anatomy_rows(reference, label, classify_errors(reference, label, d_label, d_ref), d_ref, False)
            row.update({f"ref_{key}": value for key, value in ref.items()})

        layer = layer_index(d_pred)
        for i in range(len(LAYERS)):
            in_layer = layer == i
            confident = in_layer & (confidence > CONFIDENT)
            layer_counts[i] += [in_layer.sum(), (error & in_layer).sum(), (error & in_layer & pred).sum(),
                                confident.sum(), (error & confident).sum(),
                                *[((category == code) & in_layer).sum() for code in (1, 2, 3)]]
        selected = (layer <= 2) & (layer >= 0) & (confidence > CONFIDENT)
        pooled["layer"].append(layer[selected])
        pooled["error"].append(error[selected])
        pooled["case"].append(np.full(int(selected.sum()), index, np.int16))
        for name in SIGNALS:
            pooled[name].append(scores[name][selected].astype(np.float32))
        row["seconds"] = time.time() - tick
        row["gpu_seconds"] = gpu_seconds
        case_rows.append(row)
        print(f"[{index + 1}/{len(ids)}] {case}: dice {row['dice']:.4f}, errors {row['n_err']} "
              f"(block {row['n_block']}, shift {row['n_shift']}, jitter {row['n_jitter']}), "
              f"{row['seconds']:.1f} s", flush=True)

    pooled = {name: np.concatenate(values) for name, values in pooled.items()}
    separability = []
    for i, (name, _, _) in enumerate(LAYERS[:3]):
        in_layer = pooled["layer"] == i
        labels = pooled["error"][in_layer]
        for signal in SIGNALS:
            values = pooled[signal][in_layer]
            area, precision = auroc(labels, values)
            per_case = []
            for case_index in np.unique(pooled["case"][in_layer]):
                mine = pooled["case"][in_layer] == case_index
                if min(labels[mine].sum(), (~labels[mine]).sum()) >= 10:
                    per_case.append(auroc(labels[mine], values[mine])[0])
            separability.append(dict(layer=name, signal=signal, auroc=area, ap=precision,
                                     base_rate=float(labels.mean()) if labels.size else math.nan,
                                     n=int(labels.size), n_err=int(labels.sum()),
                                     case_median_auroc=float(np.median(per_case)) if per_case else math.nan,
                                     n_cases=len(per_case)))
    for signal in SIGNALS:   # unstratified, layers 1-3 together (confound with the distance; not used)
        area, precision = auroc(pooled["error"], pooled[signal])
        separability.append(dict(layer="1-3 pooled", signal=signal, auroc=area, ap=precision,
                                 base_rate=float(pooled["error"].mean()), n=int(pooled["error"].size),
                                 n_err=int(pooled["error"].sum()), case_median_auroc=math.nan, n_cases=0))

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cases.csv", case_rows)
    write_csv(args.output / "layers.csv", [dict(layer=name, **{k: int(v) for k, v in zip(LAYER_COUNTS, counts)})
                                           for (name, _, _), counts in zip(LAYERS, layer_counts)])
    write_csv(args.output / "separability.csv", separability)
    git = subprocess.run(["git", "-C", str(PROJECT), "describe", "--always", "--dirty", "--abbrev=40"],
                         capture_output=True, text=True).stdout.strip()
    summary = dict(run=str(args.run.resolve()), dataset=dataset, seed=seed, iteration=checkpoint["iteration"],
                   run_commit=config["git_commit"], anatomy_commit=git, cases=len(case_rows),
                   reference=not args.no_reference, seconds=time.time() - started, device=str(device),
                   gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
