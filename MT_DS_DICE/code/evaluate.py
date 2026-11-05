"""Sliding-window inference and metrics, following DuCiSC's utils/test_patch.py.

Equivalent to test_single_case_first_output (padding, window starts, averaged softmax foreground,
threshold 0.5), var_all_case / test_all_case (label[label < 0] = 0; an empty prediction scores
dice = jc = hd95 = asd = 0 and still counts in the mean) and getLargestCC (26-connected components,
the skimage.measure.label default). tests/verify_ducisc_test.py checks this against the original.
"""
import itertools
import math

import h5py
import numpy as np
from scipy import ndimage
import torch
from medpy import metric

from data import SPECS, case_path, read_ids


METRICS = ("dice", "jc", "hd95", "asd")


@torch.no_grad()
def predict_volume(model, image, patch_size, stride, device):
    padding = []
    for size, patch in zip(image.shape, patch_size):
        missing = max(patch - size, 0)
        padding.append((missing // 2, missing - missing // 2))
    padded = np.pad(image, padding)
    scores = np.zeros(padded.shape, dtype=np.float32)
    counts = np.zeros(padded.shape, dtype=np.float32)
    starts = [[min(step * delta, size - patch)
               for step in range(math.ceil((size - patch) / delta) + 1)]
              for size, patch, delta in zip(padded.shape, patch_size, stride)]
    for position in itertools.product(*starts):
        crop = tuple(slice(start, start + patch) for start, patch in zip(position, patch_size))
        inputs = torch.from_numpy(np.ascontiguousarray(padded[crop][None, None])).to(device)
        foreground = model(inputs).softmax(dim=1)[0, 1].cpu().numpy()
        scores[crop] += foreground
        counts[crop] += 1
    prediction = scores / counts > 0.5
    original = tuple(slice(left, left + size)
                     for (left, _), size in zip(padding, image.shape))
    return prediction[original]


def largest_component(prediction):
    """getLargestCC; an empty prediction is returned unchanged instead of failing its assert."""
    components, count = ndimage.label(prediction, structure=np.ones((3, 3, 3)))
    if count == 0:
        return prediction
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1


def case_metrics(prediction, label):
    if np.sum(prediction) == 0:
        return dict(dice=0.0, jc=0.0, hd95=0.0, asd=0.0, empty_prediction=True)
    return dict(dice=float(metric.binary.dc(prediction, label)), jc=float(metric.binary.jc(prediction, label)),
                hd95=float(metric.binary.hd95(prediction, label)), asd=float(metric.binary.asd(prediction, label)),
                empty_prediction=False)


def summarize(rows):
    """Mean over all cases and np.std (population), as DuCiSC prints them."""
    summary = dict(cases=len(rows), empty_prediction_cases=sum(row["empty_prediction"] for row in rows))
    for key in METRICS:
        values = np.array([row[key] for row in rows])
        summary[key], summary[f"{key}_std"] = float(np.mean(values)), float(np.std(values))
    summary["per_case"] = rows
    return summary


def evaluate(model, root, dataset, patch_size, device, split=None, nms1=False, case_limit=None):
    """Metrics without post-processing; with nms1, also after keeping the largest component."""
    split = split or SPECS[dataset]["validation"]
    ids = read_ids(root, split)[:case_limit]
    was_training = model.training
    model.eval()
    rows, rows_nms1 = [], []
    try:
        for case in ids:
            with h5py.File(case_path(root, dataset, case), "r") as handle:
                image = handle["image"][:].astype(np.float32)
                label = handle["label"][:]
            label[label < 0] = 0
            prediction = predict_volume(model, image, patch_size, SPECS[dataset]["stride"], device)
            rows.append(dict(case=case, **case_metrics(prediction, label)))
            if nms1:
                rows_nms1.append(dict(case=case, **case_metrics(largest_component(prediction), label)))
    finally:
        model.train(was_training)
    result = dict(split=split, **summarize(rows))
    if nms1:
        result["nms1"] = summarize(rows_nms1)
    return result


def ducisc_line(result):
    """The summary line DuCiSC's test_all_case prints (std of Dice / JC is not scaled by 100 there)."""
    return ", ".join(f"{name}: {result[key] * scale:.2f}({result[key + '_std']:.2f})" for name, key, scale in (
        ("dice", "dice", 100), ("jc", "jc", 100), ("hd", "hd95", 1), ("asd", "asd", 1)))
