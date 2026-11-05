"""Our test protocol (code/evaluate.py) against DuCiSC's utils/test_patch.py, with fixed weights.

  CUDA_VISIBLE_DEVICES=0 python tests/verify_ducisc_test.py
For the first N test cases of LA and Pancreas (one LA_8l Student checkpoint for both):
  - test_single_case_first_output vs predict_volume: bitwise identical masks;
  - calculate_metric_percase (or DuCiSC's zeros for an empty prediction) vs case_metrics: identical values;
  - getLargestCC vs largest_component (nms=1): identical masks and metrics (skipped for an empty
    prediction, where getLargestCC would fail its assert);
  - var_all_case averages vs evaluate() means: identical; test_all_case summary line vs ducisc_line().
"""
import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(PROJECT / "tests"))
import ducisc_reference as reference
from data import SPECS, case_path, read_ids
from evaluate import case_metrics, ducisc_line, evaluate, largest_component, predict_volume, summarize
from networks import VNetDS

KEYS = ("dice", "jc", "hd95", "asd")


def ducisc_metrics(prediction, label):
    """The per-case branch of var_all_case / test_all_case."""
    return (0, 0, 0, 0) if np.sum(prediction) == 0 else reference.calculate_metric_percase(prediction, label)


def same_metrics(ours, theirs):
    return all(ours[key] == float(value) for key, value in zip(KEYS, theirs))


def ducisc_tree(data_root, dataset, ids):
    """The ./data layout DuCiSC's var_all_case expects, as symlinks into our dataset (first N cases only)."""
    tree = Path(tempfile.mkdtemp(prefix="ducisc_data_"))
    (tree / "pred").mkdir()   # test_all_case writes its summary to <test_save_path>/../
    folder = tree / "data" / dataset
    folder.mkdir(parents=True)
    (folder / "test.list").write_text("".join(f"{case}\n" for case in ids))
    if dataset == "LA":
        (folder / "2018LA_Seg_TrainingSet").symlink_to(data_root / "LA/2018LA_Seg_TrainingSet")
        paths = [folder / "2018LA_Seg_TrainingSet" / case / "mri_norm2.h5" for case in ids]
    else:
        (folder / "Pancreas_h5").mkdir()
        for case in ids:   # DuCiSC names the files <case>.h5; the local archive <case>_norm.h5
            (folder / "Pancreas_h5" / f"{case}.h5").symlink_to(case_path(data_root / "Pancreas", dataset, case))
        paths = [folder / "Pancreas_h5" / f"{case}.h5" for case in ids]
    return tree, [str(path) for path in paths]


def check_dataset(model, data_root, dataset, count, device):
    spec = SPECS[dataset]
    root = data_root / spec["directory"]
    patch, (stride_xy, _, stride_z) = spec["patch"], spec["stride"]
    ids = read_ids(root, "test")[:count]
    rows, rows_nms1, ok = [], [], True
    for case in ids:
        with h5py.File(case_path(root, dataset, case), "r") as handle:
            image, label = handle["image"][:], handle["label"][:]
        label[label < 0] = 0
        theirs, _ = reference.test_single_case_first_output(model, image, stride_xy, stride_z, patch, num_classes=2)
        ours = predict_volume(model, image.astype(np.float32), patch, spec["stride"], device)
        mask_equal = np.array_equal(theirs, ours.astype(np.int32))
        metrics = dict(case=case, **case_metrics(ours, label))
        metric_equal = same_metrics(metrics, ducisc_metrics(theirs, label))
        rows.append(metrics)
        ours_cc = largest_component(ours)
        metrics_cc = dict(case=case, **case_metrics(ours_cc, label))
        rows_nms1.append(metrics_cc)
        if np.sum(theirs) == 0:
            cc_note = "nms1 skipped: empty prediction (getLargestCC would assert); ours returns it unchanged"
            cc_equal = np.array_equal(ours_cc, ours) and metrics_cc["empty_prediction"]
        else:
            theirs_cc = reference.getLargestCC(theirs)
            cc_equal = np.array_equal(theirs_cc, ours_cc) and same_metrics(metrics_cc, ducisc_metrics(theirs_cc, label))
            cc_note = f"nms1 mask+metrics equal={cc_equal} (kept {ours_cc.sum()} of {ours.sum()} voxels)"
        print(f"  {dataset} {case}: {ours.sum()} predicted voxels; mask equal={mask_equal}; "
              f"metrics equal={metric_equal} {[round(metrics[key], 6) for key in KEYS]}; {cc_note}")
        ok &= mask_equal and metric_equal and bool(cc_equal)

    tree, paths = ducisc_tree(data_root, dataset, ids)
    cwd = os.getcwd()
    try:
        os.chdir(tree)
        with contextlib.redirect_stdout(io.StringIO()):
            averages = reference.var_all_case(model, 2, patch, stride_xy, stride_z, dataset_name=dataset)
            lines = {nms: reference.test_all_case("check", 1, model, paths, 2, patch, stride_xy, stride_z,
                                                  save_result=False, test_save_path=str(tree / "pred") + "/",
                                                  metric_detail=1, nms=nms)
                     for nms in ((0, 1) if all(row["empty_prediction"] is False for row in rows) else (0,))}
    finally:
        os.chdir(cwd)
    result = evaluate(model, root, dataset, patch, device, split="test", nms1=True, case_limit=count)
    mean_equal = all(result[key] == float(value) for key, value in zip(KEYS, averages))
    print(f"  {dataset} var_all_case averages {[round(float(v), 6) for v in averages]} == evaluate means: {mean_equal}")
    ok &= mean_equal and summarize(rows)["dice"] == result["dice"]
    for nms, line in lines.items():
        ours_line = ducisc_line(result if nms == 0 else result["nms1"])
        print(f"  {dataset} test_all_case nms={nms}: '{line}' == ours: {line == ours_line}")
        ok &= line == ours_line
    return ok


def check_empty_prediction():
    """DuCiSC's convention on a synthetic case: an empty prediction scores zeros and still counts in the mean."""
    label = np.zeros((20, 20, 20), dtype=np.float32)
    label[5:12, 5:12, 5:12] = 1
    empty = np.zeros_like(label, dtype=bool)
    full = label.astype(bool)
    rows = [dict(case="empty", **case_metrics(empty, label)), dict(case="full", **case_metrics(full, label))]
    theirs = [ducisc_metrics(empty.astype(np.int32), label), ducisc_metrics(full.astype(np.int32), label)]
    ok = all(same_metrics(row, values) for row, values in zip(rows, theirs))
    summary = summarize(rows)
    ok &= summary["empty_prediction_cases"] == 1 and summary["dice"] == 0.5 and summary["dice_std"] == 0.5
    ok &= largest_component(empty) is empty   # skipped instead of getLargestCC's assert
    print(f"  synthetic empty prediction: zeros, counted in the mean, nms skipped: {bool(ok)}")
    return bool(ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT.parent / "MT_DS/Results/seed_42/LA_8l/best_model.pth")
    parser.add_argument("--data_root", type=Path, default=PROJECT.parents[1] / "Dataset")
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args()
    device = reference.device
    model = VNetDS(n_channels=1, n_classes=2, has_dropout=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"weights: {args.checkpoint}")
    ok = all([check_dataset(model, args.data_root.resolve(), dataset, args.cases, device)
              for dataset in ("LA", "Pancreas")] + [check_empty_prediction()])
    print("PASS: masks, per-case metrics (nms 0 / 1), averages and summary lines match DuCiSC" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
