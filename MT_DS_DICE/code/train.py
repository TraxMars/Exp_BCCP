"""MT_DS_PL: Mean Teacher + four-scale deep supervision on DuCiSC's VNet with pseudo-label consistency.

L = L_sup + 0.3 * L_pl from iteration 2000 (L_sup before), where both are the four-scale mean of
(CE + Dice) / 2: L_sup to the nearest-neighbour GT, L_pl to each scale's own Teacher argmax.

The GPU is chosen by the caller through CUDA_VISIBLE_DEVICES (scripts/launch.py sets it).
"""
import argparse
import contextlib
import copy
import csv
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SPECS, VolumeDataset, TwoStreamBatchSampler
from diagnostics import boundary_diagnostics, gradient_diagnostics, labeled_band_ce
from evaluate import ducisc_line, evaluate
from networks import VNetDS
from objectives import pseudo_label_consistency, supervised_loss, update_ema
from v1_probe import FIELDS as V1_FIELDS, v1_probe


PROJECT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(SPECS), required=True)
    parser.add_argument("--data_root", type=Path, default=PROJECT.parents[1] / "Dataset")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--labelnum", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    for name, default in (("seed", 42), ("batch_size", 4), ("labeled_bs", 2), ("max_iterations", 15000),
                          ("warmup_iterations", 2000), ("base_lr", 0.01), ("ema_decay", 0.99),
                          ("consistency", 0.3), ("num_workers", 2), ("cpu_threads", 4), ("log_interval", 10),
                          ("val_interval", 200), ("val_start", 800), ("checkpoint_interval", 1000)):
        parser.add_argument(f"--{name}", type=type(default), default=default)
    # Logging-only diagnostics (diagnostics.csv) every N iterations; 0 disables them.
    parser.add_argument("--diag_interval", type=int, default=50)
    # V1 probe (v1_probe.csv, logging only) on the same steps as the diagnostics; off by default.
    parser.add_argument("--v1_probe", action="store_true")
    # Keep the training volumes in memory (sampling and augmentation are unchanged).
    parser.add_argument("--cache_volumes", action=argparse.BooleanOptionalAction, default=True)
    # Resuming restores the configuration stored in the checkpoint; other arguments are ignored.
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--patch_size", nargs=3, type=int)
    args = parser.parse_args()
    if not 0 < args.labeled_bs < args.batch_size:
        parser.error("Require 0 < labeled_bs < batch_size")
    if args.v1_probe and not args.diag_interval:
        parser.error("--v1_probe runs on diagnostics steps; it needs --diag_interval > 0")
    return args


def git_commit():
    result = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"],
                            cwd=PROJECT, capture_output=True, text=True)
    return result.stdout.strip() or None


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train(args):
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**dict(dict(v1_probe=False), **checkpoint["config"]["args"], resume=args.resume))
        args.data_root, args.output_dir = Path(args.data_root), Path(args.output_dir)
    spec = SPECS[args.dataset]
    args.data_root = args.data_root.resolve()
    root = args.data_root / spec["directory"]
    args.labelnum = args.labelnum if args.labelnum is not None else spec["labels"]
    args.patch_size = list(args.patch_size or spec["patch"])
    dataset = VolumeDataset(root, args.dataset, args.patch_size, cache=args.cache_volumes)
    if not args.labeled_bs <= args.labelnum <= len(dataset) - (args.batch_size - args.labeled_bs):
        raise ValueError("Invalid labeled count for this dataset/batch size")
    sampler = TwoStreamBatchSampler(list(range(args.labelnum)), list(range(args.labelnum, len(dataset))),
                                    args.batch_size, args.labeled_bs)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    student = VNetDS(n_channels=1, n_classes=2, has_dropout=True).to(args.device)
    teacher = copy.deepcopy(student).requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    config = dict(args=dict({key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()}, resume=None),
                  git_commit=git_commit(), labeled_ids=dataset.ids[:args.labelnum],
                  unlabeled_ids=dataset.ids[args.labelnum:])
    iteration, best_dice = 0, -1.0
    if checkpoint is not None:
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iteration, best_dice = checkpoint["iteration"], checkpoint["best_dice"]
        if (args.output_dir / "best_metrics.json").is_file():
            # best_model.pth may be newer than the last periodic checkpoint.
            best_dice = max(best_dice, json.loads((args.output_dir / "best_metrics.json").read_text())["dice"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if args.device == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(args.output_dir / "log.txt"), logging.StreamHandler(sys.stdout)])
    logging.info("config: %s", config)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                        pin_memory=args.device == "cuda", worker_init_fn=worker_init)

    def save_checkpoint():
        torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(),
                        optimizer=optimizer.state_dict(), iteration=iteration, best_dice=best_dice,
                        config=config, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None),
                   args.output_dir / "last_checkpoint.pth.tmp")
        (args.output_dir / "last_checkpoint.pth.tmp").replace(args.output_dir / "last_checkpoint.pth")

    start, data_wait = time.monotonic(), 0.0
    fields = ["iteration", "loss_supervised", "ce", "dice", "loss_pl", "pl_ce", "pl_dice",
              "consistency_weight", "loss_total", "gradient_norm", "lr", "peak_memory_mib", "elapsed_seconds",
              "data_wait_seconds"]
    with (args.output_dir / "training_metrics.csv").open("a", newline="", buffering=1) as handle, \
            (args.output_dir / "diagnostics.csv").open("a", newline="", buffering=1) as diagnostics_handle, \
            contextlib.ExitStack() as v1_files:
        metrics, diagnostics, v1_writer = csv.DictWriter(handle, fieldnames=fields), None, None
        if handle.tell() == 0:
            metrics.writeheader()
        while iteration < args.max_iterations:
            student.train()
            teacher.train()
            fetch = time.monotonic()
            for batch in loader:
                data_wait += time.monotonic() - fetch
                images = batch["image"].to(args.device, non_blocking=True)
                # Only labeled GT enters the objective.
                labels = batch["label"][:args.labeled_bs].to(args.device, non_blocking=True)
                v1_step = args.v1_probe and (iteration + 1) % args.diag_interval == 0
                if v1_step:   # the same forward, also returning its x1 / x7_up tensors
                    supervised_outputs, labeled_features = student(images[:args.labeled_bs], deepsup=True, features=True)
                    unlabeled_outputs, unlabeled_features = student(images[args.labeled_bs:], deepsup=True, features=True)
                else:
                    supervised_outputs = student(images[:args.labeled_bs], deepsup=True)
                    unlabeled_outputs = student(images[args.labeled_bs:], deepsup=True)
                loss_sup, ce, dice = supervised_loss(supervised_outputs, labels)
                with torch.no_grad():
                    teacher_outputs = teacher(images[args.labeled_bs:], deepsup=True)
                loss_pl, pl_ce, pl_dice = pseudo_label_consistency(unlabeled_outputs, teacher_outputs)
                weight = args.consistency if iteration >= args.warmup_iterations else 0.0
                loss = loss_sup
                if weight > 0:
                    loss = loss + weight * loss_pl
                if not torch.isfinite(torch.stack([loss_sup, loss_pl, loss])).all():
                    raise FloatingPointError("Nonfinite supervised/pseudo-label/total loss")
                optimizer.zero_grad(set_to_none=True)
                if args.diag_interval and (iteration + 1) % args.diag_interval == 0:
                    # The hidden GT of the unlabeled images is read only here, after the objective is built.
                    # It enters only L_true, an oracle that is differentiated by autograd.grad for the
                    # logged floats and then dropped, and a no_grad boundary function.
                    hidden = batch["label"][args.labeled_bs:].to(args.device)
                    loss_true = supervised_loss(unlabeled_outputs, hidden)[0]
                    probe = dict(iteration=iteration + 1, **gradient_diagnostics(
                        loss_sup, args.consistency * loss_pl,
                        dict(b7=student.block_seven.conv[-3].weight, b3=student.block_three.conv[-3].weight),
                        loss_true))
                    if any(parameter.grad is not None for parameter in student.parameters()):
                        raise RuntimeError("Gradient diagnostics wrote parameter gradients")
                    probe.update(boundary_diagnostics(unlabeled_outputs[-1], teacher_outputs[-1], hidden))
                    probe.update(labeled_band_ce(supervised_outputs[-1], labels))
                    if not all(isinstance(value, (int, float)) for value in probe.values()):
                        raise RuntimeError("Diagnostics must return plain numbers")
                    if diagnostics is None:
                        diagnostics = csv.DictWriter(diagnostics_handle, fieldnames=list(probe))
                        if diagnostics_handle.tell() == 0:
                            diagnostics.writeheader()
                    diagnostics.writerow(probe)
                    if v1_step:
                        # Same tensors, after the diagnostics; sampling seeded by (seed + 12345, iteration) only.
                        v1_rows = v1_probe(student, iteration + 1, labeled_features, unlabeled_features, labels, hidden,
                                           teacher_outputs[2], args.consistency * loss_pl, loss_true,
                                           (args.seed + 12345) * 1_000_003 + iteration + 1)
                        if any(parameter.grad is not None for parameter in student.parameters()):
                            raise RuntimeError("V1 probe wrote parameter gradients")
                        if v1_writer is None:
                            v1_handle = v1_files.enter_context(
                                (args.output_dir / "v1_probe.csv").open("a", newline="", buffering=1))
                            v1_writer = csv.DictWriter(v1_handle, fieldnames=V1_FIELDS)
                            if v1_handle.tell() == 0:
                                v1_writer.writeheader()
                        v1_writer.writerows(v1_rows)
                        del labeled_features, unlabeled_features
                    del loss_true
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float("inf"),
                                                               error_if_nonfinite=True)
                optimizer.step()
                update_ema(student, teacher, args.ema_decay, iteration)
                iteration += 1
                if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.max_iterations:
                    peak = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0
                    row = dict(iteration=iteration, loss_supervised=loss_sup.item(), ce=ce.item(), dice=dice.item(),
                               loss_pl=loss_pl.item(), pl_ce=pl_ce.item(), pl_dice=pl_dice.item(),
                               consistency_weight=weight, loss_total=loss.item(),
                               gradient_norm=gradient_norm.item(), lr=optimizer.param_groups[0]["lr"],
                               peak_memory_mib=peak, elapsed_seconds=time.monotonic() - start,
                               data_wait_seconds=data_wait)
                    metrics.writerow(row)
                    logging.info("iteration %d: sup=%.6f pl=%.6f weight=%.3f grad=%.4f memory=%.0fMiB", iteration,
                                 row["loss_supervised"], row["loss_pl"], weight, row["gradient_norm"], peak)
                if args.val_interval and iteration >= args.val_start and iteration % args.val_interval == 0:
                    result = dict(evaluate(student, root, args.dataset, args.patch_size, args.device), iteration=iteration)
                    with (args.output_dir / "validation_metrics.jsonl").open("a") as validation_log:
                        validation_log.write(json.dumps(result, allow_nan=False) + "\n")
                    logging.info("validation iteration %d: Dice=%.6f", iteration, result["dice"])
                    if result["dice"] > best_dice:
                        best_dice = result["dice"]
                        torch.save(student.state_dict(), args.output_dir / "best_model.pth")
                        (args.output_dir / "best_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
                if iteration == 1 or iteration % args.checkpoint_interval == 0 or iteration == args.max_iterations:
                    save_checkpoint()
                if iteration >= args.max_iterations:
                    break
                fetch = time.monotonic()
    if (args.output_dir / "best_model.pth").is_file():
        student.load_state_dict(torch.load(args.output_dir / "best_model.pth", map_location=args.device,
                                           weights_only=True))
        # Main result without post-processing (DuCiSC's default nms=0); "nms1" adds getLargestCC.
        result = evaluate(student, root, args.dataset, args.patch_size, args.device, split="test", nms1=True)
        (args.output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        logging.info("test nms0: %s | nms1: %s", ducisc_line(result), ducisc_line(result["nms1"]))
    logging.info("finished at iteration %d, best validation Dice %.6f", iteration, best_dice)


if __name__ == "__main__":
    train(arguments())
