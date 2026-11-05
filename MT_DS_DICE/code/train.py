"""Mean Teacher + four-scale deep supervision on DuCiSC's VNet; MSE or pseudo-label consistency, optional LossNorm.

The GPU is chosen by the caller through CUDA_VISIBLE_DEVICES (scripts/launch.py sets it).
"""
import argparse
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
from diagnostics import boundary_diagnostics, gradient_diagnostics
from evaluate import evaluate
from networks import VNetDS
from objectives import consistency_loss, normalize_loss, pseudo_label_consistency, supervised_loss, update_ema


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
    # mse: softmax MSE to the Teacher; pseudo_ce_dice: (CE + Dice) / 2 to the Teacher argmax of each scale.
    parser.add_argument("--consistency_type", choices=["mse", "pseudo_ce_dice"], default="mse")
    # Divide the supervised and consistency objectives each by their detached value.
    parser.add_argument("--loss_norm", action="store_true")
    # Logging-only diagnostics (diagnostics.csv) every N iterations; 0 disables them.
    parser.add_argument("--diag_interval", type=int, default=50)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    # Resuming restores the configuration stored in the checkpoint; other arguments are ignored.
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--patch_size", nargs=3, type=int)
    args = parser.parse_args()
    if not 0 < args.labeled_bs < args.batch_size:
        parser.error("Require 0 < labeled_bs < batch_size")
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
        args = argparse.Namespace(**dict(checkpoint["config"]["args"], resume=args.resume))
        args.data_root, args.output_dir = Path(args.data_root), Path(args.output_dir)
    spec = SPECS[args.dataset]
    args.data_root = args.data_root.resolve()
    root = args.data_root / spec["directory"]
    args.labelnum = args.labelnum if args.labelnum is not None else spec["labels"]
    args.patch_size = list(args.patch_size or spec["patch"])
    dataset = VolumeDataset(root, args.dataset, args.patch_size)
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
                        pin_memory=args.device == "cuda", worker_init_fn=worker_init,
                        persistent_workers=args.persistent_workers and args.num_workers > 0)

    def save_checkpoint():
        torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(),
                        optimizer=optimizer.state_dict(), iteration=iteration, best_dice=best_dice,
                        config=config, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None),
                   args.output_dir / "last_checkpoint.pth.tmp")
        (args.output_dir / "last_checkpoint.pth.tmp").replace(args.output_dir / "last_checkpoint.pth")

    start, data_wait = time.monotonic(), 0.0
    fields = ["iteration", "loss_supervised", "ce", "dice", "loss_consistency", "pl_ce", "pl_dice",
              "consistency_weight", "loss_total", "gradient_norm", "lr", "peak_memory_mib", "elapsed_seconds",
              "data_wait_seconds"]
    with (args.output_dir / "training_metrics.csv").open("a", newline="", buffering=1) as handle, \
            (args.output_dir / "diagnostics.csv").open("a", newline="", buffering=1) as diagnostics_handle:
        metrics, diagnostics = csv.DictWriter(handle, fieldnames=fields), None
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
                supervised_outputs = student(images[:args.labeled_bs], deepsup=True)
                unlabeled_outputs = student(images[args.labeled_bs:], deepsup=True)
                loss_sup, ce, dice = supervised_loss(supervised_outputs, labels)
                with torch.no_grad():
                    teacher_outputs = teacher(images[args.labeled_bs:], deepsup=True)
                if args.consistency_type == "mse":
                    loss_cons, pl_ce, pl_dice = consistency_loss(unlabeled_outputs, teacher_outputs), None, None
                else:
                    loss_cons, pl_ce, pl_dice = pseudo_label_consistency(unlabeled_outputs, teacher_outputs)
                weight = args.consistency if iteration >= args.warmup_iterations else 0.0
                norm = normalize_loss if args.loss_norm else (lambda loss: loss)
                loss = norm(loss_sup)
                if weight > 0:
                    loss = loss + weight * norm(loss_cons)
                if not torch.isfinite(torch.stack([loss_sup, loss_cons, loss])).all():
                    raise FloatingPointError("Nonfinite supervised/consistency/total loss")
                optimizer.zero_grad(set_to_none=True)
                if args.diag_interval and (iteration + 1) % args.diag_interval == 0:
                    probe = dict(iteration=iteration + 1, **gradient_diagnostics(
                        loss_sup, args.consistency * loss_cons,
                        dict(b7=student.block_seven.conv[-3].weight, b3=student.block_three.conv[-3].weight)))
                    if any(parameter.grad is not None for parameter in student.parameters()):
                        raise RuntimeError("Gradient diagnostics wrote parameter gradients")
                    # The hidden GT of the unlabeled images is read only here, after the objective is built,
                    # and only by a no_grad function that returns floats.
                    hidden = batch["label"][args.labeled_bs:].to(args.device)
                    probe.update(boundary_diagnostics(unlabeled_outputs[-1], teacher_outputs[-1], hidden))
                    if not all(isinstance(value, (int, float)) for value in probe.values()):
                        raise RuntimeError("Diagnostics must return plain numbers")
                    if diagnostics is None:
                        diagnostics = csv.DictWriter(diagnostics_handle, fieldnames=list(probe))
                        if diagnostics_handle.tell() == 0:
                            diagnostics.writeheader()
                    diagnostics.writerow(probe)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float("inf"),
                                                               error_if_nonfinite=True)
                optimizer.step()
                update_ema(student, teacher, args.ema_decay, iteration)
                iteration += 1
                if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.max_iterations:
                    peak = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0
                    row = dict(iteration=iteration, loss_supervised=loss_sup.item(), ce=ce.item(), dice=dice.item(),
                               loss_consistency=loss_cons.item(),
                               pl_ce=None if pl_ce is None else pl_ce.item(),
                               pl_dice=None if pl_dice is None else pl_dice.item(), consistency_weight=weight, loss_total=loss.item(),
                               gradient_norm=gradient_norm.item(), lr=optimizer.param_groups[0]["lr"],
                               peak_memory_mib=peak, elapsed_seconds=time.monotonic() - start,
                               data_wait_seconds=data_wait)
                    metrics.writerow(row)
                    logging.info("iteration %d: sup=%.6f cons=%.6f weight=%.3f grad=%.4f memory=%.0fMiB", iteration,
                                 row["loss_supervised"], row["loss_consistency"], weight, row["gradient_norm"], peak)
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
        result = evaluate(student, root, args.dataset, args.patch_size, args.device, split="test")
        (args.output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    logging.info("finished at iteration %d, best validation Dice %.6f", iteration, best_dice)


if __name__ == "__main__":
    train(arguments())
