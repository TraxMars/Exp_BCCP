"""Mean Teacher + four-scale deep supervision + per-objective loss normalization."""
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["LA", "Pancreas", "BraTS19"], required=True)
    parser.add_argument("--data_root", type=Path, default=PROJECT.parent / "Dataset")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--labelnum", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--labeled_bs", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=15000)
    parser.add_argument("--warmup_iterations", type=int, default=2000)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--consistency", type=float, default=0.3)
    parser.add_argument("--normalization_eps", type=float, default=1e-8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--val_start", type=int, default=800)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    # Small-patch runs are verification tools, never used by the experiment launcher.
    parser.add_argument("--patch_size", nargs=3, type=int)
    args = parser.parse_args()
    if args.max_iterations <= 0 or args.warmup_iterations < 0:
        parser.error("Iterations must be positive, warmup must be nonnegative")
    if not 0 < args.labeled_bs < args.batch_size or args.num_workers < 0:
        parser.error("Require 0 < labeled_bs < batch_size and num_workers >= 0")
    if min(args.log_interval, args.checkpoint_interval, args.cpu_threads) <= 0:
        parser.error("Log/checkpoint intervals and cpu_threads must be positive")
    if args.val_interval < 0 or args.val_start < 0:
        parser.error("Validation interval and start must be nonnegative")
    if args.patch_size and any(size < 16 or size % 16 for size in args.patch_size):
        parser.error("Patch dimensions must be positive multiples of 16")
    import math
    if not all(math.isfinite(x) for x in (args.base_lr, args.ema_decay,
                                        args.consistency, args.normalization_eps)):
        parser.error("Loss and optimizer parameters must be finite")
    if args.base_lr <= 0 or args.normalization_eps <= 0 or args.consistency < 0:
        parser.error("Invalid learning rate, normalization epsilon or consistency weight")
    if not 0 <= args.ema_decay < 1:
        parser.error("ema_decay must be in [0, 1)")
    return args


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker_init(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train(args):
    # CUDA visibility is set before importing torch, including transitive imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import copy
    import csv
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from data import SPECS, VolumeDataset, TwoStreamBatchSampler, split_manifest
    from evaluate import evaluate
    from networks import VNetDS
    from objectives import supervised_loss, consistency_loss, normalize_loss, update_ema

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the GPU host rather than the sandbox")
    args.data_root = args.data_root.resolve()
    spec = SPECS[args.dataset]
    root = args.data_root / spec["directory"]
    args.labelnum = args.labelnum if args.labelnum is not None else spec["labels"]
    args.patch_size = tuple(args.patch_size or spec["patch"])
    dataset = VolumeDataset(root, args.dataset, args.patch_size)
    if not args.labeled_bs <= args.labelnum <= len(dataset) - (args.batch_size - args.labeled_bs):
        raise ValueError("Invalid labeled count for this dataset/batch size")
    manifest = split_manifest(root, args.dataset, args.labelnum)
    sampler = TwoStreamBatchSampler(list(range(args.labelnum)),
                                    list(range(args.labelnum, len(dataset))),
                                    args.batch_size, args.labeled_bs)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    student = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).to(args.device)
    teacher = copy.deepcopy(student).requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=args.base_lr,
                                momentum=0.9, weight_decay=1e-4)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(experiment="MT_DS_LossNorm", consistency_loss="four_scale_softmax_mse",
                  supervision="four_scale_mean_ce_dice", lr_schedule="constant",
                  teacher_bn="independent_train_mode_running_stats_parameter_only_ema",
                  perturbation="independent_dropout3d_p0.5", normalization="per_objective",
                  dataset_root=str(root), train_cases=len(dataset),
                  validation_split=spec["validation"], device_name=(
                      torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"))
    iteration, best_dice = 0, -1.0
    if args.resume:
        saved_config = json.loads((args.output_dir / "config.json").read_text())
        for key in ("dataset", "labelnum", "seed", "batch_size", "labeled_bs", "patch_size",
                    "base_lr", "ema_decay", "consistency", "warmup_iterations", "normalization_eps"):
            if json.dumps(config[key]) != json.dumps(saved_config[key]):
                raise ValueError(f"Resume configuration mismatch: {key}")
        saved_manifest = json.loads((args.output_dir / "data_split_manifest.json").read_text())
        if saved_manifest != manifest:
            raise ValueError("Resume dataset split mismatch")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
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
        write_json(args.output_dir / "config.json", config)
        write_json(args.output_dir / "data_split_manifest.json", manifest)
        shutil.copytree(PROJECT / "code", args.output_dir / "code",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(PROJECT / "provenance.json", args.output_dir / "provenance.json")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(args.output_dir / "log.txt"),
                                  logging.StreamHandler(sys.stdout)])
    writer = SummaryWriter(str(args.output_dir / "tensorboard"))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                        pin_memory=args.device == "cuda", worker_init_fn=worker_init)
    common = dict(dataset=args.dataset, labelnum=args.labelnum, seed=args.seed,
                  gpu=args.gpu, pid=os.getpid(), max_iterations=args.max_iterations)

    def status(state, **extra):
        write_json(args.output_dir / "run_status.json", dict(
            common, state=state, iteration=iteration, updated_at=timestamp(),
            best_dice=best_dice if best_dice >= 0 else None, **extra))

    def save_checkpoint():
        temporary = args.output_dir / "last_checkpoint.pth.tmp"
        torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(),
                        optimizer=optimizer.state_dict(), iteration=iteration, best_dice=best_dice,
                        config=config, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(), cuda_rng=(torch.cuda.get_rng_state_all()
                            if args.device == "cuda" else None)), temporary)
        temporary.replace(args.output_dir / "last_checkpoint.pth")

    status("training")
    logging.info("config: %s", config)
    start = time.monotonic()
    start_iteration = iteration
    try:
        with (args.output_dir / "training_metrics.csv").open("a", newline="", buffering=1) as handle:
            fields = ["iteration", "loss_supervised", "ce", "dice", "loss_consistency",
                      "consistency_weight", "loss_normalized", "gradient_norm", "lr",
                      "peak_memory_mib", "elapsed_seconds"]
            metrics_writer = csv.DictWriter(handle, fieldnames=fields)
            if handle.tell() == 0:
                metrics_writer.writeheader()
            while iteration < args.max_iterations:
                student.train()
                teacher.train()
                for batch in loader:
                    images = batch["image"].to(args.device, non_blocking=True)
                    # Only labeled GT enters either objective.
                    labels = batch["label"][:args.labeled_bs].to(args.device, non_blocking=True)
                    supervised_outputs = student(images[:args.labeled_bs], deepsup=True)
                    unlabeled_outputs = student(images[args.labeled_bs:], deepsup=True)
                    loss_sup, ce, dice = supervised_loss(supervised_outputs, labels)
                    with torch.no_grad():
                        teacher_outputs = teacher(images[args.labeled_bs:], deepsup=True)
                    loss_mt = consistency_loss(unlabeled_outputs, teacher_outputs)
                    weight = args.consistency if iteration >= args.warmup_iterations else 0.0
                    loss = normalize_loss(loss_sup, args.normalization_eps)
                    if weight > 0:
                        loss = loss + weight * normalize_loss(loss_mt, args.normalization_eps)
                    if not torch.isfinite(torch.stack([loss_sup, loss_mt, loss])).all():
                        raise FloatingPointError("Nonfinite supervised/consistency/normalized loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    # Observe all gradients without altering their magnitude.
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        student.parameters(), float("inf"), error_if_nonfinite=True)
                    optimizer.step()
                    update_ema(student, teacher, args.ema_decay, iteration)
                    iteration += 1
                    if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.max_iterations:
                        elapsed = time.monotonic() - start
                        peak = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0
                        row = dict(iteration=iteration, loss_supervised=loss_sup.item(), ce=ce.item(),
                                   dice=dice.item(), loss_consistency=loss_mt.item(), consistency_weight=weight,
                                   loss_normalized=loss.item(), gradient_norm=gradient_norm.item(),
                                   lr=optimizer.param_groups[0]["lr"], peak_memory_mib=peak,
                                   elapsed_seconds=elapsed)
                        metrics_writer.writerow(row)
                        for key, value in row.items():
                            if key != "iteration":
                                writer.add_scalar(f"train/{key}", value, iteration)
                        eta = elapsed / max(iteration - start_iteration, 1) * (args.max_iterations - iteration)
                        status("training", **{key: value for key, value in row.items() if key != "iteration"},
                               eta_hours=eta / 3600)
                        logging.info("iteration %d: sup=%.6f mt=%.6f weight=%.3f grad=%.4f memory=%.0fMiB eta=%.2fh",
                                     iteration, row["loss_supervised"], row["loss_consistency"], weight,
                                     row["gradient_norm"], peak, eta / 3600)
                    if args.val_interval and iteration >= args.val_start and iteration % args.val_interval == 0:
                        status("validating")
                        result = evaluate(student, root, args.dataset, args.patch_size, args.device)
                        result["iteration"] = iteration
                        write_json(args.output_dir / "latest_validation.json", result)
                        with (args.output_dir / "validation_metrics.jsonl").open("a") as validation_log:
                            validation_log.write(json.dumps(result, allow_nan=False) + "\n")
                        writer.add_scalar("validation/dice", result["dice"], iteration)
                        logging.info("validation iteration %d: Dice=%.6f", iteration, result["dice"])
                        if result["dice"] > best_dice:
                            best_dice = result["dice"]
                            torch.save(student.state_dict(), args.output_dir / "best_model.pth")
                            torch.save(teacher.state_dict(), args.output_dir / "teacher_at_best_student.pth")
                            write_json(args.output_dir / "best_metrics.json", result)
                        status("training")
                    if iteration == 1 or iteration % args.checkpoint_interval == 0 or iteration == args.max_iterations:
                        save_checkpoint()
                    if iteration >= args.max_iterations:
                        break
        if (args.output_dir / "best_model.pth").is_file():
            student.load_state_dict(torch.load(args.output_dir / "best_model.pth",
                                               map_location=args.device, weights_only=True))
            status("testing_best_checkpoint")
            result = evaluate(student, root, args.dataset, args.patch_size, args.device, split="test")
            write_json(args.output_dir / "test_metrics.json", result)
        status("completed")
    except BaseException as exc:
        status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
               error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    train(arguments())
