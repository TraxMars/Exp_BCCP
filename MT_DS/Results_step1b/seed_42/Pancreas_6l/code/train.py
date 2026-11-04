"""Mean Teacher + four-scale deep supervision, without loss normalization."""
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
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--val_start", type=int, default=800)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    # Observation-only channel boundary contribution at D3/D2; 0 disables it.
    parser.add_argument("--cbc_interval", type=int, default=0)
    parser.add_argument("--cbc_log_interval", type=int, default=50)
    parser.add_argument("--cbc_band_radius", type=int, default=2)
    parser.add_argument("--cbc_top_fraction", type=float, default=0.25)
    # Coarse-head probe target: the supervised nearest-neighbour label, or the old soft average.
    parser.add_argument("--cbc_coarse_target", choices=["nearest", "soft"], default="nearest")
    # Logging only: CBC of the unlabeled images against their hidden GT.
    parser.add_argument("--cbc_oracle", action="store_true")
    # Step 2: channel perturbation of an unlabeled decoder tap, consistency to the Teacher.
    parser.add_argument("--fp_tap", choices=["none", "d3", "d2"], default="none")
    parser.add_argument("--fp_drop", type=float, default=0.5)
    parser.add_argument("--fp_weight", type=float, default=1.0)
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
    if args.cbc_interval < 0 or args.cbc_log_interval <= 0 or args.cbc_band_radius < 1:
        parser.error("Require cbc_interval >= 0, cbc_log_interval > 0 and cbc_band_radius >= 1")
    if not 0 < args.cbc_top_fraction <= 1:
        parser.error("cbc_top_fraction must be in (0, 1]")
    if args.cbc_oracle and not args.cbc_interval:
        parser.error("cbc_oracle requires cbc_interval > 0")
    if not 0 <= args.fp_drop < 1 or args.fp_weight < 0:
        parser.error("Require 0 <= fp_drop < 1 and fp_weight >= 0")
    import math
    if not all(math.isfinite(x) for x in (args.base_lr, args.ema_decay, args.consistency)):
        parser.error("Loss and optimizer parameters must be finite")
    if args.base_lr <= 0 or args.consistency < 0:
        parser.error("Invalid learning rate or consistency weight")
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
    from objectives import supervised_loss, consistency_loss, update_ema
    from cbc import CBCTracker
    from perturbation import TAP_FIRST_HEAD, channel_dropout, frozen_bn_statistics

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
    tracker = (CBCTracker(args.max_iterations, args.cbc_band_radius, args.cbc_top_fraction,
                          args.cbc_coarse_target)
               if args.cbc_interval else None)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(experiment="MT_DS", consistency_loss="four_scale_softmax_mse",
                  observation=("none" if tracker is None else "cbc_d3_d2_no_intervention"
                               + ("_with_oracle_logging" if args.cbc_oracle else "")),
                  oracle_gt_in_objective=False,
                  feature_perturbation=("none" if args.fp_tap == "none" else
                                        f"random_channel_dropout_{args.fp_tap}_p{args.fp_drop}"),
                  supervision="four_scale_mean_ce_dice", lr_schedule="constant",
                  teacher_bn="independent_train_mode_running_stats_parameter_only_ema",
                  perturbation="independent_dropout3d_p0.5", normalization="none",
                  dataset_root=str(root), train_cases=len(dataset),
                  validation_split=spec["validation"], device_name=(
                      torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"))
    iteration, best_dice = 0, -1.0
    if args.resume:
        saved_config = json.loads((args.output_dir / "config.json").read_text())
        for key in ("dataset", "labelnum", "seed", "batch_size", "labeled_bs", "patch_size",
                    "base_lr", "ema_decay", "consistency", "warmup_iterations", "normalization"):
            if json.dumps(config[key]) != json.dumps(saved_config[key]):
                raise ValueError(f"Resume configuration mismatch: {key}")
        for key, default in (("fp_tap", "none"), ("fp_drop", 0.5), ("fp_weight", 1.0)):
            if config[key] != saved_config.get(key, default):
                raise ValueError(f"Resume configuration mismatch: {key}")
        saved_manifest = json.loads((args.output_dir / "data_split_manifest.json").read_text())
        if saved_manifest != manifest:
            raise ValueError("Resume dataset split mismatch")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iteration, best_dice = checkpoint["iteration"], checkpoint["best_dice"]
        if tracker is not None and checkpoint.get("cbc") is not None:
            tracker.load_state_dict(checkpoint["cbc"])
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
                            if args.device == "cuda" else None),
                        cbc=tracker.state_dict() if tracker is not None else None), temporary)
        temporary.replace(args.output_dir / "last_checkpoint.pth")

    status("training")
    logging.info("config: %s", config)
    start = time.monotonic()
    start_iteration = iteration
    try:
        with (args.output_dir / "training_metrics.csv").open("a", newline="", buffering=1) as handle:
            fields = ["iteration", "loss_supervised", "ce", "dice", "loss_consistency",
                      "loss_fp", "fp_relative_change",
                      "consistency_weight", "loss_total", "gradient_norm", "lr",
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
                    # Hidden GT of the unlabeled images, for the logging-only oracle probe.
                    oracle_labels = (batch["label"][args.labeled_bs:].to(args.device, non_blocking=True)
                                     if args.cbc_oracle else None)
                    labeled_taps, unlabeled_taps = {}, {}
                    supervised_outputs = student(images[:args.labeled_bs], deepsup=True, taps=labeled_taps)
                    unlabeled_outputs = student(images[args.labeled_bs:], deepsup=True, taps=unlabeled_taps)
                    loss_sup, ce, dice = supervised_loss(supervised_outputs, labels)
                    with torch.no_grad():
                        teacher_outputs = teacher(images[args.labeled_bs:], deepsup=True)
                    loss_mt = consistency_loss(unlabeled_outputs, teacher_outputs)
                    weight = args.consistency if iteration >= args.warmup_iterations else 0.0
                    loss_fp, fp_change = None, None
                    if args.fp_tap != "none" and weight > 0:
                        clean = unlabeled_taps[args.fp_tap]
                        perturbed = channel_dropout(clean, args.fp_drop)
                        # Batch-statistics BN without touching the running statistics used at test time.
                        with frozen_bn_statistics(student):
                            perturbed_outputs = student.decode_from(args.fp_tap, perturbed, unlabeled_taps)
                        loss_fp = consistency_loss(perturbed_outputs,
                                                   teacher_outputs[TAP_FIRST_HEAD[args.fp_tap]:])
                        fp_change = ((perturbed - clean).norm() / clean.norm()).detach()
                    consistency = loss_mt if loss_fp is None else loss_mt + args.fp_weight * loss_fp
                    loss = loss_sup
                    if weight > 0:
                        loss = loss + weight * consistency
                    if not torch.isfinite(torch.stack([loss_sup, consistency, loss])).all():
                        raise FloatingPointError("Nonfinite supervised/consistency/total loss")
                    optimizer.zero_grad(set_to_none=True)
                    if tracker is not None and (iteration + 1) % args.cbc_interval == 0:
                        # Probes reuse the retained graph: no extra forward, no parameter
                        # gradients, no BN or random-stream side effects.
                        tracker.observe(iteration + 1, supervised_outputs, labels, labeled_taps,
                                        unlabeled_outputs, teacher_outputs, unlabeled_taps, oracle_labels)
                        # Neither the probes nor the oracle GT may reach any parameter gradient.
                        if any(parameter.grad is not None for parameter in student.parameters()):
                            raise RuntimeError("Observation produced parameter gradients")
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
                                   loss_fp=float("nan") if loss_fp is None else loss_fp.item(),
                                   fp_relative_change=float("nan") if fp_change is None else fp_change.item(),
                                   loss_total=loss.item(), gradient_norm=gradient_norm.item(),
                                   lr=optimizer.param_groups[0]["lr"], peak_memory_mib=peak,
                                   elapsed_seconds=elapsed)
                        metrics_writer.writerow(row)
                        for key, value in row.items():
                            if key != "iteration":
                                writer.add_scalar(f"train/{key}", value, iteration)
                        eta = elapsed / max(iteration - start_iteration, 1) * (args.max_iterations - iteration)
                        # NaN marks an inactive branch in the CSV; the JSON status stores null.
                        status("training", **{key: None if value != value else value
                                              for key, value in row.items() if key != "iteration"},
                               eta_hours=eta / 3600)
                        logging.info("iteration %d: sup=%.6f mt=%.6f weight=%.3f grad=%.4f memory=%.0fMiB eta=%.2fh",
                                     iteration, row["loss_supervised"], row["loss_consistency"], weight,
                                     row["gradient_norm"], peak, eta / 3600)
                    if tracker is not None and iteration % args.cbc_log_interval == 0:
                        tracker.log(iteration, args.output_dir)
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
