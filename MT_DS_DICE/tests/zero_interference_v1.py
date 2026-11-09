"""V1 zero interference and overhead (LA, seed 42, 100 iterations, warmup 50, no validation).

  python tests/zero_interference_v1.py --gpu 6 --work <scratch dir> [--pancreas]
Runs, one after another on one GPU:
  base  = the committed code (git archive HEAD), --diag_interval 50   (reference: the current baseline)
  off   = this code, --diag_interval 50, no --v1_probe
  v1_50 = this code, --diag_interval 50 --v1_probe
  v1_1  = this code, --diag_interval 1 --v1_probe                     (probe on every iteration)
Compares training_metrics.csv as CHANGES.md prescribes (dice, pl_dice, consistency_weight, gradient_norm, lr bitwise;
the CE-derived ce, pl_ce, loss_supervised, loss_pl, loss_total within 4 float32 ULP), the 31 diagnostics columns at
iterations 50 and 100 bitwise, the final Student / Teacher state_dict (BN buffers included) and optimizer state
bitwise, and the v1_probe.csv rows at 50 and 100 between v1_50 and v1_1 (sampling depends on (seed, iteration) only).
With --pancreas also Pancreas off / v1 (both --diag_interval 10) for the probe time there.
Writes the report and the CSVs to tests/outputs/v1/.
"""
import argparse
import csv
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import torch


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATA = PROJECT.parents[1] / "Dataset"
COMMON = ["--seed", "42", "--max_iterations", "100", "--warmup_iterations", "50", "--val_interval", "0",
          "--data_root", str(DATA)]
BITWISE = ["dice", "pl_dice", "consistency_weight", "gradient_norm", "lr"]
NEAR = ["ce", "pl_ce", "loss_supervised", "loss_pl", "loss_total"]


def run(name, code_dir, gpu, work, extra):
    output = work / name
    if output.exists():
        shutil.rmtree(output)
    command = [PYTHON, "-u", str(code_dir / "code/train.py"), "--output_dir", str(output), *COMMON, *extra]
    print(f"$ CUDA_VISIBLE_DEVICES={gpu} {' '.join(command)}", flush=True)
    log = (work / f"{name}.stdout").open("w")
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
    return output


def read(path, keep=None):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if keep is None or keep(row)]


def compare_metrics(report, reference, candidate, label):
    result = subprocess.run([PYTHON, str(PROJECT / "tests/compare_csv.py"), str(reference / "training_metrics.csv"),
                             str(candidate / "training_metrics.csv"), "--steps", "100", "--columns", *BITWISE,
                             "--near", *NEAR], capture_output=True, text=True)
    report.append(f"## training_metrics.csv: {label}\n{result.stdout.strip()}")
    return result.returncode == 0


def compare_rows(report, left, right, label, key, skip=()):
    """Bitwise equality of all value columns of the rows both files have at iterations 50 and 100."""
    keep = lambda row: row["iteration"] in ("50", "100") and row.get("family") not in skip
    a, b = {key(r): r for r in read(left, keep)}, {key(r): r for r in read(right, keep)}
    if not a or a.keys() != b.keys():
        report.append(f"## {label}\nFAIL: row keys differ ({len(a)} vs {len(b)})")
        return False
    columns = [c for c in next(iter(a.values())) if c != "iteration"]
    mismatches = [(k, c, a[k][c], b[k][c]) for k in a for c in columns if a[k][c] != b[k][c]]
    report.append(f"## {label}\n{'FAIL' if mismatches else 'PASS'}: {len(a)} rows x {len(columns)} columns, "
                  f"{len(mismatches)} mismatches" + "".join(f"\n  {m}" for m in mismatches[:5]))
    return not mismatches


def compare_states(report, runs):
    checkpoints = {name: torch.load(path / "last_checkpoint.pth", map_location="cpu", weights_only=False)
                   for name, path in runs.items()}
    reference_name, reference = next(iter(checkpoints.items()))
    ok = True
    for name, checkpoint in checkpoints.items():
        if name == reference_name:
            continue
        lines = []
        for part in ("student", "teacher"):
            left, right = reference[part], checkpoint[part]
            different = [key for key in left if not torch.equal(left[key], right[key])]
            lines.append(f"{part}: {len(left)} tensors ({sum(k.endswith(('running_mean', 'running_var', 'num_batches_tracked')) for k in left)} "
                         f"BN buffers), {len(different)} differ" + (f" e.g. {different[:3]}" if different else ""))
            ok &= not different and list(left) == list(right)
        momentum_left = [s["momentum_buffer"] for s in reference["optimizer"]["state"].values()]
        momentum_right = [s["momentum_buffer"] for s in checkpoint["optimizer"]["state"].values()]
        same_momentum = len(momentum_left) == len(momentum_right) and all(
            torch.equal(a, b) for a, b in zip(momentum_left, momentum_right))
        ok &= same_momentum and checkpoint["iteration"] == reference["iteration"] == 100
        lines.append(f"optimizer momentum buffers: {'identical' if same_momentum else 'DIFFER'}")
        report.append(f"## final state (iteration 100): {reference_name} vs {name}\n"
                      f"{'PASS' if ok else 'FAIL'}: " + "; ".join(lines))
    return ok


def overhead(report, runs, title):
    lines = [f"## overhead: {title}",
             "| run | elapsed at 100 (s) | peak allocated (MiB) | probe steps | probe s/step: mean / median / max |",
             "|---|---|---|---|---|"]
    for name, path in runs.items():
        last = read(path / "training_metrics.csv")[-1]
        probe = path / "v1_probe.csv"
        times = [float(r["value"]) for r in read(probe, lambda r: r["family"] == "meta")] if probe.exists() else []
        timing = (f"{statistics.mean(times):.3f} / {statistics.median(times):.3f} / {max(times):.3f}" if times else "—")
        lines.append(f"| {name} | {float(last['elapsed_seconds']):.1f} | {float(last['peak_memory_mib']):.0f} | "
                     f"{len(times)} | {timing} |")
    report.append("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--pancreas", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT / "tests/outputs/v1")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    base_code = args.work / "base_code"
    if base_code.exists():
        shutil.rmtree(base_code)
    base_code.mkdir()
    head = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    archive = subprocess.run(["git", "-C", str(PROJECT), "archive", "HEAD", "code"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(base_code)], input=archive.stdout, check=True)

    la = ["--dataset", "LA", "--labelnum", "8"]
    runs = dict(base=run("base", base_code, args.gpu, args.work, [*la, "--diag_interval", "50"]),
                off=run("off", PROJECT, args.gpu, args.work, [*la, "--diag_interval", "50"]),
                v1_50=run("v1_50", PROJECT, args.gpu, args.work, [*la, "--diag_interval", "50", "--v1_probe"]),
                v1_1=run("v1_1", PROJECT, args.gpu, args.work, [*la, "--diag_interval", "1", "--v1_probe"]))
    report = [f"# V1 zero interference: LA, seed 42, 100 iterations, --warmup_iterations 50 (L_pl from 50), "
              f"--val_interval 0, GPU {args.gpu} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else '?'})",
              f"# base = committed code {head} (git archive), diag every 50; off = this code without --v1_probe; "
              f"v1_50 / v1_1 = --v1_probe with diagnostics every 50 / every iteration"]
    ok = True
    for name in ("off", "v1_50", "v1_1"):
        ok &= compare_metrics(report, runs["base"], runs[name], f"base vs {name}")
    diag_key = lambda row: row["iteration"]
    for name in ("off", "v1_50", "v1_1"):
        ok &= compare_rows(report, runs["base"] / "diagnostics.csv", runs[name] / "diagnostics.csv",
                           f"diagnostics.csv rows 50 and 100, all columns: base vs {name}", diag_key)
    ok &= compare_states(report, runs)
    probe_key = lambda row: tuple(row[c] for c in ("iteration", "family", "constraint", "G", "tau", "subset", "metric"))
    ok &= compare_rows(report, runs["v1_50"] / "v1_probe.csv", runs["v1_1"] / "v1_probe.csv",
                       "v1_probe.csv rows 50 and 100 (without meta/probe_seconds): v1_50 vs v1_1", probe_key,
                       skip=("meta",))
    report.append(f"files: off wrote v1_probe.csv: {(runs['off'] / 'v1_probe.csv').exists()}; "
                  f"v1_1 diagnostics rows: {len(read(runs['v1_1'] / 'diagnostics.csv'))}; "
                  f"v1_1 probe steps: {len(read(runs['v1_1'] / 'v1_probe.csv', lambda r: r['family'] == 'meta'))}")
    overhead(report, runs, "LA (each run alone on the GPU, sequentially)")
    if args.pancreas:
        pancreas = ["--dataset", "Pancreas", "--labelnum", "6", "--diag_interval", "10"]
        extra = dict(pancreas_off=run("pancreas_off", PROJECT, args.gpu, args.work, pancreas),
                     pancreas_v1=run("pancreas_v1", PROJECT, args.gpu, args.work, [*pancreas, "--v1_probe"]))
        ok &= compare_metrics(report, extra["pancreas_off"], extra["pancreas_v1"], "Pancreas off vs v1 (diag every 10)")
        overhead(report, extra, "Pancreas, diag every 10")
        runs.update(extra)
    report.append(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    text = "\n\n".join(report) + "\n"
    print(text)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "zero_interference_v1.txt").write_text(text)
    for name, path in runs.items():
        target = args.output / "runs" / name
        target.mkdir(parents=True, exist_ok=True)
        for file in ("config.json", "training_metrics.csv", "diagnostics.csv", "v1_probe.csv"):
            if (path / file).exists():
                shutil.copy2(path / file, target / file)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
