"""Start three independent experiments with 10% or 20% labeled data."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=3, type=int, help="GPU indices for LA, Pancreas, BraTS19")
    parser.add_argument("--min_free_mib", type=int, default=12000)
    parser.add_argument("--label_percent", type=int, choices=[10, 20], default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=15000)
    parser.add_argument("--output_root", type=Path, default=PROJECT / "Results")
    parser.add_argument("--data_root", type=Path, default=PROJECT.parent / "Dataset")
    args = parser.parse_args()
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    free = {int(row.split(",")[0]): int(row.split(",")[1]) for row in raw.splitlines()}
    if args.gpus is None:
        available = sorted((gpu for gpu in free if free[gpu] >= args.min_free_mib),
                           key=lambda gpu: free[gpu], reverse=True)
        if len(available) < 3:
            raise RuntimeError(f"Need three GPUs with >= {args.min_free_mib} MiB free; available: {free}")
        args.gpus = available[:3]
    if len(set(args.gpus)) != 3 or any(free.get(gpu, 0) < args.min_free_mib for gpu in args.gpus):
        raise RuntimeError(f"Requested devices must be distinct and have sufficient free memory: {free}")
    runs = []
    for (dataset, base_labels), gpu in zip((("LA", 8), ("Pancreas", 6), ("BraTS19", 25)), args.gpus):
        labels = base_labels * (args.label_percent // 10)
        output = args.output_root.resolve() / f"seed_{args.seed}" / f"{dataset}_{labels}l"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
        command = [sys.executable, "-u", str(PROJECT / "code/train.py"),
                   "--dataset", dataset, "--labelnum", str(labels), "--seed", str(args.seed),
                   "--gpu", str(gpu), "--data_root", str(args.data_root.resolve()),
                   "--output_dir", str(output), "--max_iterations", str(args.max_iterations)]
        runs.append(dict(dataset=dataset, labelnum=labels, gpu=gpu, output_dir=str(output), command=command))
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    logs = PROJECT / "launch_logs" / stamp
    logs.mkdir(parents=True, exist_ok=False)
    manifest_path = logs / "launch_manifest.json"
    manifest = dict(created_at=stamp, timezone="Asia/Shanghai", seed=args.seed,
                    label_percent=args.label_percent,
                    gpu_free_before_launch_mib=free, runs=runs)
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    for run in runs:
        log_path = logs / f"{run['dataset']}.log"
        with log_path.open("x") as handle:
            process = subprocess.Popen(run["command"], cwd=PROJECT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        run.update(pid=process.pid, launch_log=str(log_path))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{run['dataset']}: GPU={run['gpu']} PID={process.pid} output={run['output_dir']}", flush=True)
    (PROJECT / f"launch_manifest_{args.label_percent}pct.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    if args.label_percent == 10:
        (PROJECT / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Launch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
