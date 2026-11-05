"""Queue training runs over datasets x seeds; start each one when a GPU has enough free memory.

  python scripts/launch.py --datasets LA Pancreas --seeds 42 500 1337 \\
      --output_root Results_baseline_pl

Each run writes <output_root>/seed_<s>/<dataset>_<labelnum>l. Arguments after `--` are passed to
code/train.py unchanged. The launcher returns once every run has started; runs are detached.
It refuses to start unless the working tree is clean (ignored Results* excepted), so every run's
git_commit names a committed state of the code.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
LABELS = {"LA": 8, "Pancreas": 6, "BraTS19": 25}


def free_memory():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    return {int(index): int(free) for index, free in (row.split(",") for row in raw.splitlines())}


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()


def check_traceable():
    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"Refusing to launch: uncommitted changes in {PROJECT}:\n{dirty}")
    return git("rev-parse", "HEAD")


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(LABELS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, help="Restrict placement to these GPUs")
    parser.add_argument("--memory_mib", type=int, default=7500, help="Free memory a GPU needs per run")
    parser.add_argument("--max_per_gpu", type=int, default=2)
    parser.add_argument("--poll_seconds", type=float, default=60)
    parser.add_argument("--settle_seconds", type=float, default=120,
                        help="Wait after a start so the run reaches its peak memory before the next placement")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    head = check_traceable()
    print(f"code: {head} (clean)", flush=True)
    root = (args.output_root if args.output_root.is_absolute() else PROJECT / args.output_root).resolve()
    jobs = []
    for seed in args.seeds:
        for dataset in args.datasets:
            output = root / f"seed_{seed}" / f"{dataset}_{LABELS[dataset]}l"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            jobs.append(dict(dataset=dataset, seed=seed, output_dir=str(output), command=[
                sys.executable, "-u", str(PROJECT / "code/train.py"), "--dataset", dataset,
                "--labelnum", str(LABELS[dataset]), "--seed", str(seed), "--output_dir", str(output), *extra]))
    if args.dry_run:
        for job in jobs:
            print(" ".join(job["command"]))
        return
    logs = root / "launch_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logs.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    placed = {}
    for index, job in enumerate(jobs):
        while True:
            free = free_memory()
            candidates = [gpu for gpu, memory in free.items()
                          if memory >= args.memory_mib and placed.get(gpu, 0) < args.max_per_gpu
                          and (args.gpus is None or gpu in args.gpus)]
            if candidates:
                gpu = max(candidates, key=free.get)
                break
            print(f"Waiting for {args.memory_mib} MiB free; now {free}", flush=True)
            time.sleep(args.poll_seconds)
        placed[gpu] = placed.get(gpu, 0) + 1
        logfile = logs / f"seed_{job['seed']}_{job['dataset']}.log"
        with logfile.open("x") as handle:
            process = subprocess.Popen(job["command"], cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpu)))
        job.update(gpu=gpu, pid=process.pid, log=str(logfile), started_at=datetime.now().isoformat(timespec="seconds"))
        (logs / "manifest.json").write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"seed {job['seed']} {job['dataset']}: GPU={gpu} PID={process.pid}", flush=True)
        if index + 1 < len(jobs):
            time.sleep(args.settle_seconds)
    print(f"All {len(jobs)} runs started; manifest: {logs / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
