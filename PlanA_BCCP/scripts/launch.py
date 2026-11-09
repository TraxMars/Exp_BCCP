"""Launch one step of the BCCP experiments on both baselines.

Each run copies its baseline's preserved configuration (Results/seed_*/<run>/config.json)
and adds only the arguments of the chosen step. Jobs wait for a GPU with enough free
memory; the launcher exits once every job is started.

  step1    observation only (CBC; Teacher-MSE unlabeled probe in the code of that time)
  step1b   observation only, pseudo-label CE unlabeled probe and oracle logging
           (LA runs used the soft coarse-head target; later runs the nearest one)
  step1c   observation only, gain-free probes (hard / full-resolution / D2 contrast) and oracle
  step2    random channel dropout of the unlabeled D2 tap (p=0.5, weight 1)
  step2_p0 control for step2: the same second D2 tail pass and consistency, no channel
           dropped (p=0), separating the perturbation from the extra consistency term
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


EXPERIMENT = Path(__file__).resolve().parents[1]
BASELINES = ("MT_DS", "MT_DS_LossNorm")
LABELS = {"LA": 8, "Pancreas": 6}
COPIED = ("max_iterations", "warmup_iterations", "base_lr", "ema_decay", "consistency",
          "batch_size", "labeled_bs", "num_workers", "cpu_threads", "log_interval",
          "checkpoint_interval", "val_interval", "val_start", "normalization_eps")
# Observation-only steps skip validation: evaluation cannot change the training trajectory.
STEPS = {
    "step1": ("Results_step1", ["--val_interval", "0", "--cbc_interval", "5"]),
    "step1b": ("Results_step1b", ["--val_interval", "0", "--cbc_interval", "5", "--cbc_oracle"]),
    "step1c": ("Results_step1c", ["--val_interval", "0", "--cbc_interval", "5", "--cbc_oracle",
                                  "--cbc_tracker", "1c"]),
    "step2": ("Results_step2/random_d2_p0.5", ["--fp_tap", "d2", "--fp_drop", "0.5", "--fp_weight", "1.0"]),
    "step2_p0": ("Results_step2/nodrop_d2_p0", ["--fp_tap", "d2", "--fp_drop", "0.0", "--fp_weight", "1.0"]),
}


def free_memory():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    return {int(index): int(free) for index, free in (row.split(",") for row in raw.splitlines())}


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_jobs(steps, datasets, seed, data_root):
    jobs = []
    for dataset in datasets:
        for baseline in BASELINES:
            for step in steps:
                subdirectory, extra = STEPS[step]
                name = f"{dataset}_{LABELS[dataset]}l"
                parent = EXPERIMENT / baseline / "Results" / f"seed_{seed}" / name
                output = EXPERIMENT / baseline / subdirectory / f"seed_{seed}" / name
                if output.exists():
                    raise FileExistsError(f"Refusing to overwrite {output}")
                config = json.loads((parent / "config.json").read_text())
                command = [sys.executable, "-u", str(EXPERIMENT / baseline / "code/train.py"),
                           "--dataset", dataset, "--labelnum", str(LABELS[dataset]),
                           "--seed", str(seed), "--data_root", str(data_root.resolve()),
                           "--output_dir", str(output)]
                for key in COPIED:
                    if key in config:
                        command.extend([f"--{key}", str(config[key])])
                command.extend(["--patch_size", *map(str, config["patch_size"])])
                # Step arguments come last so they override the copied baseline values.
                command.extend(extra)
                jobs.append(dict(step=step, baseline=baseline, dataset=dataset,
                                 labelnum=LABELS[dataset], output_dir=str(output),
                                 parent_config=str(parent / "config.json"), command=command))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), required=True)
    parser.add_argument("--datasets", nargs="+", choices=list(LABELS), default=["LA"])
    parser.add_argument("--gpus", type=int, nargs="+", help="One fixed GPU per job, in launch order")
    parser.add_argument("--memory_per_run_mib", type=int, default=7500)
    parser.add_argument("--max_runs_per_gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=Path, default=EXPERIMENT.parent / "Dataset")
    parser.add_argument("--poll_seconds", type=float, default=60)
    parser.add_argument("--settle_seconds", type=float, default=120)
    args = parser.parse_args()
    jobs = build_jobs(args.steps, args.datasets, args.seed, args.data_root)
    if args.gpus and len(args.gpus) != len(jobs):
        parser.error(f"--gpus needs {len(jobs)} entries")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    logs = EXPERIMENT / "launch_logs" / f"{'_'.join(args.steps)}_{stamp}"
    logs.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_at=stamp, steps=args.steps, seed=args.seed,
                    launcher_pid=os.getpid(), runs=jobs)
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    placed = {}
    for index, job in enumerate(jobs):
        if args.gpus:
            gpu = args.gpus[index]
        else:
            while True:
                free = free_memory()
                candidates = [gpu for gpu, memory in free.items()
                              if memory >= args.memory_per_run_mib
                              and placed.get(gpu, 0) < args.max_runs_per_gpu]
                if candidates:
                    gpu = max(candidates, key=free.get)
                    break
                print(f"Waiting for {args.memory_per_run_mib} MiB free; now {free}", flush=True)
                time.sleep(args.poll_seconds)
        placed[gpu] = placed.get(gpu, 0) + 1
        job["command"] += ["--gpu", str(gpu)]
        logfile = logs / f"{job['step']}_{job['baseline']}_{job['dataset']}.log"
        with logfile.open("x") as handle:
            process = subprocess.Popen(job["command"], cwd=EXPERIMENT / job["baseline"], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        job.update(gpu=gpu, pid=process.pid, launch_log=str(logfile),
                   launched_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"))
        write_json(logs / "manifest.json", manifest)
        print(f"{job['step']} {job['baseline']} {job['dataset']}: GPU={gpu} PID={process.pid}", flush=True)
        if index + 1 < len(jobs) and not args.gpus:
            # Let the new process reach its peak allocation before the next placement.
            time.sleep(args.settle_seconds)
    print(f"All runs launched; manifest: {logs / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
