"""Share of training time spent waiting for batches, with and without persistent DataLoader workers.

  CUDA_VISIBLE_DEVICES=3 python tests/bench_loader.py --output_root /tmp/bench --iterations 100
Runs code/train.py sequentially (no validation, no diagnostics) and reads data_wait_seconds and
elapsed_seconds at the last iteration of training_metrics.csv. Elapsed time starts after model and
dataset setup, so the share reflects the steady training loop.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["LA", "Pancreas"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    results = []
    for repeat in range(args.repeats):
        for dataset in args.datasets:
            for persistent in (False, True):
                output = args.output_root / f"{dataset}_persistent{int(persistent)}_r{repeat}"
                subprocess.run([sys.executable, str(PROJECT / "code/train.py"), "--dataset", dataset,
                                "--output_dir", str(output), "--max_iterations", str(args.iterations),
                                "--val_interval", "0", "--diag_interval", "0",
                                "--persistent_workers" if persistent else "--no-persistent_workers"],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                with (output / "training_metrics.csv").open(newline="") as handle:
                    last = list(csv.DictReader(handle))[-1]
                wait, elapsed = float(last["data_wait_seconds"]), float(last["elapsed_seconds"])
                results.append((dataset, persistent, repeat, wait, elapsed))
                print(f"{dataset:9s} persistent={persistent!s:5s} repeat={repeat}: wait {wait:7.2f} s / "
                      f"elapsed {elapsed:7.2f} s = {wait / elapsed:6.1%}", flush=True)
    print()
    for dataset in args.datasets:
        totals = {persistent: [sum(row[index] for row in results if row[0] == dataset and row[1] == persistent)
                               for index in (3, 4)] for persistent in (False, True)}
        (wait_off, elapsed_off), (wait_on, elapsed_on) = totals[False], totals[True]
        print(f"{dataset}: wait share {wait_off / elapsed_off:.1%} -> {wait_on / elapsed_on:.1%}; "
              f"elapsed {elapsed_off:.1f} s -> {elapsed_on:.1f} s ({1 - elapsed_on / elapsed_off:+.1%} saved)")


if __name__ == "__main__":
    main()
