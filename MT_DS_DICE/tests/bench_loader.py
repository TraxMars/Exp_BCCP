"""Speed and memory of the in-memory volume cache (--cache_volumes) against reading volumes from disk.

  CUDA_VISIBLE_DEVICES=3 python tests/bench_loader.py --output_root /tmp/bench --iterations 100
Runs code/train.py sequentially (no validation, no diagnostics) and reports the batch-wait share,
the loop time (elapsed_seconds at the last iteration, after setup), and the peak memory of the
training process tree: RSS of the main process and PSS summed over main + DataLoader workers
(PSS splits shared copy-on-write pages, so the sum is the real footprint).
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]


def tree(pid):
    pids = [pid]
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            pids += [child for text in [(task / "children").read_text()] for child in map(int, text.split())]
        except OSError:
            pass
    return pids + [grandchild for child in pids[1:] for grandchild in tree(child)[1:]]


def memory_kib(pid, field, source):
    try:
        for line in Path(f"/proc/{pid}/{source}").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def run(command):
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    peak_rss = peak_pss = 0
    while process.poll() is None:
        pids = tree(process.pid)
        peak_rss = max(peak_rss, memory_kib(process.pid, "VmRSS", "status"))
        peak_pss = max(peak_pss, sum(memory_kib(pid, "Pss", "smaps_rollup") for pid in pids))
        time.sleep(0.5)
    if process.returncode:
        raise SystemExit(f"{command} failed with {process.returncode}")
    return peak_rss / 2**20, peak_pss / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["LA", "Pancreas"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    results = []
    for repeat in range(args.repeats):
        for dataset in args.datasets:
            for cache in (False, True):
                output = args.output_root / f"{dataset}_cache{int(cache)}_r{repeat}"
                rss, pss = run([sys.executable, str(PROJECT / "code/train.py"), "--dataset", dataset,
                                "--output_dir", str(output), "--max_iterations", str(args.iterations),
                                "--val_interval", "0", "--diag_interval", "0",
                                "--cache_volumes" if cache else "--no-cache_volumes"])
                with (output / "training_metrics.csv").open(newline="") as handle:
                    last = list(csv.DictReader(handle))[-1]
                wait, elapsed = float(last["data_wait_seconds"]), float(last["elapsed_seconds"])
                results.append(dict(dataset=dataset, cache=cache, wait=wait, elapsed=elapsed, rss=rss, pss=pss))
                print(f"{dataset:9s} cache={cache!s:5s} repeat={repeat}: wait {wait:6.2f} s / loop {elapsed:6.2f} s "
                      f"= {wait / elapsed:5.1%}; peak RSS main {rss:5.2f} GiB, PSS main+workers {pss:5.2f} GiB",
                      flush=True)
    print()
    for dataset in args.datasets:
        off = [row for row in results if row["dataset"] == dataset and not row["cache"]]
        on = [row for row in results if row["dataset"] == dataset and row["cache"]]
        total = lambda rows, key: sum(row[key] for row in rows)
        print(f"{dataset}: loop time {total(off, 'elapsed'):.1f} s -> {total(on, 'elapsed'):.1f} s "
              f"(speed-up x{total(off, 'elapsed') / total(on, 'elapsed'):.2f}); wait share "
              f"{total(off, 'wait') / total(off, 'elapsed'):.1%} -> {total(on, 'wait') / total(on, 'elapsed'):.1%}; "
              f"peak PSS {max(row['pss'] for row in off):.2f} -> {max(row['pss'] for row in on):.2f} GiB, "
              f"main RSS {max(row['rss'] for row in off):.2f} -> {max(row['rss'] for row in on):.2f} GiB")


if __name__ == "__main__":
    main()
