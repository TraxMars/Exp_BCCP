"""Summarize multi-seed runs: <root>/seed_<s>/<run>/ -> one Markdown report.

  python scripts/report.py --root Results_baseline_pl --output summary/baseline_seeds.md
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parents[1]
COLUMNS = [  # (key, header, digits)
    ("best_val_dice", "best val Dice", 4), ("best_val_iteration", "@iter", 0),
    ("last5_val_dice", "last-5 val Dice", 4), ("val_dice_10k15k", "val Dice med 10k-15k", 4),
    ("val_hd95_10k15k", "val HD95 med", 2), ("val_asd_10k15k", "val ASD med", 2),
    ("test_dice", "test Dice", 4), ("test_jc", "test JC", 4), ("test_hd95", "test HD95", 2),
    ("test_asd", "test ASD", 2), ("test_worst_dice", "worst-case Dice", 4)]
PHASES = ((0, 2000), (2000, 6000), (6000, 10000), (10000, 15000))   # (start, end] iterations


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def median(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.median(values) if values else None


def run_metrics(run):
    validations = read_jsonl(run / "validation_metrics.jsonl")
    row = dict.fromkeys(key for key, _, _ in COLUMNS)
    if validations:
        best = max(validations, key=lambda entry: entry["dice"])
        late = [entry for entry in validations if 10000 <= entry["iteration"] <= 15000]
        row.update(best_val_dice=best["dice"], best_val_iteration=best["iteration"],
                   last5_val_dice=statistics.mean(entry["dice"] for entry in validations[-5:]),
                   val_dice_10k15k=median([entry["dice"] for entry in late]),
                   val_hd95_10k15k=median([entry["hd95"] for entry in late]),
                   val_asd_10k15k=median([entry["asd"] for entry in late]))
    if (run / "test_metrics.json").is_file():
        test = json.loads((run / "test_metrics.json").read_text())
        row.update(test_dice=test["dice"], test_jc=test["jc"], test_hd95=test["hd95"], test_asd=test["asd"],
                   test_worst_dice=min(case["dice"] for case in test["per_case"]))
    lines = (run / "training_metrics.csv").read_text().splitlines() if (run / "training_metrics.csv").is_file() else []
    row["last_iteration"] = int(lines[-1].split(",")[0]) if len(lines) > 1 else 0
    return row


def cell(value, digits):
    return "—" if value is None else f"{value:.{digits}f}"


def seed_table(rows):
    lines = ["| seed | iters | " + " | ".join(header for _, header, _ in COLUMNS) + " |",
             "|---" * (len(COLUMNS) + 2) + "|"]
    for seed, row in rows:
        lines.append(f"| {seed} | {row['last_iteration']} | "
                     + " | ".join(cell(row[key], digits) for key, _, digits in COLUMNS) + " |")
    summary = []
    for key, _, digits in COLUMNS:
        values = [row[key] for _, row in rows if row[key] is not None]
        if not values:
            summary.append("—")
        elif len(values) == 1:
            summary.append(f"{values[0]:.{digits}f} (n=1)")
        else:
            summary.append(f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}")
    lines.append("| mean ± sd | | " + " | ".join(summary) + " |")
    return lines


def diagnostics_table(runs):
    """Median of each diagnostics.csv column per phase, pooled over seeds."""
    rows = []
    for _, run in runs:
        if (run / "diagnostics.csv").is_file():
            with (run / "diagnostics.csv").open(newline="") as handle:
                rows += list(csv.DictReader(handle))
    if not rows:
        return ["No diagnostics.csv found.", ""]
    lines = ["| diagnostic | " + " | ".join(f"{start // 1000}k–{end // 1000}k" for start, end in PHASES) + " |",
             "|---" * (len(PHASES) + 1) + "|"]
    for column in (column for column in rows[0] if column != "iteration"):
        cells = []
        for start, end in PHASES:
            values = [float(row[column]) for row in rows if start < int(row["iteration"]) <= end and row[column]]
            cells.append(cell(median(values), 4))
        lines.append(f"| {column} | " + " | ".join(cells) + " |")
    return lines + ["", f"{len(rows)} diagnostic rows from {len(runs)} seed(s); phases are (start, end] iterations.", ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "summary/baseline_seeds.md")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else PROJECT / args.root
    runs = {}
    for run in sorted(root.glob("seed_*/*/")):
        if (run / "config.json").is_file():
            runs.setdefault(run.name, []).append((int(run.parent.name[5:]), run))
    if not runs:
        raise SystemExit(f"No runs under {root}")
    lines = [f"# Multi-seed summary: `{root}`", "",
             "Validation medians use 10000 ≤ iteration ≤ 15000; worst-case Dice is the lowest per-case test Dice;",
             "sd is the sample standard deviation across seeds.", ""]
    for name, seeds in sorted(runs.items()):
        rows = [(seed, run_metrics(run)) for seed, run in sorted(seeds)]
        lines += [f"## {name}", ""] + seed_table(rows) + ["", f"### {name}: diagnostics, median per phase", ""]
        lines += diagnostics_table(sorted(seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
