"""Summarize multi-seed runs: <root>/seed_<s>/<run>/ -> one Markdown report.

  python scripts/report.py --root Results_baseline_pl --output summary/baseline_seeds.md
Dice and JC are percentages, HD95 and ASD voxels, all with 2 decimals (DuCiSC's table format).
Rows of validation_metrics.jsonl / *.csv repeated after a resume are de-duplicated by iteration
(the last one wins).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parents[1]
COLUMNS = [  # (key, header, scale): scale 100 shows a percentage
    ("best_val_dice", "best val Dice", 100), ("best_val_iteration", "@iter", None),
    ("last5_val_dice", "last-5 val Dice", 100), ("val_dice_10k15k", "val Dice med 10k-15k", 100),
    ("val_hd95_10k15k", "val HD95 med", 1), ("val_asd_10k15k", "val ASD med", 1),
    ("test_dice", "test Dice", 100), ("test_jc", "test JC", 100), ("test_hd95", "test HD95", 1),
    ("test_asd", "test ASD", 1), ("test_worst_dice", "worst-case Dice", 100),
    ("test_empty", "empty pred.", None), ("nms1_dice", "nms1 Dice", 100), ("nms1_hd95", "nms1 HD95", 1)]
PHASES = ((0, 2000), (2000, 6000), (6000, 10000), (10000, 15000))   # (start, end] iterations


def by_iteration(rows):
    """Keep the last row of each iteration, in iteration order."""
    last = {int(row["iteration"]): row for row in rows}
    return [last[iteration] for iteration in sorted(last)]


def read_jsonl(path):
    if not path.is_file():
        return []
    return by_iteration([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return by_iteration(list(csv.DictReader(handle)))


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
                   test_worst_dice=min(case["dice"] for case in test["per_case"]),
                   test_empty=test.get("empty_prediction_cases"))
        if "nms1" in test:
            row.update(nms1_dice=test["nms1"]["dice"], nms1_hd95=test["nms1"]["hd95"])
    training = read_csv(run / "training_metrics.csv")
    row["last_iteration"] = int(training[-1]["iteration"]) if training else 0
    return row


def cell(value, scale):
    if value is None:
        return "—"
    return f"{value:.0f}" if scale is None else f"{value * scale:.2f}"


def seed_table(rows):
    lines = ["| seed | iters | " + " | ".join(header for _, header, _ in COLUMNS) + " |",
             "|---" * (len(COLUMNS) + 2) + "|"]
    for seed, row in rows:
        lines.append(f"| {seed} | {row['last_iteration']} | "
                     + " | ".join(cell(row[key], scale) for key, _, scale in COLUMNS) + " |")
    summary = []
    for key, _, scale in COLUMNS:
        values = [row[key] for _, row in rows if row[key] is not None]
        if not values:
            summary.append("—")
        elif len(values) == 1:
            summary.append(f"{cell(values[0], scale)} (n=1)")
        else:
            factor = 1 if scale is None else scale
            summary.append(f"{statistics.mean(values) * factor:.2f} ± {statistics.stdev(values) * factor:.2f}")
    lines.append("| mean ± sd | | " + " | ".join(summary) + " |")
    return lines


def diagnostics_table(runs):
    """Median of each diagnostics.csv column per phase, pooled over seeds."""
    rows = [row for _, run in runs for row in read_csv(run / "diagnostics.csv")]
    if not rows:
        return ["No diagnostics.csv found.", ""]
    columns = list(dict.fromkeys(column for row in rows for column in row if column != "iteration"))
    lines = ["| diagnostic | " + " | ".join(f"{start // 1000}k–{end // 1000}k" for start, end in PHASES) + " |",
             "|---" * (len(PHASES) + 1) + "|"]
    for column in columns:
        cells = []
        for start, end in PHASES:
            values = [float(row[column]) for row in rows
                      if start < int(row["iteration"]) <= end and row.get(column) not in (None, "")]
            value = median(values)
            cells.append("—" if value is None else f"{value:.4f}")
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
             "Dice / JC in %, HD95 / ASD in voxels. Test metrics follow DuCiSC (an empty prediction scores 0 and",
             "counts in the mean); nms1 = largest connected component. Validation medians use 10000 ≤ iteration ≤",
             "15000; worst-case Dice is the lowest per-case test Dice; sd is the sample standard deviation across seeds.",
             ""]
    for name, seeds in sorted(runs.items()):
        rows = [(seed, run_metrics(run)) for seed, run in sorted(seeds)]
        lines += [f"## {name}", ""] + seed_table(rows) + ["", f"### {name}: diagnostics, median per phase", ""]
        lines += diagnostics_table(sorted(seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
