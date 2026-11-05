"""Bitwise comparison of two training_metrics.csv files over the first N iterations.

  python tests/compare_csv.py REFERENCE CANDIDATE --steps 200 \\
      --columns loss_supervised ce dice loss_consistency gradient_norm
Values are compared as the exact strings written by csv (repr of the float), so equality is bitwise.
"""
import argparse
import csv
from pathlib import Path


def rows(path, steps):
    with path.open(newline="") as handle:
        return {int(row["iteration"]): row for row in csv.DictReader(handle) if int(row["iteration"]) <= steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--columns", nargs="+", required=True)
    args = parser.parse_args()
    reference, candidate = rows(args.reference, args.steps), rows(args.candidate, args.steps)
    if not reference or sorted(reference) != sorted(candidate) or max(reference) != args.steps:
        raise SystemExit(f"FAIL: logged iterations differ or stop early: {sorted(reference)} vs {sorted(candidate)}")
    mismatches = [(iteration, column, reference[iteration][column], candidate[iteration][column])
                  for iteration in sorted(reference) for column in args.columns
                  if reference[iteration][column] != candidate[iteration][column]]
    for mismatch in mismatches[:10]:
        print("mismatch iteration=%d %s: %s vs %s" % mismatch)
    verdict = "FAIL" if mismatches else "PASS"
    print(f"{verdict}: {len(reference)} logged iterations (1..{args.steps}) x {len(args.columns)} columns, "
          f"{len(mismatches)} mismatches")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
