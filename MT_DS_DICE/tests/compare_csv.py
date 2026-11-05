"""Bitwise comparison of two training_metrics.csv files over the first N iterations.

  python tests/compare_csv.py REFERENCE CANDIDATE --steps 200 \\
      --columns dice pl_dice gradient_norm --near ce pl_ce loss_supervised loss_pl loss_total
Values are compared as the exact strings written by csv (repr of the float), so equality is bitwise.
A column given as OLD:NEW is read as OLD from the reference and NEW from the candidate (renamed columns).
Columns under --near (values derived from the CUDA CE forward, which is not deterministic; see
check_ce_nondeterminism.py) must instead agree within --relative (default 4 float32 ULP).
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
    parser.add_argument("--near", nargs="+", default=[])
    parser.add_argument("--relative", type=float, default=4 * 2.0 ** -23)
    args = parser.parse_args()
    reference, candidate = rows(args.reference, args.steps), rows(args.candidate, args.steps)
    if not reference or sorted(reference) != sorted(candidate) or max(reference) != args.steps:
        raise SystemExit(f"FAIL: logged iterations differ or stop early: {sorted(reference)} vs {sorted(candidate)}")
    pairs = [column.split(":") if ":" in column else (column, column) for column in args.columns]
    mismatches = [(iteration, new, reference[iteration][old], candidate[iteration][new])
                  for iteration in sorted(reference) for old, new in pairs
                  if reference[iteration][old] != candidate[iteration][new]]
    for mismatch in mismatches[:10]:
        print("mismatch iteration=%d %s: %s vs %s" % mismatch)
    if mismatches:
        counts = {new: sum(1 for _, column, _, _ in mismatches if column == new) for _, new in pairs}
        print("mismatches per column:", ", ".join(f"{column}={count}" for column, count in counts.items()))
    near = [column.split(":") if ":" in column else (column, column) for column in args.near]
    worst, outside = 0.0, 0
    for iteration in sorted(reference):
        for old, new in near:
            left, right = float(reference[iteration][old]), float(candidate[iteration][new])
            gap = abs(left - right) / max(abs(left), 1e-30)
            worst, outside = max(worst, gap), outside + (gap > args.relative)
    failed = bool(mismatches) or outside > 0
    print(f"{'FAIL' if failed else 'PASS'}: {len(reference)} logged iterations (1..{args.steps}); "
          f"bitwise {len(pairs)} columns: {len(mismatches)} mismatches"
          + (f"; within {args.relative:.2e} relative {len(near)} columns: {outside} outside, max {worst:.2e}"
             if near else ""))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
