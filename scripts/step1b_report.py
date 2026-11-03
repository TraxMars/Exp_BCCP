"""Summarize step-1b diagnostics (works on partial logs) into summary/step1b.md.

Criteria, all read against the noise ceiling = split-half Spearman of the labeled EMA
(two EMAs fed alternate observations; full-data EMAs are somewhat more reliable):
  C1  stability of the pseudo-label score: unl_lag_rho (EMA vs 500 iterations earlier)
  C2  pseudo vs oracle agreement: Spearman(psi_U_pseudo, psi_U_GT)
  C3  labeled vs oracle: Spearman(psi_L, psi_U_GT); clearly below the ceiling means the
      labeled and unlabeled boundary contributions genuinely differ.
"""
import argparse
import csv
import math
from pathlib import Path
from statistics import median


EXPERIMENT = Path(__file__).resolve().parents[1]
BASELINES = ("MT_DS", "MT_DS_LossNorm")
PHASES = ((1, 2000, "warmup ≤2k"), (2001, 6000, "2k–6k"),
          (6001, 10000, "6k–10k"), (10001, 15000, "10k–15k"))
CRITERIA = (("ceiling", "lab_split_rho"), ("C1 unl_lag_rho", "unl_lag_rho"),
            ("C2 ρ(unl,orc)", "rho_unl_orc_bd"), ("C3 ρ(lab,orc)", "rho_lab_orc_bd"),
            ("ρ(lab,unl)", "rho_lab_unl_bd"), ("unl_split", "unl_split_rho"),
            ("orc_split", "orc_split_rho"), ("orc_lag", "orc_lag_rho"))
EXTRAS = (("伪带 IoU", "unl_band_iou_gt"), ("带内置信度", "unl_band_confidence"),
          ("有效图像 lab/unl/orc", ("lab_valid_images", "unl_valid_images", "orc_valid_images")),
          ("区域稳定 lab/unl/orc", ("lab_lag_rho_rg", "unl_lag_rho_rg", "orc_lag_rho_rg")),
          ("边界专属 lab/unl/orc", ("lab_residual_fraction", "unl_residual_fraction",
                                  "orc_residual_fraction")),
          ("跳过率 unl/orc", ("unl_skipped", "orc_skipped")))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def middle(rows, column):
    values = [number(row.get(column)) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return median(values) if values else math.nan


def fmt(value):
    return "–" if math.isnan(value) else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", default="LA_8l")
    args = parser.parse_args()
    lines = ["# Step 1b：伪标签 CE 无标签探针 + oracle", "",
             "各阶段中位数。ceiling = 有标签分数的奇偶拆半 Spearman（噪声上限）；",
             "C1 = 伪标签分数 500 步稳定性；C2 = ρ(ψ^U_伪, ψ^U_GT)；C3 = ρ(ψ^L, ψ^U_GT)。", ""]
    for baseline in BASELINES:
        path = EXPERIMENT / baseline / "Results_step1b" / f"seed_{args.seed}" / args.run / "channel_diagnostics.csv"
        rows = list(csv.DictReader(path.open())) if path.is_file() else []
        last = max((int(row["iteration"]) for row in rows), default=0)
        lines += [f"## {baseline} / {args.run}（已记录到第 {last} 步）", ""]
        for tap in ("d3", "d2"):
            tap_rows = [row for row in rows if row["tap"] == tap]
            lines += [f"### {tap}", "",
                      "| 阶段 | " + " | ".join(name for name, _ in CRITERIA) + " | C3 差距 |",
                      "|---|" + "---|" * (len(CRITERIA) + 1)]
            extra = ["| 阶段 | " + " | ".join(name for name, _ in EXTRAS) + " |",
                     "|---|" + "---|" * len(EXTRAS)]
            for start, end, label in PHASES:
                selected = [row for row in tap_rows if start <= int(row["iteration"]) <= end]
                if not selected:
                    continue
                values = [middle(selected, column) for _, column in CRITERIA]
                gap = values[0] - values[3]
                lines.append(f"| {label} | " + " | ".join(fmt(value) for value in values) + f" | {fmt(gap)} |")
                cells = []
                for _, columns in EXTRAS:
                    columns = (columns,) if isinstance(columns, str) else columns
                    cells.append("/".join(fmt(middle(selected, column)) for column in columns))
                extra.append(f"| {label} | " + " | ".join(cells) + " |")
            lines += [""] + extra + [""]
    output = EXPERIMENT / "summary" / "step1b.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
