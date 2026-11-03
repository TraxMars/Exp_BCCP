"""Summarize step-1 CBC diagnostics (works on partial logs) into summary/step1.md.

Gates (tentative, judged per tap on the post-warmup phases; the decision stays with us):
  G1 stability    EMA boundary ranking agrees with itself 500 iterations earlier:
                  median lag_rho >= 0.7 for both streams.
  G2 specificity  The boundary ranking is not a copy of the region ranking:
                  labeled residual_fraction >= 0.2 and rho_bd_rg <= 0.9, and the residual
                  ranking is itself stable (lag_rho_res >= 0.5). Random channel vectors
                  also give a large residual, so G2 requires the residual to be stable.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


EXPERIMENT = Path(__file__).resolve().parents[1]
BASELINES = ("MT_DS", "MT_DS_LossNorm")
RUNS = ("LA_8l", "Pancreas_6l")
PHASES = ((1, 2000, "warmup ≤2k"), (2001, 6000, "2k–6k"),
          (6001, 10000, "6k–10k"), (10001, 15000, "10k–15k"))
METRICS = ("lag_rho", "lag_jaccard", "raw_rho", "raw_jaccard", "cos_bd_rg", "rho_bd_rg",
           "residual_fraction", "lag_rho_res", "helpful_fraction", "skipped")


def read_csv(path):
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


def middle(values):
    values = [value for value in values if not math.isnan(value)]
    return median(values) if values else math.nan


def fmt(value):
    return "–" if math.isnan(value) else f"{value:.2f}"


def trajectory(run, parent):
    """Largest supervised-loss gap to the baseline log on shared iterations."""
    new = {row["iteration"]: number(row["loss_supervised"]) for row in read_csv(run / "training_metrics.csv")}
    old = {row["iteration"]: number(row["loss_supervised"]) for row in read_csv(parent / "training_metrics.csv")}
    shared = sorted(set(new) & set(old), key=int)
    if not shared:
        return "no shared iterations yet"
    gaps = [(int(key), abs(new[key] - old[key])) for key in shared]
    diverged = next((key for key, gap in gaps if gap > 1e-6), None)
    return (f"identical through {gaps[-1][0]}" if diverged is None else
            f"first gap >1e-6 at {diverged}; max |Δsup|={max(g for _, g in gaps):.4f}")


def phase_table(rows, tap):
    lines = ["| phase | stream | " + " | ".join(METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    summary = {}
    for start, end, label in PHASES:
        selected = [row for row in rows if row["tap"] == tap and start <= int(row["iteration"]) <= end]
        if not selected:
            continue
        for stream in ("lab", "unl"):
            values = {name: middle([number(row[f"{stream}_{name}"]) for row in selected])
                      for name in METRICS}
            summary[(label, stream)] = values
            lines.append(f"| {label} | {stream} | " + " | ".join(fmt(values[name]) for name in METRICS) + " |")
        cross = {name: middle([number(row[name]) for row in selected])
                 for name in ("cross_rho_bd", "cross_jaccard_bd", "cross_rho_res", "cross_jaccard_res")}
        summary[(label, "cross")] = cross
        lines.append(f"| {label} | L↔U | cross_rho_bd={fmt(cross['cross_rho_bd'])}, "
                     f"cross_jaccard_bd={fmt(cross['cross_jaccard_bd'])}, "
                     f"cross_rho_res={fmt(cross['cross_rho_res'])} |" + " |" * (len(METRICS) - 1))
    return lines, summary


def gates(summary):
    verdicts = []
    for _, _, label in PHASES[1:]:
        lab, unl = summary.get((label, "lab")), summary.get((label, "unl"))
        if lab is None:
            continue
        g1 = min(lab["lag_rho"], unl["lag_rho"]) >= 0.7 if unl else False
        g2 = (lab["residual_fraction"] >= 0.2 and lab["rho_bd_rg"] <= 0.9
              and lab["lag_rho_res"] >= 0.5)
        verdicts.append(f"{label}: G1 {'pass' if g1 else 'fail'}, G2 {'pass' if g2 else 'fail'}")
    return "; ".join(verdicts) or "no post-warmup data yet"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    lines = ["# Step 1：只观测的 CBC 诊断", "",
             "数值为各阶段中位数；lab=有标签（GT 边界带），unl=无标签（Teacher 伪边界带）。",
             "lag_* 为 EMA 分数与 500 步前的一致性；raw_* 为相邻两次原始观测的一致性；",
             "residual_fraction = 1 − cos²(ψ_bd, ψ_rg)（通道空间），lag_rho_res 为去区域分量后残差排名的稳定性。", ""]
    for baseline in BASELINES:
        for name in RUNS:
            run = EXPERIMENT / baseline / "Results_step1" / f"seed_{args.seed}" / name
            parent = EXPERIMENT / baseline / "Results" / f"seed_{args.seed}" / name
            status_path = run / "run_status.json"
            status = json.loads(status_path.read_text()) if status_path.is_file() else {}
            lines += [f"## {baseline} / {name}", "",
                      f"- 状态：{status.get('state', 'not started')}，iteration {status.get('iteration', 0)}",
                      f"- 与旧基线训练轨迹：{trajectory(run, parent)}"]
            rows = read_csv(run / "channel_diagnostics.csv")
            for tap in ("d3", "d2"):
                table, summary = phase_table(rows, tap)
                lines += [f"- {tap} 关卡：{gates(summary)}", ""] + table + [""]
    output = EXPERIMENT / "summary" / "step1.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
