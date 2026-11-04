"""Summarize step-1c diagnostics (works on partial logs) into summary/step1c.md.

Decision criteria, fixed before the runs and read on the oracle stream (unlabeled GT) in
the 10k-15k phase; "full" (full-resolution head only) is the deciding probe family, "hard"
and the D2 contrast "k" are cross-checks:
  (i)  a boundary-specific component exists on the unlabeled truth once the gain directions
       are removed: spec_free = 1 - cos^2(bd_free, rg_free) >= 0.3, the gain-free part keeps
       ||bd_free|| / ||bd|| >= 0.3 of the vector, and the residual ranking is reliable
       (orc split-half rho >= 0.7)
  (ii) the labeled stream can estimate it: its own residual ranking is reliable (lab
       split-half rho >= 0.7, otherwise the ceiling is tiny and the test is vacuous) and
       rho(lab residual, orc residual) >= 0.5 and >= 0.8 * ceiling, ceiling =
       sqrt(lab split-half * orc split-half) of the residuals
Both must hold on the same tap (and on both datasets) to continue the channel-level boundary
route; failing either stops it. rho(lab, orc) after gain removal tells whether the labeled-
guided L->U fallback survives without the shared gain direction.
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
DECISION_PHASE = "10k–15k"
SPEC, KEEP, SPLIT, RHO, RATIO = 0.3, 0.3, 0.7, 0.5, 0.8
STREAMS = ("lab", "unl", "orc")


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


def ceiling(rows):
    values = [math.sqrt(a * b) if a > 0 and b > 0 else math.nan
              for a, b in ((number(row["lab_split_res"]), number(row["orc_split_res"])) for row in rows)]
    values = [value for value in values if not math.isnan(value)]
    return median(values) if values else math.nan


def verdict(values):
    spec, keep, split, lab_split, rho, top = values
    first = spec >= SPEC and keep >= KEEP and split >= SPLIT
    second = lab_split >= SPLIT and rho >= RHO and rho >= RATIO * top
    return f"(i) {'pass' if first else 'fail'}，(ii) {'pass' if second else 'fail'}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", nargs="+", default=["LA_8l", "Pancreas_6l"])
    args = parser.parse_args()
    lines = ["# Step 1c：去增益方向后的通道边界专属性（只观测，oracle 判据）", "",
             "各阶段中位数。判据只看 orc（无标签真值）流，第 10k–15k 步，family=full 为决策探针，hard / k 为交叉检验：",
             f"(i) 去增益后 1−cos²(bd, rg) ≥ {SPEC}，剩余范数占比 ≥ {KEEP}，orc 残差拆半 ρ ≥ {SPLIT}；",
             f"(ii) lab 残差拆半 ρ ≥ {SPLIT}，且 ρ(lab 残差, orc 残差) ≥ {RHO}、≥ {RATIO} × 上限（上限 = √(lab 残差拆半 × orc 残差拆半)）。",
             "两条须在同一 tap 上同时满足（并在两个数据集上）才继续通道级边界路线。k 没有增益方向，剩余范数恒为 1，其 ψ>0 列为 K>0（通道朝正确方向区分两类）的占比。", ""]
    for run in args.runs:
        for baseline in BASELINES:
            path = EXPERIMENT / baseline / "Results_step1c" / f"seed_{args.seed}" / run / "channel_diagnostics.csv"
            if not path.is_file():
                continue
            rows = list(csv.DictReader(path.open()))
            last = max(int(row["iteration"]) for row in rows)
            lines += [f"## {baseline} / {run}（已记录到第 {last} 步）", ""]
            groups = sorted({(row["family"], row["tap"]) for row in rows},
                            key=lambda item: (("full", "hard", "k").index(item[0]), item[1]))
            decisions = []
            for family, tap in groups:
                selected = [row for row in rows if row["family"] == family and row["tap"] == tap]
                lines += [f"### {family} / {tap}", "",
                          "| 阶段 | orc 原始专属 | orc 增益占比 | 剩余范数 | **去增益专属** | **orc 残差拆半** | lab 残差拆半 "
                          "| 残差上限 | **ρ(lab残差,orc残差)** | ρ(lab,orc) 原始 | ρ(lab,orc) 去增益 | ρ(unl,orc) 去增益 "
                          "| ψ>0 占比 lab/unl/orc |",
                          "|---|" + "---|" * 12]
                for start, end, label in PHASES:
                    phase = [row for row in selected if start <= int(row["iteration"]) <= end]
                    if not phase:
                        continue
                    get = lambda column: middle(phase, column)
                    top = ceiling(phase)
                    values = [get("orc_spec"), get("orc_gain_share_bd"), get("orc_keep_bd"), get("orc_spec_free"),
                              get("orc_split_res"), get("lab_split_res"), top, get("rho_lab_orc_res"),
                              get("rho_lab_orc_bd"), get("rho_lab_orc_free"), get("rho_unl_orc_free")]
                    positive = "/".join(fmt(get(f"{stream}_pos_fraction")) for stream in STREAMS)
                    lines.append(f"| {label} | " + " | ".join(fmt(value) for value in values) + f" | {positive} |")
                    if label == DECISION_PHASE:
                        decisions.append(f"- {family} / {tap}：" + verdict(
                            [get("orc_spec_free"), get("orc_keep_bd"), get("orc_split_res"),
                             get("lab_split_res"), get("rho_lab_orc_res"), top]))
                lines.append("")
            lines += [f"判定（{DECISION_PHASE}）：", ""] + (decisions or ["- 尚未到达该阶段"]) + [""]
            # Stream-level per-head losses are repeated on every row; one family suffices.
            heads = [row for row in rows if row["family"] == "full" and row["tap"] == "d2"]
            lines += ["逐头探针损失（h1=1/4, h2=1/2, h3=全分辨率；每格为 h1/h2/h3）", "",
                      "| 阶段 | 流 | 软目标带内 CE | 软目标熵下限 | 硬目标带内 CE | 硬目标带外 CE |", "|---|---|---|---|---|---|"]
            for start, end, label in PHASES:
                phase = [row for row in heads if start <= int(row["iteration"]) <= end]
                for stream in STREAMS if phase else ():
                    cells = ["/".join(fmt(middle(phase, f"{stream}_{name}_h{index}")) for index in (1, 2, 3))
                             for name in ("soft_bd", "soft_floor", "hard_bd", "hard_rg")]
                    lines.append(f"| {label} | {stream} | " + " | ".join(cells) + " |")
            lines.append("")
    output = EXPERIMENT / "summary" / "step1c.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
