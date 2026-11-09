"""Cross-stream check of the step-1b EMA snapshots (cbc_scores.jsonl) into summary/step1b_crosscheck.md.

Asks whether the labeled boundary score carries boundary information about the unlabeled
ground truth, or only generic channel importance:
  - L.rg~O.bd close to L.bd~O.bd   -> the labeled boundary score transfers no more than region does
  - partial rho(U, O | L) near 0    -> the pseudo-label score adds nothing beyond the labeled score
  - O.bd~O.rg near 1 / low residual -> on unlabeled truth, boundary and region use the same channels
Most helpful channels are the most negative psi; chance top-25% Jaccard is about 0.14.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


EXPERIMENT = Path(__file__).resolve().parents[1]
BASELINES = ("MT_DS", "MT_DS_LossNorm")
RUNS = ("LA_8l", "Pancreas_6l")
PHASES = ((1, 2000, "warmup ≤2k"), (2001, 6000, "2k–6k"),
          (6001, 10000, "6k–10k"), (10001, 15000, "10k–15k"))
COLUMNS = ("L.bd~O.bd", "L.rg~O.bd", "L.rg~O.rg", "U.bd~O.bd", "ρ(U,O|L)", "ρ(L,O|U)",
           "L.res~O.res", "U.res~O.res", "O.bd~O.rg", "O 残差占比", "J(L,O)", "J(U,O)")


def rho(a, b):
    return spearmanr(a, b)[0]


def partial_rho(a, b, given):
    """Spearman partial correlation of a and b controlling for given."""
    ranks = [rankdata(v) for v in (a, b, given)]
    residuals = [x - np.polyval(np.polyfit(ranks[2], x, 1), ranks[2]) for x in ranks[:2]]
    return np.corrcoef(*residuals)[0, 1]


def residual(boundary, region):
    return boundary - (boundary @ region) / (region @ region) * region


def top_jaccard(a, b, fraction=0.25):
    k = max(1, round(fraction * len(a)))
    left, right = set(np.argsort(a)[:k]), set(np.argsort(b)[:k])
    return len(left & right) / len(left | right)


def snapshot_row(ema, tap):
    get = lambda stream, kind: np.asarray(ema[f"{stream}/{tap}/{kind}"])
    lab, lab_rg = get("lab", "bd"), get("lab", "rg")
    unl, unl_rg = get("unl", "bd"), get("unl", "rg")
    orc, orc_rg = get("orc", "bd"), get("orc", "rg")
    orc_cos = orc @ orc_rg / np.linalg.norm(orc) / np.linalg.norm(orc_rg)
    return [rho(lab, orc), rho(lab_rg, orc), rho(lab_rg, orc_rg), rho(unl, orc),
            partial_rho(unl, orc, lab), partial_rho(lab, orc, unl),
            rho(residual(lab, lab_rg), residual(orc, orc_rg)),
            rho(residual(unl, unl_rg), residual(orc, orc_rg)),
            rho(orc, orc_rg), 1 - orc_cos ** 2, top_jaccard(lab, orc), top_jaccard(unl, orc)]


def main():
    lines = ["# Step 1b 交叉检验：边界分数是否携带超出区域 / 通用重要性的信息", "",
             "基于每 500 步的 EMA 快照，各阶段中位数。L=有标签，U=伪标签，O=无标签 oracle（隐藏 GT）；",
             "res = 边界分数去掉本流区域分量后的残差；ρ(·,·|·) 为偏 Spearman；J 为最有帮助 25% 通道的 Jaccard（随机≈0.14）。", ""]
    for run, baseline in ((run, baseline) for run in RUNS for baseline in BASELINES):
        path = EXPERIMENT / baseline / "Results_step1b/seed_42" / run / "cbc_scores.jsonl"
        if not path.exists():
            continue
        snapshots = [json.loads(line) for line in path.read_text().splitlines() if line]
        for tap in ("d3", "d2"):
            lines += [f"## {baseline} / {run} / {tap}", "",
                      "| 阶段 | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
            for low, high, name in PHASES:
                rows = [snapshot_row(s["ema"], tap) for s in snapshots if low <= s["iteration"] <= high]
                if rows:
                    values = np.median(np.array(rows), axis=0)
                    lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
            lines.append("")
    output = EXPERIMENT / "summary" / "step1b_crosscheck.md"
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
