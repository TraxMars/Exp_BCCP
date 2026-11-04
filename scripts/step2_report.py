"""Compare step-2 runs (random D2 channel dropout) with their baselines into summary/step2.md.

Reported per run: best validation Dice and its iteration, the mean of the last five
validations (less sensitive to test-list model selection on LA/Pancreas), the final test of
the best checkpoint, and the perturbation statistics after warmup.
"""
import argparse
import csv
import json
import math
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
BASELINES = ("MT_DS", "MT_DS_LossNorm")
CONDITIONS = (("随机丢弃 p=0.5", "random_d2_p0.5"), ("不丢弃对照 p=0", "nodrop_d2_p0"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def summarize(run):
    validations = read_jsonl(run / "validation_metrics.jsonl")
    test = json.loads((run / "test_metrics.json").read_text()) if (run / "test_metrics.json").is_file() else {}
    status = json.loads((run / "run_status.json").read_text()) if (run / "run_status.json").is_file() else {}
    best = max(validations, key=lambda row: row["dice"], default=None)
    last = [row["dice"] for row in validations[-5:]]
    fp, change = [], []
    if (run / "training_metrics.csv").is_file():
        for row in csv.DictReader((run / "training_metrics.csv").open()):
            for store, key in ((fp, "loss_fp"), (change, "fp_relative_change")):
                value = float(row.get(key, "nan") or "nan")
                if not math.isnan(value):
                    store.append(value)
    mean = lambda values: sum(values) / len(values) if values else math.nan
    return dict(state=status.get("state", "not started"), iteration=status.get("iteration", 0),
                best=best["dice"] * 100 if best else math.nan, best_iteration=best["iteration"] if best else None,
                last5=mean(last) * 100 if last else math.nan,
                test={key: test.get(key) for key in ("dice", "jc", "hd95", "asd")},
                loss_fp=mean(fp), fp_change=mean(change))


def fmt(value, scale=1.0):
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value * scale:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", nargs="+", default=["LA_8l", "Pancreas_6l"])
    args = parser.parse_args()
    lines = ["# Step 2：无标签 D2 通道丢弃（权重 1）与不丢弃对照", "",
             "Dice / JC 为百分数，HD95 / ASD 为体素。LA 的在线验证与测试是同一 test.list。", "",
             "| 数据 | 基线 | 条件 | 状态 | 最佳验证 Dice（步） | 最后 5 次均值 | Test Dice | JC | HD95 | ASD | loss_fp (×1e-3) | 相对扰动 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, baseline in ((name, baseline) for name in args.runs for baseline in BASELINES):
        runs = [("基线", EXPERIMENT / baseline / "Results" / f"seed_{args.seed}" / name)]
        runs += [(label, EXPERIMENT / baseline / "Results_step2" / condition / f"seed_{args.seed}" / name)
                 for label, condition in CONDITIONS]
        for label, run in runs:
            if label != "基线" and not run.is_dir():
                continue
            result = summarize(run)
            test = result["test"]
            lines.append(
                f"| {name} | {baseline} | {label} | {result['state']} ({result['iteration']}) | "
                f"{fmt(result['best'])}（{result['best_iteration']}） | {fmt(result['last5'])} | "
                f"{fmt(test['dice'], 100)} | {fmt(test['jc'], 100)} | {fmt(test['hd95'])} | {fmt(test['asd'])} | "
                f"{fmt(result['loss_fp'], 1000)} | {fmt(result['fp_change'])} |")
    output = EXPERIMENT / "summary" / "step2.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
