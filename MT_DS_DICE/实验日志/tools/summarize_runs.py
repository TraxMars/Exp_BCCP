#!/usr/bin/env python3
"""只读汇总实验 run 的指标，供每日实验日志使用。不写任何文件，结果以 Markdown 打印到 stdout。

  python3 summarize_runs.py --date 2026-11-05 --baseline <基线 Results 目录> <仓库目录> [<仓库目录> ...]

run 目录 = <仓库>/Results*/ 下含 config.json 的目录；launch_logs/*/manifest.json 里列出但尚未建立的 run
记为"未开始"。validation_metrics.jsonl / *.csv 中同一步重复的行（resume 后）只保留最后一行。
Dice、JC 以百分数显示（文件值 ×100，2 位小数）；HD95、ASD 为体素，2 位小数；诊断量 4 位有效数字。
标准差为样本标准差（n−1）。
"""
import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

KEY_DIAG = ["band_iou", "ce_gt_in", "ce_pl_in", "ce_lab_in", "teacher_err_gt_band", "conf_wrong_095"]
# MT_DS_PL 基线（commit 49758b3）diagnostics.csv 的列；新实验多出来的列会自动列出
BASE_DIAG = set("""iteration gnorm_sup_b7 gnorm_pl_b7 cos_b7 cos_pl_true_b7 cos_sup_true_b7 frac_ch_pl_true_neg_b7
frac_ch_pl_sup_neg_b7 ch_rank_corr_b7 hit_rate_b7 gnorm_sup_b3 gnorm_pl_b3 cos_b3 cos_pl_true_b3 cos_sup_true_b3
frac_ch_pl_true_neg_b3 frac_ch_pl_sup_neg_b3 ch_rank_corr_b3 hit_rate_b3 ce_pl_in ce_gt_in ce_pl_out ce_gt_out
band_iou teacher_conf_gt_band teacher_conf_pl_band gt_band_fraction teacher_err_gt_band conf_wrong_095
conf_right_095 ce_lab_in ce_lab_out""".split())
PARAMS = ["max_iterations", "warmup_iterations", "consistency", "base_lr", "ema_decay", "batch_size", "labeled_bs",
          "val_interval", "val_start", "diag_interval", "patch_size", "resume"]
DIAG_RANGE = (6000, 15000)  # 闭区间
TEST_KEYS = ["dice", "jc", "hd95", "asd"]
ERROR_RE = re.compile(r"Traceback|Error|out of memory|Killed|non-finite|\bnan\b", re.IGNORECASE)
TIME_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(values):
    return [v for v in values if not math.isnan(v) and not math.isinf(v)]


def pct(value):
    return "—" if value is None or math.isnan(value) else f"{100 * value:.2f}"


def fix(value, digits=2):
    return "—" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def sig(value):
    return "—" if value is None or math.isnan(value) else f"{value:.4g}"


def by_iteration(rows):
    last = {}
    for row in rows:
        it = num(row.get("iteration"))
        if not math.isnan(it):
            last[int(it)] = row
    return [last[it] for it in sorted(last)]


def read_jsonl(path):
    rows = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:  # 正在写入的不完整行
                pass
    return by_iteration(rows)


def read_csv(path):
    if not path.is_file():
        return [], []
    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("iteration")]
        return reader.fieldnames or [], by_iteration(rows)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def processes():
    """[(pid, ppid, etimes, args)]，只读 ps。"""
    out = []
    for line in run_cmd(["ps", "-eo", "pid=,ppid=,etimes=,user=,args="]).splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5:
            out.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3], parts[4]))
    return out


def gpu_of_pid():
    uuid_to_index = {}
    for line in run_cmd(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            uuid_to_index[parts[1]] = parts[0]
    result = {}
    for line in run_cmd(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                         "--format=csv,noheader,nounits"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            result[int(parts[0])] = (uuid_to_index.get(parts[1], "?"), parts[2])
    return result


def main_pid_for(output_dir, procs):
    """含 `--output_dir <output_dir>` 的训练主进程（排除 DataLoader 子进程）。"""
    pattern = re.compile(r"--output_dir[= ]" + re.escape(str(output_dir)) + r"/?(\s|$)")
    matched = {p[0]: p for p in procs if "train.py" in p[4] and pattern.search(p[4])}
    mains = [p for p in matched.values() if p[1] not in matched]
    return mains[0] if mains else None


def first_last_time(path):
    if not path.is_file():
        return None, None
    lines = path.read_text(errors="replace").splitlines()
    first = next((TIME_RE.match(l).group(1) for l in lines if TIME_RE.match(l)), None)
    last = next((TIME_RE.match(l).group(1) for l in reversed(lines) if TIME_RE.match(l)), None)
    return first, last


def error_lines(path, limit=5):
    if not path or not Path(path).is_file():
        return []
    lines = [l.strip() for l in Path(path).read_text(errors="replace").splitlines() if ERROR_RE.search(l)]
    return lines[-limit:]


def updated_on(run_dir, day_start, day_end):
    for f in run_dir.rglob("*"):
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
        except OSError:
            continue
        if day_start <= mtime < day_end:
            return True
    return False


def summarize_run(run_dir, root, repo, manifest_entry, procs, gpus, day_start, day_end):
    config = read_json(run_dir / "config.json") or {}
    args = config.get("args", {})
    train_cols, train_rows = read_csv(run_dir / "training_metrics.csv")
    diag_cols, diag_rows = read_csv(run_dir / "diagnostics.csv")
    val = read_jsonl(run_dir / "validation_metrics.jsonl")
    test = read_json(run_dir / "test_metrics.json")
    best = read_json(run_dir / "best_metrics.json")
    first_time, last_time = first_last_time(run_dir / "log.txt")
    log_text = (run_dir / "log.txt").read_text(errors="replace") if (run_dir / "log.txt").is_file() else ""
    finished = re.search(r"finished at iteration (\d+), best validation Dice ([\d.]+)", log_text)
    proc = main_pid_for(run_dir, procs)
    max_it = int(num(args.get("max_iterations")) or 0) if args.get("max_iterations") else None
    current = int(num(train_rows[-1]["iteration"])) if train_rows else None

    if test is not None:
        status = "已完成"
    elif proc:
        status = "运行中"
    elif finished:
        status = "已结束但缺少 test_metrics.json"
    else:
        status = "未在运行且未完成（疑似中断）"

    vals = [(int(r["iteration"]), num(r.get("dice"))) for r in val]
    best_val = max(vals, key=lambda x: (x[1], -x[0])) if vals else None  # 并列取最早一步（与严格大于保存一致）
    last5 = [d for _, d in vals[-5:]]

    nonfinite = 0
    for row in train_rows:
        for key in ("loss_total", "loss_supervised", "loss_pl", "gradient_norm"):
            v = num(row.get(key))
            if key in row and (math.isnan(v) or math.isinf(v)):
                nonfinite += 1

    lo, hi = DIAG_RANGE
    in_range = [r for r in diag_rows if lo <= int(num(r["iteration"])) <= hi]
    extra = [c for c in diag_cols if c not in BASE_DIAG]
    diag = {}
    for col in KEY_DIAG + extra:
        if col not in diag_cols:
            diag[col] = (None, 0, 0)
            continue
        values = [num(r.get(col)) for r in in_range]
        good = finite(values)
        diag[col] = (statistics.median(good) if good else math.nan, len(good), len(values) - len(good))
    gpu = None
    if proc and proc[0] in gpus:
        gpu = f"{gpus[proc[0]][0]}（nvidia-smi，{gpus[proc[0]][1]} MiB）"
    elif manifest_entry and "gpu" in manifest_entry:
        gpu = f"{manifest_entry['gpu']}（launch manifest）"

    return dict(
        path=str(run_dir), root=str(root), repo=str(repo), args=args, commit=config.get("git_commit", "?"),
        dataset=args.get("dataset", "?"), labelnum=args.get("labelnum", "?"), seed=args.get("seed", "?"),
        status=status, pid=proc[0] if proc else None, gpu=gpu, start=first_time, last_log=last_time,
        current=current, max_it=max_it, vals=vals, best_val=best_val,
        last5=statistics.mean(last5) if last5 else math.nan, n_last5=len(last5),
        best_file=(num(best.get("dice")), best.get("iteration")) if best else None,
        test=test, finished=finished.groups() if finished else None, nonfinite=nonfinite,
        diag=diag, diag_n=len(in_range),
        diag_span=(int(num(in_range[0]["iteration"])), int(num(in_range[-1]["iteration"]))) if in_range else None,
        extra_diag=extra, updated=updated_on(run_dir, day_start, day_end),
        errors=error_lines(run_dir / "log.txt") + error_lines(manifest_entry.get("log") if manifest_entry else None),
        empty=(test.get("empty_prediction_cases"), test.get("nms1", {}).get("empty_prediction_cases")) if test else None,
    )


def mean_std(values):
    values = finite(values)
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.2f}（n=1）"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}（n={len(values)}）"


def metric_values(run):
    """用于汇总和配对的指标（Dice/JC 已 ×100）；只取已完成的 run，运行中的 run 步数不同，不参与。"""
    out = {}
    t = run["test"]
    if not t:
        return out
    for prefix, entry in (("nms0", t), ("nms1", t.get("nms1") or {})):
        for k in TEST_KEYS:
            v = num(entry.get(k))
            out[f"test_{prefix}_{k}"] = 100 * v if k in ("dice", "jc") else v
    if run["best_val"]:
        out["best_val_dice"] = 100 * run["best_val"][1]
    if run["n_last5"] == 5:
        out["last5_val_dice"] = 100 * run["last5"]
    return out


METRIC_ORDER = [f"test_{p}_{k}" for p in ("nms0", "nms1") for k in TEST_KEYS] + ["best_val_dice", "last5_val_dice"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--baseline", type=Path, help="基线 Results 根目录（配对差值的参照）")
    parser.add_argument("repos", nargs="+", type=Path)
    a = parser.parse_args()
    day_start = datetime.strptime(a.date, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)
    procs, gpus = processes(), gpu_of_pid()

    runs, pending, launchers = [], [], []
    for repo in a.repos:
        for root in sorted(p for p in repo.glob("Results*") if p.is_dir()):
            manifest = {}
            for mf in sorted(root.glob("launch_logs/*/manifest.json")):
                for entry in read_json(mf) or []:
                    manifest[str(Path(entry["output_dir"]))] = dict(entry, manifest=str(mf))
            found = set()
            for cfg in sorted(root.rglob("config.json")):
                run_dir = cfg.parent
                if "launch_logs" in run_dir.parts:
                    continue
                found.add(str(run_dir))
                runs.append(summarize_run(run_dir, root, repo, manifest.get(str(run_dir)), procs, gpus,
                                          day_start, day_end))
            for out_dir, entry in manifest.items():
                if out_dir not in found:
                    pending.append(dict(entry, root=str(root)))
    for p in procs:
        if "launch.py" in p[4] and "grep" not in p[4]:
            launchers.append(p)

    print(f"# summarize_runs.py 自动汇总（日期 {a.date}，生成于 {datetime.now():%Y-%m-%d %H:%M:%S}）\n")
    print("单位：Dice/JC 为百分数（文件值×100）；HD95/ASD 为体素；标准差为样本标准差（n−1）；"
          f"诊断中位数取 iteration ∈ [{DIAG_RANGE[0]}, {DIAG_RANGE[1]}] 的行，忽略 NaN。\n")

    print("## 1. Run 清单与状态\n")
    print("| run 路径 | 状态 | 今日有更新 | 开始时间(log.txt) | 最近日志时间 | 当前步/总步 | 最近验证 Dice@步 | GPU | PID | commit |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        last = f"{pct(r['vals'][-1][1])}@{r['vals'][-1][0]}" if r["vals"] else "—"
        print(f"| {r['path']} | {r['status']} | {'是' if r['updated'] else '否'} | {r['start'] or '—'} | "
              f"{r['last_log'] or '—'} | {r['current']}/{r['max_it']} | {last} | {r['gpu'] or '—'} | "
              f"{r['pid'] or '—'} | {r['commit'][:7]} |")
    for p in pending:
        print(f"| {p['output_dir']} | 未开始（manifest 中列出，目录尚未建立） | — | — | — | — | — | — | — | — |")
    print()
    if launchers:
        print("仍在运行的 launch.py 进程（可能在等待空闲 GPU 启动剩余 run）：")
        for p in launchers:
            print(f"- PID {p[0]}，已运行 {timedelta(seconds=p[2])}：`{p[4]}`")
        print()

    print("## 2. 关键参数（config.json → args）\n")
    print("| run 路径 | 数据集 | 标注数 | seed | " + " | ".join(PARAMS) + " | 其他参数 |")
    print("|---|---|---|---|" + "---|" * len(PARAMS) + "---|")
    known = set(PARAMS) | {"dataset", "labelnum", "seed", "data_root", "output_dir", "device", "num_workers",
                           "cpu_threads", "log_interval", "checkpoint_interval", "cache_volumes"}
    for r in runs:
        other = {k: v for k, v in r["args"].items() if k not in known}
        print(f"| {r['path']} | {r['dataset']} | {r['labelnum']} | {r['seed']} | "
              + " | ".join(str(r["args"].get(k, "—")) for k in PARAMS) + f" | {json.dumps(other) if other else '—'} |")
    print()

    done = [r for r in runs if r["test"]]
    print("## 3. 已完成 run 的 test 结果（test_metrics.json，最佳验证模型）\n")
    if done:
        print("| run 路径 | nms0 Dice | nms0 JC | nms0 HD95 | nms0 ASD | nms1 Dice | nms1 JC | nms1 HD95 | nms1 ASD | 空预测例数 nms0/nms1 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in done:
            t, t1 = r["test"], r["test"].get("nms1") or {}
            print(f"| {r['path']}/test_metrics.json | {pct(num(t.get('dice')))} | {pct(num(t.get('jc')))} | "
                  f"{fix(num(t.get('hd95')))} | {fix(num(t.get('asd')))} | {pct(num(t1.get('dice')))} | "
                  f"{pct(num(t1.get('jc')))} | {fix(num(t1.get('hd95')))} | {fix(num(t1.get('asd')))} | "
                  f"{r['empty'][0]}/{r['empty'][1]} |")
    else:
        print("（暂无 run 生成 test_metrics.json）")
    print()

    print("## 4. 验证集指标（validation_metrics.jsonl）\n")
    print("| run 路径 | 验证次数 | 最佳验证 Dice | 最佳步数 | best_metrics.json Dice@步 | 最后 5 次验证 Dice 均值 | 最近验证 Dice@步 |")
    print("|---|---|---|---|---|---|---|")
    for r in runs:
        bv = r["best_val"]
        bf = f"{pct(r['best_file'][0])}@{r['best_file'][1]}" if r["best_file"] else "—"
        l5 = pct(r["last5"]) + ("" if r["n_last5"] == 5 else f"（仅 {r['n_last5']} 次）")
        last = f"{pct(r['vals'][-1][1])}@{r['vals'][-1][0]}" if r["vals"] else "—"
        print(f"| {r['path']}/validation_metrics.jsonl | {len(r['vals'])} | {pct(bv[1]) if bv else '—'} | "
              f"{bv[0] if bv else '—'} | {bf} | {l5} | {last} |")
    print()

    print(f"## 5. 诊断量中位数（diagnostics.csv，iteration ∈ [{DIAG_RANGE[0]}, {DIAG_RANGE[1]}]）\n")
    extra_all = sorted({c for r in runs for c in r["extra_diag"]})
    cols = KEY_DIAG + extra_all
    print("| run 路径 | 覆盖步数范围 | 行数 | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for r in runs:
        span = f"{r['diag_span'][0]}–{r['diag_span'][1]}" if r["diag_span"] else "未到 6000 步"
        cells = []
        for c in cols:
            med, n, n_nan = r["diag"].get(c, (None, 0, 0))
            cells.append("无此列" if med is None else sig(med) + (f"（NaN {n_nan}）" if n_nan else ""))
        print(f"| {r['path']}/diagnostics.csv | {span} | {r['diag_n']} | " + " | ".join(cells) + " |")
    print()
    if extra_all:
        print(f"新实验新增的诊断列（不在 MT_DS_PL 基线中）：{', '.join(extra_all)}\n")

    print("## 6. 多 seed 汇总（均值 ± 样本标准差；所有指标只统计已完成的 run）\n")
    groups = {}
    for r in runs:
        groups.setdefault((r["root"], f"{r['dataset']}_{r['labelnum']}l"), []).append(r)
    print("| Results 根目录 | 数据集 | seeds（已完成/全部） | " + " | ".join(METRIC_ORDER) + " |")
    print("|---|---|---|" + "---|" * len(METRIC_ORDER))
    for (root, ds), members in sorted(groups.items()):
        values = [metric_values(m) for m in members]
        seeds_done = [str(m["seed"]) for m in members if m["test"]]
        cells = [mean_std([v[k] for v in values if k in v]) for k in METRIC_ORDER]
        print(f"| {root} | {ds} | {','.join(seeds_done) or '无'} / {','.join(str(m['seed']) for m in members)} | "
              + " | ".join(cells) + " |")
    print()

    print("## 7. 按 seed 配对的差值（实验 − 基线，同数据集同 seed）\n")
    base = str(a.baseline) if a.baseline else None
    rows = []
    for (root, ds), members in sorted(groups.items()):
        if root == base:
            continue
        ref = {m["seed"]: metric_values(m) for m in groups.get((base, ds), [])}
        for m in members:
            if m["seed"] in ref:
                mine, theirs = metric_values(m), ref[m["seed"]]
                diffs = {k: mine[k] - theirs[k] for k in METRIC_ORDER if k in mine and k in theirs}
                if diffs:
                    rows.append((root, ds, m["seed"], diffs))
    if rows:
        print("| 实验 Results 根目录 | 数据集 | seed | " + " | ".join(METRIC_ORDER) + " |")
        print("|---|---|---|" + "---|" * len(METRIC_ORDER))
        for root, ds, seed, diffs in rows:
            print(f"| {root} | {ds} | {seed} | " + " | ".join(f"{diffs[k]:+.2f}" if k in diffs else "—"
                                                              for k in METRIC_ORDER) + " |")
        print("\n配对差值的均值 ± 样本标准差：\n")
        agg = {}
        for root, ds, seed, diffs in rows:
            for k, v in diffs.items():
                agg.setdefault((root, ds), {}).setdefault(k, []).append(v)
        for (root, ds), d in agg.items():
            print(f"- {root} / {ds}：" + "；".join(f"{k} {mean_std(d[k])}" for k in METRIC_ORDER if k in d))
    else:
        print(f"（没有可与基线 {base} 配对的实验 run）")
    print()

    print("## 8. 异常检查\n")
    any_issue = False
    for r in runs:
        notes = []
        if r["nonfinite"]:
            notes.append(f"training_metrics.csv 中有 {r['nonfinite']} 个非有限值（loss/grad）")
        if any(n_nan for _, _, n_nan in r["diag"].values()):
            notes.append("关键诊断列在统计区间内有 NaN（见第 5 节）")
        if r["errors"]:
            notes.append("日志中匹配到报错关键词：" + " ‖ ".join(e[:200] for e in r["errors"]))
        if r["status"].startswith("未在运行") or r["status"].startswith("已结束但"):
            notes.append(f"状态：{r['status']}")
        if r["test"] and r["empty"] and (r["empty"][0] or r["empty"][1]):
            notes.append(f"test 中有空预测：nms0 {r['empty'][0]} 例，nms1 {r['empty'][1]} 例")
        if notes:
            any_issue = True
            print(f"- {r['path']}：" + "；".join(notes))
    if not any_issue:
        print("（未发现 NaN、报错关键词或异常中断）")


if __name__ == "__main__":
    main()
