"""Boundary-error anatomy report: Results_anatomy/seed_<s>/<run>/ -> summary/anatomy.md, with README's decision rules.

  python scripts/report_anatomy.py --root Results_anatomy --output summary/anatomy.md
Per run: shares and the other quantities from cases.csv / layers.csv / separability.csv; per dataset: mean (and
sample std) over the seeds; the decision uses the means (README.md, "Boundary-error anatomy").
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parents[1]
CATEGORIES = ("block", "shift", "jitter")
SIGNALS = ("entropy", "ds_disagreement", "flip_eval", "tta_train")
LAYERS = ("1", "2", "3", "4-6", ">6")
THRESHOLDS = [f"{t:.2f}" for t in [round(0.05 * k, 2) for k in range(1, 20)]]


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    value = float(value)
    return None if math.isnan(value) else value


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def std(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def pm(values, digits=3, scale=1):
    m, s = mean(values), std(values)
    if m is None:
        return "n/a"
    return f"{m * scale:.{digits}f}" + (f" ± {s * scale:.{digits}f}" if s is not None else "")


def run_quantities(path):
    cases = read(path / "cases.csv")
    column = lambda key: [number(row[key]) for row in cases]
    q = dict(cases=len(cases), dice=mean(column("dice")), hd95=mean(column("hd95")),
             err_gt_band=mean(column("err_gt_band")), conf_wrong_095=mean(column("conf_wrong_095")))
    gains = {name: mean(column(f"ddice_{name}")) for name in CATEGORIES}
    total = sum(gains.values())
    for name in CATEGORIES:
        q[f"share_{name}"] = gains[name] / total if total else None
        q[f"ddice_{name}"] = gains[name]
        q[f"dhd95_{name}"] = mean(column(f"dhd95_{name}"))
        q[f"voxels_{name}"] = sum(column(f"n_{name}")) / max(sum(column("n_err")), 1)
    q["ddice_all"] = mean(column("ddice_all"))
    q["reachable"] = mean(column("ddice_reach")) / q["ddice_all"] if q["ddice_all"] else None
    fp, fn = sum(column("n_shift_fp")), sum(column("n_shift_fn"))
    q["direction"] = max(fp, fn) / (fp + fn) if fp + fn else None
    q["direction_sign"] = "over (FP)" if fp >= fn else "under (FN)"
    q["cases_over"] = sum(f > n for f, n in zip(column("n_shift_fp"), column("n_shift_fn"))) / len(cases)
    curve = {t: mean(column(f"dice_t{t}")) for t in THRESHOLDS}
    best = max(curve, key=curve.get)
    q["oracle_global_t"], q["oracle_global_gain"] = float(best), curve[best] - curve["0.50"]
    q["oracle_case_gain"] = mean([max(number(row[f"dice_t{t}"]) for t in THRESHOLDS) - number(row["dice_t0.50"])
                                  for row in cases])
    if "ref_dice" in cases[0]:
        ref_gains = {name: mean(column(f"ref_ddice_{name}")) for name in CATEGORIES}
        ref_total = sum(ref_gains.values())
        q["ref_dice"] = mean(column("ref_dice"))
        for name in CATEGORIES:
            q[f"ref_share_{name}"] = ref_gains[name] / ref_total if ref_total else None

    layers = {row["layer"]: {k: int(v) for k, v in row.items() if k != "layer"} for row in read(path / "layers.csv")}
    q["layers"] = layers
    separability = read(path / "separability.csv")
    q["auroc"] = {(row["layer"], row["signal"]): number(row["auroc"]) for row in separability}
    q["ap"] = {(row["layer"], row["signal"]): number(row["ap"]) for row in separability}
    q["base_rate"] = {row["layer"]: number(row["base_rate"]) for row in separability}
    q["case_auroc"] = {(row["layer"], row["signal"]): number(row["case_median_auroc"]) for row in separability}
    for signal in SIGNALS:
        weights = [(int(row["n_err"]), number(row["auroc"])) for row in separability
                   if row["signal"] == signal and row["layer"] in ("1", "2", "3") and number(row["auroc"]) is not None]
        q[f"A_{signal}"] = sum(n * a for n, a in weights) / sum(n for n, _ in weights) if weights else None
    q["summary"] = json.loads((path / "summary.json").read_text())
    return q


def decide(q):
    """README rules 1-3 on the seed means of one dataset."""
    shares = {name: q[f"share_{name}"] for name in CATEGORIES}
    dominant = max(shares, key=shares.get)
    if shares[dominant] <= 0.5:
        top = sorted(shares, key=shares.get, reverse=True)[:2]
        kind, action = f"mixed ({top[0]} {shares[top[0]]:.2f}, {top[1]} {shares[top[1]]:.2f})", "discuss"
    elif dominant == "shift":
        kind = "shift"
        action = ("logit bias / threshold calibration (oracle gain is its upper bound)" if q["direction"] >= 0.7
                  else "shifts vary per case or region: global calibration does not apply, discuss")
    elif dominant == "jitter":
        kind, action = "jitter", "boundary tolerance or distance-softened pseudo-labels"
    else:
        kind = "block"
        action = f"new information needed (shape / position priors); U→L reaches at most {q['reachable']:.2f}"
    separating = [s for s in SIGNALS if q[f"A_{s}"] is not None and q[f"A_{s}"] >= 0.70
                  and (s == "entropy" or q["A_entropy"] is None or q[f"A_{s}"] >= q["A_entropy"] + 0.05)]
    return kind, action, separating


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=PROJECT / "Results_anatomy")
    parser.add_argument("--output", type=Path, default=PROJECT / "summary/anatomy.md")
    args = parser.parse_args()
    runs = {}
    for path in sorted(args.root.glob("seed_*/*/summary.json")):
        runs.setdefault(path.parent.name, {})[path.parent.parent.name] = run_quantities(path.parent)

    lines = [f"# Boundary-error anatomy of the V0 Teachers ({args.root.name})", "",
             "Teacher of each V0 run at iteration 15000, unlabeled training cases with their hidden GT; mode `train`"
             " (pseudo-labels as made in training) unless marked `sliding`. Mean ± sample std over the seeds. "
             "Definitions and rules: README.md, \"Boundary-error anatomy\".", ""]
    decisions = {}
    for dataset, seeds in runs.items():
        seq = list(seeds.values())
        avg = lambda key: mean([q[key] for q in seq])
        lines += [f"## {dataset} (seeds {', '.join(s.replace('seed_', '') for s in seeds)}; "
                  f"{seq[0]['cases']} cases each)", ""]
        lines += ["### Error types (contribution to Dice)", ""]
        lines += table(["", "block", "shift", "jitter"],
                       [["share of ΔDice"] + [pm([q[f"share_{c}"] for q in seq]) for c in CATEGORIES],
                        ["ΔDice (points)"] + [pm([q[f"ddice_{c}"] for q in seq], 2, 100) for c in CATEGORIES],
                        ["ΔHD95 (voxels, reported only)"] + [pm([q[f"dhd95_{c}"] for q in seq], 2) for c in CATEGORIES],
                        ["share of error voxels"] + [pm([q[f"voxels_{c}"] for q in seq]) for c in CATEGORIES],
                        ["sliding reference: share of ΔDice"] + [pm([q.get(f"ref_share_{c}") for q in seq])
                                                                 for c in CATEGORIES]])
        lines += ["",
                  f"- Dice of the pseudo-labels (mode `train`): {pm([q['dice'] for q in seq], 2, 100)}; "
                  f"sliding reference: {pm([q.get('ref_dice') for q in seq], 2, 100)}; HD95 {pm([q['hd95'] for q in seq], 2)}.",
                  f"- Shift direction consistency: {pm([q['direction'] for q in seq])} "
                  f"(majority: {', '.join(q['direction_sign'] for q in seq)}); share of cases over-segmented: "
                  f"{pm([q['cases_over'] for q in seq])}.",
                  f"- Oracle threshold (upper bound of calibration): best single threshold "
                  f"{', '.join(fmt(q['oracle_global_t'], 2) for q in seq)} → +{pm([q['oracle_global_gain'] for q in seq], 2, 100)} "
                  f"Dice points; best per case → +{pm([q['oracle_case_gain'] for q in seq], 2, 100)}.",
                  f"- Reachable by U→L (errors with |d̃| ≤ 6, share of ΔDice): {pm([q['reachable'] for q in seq])}.",
                  f"- Check against training diagnostics: Teacher error in the GT band (radius 2) "
                  f"{pm([q['err_gt_band'] for q in seq])}, conf > 0.95 among those errors "
                  f"{pm([q['conf_wrong_095'] for q in seq])}.", ""]
        lines += ["### By pseudo layer |d̃| (pooled over cases and seeds)", ""]
        rows = []
        for layer in LAYERS:
            c = {k: sum(q["layers"][layer][k] for q in seq) for k in seq[0]["layers"][layer]}
            all_errors = sum(sum(q["layers"][name]["n_err"] for name in LAYERS) for q in seq)
            rows.append([layer, fmt(c["n_err"] / c["n_voxels"]), fmt(c["n_err"] / all_errors),
                         fmt(c["n_err_fp"] / max(c["n_err"], 1)),
                         fmt(c["n_err_conf095"] / max(c["n_conf095"], 1))] +
                        [fmt(c[f"n_{name}"] / max(c["n_err"], 1)) for name in CATEGORIES])
        lines += table(["|d̃|", "error rate", "share of errors", "FP share", "error rate, conf > 0.95",
                        "block", "shift", "jitter"], rows)
        lines += ["", "### Separability of confident errors (conf > 0.95; AUROC, mean over seeds)", ""]
        rows = []
        for signal in SIGNALS:
            rows.append([signal] + [pm([q["auroc"].get((layer, signal)) for q in seq]) for layer in ("1", "2", "3")]
                        + [pm([q[f"A_{signal}"] for q in seq]), pm([q["auroc"].get(("1-3 pooled", signal)) for q in seq])]
                        + [" / ".join(fmt(mean([q["ap"].get((layer, signal)) for q in seq])) for layer in ("1", "2", "3"))])
        lines += table(["signal", "|d̃| = 1", "|d̃| = 2", "|d̃| = 3", "A (weighted)", "unstratified 1–3 (not used)",
                        "AP 1 / 2 / 3"], rows)
        lines += ["", "Error base rate among confident voxels, |d̃| = 1 / 2 / 3: " +
                  " / ".join(fmt(mean([q["base_rate"].get(layer) for q in seq])) for layer in ("1", "2", "3")) +
                  ". `entropy` ranks exactly as the confidence (two classes).", ""]
        means = {key: avg(key) for key in ("share_block", "share_shift", "share_jitter", "direction", "reachable",
                                            *[f"A_{s}" for s in SIGNALS])}
        kind, action, separating = decide(means)
        decisions[dataset] = dict(kind=kind, non_block=1 - means["share_block"])
        lines += ["### Decision (README rules 1–3)", "",
                  f"- Dominant error type: **{kind}** → {action}.",
                  f"- Signals separating confident errors (A ≥ 0.70 and ≥ A_entropy + 0.05): "
                  f"**{', '.join(separating) if separating else 'none'}**"
                  + (" → reliability weighting with it is viable." if separating
                     else " → confident errors are not detectable with these signals."),
                  f"- Non-block share: {fmt(1 - means['share_block'])}.", ""]
        lines += ["Runs: " + "; ".join(f"{s}: {q['summary']['run']} (Teacher iteration {q['summary']['iteration']}, "
                                         f"anatomy commit {q['summary']['anatomy_commit'][:7]}, "
                                         f"{q['summary']['seconds'] / 60:.0f} min)" for s, q in seeds.items()), ""]
    if decisions:
        ok = all(d["non_block"] >= 0.5 for d in decisions.values()) and len(decisions) == 2
        lines += ["## Together with V1 (README rule 4)", "",
                  "Non-block share ≥ 0.5 on both datasets: " + ("**yes**" if ok else "**no**") + " (" +
                  ", ".join(f"{k} {v['non_block']:.2f}" for k, v in decisions.items()) + "). "
                  + ("If V1 says go → V2 (scheme as V1 decides); V1 stop → discuss." if ok
                     else "V1 go → discuss (block errors dominate on a dataset); V1 stop → discuss."), ""]
    text = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
