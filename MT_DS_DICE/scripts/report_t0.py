"""T0 report: Results_T0/seed_<s>/<run>/cases.csv -> summary/T0.md, with the checks and criteria of README.md ("T0").

  python scripts/report_t0.py --root Results_T0 --anatomy Results_anatomy --output summary/T0.md
Per seed a quantity is the mean over cases; tables give mean ± sample std over the seeds. Δ_X = mean over seeds of
[mean Dice(X) − mean Dice(P0)] in Dice points; the paired interval averages each case's difference over the seeds
first, then takes the mean and 95 % t-interval over cases.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
MODES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6")
DESCRIPTION = dict(P0="batch BN + Dropout (training)", P1="batch BN, Dropout off", P2="running BN + Dropout",
                   P3="eval, single tile", P4="eval sliding window", P5="P0 × 8, mean z",
                   P6="train-mode sliding window")
COLUMNS = (("dice", "Dice", 100, 2), ("hd95", "HD95", 1, 2), ("n_fp", "FP (k vox)", 1e-3, 1),
           ("n_fn", "FN (k vox)", 1e-3, 1), ("ddice_block", "ΔDice block", 100, 2),
           ("ddice_shift", "ΔDice shift", 100, 2), ("ddice_jitter", "ΔDice jitter", 100, 2),
           ("err_gt_band", "err GT band", 1, 3), ("conf_wrong_095", "conf>0.95 | err", 1, 3),
           ("agree_p0", "Dice vs P0", 100, 2))
ANATOMY_DICE = dict(LA_8l=89.17, Pancreas_6l=72.61)   # summary/anatomy.md, mode train, seed means


def number(value):
    value = float(value)
    return None if math.isnan(value) else value


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def pm(values, scale=1, digits=2):
    values = [v * scale for v in values if v is not None]
    if not values:
        return "n/a"
    text = f"{statistics.mean(values):.{digits}f}"
    return text + (f" ± {statistics.stdev(values):.{digits}f}" if len(values) > 1 else "")


def fmt(value, digits=2, sign=False):
    return "n/a" if value is None else f"{value:{'+' if sign else ''}.{digits}f}"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def load(path):
    """{mode: {case: row}}."""
    data = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            data.setdefault(row["mode"], {})[row["case"]] = {k: (v if k in ("case", "mode") else number(v))
                                                             for k, v in row.items()}
    return data


def deltas(seeds, mode):
    """Per-seed Δ (points), their mean, and the paired per-case mean with its 95 % t-interval."""
    per_seed = {s: 100 * (mean([r["dice"] for r in d[mode].values()]) - mean([r["dice"] for r in d["P0"].values()]))
                for s, d in seeds.items()}
    cases = sorted(set.intersection(*[set(d["P0"]) for d in seeds.values()]))
    paired = [100 * mean([d[mode][c]["dice"] - d["P0"][c]["dice"] for d in seeds.values()]) for c in cases]
    m = statistics.mean(paired)
    half = stats.t.ppf(0.975, len(paired) - 1) * statistics.stdev(paired) / math.sqrt(len(paired))
    return dict(per_seed=per_seed, delta=statistics.mean(per_seed.values()), paired=m, low=m - half, high=m + half)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=PROJECT / "Results_T0")
    parser.add_argument("--anatomy", type=Path, default=PROJECT / "Results_anatomy")
    parser.add_argument("--output", type=Path, default=PROJECT / "summary/T0.md")
    args = parser.parse_args()
    runs = {}
    for path in sorted(args.root.glob("seed_*/*/summary.json")):
        runs.setdefault(path.parent.name, {})[path.parent.parent.name] = load(path.parent / "cases.csv")

    lines = [f"# T0: Teacher inference mode vs pseudo-label quality ({args.root.name})", "",
             "The 6 V0 Teachers (iteration 15000) on their unlabeled cases; per seed the mean over cases, here mean ± "
             "sample std over the seeds. Modes, checks and criteria: README.md, \"T0\".", ""]
    results = {}
    for dataset, seeds in runs.items():
        n_cases = len(next(iter(seeds.values()))["P0"])
        lines += [f"## {dataset} (seeds {', '.join(s.replace('seed_', '') for s in seeds)}; {n_cases} cases each)", ""]
        rows = []
        for mode in MODES:
            rows.append([f"{mode} {DESCRIPTION[mode]}"] + [
                pm([mean([r[key] for r in d[mode].values()]) for d in seeds.values()], scale, digits)
                for key, _, scale, digits in COLUMNS])
        lines += table(["mode", *[header for _, header, _, _ in COLUMNS]], rows) + [""]

        d_all = {mode: deltas(seeds, mode) for mode in MODES[1:]}
        results[dataset] = d_all
        rows = [[mode, fmt(d["delta"], sign=True), " / ".join(fmt(v, sign=True) for v in d["per_seed"].values()),
                 f"{fmt(d['paired'], sign=True)} [{fmt(d['low'], sign=True)}, {fmt(d['high'], sign=True)}]"]
                for mode, d in d_all.items()]
        lines += ["Δ vs P0 (Dice points):", ""]
        lines += table(["mode", "Δ (seed mean)", "per seed (" + ", ".join(s.replace("seed_", "") for s in seeds) + ")",
                        "paired per case, 95 % CI"], rows) + [""]

        p0 = mean([mean([r["dice"] for r in d["P0"].values()]) for d in seeds.values()]) * 100
        check = [f"- P0 Dice {p0:.2f} vs anatomy `train` {ANATOMY_DICE.get(dataset, math.nan):.2f}: "
                 + ("within ±1" if abs(p0 - ANATOMY_DICE.get(dataset, math.inf)) <= 1 else "**outside ±1**")]
        differences = []
        for seed, d in seeds.items():
            anatomy = args.anatomy / seed / dataset / "cases.csv"
            if anatomy.is_file():
                with anatomy.open(newline="") as handle:
                    reference = {row["case"]: float(row["ref_dice"]) for row in csv.DictReader(handle)}
                differences += [abs(d["P4"][c]["dice"] - reference[c]) for c in d["P4"] if c in reference]
        if differences:
            check.append(f"- P4 vs anatomy `sliding` per case: max |ΔDice| {max(differences):.2e} over "
                         f"{len(differences)} case-seed pairs" + (" (identical)" if max(differences) == 0 else ""))
        meta = [json.loads((args.root / s / dataset / "summary.json").read_text()) for s in seeds]
        check.append("- Dropout masks of P0 and P2 identical on the first case: "
                     + ", ".join(str(m.get("dropout_masks_p0_p2_identical_case0")) for m in meta)
                     + f"; T0 commit {meta[0]['t0_commit'][:7]}; {max(m['seconds'] for m in meta) / 60:.0f} min max")
        lines += ["Checks:", ""] + check + [""]

        if dataset.startswith("Pancreas"):
            cases = sorted(seeds[next(iter(seeds))]["P0"])
            change = {c: mean([d["P3"][c]["dice"] - d["P0"][c]["dice"] for d in seeds.values()]) for c in cases}
            top = sorted(cases, key=lambda c: abs(change[c]), reverse=True)[:5]
            rows = [[c, fmt(100 * mean([d["P0"][c]["dice"] for d in seeds.values()])),
                     fmt(100 * mean([d["P3"][c]["dice"] for d in seeds.values()])), fmt(100 * change[c], sign=True),
                     fmt(mean([d["P3"][c]["n_fp"] - d["P0"][c]["n_fp"] for d in seeds.values()]) / 1e3, 1, True),
                     fmt(mean([d["P3"][c]["n_fn"] - d["P0"][c]["n_fn"] for d in seeds.values()]) / 1e3, 1, True),
                     " / ".join(fmt(100 * (d["P3"][c]["dice"] - d["P0"][c]["dice"]), 1, True) for d in seeds.values())]
                    for c in top]
            lines += ["Largest changes P0 → P3 (mean over seeds):", ""]
            lines += table(["case", "Dice P0", "Dice P3", "Δ", "ΔFP (k vox)", "ΔFN (k vox)", "Δ per seed"], rows) + [""]

    lines += ["## Criteria (README \"T0\")", ""]
    pancreas = next((v for k, v in results.items() if k.startswith("Pancreas")), None)
    la = next((v for k, v in results.items() if k.startswith("LA")), None)
    if pancreas:
        d = {mode: pancreas[mode]["delta"] for mode in pancreas}
        r1 = d["P3"] >= 3.0 and all(v > 0 for v in pancreas["P3"]["per_seed"].values())
        important = [m for m, name in (("P1", "Dropout"), ("P2", "BN")) if d[m] >= 2.0]
        alone = [m for m in ("P1", "P2") if d["P3"] > 0 and d[m] >= 0.7 * d["P3"]]
        fix_mode = max(alone, key=lambda m: d[m]) if alone else "P3"
        fix = dict(P1="Dropout off only", P2="running BN only", P3="running BN and Dropout off (P3)")[fix_mode]
        r3 = (d["P5"] >= 0.7 * d["P3"]) if d["P3"] > 0 else None
        r4 = d["P4"] - d["P3"] >= 2.0
        lines += [
            f"- R1 (Pancreas Δ_P3 ≥ 3.0, all seeds > 0): Δ_P3 = {d['P3']:+.2f} (per seed "
            f"{' / '.join(f'{v:+.2f}' for v in pancreas['P3']['per_seed'].values())}) → **{'holds' if r1 else 'does not hold'}**.",
            f"- R2: Δ_P1 (Dropout off) = {d['P1']:+.2f}, Δ_P2 (running BN) = {d['P2']:+.2f}; important (≥ 2.0): "
            f"{', '.join({'P1': 'Dropout', 'P2': 'BN'}[m] for m in important) or 'none'}; ≥ 70 % of Δ_P3 alone: "
            f"{', '.join(alone) or 'none'} → minimal fix: **{fix}**.",
            f"- R3: Δ_P5 = {d['P5']:+.2f} vs 0.7 · Δ_P3 = {0.7 * d['P3']:+.2f} → **"
            + ({True: "mainly random noise", False: "mainly systematic bias", None: "n/a (Δ_P3 ≤ 0)"}[r3]) + "**.",
            f"- R4: Δ_P4 − Δ_P3 = {d['P4'] - d['P3']:+.2f} (≥ 2.0?) → **{'holds' if r4 else 'does not hold'}**; "
            f"Δ_P6 = {d['P6']:+.2f}."]
        if la:
            flagged = [m for m in la if abs(la[m]["delta"]) >= 1]
            lines.append(f"- LA control: " + ", ".join(f"Δ_{m} {la[m]['delta']:+.2f}" for m in la) + "; |Δ| ≥ 1: "
                         + (", ".join(flagged) if flagged else "none")
                         + ("; **Δ_P3 ≤ −1: the fix would hurt LA**" if la["P3"]["delta"] <= -1 else ""))
        if r1:
            conclusion = f"R1 holds → T1: pseudo-labels with the minimal fix ({fix}), seed 42 on both datasets."
        elif r4:
            conclusion = "R1 does not hold, R4 holds → report, discuss (repeated inference is costly)."
        else:
            conclusion = "Neither R1 nor R4 holds → report; the gap has another cause, discuss."
        lines += ["", f"**Conclusion: {conclusion}**", ""]
    text = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
