"""V1 probe report: medians of v1_probe.csv over 6000 <= iteration <= 15000 and the criteria of README.md.

  python scripts/report_v1.py --root Results_V1 --seed 42 --output summary/V1.md
Every quantity is the median over the probe steps in the window (NaN, i.e. an empty subset on that step, left
out). Rows repeated after a resume are de-duplicated by (iteration, row key), the last one wins. A criterion
whose inputs are missing counts as not holding and is marked "n/a".
"""
import argparse
import csv
import math
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parents[1]
RUNS = ("LA_8l", "Pancreas_6l")
CONSTRAINTS = ("signed", "unsigned", "band", "none")
TAUS = ("0.05", "0.1", "0.2")
KEY = ("family", "constraint", "G", "tau", "subset", "metric")
NA = "NA"


def read_probe(path, start, end):
    """{row key: [values over the probe steps in the window]} and the sorted window iterations."""
    last = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            iteration = int(row["iteration"])
            if start <= iteration <= end:
                last[(iteration, tuple(row[c] for c in KEY))] = float(row["value"])
    series = {}
    for (_, key), value in sorted(last.items()):
        series.setdefault(key, []).append(value)
    return series, sorted({iteration for iteration, _ in last})


class Medians:
    def __init__(self, series):
        self.series = series

    def __call__(self, family, metric, constraint=NA, groups=NA, tau=NA, subset=NA):
        values = [v for v in self.series.get((family, constraint, str(groups), tau, subset, metric), [])
                  if not math.isnan(v)]
        return statistics.median(values) if values else None


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def check(condition, *values):
    """True / False, or None when an input is missing."""
    return None if any(v is None for v in values) else bool(condition())


def mark(result):
    return {True: "yes", False: "**no**", None: "n/a"}[result]


def forward(m, constraint):
    by_tau = {tau: m("attention", "contamination", constraint, 4, tau, "all") for tau in TAUS}
    known = {tau: value for tau, value in by_tau.items() if value is not None}
    tau = min(known, key=known.get) if known else None
    best = known.get(tau)
    uniform = m("attention_uniform", "contamination", constraint, subset="all")
    unmasked = m("attention", "contamination", "none", 4, tau, "all") if tau else None
    swap = m("attention", "swap", constraint, 4, tau, "all") if tau else None
    checks = [(f"contamination(τ*) ≤ uniform ({fmt(best)} vs {fmt(uniform)})",
               check(lambda: best <= uniform, best, uniform)),
              (f"contamination(τ*) ≤ none(G=4, τ*) − 0.05 ({fmt(best)} vs {fmt(unmasked)} − 0.05)",
               check(lambda: best <= unmasked - 0.05, best, unmasked)),
              (f"swap(τ*) ≥ 0.1 ({fmt(swap)})", check(lambda: swap >= 0.1, swap))]
    return tau, checks


def reverse(m):
    grid = dict(constraint="unsigned", groups=4, tau="0.1")
    correction = m("reverse_label", "correction_mass", subset="all", **grid)
    harm = m("reverse_label", "harm_mass", subset="all", **grid)
    rate = m("reverse_label", "correction_rate", subset="conf095", **grid)
    cos_ul = m("reverse_grad", "cos_ul_true", subset="wrong", **grid)
    cos_pl = m("reverse_grad", "cos_pl_true", subset="wrong", **grid)
    return [(f"correction_mass > harm_mass, all ({fmt(correction, 4)} vs {fmt(harm, 4)})",
             check(lambda: correction > harm, correction, harm)),
            (f"correction_rate, conf095 ≥ 0.5 ({fmt(rate)})", check(lambda: rate >= 0.5, rate)),
            (f"cos_ul_true, wrong > 0 ({fmt(cos_ul)})", check(lambda: cos_ul > 0, cos_ul)),
            (f"cos_ul_true > cos_pl_true, wrong ({fmt(cos_ul)} vs {fmt(cos_pl)})",
             check(lambda: cos_ul > cos_pl, cos_ul, cos_pl))]


def holds(checks):
    return all(result is True for _, result in checks)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def run_section(name, m, iterations, series):
    nan = sum(math.isnan(v) for values in series.values() for v in values)
    total = sum(len(values) for values in series.values())
    lines = [f"## {name}", "",
             f"Probe steps in the window: {len(iterations)}"
             + (f" ({iterations[0]}–{iterations[-1]})" if iterations else "") + f"; NaN values left out: {nan} of {total}.",
             "", "### Attention (subset all; contamination / swap / eff_keys)", ""]
    rows = []
    for constraint in CONSTRAINTS:
        for groups in (1, 4):
            rows.append([constraint, groups] + [" / ".join(fmt(m("attention", metric, constraint, groups, tau, "all"))
                                                           for metric in ("contamination", "swap", "eff_keys"))
                                                for tau in TAUS])
        rows.append([constraint, "uniform", " / ".join(fmt(m("attention_uniform", metric, constraint, subset="all"))
                                                       for metric in ("contamination", "swap", "eff_keys")), "", ""])
    lines += table(["mask", "G", "τ = 0.05", "τ = 0.1", "τ = 0.2"], rows)
    lines += ["", "### Contamination by query |d| (G = 4, τ = 0.1)", ""]
    lines += table(["mask", "d1", "d2", "d3"],
                   [[c] + [fmt(m("attention", "contamination", c, 4, "0.1", f"d{k}")) for k in (1, 2, 3)]
                    for c in CONSTRAINTS])
    lines += ["", "### Label-level reverse correction (G = 4, τ = 0.1; uniform rows: every allowed key equally)", ""]
    rows = []
    for family in ("reverse_label", "reverse_label_uniform"):
        for constraint in CONSTRAINTS:
            grid = dict(groups=4, tau="0.1") if family == "reverse_label" else {}
            for subset in ("all", "conf095"):
                rows.append(["uniform" if family.endswith("uniform") else "attention", constraint, subset] +
                            [fmt(m(family, metric, constraint, subset=subset, **grid), 4)
                             for metric in ("correction_rate", "harm_rate", "correction_mass", "harm_mass",
                                            "wrong_mass_frac")])
    lines += table(["weights", "mask", "subset", "correction_rate", "harm_rate", "correction_mass", "harm_mass",
                    "wrong_mass_frac"], rows)
    lines += ["", "### Gradient-level reverse correction (G = 4, τ = 0.1)", ""]
    lines += table(["mask", "subset", "cos_ul_true", "cos_pl_true", "n_tokens"],
                   [[c, s] + [fmt(m("reverse_grad", metric, c, 4, "0.1", s)) for metric in ("cos_ul_true", "cos_pl_true")]
                    + [fmt(m("reverse_grad", "n_tokens", c, 4, "0.1", s), 0)]
                    for c in ("signed", "unsigned") for s in ("wrong", "wrong_conf095", "right")])
    lines += ["", "### Output sensitivity (G = 4, τ = 0.1)", ""]
    metrics = ("dp_final_band", "dce_final_band", "flip_final_band", "dp_d2", "bypass_ratio", "dp_final_replaced",
               "bypass_ratio_replaced", "replaced_fraction")
    lines += table(["mask", *metrics], [[c] + [fmt(m("sensitivity", metric, c, 4, "0.1"), 4) for metric in metrics]
                                        for c in ("signed", "unsigned")])
    lines += ["", "### Bucket profiles", ""]
    buckets = ("d-3", "d-2", "d-1", "d+1", "d+2", "d+3")
    lines += table(["metric", *buckets, "bucket_mean", "global"],
                   [[metric] + [fmt(m("profile", metric, subset=b)) for b in (*buckets, "bucket_mean", "global")]
                    for metric in ("cos_unl_pseudo_vs_true", "cos_lab_vs_unl_true", "reldiff_lab_vs_unl_true",
                                   "reldiff_lab_vs_unl")])
    lines += ["", "### Samples and time", ""]
    lines += table(["", *buckets, "outside"],
                   [[metric] + [fmt(m("sample", metric, subset=b), 0) for b in (*buckets, "outside")]
                    for metric in ("n_query", "n_key")])
    lines += ["", f"Probe time per step (median): {fmt(m('meta', 'probe_seconds'), 2)} s", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=PROJECT / "Results_V1")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--start", type=int, default=6000)
    parser.add_argument("--end", type=int, default=15000)
    parser.add_argument("--output", type=Path, default=PROJECT / "summary/V1.md")
    args = parser.parse_args()

    data = {}
    for name in args.runs:
        path = args.root / f"seed_{args.seed}" / name / "v1_probe.csv"
        series, iterations = read_probe(path, args.start, args.end) if path.is_file() else ({}, [])
        data[name] = (Medians(series), iterations, series)

    lines = [f"# V1 probe: {args.root.name}, seed {args.seed}", "",
             f"Medians over the probe steps with {args.start} ≤ iteration ≤ {args.end} (criteria: README.md, "
             f"\"V1 probe\"). τ* = the τ with the lowest median contamination (G = 4, subset all).", ""]
    if args.start != 6000 or args.end != 15000:
        lines += [f"**Window {args.start}–{args.end} differs from the criteria's 6000–15000: not a decision.**", ""]
    results, rows = {}, []
    for name, (m, iterations, _) in data.items():
        forward_checks = {c: forward(m, c) for c in ("signed", "unsigned")}
        reverse_checks = reverse(m)
        results[name] = dict(forward=all(holds(checks) for _, checks in forward_checks.values()),
                             reverse=holds(reverse_checks))
        for constraint, (tau, checks) in forward_checks.items():
            rows += [[name, f"forward, {constraint}, τ* = {tau or 'n/a'}", text, mark(result)] for text, result in checks]
        rows += [[name, "reverse, unsigned, G = 4, τ = 0.1", text, mark(result)] for text, result in reverse_checks]
        for constraint in ("signed", "unsigned"):
            bypass = m("sensitivity", "bypass_ratio", constraint, 4, "0.1")
            flag = check(lambda: bypass < 0.2, bypass)
            rows.append([name, f"sensitivity flag, {constraint}", f"bypass_ratio < 0.2 ({fmt(bypass)})",
                         {True: "**flag: x1 skip bypass is strong**", False: "no flag", None: "n/a"}[flag]])
    forward_all = all(r["forward"] for r in results.values()) and bool(results)
    reverse_all = all(r["reverse"] for r in results.values()) and bool(results)
    if forward_all and reverse_all:
        decision = "forward and reverse hold → unsigned buckets, two-way, go to V2"
    elif forward_all:
        decision = "forward holds, reverse does not → signed buckets, one-way U→L, go to V2"
    else:
        decision = "forward does not hold → stop and discuss (fallback: bucket profiles only, ablation level 2)"
    lines += ["## Criteria", ""] + table(["run", "criterion", "test (medians)", "holds"], rows) + [""]
    lines += table(["run", "forward", "reverse"],
                   [[name, mark(r["forward"]), mark(r["reverse"])] for name, r in results.items()]) + [""]
    lines += [f"**Decision: {decision}.**", ""]
    for name, (m, iterations, series) in data.items():
        lines += run_section(name, m, iterations, series)
    text = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
