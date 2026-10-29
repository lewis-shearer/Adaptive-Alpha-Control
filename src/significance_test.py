"""Paired Wilcoxon signed-rank tests across seeds, Holm-corrected across
all comparisons made. Seeds are paired (same seed = same train/test split
and initialization draw across conditions), so a paired test is the
correct one — not an unpaired t-test.

Usage: edit COMPARISONS below to list which (condition_a, condition_b,
metric) triples to test, then run `python3 -m src.significance_test`.
"""
import re
import numpy as np
from scipy.stats import wilcoxon

SEED_LINE = re.compile(r"seed[= ](\d+): acc=([\d.]+) deo=([\d.]+) dao=([\d.]+)")
FIXED_LINE = re.compile(r"alpha=([\d.]+) seed=(\d+): acc=([\d.]+) deo=([\d.]+) dao=([\d.]+)")


def parse_simple(path: str) -> dict:
    by_seed = {}
    with open(path) as f:
        for line in f:
            m = SEED_LINE.search(line)
            if m:
                seed = int(m.group(1))
                by_seed[seed] = {"acc": float(m.group(2)), "deo": float(m.group(3)), "dao": float(m.group(4))}
    return by_seed


def parse_fixed(path: str, alpha: float) -> dict:
    by_seed = {}
    with open(path) as f:
        for line in f:
            m = FIXED_LINE.match(line)
            if m and float(m.group(1)) == alpha:
                seed = int(m.group(2))
                by_seed[seed] = {"acc": float(m.group(3)), "deo": float(m.group(4)), "dao": float(m.group(5))}
    return by_seed


def paired_wilcoxon(by_seed_a: dict, by_seed_b: dict, metric: str):
    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
    a = np.array([by_seed_a[s][metric] for s in common_seeds])
    b = np.array([by_seed_b[s][metric] for s in common_seeds])
    if len(a) < 2:
        return None
    stat, p = wilcoxon(a, b)
    return {"n": len(a), "mean_a": a.mean(), "mean_b": b.mean(), "stat": stat, "p": p}


def holm_correct(p_values: list) -> list:
    order = np.argsort(p_values)
    n = len(p_values)
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def run(comparisons: list):
    """comparisons: list of (label_a, by_seed_a, label_b, by_seed_b, metric)"""
    results = []
    for label_a, data_a, label_b, data_b, metric in comparisons:
        r = paired_wilcoxon(data_a, data_b, metric)
        if r is None:
            continue
        results.append((label_a, label_b, metric, r))

    p_values = [r[3]["p"] for r in results]
    adjusted = holm_correct(p_values)

    print(f"{'comparison':<45} {'metric':<5} {'n':>3} {'mean_a':>8} {'mean_b':>8} {'p':>10} {'p_holm':>10}")
    for (label_a, label_b, metric, r), p_adj in zip(results, adjusted):
        sig = "*" if p_adj < 0.05 else " "
        print(f"{label_a + ' vs ' + label_b:<45} {metric:<5} {r['n']:>3} "
              f"{r['mean_a']:>8.4f} {r['mean_b']:>8.4f} {r['p']:>10.4g} {p_adj:>10.4g} {sig}")


if __name__ == "__main__":
    baseline = parse_simple("results/baseline_30.log")
    fixed_2 = parse_fixed("results/fixed_30.log", 2.0)
    adaptive = parse_simple("results/adaptive_30.log")
    postproc = parse_simple("results/postproc_30.log")

    comparisons = []
    for metric in ["acc", "deo", "dao"]:
        comparisons.append(("Adaptive", adaptive, "Fixed a=2.0", fixed_2, metric))
        comparisons.append(("Post-processing", postproc, "Adaptive", adaptive, metric))
        comparisons.append(("Post-processing", postproc, "Baseline", baseline, metric))
        comparisons.append(("Adaptive", adaptive, "Baseline", baseline, metric))

    run(comparisons)
