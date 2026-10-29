"""Parse the three 30-seed run logs and plot an accuracy-vs-bias Pareto frontier.

Reads the raw per-seed result lines straight out of each log file (no need
to re-run anything), computes mean+/-std per condition, and plots two
panels: accuracy vs DEO, accuracy vs DAO.
"""
import re
import numpy as np
import matplotlib.pyplot as plt

BASELINE_LOG = "results/baseline_30.log"
FIXED_LOG = "results/fixed_30.log"
ADAPTIVE_LOG = "results/adaptive_30.log"
POSTPROC_LOG = "results/postproc_30.log"

SEED_LINE = re.compile(r"acc=([\d.]+) deo=([\d.]+) dao=([\d.]+)")
FIXED_LINE = re.compile(r"alpha=([\d.]+) seed=\d+: acc=([\d.]+) deo=([\d.]+) dao=([\d.]+)")


def parse_simple(path: str) -> dict:
    accs, deos, daos = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("seed"):
                m = SEED_LINE.search(line)
                if m:
                    accs.append(float(m.group(1)))
                    deos.append(float(m.group(2)))
                    daos.append(float(m.group(3)))
    return {"acc": np.array(accs), "deo": np.array(deos), "dao": np.array(daos)}


def parse_fixed(path: str) -> dict:
    by_alpha = {}
    with open(path) as f:
        for line in f:
            m = FIXED_LINE.match(line)
            if m:
                alpha = float(m.group(1))
                by_alpha.setdefault(alpha, {"acc": [], "deo": [], "dao": []})
                by_alpha[alpha]["acc"].append(float(m.group(2)))
                by_alpha[alpha]["deo"].append(float(m.group(3)))
                by_alpha[alpha]["dao"].append(float(m.group(4)))
    return {a: {k: np.array(v) for k, v in d.items()} for a, d in by_alpha.items()}


def mean_std(arr):
    return arr.mean(), arr.std()


def main():
    baseline = parse_simple(BASELINE_LOG)
    adaptive = parse_simple(ADAPTIVE_LOG)
    fixed = parse_fixed(FIXED_LOG)
    postproc = parse_simple(POSTPROC_LOG)

    points = []  # (label, acc_mean, acc_std, deo_mean, deo_std, dao_mean, dao_std)
    for label, data in [("Baseline", baseline)] + \
                        [(f"Fixed a={a}", d) for a, d in sorted(fixed.items())] + \
                        [("Adaptive", adaptive)] + \
                        [("Post-processing", postproc)]:
        if len(data["acc"]) == 0:
            continue
        am, asd = mean_std(data["acc"])
        dm, dsd = mean_std(data["deo"])
        dam, dasd = mean_std(data["dao"])
        points.append((label, am, asd, dm, dsd, dam, dasd))

    print(f"{'condition':<14} {'acc':>14} {'deo':>14} {'dao':>14}")
    for label, am, asd, dm, dsd, dam, dasd in points:
        print(f"{label:<14} {am:.4f}+/-{asd:.4f}  {dm:.4f}+/-{dsd:.4f}  {dam:.4f}+/-{dasd:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(points)))

    for ax, metric_idx, metric_name in [(axes[0], (3, 4), "DEO"), (axes[1], (5, 6), "DAO")]:
        for (label, am, asd, dm, dsd, dam, dasd), color in zip(points, colors):
            mm, msd = (dm, dsd) if metric_name == "DEO" else (dam, dasd)
            marker = "D" if label == "Post-processing" else ("*" if label == "Adaptive" else ("s" if label == "Baseline" else "o"))
            size = 250 if label == "Post-processing" else (200 if label == "Adaptive" else 100)
            ax.errorbar(am, mm, xerr=asd, yerr=msd, fmt=marker, color=color,
                        markersize=size ** 0.5, label=label, capsize=3)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Accuracy vs {metric_name} (30 seeds, mean +/- std)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "paper/pareto_frontier.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
