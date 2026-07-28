"""
Pulls every run from the shared W&B project (baseline, fixed-alpha sweep,
dynamic alpha) and builds the ACC-vs-fairness Pareto frontier comparison
described in fixed.py's docstring: aggregate each fixed alpha to one
mean+/-std point across its 30 seeds, plot those points as a curve, and check
whether the dynamic controller's point sits on/beyond that curve rather than
cherry-picking a single "best" alpha or averaging across alphas.

Run after the fixed-alpha sweep (fixed.py and/or fixed_sweep.py) and
dynamic.py have finished:

    python analyze_results.py

Requires `wandb login` to have been run already (uses the API, not wandb.log).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

WANDB_ENTITY  = "lshearer2957-self"
WANDB_PROJECT = "final_results"

api = wandb.Api()
runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

rows = []
for run in runs:
    if run.state != "finished":
        continue
    summary = run.summary
    if not all(k in summary for k in ("ACC", "DEO", "DAO")):
        continue

    group = run.group or ""
    if group.startswith("FixedAlpha_"):
        kind = "fixed"
        alpha = float(group.replace("FixedAlpha_", ""))
    elif group == "DynamicAlpha":
        kind = "dynamic"
        alpha = None
    elif group == "Baseline":
        kind = "baseline"
        alpha = None
    else:
        continue

    rows.append({
        "kind":  kind,
        "alpha": alpha,
        "seed":  run.config.get("seed"),
        "ACC":   summary["ACC"],
        "DEO":   summary["DEO"],
        "DAO":   summary["DAO"],
    })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"No finished runs with ACC/DEO/DAO found in {WANDB_ENTITY}/{WANDB_PROJECT}")

print(f"Pulled {len(df)} finished runs: "
      f"{(df.kind == 'baseline').sum()} baseline, "
      f"{(df.kind == 'fixed').sum()} fixed-alpha, "
      f"{(df.kind == 'dynamic').sum()} dynamic\n")

# ── Aggregate: one mean+/-std point per alpha (and for baseline/dynamic) ─────
fixed_summary = (
    df[df.kind == "fixed"]
    .groupby("alpha")[["ACC", "DEO", "DAO"]]
    .agg(["mean", "std", "count"])
    .sort_index()
)
baseline_summary = df[df.kind == "baseline"][["ACC", "DEO", "DAO"]].agg(["mean", "std", "count"])
dynamic_summary  = df[df.kind == "dynamic"][["ACC", "DEO", "DAO"]].agg(["mean", "std", "count"])

print("=== Fixed-alpha sweep (mean +/- std per alpha) ===")
print(fixed_summary)
print("\n=== Baseline ===")
print(baseline_summary)
print("\n=== Dynamic Alpha ===")
print(dynamic_summary)

# ── Pareto check: is the dynamic point dominated by any single fixed alpha? ──
# "Dominated" here = some fixed alpha achieves >= ACC AND <= DAO/DEO (i.e.
# strictly as good or better on both axes) than the dynamic controller.
def dominance_check(fixed_summary, dynamic_summary, metric):
    dyn_acc = dynamic_summary.loc["mean", "ACC"]
    dyn_m   = dynamic_summary.loc["mean", metric]
    dominators = []
    for alpha, row in fixed_summary.iterrows():
        acc = row[("ACC", "mean")]
        m   = row[(metric, "mean")]
        if acc >= dyn_acc and m <= dyn_m:
            dominators.append((alpha, acc, m))
    return dyn_acc, dyn_m, dominators

for metric in ("DEO", "DAO"):
    dyn_acc, dyn_m, dominators = dominance_check(fixed_summary, dynamic_summary, metric)
    print(f"\n=== Pareto check on ACC vs {metric} ===")
    print(f"Dynamic: ACC={dyn_acc:.4f}  {metric}={dyn_m:.4f}")
    if dominators:
        print(f"Dominated by {len(dominators)} fixed-alpha point(s) (equal/better ACC AND equal/better {metric}):")
        for alpha, acc, m in dominators:
            print(f"  alpha={alpha:g}: ACC={acc:.4f}  {metric}={m:.4f}")
    else:
        print(f"NOT dominated by any single fixed alpha -- dynamic sits on/beyond the fixed-alpha frontier on this axis.")

# ── Plot: ACC vs DAO and ACC vs DEO Pareto frontiers ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, metric in zip(axes, ("DEO", "DAO")):
    alphas = fixed_summary.index.values
    acc_mean = fixed_summary[("ACC", "mean")].values
    acc_std  = fixed_summary[("ACC", "std")].values
    m_mean   = fixed_summary[(metric, "mean")].values
    m_std    = fixed_summary[(metric, "std")].values

    order = np.argsort(acc_mean)
    ax.errorbar(acc_mean[order], m_mean[order], xerr=acc_std[order], yerr=m_std[order],
                marker="o", linestyle="-", color="tab:blue", label="Fixed alpha (sweep)", capsize=3)
    for a, x, y in zip(alphas[order], acc_mean[order], m_mean[order]):
        ax.annotate(f"α={a:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.errorbar(baseline_summary.loc["mean", "ACC"], baseline_summary.loc["mean", metric],
                xerr=baseline_summary.loc["std", "ACC"], yerr=baseline_summary.loc["std", metric],
                marker="s", color="tab:gray", label="Baseline (no debiasing)", capsize=3)

    ax.errorbar(dynamic_summary.loc["mean", "ACC"], dynamic_summary.loc["mean", metric],
                xerr=dynamic_summary.loc["std", "ACC"], yerr=dynamic_summary.loc["std", metric],
                marker="*", markersize=14, color="tab:red", label="Dynamic alpha", capsize=3)

    ax.set_xlabel("ACC (higher is better)")
    ax.set_ylabel(f"{metric} (lower is better)")
    ax.set_title(f"ACC vs {metric}")
    ax.legend()
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("pareto_frontier.png", dpi=150)
print("\nSaved plot to pareto_frontier.png")
