"""Pulls full per-epoch history (not just final-epoch values) for every run
in the `final_results` W&B project, and saves it to adult/results/wandb_full_history.csv.

Used to compute real training-time stability metrics (cross-seed spread and
epoch-to-epoch volatility) instead of eyeballing chart band widths.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import wandb

ENTITY_PROJECT = "lshearer2957-self/final_results"
OUT_PATH = "adult/results/wandb_full_history.csv"


def classify(name):
    if "baseline" in name:
        return "Baseline"
    if "dynamic" in name:
        return "Dynamic"
    m = re.search(r"fixed_alpha([0-9.]+)_", name)
    if m:
        return f"Fixed_{m.group(1)}"
    return "Other"


def fetch(run):
    h = run.history(keys=["epoch", "ACC", "DEO", "DAO"], pandas=True)
    h["run_id"] = run.id
    h["name"] = run.name
    h["group"] = classify(run.name)
    return h


def main():
    api = wandb.Api()
    runs = list(api.runs(ENTITY_PROJECT, per_page=300))
    print(f"total runs: {len(runs)}")

    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(fetch, r): r for r in runs}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 50 == 0:
                print(f"{i}/{len(runs)} done")

    df = pd.concat(results, ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"saved {df.shape} to {OUT_PATH}")


if __name__ == "__main__":
    main()
