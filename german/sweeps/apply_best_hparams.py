"""
Pulls the best trial from the controller hyperparameter search (see
dynamic_hparam_sweep.py / sweep_dynamic_hparams.yaml) and patches its
ALPHA_INIT / ALPHA_LR / EMA_ALPHA / ACC_FLOOR / W_DEO / W_DAO / W_ACC values
into dynamic.py (parent german/ folder) and dynamic_sweep.py (this folder),
replacing the Adult-tuned constants currently there.

"Best" = lowest logged `objective` (DEO + DAO + accuracy-shortfall penalty,
averaged over the trial's 3 seeds) among finished trials -- consistent with
sweep_dynamic_hparams.yaml's metric.goal: minimize.

ALPHA_MIN / ALPHA_MAX are left untouched -- they were fixed search bounds in
dynamic_hparam_sweep.py, not swept values, so there's nothing to copy back
for them.

Usage:
    python german/sweeps/apply_best_hparams.py <SWEEP_ID>

<SWEEP_ID> is the entity/project/sweep_id string printed by
`wandb sweep german/sweeps/sweep_dynamic_hparams.yaml` (also visible in the W&B UI
sweep page's URL / overview tab).

Review `git diff german/` after running this, before committing -- it's a
plain text patch, not a full validation that the new values behave well.
Re-run the full 30-seed dynamic sweep (sweep_dynamic.yaml) with the patched
constants to get real paper numbers.
"""

import re
import sys

import wandb

CONFIG_KEYS = {
    "ALPHA_INIT": "alpha_init",
    "ALPHA_LR":   "alpha_lr",
    "EMA_ALPHA":  "ema_alpha",
    "ACC_FLOOR":  "acc_floor",
    "W_DEO":      "w_deo",
    "W_DAO":      "w_dao",
    "W_ACC":      "w_acc",
}
TARGET_FILES = ["german/dynamic.py", "german/sweeps/dynamic_sweep.py"]


def best_trial(sweep_id):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    finished = [r for r in sweep.runs if r.state == "finished" and "objective" in r.summary]
    if not finished:
        raise SystemExit(f"No finished runs with a logged 'objective' found in sweep {sweep_id}")
    return min(finished, key=lambda r: r.summary["objective"])


def patch_file(path, values):
    with open(path) as f:
        text = f.read()

    for const, value in values.items():
        pattern = re.compile(rf"^({const}\s*=\s*)[^\n#]+", re.MULTILINE)
        if not pattern.search(text):
            raise SystemExit(f"Could not find constant {const} in {path} -- aborting without writing.")
        text = pattern.sub(lambda m, v=value: f"{m.group(1)}{v!r}", text)

    with open(path, "w") as f:
        f.write(text)
    print(f"Patched {path}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python german/sweeps/apply_best_hparams.py <SWEEP_ID>")
    sweep_id = sys.argv[1]

    run = best_trial(sweep_id)
    values = {const: run.config[key] for const, key in CONFIG_KEYS.items()}

    print(f"Best trial: {run.id} ({run.name})")
    summary = run.summary
    print(f"  objective={summary.get('objective'):.4f}  "
          f"ACC={summary.get('ACC'):.4f}  DEO={summary.get('DEO'):.4f}  DAO={summary.get('DAO'):.4f}")
    for const, value in values.items():
        print(f"  {const} = {value}")

    for path in TARGET_FILES:
        patch_file(path, values)

    print("\nDone. Review `git diff german/` before committing, then re-run "
          "the full 30-seed dynamic sweep (sweep_dynamic.yaml) with these values.")


if __name__ == "__main__":
    main()
