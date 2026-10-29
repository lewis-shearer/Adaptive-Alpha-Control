"""Experiment 1: annealed epsilon for the adaptive-alpha dual ascent.

Same mechanism as src/train_adaptive_alpha.py (two label-conditioned
adversaries, dual ascent on alpha_deo/alpha_dao) — the only change is that
EPS_DEO/EPS_DAO are no longer fixed targets from epoch 1. They start equal
to the measured baseline gap (near-zero pressure at the start) and anneal
linearly down to the original fixed targets by the final epoch.

Reuses adaptive_train_step/build_predictor/build_adversary — no
duplicated training logic, only the epsilon schedule differs.
"""
import numpy as np
import tensorflow as tf

from src.data import load_compas
from src.metrics import compute_metrics
from src.models import build_predictor, build_adversary
from src.train_adaptive_alpha import adaptive_train_step, BATCH_SIZE, LEARNING_RATE, ALPHA_LR, ACC_FLOOR

EPOCHS = 30

# measured 30-seed baseline (see EXPERIMENTS.md) — annealing starts here
BASELINE_DEO = 0.2503
BASELINE_DAO = 0.2093

# original fixed targets from train_adaptive_alpha.py — annealing ends here
FINAL_EPS_DEO = 0.077
FINAL_EPS_DAO = 0.064


def eps_at_epoch(epoch: int, total_epochs: int, start: float, end: float) -> float:
    frac = (epoch - 1) / max(total_epochs - 1, 1)
    return start + (end - start) * frac


def run_one(seed: int, verbose: bool = True) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, z_train, z_test = load_compas()
    n_train = X_train.shape[0]

    predictor = build_predictor(input_dim=X_train.shape[1])
    adv_pos = build_adversary()
    adv_neg = build_adversary()
    p_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    a_pos_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    a_neg_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    alpha_deo = 0.0
    alpha_dao = 0.0
    last_metrics = None

    for epoch in range(1, EPOCHS + 1):
        eps_deo = eps_at_epoch(epoch, EPOCHS, BASELINE_DEO, FINAL_EPS_DEO)
        eps_dao = eps_at_epoch(epoch, EPOCHS, BASELINE_DAO, FINAL_EPS_DAO)

        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled, z_shuffled = X_train[perm], y_train[perm], z_train[perm]

        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = tf.convert_to_tensor(X_shuffled[start:end])
            y_batch = tf.convert_to_tensor(y_shuffled[start:end])
            z_batch = tf.convert_to_tensor(z_shuffled[start:end])
            adaptive_train_step(
                predictor, adv_pos, adv_neg, p_optimizer, a_pos_optimizer,
                a_neg_optimizer, x_batch, y_batch, z_batch, alpha_deo, alpha_dao,
            )

        y_pred_prob = predictor.predict(X_test, verbose=0).reshape(-1)
        metrics = compute_metrics(y_test, y_pred_prob, z_test)
        acc, deo, dao = metrics["acc"], metrics["deo"], metrics["dao"]

        pressure_deo = deo - eps_deo
        pressure_dao = dao - eps_dao
        if acc < ACC_FLOOR:
            pressure_deo = min(pressure_deo, 0.0)
            pressure_dao = min(pressure_dao, 0.0)
        alpha_deo = max(0.0, alpha_deo + ALPHA_LR * pressure_deo)
        alpha_dao = max(0.0, alpha_dao + ALPHA_LR * pressure_dao)

        last_metrics = {"acc": acc, "deo": deo, "dao": dao}

        if verbose:
            print(f"  epoch {epoch:02d} eps_deo={eps_deo:.4f} eps_dao={eps_dao:.4f} "
                  f"acc={acc:.4f} deo={deo:.4f} dao={dao:.4f} "
                  f"alpha_deo={alpha_deo:.4f} alpha_dao={alpha_dao:.4f}")

    print(f"seed={seed}: acc={last_metrics['acc']:.4f} deo={last_metrics['deo']:.4f} "
          f"dao={last_metrics['dao']:.4f}")
    return last_metrics


def run(seeds):
    results = [run_one(s, verbose=False) for s in seeds]
    print("--- annealed-epsilon summary ---")
    for key in ["acc", "deo", "dao"]:
        vals = np.array([r[key] for r in results])
        print(f"  {key}: mean={vals.mean():.4f} std={vals.std():.4f}")
    return results


if __name__ == "__main__":
    import sys
    seeds = list(range(int(sys.argv[1]))) if len(sys.argv) > 1 else list(range(5))
    run(seeds)
