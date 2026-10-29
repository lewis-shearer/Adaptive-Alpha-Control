"""Run train_adaptive_alpha's mechanism across multiple seeds, report mean+/-std.

Reuses adaptive_train_step/build_predictor/build_adversary from
train_adaptive_alpha.py so there's no duplicated training logic — only the
seed and the lack of per-epoch printing differ.
"""
import numpy as np
import tensorflow as tf

from src.data import load_compas
from src.metrics import compute_metrics
from src.models import build_predictor, build_adversary
from src.train_adaptive_alpha import (
    adaptive_train_step, EPOCHS, BATCH_SIZE, LEARNING_RATE, ALPHA_LR,
    EPS_DEO, EPS_DAO, ACC_FLOOR,
)

SEEDS = list(range(30))


def run_one_seed(seed: int) -> dict:
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

        pressure_deo = deo - EPS_DEO
        pressure_dao = dao - EPS_DAO
        if acc < ACC_FLOOR:
            pressure_deo = min(pressure_deo, 0.0)
            pressure_dao = min(pressure_dao, 0.0)
        alpha_deo = max(0.0, alpha_deo + ALPHA_LR * pressure_deo)
        alpha_dao = max(0.0, alpha_dao + ALPHA_LR * pressure_dao)

        last_metrics = {"acc": acc, "deo": deo, "dao": dao,
                         "alpha_deo": alpha_deo, "alpha_dao": alpha_dao}

    print(f"seed {seed}: acc={last_metrics['acc']:.4f} deo={last_metrics['deo']:.4f} "
          f"dao={last_metrics['dao']:.4f} alpha_deo={last_metrics['alpha_deo']:.4f} "
          f"alpha_dao={last_metrics['alpha_dao']:.4f}")
    return last_metrics


def run():
    results = [run_one_seed(s) for s in SEEDS]
    for key in ["acc", "deo", "dao", "alpha_deo", "alpha_dao"]:
        vals = np.array([r[key] for r in results])
        print(f"{key}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"min={vals.min():.4f} max={vals.max():.4f}")


if __name__ == "__main__":
    run()
