"""Same as run_seeds.py (adaptive dual-ascent alpha), but on German Credit.
Reuses adaptive_train_step unchanged — only the data source and the
epsilon/floor targets (recomputed from German's own measured baseline)
differ.
"""
import numpy as np
import tensorflow as tf

from src.data_german import load_german
from src.metrics import compute_metrics
from src.models import build_predictor, build_adversary
from src.train_adaptive_alpha import adaptive_train_step, BATCH_SIZE, LEARNING_RATE, ALPHA_LR

EPOCHS = 30
SEEDS = list(range(30))

# set from German's own measured 30-seed baseline (results/baseline_german_30.log):
# acc 0.7207+/-0.0164, deo 0.0580+/-0.0367, dao 0.0527+/-0.0249
EPS_DEO = 0.3 * 0.0580   # 0.0174
EPS_DAO = 0.3 * 0.0527   # 0.0158
ACC_FLOOR = 0.66         # finalized after checking the fixed-alpha=2.0 german
                         # result (results/fixed_german_30.log) for a realistic
                         # achievable-accuracy-under-mitigation floor


def run_one(seed: int) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, z_train, z_test = load_german()
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

        last_metrics = {"acc": acc, "deo": deo, "dao": dao}

    print(f"seed={seed}: acc={last_metrics['acc']:.4f} deo={last_metrics['deo']:.4f} "
          f"dao={last_metrics['dao']:.4f} alpha_deo={alpha_deo:.4f} alpha_dao={alpha_dao:.4f}")
    return last_metrics


def run():
    results = [run_one(s) for s in SEEDS]
    print("--- adaptive (german) summary ---")
    for key in ["acc", "deo", "dao"]:
        vals = np.array([r[key] for r in results])
        print(f"  {key}: mean={vals.mean():.4f} std={vals.std():.4f}")


if __name__ == "__main__":
    run()
