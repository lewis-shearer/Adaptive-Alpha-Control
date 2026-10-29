"""Same as src/experiments/postprocess_equalized_odds.py, but on German
Credit instead of COMPAS. Reuses every helper function unchanged — only
load_german() differs, and the moved-out import path.
"""
import numpy as np
import tensorflow as tf

from src.data_german import load_german
from src.models import build_predictor
from src.train_baseline import train_step, EPOCHS, BATCH_SIZE, LEARNING_RATE
from src.experiments.postprocess_equalized_odds import (
    roc_points, upper_concave_hull, find_best_target, apply_randomized_rule,
)

SEEDS = list(range(30))


def run_one(seed: int) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, z_train, z_test = load_german()
    n_train = X_train.shape[0]

    predictor = build_predictor(input_dim=X_train.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled = X_train[perm], y_train[perm]
        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = tf.convert_to_tensor(X_shuffled[start:end])
            y_batch = tf.convert_to_tensor(y_shuffled[start:end])
            train_step(predictor, optimizer, x_batch, y_batch)

    scores = predictor.predict(X_test, verbose=0).reshape(-1)

    rng = np.random.default_rng(seed)
    n = len(scores)
    idx = rng.permutation(n)
    calib_idx, eval_idx = idx[: n // 2], idx[n // 2:]

    s_calib, y_calib, z_calib = scores[calib_idx], y_test[calib_idx].astype(int), z_test[calib_idx].astype(int)
    s_eval, y_eval, z_eval = scores[eval_idx], y_test[eval_idx].astype(int), z_test[eval_idx].astype(int)

    pts0 = roc_points(y_calib[z_calib == 0], s_calib[z_calib == 0])
    pts1 = roc_points(y_calib[z_calib == 1], s_calib[z_calib == 1])
    hull0 = upper_concave_hull(pts0)
    hull1 = upper_concave_hull(pts1)

    prevalence0 = y_calib[z_calib == 0].mean()
    prevalence1 = y_calib[z_calib == 1].mean()
    frac0 = (z_calib == 0).mean()
    frac1 = (z_calib == 1).mean()

    _, target_fpr, target_tpr = find_best_target(hull0, hull1, prevalence0, prevalence1, frac0, frac1)

    preds = np.zeros(len(s_eval), dtype=int)
    mask0 = z_eval == 0
    mask1 = z_eval == 1
    preds[mask0] = apply_randomized_rule(s_eval[mask0], hull0, target_fpr, rng)
    preds[mask1] = apply_randomized_rule(s_eval[mask1], hull1, target_fpr, rng)

    acc = (preds == y_eval).mean()

    def tpr_fpr(yt, yp):
        pos, neg = yt == 1, yt == 0
        tpr = yp[pos].mean() if pos.any() else 0.0
        fpr = yp[neg].mean() if neg.any() else 0.0
        return tpr, fpr

    tpr0, fpr0 = tpr_fpr(y_eval[mask0], preds[mask0])
    tpr1, fpr1 = tpr_fpr(y_eval[mask1], preds[mask1])
    deo = abs(tpr0 - tpr1)
    dao = (abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0

    print(f"seed={seed}: acc={acc:.4f} deo={deo:.4f} dao={dao:.4f} "
          f"(target_fpr={target_fpr:.3f} target_tpr={target_tpr:.3f})")
    return {"acc": acc, "deo": deo, "dao": dao}


def run():
    results = [run_one(s) for s in SEEDS]
    print("--- post-processing equalized-odds (german) summary ---")
    for key in ["acc", "deo", "dao"]:
        vals = np.array([r[key] for r in results])
        print(f"  {key}: mean={vals.mean():.4f} std={vals.std():.4f}")


if __name__ == "__main__":
    run()
