"""Experiment 3: post-processing equalized-odds calibration (Hardt et al., 2016).

Completely different family from experiments 1-2: no adversary, no
training-time intervention at all. Train a PLAIN predictor (same as
train_baseline.py — full accuracy, no fairness cost during training).
Then, after training, find per-group decision thresholds that make TPR and
FPR exactly equal across groups (equalized odds), chosen to maximize
accuracy — using randomized mixing between two thresholds per group,
which is what makes any point on each group's ROC convex hull achievable.

Calibration is fit on half of the test set and evaluated on the other
half, so the reported numbers aren't optimistic from calibrating and
evaluating on the same data.
"""
import numpy as np
import tensorflow as tf

from src.data import load_compas
from src.models import build_predictor
from src.train_baseline import train_step, EPOCHS, BATCH_SIZE, LEARNING_RATE


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> list:
    thresholds = np.unique(scores)
    thresholds = np.concatenate([[thresholds.min() - 1], thresholds, [thresholds.max() + 1]])
    pts = []
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    for t in thresholds:
        pred = scores >= t
        tpr = (pred & pos).sum() / n_pos if n_pos > 0 else 0.0
        fpr = (pred & neg).sum() / n_neg if n_neg > 0 else 0.0
        pts.append((float(fpr), float(tpr), float(t)))
    return pts


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def upper_concave_hull(pts: list) -> list:
    pts = sorted(set((p[0], p[1], p[2]) for p in pts))
    hull = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def hull_tpr_at(hull: list, target_fpr: float) -> float:
    for i in range(len(hull) - 1):
        (fpr_lo, tpr_lo, _), (fpr_hi, tpr_hi, _) = hull[i], hull[i + 1]
        if fpr_lo <= target_fpr <= fpr_hi:
            if fpr_hi == fpr_lo:
                return max(tpr_lo, tpr_hi)
            frac = (target_fpr - fpr_lo) / (fpr_hi - fpr_lo)
            return tpr_lo + frac * (tpr_hi - tpr_lo)
    return hull[-1][1]


def bracket(hull: list, target_fpr: float):
    for i in range(len(hull) - 1):
        (fpr_lo, tpr_lo, t_lo), (fpr_hi, tpr_hi, t_hi) = hull[i], hull[i + 1]
        if fpr_lo <= target_fpr <= fpr_hi:
            return (fpr_lo, t_lo), (fpr_hi, t_hi)
    return hull[-1][:2] + (hull[-1][2],), hull[-1][:2] + (hull[-1][2],)


def find_best_target(hull0, hull1, prevalence0, prevalence1, frac0, frac1, n_grid=1000):
    best = None
    for fpr in np.linspace(0.0, 1.0, n_grid):
        tpr0 = hull_tpr_at(hull0, fpr)
        tpr1 = hull_tpr_at(hull1, fpr)
        tpr = min(tpr0, tpr1)
        acc = (frac0 * (prevalence0 * tpr + (1 - prevalence0) * (1 - fpr)) +
               frac1 * (prevalence1 * tpr + (1 - prevalence1) * (1 - fpr)))
        if best is None or acc > best[0]:
            best = (acc, fpr, tpr)
    return best


def apply_randomized_rule(scores: np.ndarray, hull: list, target_fpr: float, rng: np.random.Generator) -> np.ndarray:
    (fpr_lo, t_lo), (fpr_hi, t_hi) = bracket(hull, target_fpr)
    if fpr_hi == fpr_lo:
        p_hi = 0.0
    else:
        p_hi = (target_fpr - fpr_lo) / (fpr_hi - fpr_lo)
    use_hi = rng.random(len(scores)) < p_hi
    preds = np.where(use_hi, scores >= t_hi, scores >= t_lo)
    return preds.astype(int)


def run_one(seed: int) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, z_train, z_test = load_compas()
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


def run(seeds):
    results = [run_one(s) for s in seeds]
    print("--- post-processing equalized-odds summary ---")
    for key in ["acc", "deo", "dao"]:
        vals = np.array([r[key] for r in results])
        print(f"  {key}: mean={vals.mean():.4f} std={vals.std():.4f}")
    return results


if __name__ == "__main__":
    import sys
    seeds = list(range(int(sys.argv[1]))) if len(sys.argv) > 1 else list(range(5))
    run(seeds)
