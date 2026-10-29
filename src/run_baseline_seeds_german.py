"""Same as run_baseline_seeds.py, but on German Credit instead of COMPAS.
Reuses train_step/build_predictor unchanged — only load_german() differs.
"""
import numpy as np
import tensorflow as tf

from src.data_german import load_german
from src.metrics import compute_metrics
from src.models import build_predictor
from src.train_baseline import train_step, EPOCHS, BATCH_SIZE, LEARNING_RATE

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

    y_pred_prob = predictor.predict(X_test, verbose=0).reshape(-1)
    metrics = compute_metrics(y_test, y_pred_prob, z_test)
    print(f"seed={seed}: acc={metrics['acc']:.4f} deo={metrics['deo']:.4f} dao={metrics['dao']:.4f}")
    return metrics


def run():
    results = [run_one(s) for s in SEEDS]
    print("--- baseline (german) summary ---")
    for key in ["acc", "deo", "dao"]:
        vals = np.array([r[key] for r in results])
        print(f"  {key}: mean={vals.mean():.4f} std={vals.std():.4f}")


if __name__ == "__main__":
    run()
