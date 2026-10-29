"""Same as run_fixed_seeds.py, but on German Credit instead of COMPAS.
Reuses adversarial_train_step unchanged — only load_german() differs.
"""
import numpy as np
import tensorflow as tf

from src.data_german import load_german
from src.metrics import compute_metrics
from src.models import build_predictor, build_adversary
from src.train_fixed_alpha import adversarial_train_step, EPOCHS, BATCH_SIZE, LEARNING_RATE

ALPHAS = [0.5, 1.0, 2.0]
SEEDS = list(range(30))


def run_one(alpha: float, seed: int) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, z_train, z_test = load_german()
    n_train = X_train.shape[0]

    predictor = build_predictor(input_dim=X_train.shape[1])
    adversary = build_adversary()
    p_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    a_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled, z_shuffled = X_train[perm], y_train[perm], z_train[perm]
        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = tf.convert_to_tensor(X_shuffled[start:end])
            y_batch = tf.convert_to_tensor(y_shuffled[start:end])
            z_batch = tf.convert_to_tensor(z_shuffled[start:end])
            adversarial_train_step(predictor, adversary, p_optimizer, a_optimizer, x_batch, y_batch, z_batch, alpha)

    y_pred_prob = predictor.predict(X_test, verbose=0).reshape(-1)
    metrics = compute_metrics(y_test, y_pred_prob, z_test)
    print(f"alpha={alpha} seed={seed}: acc={metrics['acc']:.4f} deo={metrics['deo']:.4f} dao={metrics['dao']:.4f}")
    return metrics


def run():
    for alpha in ALPHAS:
        results = [run_one(alpha, s) for s in SEEDS]
        print(f"--- alpha={alpha} (german) summary ---")
        for key in ["acc", "deo", "dao"]:
            vals = np.array([r[key] for r in results])
            print(f"  {key}: mean={vals.mean():.4f} std={vals.std():.4f}")


if __name__ == "__main__":
    run()
