"""Experiment 2: gradient-reversal joint training (Ganin & Lempitsky, DANN-style).

Replaces the two-tape alternating predictor/adversary updates
(train_fixed_alpha.py's adversarial_train_step) with a SINGLE joint
backward pass per batch:

  y_hat = predictor(x)
  reversed = grad_reverse(y_hat, lam)   # identity forward, -lam*grad backward
  z_hat = adversary(reversed)
  total_loss = task_loss + adv_loss     # NOT subtracted — the sign flip for
                                         # the predictor's path already
                                         # happens inside grad_reverse; the
                                         # adversary's own weights are
                                         # downstream of the reversal, so
                                         # they still just minimize adv_loss
                                         # normally.
  one tape, one set of gradients, one joint step.

lam plays the same role as alpha in train_fixed_alpha.py — same model
architecture, same hyperparameters, only the UPDATE PROCEDURE differs.
This is a controlled test of whether alternating-step training was
leaving anything on the table vs. a more standard DANN-style setup.
"""
import numpy as np
import tensorflow as tf

from src.data import load_compas
from src.metrics import compute_metrics
from src.models import build_predictor, build_adversary

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def make_grad_reverse(lam: float):
    @tf.custom_gradient
    def grad_reverse(x):
        def grad(dy):
            return -lam * dy
        return tf.identity(x), grad
    return grad_reverse


def joint_train_step(predictor, adversary, optimizer, x_batch, y_batch, z_batch, lam):
    grad_reverse = make_grad_reverse(lam)
    variables = predictor.trainable_variables + adversary.trainable_variables

    with tf.GradientTape() as tape:
        y_hat = predictor(x_batch, training=True)
        y_hat = tf.squeeze(y_hat, axis=-1)
        task_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, y_hat))

        reversed_y_hat = grad_reverse(y_hat)
        z_hat = adversary(tf.expand_dims(reversed_y_hat, axis=-1), training=True)
        z_hat = tf.squeeze(z_hat, axis=-1)
        adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(z_batch, z_hat))

        total_loss = task_loss + adv_loss

    grads = tape.gradient(total_loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return float(task_loss), float(adv_loss)


def run_one(lam: float, seed: int) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, z_train, z_test = load_compas()
    n_train = X_train.shape[0]

    predictor = build_predictor(input_dim=X_train.shape[1])
    adversary = build_adversary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled, z_shuffled = X_train[perm], y_train[perm], z_train[perm]
        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = tf.convert_to_tensor(X_shuffled[start:end])
            y_batch = tf.convert_to_tensor(y_shuffled[start:end])
            z_batch = tf.convert_to_tensor(z_shuffled[start:end])
            joint_train_step(predictor, adversary, optimizer, x_batch, y_batch, z_batch, lam)

    y_pred_prob = predictor.predict(X_test, verbose=0).reshape(-1)
    metrics = compute_metrics(y_test, y_pred_prob, z_test)
    print(f"lam={lam} seed={seed}: acc={metrics['acc']:.4f} deo={metrics['deo']:.4f} dao={metrics['dao']:.4f}")
    return metrics


def run(lams, seeds):
    for lam in lams:
        results = [run_one(lam, s) for s in seeds]
        print(f"--- lam={lam} summary ---")
        for key in ["acc", "deo", "dao"]:
            vals = np.array([r[key] for r in results])
            print(f"  {key}: mean={vals.mean():.4f} std={vals.std():.4f}")


if __name__ == "__main__":
    import sys
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run([0.5, 1.0, 2.0], list(range(n_seeds)))
