"""Predictor + adversary, fixed alpha. Step 2 of the rebuild.

This is the actual Zhang et al. (2018) game. Two networks, two optimizers,
one fixed alpha controlling how hard the predictor fights the adversary.

Boilerplate (epoch loop, batching, evaluation, logging) is done for you,
same as train_baseline.py. The thing you need to fill in is
adversarial_train_step(): one full round of alternating updates for a
single batch.
"""
import numpy as np
import tensorflow as tf

from src.data import load_compas
from src.metrics import compute_metrics
from src.models import build_predictor, build_adversary

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
ALPHA = 2         # fixed for this file — this is the knob the adaptive
                      # version (step 3) will later replace with something
                      # that moves during training
SEED = 0


def adversarial_train_step(predictor, adversary, p_optimizer, a_optimizer,
                            x_batch, y_batch, z_batch, alpha) -> tuple[float, float]:
    """One batch of alternating updates. Return (predictor_loss, adversary_loss).

    TODO — two separate GradientTape blocks, run in this order:

    1. ADVERSARY STEP (predictor frozen):
       - with tf.GradientTape() as tape:
           y_hat = predictor(x_batch, training=True)   # just to get y_hat
           y_hat = tf.squeeze(y_hat, axis=-1)
           z_hat = adversary(tf.expand_dims(y_hat, axis=-1), training=True)
           z_hat = tf.squeeze(z_hat, axis=-1)
           adv_loss = tf.reduce_mean(
               tf.keras.losses.binary_crossentropy(z_batch, z_hat)
           )
       - gradients of adv_loss w.r.t. adversary.trainable_variables ONLY
         (not predictor's — predictor is frozen for this step even though
         y_hat was computed from it; that's why we only ask the tape for
         gradients w.r.t. adversary variables, never apply anything to
         the predictor here)
       - apply those gradients with a_optimizer

    2. PREDICTOR STEP (adversary frozen):
       - with tf.GradientTape() as tape:
           y_hat = predictor(x_batch, training=True)
           y_hat = tf.squeeze(y_hat, axis=-1)
           task_loss = tf.reduce_mean(
               tf.keras.losses.binary_crossentropy(y_batch, y_hat)
           )
           z_hat = adversary(tf.expand_dims(y_hat, axis=-1), training=False)
           z_hat = tf.squeeze(z_hat, axis=-1)
           adv_loss_for_predictor = tf.reduce_mean(
               tf.keras.losses.binary_crossentropy(z_batch, z_hat)
           )
           predictor_loss = task_loss - alpha * adv_loss_for_predictor
       - gradients of predictor_loss w.r.t. predictor.trainable_variables
         ONLY (adversary stays frozen — don't touch its weights here)
       - apply those gradients with p_optimizer

    Why recompute y_hat in step 2 instead of reusing step 1's: step 1's
    y_hat was recorded on a tape that's already been used/closed, and more
    importantly the predictor's weights may have just changed between
    calls in a real loop — always forward-pass fresh inside the tape
    you're about to differentiate through.

    Return (float(task_loss or predictor_loss), float(adv_loss)) — pick
    predictor_loss or task_loss for the first value, your call, just be
    consistent about which you're logging.
    """
    # --- adversary step (predictor frozen) ---
    with tf.GradientTape() as tape:
        y_hat = predictor(x_batch, training=True)
        y_hat = tf.squeeze(y_hat, axis=-1)
        z_hat = adversary(tf.expand_dims(y_hat, axis=-1), training=True)
        z_hat = tf.squeeze(z_hat, axis=-1)
        adv_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(z_batch, z_hat)
        )

    adv_grads = tape.gradient(adv_loss, adversary.trainable_variables)
    a_optimizer.apply_gradients(zip(adv_grads, adversary.trainable_variables))

    # --- predictor step (adversary frozen) ---
    with tf.GradientTape() as tape:
        y_hat = predictor(x_batch, training=True)
        y_hat = tf.squeeze(y_hat, axis=-1)
        task_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_batch, y_hat)
        )
        z_hat = adversary(tf.expand_dims(y_hat, axis=-1), training=False)
        z_hat = tf.squeeze(z_hat, axis=-1)
        adv_loss_for_predictor = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(z_batch, z_hat)
        )
        predictor_loss = task_loss - alpha * adv_loss_for_predictor

    pred_grads = tape.gradient(predictor_loss, predictor.trainable_variables)
    p_optimizer.apply_gradients(zip(pred_grads, predictor.trainable_variables))

    return float(predictor_loss), float(adv_loss)


def run():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, z_train, z_test = load_compas()
    n_train = X_train.shape[0]

    predictor = build_predictor(input_dim=X_train.shape[1])
    adversary = build_adversary()
    p_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    a_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled, z_shuffled = X_train[perm], y_train[perm], z_train[perm]

        p_losses, a_losses = [], []
        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = tf.convert_to_tensor(X_shuffled[start:end])
            y_batch = tf.convert_to_tensor(y_shuffled[start:end])
            z_batch = tf.convert_to_tensor(z_shuffled[start:end])
            p_loss, a_loss = adversarial_train_step(
                predictor, adversary, p_optimizer, a_optimizer,
                x_batch, y_batch, z_batch, ALPHA,
            )
            p_losses.append(p_loss)
            a_losses.append(a_loss)

        y_pred_prob = predictor.predict(X_test, verbose=0).reshape(-1)
        metrics = compute_metrics(y_test, y_pred_prob, z_test)

        print(
            f"epoch {epoch:02d}  p_loss={np.mean(p_losses):.4f}  a_loss={np.mean(a_losses):.4f}  "
            f"acc={metrics['acc']:.4f}  deo={metrics['deo']:.4f}  dao={metrics['dao']:.4f}"
        )


if __name__ == "__main__":
    run()
