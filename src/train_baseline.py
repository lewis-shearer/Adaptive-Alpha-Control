"""Plain predictor, no adversary. Step 1 of the rebuild.

Purpose: confirm the data pipeline and model architecture work at all,
and get a reference accuracy/bias number before any debiasing mechanism
exists. No alpha, no adversary — just BCE(y, y_hat).

Boilerplate (epoch loop, batching, evaluation, logging) is done for you
below. The one thing you need to fill in is train_step(): the actual
forward pass + loss + gradient update for a single batch.
"""
import numpy as np
import tensorflow as tf

from src.data import load_compas
from src.metrics import compute_metrics
from src.models import build_predictor

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 0


def train_step(predictor: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
                x_batch: tf.Tensor, y_batch: tf.Tensor) -> float:
    """Run one gradient step on predictor for this batch. Return the loss (float).

    TODO:
      1. Open a tf.GradientTape.
      2. Forward pass: y_hat = predictor(x_batch, training=True)
      3. Compute binary cross-entropy loss between y_batch and y_hat.
      4. Get gradients of the loss w.r.t. predictor.trainable_variables.
      5. Apply them with optimizer.apply_gradients(...).
      6. Return the loss value as a plain float.

    """
    with tf.GradientTape() as tape:
        y_hat = predictor(x_batch, training=True)
        y_hat = tf.squeeze(y_hat, axis=-1)   # (64, 1) -> (64,)
        loss = tf.keras.losses.binary_crossentropy(y_batch, y_hat)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, predictor.trainable_variables)
    optimizer.apply_gradients(zip(grads, predictor.trainable_variables))

    return float(loss)  



def run():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, z_train, z_test = load_compas()
    n_train = X_train.shape[0]

    predictor = build_predictor(input_dim=X_train.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled = X_train[perm], y_train[perm]

        epoch_losses = []
        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = tf.convert_to_tensor(X_shuffled[start:end])
            y_batch = tf.convert_to_tensor(y_shuffled[start:end])
            loss = train_step(predictor, optimizer, x_batch, y_batch)
            epoch_losses.append(loss)

        y_pred_prob = predictor.predict(X_test, verbose=0).reshape(-1)
        metrics = compute_metrics(y_test, y_pred_prob, z_test)

        print(
            f"epoch {epoch:02d}  loss={np.mean(epoch_losses):.4f}  "
            f"acc={metrics['acc']:.4f}  deo={metrics['deo']:.4f}  dao={metrics['dao']:.4f}"
        )


if __name__ == "__main__":
    run()
