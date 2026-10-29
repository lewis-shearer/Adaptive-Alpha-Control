"""Adaptive alpha via dual ascent on two fairness constraints. Step 3.

Instead of one fixed alpha, this file maintains TWO multipliers:
  alpha_deo — pressure from the DEO constraint (TPR gap, y=1 subset only)
  alpha_dao — pressure from the DAO constraint (average of TPR and FPR gap,
              both subsets)

Problem being solved:
  minimize   task loss
  subject to DEO <= EPS_DEO
             DAO <= EPS_DAO

Dual ascent update (done for you, in run()):
  pressure = metric - epsilon
  if acc < ACC_FLOOR: pressure = min(pressure, 0)   # never increase alpha
                                                      # while accuracy is
                                                      # below the floor —
                                                      # can still decrease
  alpha = max(0, alpha + ALPHA_LR * pressure)

The TODO is adaptive_train_step(): the two-SUBSET conditioned adversarial
training step. Two separate adversary networks, each trained only on the
part of the batch matching one value of y — NOT the same as
train_fixed_alpha.py's single adversary on the whole batch.
"""
import numpy as np
import tensorflow as tf

from src.data import load_compas
from src.metrics import compute_metrics
from src.models import build_predictor, build_adversary

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
ALPHA_LR = 0.5
EPS_DEO = 0.077   # 30% of this project's own measured baseline DEO
EPS_DAO = 0.064   # 30% of this project's own measured baseline DAO
ACC_FLOOR = 0.60  # below baseline's ~0.654, matches where alpha=2 fixed run
                  # already dipped to in practice — not an arbitrary guess
SEED = 0


def adaptive_train_step(predictor, adv_pos, adv_neg, p_optimizer, a_pos_optimizer,
                         a_neg_optimizer, x_batch, y_batch, z_batch,
                         alpha_deo, alpha_dao):
    """One batch of alternating updates, with TWO label-conditioned adversaries.

    TODO:

    1. Split the batch by true label (do this with tf.boolean_mask, not by
       indexing — e.g. `mask_pos = tf.equal(y_batch, 1.0)`):
         x_pos, z_pos = x where y==1 (via predictor's y_hat, not x directly —
                        actually what you need is y_hat and z per subset, so
                        compute y_hat for the WHOLE batch first in each tape,
                        then mask y_hat and z_batch by mask_pos / mask_neg)

    2. ADVERSARY STEP (predictor frozen) — train BOTH adversaries in one tape
       or two, your call:
         y_hat = predictor(x_batch, training=True); squeeze to (batch,)
         y_hat_pos = tf.boolean_mask(y_hat, mask_pos)
         z_pos = tf.boolean_mask(z_batch, mask_pos)
         z_hat_pos = adv_pos(tf.expand_dims(y_hat_pos, -1), training=True); squeeze
         adv_pos_loss = mean(BCE(z_pos, z_hat_pos))
         (mirror for adv_neg using mask_neg)
       Gradients of adv_pos_loss w.r.t. adv_pos.trainable_variables only,
       apply with a_pos_optimizer. Same for adv_neg / a_neg_optimizer.

       CAUTION: a minibatch can occasionally have zero examples of one class
       (rare with batch=64 here, but guard it) — if a subset is empty, skip
       that adversary's update for this batch (e.g. `if tf.size(z_pos) == 0:
       skip`), don't let an empty-tensor loss blow up the gradient step.

    3. PREDICTOR STEP (both adversaries frozen):
         y_hat = predictor(x_batch, training=True); squeeze
         task_loss = mean(BCE(y_batch, y_hat))
         y_hat_pos = boolean_mask(y_hat, mask_pos); z_pos = boolean_mask(z_batch, mask_pos)
         y_hat_neg = boolean_mask(y_hat, mask_neg); z_neg = boolean_mask(z_batch, mask_neg)
         z_hat_pos = adv_pos(expand_dims(y_hat_pos, -1), training=False); squeeze
         z_hat_neg = adv_neg(expand_dims(y_hat_neg, -1), training=False); squeeze
         adv_pos_loss = mean(BCE(z_pos, z_hat_pos))   # proxy for TPR-gap leakage
         adv_neg_loss = mean(BCE(z_neg, z_hat_neg))   # proxy for FPR-gap leakage

         predictor_loss = task_loss \
             - alpha_deo * adv_pos_loss \
             - alpha_dao * 0.5 * (adv_pos_loss + adv_neg_loss)

       Gradients of predictor_loss w.r.t. predictor.trainable_variables only,
       apply with p_optimizer.

    Return (float(predictor_loss), float(adv_pos_loss), float(adv_neg_loss)).
    """
    mask_pos = tf.equal(y_batch, 1.0)
    mask_neg = tf.equal(y_batch, 0.0)

    # --- adversary step (predictor frozen) ---
    with tf.GradientTape(persistent=True) as tape:
        y_hat = predictor(x_batch, training=True)
        y_hat = tf.squeeze(y_hat, axis=-1)

        y_hat_pos = tf.boolean_mask(y_hat, mask_pos)
        z_pos = tf.boolean_mask(z_batch, mask_pos)
        y_hat_neg = tf.boolean_mask(y_hat, mask_neg)
        z_neg = tf.boolean_mask(z_batch, mask_neg)

        if tf.size(z_pos) > 0:
            z_hat_pos = adv_pos(tf.expand_dims(y_hat_pos, axis=-1), training=True)
            z_hat_pos = tf.squeeze(z_hat_pos, axis=-1)
            adv_pos_loss_step1 = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(z_pos, z_hat_pos)
            )
        else:
            adv_pos_loss_step1 = None

        if tf.size(z_neg) > 0:
            z_hat_neg = adv_neg(tf.expand_dims(y_hat_neg, axis=-1), training=True)
            z_hat_neg = tf.squeeze(z_hat_neg, axis=-1)
            adv_neg_loss_step1 = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(z_neg, z_hat_neg)
            )
        else:
            adv_neg_loss_step1 = None

    if adv_pos_loss_step1 is not None:
        grads = tape.gradient(adv_pos_loss_step1, adv_pos.trainable_variables)
        a_pos_optimizer.apply_gradients(zip(grads, adv_pos.trainable_variables))

    if adv_neg_loss_step1 is not None:
        grads = tape.gradient(adv_neg_loss_step1, adv_neg.trainable_variables)
        a_neg_optimizer.apply_gradients(zip(grads, adv_neg.trainable_variables))

    del tape

    # --- predictor step (both adversaries frozen) ---
    with tf.GradientTape() as tape:
        y_hat = predictor(x_batch, training=True)
        y_hat = tf.squeeze(y_hat, axis=-1)
        task_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_batch, y_hat)
        )

        y_hat_pos = tf.boolean_mask(y_hat, mask_pos)
        z_pos = tf.boolean_mask(z_batch, mask_pos)
        y_hat_neg = tf.boolean_mask(y_hat, mask_neg)
        z_neg = tf.boolean_mask(z_batch, mask_neg)

        if tf.size(z_pos) > 0:
            z_hat_pos = adv_pos(tf.expand_dims(y_hat_pos, axis=-1), training=False)
            z_hat_pos = tf.squeeze(z_hat_pos, axis=-1)
            adv_pos_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(z_pos, z_hat_pos)
            )
        else:
            adv_pos_loss = tf.constant(0.0)

        if tf.size(z_neg) > 0:
            z_hat_neg = adv_neg(tf.expand_dims(y_hat_neg, axis=-1), training=False)
            z_hat_neg = tf.squeeze(z_hat_neg, axis=-1)
            adv_neg_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(z_neg, z_hat_neg)
            )
        else:
            adv_neg_loss = tf.constant(0.0)

        predictor_loss = task_loss \
            - alpha_deo * adv_pos_loss \
            - alpha_dao * 0.5 * (adv_pos_loss + adv_neg_loss)

    grads = tape.gradient(predictor_loss, predictor.trainable_variables)
    p_optimizer.apply_gradients(zip(grads, predictor.trainable_variables))

    return float(predictor_loss), float(adv_pos_loss), float(adv_neg_loss)


def run():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

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

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(n_train)
        X_shuffled, y_shuffled, z_shuffled = X_train[perm], y_train[perm], z_train[perm]

        p_losses = []
        for start in range(0, n_train, BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = tf.convert_to_tensor(X_shuffled[start:end])
            y_batch = tf.convert_to_tensor(y_shuffled[start:end])
            z_batch = tf.convert_to_tensor(z_shuffled[start:end])
            p_loss, _, _ = adaptive_train_step(
                predictor, adv_pos, adv_neg, p_optimizer, a_pos_optimizer,
                a_neg_optimizer, x_batch, y_batch, z_batch, alpha_deo, alpha_dao,
            )
            p_losses.append(p_loss)

        y_pred_prob = predictor.predict(X_test, verbose=0).reshape(-1)
        metrics = compute_metrics(y_test, y_pred_prob, z_test)
        acc, deo, dao = metrics["acc"], metrics["deo"], metrics["dao"]

        # dual ascent update
        pressure_deo = deo - EPS_DEO
        pressure_dao = dao - EPS_DAO
        if acc < ACC_FLOOR:
            pressure_deo = min(pressure_deo, 0.0)
            pressure_dao = min(pressure_dao, 0.0)
        alpha_deo = max(0.0, alpha_deo + ALPHA_LR * pressure_deo)
        alpha_dao = max(0.0, alpha_dao + ALPHA_LR * pressure_dao)

        print(
            f"epoch {epoch:02d}  p_loss={np.mean(p_losses):.4f}  "
            f"acc={acc:.4f}  deo={deo:.4f}  dao={dao:.4f}  "
            f"alpha_deo={alpha_deo:.4f}  alpha_dao={alpha_dao:.4f}"
        )


if __name__ == "__main__":
    run()
