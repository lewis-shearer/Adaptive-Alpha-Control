"""
Adversarial Debiasing - Zhang et al. (2018)
"Mitigating Unwanted Biases with Adversarial Learning"

Architecture:
  - Predictor P: features -> income prediction
  - Adversary A: predictor output -> gender prediction
  Training alternates: minimise predictor loss + maximise adversary loss
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import wandb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/adult.tsv"
WANDB_ENTITY  = "lshearer2957-self"
WANDB_PROJECT = "FINAL"
EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 1e-3
ALPHA       = 0.1   # adversary loss weight (λ in paper)

# SEED 40, 0, 123
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Load & Preprocess ─────────────────────────────────────────────────────────
cols = [
    "Age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Try with header first; if column count mismatches, fall back to no-header
_df = pd.read_csv(DATA_PATH, sep="\t", na_values=["?", " ?"])
if list(_df.columns) == cols:
    df = _df  # file already has correct header
else:
    # Rename whatever columns exist to our expected names
    _df.columns = cols[:len(_df.columns)]
    df = _df

df = df.dropna().reset_index(drop=True)

# Strip whitespace from all string columns
for c in df.select_dtypes(include="str").columns:
    df[c] = df[c].str.strip()

# Coerce numeric columns (guards against header-as-data surviving)
num_cols = ["Age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

# Encode target + sensitive attribute
df["income"] = LabelEncoder().fit_transform(df["income"])   # 0/1
df["gender"] = LabelEncoder().fit_transform(df["gender"])   # 0/1 (Female=0, Male=1)

# One-hot encode categoricals, scale numerics
cat_cols = ["workclass","education","marital-status","occupation",
            "relationship","race","native-country"]
num_cols = ["Age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]

df_enc = pd.get_dummies(df, columns=cat_cols)

feature_cols = [c for c in df_enc.columns if c not in ("income", "gender")]
X = df_enc[feature_cols].values.astype(np.float32)
y = df_enc["income"].values.astype(np.float32)
z = df_enc["gender"].values.astype(np.float32)   # sensitive attribute

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    X, y, z, test_size=0.2, random_state=42, stratify=y
)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_predictor(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(64, activation="relu")(inp)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)   # income prob
    return tf.keras.Model(inp, out, name="predictor")

def build_adversary():
    inp = tf.keras.Input(shape=(1,))                          # predictor output
    x   = tf.keras.layers.Dense(32, activation="relu")(inp)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # gender prob
    return tf.keras.Model(inp, out, name="adversary")

predictor = build_predictor(X_train.shape[1])
adversary  = build_adversary()

pred_optimiser = tf.keras.optimizers.Adam(LR)
adv_optimiser  = tf.keras.optimizers.Adam(LR)

bce = tf.keras.losses.BinaryCrossentropy()

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred_prob, z_true, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(float)

    acc = np.mean(y_pred == y_true)

    # DEO  = |P(ŷ=1|z=0,y=1) - P(ŷ=1|z=1,y=1)|  (equal opportunity)
    mask0_pos = (z_true == 0) & (y_true == 1)
    mask1_pos = (z_true == 1) & (y_true == 1)
    tpr0 = y_pred[mask0_pos].mean() if mask0_pos.sum() > 0 else 0.0
    tpr1 = y_pred[mask1_pos].mean() if mask1_pos.sum() > 0 else 0.0
    deo  = abs(tpr0 - tpr1)

    # DAO  = average of |TPR diff| and |FPR diff|  (equalised odds)
    mask0_neg = (z_true == 0) & (y_true == 0)
    mask1_neg = (z_true == 1) & (y_true == 0)
    fpr0 = y_pred[mask0_neg].mean() if mask0_neg.sum() > 0 else 0.0
    fpr1 = y_pred[mask1_neg].mean() if mask1_neg.sum() > 0 else 0.0
    dao  = (abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0

    return acc, deo, dao

# ── Training ──────────────────────────────────────────────────────────────────
wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name="fixed_adult_gender_123", config={
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "alpha": ALPHA,
    "model": "Zhang2018_AdversarialDebiasing"
})

n_train   = X_train.shape[0]
n_batches = n_train // BATCH_SIZE

for epoch in range(EPOCHS):
    # Shuffle
    idx = np.random.permutation(n_train)
    X_tr, y_tr, z_tr = X_train[idx], y_train[idx], z_train[idx]

    pred_losses, adv_losses = [], []

    for b in range(n_batches):
        s  = b * BATCH_SIZE
        e  = (b + 1) * BATCH_SIZE
        Xb = tf.constant(X_tr[s:e])
        yb = tf.constant(y_tr[s:e].reshape(-1, 1))
        zb = tf.constant(z_tr[s:e].reshape(-1, 1))

        # ── Step 1: update adversary (predictor frozen) ──────────────────────
        with tf.GradientTape() as tape_adv:
            y_hat   = predictor(Xb, training=False)
            z_hat   = adversary(y_hat, training=True)
            loss_adv = bce(zb, z_hat)

        grads_adv = tape_adv.gradient(loss_adv, adversary.trainable_variables)
        adv_optimiser.apply_gradients(zip(grads_adv, adversary.trainable_variables))

        # ── Step 2: update predictor (adversary frozen) ──────────────────────
        with tf.GradientTape() as tape_pred:
            y_hat     = predictor(Xb, training=True)
            z_hat     = adversary(y_hat, training=False)
            loss_pred = bce(yb, y_hat) - ALPHA * bce(zb, z_hat)  # eq. (3) in paper

        grads_pred = tape_pred.gradient(loss_pred, predictor.trainable_variables)
        pred_optimiser.apply_gradients(zip(grads_pred, predictor.trainable_variables))

        pred_losses.append(float(loss_pred))
        adv_losses.append(float(loss_adv))

    # ── Epoch eval ────────────────────────────────────────────────────────────
    y_pred_prob = predictor(tf.constant(X_test), training=False).numpy().flatten()
    acc, deo, dao = compute_metrics(y_test, y_pred_prob, z_test)

    wandb.log({
        "epoch":      epoch + 1,
        "pred_loss":  np.mean(pred_losses),
        "adv_loss":   np.mean(adv_losses),
        "ACC":        acc,
        "DEO":        deo,
        "DAO":        dao,
    })

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"pred_loss={np.mean(pred_losses):.4f}  adv_loss={np.mean(adv_losses):.4f} | "
          f"ACC={acc:.4f}  DEO={deo:.4f}  DAO={dao:.4f}")

wandb.finish()
print("\nDone. Results logged to W&B:", f"{WANDB_ENTITY}/{WANDB_PROJECT}")