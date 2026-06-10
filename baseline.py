"""
Baseline (no debiasing) - same architecture as adversarial model
for direct comparison of ACC, DEO, DAO metrics.
"""

from random import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import wandb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH     = "data/adult.tsv"
WANDB_ENTITY  = "lshearer2957-self"
WANDB_PROJECT = "FINAL"
EPOCHS        = 30
BATCH_SIZE    = 256
LR            = 1e-3

# SEED 40, 0, 123
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Load & Preprocess (identical to adversarial script) ───────────────────────
cols = [
    "Age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

_df = pd.read_csv(DATA_PATH, sep="\t", na_values=["?", " ?"])
if list(_df.columns) == cols:
    df = _df
else:
    _df.columns = cols[:len(_df.columns)]
    df = _df

df = df.dropna().reset_index(drop=True)

for c in df.select_dtypes(include="str").columns:
    df[c] = df[c].str.strip()

num_cols = ["Age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

df["income"] = LabelEncoder().fit_transform(df["income"])
df["gender"] = LabelEncoder().fit_transform(df["gender"])

cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "native-country"]
df_enc = pd.get_dummies(df, columns=cat_cols)

feature_cols = [c for c in df_enc.columns if c not in ("income", "gender")]
X = df_enc[feature_cols].values.astype(np.float32)
y = df_enc["income"].values.astype(np.float32)
z = df_enc["gender"].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    X, y, z, test_size=0.2, random_state=42, stratify=y
)

# ── Model (same architecture as predictor in adversarial script) ──────────────
inp = tf.keras.Input(shape=(X_train.shape[1],))
x   = tf.keras.layers.Dense(64, activation="relu")(inp)
x   = tf.keras.layers.Dense(32, activation="relu")(x)
out = tf.keras.layers.Dense(1,  activation="sigmoid")(x)
model = tf.keras.Model(inp, out, name="baseline")

model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss="binary_crossentropy")

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred_prob, z_true, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(float)
    acc = np.mean(y_pred == y_true)

    mask0_pos = (z_true == 0) & (y_true == 1)
    mask1_pos = (z_true == 1) & (y_true == 1)
    tpr0 = y_pred[mask0_pos].mean() if mask0_pos.sum() > 0 else 0.0
    tpr1 = y_pred[mask1_pos].mean() if mask1_pos.sum() > 0 else 0.0
    deo  = abs(tpr0 - tpr1)

    mask0_neg = (z_true == 0) & (y_true == 0)
    mask1_neg = (z_true == 1) & (y_true == 0)
    fpr0 = y_pred[mask0_neg].mean() if mask0_neg.sum() > 0 else 0.0
    fpr1 = y_pred[mask1_neg].mean() if mask1_neg.sum() > 0 else 0.0
    dao  = (abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0

    return acc, deo, dao

# ── Training ──────────────────────────────────────────────────────────────────
wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name="base_adult_gender_123", config={
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
    "model": "Baseline_NoDebiasing"
})

n_train   = X_train.shape[0]
n_batches = n_train // BATCH_SIZE

for epoch in range(EPOCHS):
    idx = np.random.permutation(n_train)
    X_tr, y_tr, z_tr = X_train[idx], y_train[idx], z_train[idx]

    losses = []
    for b in range(n_batches):
        s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        Xb = X_tr[s:e]
        yb = y_tr[s:e].reshape(-1, 1)
        loss = model.train_on_batch(Xb, yb)
        losses.append(loss)

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    acc, deo, dao = compute_metrics(y_test, y_pred_prob, z_test)

    wandb.log({
        "epoch":    epoch + 1,
        "loss":     np.mean(losses),
        "ACC":      acc,
        "DEO":      deo,
        "DAO":      dao,
        "seed":     SEED
    })

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"loss={np.mean(losses):.4f} | "
          f"ACC={acc:.4f}  DEO={deo:.4f}  DAO={dao:.4f}")

wandb.finish()
print("\nDone. Results logged to W&B:", f"{WANDB_ENTITY}/{WANDB_PROJECT}")