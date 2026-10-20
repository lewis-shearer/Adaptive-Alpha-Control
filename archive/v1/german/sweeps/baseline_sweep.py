"""
Baseline (no debiasing), W&B-Sweep-driven variant of baseline.py, for the
German Credit dataset.

baseline.py runs its own SEEDS loop in a single process. This script instead
runs ONE seed per invocation, with the seed injected by a W&B sweep
controller rather than chosen by the script itself -- so the 30 seeds can be
split across multiple machines running `wandb agent` at the same time.

Usage:
    wandb sweep german/sweeps/sweep_baseline.yaml
        -> creates the sweep on W&B, prints a SWEEP_ID (entity/project/id)

    wandb agent <SWEEP_ID>
        -> run this on every machine you want contributing runs. Each agent
           repeatedly asks W&B for the next unclaimed seed from
           sweep_baseline.yaml, runs it via `python german/sweeps/baseline_sweep.py`,
           then asks for another.

You can run this sweep's agent(s) at the same time as fixed_sweep.py's and
dynamic_sweep.py's agents (different sweeps, same or different machines) --
they log to the same final_results_german project under different groups
(Baseline / FixedAlpha_<alpha> / DynamicAlpha), so there's no collision.

Preprocessing, build_predictor, and compute_metrics are kept identical to
baseline.py / fixed.py / dynamic.py in the parent german/ folder so results stay
comparable regardless of which script (or which machine) produced them.
"""

# import packages
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import wandb

# config of hyperparameters and constants
DATA_PATH  = "data/German.tsv"
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 1e-3

# load and preprocess the dataset -- identical to baseline.py / fixed.py / dynamic.py
cols = [
    "existingchecking", "duration", "credithistory", "purpose", "creditamount",
    "savings", "employmentsince", "installmentrate", "otherdebts", "residencesince",
    "property", "otherinstallmentplans", "housing", "existingcredits", "job",
    "peopleliable", "telephone", "foreignworker", "classification", "gender"
]

df = pd.read_csv(DATA_PATH, sep="\t", na_values=["?", " ?"])
if list(df.columns) == cols:
    df = df
else:
    df.columns = cols[: len(df.columns)]
    df = df

df = df.dropna().reset_index(drop=True)
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip()

num_cols = ["duration", "creditamount", "installmentrate", "residencesince",
            "existingcredits", "peopleliable"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

df["classification"] = LabelEncoder().fit_transform(df["classification"])
df["gender"] = LabelEncoder().fit_transform(df["gender"])

cat_cols = ["existingchecking", "credithistory", "purpose", "savings",
            "employmentsince", "otherdebts", "property", "otherinstallmentplans",
            "housing", "job", "telephone", "foreignworker"]
df_enc = pd.get_dummies(df, columns=cat_cols)

feature_cols = [c for c in df_enc.columns if c not in ("classification", "gender")]
X_raw = df_enc[feature_cols].values.astype(np.float32)
y_all = df_enc["classification"].values.astype(np.float32)
z_all = df_enc["gender"].values.astype(np.float32)

scaler = StandardScaler()
X_all = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    X_all, y_all, z_all, test_size=0.2, random_state=42, stratify=y_all
)


## MODEL BUILD -- identical to baseline.py

def build_predictor(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(64, activation="relu")(inp)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1,  activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="predictor")


## Metrics -- identical to baseline.py
def compute_metrics(y_true, y_pred_prob, z_true, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(float)
    acc = np.mean(y_pred == y_true)

    mask0_pos = (z_true == 0) & (y_true == 1)
    mask1_pos = (z_true == 1) & (y_true == 1)
    tpr0 = y_pred[mask0_pos].mean() if mask0_pos.sum() > 0 else 0.0
    tpr1 = y_pred[mask1_pos].mean() if mask1_pos.sum() > 0 else 0.0
    deo = abs(tpr0 - tpr1)

    mask0_neg = (z_true == 0) & (y_true == 0)
    mask1_neg = (z_true == 1) & (y_true == 0)
    fpr0 = y_pred[mask0_neg].mean() if mask0_neg.sum() > 0 else 0.0
    fpr1 = y_pred[mask1_neg].mean() if mask1_neg.sum() > 0 else 0.0
    dao = (abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0

    return acc, deo, dao


##### ONE SWEEP-ASSIGNED RUN #####
# `wandb agent` invokes this script as `python3 german/sweeps/baseline_sweep.py
# --seed=...`, so seed is available from argv before wandb.init() ever runs --
# needed because `group` can only be set at init() time.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args, _ = parser.parse_known_args()
SEED = args.seed

wandb.init(group="Baseline", tags=["german", "Baseline_sweep"])

# rename the run so it matches baseline.py's naming convention regardless of
# which machine produced it
wandb.run.name = f"baseline_german_gender_seed{SEED}"

print(f"\n{'='*60}")
print(f"  SEED={SEED}")
print(f"{'='*60}")

np.random.seed(SEED)
tf.random.set_seed(SEED)

predictor = build_predictor(X_train.shape[1])
optimiser = tf.keras.optimizers.Adam(LR)
bce       = tf.keras.losses.BinaryCrossentropy()

n_train   = X_train.shape[0]
n_batches = n_train // BATCH_SIZE

for epoch in range(EPOCHS):
    idx = np.random.permutation(n_train)
    X_tr, y_tr, z_tr = X_train[idx], y_train[idx], z_train[idx]

    pred_losses = []

    for b in range(n_batches):
        s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        Xb = tf.constant(X_tr[s:e])
        yb = tf.constant(y_tr[s:e].reshape(-1, 1))

        with tf.GradientTape() as tape_pred:
            y_hat = predictor(Xb, training=True)
            loss_pred = bce(yb, y_hat)
        grads_pred = tape_pred.gradient(loss_pred, predictor.trainable_variables)
        optimiser.apply_gradients(zip(grads_pred, predictor.trainable_variables))

        pred_losses.append(float(loss_pred))

    y_pred_prob   = predictor(tf.constant(X_test), training=False).numpy().flatten()
    acc, deo, dao = compute_metrics(y_test, y_pred_prob, z_test)

    wandb.log({
        "epoch":     epoch + 1,
        "pred_loss": np.mean(pred_losses),
        "ACC":       acc,
        "DEO":       deo,
        "DAO":       dao,
        "seed":      SEED,
    })

    print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
          f"loss={np.mean(pred_losses):.4f} | "
          f"ACC={acc:.4f}  DEO={deo:.4f}  DAO={dao:.4f}")

wandb.finish()
print(f"\nSeed={SEED} done.")
