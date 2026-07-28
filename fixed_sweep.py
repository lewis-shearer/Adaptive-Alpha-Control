"""
Fixed-Alpha sweep, W&B-Sweep-driven variant of fixed.py.

fixed.py runs its own ALPHA_VALUES x SEEDS loop in a single process. This
script instead runs ONE (alpha, seed) combination per invocation, with alpha
and seed injected by a W&B sweep controller rather than chosen by the script
itself -- so the work can be split across multiple machines.

Usage:
    wandb sweep sweep_fixed.yaml
        -> creates the sweep on W&B, prints a SWEEP_ID (entity/project/id)

    wandb agent <SWEEP_ID>
        -> run this on every machine you want contributing runs (e.g. your
           MacBook and a second laptop). Each agent repeatedly asks W&B
           for the next unclaimed (alpha, seed) pair from the grid in
           sweep_fixed.yaml, runs it via `python fixed_sweep.py`, then asks
           for another -- so the grid gets split across machines
           automatically, with no manual work-splitting and no risk of two
           machines training the same combination.

sweep_fixed.yaml's grid covers the full 8-value ALPHA_VALUES x 30-seed grid
(240 runs total) -- this script is the intended way to produce all of them,
rather than running fixed.py's manual loop locally.

Preprocessing, build_predictor, build_adversary, and compute_metrics are
kept identical to fixed.py / dynamic.py / baseline.py so results stay
comparable regardless of which script (or which machine) produced them.
"""

# import packagesseet
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import wandb

# config of hyperparameters and constants
DATA_PATH = "data/adult.tsv"
EPOCHS    = 30
BATCH_SIZE = 256
LR        = 1e-3

# load and preprocess the dataset -- identical to fixed.py / dynamic.py / baseline.py
cols = [
    "Age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
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
X_raw = df_enc[feature_cols].values.astype(np.float32)
y_all = df_enc["income"].values.astype(np.float32)
z_all = df_enc["gender"].values.astype(np.float32)

scaler = StandardScaler()
X_all = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    X_all, y_all, z_all, test_size=0.2, random_state=42, stratify=y_all
)


## MODEL BUILDS -- identical to fixed.py

def build_predictor(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="predictor")


def build_adversary():
    inp = tf.keras.Input(shape=(1,))
    x = tf.keras.layers.Dense(32, activation="relu")(inp)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="adversary")


## Metrics -- identical to fixed.py
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
# `wandb agent` invokes this script as `python3 fixed_sweep.py --alpha=...
# --seed=...`, so alpha/seed are available from argv before wandb.init() ever
# runs -- that's needed because `group` can only be set at init() time (it
# has no post-init setter), unlike `name`/`tags`.
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--seed", type=int, required=True)
args, _ = parser.parse_known_args()
alpha = args.alpha
SEED  = args.seed

wandb.init(group=f"FixedAlpha_{alpha:g}")

# rename the run so it matches fixed.py's naming convention regardless of
# which machine produced it
wandb.run.name = f"fixed_alpha{alpha:g}_adult_gender_seed{SEED}"
wandb.run.tags = wandb.run.tags + ("FixedAlpha_sweep",)

print(f"\n{'='*60}")
print(f"  ALPHA={alpha}  SEED={SEED}")
print(f"{'='*60}")

np.random.seed(SEED)
tf.random.set_seed(SEED)

predictor      = build_predictor(X_train.shape[1])
adversary      = build_adversary()
pred_optimiser = tf.keras.optimizers.Adam(LR)
adv_optimiser  = tf.keras.optimizers.Adam(LR)
bce            = tf.keras.losses.BinaryCrossentropy()

n_train   = X_train.shape[0]
n_batches = n_train // BATCH_SIZE

for epoch in range(EPOCHS):
    idx = np.random.permutation(n_train)
    X_tr, y_tr, z_tr = X_train[idx], y_train[idx], z_train[idx]

    pred_losses, adv_losses = [], []

    for b in range(n_batches):
        s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        Xb = tf.constant(X_tr[s:e])
        yb = tf.constant(y_tr[s:e].reshape(-1, 1))
        zb = tf.constant(z_tr[s:e].reshape(-1, 1))

        with tf.GradientTape() as tape_adv:
            y_hat = predictor(Xb, training=False)
            z_hat = adversary(y_hat, training=True)
            loss_adv = bce(zb, z_hat)
        grads_adv = tape_adv.gradient(loss_adv, adversary.trainable_variables)
        adv_optimiser.apply_gradients(zip(grads_adv, adversary.trainable_variables))

        with tf.GradientTape() as tape_pred:
            y_hat = predictor(Xb, training=True)
            z_hat = adversary(y_hat, training=False)
            loss_pred = bce(yb, y_hat) - alpha * bce(zb, z_hat)
        grads_pred = tape_pred.gradient(loss_pred, predictor.trainable_variables)
        pred_optimiser.apply_gradients(zip(grads_pred, predictor.trainable_variables))

        pred_losses.append(float(loss_pred))
        adv_losses.append(float(loss_adv))

    y_pred_prob   = predictor(tf.constant(X_test), training=False).numpy().flatten()
    acc, deo, dao = compute_metrics(y_test, y_pred_prob, z_test)

    wandb.log({
        "epoch":     epoch + 1,
        "alpha":     alpha,
        "pred_loss": np.mean(pred_losses),
        "adv_loss":  np.mean(adv_losses),
        "ACC":       acc,
        "DEO":       deo,
        "DAO":       dao,
        "seed":      SEED,
    })

    print(f"  Epoch {epoch+1:3d}/{EPOCHS} | alpha={alpha:.4f} | "
          f"pred_loss={np.mean(pred_losses):.4f}  adv_loss={np.mean(adv_losses):.4f} | "
          f"ACC={acc:.4f}  DEO={deo:.4f}  DAO={dao:.4f}")

wandb.finish()
print(f"\nAlpha={alpha}, Seed={SEED} done.")
