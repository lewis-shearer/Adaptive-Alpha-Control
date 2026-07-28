"""
Baseline (no debiasing) - same architecture as adversarial model
for direct comparison of ACC, DEO, DAO metrics.

Kept structurally IDENTICAL to fixed.py / dynamic.py (same preprocessing,
same build_predictor, same compute_metrics, same per-seed model rebuild,
same seed sweep, same W&B project) -- the only difference is that there is
no adversary and no alpha term in the loss, so this is a valid ablation
rather than a confounded one.
"""

# import packages
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import wandb

# config of hyperparameters and constants
DATA_PATH     = "data/adult.tsv"
WANDB_ENTITY  = "lshearer2957-self"
WANDB_PROJECT = "final_results"
EPOCHS        = 30
BATCH_SIZE    = 256
LR            = 1e-3

SEEDS = list(range(30))  # seeds 0 – 29

# load and preprocess the dataset
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


## MODEL BUILD -- identical predictor to fixed.py / dynamic.py, no adversary

def build_predictor(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(64, activation="relu")(inp)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1,  activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="predictor")


## Metrics -- identical to fixed.py / dynamic.py
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


##### TRAINING LOOP ######

for SEED in SEEDS:
    print(f"\n{'='*60}")
    print(f"  SEED {SEED}  ({SEEDS.index(SEED)+1}/{len(SEEDS)})")
    print(f"{'='*60}")

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    predictor = build_predictor(X_train.shape[1])
    optimiser = tf.keras.optimizers.Adam(LR)
    bce       = tf.keras.losses.BinaryCrossentropy()

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"baseline_adult_gender_seed{SEED}",
        config={
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr":         LR,
            "seed":       SEED,
            "model":      "Baseline_NoDebiasing",
        },
        group="Baseline",
        reinit=True,
    )

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
    print(f"  Seed {SEED} done.")

print("\nAll seeds complete. Results logged to W&B:",
      f"{WANDB_ENTITY}/{WANDB_PROJECT}")
