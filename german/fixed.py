"""
Fixed-Alpha Baseline Sweep
An architecturally-matched control condition for comparing against the
Adaptive Alpha Control (EMA trend-aware) extension.

Identical to ../fixed.py, adapted for the German Credit dataset instead of
the Adult Income dataset:
  - target label:      classification (1 = good credit, 0 = bad credit)
  - sensitive attribute: gender (M / F)
  - different column set / one-hot columns for this dataset's features

Architecture:
  - Predictor P: features -> credit-good/bad prediction (sigmoid probability)
  - Adversary A: predictor output (y_hat) -> gender prediction
  Training alternates: minimise predictor loss + maximise adversary loss

IMPORTANT: this script is deliberately kept structurally IDENTICAL to
dynamic.py in this folder (same build_predictor, same build_adversary, same
combined-loss training step, same compute_metrics, same per-seed model
rebuild) -- the ONLY difference is how `alpha` is produced each epoch.

WHY A SWEEP INSTEAD OF ONE FIXED VALUE: see ../fixed.py's docstring -- same
reasoning applies here, so the German-dataset comparison is a fair one too.

COMPUTE NOTE: German Credit is much smaller than Adult (1,000 rows vs ~45k),
so this sweep is far cheaper here despite covering the same 8 alphas x 30
seeds grid.

Logs to a separate W&B project (final_results_german) so these runs don't
mix with the Adult Income experiments in final_results.

Run from the repo root: `python german/fixed.py`
"""

# import packages
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import wandb

# config of hyperparameters and constants
DATA_PATH     = "data/German.tsv"
WANDB_ENTITY  = "lshearer2957-self"
WANDB_PROJECT = "final_results_german"
EPOCHS        = 30
BATCH_SIZE    = 256
LR            = 1e-3

# Sweep of fixed alpha values to trace out the ACC vs DEO/DAO tradeoff curve.
# Same grid as the Adult experiments so the two datasets are comparable.
ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4708055927870487, 0.7, 1.0, 1.5]

SEEDS_PER_ALPHA = 30
SEEDS = list(range(SEEDS_PER_ALPHA))

# load and preprocess the dataset
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


## MODEL BUILDS -- identical to dynamic.py in this folder

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


## Metrics -- identical to dynamic.py in this folder
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


##### SWEEP + TRAINING LOOP #####
# Records final-epoch (ACC, DEO, DAO) per (alpha, seed) so you can build the
# tradeoff curve afterwards without re-parsing W&B.
sweep_results = []

for alpha in ALPHA_VALUES:
    print(f"\n{'#'*60}")
    print(f"  ALPHA = {alpha}")
    print(f"{'#'*60}")

    for SEED in SEEDS:
        print(f"\n{'='*60}")
        print(f"  ALPHA={alpha}  SEED {SEED}  ({SEEDS.index(SEED)+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        predictor = build_predictor(X_train.shape[1])
        adversary = build_adversary()
        pred_optimiser = tf.keras.optimizers.Adam(LR)
        adv_optimiser = tf.keras.optimizers.Adam(LR)
        bce = tf.keras.losses.BinaryCrossentropy()

        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"fixed_alpha{alpha:g}_german_gender_seed{SEED}",
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "alpha_fixed": alpha,
                "seed": SEED,
                "model": f"Zhang2018_Baseline_FixedAlphaSweep_{alpha:g}",
            },
            group=f"FixedAlpha_{alpha:g}",
            tags=["german"],
            reinit=True,
        )

        n_train = X_train.shape[0]
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

            y_pred_prob = predictor(tf.constant(X_test), training=False).numpy().flatten()
            acc, deo, dao = compute_metrics(y_test, y_pred_prob, z_test)

            wandb.log({
                "epoch": epoch + 1,
                "alpha": alpha,
                "pred_loss": np.mean(pred_losses),
                "adv_loss": np.mean(adv_losses),
                "ACC": acc,
                "DEO": deo,
                "DAO": dao,
                "seed": SEED,
            })

            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | alpha={alpha:.4f} | "
                  f"pred_loss={np.mean(pred_losses):.4f}  adv_loss={np.mean(adv_losses):.4f} | "
                  f"ACC={acc:.4f}  DEO={deo:.4f}  DAO={dao:.4f}")

        sweep_results.append({"alpha": alpha, "seed": SEED, "ACC": acc, "DEO": deo, "DAO": dao})
        wandb.finish()
        print(f"  Alpha={alpha}, Seed {SEED} done.")

# Save a local summary CSV of final-epoch metrics per (alpha, seed) for
# building the tradeoff curve / Pareto plot without depending on W&B access.
results_df = pd.DataFrame(sweep_results)
results_df.to_csv("german/fixed_alpha_sweep_results.csv", index=False)
print("\nSweep complete. Per-(alpha, seed) final metrics saved to german/fixed_alpha_sweep_results.csv")
print(results_df.groupby("alpha")[["ACC", "DEO", "DAO"]].agg(["mean", "std"]))

print("\nAll runs complete. Results logged to W&B:",
      f"{WANDB_ENTITY}/{WANDB_PROJECT}")
