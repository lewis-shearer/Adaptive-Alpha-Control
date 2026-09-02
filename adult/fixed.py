"""
Fixed-Alpha Baseline Sweep
An architecturally-matched control condition for comparing against the
Adaptive Alpha Control (EMA trend-aware) extension.

Architecture:
  - Predictor P: features -> income prediction (sigmoid probability)
  - Adversary A: predictor output (y_hat) -> gender prediction
  Training alternates: minimise predictor loss + maximise adversary loss

IMPORTANT: this script is deliberately kept structurally IDENTICAL to the
adaptive-alpha script (same build_predictor, same build_adversary, same
combined-loss training step, same compute_metrics, same per-seed model
rebuild) -- the ONLY difference is how `alpha` is produced each epoch.
Keeping everything else identical is what makes a before/after comparison of
"adaptive alpha" a valid ablation rather than a confounded one.

WHY A SWEEP INSTEAD OF ONE FIXED VALUE: reporting your adaptive controller
against a single, arbitrarily-chosen fixed alpha is a weak baseline -- if
that one value happens to be badly tuned, the adaptive method "wins" for the
wrong reason. Instead, this script trains a fixed-alpha model across a range
of ALPHA_VALUES, each for multiple seeds, so you get an accuracy-vs-fairness
tradeoff curve (a Pareto frontier over ACC vs DEO/DAO). You then plot your
adaptive controller's result against that whole curve. If your point sits
on or beyond the frontier (better ACC for equal fairness, or better fairness
for equal ACC, than any fixed alpha achieves), that's a much stronger and
more defensible claim than beating one hand-picked number.

COMPUTE NOTE: sweeping N_ALPHAS values x N_SEEDS seeds x EPOCHS epochs is
significantly more expensive than a single run. SEEDS_PER_ALPHA is set
smaller than the 30 used elsewhere in this project -- enough to get a
reasonable mean/spread per alpha without ballooning training time. Increase
it if you have the compute budget and want tighter error bars on the curve.
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

# Sweep of fixed alpha values to trace out the ACC vs DEO/DAO tradeoff curve.
# Spans roughly an order of magnitude either side of ALPHA_INIT from the
# adaptive script (0.4708) so the adaptive controller's operating range is
# well-covered by the sweep.
ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4708055927870487, 0.7, 1.0, 1.5]

# Fewer seeds per alpha than the main 30-seed runs, since this sweep is
# 8x the training cost of a single run. Bump this up if compute allows.
SEEDS_PER_ALPHA = 30
SEEDS = list(range(SEEDS_PER_ALPHA))

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
# NOTE: fixed vs. original -- pandas string columns are dtype "object", not
# "str"; select_dtypes(include="str") silently matches nothing and skips
# whitespace stripping entirely.
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


## MODEL BUILDS -- identical to the adaptive-alpha script

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


## Metrics -- identical to the adaptive-alpha script
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
            name=f"fixed_alpha{alpha:g}_adult_gender_seed{SEED}",
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "alpha_fixed": alpha,
                "seed": SEED,
                "model": f"Zhang2018_Baseline_FixedAlphaSweep_{alpha:g}",
            },
            group=f"FixedAlpha_{alpha:g}",
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
results_df.to_csv("adult/fixed_alpha_sweep_results.csv", index=False)
print("\nSweep complete. Per-(alpha, seed) final metrics saved to adult/fixed_alpha_sweep_results.csv")
print(results_df.groupby("alpha")[["ACC", "DEO", "DAO"]].agg(["mean", "std"]))

print("\nAll runs complete. Results logged to W&B:",
      f"{WANDB_ENTITY}/{WANDB_PROJECT}")
