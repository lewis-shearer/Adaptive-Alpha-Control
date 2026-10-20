"""
Adaptive Alpha Control: A Trend-Aware Dynamic Weighting Mechanism for Stable Adversarial Debiasing

An extension of:
Adversarial Debiasing - Zhang et al. (2018)
"Mitigating Unwanted Biases with Adversarial Learning"

Architecture:
  - Predictor P: features -> income prediction
  - Adversary A: predictor output -> gender prediction
  Training alternates: minimise predictor loss + maximise adversary loss

With a dynamic alpha controller driven by the models accuracy, differential in equal opportunity (DEO) and differential in average odds (DAO) metrcis during training. 
And uses EMA to smooth the trends in these metrics
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

# constants found through hyperparam tuning with wandb sweeps
ALPHA_INIT    = 0.4708055927870487
ALPHA_MIN     = 0.01
ALPHA_MAX     = 1.004835831178367
ALPHA_LR      = 0.4468379572755423

ACC_FLOOR     = 0.84
W_DEO         = 0.668900557521594
W_DAO         = 1.9288807992921904
W_ACC         = 4.645321445267202

EMA_ALPHA     = 0.3601034153775618

SEEDS = list(range(30))  # seeds 0 – 29

# load and preprocess the dataset
cols = [
    "Age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# laod in dataset etc
df = pd.read_csv(DATA_PATH, sep="\t", na_values=["?", " ?"])
if list(df.columns) == cols:
    df = df
else:
    # rename columns if they dont match expected
    df.columns = cols[: len(df.columns)]
    df = df

# clean up dataset 
df = df.dropna().reset_index(drop=True)
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip()

# convert categorical columns to numeric
num_cols = ["Age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

# encode target and sensitive attribute
# target = income > 50K 
# sensitive = gender (female=0, male=1)
df["income"] = LabelEncoder().fit_transform(df["income"])
df["gender"] = LabelEncoder().fit_transform(df["gender"])

# one hot encoding 
cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "native-country"]
df_enc = pd.get_dummies(df, columns=cat_cols)

# prepare features and labels
feature_cols = [c for c in df_enc.columns if c not in ("income", "gender")]
X_raw = df_enc[feature_cols].values.astype(np.float32)
y_all = df_enc["income"].values.astype(np.float32)
z_all = df_enc["gender"].values.astype(np.float32)

# scale features
scaler = StandardScaler()
X_all  = scaler.fit_transform(X_raw)

# traian and test split datasets 
X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    X_all, y_all, z_all, test_size=0.2, random_state=42, stratify=y_all
)




## MODELS BUILDS

def build_predictor(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(64, activation="relu")(inp)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1,  activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="predictor")

def build_adversary():
    inp = tf.keras.Input(shape=(1,))
    x   = tf.keras.layers.Dense(32, activation="relu")(inp)
    out = tf.keras.layers.Dense(1,  activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="adversary")



## Metrics and History Tracking
def compute_metrics(y_true, y_pred_prob, z_true, threshold=0.5):
    # Compute accuracy, DEO, and DAO metrics
    y_pred = (y_pred_prob >= threshold).astype(float)
    acc    = np.mean(y_pred == y_true)

    # Differential Equal Opportunity (DEO)
    mask0_pos = (z_true == 0) & (y_true == 1) # Mask for group 0 with positive true labels
    mask1_pos = (z_true == 1) & (y_true == 1) # Mask for group 1 with positive true labels
    # True Positive Rate for group 0
    if mask0_pos.sum() > 0:
        tpr0 = y_pred[mask0_pos].mean() 
    else:
        tpr0 = 0.0
    # True Positive Rate for group 1
    if mask1_pos.sum() > 0:
        tpr1 = y_pred[mask1_pos].mean() 
    else:
        tpr1 = 0.0

    deo  = abs(tpr0 - tpr1)




    mask0_neg = (z_true == 0) & (y_true == 0) # Mask for group 0 with negative true labels
    mask1_neg = (z_true == 1) & (y_true == 0) # Mask for group 1 with negative true labels
    # False Positive Rate for group 0
    if mask0_neg.sum() > 0:
        fpr0 = y_pred[mask0_neg].mean() 
    else:
        fpr0 = 0.0
    # False Positive Rate for group 1
    if mask1_neg.sum() > 0:
        fpr1 = y_pred[mask1_neg].mean()
    else:
        fpr1 = 0.0

    dao  = (abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0

    return acc, deo, dao






# History tracker 

class MetricHistory:
    #  intislise the ema alpha and the deltas 
    def __init__(self, ema_alpha=0.7):
        self.ema_alpha = ema_alpha
        self.history   = []
        self.ema_ddeo  = 0.0 # ema delta deo
        self.ema_ddao  = 0.0 # ema delta dao
        self.ema_dacc  = 0.0 # ema delta acc

    
    def update(self, acc, deo, dao):
        if self.history:
            # get latest metric from history
            prev_acc, prev_deo, prev_dao = self.history[-1]
            # calc delta for metrics
            d_deo = deo - prev_deo
            d_dao = dao - prev_dao
            d_acc = acc - prev_acc
            # update the EMA of the deltas using the specified alpha
            self.ema_ddeo = self.ema_alpha * d_deo + (1 - self.ema_alpha) * self.ema_ddeo
            self.ema_ddao = self.ema_alpha * d_dao + (1 - self.ema_alpha) * self.ema_ddao
            self.ema_dacc = self.ema_alpha * d_acc + (1 - self.ema_alpha) * self.ema_dacc
        self.history.append((acc, deo, dao))

    # Compute the pressure based on the EMA of deltas and weights
    def pressure(self, acc):
        # if bias is going up -> pressure becomes positive -> alpha increases -> adversary loss weighted more
        # if bias is going down -> pressure becomes negative -> alpha decreases -> adversary loss weighted less
        p = (W_DEO * self.ema_ddeo
           + W_DAO * self.ema_ddao
           - W_ACC * self.ema_dacc)
        # applay penalty if accuracy drops below specified threshold
        if acc < ACC_FLOOR:
            p = min(p, -0.2)
        return p
    

    # make a string version of the trend for log during training to understand better whats happening and debugging purpose
    def trend_str(self):
        deo_trend = "↑" if self.ema_ddeo > 0.005  else ("↓" if self.ema_ddeo < -0.005  else "→")
        dao_trend = "↑" if self.ema_ddao > 0.005  else ("↓" if self.ema_ddao < -0.005  else "→")
        acc_trend = "↑" if self.ema_dacc > 0.001  else ("↓" if self.ema_dacc < -0.001  else "→")
        return f"DEO{deo_trend} DAO{dao_trend} ACC{acc_trend}"


##### TRAINING LOOP ######

for SEED in SEEDS:
    print(f"\n{'='*60}")
    print(f"  SEED {SEED}  ({SEEDS.index(SEED)+1}/{len(SEEDS)})")
    print(f"{'='*60}")

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    predictor      = build_predictor(X_train.shape[1])
    adversary      = build_adversary()
    pred_optimiser = tf.keras.optimizers.Adam(LR)
    adv_optimiser  = tf.keras.optimizers.Adam(LR)
    bce            = tf.keras.losses.BinaryCrossentropy()
    #  init the wandb run for this run 
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"dynamic_adult_gender_seed{SEED}",
        config={
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "alpha_init":  ALPHA_INIT,
            "alpha_min":   ALPHA_MIN,
            "alpha_max":   ALPHA_MAX,
            "alpha_lr":    ALPHA_LR,
            "acc_floor":   ACC_FLOOR,
            "w_deo":       W_DEO,
            "w_dao":       W_DAO,
            "w_acc":       W_ACC,
            "ema_alpha":   EMA_ALPHA,
            "seed":        SEED,
            "model":       "Zhang2018_AdaptiveAlpha_EMA",
        },
        group="DynamicAlpha",
        reinit=True,
    )
    # init alpha and trackers
    alpha = ALPHA_INIT
    history = MetricHistory(ema_alpha=EMA_ALPHA)
    n_train = X_train.shape[0]
    n_batches = n_train // BATCH_SIZE
     
     # training start: 
    for epoch in range(EPOCHS):
        # shuffle
        idx = np.random.permutation(n_train)
        # X_tr = train features, y_tr = train labels, z_tr = sensitive attribute labels
        X_tr, y_tr, z_tr = X_train[idx], y_train[idx], z_train[idx]

        # init lists of loss per epoch
        pred_losses, adv_losses = [], []

        # iterarte through batches
        for b in range(n_batches):
            # get first and last index of batch
            s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            
            Xb = tf.constant(X_tr[s:e])
            yb = tf.constant(y_tr[s:e].reshape(-1, 1))
            zb = tf.constant(z_tr[s:e].reshape(-1, 1))

            # compute adversary loss and update adversary weights
            # goal to train adversary head to spot gender bias
            # predicts gender from predictor output (y_hat)
            # if successfull then predictor ouput contains structural footprint of gender
            with tf.GradientTape() as tape_adv:
                y_hat = predictor(Xb, training=False)
                z_hat = adversary(y_hat, training=True)
                loss_adv = bce(zb, z_hat)
            grads_adv = tape_adv.gradient(loss_adv, adversary.trainable_variables)
            adv_optimiser.apply_gradients(zip(grads_adv, adversary.trainable_variables))

            # compute predictor loss and update predictor weights
            # goal to optimise to hit targets while blinding adversary 
            # penalises the adversary when its acts well
            with tf.GradientTape() as tape_pred:
                y_hat = predictor(Xb, training=True)
                z_hat = adversary(y_hat, training=False)
                # alpha is the one dynamically updated per epoch to adjust how much the adversary loss is weighted in the predictor loss
                loss_pred = bce(yb, y_hat) - alpha * bce(zb, z_hat)
            grads_pred = tape_pred.gradient(loss_pred, predictor.trainable_variables)
            pred_optimiser.apply_gradients(zip(grads_pred, predictor.trainable_variables))
            
            # append the losses to the lists for logging
            pred_losses.append(float(loss_pred))
            adv_losses.append(float(loss_adv))

        y_pred_prob   = predictor(tf.constant(X_test), training=False).numpy().flatten()
        acc, deo, dao = compute_metrics(y_test, y_pred_prob, z_test)

        # passes on the metrics so they can be used to update alpha for next epoch
        history.update(acc, deo, dao)
        pressure  = history.pressure(acc)
        # sets strict boundries to alpha so it does not go to high or low and destabilise training
        new_alpha = float(np.clip(alpha + ALPHA_LR * pressure, ALPHA_MIN, ALPHA_MAX))
        
        # log metrics to wandb
        wandb.log({
            "epoch":     epoch + 1,
            "pred_loss": np.mean(pred_losses),
            "adv_loss":  np.mean(adv_losses),
            "ACC":       acc,
            "DEO":       deo,
            "DAO":       dao,
            "alpha":     alpha,
            "pressure":  pressure,
            "ema_ddeo":  history.ema_ddeo,
            "ema_ddao":  history.ema_ddao,
            "ema_dacc":  history.ema_dacc,
            "seed":      SEED,
        })

        # print metrics to console for monitoring
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"loss={np.mean(pred_losses):.4f} | "
              f"ACC={acc:.4f}  DEO={deo:.4f}  DAO={dao:.4f} | "
              f"[{history.trend_str()}]  pressure={pressure:+.3f}  "
              f"alpha: {alpha:.4f} → {new_alpha:.4f}")

        # update alpha for next epoch
        alpha = new_alpha

    wandb.finish()
    print(f"  Seed {SEED} done.")

print("\nAll seeds complete. Results logged to W&B:",
      f"{WANDB_ENTITY}/{WANDB_PROJECT}")
