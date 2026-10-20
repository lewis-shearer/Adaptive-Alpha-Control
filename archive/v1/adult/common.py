"""
Shared preprocessing, model architecture, and metrics for the Adult Income
experiment (baseline.py / fixed.py / dynamic.py).

Kept in one place deliberately: baseline/fixed/dynamic must use byte-identical
preprocessing, architecture, and metric definitions for the three-way
comparison to be a valid ablation (only the alpha mechanism should differ).
A preprocessing bug once had to be fixed in three separate copies of this
code -- see README.md's "Experimental setup" section -- this module is the
fix for that class of problem.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "data/adult.tsv"

_COLS = [
    "Age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
_NUM_COLS = ["Age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
_CAT_COLS = ["workclass", "education", "marital-status", "occupation",
             "relationship", "race", "native-country"]


def load_data(data_path=DATA_PATH):
    """Load and preprocess Adult Income, returning the same 80/20 stratified
    train/test split (random_state=42) used by every condition."""
    df = pd.read_csv(data_path, sep="\t", na_values=["?", " ?"])
    if list(df.columns) != _COLS:
        df.columns = _COLS[: len(df.columns)]

    df = df.dropna().reset_index(drop=True)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    for c in _NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    df["income"] = LabelEncoder().fit_transform(df["income"])
    df["gender"] = LabelEncoder().fit_transform(df["gender"])

    df_enc = pd.get_dummies(df, columns=_CAT_COLS)

    feature_cols = [c for c in df_enc.columns if c not in ("income", "gender")]
    X_raw = df_enc[feature_cols].values.astype(np.float32)
    y_all = df_enc["income"].values.astype(np.float32)
    z_all = df_enc["gender"].values.astype(np.float32)

    X_all = StandardScaler().fit_transform(X_raw)

    return train_test_split(
        X_all, y_all, z_all, test_size=0.2, random_state=42, stratify=y_all
    )


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


def compute_metrics(y_true, y_pred_prob, z_true, threshold=0.5):
    """ACC, DEO (equal-opportunity gap), DAO (average-odds gap)."""
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


class MetricHistory:
    """Adaptive Alpha Control's controller state: tracks EMA-smoothed
    epoch-to-epoch deltas of ACC/DEO/DAO and turns them into a single
    'pressure' signal that drives the alpha update (dynamic.py only)."""

    def __init__(self, w_deo, w_dao, w_acc, acc_floor, ema_alpha=0.7):
        self.w_deo = w_deo
        self.w_dao = w_dao
        self.w_acc = w_acc
        self.acc_floor = acc_floor
        self.ema_alpha = ema_alpha
        self.history = []
        self.ema_ddeo = 0.0
        self.ema_ddao = 0.0
        self.ema_dacc = 0.0

    def update(self, acc, deo, dao):
        if self.history:
            prev_acc, prev_deo, prev_dao = self.history[-1]
            d_deo = deo - prev_deo
            d_dao = dao - prev_dao
            d_acc = acc - prev_acc
            self.ema_ddeo = self.ema_alpha * d_deo + (1 - self.ema_alpha) * self.ema_ddeo
            self.ema_ddao = self.ema_alpha * d_dao + (1 - self.ema_alpha) * self.ema_ddao
            self.ema_dacc = self.ema_alpha * d_acc + (1 - self.ema_alpha) * self.ema_dacc
        self.history.append((acc, deo, dao))

    def pressure(self, acc):
        # bias trending up -> pressure positive -> alpha increases -> lean harder on adversary
        # accuracy trending up -> pressure pulled down -> don't double-charge for a free fairness gain
        p = (self.w_deo * self.ema_ddeo
             + self.w_dao * self.ema_ddao
             - self.w_acc * self.ema_dacc)
        if acc < self.acc_floor:
            p = min(p, -0.2)
        return p

    def trend_str(self):
        deo_trend = "↑" if self.ema_ddeo > 0.005 else ("↓" if self.ema_ddeo < -0.005 else "→")
        dao_trend = "↑" if self.ema_ddao > 0.005 else ("↓" if self.ema_ddao < -0.005 else "→")
        acc_trend = "↑" if self.ema_dacc > 0.001 else ("↓" if self.ema_dacc < -0.001 else "→")
        return f"DEO{deo_trend} DAO{dao_trend} ACC{acc_trend}"
