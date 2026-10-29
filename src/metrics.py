"""Fairness and accuracy metrics, computed on held-out predictions.

DEO (Difference in Equal Opportunity): |TPR_group0 - TPR_group1|
  Only compares true-positive rates. Answers: "among people who actually
  recidivated, is the model equally likely to catch it regardless of race?"

DAO (Difference in Average Odds): mean(|TPR gap|, |FPR gap|)
  Adds false-positive-rate parity on top of DEO. Answers the DEO question
  AND: "among people who did NOT recidivate, is the model equally likely
  to wrongly flag them regardless of race?"

Both are 0 = perfectly fair (by this definition), higher = more biased.
"""
import numpy as np


def _tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    pos = y_true == 1
    neg = y_true == 0
    tpr = y_pred[pos].mean() if pos.any() else 0.0
    fpr = y_pred[neg].mean() if neg.any() else 0.0
    return float(tpr), float(fpr)


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, z: np.ndarray) -> dict:
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = y_true.astype(int)
    z = z.astype(int)

    acc = float((y_pred == y_true).mean())

    tpr0, fpr0 = _tpr_fpr(y_true[z == 0], y_pred[z == 0])
    tpr1, fpr1 = _tpr_fpr(y_true[z == 1], y_pred[z == 1])

    deo = abs(tpr0 - tpr1)
    dao = (abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0

    return {
        "acc": acc,
        "deo": deo,
        "dao": dao,
        "tpr_group0": tpr0,
        "tpr_group1": tpr1,
        "fpr_group0": fpr0,
        "fpr_group1": fpr1,
    }
