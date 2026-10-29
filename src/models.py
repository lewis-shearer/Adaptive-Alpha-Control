"""Predictor and adversary architectures — YOUR CODE GOES HERE.

Nothing is implemented below on purpose. Build these from what you already
know about Dense/Sequential models in TensorFlow/Keras, guided by the
explanation already covered:

Predictor:
  - Input: the feature vector (X_train.shape[1] columns from src/data.py).
  - Output: a single sigmoid probability (predicted probability of
    two_year_recid == 1).
  - Hidden layers are your call — start simple (e.g. two Dense/ReLU layers)
    and only make it bigger if training underfits.

Adversary:
  - Input: the predictor's output — ONE number (y_hat). Not the raw
    features. Not the true label y. This is the whole point: if the
    adversary can guess race from y_hat alone, y_hat is leaking race.
  - Output: a single sigmoid probability (predicted probability that
    z == 1, i.e. African-American).

Keep both as plain tf.keras.Model or tf.keras.Sequential subclasses/
instances — whatever you're already comfortable with from your CNN/linear
regression work. The architecture itself isn't the hard part here; the
training loop in train_baseline.py (and later train_fixed_alpha.py) is.
"""
import tensorflow as tf


def build_predictor(input_dim: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_adversary() -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
