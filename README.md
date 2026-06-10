# Adaptive Alpha Control

**Adaptive Alpha Control: A Trend-Aware Dynamic Weighting Mechanism for Stable Adversarial Debiasing**

An extension of the adversarial debiasing framework proposed by Zhang et al. (2018) that replaces a fixed adversarial weight (`α`) with a dynamic controller driven by fairness and accuracy trends during training.

The system continuously adjusts adversarial pressure using changes in:

* Accuracy (ACC)
* Difference in Equal Opportunity (DEO)
* Difference in Adversarial Outcomes (DAO)

The objective is to improve fairness while maintaining predictive performance, avoiding the instability and manual tuning associated with fixed adversarial weights.

---

# Overview

Traditional adversarial debiasing trains a predictor and an adversary simultaneously.

The predictor attempts to make accurate predictions, while the adversary attempts to recover a sensitive attribute (gender in this work) from the predictor's output.

The predictor loss is:

L_pred = BCE(y, ŷ) − α · BCE(z, ẑ)

where:

* y = target label
* ŷ = predicted label
* z = sensitive attribute
* ẑ = adversary prediction
* α = adversarial weighting coefficient

In the original framework, α remains constant throughout training.

Adaptive Alpha Control replaces this fixed value with a dynamic controller that updates α after every epoch based on observed fairness and performance trends.

---

# Key Features

* Dynamic adversarial weighting
* Trend-aware fairness control
* Exponential Moving Average (EMA) smoothing
* Accuracy protection via safety floor
* Multi-seed evaluation
* Full Weights & Biases logging
* Reproduction of Zhang et al. (2018) architecture

---

# Model Architecture

## Predictor

Input (99 features)

→ Dense(64, ReLU)

→ Dense(32, ReLU)

→ Dense(1, Sigmoid)

## Adversary

Prediction Probability ŷ

→ Dense(32, ReLU)

→ Dense(1, Sigmoid)

The adversary only observes the predictor output, preventing trivial recovery of sensitive information from raw inputs.

---

# Adaptive Alpha Controller

After every epoch:

1. Evaluate ACC, DEO, and DAO.
2. Compute metric changes.
3. Smooth changes using exponential moving averages.
4. Compute a pressure signal.
5. Update α for the next epoch.

Pressure signal:

pressure =
(W_DEO × EMA(ΔDEO))
+
(W_DAO × EMA(ΔDAO))
-------------------

(W_ACC × EMA(ΔACC))

Alpha update:

α ← clip(
α + α_lr × pressure,
α_min,
α_max
)

If accuracy falls below the configured floor, fairness pressure is overridden to protect predictive performance.

---

# Fairness Metrics

## Accuracy (ACC)

Overall prediction accuracy.

Higher is better.

---

## Difference in Equal Opportunity (DEO)

Measures disparity in true positive rates:

DEO = |TPR₀ − TPR₁|

Lower is better.

---

## Difference in Adversarial Outcomes (DAO)

Measures disparity in both true positive and false positive rates:

DAO =
(|TPR₀ − TPR₁| + |FPR₀ − FPR₁|) / 2

Lower is better.

---

# Dataset

UCI Adult Income Dataset

Task:

Predict whether annual income exceeds $50,000.

Sensitive Attribute:

Gender

After preprocessing:

* ~45k records
* One-hot encoded categorical features
* Standardised numerical features
* 99 predictor features

---

# Hyperparameters

| Parameter           | Value        |
| ------------------- | ------------ |
| Epochs              | 30           |
| Batch Size          | 256          |
| Learning Rate       | 1e-3         |
| Alpha Initial       | 0.4708055928 |
| Alpha Minimum       | 0.01         |
| Alpha Maximum       | 1.0048358312 |
| Alpha Learning Rate | 0.4468379573 |
| Accuracy Floor      | 0.84         |
| EMA Smoothing       | 0.3601034154 |
| W_DEO               | 0.6689005575 |
| W_DAO               | 1.9288807993 |
| W_ACC               | 4.6453214453 |

---

# Experimental Setup

The repository evaluates:

### Base Model

No adversarial debiasing.

### Fixed Alpha

Adversarial debiasing with constant α.

### Dynamic Alpha

Adaptive Alpha Control with trend-aware updates.

Experiments are repeated across multiple random seeds for robustness.

---

# Results

Mean results reported in the accompanying paper:

| Model         | ACC ↑  | DEO ↓  | DAO ↓  |
| ------------- | ------ | ------ | ------ |
| Base Model    | 0.8450 | 0.0789 | 0.0811 |
| Fixed Alpha   | 0.8457 | 0.0670 | 0.0730 |
| Dynamic Alpha | 0.8442 | 0.0284 | 0.0481 |

Adaptive Alpha Control achieves substantially lower bias while maintaining comparable predictive accuracy.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/lewis-shearer/Adaptive-Alpha-Control.git
cd Adaptive-Alpha-Control
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

* Python 3.10+
* TensorFlow
* NumPy
* Pandas
* Scikit-learn
* Weights & Biases

Example:

```bash
pip install tensorflow numpy pandas scikit-learn wandb
```

---

# Running Experiments

Ensure the Adult dataset is located at:

```text
data/adult.tsv
```

Run the seed sweep:

```bash
python adaptive_alpha_sweep.py
```

The script:

* Trains 30 seeds
* Logs every run to Weights & Biases
* Records fairness metrics
* Tracks alpha trajectories
* Saves complete experiment history

---

# Weights & Biases Logging

Each run logs:

* ACC
* DEO
* DAO
* Alpha
* Pressure
* EMA ΔDEO
* EMA ΔDAO
* EMA ΔACC
* Predictor loss
* Adversary loss

Run naming convention:

```text
dynamic_adult_gender_<seed>
```

Example:

```text
dynamic_adult_gender_0
dynamic_adult_gender_1
dynamic_adult_gender_2
...
```

---

# Repository Structure

```text
Adaptive-Alpha-Control/
│
├── data/
│   └── adult.tsv
│
├── adaptive_alpha_sweep.py
│
├── paper/
│   └── Adaptive_Alpha_Control.pdf
│
├── README.md
│
└── requirements.txt
```

---

# Ethical Considerations

Adaptive Alpha Control improves measured fairness metrics but does not guarantee fairness in an absolute sense.

Important limitations include:

* Binary treatment of gender
* Dependence on historical census data
* Metric-specific fairness definitions
* Potential proxy discrimination
* Fairness–accuracy trade-offs encoded through controller weights

The system should be viewed as a tool for bias mitigation rather than a complete fairness solution.


---

# Acknowledgements

This project builds upon:

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018).

*Mitigating Unwanted Biases with Adversarial Learning.*

Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society.

---


