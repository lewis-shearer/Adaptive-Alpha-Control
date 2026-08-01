# Adaptive Alpha Control

**Adaptive Alpha Control: A Trend-Aware Dynamic Weighting Mechanism for Stable Adversarial Debiasing**

An extension of the adversarial debiasing framework proposed by Zhang et al. (2018) that replaces a fixed adversarial weight (`α`) with a dynamic controller driven by fairness and accuracy trends during training.

The system continuously adjusts adversarial pressure using changes in:

* Accuracy (ACC)
* Difference in Equal Opportunity (DEO)
* Difference in Average Odds (DAO)

The objective is to improve fairness while maintaining predictive performance, avoiding the instability and manual tuning associated with fixed adversarial weights.

---

# Overview

Traditional adversarial debiasing trains a predictor and an adversary simultaneously.

The predictor attempts to predict income from census features, while the adversary attempts to recover a sensitive attribute (gender) from the predictor's output probability alone (not from the raw features). This forces the predictor to produce outputs that don't leak gender information if it wants to fool the adversary.

The predictor loss is:

```
L_pred = BCE(y, ŷ) − α · BCE(z, ẑ)
```

where:

* y = target label (income > $50K)
* ŷ = predicted label
* z = sensitive attribute (gender)
* ẑ = adversary's prediction of z from ŷ
* α = adversarial weighting coefficient

Training alternates each batch: the adversary is updated to get *better* at recovering gender from `ŷ`, then the predictor is updated to minimise its own prediction loss while *maximising* the adversary's loss (via the `−α · BCE(z, ẑ)` term) — i.e. trying to blind the adversary.

In Zhang et al.'s original framework, `α` is a constant chosen up front. This repository implements and compares three variants of this setup, sharing the same architecture, data pipeline, and metrics:

| Script | What it does |
| --- | --- |
| [`baseline.py`](baseline.py) | Predictor only, no adversary, no debiasing. Establishes the accuracy/fairness of the model with no intervention. |
| [`fixed.py`](fixed.py) | Adversarial debiasing with a **constant** α, swept across 8 values, each run for multiple seeds — traces out an accuracy-vs-fairness tradeoff curve to compare against. |
| [`dynamic.py`](dynamic.py) | Adaptive Alpha Control — α is recomputed every epoch from the trend in ACC/DEO/DAO. |

---

# Why a fixed-alpha *sweep* instead of one fixed value

Comparing an adaptive controller against a single, arbitrarily-chosen fixed α is a weak baseline — if that one value happens to be badly tuned, the adaptive method "wins" for the wrong reason.

`fixed.py` instead trains the fixed-alpha model across a range of `ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4708, 0.7, 1.0, 1.5]` (roughly an order of magnitude either side of the dynamic controller's initial α), each for multiple seeds. This produces a Pareto frontier of ACC vs DEO/DAO. The dynamic controller's result is then judged against that whole curve, not one point on it — if it sits on or beyond the frontier, that's a materially stronger claim than beating a single hand-picked number.

Because this sweep is 8× the training cost of a single run, `fixed.py` uses fewer seeds per alpha (`SEEDS_PER_ALPHA`) than the 30 used in `dynamic.py`; that constant can be raised if compute allows.

---

# Model Architecture

Identical predictor and adversary are used across all three scripts, so comparisons aren't confounded by architecture differences.

## Predictor

Input (99 features)

→ Dense(64, ReLU)

→ Dense(32, ReLU)

→ Dense(1, Sigmoid)

## Adversary

Input: predictor output ŷ (a single probability, not the raw features)

→ Dense(32, ReLU)

→ Dense(1, Sigmoid)

Because the adversary only ever sees `ŷ`, it can only recover gender if the predictor's output itself carries a signal correlated with gender — it has no access to the raw features to find that correlation elsewhere.

---

# Adaptive Alpha Controller (`dynamic.py`)

After every epoch:

1. Evaluate ACC, DEO, and DAO on the held-out test set.
2. Compute the change in each metric since the previous epoch (`Δ`).
3. Smooth each `Δ` with an exponential moving average (EMA) — this stops the controller from over-reacting to noisy epoch-to-epoch swings.
4. Combine the smoothed deltas into a single pressure signal.
5. Update α for the next epoch from that pressure.

Pressure signal:

```
pressure = (W_DEO × EMA(ΔDEO)) + (W_DAO × EMA(ΔDAO)) − (W_ACC × EMA(ΔACC))
```

Intuition: if bias (DEO/DAO) is trending upward, pressure goes positive and α increases next epoch, weighting the adversary term more heavily. If accuracy is improving, that pulls pressure back down, damping the adversarial push so accuracy isn't sacrificed for a fairness gain that's already happening on its own.

Alpha update:

```
α ← clip(α + α_lr × pressure, α_min, α_max)
```

**Accuracy floor override:** if accuracy on the current epoch drops below `ACC_FLOOR` (0.84), the controller ignores the computed pressure and forces `pressure = min(pressure, -0.2)` — i.e. it unconditionally pushes α down to relieve pressure on the predictor and recover accuracy, regardless of what the fairness trend says.

---

# Fairness Metrics

All three scripts compute these identically, from the test-set predictions at a 0.5 threshold.

## Accuracy (ACC)

Overall prediction accuracy. Higher is better.

## Difference in Equal Opportunity (DEO)

Disparity in true positive rate between gender groups:

```
DEO = |TPR₀ − TPR₁|
```

Lower is better.

## Difference in Average Odds (DAO)

Average of the TPR and FPR disparities between gender groups:

```
DAO = (|TPR₀ − TPR₁| + |FPR₀ − FPR₁|) / 2
```

Lower is better.

---

# Dataset

**UCI Adult Income Dataset** (`data/adult.tsv`)

Task: predict whether annual income exceeds $50,000.

Sensitive attribute: gender (binary in this dataset — see Ethical Considerations).

Preprocessing (shared across all three scripts):

* Drop rows with missing values
* Strip whitespace from string columns
* One-hot encode categorical columns (workclass, education, marital-status, occupation, relationship, race, native-country)
* Standardise numeric columns
* 80/20 train/test split, stratified on income, `random_state=42`
* ~45k records, 99 predictor features after encoding

`data/German.tsv` (German Credit dataset) is also present but not currently used by any script — it's there for possible future extension to a second dataset/sensitive-attribute setting, not part of the current experiments.

---

# Hyperparameters (`dynamic.py`)

Found through hyperparameter tuning with W&B sweeps.

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

`fixed.py` reuses the same epochs/batch-size/LR, but sweeps α itself rather than using a controller (see `ALPHA_VALUES` above). `baseline.py` reuses the same epochs/batch-size/LR with no α and no adversary at all.

---

# Experimental Setup

All three scripts now share identical data loading, preprocessing, `build_predictor`, `compute_metrics`, the same 30-seed sweep (seeds 0–29), the same batch/epoch loop structure, and the same W&B project — so the only thing that varies between them is how (or whether) α and the adversary are used. This makes the three-way comparison a valid ablation rather than three independently-configured experiments.

| Script | Seeds | W&B project | W&B group |
| --- | --- | --- | --- |
| `baseline.py` | 30 (seeds 0–29) | `final_results` | `Baseline` |
| `fixed.py` | 30 per alpha × 8 alphas | `final_results` | `FixedAlpha_<alpha>` |
| `dynamic.py` | 30 (seeds 0–29) | `final_results` | `DynamicAlpha` |

Two things previously broke that comparability and have since been fixed:

* **Preprocessing bug**: `baseline.py` and `dynamic.py` used `df.select_dtypes(include="str")` to find categorical columns for whitespace-stripping. Pandas string columns are dtype `object`, not `str`, so that call silently matched nothing and the stripping never ran — while `fixed.py` already used the correct `include="object"`. All three scripts now use `include="object"`, so they train on identically-encoded features.
* **`baseline.py` scope**: it previously trained a single seed (123) into a separate W&B project (`FINAL`). It's now rewritten to loop over the same 30 seeds, log to the same project, and use the same `tf.GradientTape` training-step structure as the other two scripts (predictor-only, no adversary).

---

# Results

Mean ± std over 30 seeds per row, from W&B.

**These numbers predate the fixes above** — the `Base Model` row came from an older, differently-coded baseline script (single/legacy seed loop, buggy preprocessing), and `Dynamic Alpha` was run before the `include="object"` fix, so its feature encoding didn't match `fixed.py`'s. They're kept here as the last known values; all three scripts need to be rerun on the now-coherent code before this table is trustworthy for the paper.

| Model | ACC ↑ | DEO ↓ | DAO ↓ |
| --- | --- | --- | --- |
| Base Model | 0.8219 ± 0.0089 | 0.0720 ± 0.0211 | 0.0833 ± 0.0150 |
| Fixed Alpha (α = 0.05) | 0.8455 ± 0.0017 | 0.0746 ± 0.0272 | 0.0768 ± 0.0162 |
| Fixed Alpha (α = 0.1) | 0.8452 ± 0.0017 | 0.0684 ± 0.0218 | 0.0735 ± 0.0123 |
| Fixed Alpha (α = 0.2) | 0.8447 ± 0.0018 | 0.0446 ± 0.0271 | 0.0588 ± 0.0161 |
| Dynamic Alpha | 0.8450 ± 0.0019 | 0.0298 ± 0.0195 | 0.0483 ± 0.0115 |

The remaining `fixed.py` sweep values (0.3, 0.4708, 0.7, 1.0, 1.5) haven't been run yet either. Once everything is rerun, `fixed_alpha_sweep_results.csv` will hold per-(alpha, seed) metrics for the full tradeoff curve, and this table should be replaced with the fresh numbers.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/lewis-shearer/Adaptive-Alpha-Control.git
cd Adaptive-Alpha-Control
```

Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn wandb
```

Requirements:

* Python 3.10+
* TensorFlow
* NumPy
* Pandas
* Scikit-learn
* Weights & Biases (`wandb login` before running any script)

---

# Running Experiments

Ensure the Adult dataset is located at:

```text
data/adult.tsv
```

Run whichever variant you want directly:

```bash
python baseline.py   # no debiasing, 30 seeds
python fixed.py       # 8-value alpha sweep, 30 seeds each
python dynamic.py     # adaptive alpha controller, 30 seeds
```

`fixed.py` additionally writes a local summary to `fixed_alpha_sweep_results.csv` (per-alpha, per-seed final ACC/DEO/DAO) so the tradeoff curve can be rebuilt without depending on W&B access.

---

# Running the Fixed-Alpha Sweep Across Multiple Machines

`fixed.py` runs its whole `ALPHA_VALUES × SEEDS` grid sequentially in one process. [`fixed_sweep.py`](fixed_sweep.py) + [`sweep_fixed.yaml`](sweep_fixed.yaml) run the same grid through an actual **W&B Sweep**, so the work can be split across several machines (e.g. a MacBook plus Windows laptops) instead of one machine grinding through all of it.

How it works: `wandb agent` on each machine repeatedly asks the sweep controller for the next unclaimed `(alpha, seed)` pair, runs it, and asks for another — so all machines drain the same shared grid in parallel with no manual work-splitting and no chance of two machines training the same combination.

```bash
# once, on any one machine — creates the sweep and prints a SWEEP_ID
wandb sweep sweep_fixed.yaml

# on every machine you want contributing runs (repeat per machine)
wandb agent <entity>/<project>/<SWEEP_ID>
```

`sweep_fixed.yaml`'s grid only covers the alpha values not already produced by `fixed.py`'s manual runs (0.05, 0.1, 0.2). Edit the `alpha.values` list there if you want the full 8-value grid re-run through the sweep instead, for consistency.

Each sweep-assigned run still calls `build_predictor`/`build_adversary`/`compute_metrics` and logs `ACC`/`DEO`/`DAO`/`alpha`/`pred_loss`/`adv_loss` exactly like `fixed.py`, and is renamed to the same `fixed_alpha<alpha>_adult_gender_seed<seed>` convention — so results from the manual runs and the sweep runs are interchangeable in the same `final_results` project. There's no shared local CSV across machines though (`fixed.py`'s in-process `sweep_results` list doesn't exist here) — pull final numbers back afterward via a W&B export or the `wandb.Api()`.

---

# Weights & Biases Logging

`dynamic.py` logs, per epoch: `ACC`, `DEO`, `DAO`, `alpha`, `pressure`, `ema_ddeo`, `ema_ddao`, `ema_dacc`, `pred_loss`, `adv_loss`.

`fixed.py` logs, per epoch: `ACC`, `DEO`, `DAO`, `alpha`, `pred_loss`, `adv_loss`.

`baseline.py` logs, per epoch: `ACC`, `DEO`, `DAO`, `pred_loss`.

Run naming conventions:

```text
baseline_adult_gender_seed<seed>            # baseline.py
fixed_alpha<alpha>_adult_gender_seed<seed>  # fixed.py
dynamic_adult_gender_seed<seed>             # dynamic.py
```

---

# Repository Structure

```text
Adaptive-Alpha-Control/
│
├── data/
│   ├── adult.tsv        # used by all three scripts
│   └── German.tsv       # present, not currently used
│
├── baseline.py           # no debiasing
├── fixed.py               # fixed-alpha sweep (single-machine, manual loop)
├── fixed_sweep.py         # fixed-alpha sweep (W&B-Sweep-driven, one (alpha, seed) per run)
├── sweep_fixed.yaml       # grid config for fixed_sweep.py
├── dynamic.py             # adaptive alpha controller
│
└── README.md
```

---

# Ethical Considerations

Adaptive Alpha Control improves measured fairness metrics but does not guarantee fairness in an absolute sense.

Important limitations include:

* Binary treatment of gender, inherited from how the UCI Adult dataset encodes this attribute
* Dependence on historical census data
* Metric-specific fairness definitions (DEO/DAO don't capture every notion of fairness)
* Potential proxy discrimination through correlated features
* Fairness–accuracy trade-offs encoded through controller weights (`W_DEO`, `W_DAO`, `W_ACC`), which were themselves tuned via a hyperparameter sweep

The system should be viewed as a tool for bias mitigation rather than a complete fairness solution.

---

# Acknowledgements

This project builds upon:

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018).

*Mitigating Unwanted Biases with Adversarial Learning.*

Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society.
