# Adaptive Alpha Control

**A smarter way to balance fairness and accuracy when training a machine learning model.**

This project changes how "adversarial debiasing" works. Normally, this technique uses one fixed setting (called alpha, or `α`) to control how hard the model tries to be fair. Picking that number is a guessing game — too low and the model stays biased, too high and it loses accuracy.

Instead of guessing, this project builds a controller that watches the model's fairness and accuracy as it trains, and automatically adjusts alpha every epoch. The goal: get fairer predictions without giving up much accuracy, and without having to hand-tune a number in advance.

---

## What is "adversarial debiasing"?

Imagine two models training at the same time:

1. **The predictor** — tries to guess whether someone's income is above $50k, using their census data.
2. **The adversary** — tries to guess someone's gender, but it only gets to see the predictor's guess (not the original data).

If the adversary can guess gender just from the predictor's output, that means the predictor's guesses are leaking gender information — a sign of bias. So the predictor is trained to do two things at once: get the income prediction right, **and** make it hard for the adversary to guess gender from that prediction.

The knob that balances these two goals is `α` (alpha). A small alpha means the model barely cares about fooling the adversary. A large alpha means it cares a lot — sometimes too much, at the cost of accuracy.

This repo compares three ways of setting that knob:

| Script | What it does |
| --- | --- |
| [`baseline.py`](baseline.py) | No adversary at all. Just trains the predictor normally, so we have a "no fairness effort" reference point. |
| [`fixed.py`](fixed.py) | Uses a fixed alpha, but tries 8 different values (and multiple random seeds for each) so we get a full curve of fairness vs. accuracy trade-offs, not just one guess. |
| [`dynamic.py`](dynamic.py) | **Adaptive Alpha Control** — alpha changes automatically every epoch based on how fairness and accuracy are trending. |

---

## Why test a whole range of fixed alphas, not just one?

If we only compared the adaptive controller against one fixed alpha, and that one number happened to be a bad choice, the adaptive method would look better for the wrong reason.

So `fixed.py` runs the model across a spread of alpha values — `[0.05, 0.1, 0.2, 0.3, 0.4708, 0.7, 1.0, 1.5]` — each with several random seeds. Plotting all of these gives a curve of "best possible trade-offs" (a Pareto frontier). The adaptive controller is then judged against that entire curve, not a single point. If it lands on or beyond the curve, that's a much stronger result.

This sweep costs 8x the compute of a single run, so `fixed.py` uses fewer seeds per alpha than `dynamic.py` does. That number can be raised later if more compute is available.

---

## Model setup

The predictor and adversary are identical across all three scripts, so the comparison is fair — nothing else changes except how alpha is used.

**Predictor:** 99 input features → 64 neurons → 32 neurons → 1 output (income yes/no)

**Adversary:** takes only the predictor's single output number → 32 neurons → 1 output (guess at gender)

Because the adversary never sees the original data — only the predictor's final answer — it can only guess gender correctly if the predictor's answer itself is carrying a gender signal.

---

## How the adaptive controller works (`dynamic.py`)

At the end of every training epoch, the controller:

1. Measures accuracy and two fairness scores (DEO and DAO — explained below) on held-out test data.
2. Looks at how much each of these changed since last epoch.
3. Smooths out noisy jumps using a moving average, so one weird epoch doesn't cause an overreaction.
4. Combines these into a single "pressure" number.
5. Uses that pressure to raise or lower alpha for the next epoch.

In plain terms: if bias is creeping up, pressure rises and alpha goes up next epoch, leaning harder on the adversary. If accuracy is improving on its own, that pulls pressure back down, so the model doesn't sacrifice accuracy it didn't need to.

**Safety net:** if accuracy ever drops below 0.84, the controller ignores everything else and forces alpha down, to protect accuracy first.

---

## Fairness metrics, explained simply

All three scripts measure the same things, on the same held-out test data:

- **ACC (Accuracy)** — how often the model gets the income prediction right. Higher is better.
- **DEO (Difference in Equal Opportunity)** — among people who actually earn over $50k, does the model catch them at the same rate regardless of gender? Lower is better (0 = perfectly equal).
- **DAO (Difference in Average Odds)** — a broader version of DEO that also checks false positives, not just true positives. Lower is better.

---

## Results

Charts below are averaged over many training runs (shaded band = spread across seeds), comparing:

- 🟢 **Zhang2018_AdaptiveAlpha_EMA** — the adaptive controller (this project's method)
- 🔴 **Baseline_NoDebiasing** — no fairness effort at all
- 🔵 **Fixed-alpha runs** — the fixed-alpha sweep average

### Accuracy over training

![Accuracy over training](results/acc_over_epochs.png)

Accuracy is close across all three — the adaptive controller trades only a small amount of accuracy compared to the baseline.

### Equal Opportunity gap over training (lower is better)

![DEO over training](results/deo_over_epochs.png)

The adaptive controller (green) keeps this gap far lower than the baseline (red) and the fixed-alpha runs (blue) throughout training.

### Average Odds gap over training (lower is better)

![DAO over training](results/dao_over_epochs.png)

Same pattern — the adaptive controller stays noticeably more fair throughout training, not just at the end.

### Accuracy vs. fairness trade-off (Pareto frontier)

![Pareto frontier](pareto_frontier.png)

This plots every fixed-alpha run against the adaptive controller, so you can see the whole trade-off curve at once, not just one point on it.

### Final numbers (mean ± std over 30 runs each)

> ⚠️ These numbers are from an earlier version of the code, before a preprocessing bug was fixed (see [Experimental Setup](#experimental-setup)). They're kept here as the last known values — all three scripts need a fresh run on the current code before this table should be treated as final.

| Model | ACC ↑ | DEO ↓ | DAO ↓ |
| --- | --- | --- | --- |
| Baseline (no debiasing) | 0.8219 ± 0.0089 | 0.0720 ± 0.0211 | 0.0833 ± 0.0150 |
| Fixed Alpha (α = 0.05) | 0.8455 ± 0.0017 | 0.0746 ± 0.0272 | 0.0768 ± 0.0162 |
| Fixed Alpha (α = 0.1) | 0.8452 ± 0.0017 | 0.0684 ± 0.0218 | 0.0735 ± 0.0123 |
| Fixed Alpha (α = 0.2) | 0.8447 ± 0.0018 | 0.0446 ± 0.0271 | 0.0588 ± 0.0161 |
| **Dynamic Alpha (adaptive)** | 0.8450 ± 0.0019 | **0.0298 ± 0.0195** | **0.0483 ± 0.0115** |

The remaining fixed-alpha values (0.3, 0.4708, 0.7, 1.0, 1.5) haven't been run yet. Once everything is re-run on the fixed code, `fixed_alpha_sweep_results.csv` will hold the full set of numbers and this table will be updated.

---

## Dataset

**UCI Adult Income Dataset** (`data/adult.tsv`)

- **Task:** predict whether someone's annual income is above $50,000.
- **Sensitive attribute:** gender (recorded as binary in this dataset — see [Ethical Considerations](#ethical-considerations)).

Before training, the data is cleaned up the same way in all three scripts:

- Rows with missing values are dropped
- Extra whitespace is stripped from text columns
- Categorical columns (workclass, education, marital status, occupation, relationship, race, native country) are one-hot encoded
- Numeric columns are standardised
- Split 80/20 into train/test, stratified by income, `random_state=42`
- ~45,000 records, 99 features per row after encoding

There's also a `data/German.tsv` (German Credit dataset) sitting in the repo — it isn't used by any script yet. It's there in case this project gets extended to a second dataset later.

---

## Hyperparameters (`dynamic.py`)

Found via a Weights & Biases hyperparameter sweep.

| Parameter | Value |
| --- | --- |
| Epochs | 30 |
| Batch Size | 256 |
| Learning Rate | 1e-3 |
| Alpha Initial | 0.4708055928 |
| Alpha Minimum | 0.01 |
| Alpha Maximum | 1.0048358312 |
| Alpha Learning Rate | 0.4468379573 |
| Accuracy Floor | 0.84 |
| EMA Smoothing | 0.3601034154 |
| W_DEO | 0.6689005575 |
| W_DAO | 1.9288807993 |
| W_ACC | 4.6453214453 |

`fixed.py` reuses the same epochs, batch size, and learning rate, but sweeps alpha itself instead of using a controller. `baseline.py` reuses the same settings too, minus alpha and the adversary entirely.

---

## Experimental Setup

All three scripts now share the same data loading, preprocessing, model-building code, metric calculations, the same 30 random seeds (0–29), and log to the same W&B project. The only thing that differs between them is how (or whether) alpha and the adversary are used — so this is a clean comparison, not three separately-configured experiments.

| Script | Seeds | W&B project | W&B group |
| --- | --- | --- | --- |
| `baseline.py` | 30 (seeds 0–29) | `final_results` | `Baseline` |
| `fixed.py` | 30 per alpha × 8 alphas | `final_results` | `FixedAlpha_<alpha>` |
| `dynamic.py` | 30 (seeds 0–29) | `final_results` | `DynamicAlpha` |

Two bugs used to break this comparison, and have since been fixed:

- **Preprocessing bug:** `baseline.py` and `dynamic.py` were looking for text columns using the wrong pandas dtype filter, so whitespace-stripping silently never ran. `fixed.py` already had it right. All three now use the same, correct filter.
- **`baseline.py` scope:** it used to train only one seed into a separate, older W&B project. It's now rewritten to run the same 30 seeds, log to the same project, and use the same training loop structure as the other two scripts.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/lewis-shearer/Adaptive-Alpha-Control.git
cd Adaptive-Alpha-Control
```

Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn wandb
```

You'll need:

- Python 3.10+
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- A Weights & Biases account (`wandb login` before running anything)

---

## Running Experiments

Make sure the dataset is here:

```text
data/adult.tsv
```

Then run whichever version you want:

```bash
python baseline.py   # no debiasing, 30 seeds
python fixed.py       # 8-value alpha sweep, 30 seeds each
python dynamic.py     # adaptive alpha controller, 30 seeds
```

`fixed.py` also saves a local summary to `fixed_alpha_sweep_results.csv`, so you can rebuild the trade-off curve without needing W&B access.

---

## Running the Fixed-Alpha Sweep Across Multiple Machines

`fixed.py` runs its entire grid of alphas and seeds one at a time on a single machine. If you'd rather split the work across several machines (say, a MacBook plus a couple of Windows laptops), use [`fixed_sweep.py`](fixed_sweep.py) with [`sweep_fixed.yaml`](sweep_fixed.yaml) instead — this runs the same grid as a proper **W&B Sweep**.

Each machine just asks W&B for the next unclaimed (alpha, seed) pair, runs it, and asks for another. So all your machines chip away at the same grid at once, with no manual splitting and no risk of two machines doing the same work twice.

```bash
# once, on any one machine — creates the sweep and prints a SWEEP_ID
wandb sweep sweep_fixed.yaml

# on every machine you want contributing runs (repeat per machine)
wandb agent <entity>/<project>/<SWEEP_ID>
```

By default, `sweep_fixed.yaml`'s grid only covers the alpha values not already produced manually (0.05, 0.1, 0.2). Edit the `alpha.values` list there if you want the full 8-value grid run this way instead.

Every run from the sweep logs the same metrics and follows the same naming convention as a manual `fixed.py` run, so results from both are interchangeable in the same `final_results` project. There's no shared local CSV across machines though — pull the final numbers back afterward via a W&B export or `wandb.Api()`.

---

## Weights & Biases Logging

- `dynamic.py` logs, per epoch: `ACC`, `DEO`, `DAO`, `alpha`, `pressure`, `ema_ddeo`, `ema_ddao`, `ema_dacc`, `pred_loss`, `adv_loss`.
- `fixed.py` logs, per epoch: `ACC`, `DEO`, `DAO`, `alpha`, `pred_loss`, `adv_loss`.
- `baseline.py` logs, per epoch: `ACC`, `DEO`, `DAO`, `pred_loss`.

Run naming:

```text
baseline_adult_gender_seed<seed>            # baseline.py
fixed_alpha<alpha>_adult_gender_seed<seed>  # fixed.py
dynamic_adult_gender_seed<seed>             # dynamic.py
```

---

## Repository Structure

```text
Adaptive-Alpha-Control/
│
├── data/
│   ├── adult.tsv        # used by all three scripts
│   └── German.tsv       # present, not currently used
│
├── results/               # chart images used in this README
│
├── baseline.py            # no debiasing
├── fixed.py               # fixed-alpha sweep (single machine)
├── fixed_sweep.py         # fixed-alpha sweep (W&B Sweep, multi-machine)
├── sweep_fixed.yaml       # grid config for fixed_sweep.py
├── dynamic.py             # adaptive alpha controller
├── analyze_results.py     # builds the Pareto frontier plot
├── pareto_frontier.png    # accuracy vs. fairness trade-off chart
│
└── README.md
```

---

## Ethical Considerations

Adaptive Alpha Control improves the fairness numbers it's measured on — it doesn't guarantee fairness in a broader sense. A few important limits:

- Gender is treated as binary here, because that's how the UCI Adult dataset records it
- The model learns from historical census data, which carries its own historical biases
- DEO and DAO are specific definitions of fairness — they don't capture every notion of what "fair" means
- Other features correlated with gender could still let bias leak through indirectly
- The trade-off between fairness and accuracy is shaped by controller settings (`W_DEO`, `W_DAO`, `W_ACC`), which were themselves chosen via a hyperparameter sweep, not first principles

This project is a tool for reducing bias — not a complete solution to fairness.

---

## Acknowledgements

This project builds on:

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). *Mitigating Unwanted Biases with Adversarial Learning.* Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society.
