# Experiment log

Validated core (don't touch): src/data.py, metrics.py, models.py,
train_baseline.py, train_fixed_alpha.py, train_adaptive_alpha.py.
30-seed results for these are in README-worthy form already — see the
table below as the baseline to beat.

## Current best (30 seeds each)

| condition | acc | DEO | DAO |
|---|---|---|---|
| Baseline | 0.656 ± 0.005 | 0.250 ± 0.014 | 0.209 ± 0.016 |
| Fixed a=2.0 (best fixed) | 0.616 ± 0.016 | 0.172 ± 0.029 | 0.139 ± 0.028 |
| Adaptive (dual ascent) | 0.609 ± 0.011 | 0.125 ± 0.025 | 0.144 ± 0.017 |

Goal: beat Adaptive's row on acc AND DEO AND DAO simultaneously, without
cheating by re-tuning against the same 30 seeds repeatedly (quick-test new
ideas at 5 seeds first; only spend a full 30-seed run on something that
already looks promising at 5).

## Experiment 1: Annealed epsilon

Idea: instead of fixed EPS_DEO/EPS_DAO targets from epoch 1 (which makes
alpha climb aggressively right away), start epsilon loose (= baseline's
own gap, so ~zero pressure early) and anneal it down linearly to the
existing target by the final epoch. Lets the predictor learn the task
first before fairness pressure ramps up, hopefully preserving more
accuracy for the same final DEO/DAO.

Status: DONE. 5-seed quick test result: acc 0.626±0.009, DEO 0.164±0.020,
DAO 0.151±0.024 — vs current best adaptive (0.609/0.125/0.144). Accuracy
genuinely improved, but DEO got worse and DAO is roughly tied. Does not
dominate the existing result — it's a different point on the same
frontier (trades DEO for accuracy), not a strictly better one. Not worth
a full 30-seed validation on its own; could be worth reporting in the
paper as a "what if you soften the start" ablation, not as the headline
method.

## Experiment 2: Gradient-reversal joint training

Idea: replace the two-tape alternating predictor/adversary updates with a
single joint backward pass, using a gradient-reversal layer (forward
pass-through, backward negates+scales the gradient) between predictor and
adversary. Tests whether the alternating-update implementation itself was
leaving anything on the table vs. a more standard domain-adversarial
training setup.

Status: DONE. 5-seed quick test at lam=0.5/1.0/2.0 reproduces the
alternating-training fixed-alpha numbers almost exactly (e.g. lam=2.0:
acc 0.613±0.007, DEO 0.165±0.028, DAO 0.130±0.029 vs fixed alpha=2.0's
0.616/0.172/0.139 at 30 seeds). Confirms the two training procedures are
equivalent — GRL isn't a worse or better implementation, just a
different-looking one. Does not beat adaptive. Not worth a 30-seed run.

## Experiment 3: Post-processing equalized-odds calibration (Hardt et al. 2016)

Idea: completely different family — no adversary, no training-time
intervention. Train a plain unconstrained predictor, then calibrate
per-group decision thresholds post-hoc (via randomized mixing between two
thresholds per group) to exactly satisfy equalized odds, chosen to
maximize accuracy. Calibrated on half the test set, evaluated on the
other half (held out from calibration) to avoid optimistic bias.

Status: DONE, 30-seed validated. acc 0.631±0.020, DEO 0.061±0.045, DAO
0.060±0.040 — beats every mitigation method tried (fixed-alpha sweep,
adaptive dual ascent) on all three axes simultaneously:
  - vs baseline: acc -3.8%, DEO -76%, DAO -71%
  - vs adaptive (dual ascent): acc +3.6%, DEO -51%, DAO -58%
This is the first technique that actually pushes the frontier outward
rather than sliding along it. Likely explanation: adversarial training
(experiments 1-2, and the original adaptive mechanism) has to fight an
adversary DURING training, which costs accuracy as a side effect of the
optimization process itself. Post-processing never touches training at
all — it trains a plain, maximally-accurate predictor first, then solves
a direct, convex calibration problem for the exact fairness definition
(equalized odds) being optimized. It's not fighting proxy signals through
noisy minimax dynamics, it's solving the actual target problem directly.

DEO/DAO stds are fairly large (0.045, 0.040) relative to their means —
worth flagging honestly in the paper as a property of this method (known
in the literature: randomized post-processing trades some outcome
variance for the accuracy/fairness gain), not hidden.

STOP HERE for new techniques — this is a strong, validated, genuinely
different result. Next step is writing it up, not chasing further
tweaks against these same 30 seeds.

## Cross-dataset finding (German Credit, 2026-10-26)

Post-processing's dominance on COMPAS does NOT replicate on German
Credit — there it's the WORST method (acc 0.711 vs ~0.72-0.73 for
everything else, DAO 0.105 vs ~0.05 for everything else, and roughly
2x the variance of every other method on every metric).

Likely explanation: German's test set is only 200 rows. Post-processing
calibration splits that in half (100/100) and further splits by
sensitive-attribute group, leaving ~30-70 examples to estimate each
group's ROC curve from — too little data for a reliable calibration.
Post-processing's COMPAS win likely depended on COMPAS's much larger test
set (~1000+ rows). Adversarial methods (fixed-alpha, adaptive) use the
full training set and don't have this weakness — consistent with them
staying stable across both datasets while post-processing doesn't.

This is the headline cross-dataset finding for the paper: the right
debiasing approach depends on how much held-out data is available for
calibration, not a universal ranking of methods.

CONFIRMED via paired Wilcoxon + Holm correction (src/significance_test.py):
  - DAO: post-processing vs adaptive p_holm=0.00097, vs fixed a=2.0
    p_holm=0.00085, vs baseline p_holm=0.0023 — all significant. This is
    the real, confirmed part of the reversal.
  - DEO and accuracy differences: NOT significant after correction (p_holm
    = 1 in every case) — directionally consistent with DAO's story but
    can't be claimed as a real effect at n=30 on this small a test set.
  - Adaptive vs baseline on German: no significant difference on any
    metric — consistent with German's low baseline bias giving the
    adversarial mechanism little to fix.

Precise claim for the paper: post-processing calibration becomes
unreliable specifically on the COMPOUND fairness metric (DAO, which
needs per-group FPR estimates on top of TPR) when the calibration set is
small — not a blanket "post-processing is worse on German."
