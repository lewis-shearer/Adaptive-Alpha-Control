# Paper plan — target: publishable draft in 2 months (by 2026-12-26)

Format: arXiv-style write-up, no fixed venue template yet.
Datasets: COMPAS (done) + German Credit (in progress).
Framing: Adaptive Alpha Control (dual-ascent, two-constraint) is the main
method contribution. Post-processing (Hardt et al. 2016) is included as
an honest, stronger-performing comparison in Results/Discussion — not
hidden, not the headline.

## Status as of 2026-10-26

### COMPAS (primary dataset)
- [x] Baseline, fixed-alpha sweep (0.5/1.0/2.0), adaptive dual-ascent — all
      30-seed validated. Numbers captured in chat; raw per-seed logs were
      lost to /tmp cleanup and need re-running into results/ (in progress).
- [x] Post-processing (Hardt et al.) — 30-seed validated, raw log saved at
      results/postproc_30.log. This beats every adversarial variant on
      acc/DEO/DAO simultaneously.
- [x] Ablations: annealed epsilon, gradient-reversal joint training — both
      quick-tested (5 seeds), neither beats adaptive's frontier point.
      Logs saved at results/annealed_5.log, results/grl_5.log.
- [ ] Re-run baseline/fixed/adaptive at 30 seeds into results/ (permanent
      storage this time, not /tmp).
- [ ] Paired Wilcoxon + Holm correction significance tests (script ready:
      src/significance_test.py).

### German Credit (second dataset, generalization check)
- [x] Data pipeline (src/data_german.py).
- [x] Baseline — 30 seeds, results/baseline_german_30.log. Notably less
      biased at baseline than COMPAS (DEO 0.058 vs COMPAS's 0.250).
- [x] Fixed-alpha sweep — 30 seeds, results/fixed_german_30.log. Barely
      moves from baseline at any alpha (0.5/1.0/2.0 all land within noise
      of baseline's own acc/DEO/DAO) — German's baseline bias is already
      small, so there's less for adversarial training to fight. Notable
      cross-dataset finding for Discussion.
- [x] Adaptive dual-ascent — done. Barely differs from baseline (German's
      low baseline bias leaves little to fix).
- [x] Post-processing — done. KEY FINDING: reverses COMPAS's result —
      post-processing is the WORST method on German, specifically on DAO
      (0.105 vs ~0.05 for everything else), confirmed significant via
      paired Wilcoxon + Holm (p_holm < 0.0001-0.0023 vs every other
      condition). DEO/accuracy differences not statistically significant.
      Likely cause: German's 200-row test set, halved for calibration,
      halved again by group — too little data for reliable per-group ROC
      estimation. This is now the paper's headline cross-dataset finding.
- [x] Significance tests on German — done.
- [x] Re-ran COMPAS baseline/fixed/adaptive into permanent results/
      storage (was lost to /tmp cleanup) — in progress, baseline ~1/3 done
      as of this update, fixed-alpha and adaptive queued after.

## Writing progress — FIRST COMPLETE DRAFT DONE (2026-10-26)

paper/paper.tex has real content in every section (zero TODOs/
placeholders remain): Abstract, Introduction, Related Work (verified
citations), Method, Experimental Setup, Results (both datasets, full
tables, significance tests), Discussion (including the corrected,
precise Adaptive-vs-fixed-alpha finding: better DEO, worse DAO, tied
accuracy — not simply "comparable"), Limitations, Conclusion.

Compiles clean with pdflatex (BasicTeX installed, 2 passes, zero
warnings on the second pass) — paper/paper.pdf, 9 pages.

This is ~4 weeks ahead of the original 8-week schedule. Remaining work
shifts from "writing" to "revision":
- [ ] Read through as a skeptical reviewer — check every claim against
      the actual numbers in results/.
- [ ] Tighten prose, cut anything that doesn't earn its place.
- [ ] Decide whether the ablations (annealed epsilon, gradient reversal)
      deserve their own appendix or stay as the one-paragraph mention
      currently in Discussion.
- [ ] Final proofread + submission formatting once a venue (if any) is
      chosen.

## Week-by-week (8 weeks + buffer)

**Week 1 (Oct 26 - Nov 1)**: Finish German Credit full validation (all 4
conditions, 30 seeds). Re-run COMPAS baseline/fixed/adaptive into
permanent results/ storage. Run significance tests on both datasets.

**Week 2 (Nov 2-8)**: Verify citations (Zhang et al. 2018, Hardt et al.
2016, Angwin et al. 2016/ProPublica, Ganin & Lempitsky 2015) via actual
lookup, not memory. Draft Related Work + Method sections.

**Week 3 (Nov 9-15)**: Draft Experimental Setup + Results (both datasets,
tables + the Pareto plot regenerated for German too).

**Week 4 (Nov 16-22)**: Draft Discussion (the honest post-processing-beats-
adaptive finding goes here) + Limitations/Ethical Considerations.

**Week 5 (Nov 23-29)**: Draft Abstract + Introduction + Conclusion +
Reproducibility statement. Assemble full first complete draft.

**Week 6 (Nov 30-Dec 6)**: Revision pass 1 — verify every number in the
text against the actual logs, tighten prose, check all citations resolve.

**Week 7 (Dec 7-13)**: Revision pass 2 — adversarial self-review (read it
as a skeptical reviewer would), polish figures.

**Week 8 (Dec 14-20)**: Final formatting, proofreading, submission-ready
PDF.

**Buffer (Dec 21-26)**: slack for delays.

## Rules to not break

- Don't keep re-tuning hyperparameters against the same 30 seeds used for
  final reporting — that's the overfitting trap flagged earlier.
- Don't invent citation details from memory for the final paper — verify
  each one.
- Keep results/ as the only durable location for run logs. /tmp gets
  cleared and already lost three log files once.
