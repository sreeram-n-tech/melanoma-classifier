# Leakage-ablation review

Read-only assessment of the honest 5-fold harness (`run_all.py`,
`experiments/run_kfold_honest.py`, `finalize.py`, `model/evaluate.py`) and the
results in `model_output/`. No code changed. Prioritized: interpretation first,
then method, then the unfinished steps, then minor notes.

## Verdict up front

The harness mechanics are **correct**. The *conclusion the report would print is
not*. The recomputed paired deltas (masked − baseline):

| fold | Δspec | Δsens | Δauc | drift(|Δsens|>0.02) |
|------|-------|-------|------|------|
| 0 | −0.0219 | +0.0137 | +0.0020 | no |
| 1 | **−0.0929** | **+0.0235** | +0.0121 | **YES** |
| 2 | −0.0061 | −0.0088 | −0.0187 | no |
| 3 | +0.0481 | −0.0144 | +0.0105 | no |
| 4 | −0.0023 | −0.0122 | −0.0001 | no |

- **Mean Δspec (all 5): −0.0150 ± 0.0455** → ~0.33 pooled-std. Within noise.
- **Mean Δspec (drift-guarded, excl. fold1): +0.0044** → masking marginally *better*.
- **Mean Δauc: +0.0012** → flat / slightly up.

The −0.015 headline is produced by **one fold (fold1)** that the study's *own*
drift guard flags as confounded. Remove it and the effect flips sign to ~0. AUC
does not drop. **There is no detectable leakage at this training scale** — a
legitimate, honest negative result. It must not be reported as "leakage quantified."

---

## P1 — Report verdict keys on the wrong number (misleading conclusion)

`run_all.py:474-481`, `write_report`. The verdict branches on
`d_all = mean_d_spec_all` (the all-folds mean). With `d_all=-0.015 < -0.01` it
prints *"masking caused a specificity drop … the baseline exploited
artifact→melanoma shortcuts (leakage quantified)."* That statement is false here:
the drop is within noise and is driven by the drifted fold.

**Why it matters:** this is the single sentence a reader takes away. It asserts a
causal shortcut the data does not support.

**Change:** drive the verdict from the **drift-guarded** estimate
(`mean_d_spec_clean`, currently computed at `run_all.py:353` but unused in the
verdict) **and** gate it on significance vs pooled std. Concretely: if
`abs(mean_d_spec_clean) < std_d_spec_all` → print "no detectable effect (within
fold noise)". Only claim a direction when the clean mean exceeds the pooled std
*and* Δauc moves the same way. Also print both numbers (all vs drift-guarded) so
the fold1 dependence is visible.

## P2 — spec@95 is compared at *unequal achieved sensitivities* (confound)

`run_all.py:242-247` (calibrate) + `evaluate_fold` scoring. The threshold is
frozen on inner_val at 95% sens, then applied to held_out. Achieved held_out sens
is not 95% and differs per fold **and per arm** (baseline 0.915–0.976; masked
0.902–0.962). Specificity is the free variable on the ROC, so comparing spec
across two arms sitting at *different* sensitivities mixes the artifact effect
with calibration-transfer error. fold1 is the textbook case: masked bought
+2.35% sens and "paid" −9.3% spec — that is the sens/spec trade-off, not leakage.

**Why it matters:** the primary metric is partly measuring where each arm landed
on its ROC, not the arms' quality difference.

**Change (either):**
1. Make AUC (threshold-free) the **primary** metric; spec@95 secondary. AUC here
   is flat, which is the cleanest read. Low effort — numbers already computed.
2. Or compare spec at a *common held_out* sensitivity: interpolate each arm's
   held_out ROC to exactly 0.95 sens and read spec there (`roc_curve` is already
   imported via `tune_threshold`). This removes the drift-guard hack entirely.

The τ=0.02 drift guard (`run_all.py:47,327`) is a blunt post-hoc patch for this;
option 2 makes it unnecessary.

## P3 — Grad-CAM never ran; report.md / paired_comparison.json were never written

`run_all.py:612-618`, `main`. The Unicode `Δ` crash fired inside
`paired_comparison` (log ends at masked fold4 eval, 2026-07-01 23:07), which is
*before* `run_gradcam` (615) and `write_report` (618). Confirmed on disk:
`gradcam_mechanistic/` absent, `report.md` absent, `model_output/paired_comparison.json`
absent.

**So the answer to "what did Grad-CAM show" is: it has not run.** No mechanistic
evidence exists yet. The Unicode fix is already in `run_all.py` (delta/+-,
utf-8 log open, stdout reconfigure). Re-running `main()` now skips both arms
(summary JSONs present) and executes only paired → gradcam → report. Nothing to
change for this beyond running the tail.

## P4 — N=5, correlated folds, no CI → don't state directional claims

`run_all.py:317-361`, `paired_comparison`. Reports mean ± pstdev only. The 5
StratifiedGroupKFold folds share overlapping training data, so they are **not
independent**; the ±std understates correlation and cannot support a significance
claim. Effect size is ~0.3σ regardless.

**Change:** report the paired spread as descriptive (state "descriptive, folds
not independent"), add a paired 95% CI or a sign summary, and keep the wording
non-directional. This is presentation, not new math.

---

## Minor / correctness-clean, document only

- **T fit on single-view, threshold on TTA** (`run_all.py:242-247`). Deliberate
  and *defensible* — fitting temperature on single-view avoids TTA leaking into
  calibration. Consequence: the TTA probs aren't strictly NLL-calibrated, but T
  and threshold are applied consistently to TTA probs on both inner_val and
  held_out, so the operating point transfers and AUC is unaffected. No bug; add a
  one-line comment so it isn't "fixed" later by mistake.
- **`>=` threshold semantics** (`finalize.py metrics` vs `evaluate.py tune_threshold`).
  `roc_curve` operating points correspond to `score >= threshold`, which is what
  `metrics` applies. Consistent. No off-by-one.
- **Harness plumbing is sound:** both arms use the *same* `fold{i}.json` splits →
  the comparison is properly paired; `best_model.pt` selected by inner_val AUC in
  both arms; `collect_logits` averages in logit space on one code path for
  calibration/threshold/scoring (no train/serve skew). Sensitivity abort < 0.90
  guard is active and never tripped.

## Suggested order to apply

1. P3 — run the fixed tail (gradcam + report). Gets the missing artifacts + lets
   you see the 6 FN CAMs before finalizing wording.
2. P1 — rewrite the verdict logic around the drift-guarded, significance-gated
   number.
3. P2 option 1 — promote AUC to primary in the report table/prose (cheap). P2
   option 2 (common-sensitivity spec) only if you want to keep spec as primary.
4. P4 — soften stats language / add CI.

Bottom line for the writeup: **honest negative result — masking ruler+vignette
artifacts did not change discrimination (ΔAUC ≈ 0) or specificity beyond fold
noise; the apparent −0.015 spec drop is a single drifted fold.**
