# PLAN — EfficientNet-B3 Backbone Experiment (capacity hypothesis)

> Plan only. This file is the deliverable. **Do NOT code or train until approved.**

## Context

Baseline (EffNet-**B0**, 224px, `off` preprocessing, unfreeze19, 5+10 epochs, LR 1e-4,
seed 42) plateaus at **AUC ≈ 0.904 ± 0.007**. Two prior strict ablations were honest nulls
on AUC:
- **Artifact masking** — null on AUC.
- **Spatial augmentation** ([[spatial-aug-experiment]]) — ΔAUC −0.006 (neutral), though it
  did move attention onto lesions.

Both nulls point away from "artifacts / attention" as the ceiling and toward **capacity or
data**. This experiment tests the **capacity** leg: swap B0 → **B3** (~5M → ~12M params,
deeper/wider), everything else held as close to identical as the harness allows, and pair
against the existing baseline summary. **An honest null is a valid, pre-registered outcome.**

## Grounded facts that shape the design (verified in code)

- **`build_model` only knows B0 / mobilenet_v2** (`model/model.py:9-24`). Adding B3 is a
  ~6-line branch mirroring the B0 one: torchvision `efficientnet_b3` +
  `EfficientNet_B3_Weights.DEFAULT`, same `net.classifier[1]` (Dropout→Linear) and same
  `net.features` container. So **Grad-CAM (`net.features[-1]`), `freeze_backbone`, and
  `unfreeze_last_n_blocks` work on B3 unchanged.**
- **The eval + CAM pipeline is arch-agnostic already.** `train.py`'s `maybe_save_best` writes
  `"arch": args.arch` into `best_model.pt`; `evaluate_fold` and `run_gradcam` both do
  `build_model(ckpt.get("arch", "efficientnet_b0"), …)` (`run_all.py:233,400`). Passing
  `--arch efficientnet_b3` propagates through scoring **and** Grad-CAM with **no eval-code
  change — at 224px.**
- **`evaluate_fold` is 224-locked.** It builds loaders at `img_size=IMG_SIZE` (=224) on the
  224 `off` cache. B3@224 reuses it byte-for-byte; **B3@300 would require a new `_300` cache
  (`dataset.py --img_size 300`) AND resolution-plumbing into eval** — materially more work
  and more surface for train/serve skew. This is the core reason 224 goes first.
- **Train/val gap is already on disk.** `training_log.csv` logs per-epoch
  `train_loss,val_loss,val_auc` (`train.py:251,255`). The overfitting guard needs **no
  training change** — just read the CSV per fold.
- **The control already exists.** `model_output/kfold_honest_baseline_summary.json` (B0,
  same recipe/seed/splits). One new arm trains; no baseline retrain.
- **Two hard-abort guards already exist and are reused verbatim:** sens floor 0.90
  (`SENS_ABORT`, `evaluate_fold`) and NaN-loss (`_nan_in_log`). The B3 experiment adds **no
  new guard** (no preflight needed — no augmentation change).

## Design decision 1 — INPUT RESOLUTION

**Recommendation: run B3 at 224px first. Native-300 is an OPTIONAL second run, only if 224
shows promise.** Resolution is an explicit flag (`--img_size`, already exists).

| | B3 @ 224 (primary) | B3 @ 300 (optional 2nd) |
|---|---|---|
| Confound | Clean-*er*: same resolution as baseline → isolates capacity | Confounds capacity **with** resolution |
| Harness reuse | `evaluate_fold` + CAM unchanged; existing `off` 224 cache | Needs new `_300` cache **and** eval-resolution plumbing |
| Compute | ~baseline cost ×(B3/B0) | + larger images, ~1.8× the pixels |

**Tradeoff stated honestly in the report:** B3's *native* training resolution is 300; forcing
it to 224 slightly handicaps B3 (it was pretrained at 300). So B3@224 is the clean
**capacity-at-fixed-resolution** test; if it's neutral/negative, 300 disambiguates "capacity
doesn't help" from "B3 needs its native resolution." The report will state explicitly which
resolution was used and that **224 is the clean single-resolution comparison, 300 is not.**

## Design decision 2 — RECIPE (hold vs adapt)

B3 has more capacity than B0, so the B0-tuned recipe may over/under-fit B3. The tension:
**tune B3 → confounds arch with recipe; hold everything → B3 may lose because its recipe
wasn't retuned, not because capacity doesn't help.** Resolution:

**Primary arm holds the recipe identical** (a genuine one-knob swap: B0→B3), and the
**train/val-gap log tells us the failure mode** if B3 loses. A regularized B3 is a
*pre-registered conditional follow-up*, not auto-run.

| Knob | Baseline (B0) | B3 primary | Hold / adapt + justification |
|---|---|---|---|
| `--arch` | efficientnet_b0 | **efficientnet_b3** | **THE variable.** |
| `--img_size` | 224 | 224 | **Hold** (decision 1). |
| `--lr_stage2` | 1e-4 | 1e-4 | **Hold.** Cleanest comparison; 1e-4 is already conservative for fine-tuning. |
| `--lr_stage1` | 1e-3 | 1e-3 | **Hold.** Head-only warmup, arch-insensitive. |
| epochs (s1/s2) | 5 / 10 | 5 / 10 | **Hold.** Changing budget confounds arch with schedule; the gap log flags if B3 overfits inside 10. |
| `--unfreeze_blocks` | 19 | 19 | **Hold.** B0 and B3 both have 9 top-level `features` children, so `children[-19:]` = **the entire backbone unfrozen** for both → same "fully fine-tuned" semantics, apples-to-apples. |
| `--weight_decay` | 0 (Adam) | 0 (Adam) | **Hold for primary.** wd=1e-4 (AdamW) is the *conditional* overfit lever (see below), not in the primary. |
| `--batch_size` | 32 | 32 | **Hold**; drop to 16 **only on CUDA OOM** (B3@224 costs more VRAM). If dropped, document it — smaller batch perturbs LR dynamics. |
| seed / loss / preprocessing / splits / TTA / calibration | 42 / weighted_ce / off / honest_splits | identical | **Hold** — the honest protocol is untouched. |

**Honesty flag for the report (required):** even at 224 with an identical recipe, B0→B3
changes depth, width, and stochastic-depth internally, and the recipe was tuned on B0. So
this is an **architecture comparison, not a strict single-variable ablation** — stated
plainly in the report.

## Design decision 3 — OVERFITTING GUARD

B3 (~12M params) on ~10k images is overfit-prone. **Per-fold train/val gap logging, read
from the existing `training_log.csv` (no training-code change):**

- For each fold, at the best-val-AUC epoch record `train_loss`, `val_loss`, `val_auc`, and
  the **gap = val_loss − train_loss**; also flag if `val_auc` **peaked early then declined**
  (classic overfit signature) by comparing peak-epoch to final-epoch val_auc.
- **Flag a fold** when `gap > GAP_FLAG` (heuristic; default 0.15 — the baseline B0 gap sets
  the reference and is printed alongside) **or** val_auc dropped > 0.01 from its peak by the
  final epoch. `# ponytail: heuristic threshold, tune against the B0 baseline gap if noisy.`
- Flags are **reported, not aborting** (unless a fold trips the existing sens<0.90 abort).
  A blown-out gap is the signal that the *conditional* regularized-B3 follow-up
  (`--weight_decay 1e-4` + `--early_stop_patience 3`, or fewer stage-2 epochs) is warranted —
  written into the report as the recommended next step, **not launched automatically.**

## Success / failure criteria (declared up front)

- **Primary — paired 5-fold ΔAUC (B3 − baseline):**
  - *Success:* mean ΔAUC > pooled SD **and** all folds hold sens ≥ 0.90.
  - *Neutral:* |mean ΔAUC| < pooled SD → reported honestly as no detectable effect.
  - *Failure:* mean ΔAUC < −pooled SD, or any fold's held-out sens < 0.90 (hard abort).
- **Secondary:** spec@95 (frozen-threshold specificity) and center-energy on the 6 tracked
  FN IDs + 3 control TPs (before = B0 `honest_baseline_fold{N}`, after = B3
  `honest_backbone_fold{N}`, same `off` images). Neutral secondary is a valid outcome.
- **Power limit, stated in the report:** the 5 StratifiedGroupKFold folds share overlapping
  training data → not independent; baseline `std_auc` ≈ 0.0074, so a genuine gain below
  **~0.01 AUC** sits inside fold noise and reads **neutral**. A neutral read at N=5 is a
  power ceiling, **not** proof capacity can't help. CIs are descriptive (within-study), not
  population inference. **Do not tune toward a target.**

## Standalone detached runner: `experiments/run_backbone.py`

Modeled on `run_augment.py` (import `run_all` helpers, don't fork them): idempotent,
resumable, no Claude in the loop. **No preflight** (no augmentation change → nothing to
gate). Sequence:

1. **Pairing-integrity assert** — reuse the `run_augment` pattern: abort unless each fold's
   `n_inner_val`/`n_held_out` in `kfold_honest_baseline_summary.json` match the current
   `honest_splits/fold*.json` counts (proves the baseline was scored on the same splits).
2. **Train arm `backbone`, 5 folds** — own subprocess cmd (can't reuse `run_all.train_fold`;
   it hardcodes `--arch efficientnet_b0`). Exact baseline command **with `--arch
   efficientnet_b3`** (+ `--img_size` from a runner flag, default 224); run_name
   `honest_backbone_fold{N}`; skip if complete (`is_training_complete`), delete `best_model.pt`
   + `--resume` if partial; NaN-loss abort via `_nan_in_log`.
3. **Evaluate** each fold via `run_all.evaluate_fold(ckpt, fold, "off")` — unchanged; arch
   read from the checkpoint; sens<0.90 → abort. Write
   `model_output/kfold_honest_backbone_summary.json` (skip whole arm if it exists).
4. **Train/val-gap read** — parse each fold's `training_log.csv`, emit per-fold gap + flags.
5. **Paired comparison** vs the existing baseline summary (reuse `run_all.paired_comparison`
   + a t-based 95% CI on the per-fold ΔAUC list) → `model_output/paired_backbone_comparison.json`.
6. **Before/after Grad-CAM** on the 6 FN IDs + 3 TPs (B0 vs B3 checkpoint, same off-cache
   images) → `gradcam_backbone/` + `center_energy.json`.
7. **Write `report_backbone.md`** — AUC primary + CI, spec@95, Δcenter-energy table, the
   per-fold train/val-gap table with flags, the power-limit caveat, the
   "architecture-comparison-not-strict-ablation" honesty flag, the resolution note, and an
   honest verdict gated on ΔAUC vs pooled SD. Include the conditional regularized-B3 /
   native-300 next-step recommendation.

Run detached: `python experiments/run_backbone.py` (default 224). Optional 2nd:
`python experiments/run_backbone.py --img_size 300` **only after** the 300 cache is built —
the runner aborts with a clear message if `cache_dir_for("off", 300)` is missing.

## Files that change / get added

| File | Change |
|---|---|
| `model/model.py` | Add an `efficientnet_b3` branch to `build_model` (mirror B0: weights, `classifier[1]` reset, `backbone = net.features`) and add `"efficientnet_b3"` to `get_target_layer`'s arch tuple. |
| `model/train.py` | Add `"efficientnet_b3"` to the `--arch` `choices` list (line 334). No other change. |
| `experiments/run_backbone.py` | **New** standalone runner (above). Imports `evaluate_fold, is_training_complete, _csv_epoch_count, _nan_in_log, load_cached_bgr, paired_comparison, MODEL_OUTPUT, SPLIT_DIR, N_FOLDS, TOTAL_EPOCHS, TAU_SENS, FN_IDS` from `run_all`. |
| `PLAN_backbone.md` | **New** — this file. |

**No change** to loss, optimizer, seed, epochs, fold/grouping logic, the honest calibration
protocol, the augmentation path, or `run_all.py`'s locked summaries. **No checkpoints,
caches, or splits deleted.**

## Verification

- **Arch isolation:** diff B3 vs B0 `run_config.json` — only `arch` (and `img_size` iff the
  300 run) differs; recipe, seed, splits, preprocessing identical.
- **B3 loads + fine-tunes:** first fold reaches Stage 2 without shape/NaN error; checkpoint
  records `"arch": "efficientnet_b3"`, so eval/CAM pick it up automatically.
- **Splits untouched:** runner reads `honest_splits/fold*.json`, never writes them; baseline
  summary read-only.
- **Apples-to-apples pairing:** same fold jsons, seed, recipe, `off` eval cache (@224);
  comparison paired per fold; integrity assert passes.
- **Overfit guard fires:** per-fold gap table renders; a fold with `gap > GAP_FLAG` or a
  peak-then-decline val_auc is flagged in the report.
- **End-to-end read-out:** `kfold_honest_backbone_summary.json`,
  `paired_backbone_comparison.json`, `gradcam_backbone/` (+ center_energy.json),
  `report_backbone.md` (verdict + power caveat + honesty/resolution flags + gap table).
