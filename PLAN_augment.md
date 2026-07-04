# PLAN — Spatial-Augmentation Ablation (break positional bias)

> Status: **plan only — not yet executed.** Nothing is trained until this plan is run via
> `experiments/run_augment.py`.

## Context

The Grad-CAM audit (center-energy metric, fraction of CAM energy inside the inner 70% of
the frame) showed a **positional/compositional bias**: the final model localizes lesions
it gets *right* (control TPs, center-energy 0.57–0.81) but attends to borders/blank skin
on its *false negatives* (0.17–0.44). Hypothesis: **stronger spatial augmentation
decorrelates lesion position from label**, moving attention onto the lesion and improving
discrimination. This is a single strict ablation: the ONLY change vs the current baseline
is the train-time spatial augmentation.

**Grounded facts that shape the design (verified in code):**
- The baseline is **not** augmentation-free. `HAM10000Dataset` (`model/dataset.py:228-241`)
  already applies `RandomResizedCrop(224, scale=(0.85,1.0))` + `RandomRotation(20)` + flips
  on the train split. The treatment *intensifies* spatial aug; the contrast must be clearly
  stronger than 0.85-scale crops or it reads neutral for lack of contrast.
- **The control already exists on disk.** `model_output/kfold_honest_baseline_summary.json`
  is the `off`-preprocessing, `unfreeze19`, 10-epoch, seed-42 arm trained *without* extra
  spatial aug, on the byte-identical `experiments/honest_splits/fold*.json`. So this
  experiment trains **one new arm** and pairs it against that summary — **no baseline retrain.**
- Augmentation is already train-time only (`augment=True` only for the train loader,
  `make_dataloaders`, `dataset.py:261-270`). Val/test read the resize-only path → "never
  touches val/test" is free.
- Two of the three hard-abort guards already exist and are reused verbatim: **sens floor
  0.90** (`SENS_ABORT`, `run_all.evaluate_fold`) and **NaN-loss** (`_nan_in_log`). Only the
  **lesion-in-frame preflight** is new.

## The change (chosen: Moderate, no translate)

Train-time transform on the treatment arm only, everything else identical:

```
RandomHorizontalFlip()
RandomVerticalFlip()
RandomResizedCrop(224, scale=(0.55,1.0), ratio=(0.8,1.25))   # was scale=(0.85,1.0)
RandomRotation(30)                                            # was 20
ColorJitter(0.2,0.2,0.1)                                      # UNCHANGED — spatial-only ablation
```

**Why these magnitudes / the tradeoff.** RandomResizedCrop crops *within* the image and
resizes to fill the frame → **zero padding artifact** (unlike RandomAffine translate, which
pads exposed borders black and would re-introduce the border cue we are trying to suppress).
Scale lower bound 0.55 → worst-case crop side ≈ √0.55 ≈ 0.74 of the frame, positioned
randomly, so a centrally-framed HAM dermoscopy lesion stays in view with margin while its
relative position and scale vary enough to break "border-hugging = melanoma."
- **Too aggressive** (e.g. scale 0.3–0.4): a small off-center crop can land on blank
  peri-lesional skin and exclude the lesion → an image still labeled `mel` with no lesion in
  it → label noise, worst on the minority class; also risks the 0.90 sens floor.
- **Too mild** (≈ baseline 0.85): indistinguishable from the control → the experiment reads
  neutral because there is no contrast, not because the hypothesis is false.

## Preflight lesion-in-frame gate (the new hard-abort, runs BEFORE any training)

No lesion masks exist in HAM10000, so the check combines an **automatic analytic proxy**
(the unattended abort) with a **visual grid** (catches what the proxy's assumption can't):

1. **Analytic retention proxy — the abort.** Monte-Carlo the *exact* crop distribution the
   training uses by calling `torchvision.transforms.RandomResizedCrop.get_params` with
   `scale=(0.55,1.0), ratio=(0.8,1.25)` (reuse torchvision's own sampler — no reimplementation).
   Draw ~2000 crop boxes; for each, test whether the **central disk** (radius 0.25·side,
   image-centered) is ≥70% inside the box. **Abort if <95% of sampled crops retain the
   central disk.** Assumption, documented in the artifact: the lesion lies within the central
   ~25%-radius region — true for curated HAM dermoscopy framing; eccentric lesions are the
   residual risk the visual grid surfaces. (Moderate/no-translate passes this comfortably.)
2. **Visual grid — the eyeball.** Render K=8 augmented samples for ~12 images (the 6 tracked
   FN IDs + a few random `mel`/`nv`) into `augment_qc/preflight_grid.jpg`. Written before
   training; both gates must pass. The grid is the human-checkable "crops don't remove the
   lesion" artifact.

Skip guard: if `augment_qc/preflight_PASS.txt` exists, the gate is skipped on resume.

## Harness reuse (nothing about the honest protocol changes)

- **Splits:** read `experiments/honest_splits/fold{0..4}.json` **as-is, byte-identical**.
  Not regenerated.
- **Eval:** reuse `run_all.evaluate_fold(ckpt, fold, "off")` unchanged — calibrate T on
  single-view inner_val, tune 95%-sens threshold on TTA inner_val, score held_out with TTA
  + frozen (T, threshold). The augment arm evaluates on the **`off` cache** (augmentation is
  train-time only), so eval images are identical to the baseline arm's.
- **Pairing:** reuse `run_all.paired_comparison(baseline_summary, augment_summary)` (per-fold
  Δauc/Δspec/Δsens, drift guard τ=0.02, t-based 95% CI). Additionally CI the per-fold **ΔAUC**
  list with the same t-table, since AUC is the primary metric here.
- **Grad-CAM / center-energy:** reuse `GradCAM` + `overlay_heatmap` + the center-energy
  formula (`run_all.run_gradcam`, `run_all.py:406-432`). Run it **twice on the `off` cache,
  same images** — once with the baseline checkpoint (`honest_baseline_fold{N}`, the "before")
  and once with the augment checkpoint (`honest_augment_fold{N}`, the "after") — for the 6 FN
  IDs + 3 control TPs, to get paired before/after center-energy per ID.

New arm: `augment`, preprocessing `off`, checkpoints `model_output/honest_augment_fold{N}/`,
summary `model_output/kfold_honest_augment_summary.json`.

## Success / failure criteria (declared up front)

- **Primary — paired 5-fold ΔAUC (augment − baseline):**
  - *Success:* mean ΔAUC > pooled SD (outside fold noise) **and** all folds hold sens ≥ 0.90.
  - *Neutral:* |mean ΔAUC| < pooled SD → report honestly as no detectable effect.
  - *Failure:* mean ΔAUC < −pooled SD, or any fold's held-out sens < 0.90 (hard abort).
- **Secondary — mechanistic, Δcenter-energy on the 6 tracked FN IDs** (baseline model →
  augment model, same off-cache images): does attention move onto the lesion (center-energy
  ↑), and how many FNs flip to correct at the frozen threshold? A neutral/negative mechanistic
  result is a valid outcome and is reported.
- **Power limit, stated in the report:** the 5 StratifiedGroupKFold folds share overlapping
  training data → not independent; baseline `std_auc` ≈ 0.0074, so a genuine gain below
  ~0.01 AUC will sit inside fold noise and read **neutral**. A neutral read at N=5 is *not*
  evidence the hypothesis is false — it is a power ceiling. CIs are descriptive
  (within-study variance), not population inference.

## Standalone detached runner: `experiments/run_augment.py`

Modeled on `run_all.py` (import its helpers, don't fork them): idempotent, resumable, no
Claude in the loop. Sequence:
1. **Preflight gate** (analytic proxy + visual grid) — abort on fail; skip if `preflight_PASS.txt`.
2. **Train arm `augment`, 5 folds** — the exact baseline command from `run_all.train_fold`
   (`efficientnet_b0`, `--preprocessing off`, stage1 5 / stage2 10, `--lr_stage2 1e-4`,
   `--unfreeze_blocks 19`, `--seed 42`, `--split_json fold{N}.json`, `--resume`) **plus the
   single new flag `--spatial_aug`**. Per-fold skip/resume via `is_training_complete`
   (best_model.pt + ≥15 CSV rows) → partial deletes best_model.pt and resumes from
   last_checkpoint. NaN-loss abort via `_nan_in_log`.
3. **Evaluate** each fold (`run_all.evaluate_fold(ckpt, fold, "off")`); sens<0.90 → abort.
   Write `kfold_honest_augment_summary.json` (skip whole arm if it exists).
4. **Paired comparison** vs the existing `kfold_honest_baseline_summary.json` (+ ΔAUC CI) →
   `model_output/paired_augment_comparison.json`.
5. **Before/after Grad-CAM** on the 6 FN IDs + 3 TPs → `gradcam_augment/`.
6. **Write `report_augment.md`** — AUC primary, Δcenter-energy table, power-limit caveat,
   honest verdict gated on drift-guarded ΔAUC vs pooled SD (mirror the P1/P2/P4 logic already
   in `run_all.write_report`).

Run detached: `python experiments/run_augment.py` (background/`nohup`-style), no polling.

## Files that change / get added

| File | Change |
|---|---|
| `model/dataset.py` | Add `spatial_aug=False` param to `HAM10000Dataset.__init__` and `make_dataloaders` (mirror the existing `strong_aug` plumbing). When set, swap the `geometric` list for the moderate spatial transform above. **No change to val/test path.** |
| `model/train.py` | Add `--spatial_aug` flag (mirror `--strong_aug`, `train.py:372`); thread into `make_dataloaders(..., spatial_aug=args.spatial_aug)`. |
| `experiments/run_augment.py` | **New** standalone runner (above). Imports `evaluate_fold`, `paired_comparison`, `is_training_complete`, `_csv_epoch_count`, `_nan_in_log`, `GradCAM`/`overlay_heatmap`/center-energy from `run_all`. |
| `PLAN_augment.md` | **This file.** |

No change to architecture, loss, optimizer, seed, epochs, fold/grouping logic, the honest
calibration protocol, or `run_all.py`'s locked results (its summaries exist → it stays put).

## Verification

- **Augmentation isolation:** assert the treatment train transform differs from baseline
  *only* in the geometric block (crop scale/ratio + rotation); ColorJitter, Normalize,
  flips, and the entire val/test transform are byte-identical. Diff the two `run_config.json`
  to prove only `--spatial_aug` differs.
- **Preflight gate:** `augment_qc/preflight_grid.jpg` renders; analytic retention ≥95% before
  training starts; run aborts cleanly (writes `STOP_REPORT.txt`) if a magnitude is later
  pushed past the lesion-retention bound.
- **Splits untouched:** `run_augment.py` reads `honest_splits/fold*.json` and never writes
  them; baseline summary is read-only.
- **Apples-to-apples pairing:** augment and baseline use the same fold jsons, seed, recipe,
  and the same `off` eval cache; comparison is properly paired per fold.
- **End-to-end read-out:** `kfold_honest_augment_summary.json` (5 folds' AUC/spec/sens +
  mean±std), `paired_augment_comparison.json` (ΔAUC primary + CI), `gradcam_augment/`
  (before/after overlays + per-ID center-energy), `report_augment.md` (verdict + power caveat).
