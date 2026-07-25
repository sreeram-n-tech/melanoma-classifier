# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A melanoma binary classifier (EfficientNet-B0 / MobileNetV2 on HAM10000) plus a
**leak-proof evaluation harness**. The point of the repo is not the score — it's
that every experiment reports honestly through a harness that's hard to fool.
Four experiments (artifact masking, spatial aug, bigger backbone, learning curve)
all returned honest nulls on AUC (~0.904); the write-ups keep the null results.
When adding experiments or "improvements," preserve that honesty — don't tune on
held-out data, and report neutral/negative results as such. `EXPLAINER.md` is the
full deep-dive; `README.md` is the summary.

## Environment & data

- Python 3.10+, PyTorch. `pip install -r requirements.txt`. GPU strongly preferred
  (developed on an RTX 3050). `python -c "import torch; print(torch.cuda.is_available())"`.
- Data is **not** in the repo (gitignored). Needs `data/raw/HAM10000_metadata.csv`
  and `data/raw/images/` (all ~10,015 .jpg in one folder), from Kaggle HAM10000.
- Trained weights (`model_output/*.pt`, ~2.6 GB) are gitignored. Only curated
  metrics/summary JSON+CSV under `results/` are committed.
- Run all commands **from the repo root** (`G:\srip`), not from inside `model/`.

## Core commands

```bash
# Build preprocessing caches (build BOTH on and off — the ablation arm needs both)
python model/dataset.py --build_preprocessed --preprocessing on
python model/dataset.py --build_preprocessed --preprocessing off

# Train (two-stage: frozen head, then fine-tune last N blocks). Checkpoints -> model_output/<run_tag>/
python model/train.py --arch efficientnet_b0 --loss weighted_ce --preprocessing off
python model/train.py ... --resume        # resume exactly where Ctrl+C left off (weights + EMA)

# Evaluate one checkpoint (metrics at 0.5 AND at the 95%-sensitivity threshold + ROC plot)
python model/evaluate.py --checkpoint model_output/<run>/best_model.pt

# Grad-CAM overlays
python model/gradcam.py --checkpoint model_output/<run>/best_model.pt --image_dir data/preprocessed --n_samples 8
```

There is **no test suite / linter / build step** — this is a research repo.
"Running a single test" = training/evaluating one config. Validate changes by
running the relevant experiment runner and checking the summary JSON it writes.

## The honest evaluation harness (the important architecture)

The trustworthy comparisons run through `experiments/run_kfold_honest.py`, driven
by pre-frozen split files in `experiments/honest_splits/fold{0-4}.json`
(generated once by `experiments/make_honest_splits.py`; committed for reproducibility).
Per fold it: trains on `inner_train` → fits temperature `T` on `inner_val` →
tunes the 95%-sensitivity threshold on `inner_val` → scores `held_out` **once**
with frozen `(T, threshold)`. It **hard-aborts** on NaN loss or if held-out
sensitivity < 0.90.

Non-negotiable anti-leakage invariants — do not break these:
- **Lesion-aware splits.** HAM10000 photographs the same lesion multiple times.
  Splits group by `lesion_id` (`StratifiedGroupKFold`), never `image_id`. Splitting
  by image was a real bug caught earlier that silently inflated AUC.
- **Three-way per fold:** `inner_train` trains, `inner_val` calibrates/tunes,
  `held_out` is scored once and never used for training or tuning.
- **Frozen threshold** fixed on `inner_val`, applied unchanged to `held_out`.
- Temperature scaling and TTA (flips) applied consistently to all arms.

`finalize.py` (`collect_logits`, `fit_temperature`, `metrics`, `probs_pos`) and
`evaluate.py:tune_threshold` are the shared calibration/threshold primitives —
the honest harness imports them so the deployed and evaluated numbers use the
same code path. Reuse these rather than re-implementing.

## Training internals (`model/train.py`)

- **Two stages, automatic:** Stage 1 trains the classifier head with the backbone
  frozen; Stage 2 unfreezes the last `--unfreeze_blocks` blocks and fine-tunes at
  lower LR. Cosine schedule only ever applies to Stage 2.
- **Recipe upgrades are all OFF by default** so old runs reproduce bit-for-bit:
  `--schedule cosine`, `--warmup_epochs`, `--early_stop_patience`, `--ema`,
  `--label_smoothing`, `--mixup`/`--cutmix`, `--strong_aug`, `--spatial_aug`,
  `--weight_decay` (>0 switches Adam→AdamW), `--img_size` (uses a separate `_N` cache).
  Keep this discipline: new knobs default to the original behavior.
- **`run_tag`** (the `model_output/` dir name) is derived from arch/loss/preprocessing/
  mode. Use `--run_name` for new-recipe A/B runs so they don't clobber a locked run's checkpoints.
- Saves `best_model.pt` (best val AUC) and `last_checkpoint.pt` (every epoch, for `--resume`).

## Preprocessing modes

`--preprocessing` selects the image cache, each a separate dir under `data/`:
`on`=full preprocess, `off`=resize-only (raw), `color_norm`, `devig` (devignette),
`devig_ruler` (devignette + ruler masking, the artifact-masking ablation arm).
`model/model.py` supports arches `mobilenet_v2`, `efficientnet_b0`, `efficientnet_b3`.

## Layout

- `model/` — `dataset.py` (caches, lesion-aware splits, dataloaders), `model.py`
  (backbones + freeze helpers), `train.py`, `evaluate.py`, `losses.py` (focal), `gradcam.py`.
- `experiments/` — runner scripts (`run_kfold_honest.py` is the honest one;
  `run_augment/backbone/learning_curve.py` are the four studies) + `honest_splits/`.
- Root utilities: `finalize.py`, `compare_roc.py` (bootstrap CIs), `external_eval.py`
  (frozen ckpt on PH2/ISIC), `analyze.py`, `run_leakage_study.py` (master orchestrator).
- `PLAN_*.md` = pre-experiment plans; `report*.md` = result write-ups (keep nulls honest).
