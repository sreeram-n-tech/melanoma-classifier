# Melanoma Classifier — an honest evaluation study on HAM10000

An EfficientNet-B0 melanoma classifier (AUC **0.904**) paired with a leak-proof
evaluation harness, used to run four honest experiments that probe *why* the
model's accuracy plateaus — and rule out the obvious suspects one by one.

> The headline isn't the score. It's that every experiment here reports its
> result honestly, including the null ones, through an evaluation harness built
> to be hard to fool. For the full, beginner-friendly deep-dive see
> **[EXPLAINER.md](EXPLAINER.md)**.

## Key results

Baseline: **EfficientNet-B0**, ImageNet-pretrained, 224×224, 5-fold
lesion-aware cross-validation.

| Metric | Value |
|---|---|
| **AUC** | **0.9042 ± 0.0074** |
| Sensitivity | 0.9379 ± 0.0216 |
| Specificity @ 95% sensitivity | 0.6619 ± 0.0395 |

The `±` is the standard deviation across the 5 folds. As a rule of thumb, a
change in AUC smaller than ~0.01 is inside that fold noise and reads as neutral.

### The four experiments (all honest nulls on the score)

| Experiment | Question | Result vs. baseline | Takeaway |
|---|---|---|---|
| **1. Artifact masking** | Cheating off rulers/vignettes? | AUC 0.9054, **Δ +0.0012** | Not cheating off artifacts — but 3/6 hard misses had *no* artifact, exposing a positional bias. |
| **2. Spatial augmentation** | Does breaking the framing habit help? | AUC 0.8983, **Δ −0.0059** | No score gain, **but** attention moved onto the lesion (+0.097 center-energy). Fixed the habit, not the ceiling. |
| **3. Bigger model (B3)** | Is the model too small? | AUC 0.8956, **Δ −0.0086** | No gain; B3 overfit on all 5 folds. Capacity isn't the limit. |
| **4. Learning curve** | Do we just need more data? | 25%→0.863 … 100%→0.904; last step **+0.0063** | Curve flattened before 100%. Quantity isn't the limit. |

**Conclusion:** four suspects tested, four cleared. By elimination the ~0.904
ceiling is best explained by **data quality and the intrinsic difficulty** of
some single-photo melanoma calls — stated within the power limits of a 5-fold
study. Full reasoning and caveats in [EXPLAINER.md](EXPLAINER.md).

## How the evaluation harness works

Every experiment is a comparison against the baseline, and the comparison is
only trustworthy because both arms run through the same pre-frozen, leak-proof
harness:

- **Lesion-aware splits** — HAM10000 photographs the same lesion multiple times.
  Splits are grouped by `lesion_id` (`StratifiedGroupKFold`) so no lesion ever
  appears in both train and test. This is the single most important anti-leakage
  safeguard.
- **Three-way split per fold** — `inner_train` (~64%) trains, `inner_val` (~16%)
  calibrates, `held_out` (~20%) is scored once at the end and never used for
  training or tuning.
- **Frozen decision threshold** — the cutoff is fixed on `inner_val` at the
  95%-sensitivity point, then applied unchanged to `held_out`. No picking the
  flattering cutoff after seeing the answers.
- **Temperature scaling** — a one-number calibration of confidence that doesn't
  change rankings (so it doesn't change AUC).
- **TTA** — test-time augmentation (flips) applied consistently to all arms.
- **Drift guard** — auto-flags any fold that behaved oddly vs. the others, so a
  weird fold is reported honestly instead of silently skewing the mean.

## Setup

Requires Python 3.10+ and (recommended) an NVIDIA GPU for training.

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Unix:     source venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # True = GPU visible
```

### Get the data

The **HAM10000** dataset is not included (see `.gitignore`). Download it from
[Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and
arrange it as:

```
data/raw/HAM10000_metadata.csv
data/raw/images/            # all ~10,015 .jpg files in one folder
```

### Run it

```bash
# Build the preprocessing caches (both needed for the ablation arm)
python model/dataset.py --build_preprocessed --preprocessing on
python model/dataset.py --build_preprocessed --preprocessing off

# Train the baseline
python model/train.py --arch efficientnet_b0 --loss weighted_ce --preprocessing off

# Evaluate a checkpoint (metrics at 0.5 and at the sensitivity-tuned threshold)
python model/evaluate.py --checkpoint model_output/<run>/best_model.pt

# Reproduce the honest study end-to-end (splits → both arms → Grad-CAM → report)
python run_leakage_study.py
```

See **[HOW_TO_RUN.md](HOW_TO_RUN.md)** for the full step-by-step walkthrough.

### Try the demo

A small FastAPI web app wraps the classifier (upload a lesion image → calibrated
probability + Grad-CAM). Weights aren't in the repo, so drop a `best_model.pt`
into `webapp/models/effb0/` first, then:

```bash
pip install -r webapp/requirements.txt
uvicorn webapp.app:app --reload      # open http://127.0.0.1:8000/
```

See **[webapp/README.md](webapp/README.md)** for details.

## Project structure

```
model/                    Core: model, dataset, training, evaluation, losses, Grad-CAM
preprocessing/            Devignette + ruler-masking transforms
experiments/              Experiment runners + frozen CV split definitions
  honest_splits/          The 5 frozen fold definitions (reproducibility)
results/                  Committed metrics & summaries (JSON/CSV) — weights excluded
webapp/                   FastAPI demo app (code only; model weights excluded)

finalize.py               Freeze threshold + temperature-calibrate → deploy.json
compare_roc.py            ROC / spec-at-sensitivity with bootstrap CIs
external_eval.py          Run a frozen checkpoint on a different dataset (PH2/ISIC)
analyze.py                Subgroup metrics + Grad-CAM error audit
run_leakage_study.py      Master orchestrator for the artifact-leakage study
run_all.py                Self-contained pipeline runner

EXPLAINER.md              Full technical deep-dive (start here)
HOW_TO_RUN.md             Step-by-step run guide
report*.md                Per-experiment result write-ups
PLAN_*.md                 Pre-experiment plans
```

Trained weights (`model_output/*.pt`, ~2.6 GB) and the dataset are intentionally
excluded from the repo; the small metrics/summary files that back every number
above are committed under `results/`.

## License & intended use

Research and educational use. This is **not** a medical device and must not be
used for clinical diagnosis.
