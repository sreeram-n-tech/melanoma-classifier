# PLAN — Vignette-Removal Ablation (constant-fill mask)

## Context

**Why this experiment.** A Grad-CAM audit of the 8 test-set false negatives in the
final model (`model_output/effb0_unfreeze19_main/`) found 6 "attention failures":
activation concentrates on the circular dermoscope **vignette / dark border** instead
of the lesion. Hypothesis: the dark vignette ring is a salient spurious feature the
model latched onto; neutralising it should move attention back onto the lesion and
recover some missed melanomas **without** hurting the clinically critical operating
point (specificity at 95% sensitivity).

**The intended change is a strict ablation.** The ONLY thing that varies between the
baseline and treatment arms is **vignette removal in the cached preprocessing**.
Architecture (EfficientNet-B0), recipe (10 Stage-2 epochs, constant LR 1e-4, 19 blocks
unfrozen, seed 42), the lesion-ID-aware fold definitions, and the evaluation harness are
identical across both arms.

**Two design decisions are already locked (user-confirmed):**
- **Removal style = constant-fill mask** — detect the dark border, repaint it with a
  per-image neutral fill, leaving the field *in place*. Lesion **scale and framing are
  unchanged**, so this is the cleanest isolation of "does the border carry signal the
  model exploits." (Trade-off: a faint residual ring edge may remain — see Risks.)
- **Eval metric = honest k-fold finalize** — a new harness (used identically for BOTH
  arms) that, per fold, fits temperature + freezes the 95%-sens threshold on an **inner
  validation set carved from the training folds**, then scores the untouched held-out
  fold with TTA. See "Why new eval code is required" below.

---

## What the code actually does today (grounded findings)

**Preprocessing & cache (`model/dataset.py`).**
- `IMG_SIZE = 224`. Images are preprocessed **once** and cached as PNG; every consumer
  (train/val/test/finalize/TTA/Grad-CAM) reads PNGs from a cache dir. This means **a
  transform baked into the cache is automatically identical at train, val, test,
  inference and inside TTA — there is no train/serve skew possible.** This is the key
  architectural lever for the ablation.
- `build_preprocessed_cache(df, raw_dir, out_dir, mode=..., img_size)` writes the cache;
  `mode` ∈ {`full` = hair-removal + CLAHE, `raw` = resize-only, `color_norm`}.
- `cache_dir_for(preprocessing, img_size)` and
  `_BASE_CACHE_DIR = {"on": PREPROCESSED_DIR, "off": RAW_CACHE_DIR, "color_norm": COLOR_NORM_CACHE_DIR}`
  map the `--preprocessing` flag to a cache directory.
- Transforms: val/test = `Resize(224) + ToTensor + Normalize(ImageNet)`; train adds
  flips/rotation/`RandomResizedCrop`/jitter. Normalisation is shared.
- **The final model uses `preprocessing="off"`** (confirmed in
  `effb0_unfreeze19_main/deploy.json`) → the resize-only cache `data/no_preprocessing_cache`.
  **So the baseline has NO hair-removal/CLAHE.** The treatment must therefore be
  "resize-only **+ devignette**" — adding nothing else — or the change would be confounded.

**Training (`model/train.py`, `model/model.py`).**
- Recipe knobs: `--arch efficientnet_b0`, `--epochs_stage2 10`, `--lr_stage2 1e-4`,
  `--unfreeze_blocks 19`, `--seed 42`. Stage-1 freezes backbone; Stage-2 unfreezes the
  last N top-level blocks (`unfreeze_last_n_blocks`).
- K-fold split: `lesion_aware_kfold(df, n_splits=5, seed=42)` uses `StratifiedGroupKFold`
  grouped by `lesion_id` (no lesion crosses folds). In current fold mode,
  `train.py` sets **`test_df = val_df` (the held-out fold)** — i.e. the held-out fold is
  used both for model selection AND scoring (no separate inner-val).
- `image_dir = cache_dir_for(args.preprocessing, args.img_size)` is the only line that
  selects the cache.

**Honest single-split harness (`finalize.py`).**
- `fit_temperature` (LBFGS NLL on **validation** logits) → produces `T` (1.893 in the
  current deploy). `tune_threshold(..., target_sensitivity=0.95)` freezes the threshold
  on **validation** (0.194 currently). `collect_logits(..., tta=True)` averages raw
  logits over 4 flip views (`TTA_OPS`). `metrics()` reports AUC + sens + spec.
- **But `finalize.py` runs only on the single `lesion_aware_split` (70/15/15)** — this is
  where `effb0_unfreeze19_main`, T=1.893, thr=0.194 and the 8 FNs come from.

**K-fold scoring today (`experiments/run_kfold.py` → `model/evaluate.py`).**
- `evaluate.py` in fold mode tunes the threshold **on the held-out fold itself**
  (`tune_threshold(labels, probs_pos)`), with **no temperature scaling and no TTA** →
  optimistic, NOT the honest protocol. Existing kfold summaries are `effb0_baseline` and
  `effb0_cosine10` — **none at the `unfreeze19` recipe.**

**Why new eval code is required.** The metric the experiment compares on — "5-fold mean
± std of the *honest* spec@95" — does not exist as one artifact. `finalize.py` is honest
but single-split; `run_kfold.py` is 5-fold but optimistic. We build ONE honest k-fold
harness and run **both** arms through it (so "harness identical between arms" still holds).

**Grad-CAM audit (`model/gradcam.py`, `analyze.py`).** Target layer `net.features[-1]`.
`analyze.py::gradcam_mistakes` selects the `k` most-confident FPs and least-confident FNs
**dynamically** (not hardcoded) and writes overlays to
`model_output/effb0_unfreeze19_main/analysis/false_neg_*.png`. The 6 attention-failure
IDs live in those filenames — **read them from that folder and confirm the exact 6**
before the mechanistic check (do not hardcode from memory).

**PH2 external (`prepare_ph2.py`, `external_eval.py`).** PH2 BMPs are copied raw and
loaded **without** going through the HAM cache → devignetting PH2 needs the transform
applied in the PH2 path explicitly. PH2 is a different scanner with different border
characteristics. Treated as a **secondary, optional** re-run (see Risks).

---

## Design decisions — resolved, with trade-offs

**1. Detection: per-image, not fixed geometric crop.** *Recommendation: per-image
detection with a no-op guard.* Not every HAM10000 image has a vignette; a fixed
geometric crop would destroy borderless/full-frame images (cut real lesion tissue). The
detector must be a **safe no-op** when no dark border ring is present. Trade-off:
per-image adds detection-failure risk (mis-masking a dark lesion), mitigated by a
conservative corner-darkness gate + connectivity-to-border requirement.

**2. Removal style: constant-fill mask (LOCKED).** Detect the dark ring; repaint masked
pixels with a **per-image neutral fill** = median/mean colour of the in-field region
(minimises a new high-contrast edge vs a fixed grey). Field stays in place → **lesion
scale and framing identical to baseline**, so the ablation isolates "border content"
alone. Trade-off vs the alternatives: crop-to-field would have enlarged the lesion (extra
confound); circular-zero would inject a synthetic edge. Constant-fill is the most
conservative; residual faint ring is the main remaining risk.

**3. Application point: baked into the cache → zero skew.** The devignette transform is
applied inside `build_preprocessed_cache` and written to a dedicated cache dir
`data/devignette_cache`. Because train/val/test/finalize/TTA/Grad-CAM all read PNGs from
the resolved cache dir, the SAME pixels are seen everywhere. TTA flips operate on the
already-devignetted tensor, so TTA inherits it for free. **Confirmed: no train/serve skew.**

**4. Calibration re-fit end to end.** The old `T=1.893` / `thr=0.194` are tied to the old
(`off`) input distribution and are **NOT reused**. The honest k-fold harness re-fits `T`
on each fold's inner-val and re-freezes the 95%-sens threshold on each fold's inner-val,
then applies to the held-out fold. Per-fold T and threshold are reported.

---

## Reporting the full operating point per fold (not just specificity)

The frozen-threshold protocol freezes the 95%-sens threshold on **inner-val**, then
applies it untouched to the held-out fold. Therefore the **achieved sensitivity on the
held-out fold floats** — it is 95% on inner-val, *not exactly* 95% on the scored data. A
bare "spec@95" is misleading because the 95 is not held fixed on the held-out fold.

**Each per-fold record (both arms) must store:** `frozen_threshold`,
`achieved_sensitivity` (on the held-out fold), `achieved_specificity` (on the held-out
fold), and `auc`. The aggregate summary reports mean ± std of each, plus the mean achieved
sensitivity (to show how far it drifts from 0.95).

## Success / failure criteria — PAIRED, per fold

Both arms run on **byte-identical** fold splits/seed (see split-json note below), so the
comparison is **paired per fold**: for each of the 5 folds, compute
`Δspec_f = spec_devig,f − spec_off,f` and `Δsens_f = sens_devig,f − sens_off,f` **on the
same held-out fold**.

- **Confounding guard (per fold):** a fold's `Δspec_f` is only interpretable if that
  fold's **achieved sensitivities are comparable across arms**, i.e. `|Δsens_f| ≤ τ`
  (tolerance, default **τ = 0.02**). If `|Δsens_f| > τ`, **flag the fold**: its specificity
  delta is confounded by operating-point drift, not a clean effect — report it but exclude
  it from the headline judgement (and say so).
- **Primary judgement:** on the (unflagged) **5 paired specificity deltas**, judge by the
  **mean paired Δspec relative to the spread** of the paired deltas (e.g. mean vs std of
  the deltas / a paired sign reading):
  - *Success:* mean paired Δspec **clearly positive relative to the spread** of the 5
    deltas, **and** AUC non-inferior (mean paired ΔAUC ≥ ~0).
  - *Neutral:* paired Δspec within noise (spread overlaps 0).
  - *Failure:* mean paired Δspec negative.
- **Statistical-power caveat (stated honestly):** with only **5 folds**, this paired test
  reliably detects a **large** shift but a **small real improvement may read neutral**. A
  null is an honest, valid outcome — **not something to chase** by tweaking the protocol.
- **Secondary:** 5-fold mean **AUC (± std)**, also reported paired (mean ΔAUC).
- **Mechanistic:** re-run Grad-CAM on the **same 6 attention-failure IDs** (each scored
  out-of-fold by the `devig` checkpoint of the fold that holds it). Report (a) how many are
  now classified correctly at the frozen threshold, and (b) whether CAM mass moved onto the
  lesion. The central-vs-border CAM-energy proxy **assumes a centered lesion**: for each ID,
  **first check whether the lesion is actually central**; for **off-center** lesions, rely
  on the **visual overlay comparison**, not the central-energy number. A neutral/negative
  result is a valid outcome.

---

## Risks (flagged)

- **Residual ring edge (from constant-fill):** the model may key on the masked region's
  boundary instead of the original border. Mechanistic CAM check will catch this.
- **Scale vs 224 input:** *largely neutralised* by the constant-fill choice (field not
  cropped/resized differently). Only secondary effect: fill changes the global colour
  statistics slightly.
- **Currently-correct predictions could flip.** Removing border content may shift some
  current TNs/TPs. This is exactly why the primary metric is on the **aggregate**, not the
  6 cases.
- **Detection false-mask:** a genuinely dark lesion abutting the frame could be partly
  masked. Mitigate with corner-gate + border-connectivity; log per-image mask area and
  the no-op rate; eyeball a sample (reuse the `preprocessing_check/` side-by-side pattern).
- **PH2 external validation:** PH2 has different border characteristics and bypasses the
  HAM cache. Re-running PH2 requires applying `remove_vignette` in the PH2 path. Treated
  as **secondary/optional**; if run, interpret cautiously (the no-op guard should leave
  borderless PH2 images unchanged).

---

## Files that change / get added

| File | Change |
|---|---|
| `preprocessing/preprocess.py` | **Add** `remove_vignette(img_bgr, ...)`: per-image dark-ring detection (cv2, BGR) + constant-fill, **safe no-op** on borderless images. Deterministic. |
| `model/dataset.py` | **Add** `"devig"` to `_BASE_CACHE_DIR` → `data/devignette_cache`; add a `"devignette"` branch in `build_preprocessed_cache` (= resize-only **+** `remove_vignette`, no hair/CLAHE). |
| `model/train.py` | **Add** `"devig"` to `--preprocessing` choices; **add** optional `--split_json` to load explicit lesion-aware **inner_train/inner_val/test** id-lists (so one training code path serves the honest folds). |
| `model/evaluate.py`, `finalize.py` | **Add** `"devig"` to their image-dir maps (keeps the single-split tools usable on the new cache). No logic change. |
| `experiments/make_honest_splits.py` | **New, run ONCE.** Builds the 5 lesion-aware folds (`lesion_aware_kfold`, seed 42); per fold carves a lesion-aware inner-val from the 4 training folds (reuse `lesion_aware_split`). Writes `experiments/honest_splits/fold{0..4}.json` (inner_train/inner_val/held-out `image_id` lists). **Both arms consume these exact files** → splits are byte-identical, not regenerated per arm. |
| `experiments/run_kfold_honest.py` | **New** orchestrator: for each fold, loads `honest_splits/fold{f}.json`, trains via `train.py --split_json`, then runs a per-fold finalize **reusing `finalize.py` helpers** (`collect_logits`, `fit_temperature`, `probs_pos`, `tune_threshold`, `metrics`, TTA on) — inner-val for T+threshold, held-out fold for scoring. Records per fold: `frozen_threshold`, `achieved_sensitivity`, `achieved_specificity`, `auc`, `T`. Aggregates → `kfold_honest_<arm>_summary.json`. Run for `off` (baseline) and `devig` (treatment) against the **same split files**. |
| `analyze.py` (reuse) | Mechanistic Grad-CAM on the 6 IDs against `data/devignette_cache` + the holding fold's checkpoint. |

No changes to architecture, loss, augmentation, seed, or the fold/grouping logic.

---

## Ordered execution checklist

1. **Confirm the 6 attention-failure IDs** by reading
   `model_output/effb0_unfreeze19_main/analysis/false_neg_*.png` filenames; record them.
2. **Implement `remove_vignette`** in `preprocessing/preprocess.py` (cv2/BGR, per-image
   detection + constant-fill, no-op guard). Add a tiny visual check (reuse the
   `preprocessing_check/` side-by-side pattern) on ~10 images incl. borderless ones;
   verify no-op on borderless and clean fill on vignetted. Log mask-area / no-op rate.
3. **Wire the `"devig"` cache** into `model/dataset.py` (`_BASE_CACHE_DIR` +
   `build_preprocessed_cache` `"devignette"` branch) and add `"devig"` to the maps in
   `train.py`, `evaluate.py`, `finalize.py`.
4. **Build the cache** `data/devignette_cache` via `build_preprocessed_cache(mode="devignette")`.
5. **Add `--split_json`** to `model/train.py` (load inner_train/inner_val/test id-lists).
6. **Generate splits ONCE:** `experiments/make_honest_splits.py` →
   `experiments/honest_splits/fold{0..4}.json`. Both arms reuse these identical files.
7. **Write `experiments/run_kfold_honest.py`** (consumes the split files; per-fold honest
   finalize; records frozen_threshold + achieved_sens + achieved_spec + auc + T; aggregates).
8. **Run BASELINE arm:** `run_kfold_honest.py --preprocessing off --arch efficientnet_b0
   --unfreeze_blocks 19 --epochs_stage2 10 --lr_stage2 1e-4 --seed 42`
   → `kfold_honest_off_summary.json` (this is the missing honest baseline).
9. **Run TREATMENT arm:** same command with `--preprocessing devig`, **same split files**
   → `kfold_honest_devig_summary.json`.
10. **Compare paired:** per fold compute Δspec_f and Δsens_f on the same held-out fold;
    flag folds with |Δsens_f| > τ (0.02); judge mean paired Δspec vs spread + AUC
    non-inferiority, with the 5-fold power caveat. Report the full operating point per fold.
11. **Mechanistic Grad-CAM:** for each of the 6 IDs, score out-of-fold with its fold's
    `devig` checkpoint; report #now-correct; for centered lesions use central-vs-border CAM
    energy, for off-center lesions use the visual overlay comparison vs the old overlays.
12. **(Optional, secondary) PH2:** apply `remove_vignette` in the PH2 path, re-run
    `external_eval.py` with the new frozen per-fold (or refit) calibration; interpret with
    the border-difference caveat.
13. **Write up** the result honestly (including neutral/negative), with the per-fold
    operating-point table, the paired deltas, and the mechanistic findings.

---

## Verification

- **Devignette correctness:** side-by-side overlays on vignetted + borderless samples;
  assert pixel-identical output on borderless images (no-op), and that masked area lands
  on the dark ring only.
- **No skew:** confirm `cache_dir_for("devig")` resolves to `data/devignette_cache` and
  that train/finalize/Grad-CAM all read it (single source of truth).
- **Calibration is fresh:** assert per-fold T ≠ 1.893 and threshold ≠ 0.194 (they are
  refit), and that threshold is frozen on inner-val, never on the held-out fold.
- **Apples-to-apples:** baseline and treatment use identical fold seeds, split jsons,
  recipe flags; diff the two run configs to prove only `--preprocessing` differs.
- **Aggregate read-out:** `kfold_honest_{off,devig}_summary.json` each contain 5 folds'
  AUC + spec@95 and the mean±std; primary comparison computed from these.
