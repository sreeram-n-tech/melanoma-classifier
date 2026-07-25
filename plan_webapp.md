# Plan — Melanoma Classifier Research Demo Web App

> No app code is written until you've reviewed this and answered the Open Questions.

---

## Context

You have a finished melanoma-classification research project whose entire identity is
**honest evaluation** — leak-proof splits, calibrated probabilities, reported null results,
visible error rates. Your mentor asked for a web app that lets a user **upload a dermoscopy
image, pick a model, toggle preprocessing, and compare** — as a *research/educational*
demonstration, explicitly **not** a diagnostic product.

The models, the calibrated inference pipeline, and the preprocessing techniques are all
**already built and real** in this repo. The web app's job is to expose them faithfully:
its numbers must match your research reports, and its framing must never overclaim. This
plan designs that app in a **new, isolated `webapp/` folder** — the training code,
experiment scripts, and checkpoints are touched **read-only** (checkpoints are *copied*, never
moved or modified).

**Design-skill note:** you asked for the `impeccable` skill. No such skill is installed on
this machine (I searched `~/.claude`, the plugin marketplaces, and the repo). I used the
installed **`frontend-design`** skill instead — its SKILL.md drives the design section below.
If `impeccable` is a private skill you can install, say so and I'll re-run the design pass
through it.

---

## 1. Recommended tech stack

**Recommendation: FastAPI + server-rendered Jinja templates + a thin vanilla-JS `fetch` layer.
Self-hosted fonts/CSS, no build tooling, CPU-only. PyTorch model loaded once at startup,
in-process.**

### Why — the comparison you asked for

| Approach | Speed to demo | Design ceiling | Fit for this brief |
|---|---|---|---|
| **Gradio** | Fastest (hours) | **Low** — you fight its theme to get anywhere bespoke; the "calm credible clinical" look and the custom calibration readout are hard/janky | Great for a throwaway ML demo; **wrong** when design credibility and a persistent, precisely-placed safety band are first-class requirements |
| **Streamlit** | Fast | Low–medium | Rerun-on-interaction model makes before/after toggles and always-visible disclaimers awkward; same design ceiling problem |
| **FastAPI + Jinja + vanilla JS** ✅ | Medium (days) | **High** — real HTML/CSS = full control of hierarchy, type, the signature element, and safety copy | **Best fit**: single Python codebase (no npm/React), reuses your PyTorch code in-process, self-contained, deployable as one `uvicorn` process |
| FastAPI + React/Next SPA | Slow | Highest | Overkill for an internship deliverable — two toolchains, build pipeline, CORS. Not recommended |

The two things that break the "just use Gradio" default here are (a) you invoked a design
skill and want a **bespoke, credible** look, and (b) the **safety/honesty UI is
non-negotiable and needs exact placement/wording**. Both need real HTML/CSS. FastAPI+Jinja
buys that control while staying a single, small Python app — the right amount of work for a
time-boxed internship. Gradio remains the honest fallback if time collapses (see Open
Questions).

### Dependencies (`webapp/requirements.txt`)
`fastapi`, `uvicorn[standard]`, `torch`, `torchvision`, `opencv-python`, `pillow`, `numpy`,
`jinja2`, `python-multipart`. (No pandas/matplotlib/sklearn in the serving process — see §3.)

---

## 2. Design direction (via `frontend-design`)

The brief pins the aesthetic to "clean, calm, credible clinical/research." The brief's words
win — so this is disciplined, not flashy. But *clinical* is a **style**, and the design must
never imply *clinically validated*. The distinctive, non-templated angle comes from the
project's real soul: **calibrated honesty**, not "AI detects cancer." The app should feel like
a **precision lab instrument that is honest about its own uncertainty**.

### Design tokens

**Palette** (derived from the dermatoscope + a safety-driven color decision — deliberately
cool where the AI-default is warm-cream; amber-not-red for the referral signal):
```
--ink:    #1B2430   /* deep cool slate — primary text (not pure black) */
--paper:  #F7F8F9   /* cool off-white app surface */
--card:   #FFFFFF   /* raised surfaces */
--line:   #E4E7EA   /* hairline dividers/borders */
--dermis: #0E6E78   /* muted dermatoscopy teal — primary accent, used sparingly */
--refer:  #B5701B   /* considered amber/ochre — the "would refer" signal, NEVER red */
--muted:  #5E6B78   /* secondary text, labels, captions */
--wash:   #EAF1F1   /* faint teal wash for the below-threshold track region */
```
Color semantics are a **safety choice**: red = alarm/diagnosis, so the elevated-risk state is
**amber ("worth review")**, and the below-threshold state is calm teal/neutral. No screaming
red "MELANOMA" state anywhere.

**Typography** — one family for cohesion, personality from weight/scale + a monospace
signature. Self-hosted (OFL), **no CDN** (offline-capable, clinical-credible):
- Headings & body: **IBM Plex Sans** — genuine institutional/technical heritage; reads
  credible without the "startup Inter" or "editorial serif" defaults.
- **Numeric readouts** (probability, temperature, threshold, AUC/sens/spec): **IBM Plex Mono**
  — the signature type move. Every *honest number* is set as lab telemetry, reinforcing
  "instrument reading, not verdict."
- Type scale: 12 / 14 / 16 / 20 / 28 / 40, intentional line-heights, tightened heading tracking.
- Optional swap: **Space Grotesk** for headings if you want more edge (mentioned, not default).

**Signature element — the "Calibration Readout"** (the one thing the page is remembered by;
spend the boldness here, keep everything else quiet):
A horizontal 0→1 probability track rendered like a precision scale, showing:
1. the **calibrated P(melanoma)** as a labeled marker (IBM Plex Mono),
2. the **frozen referral threshold (0.194)** as a fixed, labeled gate on the track,
3. the region **above** the gate tinted `--refer` (amber), **below** tinted `--wash`,
4. the naïve **0.5 cutoff** as a faint secondary tick, labeled "not used — shown for contrast,"
5. a quiet animated fill to the value on result (respecting `prefers-reduced-motion`).

This turns "62% melanoma" into an honest, self-explaining instrument reading — it *is* the
project's thesis made visual.

**Motion/restraint:** one arrival transition + the track fill. Nothing else animates (extra
motion reads AI-generated). Quality floor, unannounced: responsive to mobile, visible keyboard
focus, reduced-motion honored, sufficient contrast.

### Layout & user flow (section by section)

Desktop = a two-pane "workbench"; mobile = single column (controls first, reading below).

```
┌───────────────────────────────────────────────────────────────────────┐
│  ▍Melanoma Classifier — Research Demonstration        [not a medical    │  identity bar
│                                                        device]          │
├───────────────────────────────────────────────────────────────────────┤
│  ⚠ Research/educational demonstration only. Not a medical device, not   │  PERSISTENT
│    for diagnosis, not a substitute for a dermatologist. Real concern →   │  disclaimer band
│    see a doctor.                                                         │  (sticky, always visible)
├──────────────────────────────┬────────────────────────────────────────┤
│  SPECIMEN & CONTROLS          │  READING                                │
│  ┌────────────────────────┐   │  ┌───────────────────────────────────┐  │
│  │  [ drop / upload image ]│   │  │  Before → After (when preproc on) │  │
│  │  Expects DERMOSCOPY,    │   │  └───────────────────────────────────┘  │
│  │  not phone photos.      │   │  ┌───────────────────────────────────┐  │
│  └────────────────────────┘   │  │  ▓▓▓ CALIBRATION READOUT ▓▓▓       │  │  ← signature
│  Model:  (o) EfficientNet-B0  │  │  0 ──●── |0.194| ─────── 1         │  │
│          ( ) MobileNetV2      │  │  P(melanoma) = 0.62   → REFER (amber)│ │
│          ( ) EfficientNet-B3* │  │                                     │  │
│  Preprocessing:               │  │  Model performance (held-out):      │  │
│   [ ] Hair removal            │  │   AUC 0.899 · Sens 0.95 · Spec 0.59 │  │  honesty stats
│   [ ] CLAHE contrast          │  └───────────────────────────────────┘  │
│   [ ] Color normalization     │  ┌───────────────────────────────────┐  │
│   [ ] Vignette removal        │  │  Plain-language: what this means   │  │  explainer panel
│   [ ] Ruler masking           │  │  (probability, threshold, limits)  │  │
│   [ ] Noise filter (erodes    │  └───────────────────────────────────┘  │
│        texture — off)         │                                         │
│         [ Run reading ]       │                                         │
├──────────────────────────────┴────────────────────────────────────────┤
│  Footer: research project · honest-evaluation ethos · not a medical dev │
└───────────────────────────────────────────────────────────────────────┘
```

**Flow:** upload → (optionally toggle preprocessing) → pick model → **Run reading** → the
right pane fills: before/after (if preproc on), the calibration readout, the flag, the model's
honest stats, and the plain-language panel. Empty state = a directive invitation, not a blank
box. Error state = plain, in the interface's voice ("That file isn't an image we can read —
upload a JPG or PNG dermoscopy image.").

---

## 3. Serving the model — preserving the honest inference pipeline

The reusable inference math already lives in `finalize.py` / `model/dataset.py` /
`model/evaluate.py`, **but** those operate on a `DataLoader` built from cached PNGs keyed by
`image_id` (`make_dataloaders`) — they can't serve one arbitrary uploaded image. So the webapp
reuses the **math** and adds a thin single-image path. Numeric parity is the requirement.

### Reuse map (single source of truth for the honest math)
- `from model import build_model` — clean import (torch/torchvision only, no side effects).
- **Vendor** (tiny, stable) into `webapp/inference.py`, each with a `# mirrors finalize.py`
  comment + a parity check, to keep the web process free of pandas/matplotlib (dragged in by
  importing `dataset`/`evaluate`):
  - the 4 TTA flip ops (`finalize.TTA_OPS`),
  - temperature scaling `probs_pos(logits, T) = softmax(logits / T)[:,1]`,
  - the eval transform, **exactly** `Resize((224,224)) → ToTensor → Normalize(ImageNet
    mean/std)` (mirrors `dataset.py` `augment=False`).
- Per-model `(temperature, threshold, tta, stats)` are **read from that model's `deploy.json`**
  — never hard-coded, never recomputed (recomputing needs the absent HAM10000 set).

### `infer_one(pil_img, model_entry, preprocess_fns)` — the single-image path
1. If preprocessing toggles are on: `pil → BGR uint8 np`, run the `preprocess.py` fn(s), `→ pil`.
2. `x = transform(pil).unsqueeze(0)` — 224², ImageNet-normalized.
3. `logits = mean over TTA_OPS if model_entry.tta else single view` — logit-space averaging,
   exactly as `collect_logits` does for `nets=[net]`.
4. `p = probs_pos(logits, model_entry.T)` — calibrated P(melanoma).
5. `above = p >= model_entry.threshold`.
6. return `{prob: p, above_threshold: above, threshold, temperature, tta, stats}`.

**TTA must match each model's `deploy.json.tta`** (B0=on, MobileNet=off) so the displayed
number corresponds to the recorded operating point. The 0.5 cutoff is computed too, but only
for the "shown for contrast" faint tick — the **decision uses 0.194** (or the model's own
threshold).

### Model registry (loaded once at startup)
```
MODELS = {
  "effb0":      { arch:"efficientnet_b0", ckpt: models/effb0/best_model.pt,       deploy: .../deploy.json },
  "mobilenetv2":{ arch:"mobilenet_v2",    ckpt: models/mobilenetv2/best_model.pt,  deploy: .../deploy.json },
  # "effb3":    uncalibrated — see Open Question 2
}
```
Checkpoints + their `deploy.json` are **copied** into `webapp/models/<name>/` so the app is
self-contained and `model_output/` is never touched. Sizes: B0 ≈15.6 MB, MobileNet ≈8.7 MB
(B3 ≈41 MB only if included).

### Proposed `webapp/` structure
```
webapp/
  app.py                    FastAPI routes: GET / (page), POST /predict (multipart → JSON)
  inference.py              registry + infer_one + vendored honest-math (parity-checked)
  preprocessing_bridge.py   PIL↔cv2 + toggle→preprocess.py fn mapping + ordering
  models/effb0/…  models/mobilenetv2/…    (copied best_model.pt + deploy.json)
  templates/index.html      Jinja
  static/  css/  js/  fonts/(self-hosted IBM Plex OFL)
  requirements.txt   README.md
```

---

## 4. Wiring model & preprocessing selection

**Model selection:** radio in the UI → `/predict` param → registry lookup → `infer_one` with
that entry's arch/ckpt/T/threshold/tta. The reading pane **always** shows the selected model's
own honest stats from its `deploy.json` (single source of truth), so switching models is an
honest comparison, not a hidden swap.

**Preprocessing selection** (all real, from `preprocessing/preprocess.py`):

| Toggle | Function | Notes |
|---|---|---|
| Hair removal | `remove_hair` | DullRazor blackhat→inpaint |
| CLAHE contrast | `enhance_contrast` | LAB L-channel CLAHE |
| Color normalization | `color_normalize` | Shades-of-Gray constancy |
| Vignette removal | `remove_vignette` | **no-op if no dark corners** → UI: "no vignette detected" |
| Ruler masking | `remove_ruler` | **no-op if no ruler** → UI: "no ruler detected" |
| Noise filter | `denoise` | **off by default**, flagged "erodes real lesion texture" |

Multiple-on order follows `preprocess.py`'s own `preprocess_image` (hair → [denoise] →
contrast), with color/vignette/ruler applied in a fixed, documented order first. The **Before →
After** pair renders above the readout so the user sees exactly what the operation did (some
ops legitimately produce *no visible change* — that's honest and gets a caption, not a hidden
result).

---

## 5. Safety & honesty elements — where each lives

| # | Element | Location | Draft copy (interface voice) |
|---|---|---|---|
| 1 | **Persistent disclaimer band** | Sticky, top of every view | "Research/educational demonstration only. Not a medical device, not for diagnosis, not a substitute for a dermatologist. If you have a real concern about a skin lesion, see a doctor." |
| 2 | **Dermoscopy-only upload note** | At the upload control | "Expects dermoscopy images (dermatoscope close-ups). Ordinary phone photos will give unreliable results — the models were trained only on dermoscopy." |
| 3 | **Result framing** | Calibration readout + flag | "This research model *estimates* the probability of melanoma." Never "diagnosis"/"detected." |
| 4 | **Known error rates always shown** | Reading pane, under the readout | The model's own AUC / sensitivity / specificity from `deploy.json` — the estimate is never shown without its uncertainty. |
| 5 | **Threshold explanation** | On the readout | "0.194 = the model's referral threshold, tuned for 95% sensitivity on validation — not a naïve 50%." |
| 6 | **Off-distribution caveat** | Appears whenever any preprocessing toggle is on | "This model was trained on *raw* images. With preprocessing applied, the input is off-distribution — treat this estimate as **illustrative**; it is **not** backed by the reported AUC." Readout styled 'provisional' (hatched) in this state. |
| 7 | **Uncalibrated caveat** | Only if B3 is included | "Uncalibrated: raw model output, no frozen threshold, not comparable to the other models." |
| 8 | **Footer** | Bottom | Restates research/educational + not-a-medical-device; links to the project's honest-evaluation ethos (README/EXPLAINER). |

The **off-distribution honesty (#6) is the subtle-but-critical one**: your primary B0 was
trained `preprocessing="off"`, so *every* preprocessing toggle feeds it off-distribution input.
The app shows the before/after (genuinely educational) but must **never present an
authoritative-looking number for a preprocessed image without the caveat.**

---

## 6. Phased build order (smallest working thing first)

- **Phase 0 — End-to-end skeleton.** Scaffold `webapp/`; copy B0 `best_model.pt` + `deploy.json`;
  `inference.py` (single model, TTA per deploy.json, no preprocessing); `POST /predict`
  (upload → JSON); one minimal HTML page (upload → calibrated prob + flag as text).
  **Deliverable: upload → one model → honest prediction, numbers matching `deploy.json`.**
- **Phase 1 — Designed prediction UI + safety.** Implement the `frontend-design` token system,
  the Calibration Readout signature element, the amber flag, the stats panel, the
  plain-language panel, the persistent disclaimer, upload note, footer. Accessibility/responsive
  floor. **Deliverable: the single-model app looks credible and is fully honest.**
- **Phase 2 — Model selector.** Registry with B0 + MobileNetV2 (both calibrated); selector UI;
  per-model stats + TTA handling; B3 per Open Question 2. **Deliverable: switch/compare models honestly.**
- **Phase 3 — Preprocessing selector + before/after.** Wire `preprocess.py` via
  `preprocessing_bridge.py`; toggles; PIL↔BGR; before/after render; no-op notices;
  off-distribution caveat (#6); denoise destructive warning. **Deliverable: try/compare
  preprocessing, honestly labeled.**
- **Phase 4 — Compare & polish (stretch).** Optional side-by-side compare (models ×
  preprocessing); final copy pass; empty/error states; a11y + mobile QA; `webapp/README.md`.

**Time risks (flagged):**
- *Design fidelity* (Phase 1) is the main time sink — mitigated because this plan locks the
  token system, so Phase 1 is execution, not exploration.
- *PIL↔cv2 BGR round-tripping* (Phase 3) — channel-order/dtype bugs; plus no-op/denoise
  semantics need UI handling. Small but fiddly.
- *B3 calibration gap* — decide in Open Question 2 **before** Phase 2 to avoid rework.
- *Numeric parity* — low risk given we vendor the exact math + a parity check, but must be
  verified in Phase 0.
- The build phase (not this plan) will use the **`full-output-enforcement`** skill so generated
  files are complete, no placeholders.

---

## 7. Open questions (decide before building)

1. **Tech stack** — accept **FastAPI + Jinja + vanilla JS** (recommended), or prefer **Gradio**
   (fastest, much lower design ceiling — the bespoke look and calibration readout suffer)?
2. **EfficientNet-B3** — it has **no calibrated `deploy.json`** and can't get one without the
   absent HAM10000 val set. **(a)** Include it labeled "uncalibrated — raw output, no threshold,
   not comparable" (recommended: honors the 3-model ask, stays honest), **(b)** omit it (offer
   only the two calibrated models), or **(c)** you supply HAM10000 later so I can run `finalize.py`
   to calibrate it (future work, out of scope now)?
3. **Preprocessing → prediction when off-distribution** — **(a)** show the number with a
   prominent "illustrative / not AUC-backed" caveat (recommended, matches your brief), or
   **(b)** show before/after but **withhold** the authoritative number until toggles are off?
4. **Deployment** — **local demo only** (recommended; copy checkpoints into `webapp/`, run
   `uvicorn` on localhost for your mentor), or **hosted/shareable** (needs a box that has the
   `.pt` files — note they're gitignored and B3 is ~41 MB)?
5. **MobileNet variant** — use the honest baseline `mobilenet_v2_weighted_ce_off_unfreeze19_main`
   (AUC 0.883 / sens 0.886) or a recipe variant like `mv2_recipe_v4` (AUC 0.874 / sens 0.904)?
   Recommend the honest baseline for the cleanest comparison. (Minor.)
6. **Compare view** (Phase 4 side-by-side) — in scope for the internship timeline, or a stretch goal?

---

## 8. Verification (how we'll test end-to-end)

- **Phase 0 parity:** run `uvicorn webapp.app:app`; POST a sample image; assert `prob ∈ [0,1]`
  and `flag == (prob ≥ 0.194)`. Confirm the honest math: temperature actually applied
  (T=1 vs 1.893 shifts prob as expected), TTA averages 4 flips in logit space. If any HAM10000
  image is available locally, cross-check one output against a `finalize.py`-style hand
  calculation; otherwise assert internal consistency.
- **Phase 1:** visual QA in mobile + desktop widths; keyboard-only navigation; reduced-motion;
  confirm the disclaimer band is visible in every state (idle, result, error).
- **Phase 2:** switch models → each shows **its own** T/threshold/stats and the number changes
  accordingly; displayed stats **equal** the `deploy.json` values (single source of truth).
- **Phase 3:** toggle each preprocessing → before/after renders; no-op images show the
  "no artifact detected" caption; the off-distribution caveat (#6) appears; denoise shows the
  destructive warning.
- **Cross-cutting honesty check:** no view ever shows a bare probability without (a) its model's
  error rates and (b) any applicable caveat.
