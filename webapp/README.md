# Melanoma Classifier — Research Demonstration (web app)

A local, research/educational web demo of the melanoma classifier. **Not a medical
device, not for diagnosis.** It serves the trained models from this repo behind an
image upload and shows the model's *calibrated* melanoma probability with its known
error rates.

This folder is self-contained: the serving checkpoints are **copied** into
`models/` — the research training code and `model_output/` are never modified.

## Status: Phase 0-4 all done (design polish + compare view, presentation only)

- Upload a dermoscopy image → pick **EfficientNet-B0** or **MobileNetV2** → calibrated
  P(melanoma) + referral flag at that model's own frozen threshold. Each model's
  temperature/threshold/TTA/stats come from its own `deploy.json` — never shared.
- Full `frontend-design` token system (IBM Plex Sans/Mono stack, amber-not-red
  referral color, 12/14/16/20/28/40 type scale), the **Calibration Readout**
  signature element (0→1 track, labeled threshold gate, faint 0.50 tick shown
  "not used"), a sticky disclaimer band, dermoscopy-only upload note, model
  stats panel, and a plain-language explainer panel. Accessibility floor:
  keyboard focus rings, `prefers-reduced-motion` respected, responsive
  two-pane → single-column layout below 860px.
- **Preprocessing toggles** (`preprocessing_bridge.py`, wrapping
  `preprocessing/preprocess.py` read-only): color normalization, vignette removal,
  ruler masking, hair removal, noise filter (off by default, flagged destructive),
  CLAHE contrast — applied in a fixed order matching
  `model/dataset.py:build_preprocessed_cache`. Every model here was trained with
  `--preprocessing off`, so any toggle makes the input off-distribution: the
  reading gets a **PROVISIONAL** badge, a hatched readout overlay, and a caveat
  that the number isn't backed by the reported AUC — the calibrated number is never
  shown bare. Before/after images render when any toggle is on; no-op-capable steps
  (vignette, ruler) report "no vignette/ruler detected" honestly instead of implying
  an effect that didn't happen.
- The numbers match each model's own `models/<key>/deploy.json` — verified by
  `parity_check.py` (loops over every registered model).
- `inference.py` is unchanged since Phase 0; preprocessing runs entirely before
  the honest inference path, never inside it.

Font note: IBM Plex Sans/Mono (Regular + SemiBold, the only weights actually
used) are self-hosted as `.woff2` in `static/fonts/` (OFL-licensed, vendored
from the IBM/plex repo — see `static/fonts/LICENSE.txt`), wired via
`@font-face` in `static/css/style.css`. No CDN call, no network dependency.

### Design polish pass (presentation only, `inference.py`/registry/parity/preprocessing_bridge untouched)

- Self-hosted IBM Plex fonts (above).
- Fixed a real bug in the Calibration Readout: the gate/naive/marker text labels
  sit just outside the track box (`top:-22px` etc.), but the track had
  `overflow: hidden` on itself, which silently clipped all three labels
  invisible. Fixed by moving the clip onto a new inner `.readout-fill` layer
  that holds only the colored regions, leaving the track free to show its
  labels. Also gave the marker label more clearance above the gate label so
  they don't visually collide when the probability lands near the threshold.
- Unified the disclaimer band's colors with the `--refer`/`--refer-text` design
  tokens instead of one-off hex values, so every "amber" surface in the app
  (disclaimer, referral flag, provisional badge) reads as one consistent signal.
- AUC/Sensitivity/Specificity values now render in IBM Plex Mono (matching the
  plan's "every honest number is lab telemetry" rule), replacing plain bold text.
- Small identity-bar accent (a 4px teal tick before the title) per the plan's
  own mockup, echoing `--dermis` without competing with the readout.

### Post-polish bugfix (presentation only)

Real-browser testing surfaced two symptoms that traced back to one CSS bug:
`.before-after` and `.badge` both set `display` unconditionally in the
stylesheet, which — per the CSS cascade — beats the browser's native
`[hidden] { display: none }` user-agent rule regardless of selector
specificity (any author-origin rule outranks any user-agent-origin rule).
Net effect: the before/after image panel and the PROVISIONAL badge were
*always* rendered, independent of what JS set `.hidden` to. That's why the
before/after pane showed bare alt-text placeholders (no `src` was ever
assigned when preprocessing wasn't applied) and why the PROVISIONAL badge
appeared with zero toggles checked. `preprocessing_applied` itself was
already correct on the backend (verified via direct curl with no
`preprocess` field — returns `False`) — this was never a logic bug. Fixed
with one rule replacing the old narrow `#result[hidden], #empty-state[hidden]`
patch:
```css
[hidden] { display: none !important; }
```

### Phase 4 — side-by-side model compare view

A "Compare both models side by side" checkbox next to the model select. When
checked, the model dropdown is disabled (comparison always runs every
registered model, currently EfficientNet-B0 and MobileNetV2) and **Run
reading** posts to a new `POST /compare` route instead of `/predict`.

`/compare` (in `app.py`) reuses the exact same validation/decode/preprocess
helper as `/predict`, then calls `inference.infer_pil` once per registered
model on the **same** preprocessed image — `inference.py` itself was not
touched, and there is no ensembling: each model's probability, threshold,
temperature, and stats stay its own, exactly as `/predict` already returns
them. The only new field is `disagree` — whether the two models' own
frozen-threshold decisions differ — computed as a plain boolean comparison,
never as a combined/averaged score.

The UI shows two columns (one per model, labeled with the model's own name),
each with its own calibration readout, probability, referral flag, and
held-out stats — reusing a shared `fillReading()` JS function so the single
and compare views can't drift out of sync. Above the columns, a verdict band
states plainly whether the models agree or disagree (disagreement is styled
amber, same semantic as the "would refer" state — worth a closer look, not
an alarm). Preprocessing toggles apply identically to both columns (one
shared before/after image, one shared off-distribution caveat, since the
input the two models see is identical); each column's own PROVISIONAL badge
still reflects that model's own `preprocessing_applied` flag independently.

Verified via curl: `/compare` with no toggles returns both models' honest
readings and `disagree: false` (both agreed on the test image); with a
toggle on, both are correctly marked `preprocessing_applied: true` and share
one `processed_image_b64`. `parity_check.py` re-run after this change —
both models still PASS (no inference math touched).

### Phase 4 close-out — copy pass, B3, a11y/mobile QA

- **Copy pass:** fixed a leftover singular ("the model was trained only on
  dermoscopy") from before the model selector existed — now "these models,"
  matching the fact that two models are selectable and compare mode runs both.
- **Open Question 2 (B3), resolved:** EfficientNet-B3 stays out of the
  registry. It has no `deploy.json` — calibrating it needs the original
  HAM10000 validation split, which this demo doesn't have. Rather than bolt
  on an "uncalibrated" third model/UI state this late, the explainer panel
  ("What does this number mean?") now says plainly why it's excluded, so the
  gap is disclosed instead of silently absent.
- **Empty/error states:** already correct — Run reading stays disabled until
  a file is chosen (no empty-submission error path exists), `/predict` and
  `/compare` share one validation helper so both reject bad input with the
  same plain-voice message, and a failed request clears both result panes
  back to the empty-state invitation card while the error shows in
  interface voice near the button.
- **A11y:** added `aria-live="polite"` to `#result` and `#compare-result` so
  screen readers announce a new reading without requiring focus to move;
  `#err` already used `role="alert"`. Verified color is never the only
  signal (flag/badge/verdict all pair color with text), focus-visible rings
  cover every native control including the new compare checkbox, and the
  Calibration Readout's `aria-label` updates for both single and compare
  columns.
- **Mobile QA:** `.compare-columns` stacks to one column below 1000px and
  goes two-up at 1000px+, inside a `.wrap` capped at 1040px — reviewed at
  narrow (<860px, single-pane workbench), tablet (860–999px, two-pane
  workbench + stacked compare columns), and desktop (1000px+, two-pane
  workbench + side-by-side compare columns) widths; no overflow, sufficient
  column width for the readout and mono numbers at every band.

This closes out Phase 4 exactly as `plan_webapp.md` scoped it — no phases
remain open.

### Visual redesign — editorial hero + "Precision Analysis Bench" (presentation only)

Full re-skin to an editorial, medical-instrument look, using an AI-generated
Stitch mockup (`stitch_dermoscopy_research_instrument/`) as a *visual*
reference only — no code, data, fake models, or invented numbers were carried
over from it. `inference.py`, the model registry, parity logic, and
`preprocessing_bridge.py` were **not touched**; the only backend change is
`app.py`'s `index()` route now also passes the EfficientNet-B0 `ModelEntry`
(`primary`, via the existing public `inference.get_model()`) so the new
explainer section can render its real threshold instead of a placeholder.

- **Hero:** a real HAM10000 melanoma sample (`ISIC_0033848`, CC BY-NC 4.0,
  credited in a small caption) behind a dark scrim, "Melanoma — Reading the
  Skin" headline, honest subhead, scroll-to-explore link into the explainer.
- **"Dermoscopy & Models" section:** the existing transparency/honesty copy,
  plus a decorative calibration diagram that shows **only the frozen
  threshold** (`{{ primary.threshold }}`, currently 0.194) — no probability
  marker, no invented value (the mockup's own hard-coded "0.62" was
  deliberately dropped).
- **"Precision Analysis Bench":** the existing functional controls/reading
  panels, restyled into the mockup's sharp-cornered, mono-numeral,
  amber-for-caution instrument language, with every element ID/class `app.js`
  depends on preserved verbatim — no JS changes were needed or made.
- Kept, unchanged in behavior: all six real preprocessing toggles (noise off
  by default with its warning), the live PROVISIONAL badge/hatched-readout/
  caveat state, the compare-both-models view wired to `/compare`, the
  disclaimer band, the "what does this number mean" panel, the
  dermoscopy-only upload note, and error rates shown beside every
  probability.
- New design tokens keep the existing amber-not-red / IBM Plex Sans+Mono
  system; every new color pairing was checked against WCAG AA (4.5:1) by
  hand before use, including the mockup's own badge colors, which did not
  quite clear AA for text and were not reused as-is.
- Verified after the rewrite: `/`, `/predict`, `/compare` all curl-tested
  (200s, real numbers, `preprocessing_applied` toggling correctly with a real
  `processed_image_b64`), and `parity_check.py` re-run — both models PASS.

## Run (from the repo root, `G:\srip`)

```bash
pip install -r webapp/requirements.txt      # torch etc. are already present in this repo
uvicorn webapp.app:app --reload
# open http://127.0.0.1:8000/
```

## Verify parity with the research numbers

```bash
python webapp/parity_check.py
```
Runs the web app's own inference path over the same held-out test split
(`lesion_aware_split`, seed 42) and the same `off` image cache that `finalize.py`
used, then compares AUC / sensitivity / specificity to `deploy.json`. Expected:
all three match within 1e-3.

## How inference stays honest

`inference.py` reuses the exact research math: `build_model` is imported from
`model/`, and the eval transform (`Resize 224 → ToTensor → ImageNet Normalize`),
the 4-flip TTA, and temperature scaling are vendored verbatim from
`model/dataset.py` and `finalize.py`. Temperature (T=1.893) and threshold (0.194)
are read from `deploy.json`, never recomputed.

Note: `parity_check.py` reads the pre-built 224² `off` cache, so its pixels are
identical to what `finalize.py` saw. A live upload is resized on the fly by the same
`Resize((224,224))` transform (PIL bilinear); this can differ by a hair from the
cache's interpolation but is the standard eval transform and does not affect the
frozen operating point.
