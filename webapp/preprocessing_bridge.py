"""
webapp/preprocessing_bridge.py

Bridges the web app's PIL image world to preprocessing/preprocess.py's OpenCV
(BGR, uint8) world, and maps UI toggle names to those functions in a fixed,
documented order — imported (read-only), not copied.

Order when multiple toggles are on: color_normalize -> remove_vignette ->
remove_ruler run first (in that order), then preprocess_image's own internal
order (remove_hair -> [denoise] -> enhance_contrast). This mirrors
model/dataset.py:build_preprocessed_cache, which also runs preprocessing on
the full-resolution raw image BEFORE the 224x224 resize.

OFF-DISTRIBUTION WARNING: every model currently served
(webapp/inference.py:_REGISTRY_SPEC) was trained with `--preprocessing off`
(raw images only) and calibrated (temperature + threshold) on that same raw
distribution. None of these toggles were seen during training or
calibration. Any image that passes through apply_toggles() with a non-empty
`enabled` set is, by definition, off that distribution — the model's
calibrated probability is no longer backed by its reported AUC. Callers
(app.py) MUST surface this as illustrative, never as an authoritative number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "preprocessing"))
from preprocess import (  # noqa: E402  (research code, imported not copied)
    color_normalize,
    denoise,
    enhance_contrast,
    remove_hair,
    remove_ruler,
    remove_vignette,
)

# (key, function, UI label, no-op noun for "no <noun> detected" caption
#  (None if this step can't no-op), warning shown in the UI)
_STEPS = [
    ("color_normalize", color_normalize, "Color normalization", None, None),
    ("remove_vignette", remove_vignette, "Vignette removal", "vignette", None),
    ("remove_ruler", remove_ruler, "Ruler masking", "ruler", None),
    ("remove_hair", remove_hair, "Hair removal", None, None),
    ("denoise", denoise, "Noise filter", None,
     "Erodes real lesion texture — even mild settings measurably reduced sharpness "
     "in testing on this dataset. Off by default; use with caution."),
    ("enhance_contrast", enhance_contrast, "CLAHE contrast", None, None),
]

TOGGLE_KEYS = [key for key, *_ in _STEPS]


def toggle_catalog() -> list[dict]:
    """UI-facing metadata for the toggle checklist, in application order."""
    return [
        {"key": key, "label": label, "can_noop": noop_noun is not None, "warning": warning}
        for key, _fn, label, noop_noun, warning in _STEPS
    ]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_toggles(pil_img: Image.Image, enabled: set[str]) -> tuple[Image.Image, list[dict]]:
    """Runs the enabled toggles, in the fixed pipeline order, on the image at
    its native resolution (matching build_preprocessed_cache's ordering).
    Returns the processed PIL image plus a per-step report (applied vs.
    no-op) for honest UI captions.

    If `enabled` is empty, returns (pil_img, []) unchanged with no BGR
    round-trip at all — the untouched case never touches OpenCV.
    """
    if not enabled:
        return pil_img, []

    img = pil_to_bgr(pil_img)
    report = []
    for key, fn, label, noop_noun, warning in _STEPS:
        if key not in enabled:
            continue
        before = img
        img = fn(img)
        is_noop = noop_noun is not None and (img is before)
        caption = f"No {noop_noun} detected — image unchanged." if is_noop else "Applied."
        report.append({
            "key": key, "label": label, "noop": is_noop, "caption": caption, "warning": warning,
        })

    return bgr_to_pil(img), report
