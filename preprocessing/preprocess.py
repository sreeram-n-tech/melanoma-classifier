"""
preprocessing/preprocess.py

Deterministic preprocessing pipeline:
  1. Hair removal  (morphological blackhat -> threshold -> inpaint, "DullRazor"-style)
  2. Contrast enhancement (CLAHE on the L channel in LAB space)

This is deterministic -- the same input always produces the same output -- so
it's run ONCE per image and cached to disk (see dataset.py:build_preprocessed_cache),
rather than recomputed every epoch. Random augmentation (flip/rotate/zoom/color
jitter) is a separate, intentionally-live step applied at training time.

Denoising was REMOVED from the default pipeline after testing on real HAM10000
images: even mild fastNlMeansDenoising (h=1) cut measured image sharpness by
~40%, and h=3+ produced a visible "watercolor blob" look across the whole
lesion -- HAM10000 dermoscopy images are clean enough (dermatoscope-captured,
not phone photos) that there's very little sensor noise to remove, so
denoising mostly just erases real pigment-network/vessel texture instead. The
function is kept below in case you want to experiment, but isn't called by
default.

Also includes a standalone CLI to visually validate the pipeline on a handful
of images before running it over the full dataset -- check the lesion isn't
distorted, real texture survives, AND actual hairs fade out, before moving on.
"""
import argparse
import os

import cv2
import numpy as np


def remove_hair(img, kernel_size=9, thresh=20, inpaint_radius=5):
    """Detects hair-like structures via a morphological blackhat filter, then
    inpaints over them so they don't get learned as a spurious feature.

    These defaults (kernel_size=9, thresh=20) were tuned against real
    HAM10000 images, not synthetic ones -- thresh=30 from an earlier pass
    barely caught any real hair (real hairs here are lower-contrast than
    a synthetic test suggested). Classical hair removal like this won't make
    hairs perfectly invisible, especially where they cross directly over
    pigmented lesion texture -- it fades them, which is the realistic outcome
    for this style of algorithm. If real hairs are still clearly un-faded in
    your validation images, lower thresh a bit further; if lesion texture/dots
    start visibly eroding, raise it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
    # Dilate slightly so inpainting gets a clean, fully-covered edge to
    # blend from instead of patchy single-pixel-wide gaps.
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(img, hair_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    return inpainted


def denoise(img, h=3):
    """NOT called by default -- see module docstring. Kept for experimentation
    only; even h=1 measurably erases real lesion texture on this dataset."""
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
    )


def enhance_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def color_normalize(img, p=6):
    """Shades-of-Gray color constancy (Finlayson & Trezzi, 2004). Estimates
    the illuminant color cast per-channel (using a Minkowski p-norm average,
    p=6 is the standard default) and rescales channels so the average color
    is neutral gray. Removes device/lighting color casts (e.g. an overly red
    or yellow tint) WITHOUT blurring -- it's a per-pixel linear scale, not a
    smoothing filter, so unlike denoise() it doesn't erase real texture.
    NOT proven to help this model yet -- test with an ablation before
    trusting it, same as every other step in this file."""
    img_f = img.astype(np.float32)
    illum = np.power(np.mean(np.power(img_f, p), axis=(0, 1)), 1.0 / p)
    illum_norm = illum / np.linalg.norm(illum)
    correction = illum_norm.mean() / illum_norm
    out = img_f * correction
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_vignette(img, corner_frac=0.12, dark_thresh=40,
                    min_radius_frac=0.35, max_radius_frac=0.52):
    """Detect and neutralise the dark circular dermoscope vignette by filling
    the border region with the median in-field colour.

    Safe no-op: returns the original array (same object, NOT a copy) when
    fewer than 3 of the 4 corner patches have median grayscale below
    dark_thresh.  Callers can test ``result is img`` to detect the no-op.

    Detection:
      1. Corner-darkness gate -- 4 corner patches of size
         ``corner_frac * min(H,W)``; fire only when >= 3 are dark.
      2. Hough circle on a Gaussian-blurred grayscale image.  If Hough finds
         nothing, fall back to a centred circle at 44% of min(H,W).
      3. Fill all pixels outside the circle with the per-channel median of
         the in-field pixels (minimises the residual edge vs a fixed grey).

    Args:
        img:              BGR uint8 ndarray, full-resolution raw image.
        corner_frac:      fraction of min(H,W) used as the corner-patch side.
        dark_thresh:      grayscale median threshold (0-255); vignettes are
                          typically < 20, borderless corners typically > 50.
        min_radius_frac:  min Hough radius as fraction of min(H,W).
        max_radius_frac:  max Hough radius as fraction of min(H,W).

    Returns:
        BGR uint8 ndarray, same shape as img.  Pixel-identical to input when
        the no-op gate fires (same object, not a copy).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Corner-darkness gate
    c = max(1, int(min(h, w) * corner_frac))
    corners = [gray[:c, :c], gray[:c, w - c:], gray[h - c:, :c], gray[h - c:, w - c:]]
    n_dark = sum(int(np.median(p)) < dark_thresh for p in corners)
    if n_dark < 3:
        return img  # no-op: pixel-identical, same object

    # 2. Locate the circular field via Hough
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    min_r = int(min(h, w) * min_radius_frac)
    max_r = int(min(h, w) * max_radius_frac)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5,
        minDist=min(h, w),   # expect exactly one dominant circle
        param1=50, param2=30,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        dists = [abs(cx - w // 2) + abs(cy - h // 2) for cx, cy, _ in circles]
        cx_c, cy_c, r = circles[int(np.argmin(dists))]
    else:
        # Hough failed: conservative centred-circle fallback
        cx_c, cy_c = w // 2, h // 2
        r = int(min(h, w) * 0.44)

    # 3. Fill outside-circle pixels with median in-field colour
    Y, X = np.ogrid[:h, :w]
    outside = (X - cx_c) ** 2 + (Y - cy_c) ** 2 > r ** 2
    inside = ~outside
    fill = np.array([int(np.median(img[:, :, ch][inside])) for ch in range(3)],
                    dtype=np.uint8)
    out = img.copy()
    out[outside] = fill
    return out


def remove_ruler(img, cv_thresh=0.30, min_thresh=40, strip_px=20):
    """Detect and mask ruler / ink-mark strips along image edges.

    Detection criterion (matches measure_ruler_prevalence.py): on the same side,
    profile_cv > cv_thresh  AND  outer-3px minimum grayscale < min_thresh.
    Both signals required: high cv alone fires on skin texture; dark outer pixels
    alone fire on the circular vignette (handled by remove_vignette).

    Safe no-op: returns the original array (same object) when no side fires.
    Callers can test ``result is img``.

    Fill: median BGR of the in-field region (all non-flagged-strip pixels).

    Args:
        img:        BGR uint8 ndarray (any resolution).
        cv_thresh:  profile coefficient-of-variation threshold (default 0.30).
        min_thresh: outer-3px minimum grayscale below which ticks are confirmed.
        strip_px:   width of edge strip to test and mask (default 20 px).

    Returns:
        BGR uint8 ndarray, same shape.  Pixel-identical (same object) when no
        side triggers.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

    def _stats(strip, outer, axis):
        profile = strip.mean(axis=axis)
        cv = float(profile.std() / (profile.mean() + 1e-9))
        return cv, float(outer.min())

    checks = {
        "top":    _stats(gray[:strip_px, :], gray[:3, :], 0),
        "bottom": _stats(gray[h-strip_px:, :], gray[h-3:, :], 0),
        "left":   _stats(gray[:, :strip_px], gray[:, :3], 1),
        "right":  _stats(gray[:, w-strip_px:], gray[:, w-3:], 1),
    }
    flagged = [s for s, (cv, mn) in checks.items() if cv > cv_thresh and mn < min_thresh]
    if not flagged:
        return img  # no-op: same object

    field = np.ones((h, w), dtype=bool)
    for s in flagged:
        if s == "top":    field[:strip_px, :] = False
        elif s == "bottom": field[h-strip_px:, :] = False
        elif s == "left":   field[:, :strip_px] = False
        elif s == "right":  field[:, w-strip_px:] = False

    fill = np.array([int(np.median(img[:, :, ch][field])) for ch in range(3)],
                    dtype=np.uint8)
    out = img.copy()
    for s in flagged:
        if s == "top":    out[:strip_px, :] = fill
        elif s == "bottom": out[h-strip_px:, :] = fill
        elif s == "left":   out[:, :strip_px] = fill
        elif s == "right":  out[:, w-strip_px:] = fill
    return out


def validate_vignette_on_samples(raw_dir, out_dir, image_ids):
    """Write 3-panel side-by-sides (original | border-in-red | devignetted)
    for each image_id and return per-image stat dicts.

    Each stat dict: image_id, noop (bool), n_dark_corners (int),
    masked_px (int), masked_frac (float).
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = []
    for image_id in image_ids:
        src = os.path.join(raw_dir, f"{image_id}.jpg")
        img = cv2.imread(src)
        if img is None:
            print(f"  WARNING: cannot read {src}, skipping")
            continue
        h, w = img.shape[:2]

        # Replicate corner gate for stats (same params as remove_vignette defaults)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = max(1, int(min(h, w) * 0.12))
        corners = [gray[:c, :c], gray[:c, w - c:], gray[h - c:, :c], gray[h - c:, w - c:]]
        n_dark = sum(int(np.median(p)) < 40 for p in corners)

        result = remove_vignette(img)
        is_noop = result is img

        if is_noop:
            masked_px = 0
        else:
            masked_px = int(np.any(result != img, axis=2).sum())
        masked_frac = masked_px / (h * w)

        stats.append({
            "image_id": image_id, "noop": is_noop,
            "n_dark_corners": n_dark,
            "masked_px": masked_px, "masked_frac": float(masked_frac),
        })

        # 3-panel: original | border-region highlighted red | devignetted
        highlight = img.copy()
        if not is_noop:
            border_mask = np.any(result != img, axis=2)
            highlight[border_mask] = [0, 0, 255]  # BGR red
        panel = np.concatenate([img, highlight, result], axis=1)
        label = ("NOOP" if is_noop
                 else f"masked={masked_frac:.1%}  dark_corners={n_dark}")
        cv2.putText(panel, f"{image_id}  {label}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"vignette_{image_id}.jpg"), panel)

    return stats


def preprocess_image(img, kernel_size=9, hair_thresh=20, inpaint_radius=5,
                      apply_denoise=False, denoise_h=3, clip_limit=2.0):
    """Full pipeline, in order: hair removal -> (optional denoise) -> contrast
    enhancement. apply_denoise defaults to False -- see module docstring."""
    img = remove_hair(img, kernel_size=kernel_size, thresh=hair_thresh, inpaint_radius=inpaint_radius)
    if apply_denoise:
        img = denoise(img, h=denoise_h)
    img = enhance_contrast(img, clip_limit=clip_limit)
    return img


def validate_on_samples(test_dir, out_dir, n_images=10, kernel_size=9, hair_thresh=20,
                         inpaint_radius=5, apply_denoise=False, denoise_h=3, clip_limit=2.0):
    """Writes side-by-side before/after comparisons for a handful of images.
    Open these and check: lesion texture (pigment network, dots, vessels)
    still clearly visible and not blurred/blotchy, AND real hairs visibly
    faded compared to the original. If hairs are still fully solid/un-faded,
    lower hair_thresh; if texture looks eroded or blotchy, raise hair_thresh
    or leave apply_denoise off."""
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:n_images]
    for fname in files:
        img = cv2.imread(os.path.join(test_dir, fname))
        if img is None:
            print(f"  WARNING: could not read {fname}, skipping")
            continue
        processed = preprocess_image(img, kernel_size=kernel_size, hair_thresh=hair_thresh,
                                      inpaint_radius=inpaint_radius, apply_denoise=apply_denoise,
                                      denoise_h=denoise_h, clip_limit=clip_limit)
        side_by_side = np.concatenate([img, processed], axis=1)
        cv2.imwrite(os.path.join(out_dir, f"compare_{fname}"), side_by_side)
    print(f"Wrote {len(files)} before/after comparisons to {out_dir}/. "
          f"Check: lesion texture still clearly visible (not blurry/blotchy), "
          f"AND real hairs visibly faded vs. the original.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--out_dir", default="preprocessing_check")
    parser.add_argument("--n_images", type=int, default=10)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--hair_thresh", type=int, default=20,
                         help="Lower = more aggressive hair detection, but risks "
                              "flagging real lesion texture/dots as hair. Raise if "
                              "texture looks eroded; lower if real hairs are clearly "
                              "un-faded.")
    parser.add_argument("--inpaint_radius", type=int, default=5)
    parser.add_argument("--apply_denoise", action="store_true",
                         help="Off by default -- testing showed this erases real "
                              "lesion texture on HAM10000. Pass this flag only if "
                              "you have a specific reason to try it.")
    parser.add_argument("--denoise_h", type=int, default=3)
    parser.add_argument("--clip_limit", type=float, default=2.0)
    args = parser.parse_args()
    validate_on_samples(args.test_dir, args.out_dir, args.n_images, args.kernel_size,
                         args.hair_thresh, args.inpaint_radius, args.apply_denoise,
                         args.denoise_h, args.clip_limit)
