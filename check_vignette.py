"""
check_vignette.py  --  visual validation of remove_vignette() before building
                       the full devignette cache.

Writes 3-panel JPEGs to vignette_check/:
    original | border-highlighted-in-red | devignetted

Also prints per-image stats (no-op, masked fraction) and asserts pixel
identity for every no-op image.

Usage (run from project root):
    python check_vignette.py
    python check_vignette.py --n_sample 80 --n_panels 15 --seed 7
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "preprocessing"))
from preprocess import remove_vignette, validate_vignette_on_samples

RAW_DIR = "data/raw/images"
OUT_DIR = "vignette_check"


def quick_stats(image_id):
    """Load one image, run remove_vignette, return stat dict. No disk writes."""
    img = cv2.imread(os.path.join(RAW_DIR, f"{image_id}.jpg"))
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c = max(1, int(min(h, w) * 0.12))
    corners = [gray[:c, :c], gray[:c, w - c:], gray[h - c:, :c], gray[h - c:, w - c:]]
    n_dark = sum(int(np.median(p)) < 40 for p in corners)
    result = remove_vignette(img)
    is_noop = result is img
    masked_px = 0 if is_noop else int(np.any(result != img, axis=2).sum())
    return {
        "image_id": image_id, "noop": is_noop,
        "n_dark_corners": n_dark,
        "masked_px": masked_px, "masked_frac": masked_px / (h * w),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=50,
                        help="Images to sample for aggregate stats")
    parser.add_argument("--n_panels", type=int, default=12,
                        help="Side-by-side images to write to vignette_check/")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_jpgs = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg"))
    chosen = rng.choice(all_jpgs, size=min(args.n_sample, len(all_jpgs)), replace=False)
    sample_ids = [os.path.splitext(f)[0] for f in chosen]

    # --- Stats pass (no I/O writes) ---
    print(f"Computing stats on {len(sample_ids)} images ...")
    all_stats = [s for s in (quick_stats(i) for i in sample_ids) if s is not None]

    noops = [s for s in all_stats if s["noop"]]
    vignetted = [s for s in all_stats if not s["noop"]]

    print(f"\n=== Aggregate stats ({len(all_stats)} images) ===")
    print(f"  No-op (borderless) : {len(noops):3d} / {len(all_stats)} = {len(noops)/len(all_stats):.1%}")
    print(f"  Vignetted (filled) : {len(vignetted):3d} / {len(all_stats)}")
    if vignetted:
        fracs = [s["masked_frac"] for s in vignetted]
        print(f"  Masked-px fraction : "
              f"min={min(fracs):.1%}  "
              f"p25={np.percentile(fracs, 25):.1%}  "
              f"median={np.median(fracs):.1%}  "
              f"p75={np.percentile(fracs, 75):.1%}  "
              f"max={max(fracs):.1%}")

    # --- Pixel-identity assertion for every no-op ---
    print(f"\nVerifying pixel identity for {len(noops)} no-op images ...")
    failures = 0
    for s in noops:
        img = cv2.imread(os.path.join(RAW_DIR, f"{s['image_id']}.jpg"))
        result = remove_vignette(img)
        if result is not img:
            print(f"  FAIL pixel-identity: {s['image_id']}")
            failures += 1
    if failures == 0:
        print(f"  All {len(noops)} no-op images returned same object. OK.")
    else:
        print(f"  {failures} FAILURES -- check corner_frac / dark_thresh.")

    # --- Select panel images: variety of types ---
    # 2 highest-masked (stress-test the fill), up to 4 typical vignetted, up to 4 no-ops
    by_mask_desc = sorted(vignetted, key=lambda s: -s["masked_frac"])
    panel = []
    seen = set()
    for s in by_mask_desc[:2]:            # highest masked (potential false-mask worst-case)
        panel.append(s); seen.add(s["image_id"])
    for s in vignetted:                   # typical vignetted
        if s["image_id"] not in seen and len(panel) < 6:
            panel.append(s); seen.add(s["image_id"])
    for s in noops:                       # borderless
        if s["image_id"] not in seen and len(panel) < args.n_panels:
            panel.append(s); seen.add(s["image_id"])
    panel = panel[:args.n_panels]
    panel_ids = [s["image_id"] for s in panel]

    # --- Write panels ---
    print(f"\nWriting {len(panel_ids)} panels to {OUT_DIR}/ ...")
    panel_stats = validate_vignette_on_samples(RAW_DIR, OUT_DIR, panel_ids)

    n_noop_panels = sum(1 for s in panel_stats if s["noop"])
    n_vig_panels = sum(1 for s in panel_stats if not s["noop"])
    print(f"\n{'IMAGE_ID':<25} {'NOOP':<6} {'DARK_CRNRS':<12} MASKED_FRAC")
    print("-" * 60)
    for s in panel_stats:
        print(f"{s['image_id']:<25} {'YES' if s['noop'] else 'no':<6} "
              f"{s['n_dark_corners']:<12} {s['masked_frac']:.1%}")

    print(f"\nPanel summary: {n_vig_panels} vignetted, {n_noop_panels} no-op")
    print(f"\nDone.  Open {OUT_DIR}/*.jpg to inspect.")
    print("  Panel 1: original  |  Panel 2: border-in-red  |  Panel 3: devignetted")
    print("  Vignetted: border should be cleanly filled; lesion pixels unchanged.")
    print("  No-op: Panel 1 and 3 should look identical (border-in-red shows nothing).")

    if n_noop_panels < 3:
        print(f"\nWARNING: only {n_noop_panels} no-op images in panels -- "
              f"consider --n_sample 100 or --n_panels 16.")


if __name__ == "__main__":
    main()
