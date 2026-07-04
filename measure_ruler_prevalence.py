"""
measure_ruler_prevalence.py

Full-dataset scan for ruler / ink-mark artifacts.

Detection criterion (two signals required on the same side):
  profile_cv > CV_THRESH     -- periodic variation along the edge
  outer3_min  < MIN_THRESH   -- some near-black pixels (tick marks / ink)

Both are needed: high cv alone fires on skin texture; dark outer edge alone fires
on circular vignettes.  Together they characterise the alternating-marks pattern
seen on ISIC_0026158.

Output:
  - Per-image flag: ruler_any (bool), which sides fired, max cv, outer3_min
  - Prevalence breakdown by dx class (mel vs benign vs per-class)
  - Distribution of max_profile_cv and global_outer3_min across all images
  - List of flagged image IDs written to ruler_flagged.txt

Run from project root:
    python measure_ruler_prevalence.py
"""
import os
import time

import cv2
import numpy as np
import pandas as pd

RAW_DIR    = "data/raw/images"
META_PATH  = "data/raw/HAM10000_metadata.csv"
OUT_TXT    = "ruler_flagged.txt"
STRIP_PX   = 20     # width of edge strip to analyse
OUTER_PX   = 3      # outermost pixels for dark-minimum check
CV_THRESH  = 0.30   # primary: profile coefficient-of-variation
MIN_THRESH = 40     # corroborating: outer3_min below this
# Also report at stricter / looser thresholds for sensitivity check
CV_LO, CV_HI = 0.20, 0.45


def border_stats(gray, h, w):
    """Return per-side dict {mean, outer3_min, profile_cv} for 4 edge strips."""
    strips = {
        "top":    gray[:STRIP_PX, :],
        "bottom": gray[h - STRIP_PX:, :],
        "left":   gray[:, :STRIP_PX],
        "right":  gray[:, w - STRIP_PX:],
    }
    outers = {
        "top":    gray[:OUTER_PX, :],
        "bottom": gray[h - OUTER_PX:, :],
        "left":   gray[:, :OUTER_PX],
        "right":  gray[:, w - OUTER_PX:],
    }
    out = {}
    for side in ("top", "bottom", "left", "right"):
        strip = strips[side].astype(float)
        outer = outers[side]
        # 1-D profile: mean perpendicular to the edge direction
        profile = strip.mean(axis=0 if side in ("top", "bottom") else 1)
        cv = float(profile.std() / (profile.mean() + 1e-9))
        out[side] = {
            "mean":        float(strip.mean()),
            "outer3_min":  float(outer.min()),
            "profile_cv":  cv,
            "ruler_flag":  cv > CV_THRESH and outer.min() < MIN_THRESH,
        }
    return out


def main():
    meta = pd.read_csv(META_PATH)
    label_map = meta.set_index("image_id")["dx"].to_dict()

    rows = []
    t0 = time.time()
    all_jpgs = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg"))
    n = len(all_jpgs)
    print(f"Scanning {n} images ...")

    for i, fname in enumerate(all_jpgs):
        img_id = os.path.splitext(fname)[0]
        img = cv2.imread(os.path.join(RAW_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        stats = border_stats(img, h, w)

        # Circular-vignette gate (already characterised separately)
        c = max(1, int(min(h, w) * 0.12))
        corners = [img[:c, :c], img[:c, w-c:], img[h-c:, :c], img[h-c:, w-c:]]
        n_dark = sum(int(np.median(p)) < 40 for p in corners)
        is_vignette = n_dark >= 3

        ruler_sides = [s for s, v in stats.items() if v["ruler_flag"]]
        ruler_any = len(ruler_sides) > 0 and not is_vignette  # don't double-count vignette

        max_cv  = max(v["profile_cv"]  for v in stats.values())
        min_o3  = min(v["outer3_min"]  for v in stats.values())

        # Threshold-sensitivity counts (non-vignette only)
        ruler_lo = (not is_vignette and
                    any(v["profile_cv"] > CV_LO and v["outer3_min"] < MIN_THRESH
                        for v in stats.values()))
        ruler_hi = (not is_vignette and
                    any(v["profile_cv"] > CV_HI and v["outer3_min"] < MIN_THRESH
                        for v in stats.values()))

        rows.append({
            "image_id":    img_id,
            "dx":          label_map.get(img_id, "?"),
            "is_vignette": is_vignette,
            "ruler_any":   ruler_any,
            "ruler_lo":    ruler_lo,
            "ruler_hi":    ruler_hi,
            "ruler_sides": ",".join(ruler_sides),
            "max_cv":      round(max_cv, 4),
            "min_outer3":  round(min_o3, 1),
            "n_dark_corners": n_dark,
        })

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n}  ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s.\n")

    # ── Overall artifact prevalence ──────────────────────────────────────────
    total = len(df)
    n_vig    = df["is_vignette"].sum()
    n_ruler  = df["ruler_any"].sum()
    n_either = (df["is_vignette"] | df["ruler_any"]).sum()

    print(f"=== Overall artifact prevalence (N={total}) ===")
    print(f"  Circular vignette (>=3 dark corners)        : {n_vig:4d}  ({n_vig/total:.1%})")
    print(f"  Ruler / ink marks (cv>{CV_THRESH}, outer3<{MIN_THRESH}) : {n_ruler:4d}  ({n_ruler/total:.1%})")
    print(f"    of which -- looser cv>{CV_LO}             : {df['ruler_lo'].sum():4d}  ({df['ruler_lo'].sum()/total:.1%})")
    print(f"    of which -- stricter cv>{CV_HI}           : {df['ruler_hi'].sum():4d}  ({df['ruler_hi'].sum()/total:.1%})")
    print(f"  Either artifact                              : {n_either:4d}  ({n_either/total:.1%})")

    # ── Class breakdown ──────────────────────────────────────────────────────
    print(f"\n=== Ruler prevalence by dx class ===")
    print(f"  {'dx':<8} {'n_images':>9} {'n_ruler':>8} {'ruler%':>8}  "
          f"{'n_vig':>6} {'vig%':>6}")
    print(f"  {'-'*52}")
    order = ["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"]
    for dx in order:
        sub = df[df["dx"] == dx]
        if len(sub) == 0:
            continue
        nr = sub["ruler_any"].sum()
        nv = sub["is_vignette"].sum()
        print(f"  {dx:<8} {len(sub):>9} {nr:>8} {nr/len(sub):>8.1%}  "
              f"{nv:>6} {nv/len(sub):>6.1%}")
    # Melanoma vs rest
    mel  = df[df["dx"] == "mel"]
    rest = df[df["dx"] != "mel"]
    print(f"\n  {'mel':<8} {len(mel):>9} {mel['ruler_any'].sum():>8} "
          f"{mel['ruler_any'].mean():>8.1%}")
    print(f"  {'non-mel':<8} {len(rest):>9} {rest['ruler_any'].sum():>8} "
          f"{rest['ruler_any'].mean():>8.1%}")
    ratio = (mel['ruler_any'].mean() / (rest['ruler_any'].mean() + 1e-9))
    print(f"  Ruler prevalence ratio mel/non-mel: {ratio:.2f}x")

    # ── Which side flags most often ──────────────────────────────────────────
    print(f"\n=== Ruler-side distribution (among flagged images) ===")
    flagged = df[df["ruler_any"]]
    for side in ("top", "bottom", "left", "right"):
        cnt = flagged["ruler_sides"].str.contains(side).sum()
        print(f"  {side:<8}: {cnt:4d} / {len(flagged)} ({cnt/max(len(flagged),1):.1%})")

    # ── max_cv distribution ──────────────────────────────────────────────────
    print(f"\n=== max_profile_cv distribution across all images ===")
    for pct in (50, 75, 90, 95, 99):
        val = np.percentile(df["max_cv"], pct)
        print(f"  p{pct:<3}: {val:.3f}")
    print(f"  max:  {df['max_cv'].max():.3f}  (image {df.loc[df['max_cv'].idxmax(),'image_id']})")

    # ── Write flagged IDs ────────────────────────────────────────────────────
    flagged_ids = df[df["ruler_any"]]["image_id"].tolist()
    with open(OUT_TXT, "w") as f:
        for img_id in flagged_ids:
            row = df[df["image_id"] == img_id].iloc[0]
            f.write(f"{img_id}  dx={row['dx']}  sides={row['ruler_sides']}  "
                    f"max_cv={row['max_cv']:.3f}  min_outer3={row['min_outer3']:.0f}\n")
    print(f"\n  Flagged list -> {OUT_TXT}  ({len(flagged_ids)} images)")


if __name__ == "__main__":
    main()
