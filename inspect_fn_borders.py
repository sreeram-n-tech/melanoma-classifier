"""
inspect_fn_borders.py  --  characterize border attention for the 4 non-vignette
attention-failure false-negative IDs.

For each ID writes a 3-panel JPEG to border_inspect/:
    original (224x224) | Grad-CAM overlay | zone-energy annotated

Also prints per-ID text stats:
  - CAM energy fractions in 9 non-overlapping zones (3x3 grid)
  - Border vs centre aggregate energy ratio
  - Raw image border analysis per side: mean brightness, outer-3px stats,
    profile coefficient-of-variation (high CV -> ruler/marks vs uniform dark)
  - Per-side classification: dark frame / ruler / normal

Run from project root:
    python inspect_fn_borders.py
"""
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Mirror analyze.py's sys.path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, ROOT)

from gradcam import GradCAM, overlay_heatmap   # noqa: E402  (model/ on path)
from model import build_model, get_target_layer  # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

CHECKPOINT = "model_output/effb0_unfreeze19_main/best_model.pt"
CACHE_DIR  = "data/no_preprocessing_cache"   # same distribution model was trained on
RAW_DIR    = "data/raw/images"
OUT_DIR    = "border_inspect"
IMG_SIZE   = 224
BORDER_PX  = 34   # outermost 15 % of 224 px ≈ 34 px

TARGET_IDS = ["ISIC_0032569", "ISIC_0034222", "ISIC_0026158", "ISIC_0025791"]


# ── model ──────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    net, _ = build_model(ckpt["arch"], num_classes=2, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.to(device).eval()
    return net, ckpt["arch"]


# ── CAM spatial analysis ───────────────────────────────────────────────────────

def cam_zone_energy(cam):
    """
    Split the (H, W) CAM (0-1 normalised) into 9 non-overlapping zones:

          TL  |  TOP  |  TR
         ─────┼───────┼─────
         LEFT |  CTR  | RIGHT
         ─────┼───────┼─────
          BL  |  BOT  |  BR

    Returns dict zone -> {energy_frac, pixel_frac, ratio}.
    ratio > 1 means attention is disproportionately in that zone.
    """
    h, w = cam.shape
    b = BORDER_PX
    total_e = float(cam.sum()) + 1e-9
    total_px = h * w

    def zone(r0, r1, c0, c1):
        patch = cam[r0:r1, c0:c1]
        px = (r1 - r0) * (c1 - c0)
        e = float(patch.sum()) / total_e
        p = px / total_px
        return {"energy_frac": e, "pixel_frac": p, "ratio": e / (p + 1e-9)}

    z = {
        "TL":    zone(0,   b,   0,   b),
        "TOP":   zone(0,   b,   b,   w-b),
        "TR":    zone(0,   b,   w-b, w),
        "LEFT":  zone(b,   h-b, 0,   b),
        "CTR":   zone(b,   h-b, b,   w-b),
        "RIGHT": zone(b,   h-b, w-b, w),
        "BL":    zone(h-b, h,   0,   b),
        "BOT":   zone(h-b, h,   b,   w-b),
        "BR":    zone(h-b, h,   w-b, w),
    }
    border_e = sum(v["energy_frac"] for k, v in z.items() if k != "CTR")
    border_p = sum(v["pixel_frac"]  for k, v in z.items() if k != "CTR")
    z["BORDER"] = {"energy_frac": border_e, "pixel_frac": border_p,
                   "ratio": border_e / (border_p + 1e-9)}
    return z


# ── raw border analysis ────────────────────────────────────────────────────────

def analyze_raw_border(raw_path, strip_px=20):
    """
    Analyse the 4 edge strips of the full-resolution raw JPG.

    Per side:
      mean / std of grayscale brightness across the whole strip
      outer3_mean / outer3_min  — outermost 3 px only
      profile_cv  — coefficient of variation of the 1-D mean profile along the
                    edge (high CV → periodic variation → ruler / ink marks)
    """
    img = cv2.imread(raw_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    h, w = gray.shape

    sides = {
        "top":    (gray[:strip_px, :],  gray[:3, :],  gray[:strip_px, :].mean(axis=0)),
        "bottom": (gray[h-strip_px:, :], gray[h-3:, :], gray[h-strip_px:, :].mean(axis=0)),
        "left":   (gray[:, :strip_px],  gray[:, :3],  gray[:, :strip_px].mean(axis=1)),
        "right":  (gray[:, w-strip_px:], gray[:, w-3:], gray[:, w-strip_px:].mean(axis=1)),
    }
    result = {}
    for side, (strip, outer, profile) in sides.items():
        result[side] = {
            "mean":        float(strip.mean()),
            "std":         float(strip.std()),
            "outer3_mean": float(outer.mean()),
            "outer3_min":  float(outer.min()),
            "profile_std": float(profile.std()),
            "profile_cv":  float(profile.std() / (profile.mean() + 1e-9)),
        }
    return result


def classify_side(s):
    o3 = s["outer3_mean"]
    cv = s["profile_cv"]
    mean = s["mean"]
    if o3 < 8:
        border_type = "DARK FRAME (near-black outer edge, uniform)"
    elif o3 < 25:
        if cv > 0.30:
            border_type = "RULER / INK MARKS (dark outer edge, high along-edge variation)"
        else:
            border_type = "dark border (moderate, uniform)"
    elif cv > 0.30:
        border_type = "RULER / INK MARKS (bright outer edge, high along-edge variation)"
    elif mean < 50:
        border_type = "dark content (skin/lesion reaching edge, no frame)"
    else:
        border_type = "normal (bright, no artifact)"
    return border_type


# ── visualisation ──────────────────────────────────────────────────────────────

def draw_zone_overlay(img_bgr, zones):
    """Annotate img_bgr in-place with zone energy fractions."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    b = BORDER_PX

    # Green rectangle marking the border / center boundary
    cv2.rectangle(out, (b, b), (w - b - 1, h - b - 1), (0, 220, 0), 1)

    def put(r, c, text):
        cv2.putText(out, text, (c, r), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0, 255, 255), 1, cv2.LINE_AA)

    # Corners
    put(12, 2,     f"TL={zones['TL']['energy_frac']:.2f}")
    put(12, w-b+1, f"TR={zones['TR']['energy_frac']:.2f}")
    put(h-4, 2,    f"BL={zones['BL']['energy_frac']:.2f}")
    put(h-4, w-b+1,f"BR={zones['BR']['energy_frac']:.2f}")
    # Sides
    put(12, b+4,   f"TOP={zones['TOP']['energy_frac']:.2f}")
    put(h-4, b+4,  f"BOT={zones['BOT']['energy_frac']:.2f}")
    put(h//2, 1,   f"L={zones['LEFT']['energy_frac']:.2f}")
    put(h//2, w-b+1,f"R={zones['RIGHT']['energy_frac']:.2f}")
    # Centre
    put(h//2, b+5, f"CTR={zones['CTR']['energy_frac']:.2f}")
    # Border aggregate
    put(h-4+0, b+40, f"[BORDER total={zones['BORDER']['energy_frac']:.2f} "
                     f"ratio={zones['BORDER']['ratio']:.1f}x]")
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}\n")

    net, arch = load_model(CHECKPOINT, device)
    cam_gen = GradCAM(net, get_target_layer(net, arch))

    summary_rows = []

    for img_id in TARGET_IDS:
        print(f"\n{'='*64}")
        print(f"  {img_id}")
        print(f"{'='*64}")

        # ── Grad-CAM ──
        cache_path = os.path.join(CACHE_DIR, f"{img_id}.png")
        pil = Image.open(cache_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        np_img = np.array(pil)        # RGB uint8
        tensor = NORM(T.ToTensor()(pil)).unsqueeze(0).to(device)
        tensor.requires_grad_()
        cam = cam_gen.generate(tensor, class_idx=1)
        overlay = overlay_heatmap(np_img, cam)   # RGB

        # ── Zone energy ──
        zones = cam_zone_energy(cam)
        print(f"\n  CAM zone energy  (BORDER_PX={BORDER_PX}, outer 15% of 224):")
        print(f"  {'Zone':<8} {'EnergyFrac':>11} {'PixelFrac':>10} {'Ratio':>7}")
        for z in ["TL", "TOP", "TR", "LEFT", "CTR", "RIGHT", "BL", "BOT", "BR"]:
            e = zones[z]["energy_frac"]
            p = zones[z]["pixel_frac"]
            r = zones[z]["ratio"]
            print(f"  {z:<8} {e:>11.3f} {p:>10.3f} {r:>7.2f}x")
        bd = zones["BORDER"]
        print(f"  {'BORDER':>8} {bd['energy_frac']:>11.3f} "
              f"{bd['pixel_frac']:>10.3f} {bd['ratio']:>7.2f}x  <-- aggregate")

        # dominant non-center zone
        dom = max(("TL","TOP","TR","LEFT","RIGHT","BL","BOT","BR"),
                  key=lambda z: zones[z]["energy_frac"])

        # ── Raw border ──
        raw_path = os.path.join(RAW_DIR, f"{img_id}.jpg")
        bstats = analyze_raw_border(raw_path)
        if bstats:
            print(f"\n  Raw border analysis (20 px strips, full-res JPG):")
            print(f"  {'Side':<8} {'mean':>6} {'std':>5} {'outer3_mean':>12} "
                  f"{'outer3_min':>11} {'profile_cv':>11}  classification")
            print(f"  {'-'*90}")
            for side in ["top", "bottom", "left", "right"]:
                s = bstats[side]
                cls = classify_side(s)
                print(f"  {side:<8} {s['mean']:>6.1f} {s['std']:>5.1f} "
                      f"{s['outer3_mean']:>12.1f} {s['outer3_min']:>11.1f} "
                      f"{s['profile_cv']:>11.3f}  {cls}")

        # ── Composite panel ──
        img_bgr     = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        zone_vis    = draw_zone_overlay(img_bgr.copy(), zones)
        panel = np.concatenate([img_bgr, overlay_bgr, zone_vis], axis=1)
        cv2.putText(panel, f"{img_id}  (BORDER={bd['energy_frac']:.2f}, "
                            f"ratio={bd['ratio']:.1f}x, dom={dom})",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out_path = os.path.join(OUT_DIR, f"border_{img_id}.jpg")
        cv2.imwrite(out_path, panel)
        print(f"\n  Panel -> {out_path}")

        summary_rows.append({
            "image_id":            img_id,
            "border_energy_frac":  round(bd["energy_frac"], 3),
            "border_energy_ratio": round(bd["ratio"], 2),
            "ctr_energy_frac":     round(zones["CTR"]["energy_frac"], 3),
            "dom_zone":            dom,
        })

    # ── Cross-ID summary ──
    print(f"\n\n{'='*64}")
    print("CROSS-ID SUMMARY")
    print(f"{'='*64}")
    print(f"  {'ID':<20} {'BORDER_FRAC':>12} {'BORDER_RATIO':>13} "
          f"{'CTR_FRAC':>10} {'DOM_ZONE':>9}")
    print(f"  {'-'*68}")
    for r in summary_rows:
        print(f"  {r['image_id']:<20} {r['border_energy_frac']:>12.3f} "
              f"{r['border_energy_ratio']:>13.2f}x "
              f"{r['ctr_energy_frac']:>10.3f} {r['dom_zone']:>9}")

    # uniform-attention baseline for interpretation
    border_px_frac = 1 - ((IMG_SIZE - 2*BORDER_PX)**2) / (IMG_SIZE**2)
    print(f"\n  Baseline (uniform attention): border_frac = {border_px_frac:.3f}, ratio = 1.00x")
    print(f"  Ratio >> 1 means attention is disproportionately at the border.")
    print(f"\n  Panels (original | CAM overlay | zone-energy annotated) -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
