"""
analyze.py  (project root)

Error analysis on the HAM10000 test split for a finalized checkpoint:

  1. Subgroup metrics -- AUC / sensitivity / specificity broken down by
     localization, sex, and age band, at the FROZEN deploy threshold. Exposes
     where the model is weak (e.g. a body site or age group it misses), which a
     single headline number hides.

  2. Grad-CAM artifact audit -- saves heatmap overlays for the most confident
     MISTAKES (top false positives + top false negatives). If the model is firing
     on rulers / ink / vignette instead of the lesion, you'll see it here, and
     that's a concrete fix (crop/mask) rather than a mystery.

Reads deploy.json (from finalize.py) for the temperature + threshold.

Usage (from project root):
    python analyze.py --checkpoint model_output/mobilenet_v2_weighted_ce_off_unfreeze19_main/best_model.pt
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

# finalize import also puts model/ on sys.path for the gradcam/model imports below.
from finalize import collect_logits, load_nets, probs_pos, resolve_image_dir  # noqa: E402
from dataset import lesion_aware_split, load_metadata, make_dataloaders  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
from model import build_model, get_target_layer  # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def subgroup_table(df, threshold, by):
    """Per-group n / melanoma count / AUC / sens / spec at the frozen threshold."""
    print(f"\n--- by {by} ---")
    print(f"{by:<18} {'n':>5} {'n_mel':>6} {'AUC':>7} {'sens':>7} {'spec':>7}")
    for key, g in df.groupby(by):
        n, n_pos = len(g), int(g["label"].sum())
        preds = (g["prob"].values >= threshold).astype(int)
        tp = int(((preds == 1) & (g["label"] == 1)).sum())
        fn = int(((preds == 0) & (g["label"] == 1)).sum())
        tn = int(((preds == 0) & (g["label"] == 0)).sum())
        fp = int(((preds == 1) & (g["label"] == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        try:
            auc = roc_auc_score(g["label"], g["prob"]) if 0 < n_pos < n else float("nan")
        except ValueError:
            auc = float("nan")
        print(f"{str(key):<18} {n:>5} {n_pos:>6} {auc:>7.3f} {sens:>7.3f} {spec:>7.3f}")


def age_band(age):
    if pd.isna(age):
        return "unknown"
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    return "60+"


def gradcam_mistakes(df, checkpoint, image_dir, img_size, threshold, out_dir, k=6):
    """Grad-CAM overlays for the top-k most confident false positives and false
    negatives -- the cases most worth eyeballing for artifact reliance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    net, _ = build_model(ckpt["arch"], num_classes=2, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.to(device).eval()
    cam_gen = GradCAM(net, get_target_layer(net, ckpt["arch"]))

    preds = (df["prob"].values >= threshold).astype(int)
    fp = df[(preds == 1) & (df["label"] == 0)].nlargest(k, "prob")
    fn = df[(preds == 0) & (df["label"] == 1)].nsmallest(k, "prob")

    os.makedirs(out_dir, exist_ok=True)
    for tag, group in [("false_pos", fp), ("false_neg", fn)]:
        for _, row in group.iterrows():
            path = os.path.join(image_dir, f"{row['image_id']}.png")
            if not os.path.exists(path):
                print(f"  (skip {row['image_id']}: not in cache {image_dir})")
                continue
            pil = Image.open(path).convert("RGB").resize((img_size, img_size))
            np_img = np.array(pil)
            tensor = NORM(T.ToTensor()(pil)).unsqueeze(0).to(device)
            tensor.requires_grad_()
            cam = cam_gen.generate(tensor, class_idx=1)  # explain the melanoma logit
            overlay = overlay_heatmap(np_img, cam)
            side = np.concatenate([np_img, overlay], axis=1)
            out = os.path.join(out_dir, f"{tag}_{row['prob']:.2f}_{row['image_id']}.png")
            cv2.imwrite(out, cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    print(f"\nSaved Grad-CAM mistake overlays to {out_dir}/ "
          f"({len(fp)} false-pos, {len(fn)} false-neg). Open them and check the heatmap "
          f"sits on the lesion, not on hair/ruler/ink/border.")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(run_dir, "deploy.json")) as f:
        deploy = json.load(f)
    T_scale, threshold, img_size = deploy["temperature"], deploy["threshold"], deploy["img_size"]

    image_dir, _ = resolve_image_dir(args.checkpoint)
    df_all = load_metadata(task_mode="binary")
    _, val_df, test_df = lesion_aware_split(df_all, seed=args.seed)

    _, _, test_loader = make_dataloaders(val_df, val_df, test_df, image_dir, img_size=img_size)
    nets = load_nets([args.checkpoint], device)
    logits, labels = collect_logits(nets, test_loader, device, tta=args.tta)
    probs = probs_pos(logits, T_scale)

    # test_loader has shuffle=False, so probs line up row-for-row with test_df.
    test_df = test_df.reset_index(drop=True).copy()
    assert len(test_df) == len(probs), "row/probability misalignment"
    test_df["prob"] = probs
    test_df["age_band"] = test_df["age"].map(age_band)

    print(f"=== Subgroup analysis (T={T_scale:.3f}, frozen threshold={threshold:.4f}) ===")
    subgroup_table(test_df, threshold, "localization")
    subgroup_table(test_df, threshold, "sex")
    subgroup_table(test_df, threshold, "age_band")

    gradcam_mistakes(test_df, args.checkpoint, image_dir, img_size, threshold,
                     out_dir=os.path.join(run_dir, "analysis"), k=args.k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--k", type=int, default=6, help="How many top mistakes to visualize per type.")
    args = parser.parse_args()
    main(args)
