"""
sanity_check_gradcam.py

Grad-CAM on top-k highest-confidence TRUE POSITIVES and TRUE NEGATIVES.
If these show on-lesion attention, then the FN off-lesion pattern is a real
attention failure, not a Grad-CAM computation artifact (e.g. padding bias).

Usage:
    python sanity_check_gradcam.py \
        --checkpoint model_output/effb0_unfreeze19_main/best_model.pt --tta
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, "model")
from dataset import lesion_aware_split, load_metadata, make_dataloaders
from finalize import collect_logits, load_nets, probs_pos, resolve_image_dir
from gradcam import GradCAM, overlay_heatmap
from model import build_model, get_target_layer

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.dirname(args.checkpoint)

    with open(os.path.join(run_dir, "deploy.json")) as f:
        deploy = json.load(f)
    T_scale, threshold, img_size = deploy["temperature"], deploy["threshold"], deploy["img_size"]

    image_dir, _ = resolve_image_dir(args.checkpoint)
    df_all = load_metadata(task_mode="binary")
    _, val_df, test_df = lesion_aware_split(df_all, seed=42)
    _, _, test_loader = make_dataloaders(val_df, val_df, test_df, image_dir, img_size=img_size)

    nets = load_nets([args.checkpoint], device)
    logits, labels = collect_logits(nets, test_loader, device, tta=args.tta)
    probs = probs_pos(logits, T_scale)

    test_df = test_df.reset_index(drop=True).copy()
    test_df["prob"] = probs
    preds = (probs >= threshold).astype(int)

    tp = test_df[(preds == 1) & (test_df["label"] == 1)].nlargest(args.k, "prob")
    tn = test_df[(preds == 0) & (test_df["label"] == 0)].nsmallest(args.k, "prob")

    ckpt = torch.load(args.checkpoint, map_location=device)
    net, _ = build_model(ckpt["arch"], num_classes=2, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.to(device).eval()
    cam_gen = GradCAM(net, get_target_layer(net, ckpt["arch"]))

    out_dir = os.path.join(run_dir, "analysis_sanity")
    os.makedirs(out_dir, exist_ok=True)

    for tag, group, class_idx in [("true_pos", tp, 1), ("true_neg", tn, 0)]:
        for _, row in group.iterrows():
            path = os.path.join(image_dir, f"{row['image_id']}.png")
            if not os.path.exists(path):
                continue
            pil = Image.open(path).convert("RGB").resize((img_size, img_size))
            np_img = np.array(pil)
            tensor = NORM(T.ToTensor()(pil)).unsqueeze(0).to(device)
            tensor.requires_grad_()
            cam = cam_gen.generate(tensor, class_idx=class_idx)
            overlay = overlay_heatmap(np_img, cam)
            side = np.concatenate([np_img, overlay], axis=1)
            fname = f"{tag}_prob{row['prob']:.3f}_{row['image_id']}.png"
            cv2.imwrite(os.path.join(out_dir, fname), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

    print(f"Saved {args.k} TP and {args.k} TN Grad-CAMs to {out_dir}/")
    print("Expected: TPs should show on-lesion heatmap (melanoma attended correctly).")
    print("If TPs ALSO show corner/edge activation: likely a padding artifact, not a real finding.")
    print("If TPs show clean on-lesion heatmap: FN corner-activation is a genuine attention failure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()
    main(args)
