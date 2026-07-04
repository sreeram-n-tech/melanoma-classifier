"""
external_eval.py  (project root)

Generalisation test: run a FROZEN checkpoint (no retraining) on a *different*
dataset -- PH2, ISIC 2019/2020, etc. -- applying the temperature + decision
threshold from deploy.json untouched. A model that only works on HAM10000's
test split hasn't been shown to generalise; this is the number that proves it.

This stays dataset-agnostic: you point it at an images folder + a labels CSV and
say which column is the filename and which values count as melanoma. (PH2 and
ISIC both ship a CSV/spreadsheet you can pass directly.)

Usage (from project root):
    python external_eval.py \
        --checkpoint model_output/mobilenet_v2_weighted_ce_off_unfreeze19_main/best_model.pt \
        --images_dir data/external/ph2/images \
        --labels_csv data/external/ph2/labels.csv \
        --image_col image --label_col diagnosis --positive_values melanoma,mel

Reads deploy.json (written by finalize.py) from the first checkpoint's run dir
for the frozen temperature + threshold -- run finalize.py first.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Reuse the exact load/forward/calibrate/metrics helpers from finalize.py
# (importing it also puts model/ on sys.path).
from finalize import collect_logits, load_nets, metrics, probs_pos  # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_EXTS = ["", ".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]


class ExternalDataset(Dataset):
    """Loads (image, binary_label) using the SAME inference transform as
    training val/test: resize -> ToTensor -> ImageNet normalize."""
    def __init__(self, rows, images_dir, img_size):
        self.rows = rows  # list of (filename, label)
        self.images_dir = images_dir
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), NORM])

    def __len__(self):
        return len(self.rows)

    def _resolve(self, filename):
        for ext in _EXTS:
            path = os.path.join(self.images_dir, f"{filename}{ext}")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No image for '{filename}' in {self.images_dir} "
                                f"(tried extensions {_EXTS[1:]}).")

    def __getitem__(self, idx):
        filename, label = self.rows[idx]
        img = Image.open(self._resolve(filename)).convert("RGB")
        return self.transform(img), label


def build_rows(labels_csv, image_col, label_col, positive_values):
    df = pd.read_csv(labels_csv)
    for col in (image_col, label_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not in {labels_csv}. Found: {list(df.columns)}")
    positives = {v.strip().lower() for v in positive_values.split(",")}
    rows = []
    for _, r in df.iterrows():
        label = int(str(r[label_col]).strip().lower() in positives)
        rows.append((str(r[image_col]), label))
    return rows


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deploy_path = args.deploy or os.path.join(os.path.dirname(args.checkpoint[0]), "deploy.json")
    if not os.path.exists(deploy_path):
        raise FileNotFoundError(f"{deploy_path} not found -- run finalize.py first.")
    with open(deploy_path) as f:
        deploy = json.load(f)
    T_scale, threshold, img_size = deploy["temperature"], deploy["threshold"], deploy["img_size"]

    rows = build_rows(args.labels_csv, args.image_col, args.label_col, args.positive_values)
    n_pos = sum(lbl for _, lbl in rows)
    print(f"External set: {len(rows)} images, {n_pos} melanoma ({n_pos / len(rows):.1%}).")

    loader = DataLoader(ExternalDataset(rows, args.images_dir, img_size),
                        batch_size=args.batch_size, shuffle=False, num_workers=2)
    nets = load_nets(args.checkpoint, device)
    logits, labels = collect_logits(nets, loader, device, tta=args.tta)
    probs = probs_pos(logits, T_scale)

    ext_tuned = metrics(labels, probs, threshold)
    ext_default = metrics(labels, probs, 0.5)

    print(f"\n=== External eval (frozen T={T_scale:.3f}, threshold={threshold:.4f}"
          f"{', +TTA' if args.tta else ''}) ===")
    print(f"-- @ frozen threshold --  AUC {ext_tuned['auc']:.4f} | "
          f"sens {ext_tuned['sensitivity']:.4f} | spec {ext_tuned['specificity']:.4f}")
    print(f"-- @ default 0.5      --  AUC {ext_default['auc']:.4f} | "
          f"sens {ext_default['sensitivity']:.4f} | spec {ext_default['specificity']:.4f}")
    print(f"\nFor reference, internal HAM10000 test AUC was {deploy.get('test_auc'):.4f}. "
          f"A large drop here = limited generalisation (expected to some degree across "
          f"devices/populations -- report it honestly).")

    out = os.path.join(os.path.dirname(deploy_path), "external_eval.json")
    with open(out, "w") as f:
        json.dump({"labels_csv": args.labels_csv, "n": len(rows), "n_pos": n_pos,
                   "tta": args.tta, "at_frozen_threshold": ext_tuned,
                   "at_default_0.5": ext_default,
                   "internal_test_auc": deploy.get("test_auc")}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, nargs="+")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--image_col", default="image")
    parser.add_argument("--label_col", default="diagnosis")
    parser.add_argument("--positive_values", default="melanoma,mel",
                        help="Comma-separated label values that mean melanoma (case-insensitive).")
    parser.add_argument("--deploy", default=None, help="Path to deploy.json (default: next to checkpoint).")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
