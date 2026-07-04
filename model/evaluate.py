"""
model/evaluate.py

Loads a trained checkpoint, runs inference on its held-out test set, tunes the
decision threshold off the ROC curve, and reports precision/recall/F1,
sensitivity/specificity, ROC-AUC, a confusion matrix, and a saved plot.

Threshold tuning default: lowest threshold that still hits a target
sensitivity (default 0.95) -- i.e. optimized to catch melanoma, accepting more
false alarms, rather than the default 0.5 cutoff which has no clinical
justification for a screening tool. Falls back to Youden's J statistic if the
target sensitivity isn't reachable at all.

Usage (from the project root):
    python model/evaluate.py --checkpoint model_output/mobilenet_v2_weighted_ce_on_main/best_model.pt
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from dataset import (
    COLOR_NORM_CACHE_DIR, DEVIGNETTE_CACHE_DIR, DEVIGNETTE_RULER_CACHE_DIR, IMG_SIZE, PREPROCESSED_DIR, RAW_CACHE_DIR,
    lesion_aware_kfold, lesion_aware_split, load_metadata, make_dataloaders,
)
from model import build_model


def tune_threshold(labels, probs_pos, target_sensitivity=0.95):
    """
    Picks the lowest threshold that still achieves >= target_sensitivity
    (recall on the melanoma class). Falls back to Youden's J (max
    sensitivity + specificity - 1) if no threshold reaches the target.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs_pos)
    sensitivity = tpr
    specificity = 1 - fpr

    candidates = np.where(sensitivity >= target_sensitivity)[0]
    if len(candidates) > 0:
        best_idx = candidates[np.argmax(specificity[candidates])]
        method = f"target_sensitivity>={target_sensitivity}"
    else:
        j_scores = sensitivity + specificity - 1
        best_idx = int(np.argmax(j_scores))
        method = "youdens_j (target sensitivity unreachable)"
    return float(thresholds[best_idx]), method, fpr, tpr


def evaluate(checkpoint_path, target_sensitivity=0.95, fold=None, n_folds=5, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch = ckpt["arch"]
    task_mode = ckpt["task_mode"]
    if task_mode != "binary":
        raise NotImplementedError("evaluate.py currently supports binary classification only.")

    run_dir = os.path.dirname(checkpoint_path)
    run_config_path = os.path.join(run_dir, "run_config.json")
    preprocessing = "on"
    if os.path.exists(run_config_path):
        with open(run_config_path) as f:
            preprocessing = json.load(f).get("preprocessing", "on")
    image_dir = {"on": PREPROCESSED_DIR, "off": RAW_CACHE_DIR,
                 "color_norm": COLOR_NORM_CACHE_DIR,
                 "devig": DEVIGNETTE_CACHE_DIR,
                 "devig_ruler": DEVIGNETTE_RULER_CACHE_DIR}[preprocessing]

    df = load_metadata(task_mode=task_mode)
    if fold is not None:
        splits = list(lesion_aware_kfold(df, n_splits=n_folds, seed=seed))
        _, test_df = splits[fold]
    else:
        _, _, test_df = lesion_aware_split(df, seed=seed)

    net, _ = build_model(arch, num_classes=2, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.to(device)
    net.eval()

    # train/val loaders below are unused placeholders; make_dataloaders needs all three.
    _, _, test_loader = make_dataloaders(test_df, test_df, test_df, image_dir, img_size=IMG_SIZE)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            probs = torch.softmax(net(imgs), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    probs_pos = probs[:, 1]

    auc = roc_auc_score(labels, probs_pos)
    threshold, method, fpr, tpr = tune_threshold(labels, probs_pos, target_sensitivity)

    preds_default = (probs_pos >= 0.5).astype(int)
    preds_tuned = (probs_pos >= threshold).astype(int)

    print(f"\n=== Evaluation: {checkpoint_path} ===")
    print(f"ROC-AUC: {auc:.4f}")
    print("\n--- Default threshold (0.5) ---")
    print(classification_report(labels, preds_default, target_names=["not_melanoma", "melanoma"]))
    print(f"\n--- Tuned threshold ({threshold:.3f}, method: {method}) ---")
    print(classification_report(labels, preds_tuned, target_names=["not_melanoma", "melanoma"]))

    cm = confusion_matrix(labels, preds_tuned)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    print(f"\nAt tuned threshold: sensitivity {sensitivity:.4f}, specificity {specificity:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].scatter([1 - specificity], [sensitivity], color="red", zorder=5,
                     label=f"tuned thresh = {threshold:.2f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["not_mel", "mel"])
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["not_mel", "mel"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    out_path = os.path.join(run_dir, "evaluation_report.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plots to {out_path}")

    results = {
        "checkpoint": checkpoint_path, "auc": float(auc), "tuned_threshold": threshold,
        "threshold_method": method, "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
    with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target_sensitivity", type=float, default=0.95)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.target_sensitivity, args.fold, args.n_folds)
