"""
compare_roc.py  (project root)

Full ROC comparison between two (or more) checkpoints on the TEST split.

Prints:
  1. Specificity at matched sensitivity levels (80%, 85%, 90%, 93%, 95%, 97%)
     so you can compare models at the same operating point instead of each
     model's own frozen threshold.
  2. Full ROC table (every unique sensitivity step) for each model.
  3. AUC with bootstrap 95% CI so you can tell signal from noise.
     With n_pos=167 in the HAM10000 test split, SE(AUC) ≈ 0.025 --
     differences smaller than ~0.02 are not reliably distinguishable.

Usage (from project root):
    python compare_roc.py \
        --checkpoints \
            model_output/mobilenet_v2_weighted_ce_off_unfreeze19_main/best_model.pt \
            model_output/mv2_recipe_v3/best_model.pt \
        --labels baseline v3
"""
import argparse
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, "model")
from dataset import IMG_SIZE, lesion_aware_split, load_metadata, make_dataloaders
from model import build_model


# ── inference ────────────────────────────────────────────────────────────────

def load_and_infer(checkpoint_path, test_loader, device):
    """Returns (probs_melanoma, labels) for the test set."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    net, _ = build_model(ckpt["arch"], num_classes=2, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.to(device).eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            p = torch.softmax(net(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
            all_probs.append(p)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── AUC bootstrap CI ─────────────────────────────────────────────────────────

def bootstrap_auc(labels, probs, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(labels), len(labels), replace=True)
        lb, pb = labels[idx], probs[idx]
        if lb.sum() == 0 or lb.sum() == len(lb):
            continue
        aucs.append(roc_auc_score(lb, pb))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── matched operating points ──────────────────────────────────────────────────

TARGET_SENS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]

def spec_at_sens(fpr, tpr, thresholds, target):
    """Specificity (= 1-FPR) at the lowest threshold reaching >= target sensitivity.
    Returns (specificity, threshold, actual_sensitivity) or (nan, nan, nan)."""
    candidates = np.where(tpr >= target)[0]
    if len(candidates) == 0:
        return float("nan"), float("nan"), float("nan")
    # Among candidates, take the one with highest specificity (lowest FPR).
    best = candidates[np.argmin(fpr[candidates])]
    return float(1 - fpr[best]), float(thresholds[best]), float(tpr[best])


# ── printing ─────────────────────────────────────────────────────────────────

def print_matched_table(results):
    """results: list of (label, auc, ci_lo, ci_hi, fpr, tpr, thresholds)"""
    col_w = 14
    header = f"{'sens_target':>12}  " + "  ".join(
        f"{'spec@'+r[0]:>{col_w}}" for r in results)
    print(header)
    print("-" * len(header))
    for t in TARGET_SENS:
        row = f"{t:>12.0%}  "
        for label, auc, ci_lo, ci_hi, fpr, tpr, thr in results:
            sp, th, actual_s = spec_at_sens(fpr, tpr, thr, t)
            cell = f"{sp:.4f} (thr={th:.3f})" if not np.isnan(sp) else "n/a"
            row += f"{cell:>{col_w}}  "
        print(row)


def print_full_roc(label, fpr, tpr, thresholds, probs, labels):
    """Print one row per UNIQUE sensitivity step, deduplicated."""
    print(f"\n--- Full ROC: {label} ---")
    print(f"{'threshold':>12}  {'sensitivity':>13}  {'specificity':>13}  {'PPV':>8}")
    prev_tpr = -1.0
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    for th, tp_rate, fp_rate in zip(thresholds, tpr, fpr):
        if abs(tp_rate - prev_tpr) < 0.005:  # skip near-duplicates
            continue
        prev_tpr = tp_rate
        sp = 1.0 - fp_rate
        tp = tp_rate * n_pos
        fp = fp_rate * n_neg
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        print(f"{th:>12.4f}  {tp_rate:>13.4f}  {sp:>13.4f}  {ppv:>8.4f}")


# ── main ─────────────────────────────────────────────────────────────────────

def resolve_image_dir(checkpoint_path):
    import os, json as _json
    from dataset import COLOR_NORM_CACHE_DIR, PREPROCESSED_DIR, RAW_CACHE_DIR
    run_dir = os.path.dirname(checkpoint_path)
    cfg = os.path.join(run_dir, "run_config.json")
    preprocessing = "on"
    if os.path.exists(cfg):
        with open(cfg) as f:
            preprocessing = _json.load(f).get("preprocessing", "on")
    return {"on": PREPROCESSED_DIR, "off": RAW_CACHE_DIR,
            "color_norm": COLOR_NORM_CACHE_DIR}[preprocessing]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # All checkpoints must share preprocessing config; use the first one's.
    image_dir = resolve_image_dir(args.checkpoints[0])
    df = load_metadata(task_mode="binary")
    _, val_df, test_df = lesion_aware_split(df, seed=args.seed)
    _, _, test_loader = make_dataloaders(
        val_df, val_df, test_df, image_dir, img_size=IMG_SIZE)

    n_pos = int((test_df["label"] == 1).sum())
    n_neg = int((test_df["label"] == 0).sum())
    se_approx = (0.883 * 0.117 / n_pos) ** 0.5  # rough, using 0.883 as AUC proxy
    print(f"\nTest set: {len(test_df)} images, {n_pos} melanoma, {n_neg} benign")
    print(f"Approx SE(AUC) ≈ {se_approx:.3f} — differences < {2*se_approx:.2f} "
          f"are not reliably distinguishable on this test set size.\n")

    labels_ref = None
    results = []

    for ckpt_path, lbl in zip(args.checkpoints, args.labels):
        probs, labels = load_and_infer(ckpt_path, test_loader, device)
        if labels_ref is None:
            labels_ref = labels
        else:
            assert (labels == labels_ref).all(), "label mismatch between checkpoints"

        auc = roc_auc_score(labels, probs)
        ci_lo, ci_hi = bootstrap_auc(labels, probs)
        fpr, tpr, thresholds = roc_curve(labels, probs)

        print(f"{lbl}:  AUC = {auc:.4f}  (95% CI {ci_lo:.4f}–{ci_hi:.4f})")
        results.append((lbl, auc, ci_lo, ci_hi, fpr, tpr, thresholds, probs, labels))

    print(f"\n=== Specificity at matched sensitivity levels ===")
    print_matched_table([(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in results])

    for lbl, auc, ci_lo, ci_hi, fpr, tpr, thresholds, probs, labels in results:
        print_full_roc(lbl, fpr, tpr, thresholds, probs, labels)

    # Quick CI overlap summary
    print(f"\n=== Overlap check ===")
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            a, b = results[i], results[j]
            overlap = a[2] < b[3] and b[2] < a[3]
            print(f"  {a[0]} CI [{a[2]:.4f}–{a[3]:.4f}] vs "
                  f"{b[0]} CI [{b[2]:.4f}–{b[3]:.4f}]: "
                  f"{'CIs OVERLAP — difference not significant' if overlap else 'CIs do NOT overlap — difference is significant'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Short label for each checkpoint (same order).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.labels):
        parser.error("--checkpoints and --labels must have the same length")
    main(args)
