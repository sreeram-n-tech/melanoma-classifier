"""
finalize.py  (project root)

Turns a trained checkpoint into an HONEST, calibrated, deployable operating point
-- in a single val -> test pass -- and writes model_output/<run>/deploy.json.

Why this exists (see the diagnosis in the plan):
  * The old evaluate.py tuned the decision threshold on the TEST set and then
    reported on that same set -> optimistic. Here the threshold is frozen on a
    held-out VALIDATION split and applied untouched to TEST.
  * k-fold tuned thresholds swung 0.05 -> 0.23 for the same 95%-sensitivity
    target, i.e. the raw operating point isn't transferable. Temperature scaling
    (Guo et al. 2017) fixes the probability scale so a fixed threshold means
    something, and so a displayed "62% melanoma" is meaningful.

Also supports, with explicit honesty caveats:
  * --tta   : test-time augmentation (average logits over flips), ~free robustness.
  * passing MULTIPLE --checkpoint paths : 5-fold ensemble (average logits). NOTE
    the fold models' training lesions can overlap the main test split, so the
    ensemble TEST number is a robustness indicator, not a clean headline.

Usage (run from project root):
    python finalize.py --checkpoint model_output/mobilenet_v2_weighted_ce_off_unfreeze19_main/best_model.pt
    python finalize.py --checkpoint <main> --tta
    python finalize.py --checkpoint <fold0> <fold1> <fold2> <fold3> <fold4>   # ensemble
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

# model/ holds dataset.py, model.py, evaluate.py and is what those scripts put on
# sys.path[0] when run directly; mirror that so we can reuse them unchanged.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

from dataset import (  # noqa: E402
    COLOR_NORM_CACHE_DIR, DEVIGNETTE_CACHE_DIR, DEVIGNETTE_RULER_CACHE_DIR, IMG_SIZE, PREPROCESSED_DIR, RAW_CACHE_DIR,
    lesion_aware_split, load_metadata, make_dataloaders,
)
from evaluate import tune_threshold  # noqa: E402  (reused, not re-implemented)
from model import build_model  # noqa: E402

# TTA views: identity, horizontal flip, vertical flip, both. Dermoscopy images
# have no canonical orientation, so these are all label-preserving.
TTA_OPS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.flip(x, dims=[2, 3]),
]


def load_nets(checkpoint_paths, device):
    """Loads one or more checkpoints into eval-mode models. All must share arch
    and binary task mode (enforced) so their logits can be averaged."""
    nets, archs = [], []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device)
        if ckpt.get("task_mode", "binary") != "binary":
            raise NotImplementedError(f"{path}: finalize.py supports binary only.")
        net, _ = build_model(ckpt["arch"], num_classes=2, pretrained=False)
        net.load_state_dict(ckpt["model_state"])
        net.to(device).eval()
        nets.append(net)
        archs.append(ckpt["arch"])
    if len(set(archs)) > 1:
        raise ValueError(f"Cannot ensemble different archs: {archs}")
    return nets


def collect_logits(nets, loader, device, tta=False):
    """Mean raw logits across all models (and, if tta, across flip views).
    Averaging in logit space keeps a single code path that calibration,
    thresholding and reporting all consume identically."""
    ops = TTA_OPS if tta else TTA_OPS[:1]
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            acc = None
            for net in nets:
                for op in ops:
                    out = net(op(imgs))
                    acc = out if acc is None else acc + out
            acc = acc / (len(nets) * len(ops))
            all_logits.append(acc.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels).numpy()


def fit_temperature(logits, labels):
    """Single-scalar temperature scaling: minimise NLL of softmax(logits / T) on
    the validation set. T > 1 softens over-confident logits, T < 1 sharpens."""
    logits = logits.detach()
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    nll = torch.nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nll(logits / T.clamp(min=1e-3), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def probs_pos(logits, T):
    """P(melanoma) after temperature scaling."""
    return torch.softmax(logits / T, dim=1)[:, 1].numpy()


def metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    auc = roc_auc_score(labels, probs)
    return {"auc": float(auc), "sensitivity": float(sens), "specificity": float(spec)}


def resolve_image_dir(first_checkpoint):
    """Match the preprocessing variant the checkpoint was trained with, exactly
    as evaluate.py does (reads its run_config.json)."""
    run_dir = os.path.dirname(first_checkpoint)
    preprocessing = "on"
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            preprocessing = json.load(f).get("preprocessing", "on")
    return {"on": PREPROCESSED_DIR, "off": RAW_CACHE_DIR,
            "color_norm": COLOR_NORM_CACHE_DIR,
            "devig": DEVIGNETTE_CACHE_DIR,
            "devig_ruler": DEVIGNETTE_RULER_CACHE_DIR}[preprocessing], preprocessing


def finalize(checkpoint_paths, target_sensitivity=0.95, tta=False, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir, preprocessing = resolve_image_dir(checkpoint_paths[0])

    df = load_metadata(task_mode="binary")
    _, val_df, test_df = lesion_aware_split(df, seed=seed)
    # make_dataloaders wants three frames; we only use the val and test loaders.
    _, val_loader, test_loader = make_dataloaders(
        val_df, val_df, test_df, image_dir, img_size=IMG_SIZE)

    nets = load_nets(checkpoint_paths, device)
    is_ensemble = len(nets) > 1

    # --- Calibrate on validation (single view, no TTA, for a stable T) ---
    val_logits, val_labels = collect_logits(nets, val_loader, device, tta=False)
    T = fit_temperature(val_logits, val_labels)

    # --- Freeze threshold on CALIBRATED validation probabilities ---
    val_probs = probs_pos(val_logits if not tta
                          else collect_logits(nets, val_loader, device, tta=True)[0], T)
    threshold, method, _, _ = tune_threshold(val_labels, val_probs, target_sensitivity)

    # --- Apply T + frozen threshold, untouched, to TEST ---
    test_logits, test_labels = collect_logits(nets, test_loader, device, tta=tta)
    test_probs = probs_pos(test_logits, T)
    test_tuned = metrics(test_labels, test_probs, threshold)
    test_default = metrics(test_labels, test_probs, 0.5)

    # --- Report ---
    print(f"\n=== finalize: {len(nets)} model(s){' + TTA' if tta else ''} "
          f"(preprocessing={preprocessing}) ===")
    print(f"Fitted temperature T = {T:.4f}")
    print(f"Frozen threshold = {threshold:.4f}  (chosen on VALIDATION, method: {method})")
    print(f"\n-- TEST @ default 0.5 --  AUC {test_default['auc']:.4f} | "
          f"sens {test_default['sensitivity']:.4f} | spec {test_default['specificity']:.4f}")
    print(f"-- TEST @ frozen thr --  AUC {test_tuned['auc']:.4f} | "
          f"sens {test_tuned['sensitivity']:.4f} | spec {test_tuned['specificity']:.4f}")
    print("(AUC is threshold-independent; sens/spec move with the frozen threshold.)")

    deploy = {
        "checkpoints": checkpoint_paths,
        "ensemble": is_ensemble,
        "tta": tta,
        "preprocessing": preprocessing,
        "img_size": IMG_SIZE,
        "temperature": T,
        "threshold": threshold,
        "threshold_source": "validation",
        "threshold_method": method,
        "test_auc": test_tuned["auc"],
        "test_sensitivity": test_tuned["sensitivity"],
        "test_specificity": test_tuned["specificity"],
    }
    if is_ensemble:
        deploy["caveat"] = ("Ensemble TEST metrics are a robustness indicator only: "
                            "fold training lesions may overlap the main test split.")
    out_path = os.path.join(os.path.dirname(checkpoint_paths[0]), "deploy.json")
    with open(out_path, "w") as f:
        json.dump(deploy, f, indent=2)
    print(f"\nWrote {out_path}")
    return deploy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="One checkpoint, or several to average as an ensemble.")
    parser.add_argument("--target_sensitivity", type=float, default=0.95)
    parser.add_argument("--tta", action="store_true",
                        help="Average logits over flip views at inference.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training split seed (default 42).")
    args = parser.parse_args()
    finalize(args.checkpoint, args.target_sensitivity, args.tta, args.seed)
