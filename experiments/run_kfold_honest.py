"""
experiments/run_kfold_honest.py

Honest per-fold evaluation harness for the leakage study.  Per fold:
  1. Train model on inner_train, selecting best checkpoint by inner_val AUC.
  2. Fit temperature T on single-view inner_val logits (NLL / LBFGS).
  3. Tune 95%-sensitivity threshold on TTA inner_val probabilities.
  4. Score held_out with TTA + frozen (T, threshold).
  5. Record: fold, T, frozen_threshold, achieved_sensitivity,
             achieved_specificity, auc.

Hard aborts:
  - Training crashes or stdout contains NaN in a loss line.
  - achieved_sensitivity on held_out < SENS_ABORT (0.90).

Usage:
    python experiments/run_kfold_honest.py --preprocessing off --arm baseline
    python experiments/run_kfold_honest.py --preprocessing devig_ruler --arm masked
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, ROOT)

from dataset import IMG_SIZE, load_metadata, make_dataloaders, _BASE_CACHE_DIR  # noqa: E402
from finalize import collect_logits, fit_temperature, metrics, probs_pos         # noqa: E402
from evaluate import tune_threshold                                               # noqa: E402
from model import build_model                                                     # noqa: E402

SPLIT_DIR        = os.path.join(ROOT, "experiments", "honest_splits")
MODEL_OUTPUT     = os.path.join(ROOT, "model_output")
N_FOLDS          = 5
TARGET_SENS      = 0.95
SENS_ABORT       = 0.90
LOG_FILE         = os.path.join(ROOT, "run_log.txt")


# ── helpers ──────────────────────────────────────────────────────────────────

def log(msg, also_print=True):
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _nan_in_log(path):
    try:
        with open(path) as f:
            content = f.read().lower()
        for line in content.split("\n"):
            if "nan" in line and ("loss" in line or "train" in line or "val" in line):
                return True
    except OSError:
        pass
    return False


# ── per-fold training ─────────────────────────────────────────────────────────

def train_fold(preprocessing, fold_idx, arm_name, extra_train_args=None):
    """Invoke train.py for one fold; return path to best_model.pt."""
    run_tag   = f"honest_{arm_name}_fold{fold_idx}"
    out_dir   = os.path.join(MODEL_OUTPUT, run_tag)
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    log_path  = os.path.join(out_dir, "train_stdout.log")
    split_json = os.path.join(SPLIT_DIR, f"fold{fold_idx}.json")

    if os.path.exists(ckpt_path):
        log(f"  fold{fold_idx}: checkpoint already exists, skipping training.")
        return ckpt_path

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(ROOT, "model", "train.py"),
        "--arch",           "efficientnet_b0",
        "--preprocessing",  preprocessing,
        "--epochs_stage1",  "5",
        "--epochs_stage2",  "10",
        "--lr_stage2",      "1e-4",
        "--unfreeze_blocks","19",
        "--seed",           "42",
        "--split_json",     split_json,
        "--run_name",       run_tag,
        "--resume",                        # no-op if last_checkpoint.pt absent
    ]
    if extra_train_args:
        cmd.extend(extra_train_args)

    log(f"  fold{fold_idx}: launching {run_tag} ...")
    t0 = time.time()
    with open(log_path, "w") as lf:
        result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"[HARD ABORT] Training failed for {run_tag} "
            f"(exit code {result.returncode}). See {log_path}"
        )
    if _nan_in_log(log_path):
        raise RuntimeError(
            f"[HARD ABORT] NaN detected in training log for {run_tag}. "
            f"See {log_path}"
        )

    log(f"  fold{fold_idx}: training done in {elapsed/60:.1f} min.")
    return ckpt_path


# ── per-fold evaluation ───────────────────────────────────────────────────────

def evaluate_fold(ckpt_path, fold_idx, preprocessing):
    """Calibrate on inner_val; score held_out.  Returns result dict."""
    fold_json = os.path.join(SPLIT_DIR, f"fold{fold_idx}.json")
    with open(fold_json) as f:
        splits = json.load(f)

    image_dir = _BASE_CACHE_DIR[preprocessing]
    df        = load_metadata()
    iv_df     = df[df["image_id"].isin(splits["inner_val"])].reset_index(drop=True)
    ho_df     = df[df["image_id"].isin(splits["held_out"])].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    net, _ = build_model(ckpt.get("arch", "efficientnet_b0"), num_classes=2, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.to(device).eval()

    # dataloaders: pass same df for train/val/test; we only use val_loader / test_loader
    _, val_loader, _  = make_dataloaders(iv_df, iv_df, iv_df, image_dir,
                                         batch_size=64, img_size=IMG_SIZE)
    _, _, test_loader = make_dataloaders(ho_df, ho_df, ho_df, image_dir,
                                         batch_size=64, img_size=IMG_SIZE)

    # 1. Fit T on single-view inner_val (no TTA for numerical stability)
    val_logits_sv, val_labels = collect_logits([net], val_loader, device, tta=False)
    T = fit_temperature(val_logits_sv, val_labels)

    # 2. Tune threshold on TTA inner_val
    val_logits_tta, _ = collect_logits([net], val_loader, device, tta=True)
    val_probs         = probs_pos(val_logits_tta, T)
    threshold, method, _, _ = tune_threshold(val_labels, val_probs, TARGET_SENS)

    # 3. Score held_out with TTA + frozen T + threshold
    test_logits, test_labels = collect_logits([net], test_loader, device, tta=True)
    test_probs               = probs_pos(test_logits, T)
    m                        = metrics(test_labels, test_probs, threshold)

    result = {
        "fold":                  fold_idx,
        "T":                     float(T),
        "frozen_threshold":      float(threshold),
        "threshold_method":      method,
        "achieved_sensitivity":  float(m["sensitivity"]),
        "achieved_specificity":  float(m["specificity"]),
        "auc":                   float(m["auc"]),
        "n_inner_val":           len(iv_df),
        "n_held_out":            len(ho_df),
    }

    if result["achieved_sensitivity"] < SENS_ABORT:
        raise RuntimeError(
            f"[HARD ABORT] fold{fold_idx}: achieved_sensitivity "
            f"{result['achieved_sensitivity']:.3f} < {SENS_ABORT}. "
            f"Calibration may have broken."
        )
    return result


# ── arm runner ────────────────────────────────────────────────────────────────

def run_arm(preprocessing, arm_name, extra_train_args=None):
    """Train and evaluate all 5 folds for one preprocessing arm."""
    log(f"=== ARM '{arm_name}' (preprocessing={preprocessing}) START ===")
    fold_results = []

    for fold_idx in range(N_FOLDS):
        log(f"--- fold{fold_idx} ---")
        ckpt_path = train_fold(preprocessing, fold_idx, arm_name, extra_train_args)
        log(f"  fold{fold_idx}: evaluating ...")
        result    = evaluate_fold(ckpt_path, fold_idx, preprocessing)
        fold_results.append(result)
        log(f"  fold{fold_idx}: T={result['T']:.3f}  thr={result['frozen_threshold']:.3f}  "
            f"sens={result['achieved_sensitivity']:.3f}  "
            f"spec={result['achieved_specificity']:.3f}  auc={result['auc']:.4f}")

    specs = [r["achieved_specificity"]  for r in fold_results]
    senss = [r["achieved_sensitivity"]  for r in fold_results]
    aucs  = [r["auc"]                   for r in fold_results]

    summary = {
        "arm":              arm_name,
        "preprocessing":    preprocessing,
        "folds":            fold_results,
        "mean_sensitivity": float(np.mean(senss)),
        "std_sensitivity":  float(np.std(senss)),
        "mean_specificity": float(np.mean(specs)),
        "std_specificity":  float(np.std(specs)),
        "mean_auc":         float(np.mean(aucs)),
        "std_auc":          float(np.std(aucs)),
    }

    out_path = os.path.join(MODEL_OUTPUT, f"kfold_honest_{arm_name}_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"=== ARM '{arm_name}' DONE ===")
    log(f"    spec@val95: {np.mean(specs):.4f} +/- {np.std(specs):.4f}")
    log(f"    AUC:        {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    log(f"    Summary  -> {out_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Honest k-fold evaluation for one arm.")
    ap.add_argument("--preprocessing", required=True,
                    choices=["on", "off", "color_norm", "devig", "devig_ruler"],
                    help="Preprocessing mode (selects cache directory).")
    ap.add_argument("--arm", required=True,
                    help="Short arm name used in output file names (e.g. 'baseline', 'masked').")
    args = ap.parse_args()
    run_arm(args.preprocessing, args.arm)
