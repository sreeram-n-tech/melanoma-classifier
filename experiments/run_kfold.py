"""
experiments/run_kfold.py

Runs lesion-ID-aware K-fold cross-validation for a given config (default:
5 folds) and reports mean +/- std metrics across folds -- the standard way
to show test-set numbers aren't a fluke of one particular train/test split.

Usage (from the project root):
    python experiments/run_kfold.py
    python experiments/run_kfold.py --arch efficientnet_b0 --n_folds 5
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_kfold(arch, loss, preprocessing, n_folds, target_sensitivity,
              unfreeze_blocks=3, run_name_prefix=None, extra_train_args=None):
    unfreeze_suffix = "" if unfreeze_blocks == 3 else f"_unfreeze{unfreeze_blocks}"
    extra_train_args = extra_train_args or []
    fold_results = []
    for fold in range(n_folds):
        # If a custom prefix is given (new recipe), use it so results don't
        # overwrite the baseline k-fold dirs.
        if run_name_prefix:
            tag = f"{run_name_prefix}_fold{fold}"
        else:
            tag = f"{arch}_{loss}_{preprocessing}{unfreeze_suffix}_fold{fold}"
        print(f"\n=== Fold {fold + 1}/{n_folds} ({tag}) ===")
        train_cmd = [sys.executable, os.path.join(ROOT, "model", "train.py"),
                     "--arch", arch, "--loss", loss, "--preprocessing", preprocessing,
                     "--fold", str(fold), "--n_folds", str(n_folds),
                     "--unfreeze_blocks", str(unfreeze_blocks),
                     "--run_name", tag] + extra_train_args
        subprocess.run(train_cmd, cwd=ROOT, check=True)
        ckpt = os.path.join(ROOT, "model_output", tag, "best_model.pt")
        subprocess.run([sys.executable, os.path.join(ROOT, "model", "evaluate.py"),
                         "--checkpoint", ckpt, "--fold", str(fold), "--n_folds", str(n_folds),
                         "--target_sensitivity", str(target_sensitivity)], cwd=ROOT, check=True)
        with open(os.path.join(ROOT, "model_output", tag, "eval_results.json")) as f:
            fold_results.append(json.load(f))

    aucs = [r["auc"] for r in fold_results]
    sens = [r["sensitivity"] for r in fold_results]
    spec = [r["specificity"] for r in fold_results]

    desc = run_name_prefix if run_name_prefix else f"{arch}_{loss}_{preprocessing}{unfreeze_suffix}"
    print(f"\n=== {n_folds}-Fold CV Summary ({desc}) ===")
    print(f"AUC:          {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"Sensitivity:  {np.mean(sens):.4f} +/- {np.std(sens):.4f}")
    print(f"Specificity:  {np.mean(spec):.4f} +/- {np.std(spec):.4f}")

    summary = {
        "per_fold": fold_results,
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "mean_sensitivity": float(np.mean(sens)), "std_sensitivity": float(np.std(sens)),
        "mean_specificity": float(np.mean(spec)), "std_specificity": float(np.std(spec)),
    }
    out_path = os.path.join(ROOT, "model_output", f"kfold_{desc}_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="mobilenet_v2")
    parser.add_argument("--loss", default="weighted_ce")
    parser.add_argument("--preprocessing", default="on")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--target_sensitivity", type=float, default=0.95)
    parser.add_argument("--unfreeze_blocks", type=int, default=3)
    parser.add_argument("--run_name_prefix", default=None,
                        help="Prefix for fold output dirs, e.g. 'mv2_v4'. "
                             "Keeps new-recipe folds separate from baseline folds.")
    # Recipe flags forwarded verbatim to train.py.
    parser.add_argument("--epochs_stage1", type=int, default=None)
    parser.add_argument("--epochs_stage2", type=int, default=None)
    parser.add_argument("--schedule", default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--strong_aug", action="store_true")
    args = parser.parse_args()

    # Build extra_train_args from only the flags that were explicitly set.
    extra = []
    for flag, val in [("--epochs_stage1", args.epochs_stage1),
                      ("--epochs_stage2", args.epochs_stage2),
                      ("--schedule", args.schedule),
                      ("--warmup_epochs", args.warmup_epochs),
                      ("--ema_decay", args.ema_decay),
                      ("--mixup_alpha", args.mixup_alpha),
                      ("--weight_decay", args.weight_decay),
                      ("--label_smoothing", args.label_smoothing)]:
        if val is not None:
            extra += [flag, str(val)]
    for flag, enabled in [("--ema", args.ema), ("--mixup", args.mixup),
                           ("--cutmix", args.cutmix), ("--strong_aug", args.strong_aug)]:
        if enabled:
            extra.append(flag)

    run_kfold(args.arch, args.loss, args.preprocessing, args.n_folds,
              args.target_sensitivity, args.unfreeze_blocks,
              run_name_prefix=args.run_name_prefix, extra_train_args=extra)
