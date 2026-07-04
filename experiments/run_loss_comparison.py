"""
experiments/run_loss_comparison.py

Trains the same architecture with class-weighted cross-entropy vs. focal
loss, and reports which handles HAM10000's class imbalance (melanoma is a
small minority class) better.

Usage (from the project root):
    python experiments/run_loss_comparison.py
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(arch, loss, preprocessing):
    tag = f"{arch}_{loss}_{preprocessing}_main"
    print(f"\n=== Training: loss={loss} ===")
    subprocess.run([sys.executable, os.path.join(ROOT, "model", "train.py"),
                     "--arch", arch, "--loss", loss, "--preprocessing", preprocessing],
                    cwd=ROOT, check=True)
    ckpt = os.path.join(ROOT, "model_output", tag, "best_model.pt")
    subprocess.run([sys.executable, os.path.join(ROOT, "model", "evaluate.py"),
                     "--checkpoint", ckpt], cwd=ROOT, check=True)
    with open(os.path.join(ROOT, "model_output", tag, "eval_results.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="mobilenet_v2")
    parser.add_argument("--preprocessing", default="on")
    args = parser.parse_args()

    losses = ["weighted_ce", "focal"]
    results = {l: run(args.arch, l, args.preprocessing) for l in losses}

    print("\n=== Loss Function Comparison: Summary ===")
    print(f"{'metric':<15}{'weighted_ce':<18}{'focal':<18}")
    for metric in ["auc", "sensitivity", "specificity"]:
        print(f"{metric:<15}{results['weighted_ce'][metric]:<18.4f}{results['focal'][metric]:<18.4f}")

    out_path = os.path.join(ROOT, "model_output", "loss_comparison_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {out_path}")
