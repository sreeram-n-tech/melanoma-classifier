"""
experiments/run_architecture_comparison.py

Trains MobileNetV2 and EfficientNetB0 under the same loss/preprocessing
config and reports which wins -- strengthens the report with a real
comparison instead of just picking one architecture by assumption.

Usage (from the project root):
    python experiments/run_architecture_comparison.py
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(arch, loss, preprocessing):
    tag = f"{arch}_{loss}_{preprocessing}_main"
    print(f"\n=== Training: arch={arch} ===")
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
    parser.add_argument("--loss", default="weighted_ce")
    parser.add_argument("--preprocessing", default="on")
    args = parser.parse_args()

    archs = ["mobilenet_v2", "efficientnet_b0"]
    results = {a: run(a, args.loss, args.preprocessing) for a in archs}

    print("\n=== Architecture Comparison: Summary ===")
    print(f"{'metric':<15}{'mobilenet_v2':<18}{'efficientnet_b0':<18}")
    for metric in ["auc", "sensitivity", "specificity"]:
        print(f"{metric:<15}{results['mobilenet_v2'][metric]:<18.4f}{results['efficientnet_b0'][metric]:<18.4f}")

    out_path = os.path.join(ROOT, "model_output", "architecture_comparison_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {out_path}")
