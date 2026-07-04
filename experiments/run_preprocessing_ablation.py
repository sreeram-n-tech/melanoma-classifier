"""
experiments/run_preprocessing_ablation.py

Trains the SAME config (architecture, loss) twice -- once on the
preprocessed cache, once on the raw-resized-only cache -- and reports a
side-by-side comparison. Turns "preprocessing helps" from an assumption
into a measured result.

Usage (from the project root):
    python experiments/run_preprocessing_ablation.py
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(arch, loss, preprocessing):
    tag = f"{arch}_{loss}_{preprocessing}_main"
    print(f"\n=== Training: preprocessing={preprocessing} ===")
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
    parser.add_argument("--loss", default="weighted_ce")
    args = parser.parse_args()

    results = {"on": run(args.arch, args.loss, "on"), "off": run(args.arch, args.loss, "off")}

    print("\n=== Preprocessing Ablation: Summary ===")
    print(f"{'metric':<15}{'with preprocessing':<22}{'without preprocessing':<22}")
    for metric in ["auc", "sensitivity", "specificity"]:
        print(f"{metric:<15}{results['on'][metric]:<22.4f}{results['off'][metric]:<22.4f}")

    out_path = os.path.join(ROOT, "model_output", "preprocessing_ablation_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {out_path}")
