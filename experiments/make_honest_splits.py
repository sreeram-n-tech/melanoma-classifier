"""
experiments/make_honest_splits.py

Run ONCE to generate the 5 lesion-aware fold split definitions shared by
BOTH ablation arms (baseline and treatment).  Each fold JSON contains three
disjoint, lesion-ID-clean partitions:

    inner_train  ~64% — training data
    inner_val    ~16% — calibration: temperature + threshold fitting
    held_out     ~20% — evaluation only (never seen during training/calibration)

Outer folds produced by lesion_aware_kfold (StratifiedGroupKFold, n=5, seed=42).
Inner-val carved from inner with train_test_split(test_size=0.20, stratified,
seed=42, lesion-level).

Writes:
    experiments/honest_splits/fold{0..4}.json

Run from project root:
    python experiments/make_honest_splits.py
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, ROOT)

from dataset import lesion_aware_kfold, load_metadata  # noqa: E402

OUT_DIR        = os.path.join(ROOT, "experiments", "honest_splits")
INNER_VAL_FRAC = 0.20
SEED           = 42
N_FOLDS        = 5


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df    = load_metadata()
    folds = list(lesion_aware_kfold(df, n_splits=N_FOLDS, seed=SEED))

    for fold_idx, (inner_df, held_out_df) in enumerate(folds):
        lesion_labels = inner_df.groupby("lesion_id")["label"].max()
        lesion_ids    = lesion_labels.index.values
        labels        = lesion_labels.values

        inner_train_lesions, inner_val_lesions = train_test_split(
            lesion_ids, test_size=INNER_VAL_FRAC,
            stratify=labels, random_state=SEED,
        )
        inner_train_df = inner_df[inner_df["lesion_id"].isin(inner_train_lesions)]
        inner_val_df   = inner_df[inner_df["lesion_id"].isin(inner_val_lesions)]

        split_data = {
            "fold":        fold_idx,
            "inner_train": inner_train_df["image_id"].tolist(),
            "inner_val":   inner_val_df["image_id"].tolist(),
            "held_out":    held_out_df["image_id"].tolist(),
        }

        # Sanity: all three partitions are disjoint
        it = set(split_data["inner_train"])
        iv = set(split_data["inner_val"])
        ho = set(split_data["held_out"])
        assert len(it & iv) == 0, f"fold{fold_idx}: inner_train / inner_val overlap"
        assert len(it & ho) == 0, f"fold{fold_idx}: inner_train / held_out overlap"
        assert len(iv & ho) == 0, f"fold{fold_idx}: inner_val / held_out overlap"
        assert len(it) + len(iv) + len(ho) == len(df), (
            f"fold{fold_idx}: total {len(it)+len(iv)+len(ho)} != dataset {len(df)}"
        )

        # Mel prevalence check (sanity)
        it_mel = inner_train_df["label"].mean()
        iv_mel = inner_val_df["label"].mean()
        ho_mel = held_out_df["label"].mean()

        out_path = os.path.join(OUT_DIR, f"fold{fold_idx}.json")
        with open(out_path, "w") as f:
            json.dump(split_data, f, separators=(",", ":"))

        print(f"fold{fold_idx}: inner_train={len(it)} (mel={it_mel:.1%})  "
              f"inner_val={len(iv)} (mel={iv_mel:.1%})  "
              f"held_out={len(ho)} (mel={ho_mel:.1%})")

    print(f"\nDone -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
