"""
webapp/parity_check.py

Proves the web app's inference path reproduces the numbers frozen in each
registered model's own `webapp/models/<key>/deploy.json`.

For every model in the registry it runs the web app's OWN inference functions
(`inference.TRANSFORM`, `inference._mean_logits`, `inference._prob_pos`, that
model's own temperature/threshold/TTA read from ITS deploy.json) over the SAME
held-out test split that `finalize.py` used (`lesion_aware_split(df, seed=42)`),
reading the SAME `off` cache (`data/no_preprocessing_cache/`). It then compares
AUC / sensitivity / specificity against that model's deploy.json.

If the webapp path is faithful, the three numbers match deploy.json to floating
point per model. AUC is threshold-independent; sensitivity/specificity depend on
each model's own frozen temperature and threshold, both taken straight from its
deploy.json — never shared or borrowed across models.

Read-only: it loads the copied checkpoints and reads the research metadata + image
cache. It does not modify any research file.

Run from the repo root (G:\\srip):
    python webapp/parity_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))          # webapp/inference.py
sys.path.insert(0, str(ROOT / "model"))  # research dataset.py (read-only)

import inference  # noqa: E402  (webapp inference — the code under test)
from dataset import RAW_CACHE_DIR, lesion_aware_split, load_metadata  # noqa: E402

OFF_CACHE = ROOT / RAW_CACHE_DIR  # data/no_preprocessing_cache
METADATA = ROOT / "data" / "raw" / "HAM10000_metadata.csv"

TOL = 1e-3


def check_model(key: str, test_df) -> bool:
    entry = inference.get_model(key)
    deploy = entry.deploy_raw

    probs, labels = [], []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = Image.open(OFF_CACHE / f"{row['image_id']}.png")
            x = inference.TRANSFORM(img.convert("RGB")).unsqueeze(0)
            logits = inference._mean_logits(entry.net, x, entry.tta)
            probs.append(inference._prob_pos(logits, entry.temperature))
            labels.append(int(row["label"]))

    probs = np.asarray(probs)
    labels = np.asarray(labels)

    auc = float(roc_auc_score(labels, probs))
    preds = (probs >= entry.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    rows = [
        ("AUC", auc, deploy["test_auc"]),
        ("Sensitivity", sens, deploy["test_sensitivity"]),
        ("Specificity", spec, deploy["test_specificity"]),
    ]

    print(f"\n=== {entry.display_name} ({key}) ===")
    print(f"Frozen operating point from deploy.json: T={entry.temperature:.4f}, "
          f"threshold={entry.threshold:.4f}, tta={entry.tta}\n")
    print(f"{'metric':<12}{'webapp':>10}{'deploy.json':>14}{'diff':>12}")
    ok = True
    for name, got, ref in rows:
        d = got - ref
        ok = ok and abs(d) < TOL
        print(f"{name:<12}{got:>10.4f}{ref:>14.4f}{d:>+12.5f}")
    return ok


def main() -> int:
    df = load_metadata(metadata_path=str(METADATA), task_mode="binary")
    _, _, test_df = lesion_aware_split(df, seed=42)
    print(f"Test images scored: {len(test_df)}  (held-out split, seed=42, 'off' cache)")

    results = {}
    for m in inference.available_models():
        results[m["key"]] = check_model(m["key"], test_df)

    print()
    ok = all(results.values())
    for key, passed in results.items():
        print(f"{key:<14} {'PASS' if passed else 'MISMATCH'}")
    print("\nPARITY:", "PASS - all registered models reproduce their deploy.json within 1e-3"
          if ok else "MISMATCH - investigate the model(s) marked above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
