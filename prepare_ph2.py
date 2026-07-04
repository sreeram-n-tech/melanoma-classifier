"""
prepare_ph2.py

Prepares the PH2 dataset for external_eval.py:
  1. Flattens the nested PH2 image structure into data/external/ph2/images/
  2. Writes data/external/ph2/labels.csv with columns: image_id, diagnosis

Usage (from project root, after extracting PH2Dataset/ somewhere):
    python prepare_ph2.py --ph2_root path/to/PH2Dataset

PH2 xlsx structure:
  - Rows 1-12: legend tables (skipped)
  - Row 13: header
  - Row 14+: data; image IDs in column "Image Name"
  - Diagnosis one-hot across 3 columns ("Common Nevus", "Atypical Nevus", "Melanoma")
    with "X" marking the positive class for each row
  - Melanoma X -> "melanoma"; anything else -> "nevus"
"""
import argparse
import os
import shutil

import pandas as pd


def find_annotation_file(ph2_root):
    xlsx, txt = None, None
    for dirpath, _, files in os.walk(ph2_root):
        for f in files:
            path = os.path.join(dirpath, f)
            if f.endswith(".xlsx") or f.endswith(".xls"):
                xlsx = path
            elif f == "PH2_dataset.txt":
                txt = path
    return xlsx or txt  # prefer Excel; fall back to txt


def parse_annotations(ann_path):
    """Returns a dict of {image_id: label_str} where label_str is 'melanoma' or 'nevus'."""
    ext = os.path.splitext(ann_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        # Header at row 13 (0-indexed skiprows=12 skips rows 1-12, row 13 becomes header).
        df = pd.read_excel(ann_path, header=0, skiprows=12)
        df.columns = [str(c).strip() for c in df.columns]

        # Find image ID column (column A, "Image Name").
        id_col = None
        for c in df.columns:
            if "image" in c.lower() or "name" in c.lower():
                id_col = c
                break
        if id_col is None:
            print(f"Columns found: {list(df.columns)}")
            raise ValueError("Could not find image ID column. Check xlsx structure.")

        # Find the three one-hot diagnosis columns by keyword.
        mel_col, nevus_cols = None, []
        for c in df.columns:
            cl = c.lower()
            if "melanoma" in cl:
                mel_col = c
            elif "nevus" in cl or "common" in cl or "atypical" in cl:
                nevus_cols.append(c)

        if mel_col is None:
            print(f"Columns found: {list(df.columns)}")
            raise ValueError("Could not find Melanoma column. Check xlsx structure.")

        print(f"  Image ID column : '{id_col}'")
        print(f"  Melanoma column : '{mel_col}'")
        print(f"  Nevus columns   : {nevus_cols}")

        result = {}
        for _, row in df.iterrows():
            img_id = str(row[id_col]).strip()
            if img_id in ("nan", "", "NaN"):
                continue
            is_melanoma = str(row[mel_col]).strip().upper() == "X"
            result[img_id] = "melanoma" if is_melanoma else "nevus"
        return result
    elif ext == ".txt":
        # Some distributions include a text annotation file
        result = {}
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_id = parts[0]
                    diag = parts[-1]
                    label = "melanoma" if diag in ("2", "Melanoma", "melanoma") else "nevus"
                    result[img_id] = label
        return result
    else:
        raise ValueError(f"Unknown annotation format: {ann_path}")


def main(args):
    ph2_root = args.ph2_root
    out_images = os.path.join("data", "external", "ph2", "images")
    out_csv = os.path.join("data", "external", "ph2", "labels.csv")
    os.makedirs(out_images, exist_ok=True)

    # Find annotation file
    ann_path = args.annotation or find_annotation_file(ph2_root)
    if ann_path is None:
        print("WARNING: No annotation file found. Will label all images as 'unknown'.")
        annotations = {}
    else:
        print(f"Found annotation file: {ann_path}")
        annotations = parse_annotations(ann_path)
        print(f"Parsed {len(annotations)} annotations. "
              f"Melanoma: {sum(1 for v in annotations.values() if v == 'melanoma')}, "
              f"Nevus: {sum(1 for v in annotations.values() if v == 'nevus')}")

    # Walk PH2 image tree and flatten
    rows = []
    for dirpath, _, files in os.walk(ph2_root):
        for fname in files:
            if not fname.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
                continue
            # PH2 image IDs look like IMD001, IMD002, ...
            stem = os.path.splitext(fname)[0]
            # Skip masks or other derived files (e.g. IMD001_lesion)
            if "_" in stem and not stem.startswith("IMD"):
                continue
            img_id = stem.split("_")[0] if "_" in stem else stem
            if not img_id.startswith("IMD"):
                continue
            src = os.path.join(dirpath, fname)
            dst = os.path.join(out_images, f"{img_id}.bmp")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            label = annotations.get(img_id, annotations.get(stem, "unknown"))
            rows.append({"image_id": img_id, "diagnosis": label})

    # Deduplicate (nested structure may find same image twice)
    df = pd.DataFrame(rows).drop_duplicates("image_id").sort_values("image_id")
    df.to_csv(out_csv, index=False)

    n_mel = (df["diagnosis"] == "melanoma").sum()
    n_nev = (df["diagnosis"] == "nevus").sum()
    print(f"\nFlattened {len(df)} images to {out_images}/")
    print(f"  Melanoma: {n_mel}, Nevus/other: {n_nev}")
    print(f"  Written: {out_csv}")
    print(f"\nNext step:")
    print(f'  python external_eval.py \\')
    print(f'    --checkpoint model_output/effb0_unfreeze19_main/best_model.pt \\')
    print(f'    --images_dir {out_images} \\')
    print(f'    --labels_csv {out_csv} \\')
    print(f'    --image_col image_id --label_col diagnosis \\')
    print(f'    --positive_values melanoma --tta')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ph2_root", required=True,
                        help="Path to the extracted PH2Dataset/ directory")
    parser.add_argument("--annotation", default=None,
                        help="Explicit path to annotation file (xlsx or txt). "
                             "Auto-detected if omitted; xlsx is preferred over txt.")
    args = parser.parse_args()
    main(args)
