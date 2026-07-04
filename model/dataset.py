"""
model/dataset.py

Loads HAM10000 metadata, builds lesion-ID-aware stratified splits (fixing a
real bug from v1: splitting by image_id let multiple photos of the SAME
lesion land in both train and test, quietly inflating reported accuracy),
and provides a PyTorch Dataset + DataLoader builder.

Also handles:
  - bulk preprocessing (caching preprocessed images to disk so training
    doesn't redo deterministic hair-removal/CLAHE every epoch)
  - an explicit on/off preprocessing cache, so the SAME pipeline can produce
    a "no preprocessing" cache for the with/without ablation experiment
  - lesion-ID-aware K-fold splits, for the K-fold cross-validation experiment

Run directly to build a cache:
    python model/dataset.py --build_preprocessed --preprocessing on
    python model/dataset.py --build_preprocessed --preprocessing off
(run from the project root, i.e. inside melanoma-core/)
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SGKF = True
except ImportError:
    HAVE_SGKF = False

# Make `preprocessing/` importable when this file is run as a script from
# inside model/ (its own directory is auto-added to sys.path, the project
# root is not, so we add it explicitly).
sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocessing.preprocess import color_normalize, preprocess_image, remove_vignette, remove_ruler  # noqa: E402

TASK_MODE = "binary"  # "binary" or "multiclass" -- change here, or pass --task_mode
IMG_SIZE = 224

RAW_IMAGES_DIR = "data/raw/images"
METADATA_PATH = "data/raw/HAM10000_metadata.csv"
PREPROCESSED_DIR = "data/preprocessed"
RAW_CACHE_DIR = "data/no_preprocessing_cache"  # resized-only cache, for the ablation
COLOR_NORM_CACHE_DIR = "data/color_norm_cache"  # color-normalization-ONLY cache,
# isolated from hair removal/CLAHE so its effect can be measured on its own
DEVIGNETTE_CACHE_DIR      = "data/devignette_cache"       # resize + vignette removal only
DEVIGNETTE_RULER_CACHE_DIR = "data/devig_ruler_cache"     # resize + vignette + ruler masking

MULTICLASS_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

_BASE_CACHE_DIR = {
    "on":         PREPROCESSED_DIR,
    "off":        RAW_CACHE_DIR,
    "color_norm": COLOR_NORM_CACHE_DIR,
    "devig":      DEVIGNETTE_CACHE_DIR,
    "devig_ruler": DEVIGNETTE_RULER_CACHE_DIR,
}


def cache_dir_for(preprocessing, img_size=IMG_SIZE):
    """Cache directory for a (preprocessing, resolution) pair. The default 224
    keeps the original dir names; other sizes get a `_<size>` suffix so a
    higher-res cache never collides with (or gets skipped against) the 224 one."""
    base = _BASE_CACHE_DIR[preprocessing]
    return base if img_size == IMG_SIZE else f"{base}_{img_size}"


def load_metadata(metadata_path=METADATA_PATH, task_mode=TASK_MODE):
    df = pd.read_csv(metadata_path)
    if task_mode == "binary":
        df["label"] = (df["dx"] == "mel").astype(int)
    else:
        label_map = {name: i for i, name in enumerate(MULTICLASS_LABELS)}
        df["label"] = df["dx"].map(label_map)
    return df


def lesion_aware_split(df, test_size=0.15, val_size=0.15, seed=42):
    """
    Splits by lesion_id, not by image_id, so every image of a given lesion
    stays in the same split. (v1 split by image and leaked lesions across
    train/test, which quietly inflated reported test accuracy.)
    """
    lesion_labels = df.groupby("lesion_id")["label"].max()  # one row per lesion
    lesion_ids = lesion_labels.index.values
    labels = lesion_labels.values

    train_ids, temp_ids, train_y, temp_y = train_test_split(
        lesion_ids, labels, test_size=(test_size + val_size),
        stratify=labels, random_state=seed,
    )
    relative_val = val_size / (test_size + val_size)
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_y, test_size=(1 - relative_val),
        stratify=temp_y, random_state=seed,
    )

    train_df = df[df["lesion_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def lesion_aware_kfold(df, n_splits=5, seed=42):
    """
    Yields (train_df, val_df) pairs for K-fold CV, keeping all images of a
    lesion in the same fold. Uses StratifiedGroupKFold if available
    (scikit-learn >= 1.1); falls back to a manual grouped-stratified chunk
    split otherwise.
    """
    lesion_labels = df.groupby("lesion_id")["label"].max()
    lesion_ids = lesion_labels.index.values
    labels = lesion_labels.values

    if HAVE_SGKF:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in sgkf.split(lesion_ids, labels, groups=lesion_ids):
            train_lesions = set(lesion_ids[train_idx])
            val_lesions = set(lesion_ids[val_idx])
            yield (
                df[df["lesion_id"].isin(train_lesions)].reset_index(drop=True),
                df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True),
            )
    else:
        rng = np.random.RandomState(seed)
        folds = [[] for _ in range(n_splits)]
        for class_val in np.unique(labels):
            class_lesions = lesion_ids[labels == class_val].copy()
            rng.shuffle(class_lesions)
            chunks = np.array_split(class_lesions, n_splits)
            for i, chunk in enumerate(chunks):
                folds[i].extend(chunk.tolist())
        for i in range(n_splits):
            val_lesions = set(folds[i])
            train_lesions = set(lesion_ids) - val_lesions
            yield (
                df[df["lesion_id"].isin(train_lesions)].reset_index(drop=True),
                df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True),
            )


def get_class_weights(df, task_mode=TASK_MODE):
    classes = np.unique(df["label"])
    weights = compute_class_weight("balanced", classes=classes, y=df["label"])
    return torch.tensor(weights, dtype=torch.float32)


def build_preprocessed_cache(df, raw_dir=RAW_IMAGES_DIR, out_dir=PREPROCESSED_DIR,
                              mode="full", img_size=IMG_SIZE):
    """
    Runs the requested preprocessing variant once per image and caches the
    PNG result, so training doesn't redo it every epoch. Safe to re-run --
    already-cached images are skipped.

    mode:
      "full"       -> hair removal + CLAHE (preprocess_image, no denoise)
      "raw"        -> just resize, no processing at all
      "color_norm" -> ONLY Shades-of-Gray color normalization, isolated from
                      hair removal/CLAHE so its effect can be measured on its
                      own rather than mixed into a combo already shown to
                      underperform.
    """
    os.makedirs(out_dir, exist_ok=True)
    skipped, done = 0, 0
    for image_id in df["image_id"]:
        out_path = os.path.join(out_dir, f"{image_id}.png")
        if os.path.exists(out_path):
            skipped += 1
            continue
        src_path = os.path.join(raw_dir, f"{image_id}.jpg")
        img = cv2.imread(src_path)
        if img is None:
            print(f"  WARNING: could not read {src_path}, skipping")
            continue
        if mode == "full":
            img = preprocess_image(img)
        elif mode == "color_norm":
            img = color_normalize(img)
        elif mode == "devignette":
            img = remove_vignette(img)
        elif mode == "devignette_ruler":
            img = remove_ruler(remove_vignette(img))
        # mode == "raw": no processing
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(out_path, img)
        done += 1
        if done % 500 == 0:
            print(f"  ...{done} images processed")
    print(f"Done. {done} newly processed, {skipped} already cached.")


class HAM10000Dataset(Dataset):
    """
    Reads pre-cached PNGs (already hair-removed/denoised/CLAHE'd, or just
    resized -- whichever cache dir is passed in) and applies LIVE augmentation
    only when augment=True (i.e. only on the train split). Deterministic
    preprocessing is never repeated here -- it already happened once in
    build_preprocessed_cache().

    strong_aug=True adds RandAugment (before ToTensor) + RandomErasing (after
    Normalize) on top of the baseline geometric/colour augs -- extra
    regularisation to close the val->test gap. Off by default so existing runs
    reproduce exactly.

    Note on img_size: the on-disk cache is written at the cache's own resolution
    (224 by default). Asking for a larger img_size here just upsamples those
    cached PNGs -- to truly train at higher resolution, rebuild the cache at that
    size first (see build_preprocessed_cache / the --img_size CLI flag).
    """
    NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __init__(self, df, image_dir, img_size=IMG_SIZE, augment=False, strong_aug=False,
                 spatial_aug=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size

        if augment:
            # spatial_aug: wider random-resized-crop + more rotation to decorrelate lesion
            # position from label (breaks the positional/compositional bias). RRC crops WITHIN
            # the image and refills the frame -> no black padding. ColorJitter is held constant
            # so this is a spatial-only ablation vs the baseline geometric block.
            # else-branch values reproduce the baseline byte-for-byte (ratio default = (3/4,4/3)).
            crop_scale = (0.55, 1.0) if spatial_aug else (0.85, 1.0)
            crop_ratio = (0.8, 1.25) if spatial_aug else (3 / 4, 4 / 3)
            rot_deg    = 30 if spatial_aug else 20
            geometric = [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(rot_deg),
                T.RandomResizedCrop(img_size, scale=crop_scale, ratio=crop_ratio),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            ]
            if strong_aug:
                geometric.append(T.RandAugment(num_ops=2, magnitude=7))
            tail = [T.ToTensor(), self.NORM]
            if strong_aug:
                tail.append(T.RandomErasing(p=0.25, scale=(0.02, 0.15)))
            self.transform = T.Compose(geometric + tail)
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                self.NORM,
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['image_id']}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = int(row["label"])
        return img, label


def make_dataloaders(train_df, val_df, test_df, image_dir, batch_size=32,
                     img_size=IMG_SIZE, strong_aug=False, spatial_aug=False):
    train_ds = HAM10000Dataset(train_df, image_dir, img_size, augment=True,
                               strong_aug=strong_aug, spatial_aug=spatial_aug)
    val_ds = HAM10000Dataset(val_df, image_dir, img_size, augment=False)
    test_ds = HAM10000Dataset(test_df, image_dir, img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_preprocessed", action="store_true")
    parser.add_argument("--preprocessing", choices=["on", "off", "color_norm", "devig", "devig_ruler"], default="on")
    parser.add_argument("--task_mode", choices=["binary", "multiclass"], default=TASK_MODE)
    parser.add_argument("--metadata", default=METADATA_PATH)
    parser.add_argument("--raw_dir", default=RAW_IMAGES_DIR)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE,
                        help="Cache resolution. Non-224 sizes build a separate `_<size>` cache "
                             "for true higher-res training (see cache_dir_for).")
    args = parser.parse_args()

    df = load_metadata(args.metadata, args.task_mode)

    if args.build_preprocessed:
        mode_map = {"on": "full", "off": "raw", "color_norm": "color_norm", "devig": "devignette", "devig_ruler": "devignette_ruler"}
        mode = mode_map[args.preprocessing]
        out_dir = cache_dir_for(args.preprocessing, args.img_size)
        print(f"Building cache (mode={mode}, img_size={args.img_size}) -> {out_dir}")
        build_preprocessed_cache(df, args.raw_dir, out_dir, mode=mode, img_size=args.img_size)
