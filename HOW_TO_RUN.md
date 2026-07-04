# Melanoma Detection — PyTorch Core (v2, verified)

This replaces any earlier training code (TensorFlow version, or whatever
ChatGPT/Antigravity generated). Switched to PyTorch because recent TensorFlow
versions dropped native GPU support on Windows — PyTorch works with your
RTX 3050 directly, no WSL2 needed.

## What this version fixes / adds, and why it matters

- **Lesion-ID-aware splitting** (train/val/test AND K-fold) — every image of
  the same lesion stays in one split. Splitting by image_id instead of
  lesion_id (a real bug we caught earlier) lets the same lesion appear in
  both train and test, which quietly inflates reported accuracy/AUC. This is
  the main reason to redo training rather than trust unverified numbers.
- **ROC-based threshold tuning** (default: lowest threshold that still hits
  95% sensitivity) reported *alongside* the default 0.5 cutoff — not instead
  of it, so you can see the real tradeoff rather than one cherry-picked number.
- **Brightness/contrast/saturation augmentation**, on top of the existing
  flip/rotate/zoom.
- **Grad-CAM** included.
- **Checkpoint resume** — safe to Ctrl+C and pick back up mid-training,
  without restarting Stage 1 from scratch.
- **Four ready-to-run comparison experiments**: preprocessing on/off,
  MobileNetV2 vs EfficientNetB0, focal vs weighted cross-entropy loss,
  5-fold cross-validation.

All of this was tested end-to-end (split → cache → train → resume →
evaluate → Grad-CAM) on a synthetic dataset before being handed to you —
the logic is verified, even though the real ~10k-image run will be your
first time executing it against HAM10000.

## 0. Reuse what you already have — don't re-download anything

Keep your existing:
- `data/raw/HAM10000_metadata.csv`
- `data/raw/images/` (merged folder of all ~10,015 .jpg files)

Just place/point this project at the same paths. The Kaggle download and
folder merge you already did are not wasted.

## 1. Environment

```powershell
cd path\to\melanoma-core
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `True NVIDIA GeForce RTX 3050 ...`. If it prints `False`, stop
and tell me the exact error before training anything — that means PyTorch
isn't seeing your GPU and training would silently run on CPU (very slow).

## 2. Validate preprocessing on a few images first

```powershell
python preprocessing\preprocess.py --test_dir data\raw\images --out_dir preprocessing_check --n_images 10
```
Open `preprocessing_check/` and actually look — confirm lesions aren't
distorted and edges are still visible before processing all ~10k images.

## 3. Build the preprocessing caches (build BOTH — you need both for the ablation experiment)

```powershell
python model\dataset.py --build_preprocessed --preprocessing on
python model\dataset.py --build_preprocessed --preprocessing off
```
First run takes a while over ~10k images. Safe to re-run — already-cached
images are skipped, so an interrupted run just continues where it left off.

## 4. Train the main model

```powershell
python model\train.py --arch mobilenet_v2 --loss weighted_ce --preprocessing on
```
Checkpoints/logs land in `model_output/mobilenet_v2_weighted_ce_on_main/`.
If you need to stop partway (Ctrl+C), resume exactly where you left off:
```powershell
python model\train.py --arch mobilenet_v2 --loss weighted_ce --preprocessing on --resume
```

## 5. Evaluate

```powershell
python model\evaluate.py --checkpoint model_output\mobilenet_v2_weighted_ce_on_main\best_model.pt
```
Prints metrics at BOTH the default 0.5 threshold and a sensitivity-tuned
threshold, and saves `evaluation_report.png` (ROC curve + confusion matrix).

## 6. Grad-CAM

```powershell
python model\gradcam.py --checkpoint model_output\mobilenet_v2_weighted_ce_on_main\best_model.pt --image_dir data\preprocessed --n_samples 8
```

## 7. The four comparison experiments (this is your "week 2")

Each one trains + evaluates every config in the comparison and writes a
summary JSON to `model_output/` automatically — no manual bookkeeping:

```powershell
python experiments\run_preprocessing_ablation.py
python experiments\run_architecture_comparison.py
python experiments\run_loss_comparison.py
python experiments\run_kfold.py --n_folds 5
```
Heads up: `run_kfold.py` trains 5 full models (one per fold), so budget
roughly 5x the time of one main run for it.

## 8. Handoff to the website

Whichever config wins (`model_output/<tag>/best_model.pt`) is the only file
the website needs. Back to Antigravity for the Flask backend + frontend —
the backend just loads this checkpoint and calls it for predictions.
