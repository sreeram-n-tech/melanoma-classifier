"""
experiments/run_augment.py -- Spatial-augmentation ablation runner.

Trains ONE new arm ("augment" = the exact baseline recipe + --spatial_aug) on the
byte-identical honest_splits, then pairs it against the EXISTING baseline summary. No
baseline retrain. Standalone / detached / idempotent / resumable. Reuses the honest
scoring protocol and the sens-floor / NaN-loss hard-abort guards from run_all.py; adds
one new guard -- the lesion-in-frame preflight gate.

Usage (from project root G:\\srip):
    python experiments/run_augment.py --preflight-only   # gate + QC grid, then STOP
    python experiments/run_augment.py                    # full run (skips preflight if PASS.txt)

Idempotent skip guards:
  - preflight:  skipped if augment_qc/preflight_PASS.txt exists
  - each fold:  skipped if best_model.pt + >=15 CSV rows; partial -> delete best, --resume
  - arm:        skipped if model_output/kfold_honest_augment_summary.json exists
  - compare/gradcam/report: always recomputed (cheap)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "preprocessing"))
sys.path.insert(0, ROOT)  # so `import run_all` resolves

import run_all  # noqa: E402  (honest scoring protocol + shared helpers, single source of truth)
from run_all import (  # noqa: E402
    evaluate_fold, is_training_complete, _csv_epoch_count, _nan_in_log,
    load_cached_bgr, MODEL_OUTPUT, SPLIT_DIR, N_FOLDS, TOTAL_EPOCHS, TAU_SENS, FN_IDS,
)

ARM              = "augment"
LOG_FILE         = os.path.join(ROOT, "run_augment_log.txt")
QC_DIR           = os.path.join(ROOT, "augment_qc")
PASS_MARKER      = os.path.join(QC_DIR, "preflight_PASS.txt")
BASELINE_SUMMARY = os.path.join(MODEL_OUTPUT, "kfold_honest_baseline_summary.json")
CONTROL_TPS      = ["ISIC_0031784", "ISIC_0032699", "ISIC_0027261"]

# --- treatment spatial aug (must match model/dataset.py spatial_aug block) ---
IMG_SIZE        = 224
CROP_SCALE      = (0.55, 1.0)
CROP_RATIO      = (0.8, 1.25)
ROT_DEG         = 30
RETAIN_MIN_FRAC = 0.70   # a crop "retains" the lesion if >=70% of the central disk is inside it
RETAIN_GATE     = 0.95   # >=95% of sampled crops must retain, else hard-abort


# ── logging / abort ───────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def abort(reason):
    log(f"[HARD ABORT] {reason}")
    with open(os.path.join(ROOT, "STOP_REPORT.txt"), "w") as f:
        f.write(f"HARD ABORT (run_augment): {reason}\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.exit(1)


# ── preflight lesion-in-frame gate ────────────────────────────────────────────

def _pick_qc_ids(df):
    ids = [i for i in FN_IDS]
    rng = np.random.default_rng(42)
    for dx in ("mel", "nv"):
        pool = df[df["dx"] == dx]["image_id"].tolist()
        if pool:
            ids += list(rng.choice(pool, size=min(3, len(pool)), replace=False))
    return ids


def preflight():
    """Analytic central-disk retention (unattended abort) + visual QC grid (eyeball).
    Runs BEFORE any training. Writes PASS marker on success so resume skips it."""
    import torch
    from torchvision import transforms as T
    from PIL import Image
    from dataset import _BASE_CACHE_DIR, load_metadata  # noqa: E402

    os.makedirs(QC_DIR, exist_ok=True)
    if os.path.exists(PASS_MARKER):
        log("Preflight: PASS marker present. Skipping gate.")
        return

    log("=== Preflight lesion-in-frame gate ===")

    # (1) Analytic proxy: Monte-Carlo torchvision's OWN crop sampler; check that the
    # central disk (the region a curated dermoscopy lesion sits in) stays in the crop.
    side = IMG_SIZE
    yy, xx = np.ogrid[:side, :side]
    disk = ((yy - side / 2) ** 2 + (xx - side / 2) ** 2) <= (0.25 * side) ** 2
    disk_area = float(disk.sum())
    dummy = torch.zeros(3, side, side)
    n, retained = 2000, 0
    for _ in range(n):
        i, j, h, w = T.RandomResizedCrop.get_params(dummy, CROP_SCALE, CROP_RATIO)
        if disk[i:i + h, j:j + w].sum() / disk_area >= RETAIN_MIN_FRAC:
            retained += 1
    frac = retained / n
    log(f"  central-disk retention: {frac:.3f} of {n} crops keep >={RETAIN_MIN_FRAC:.0%} "
        f"of the disk  (gate >= {RETAIN_GATE:.0%})")

    # (2) Visual grid: spatial ops only (ColorJitter/normalize omitted -- geometry is
    # what decides lesion-in-frame). Original + 6 augmented samples per row.
    geo = T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(ROT_DEG),
        T.RandomResizedCrop(side, scale=CROP_SCALE, ratio=CROP_RATIO),
    ])
    image_dir = os.path.join(ROOT, _BASE_CACHE_DIR["off"])
    df = load_metadata()
    rows_img = []
    for iid in _pick_qc_ids(df):
        bgr = load_cached_bgr(iid, image_dir)
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (side, side))
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cells = [bgr]
        for _ in range(6):
            cells.append(cv2.cvtColor(np.array(geo(pil)), cv2.COLOR_RGB2BGR))
        rows_img.append(np.concatenate(cells, axis=1))
    grid_path = os.path.join(QC_DIR, "preflight_grid.jpg")
    if rows_img:
        cv2.imwrite(grid_path, np.concatenate(rows_img, axis=0))
        log(f"  QC grid -> {grid_path}  ({len(rows_img)} images x 6 augs)")
    else:
        log("  WARNING: no QC images found in the off cache; grid not written.")

    if frac < RETAIN_GATE:
        abort(f"preflight retention {frac:.3f} < {RETAIN_GATE:.2f} -- crops drop the lesion "
              f"too often; loosen CROP_SCALE lower bound.")
    with open(PASS_MARKER, "w") as f:
        f.write(f"retention={frac:.4f}\nmin_frac={RETAIN_MIN_FRAC}\ngate={RETAIN_GATE}\n"
                f"grid=augment_qc/preflight_grid.jpg\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log(f"  Preflight PASSED (retention {frac:.3f}).")


# ── pairing integrity ─────────────────────────────────────────────────────────

def assert_paired_integrity(baseline_summary):
    """Prove the existing baseline summary was scored on the CURRENT honest_splits:
    per-fold inner_val/held_out counts must match. Guards against pairing the augment
    arm against a baseline computed on different splits/seed."""
    b = {r["fold"]: r for r in baseline_summary["folds"]}
    for fi in range(N_FOLDS):
        with open(os.path.join(SPLIT_DIR, f"fold{fi}.json")) as f:
            s = json.load(f)
        n_iv, n_ho = len(s["inner_val"]), len(s["held_out"])
        r = b.get(fi)
        if r is None:
            abort(f"baseline summary missing fold{fi}")
        if r.get("n_inner_val") != n_iv or r.get("n_held_out") != n_ho:
            abort(f"fold{fi}: baseline split sizes (iv={r.get('n_inner_val')}, "
                  f"ho={r.get('n_held_out')}) != current splits (iv={n_iv}, ho={n_ho}). "
                  f"Baseline was scored on DIFFERENT splits -- refusing to pair.")
    log("  Pairing integrity OK: baseline was scored on the current honest_splits.")


# ── training ──────────────────────────────────────────────────────────────────

def train_augment_fold(fold_idx):
    run_tag = f"honest_{ARM}_fold{fold_idx}"
    out_dir = os.path.join(MODEL_OUTPUT, run_tag)
    best_pt = os.path.join(out_dir, "best_model.pt")
    last_pt = os.path.join(out_dir, "last_checkpoint.pt")
    split_json = os.path.join(SPLIT_DIR, f"fold{fold_idx}.json")
    log_path = os.path.join(out_dir, "train_stdout.log")

    if is_training_complete(out_dir):
        log(f"  {run_tag}: already complete. Skipping.")
        return best_pt
    n_done = _csv_epoch_count(out_dir)
    if os.path.exists(best_pt) and n_done < TOTAL_EPOCHS:
        log(f"  {run_tag}: partial ({n_done}/{TOTAL_EPOCHS}). Deleting best_model.pt; resuming.")
        os.remove(best_pt)
    os.makedirs(out_dir, exist_ok=True)
    log(f"  {run_tag}: training (resume={os.path.exists(last_pt)}, done={n_done}) ...")

    cmd = [
        sys.executable, os.path.join(ROOT, "model", "train.py"),
        "--arch",            "efficientnet_b0",
        "--preprocessing",   "off",
        "--epochs_stage1",   "5",
        "--epochs_stage2",   "10",
        "--lr_stage2",       "1e-4",
        "--unfreeze_blocks", "19",
        "--seed",            "42",
        "--split_json",      split_json,
        "--run_name",        run_tag,
        "--spatial_aug",     # the ONLY difference vs the baseline arm
        "--resume",
    ]
    t0 = time.time()
    with open(log_path, "a") as lf:
        r = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        abort(f"Training failed for {run_tag} (exit {r.returncode}). See {log_path}")
    if _nan_in_log(log_path):
        abort(f"NaN detected in training output for {run_tag}. See {log_path}")
    if not os.path.exists(best_pt):
        abort(f"Training finished but best_model.pt missing for {run_tag}")
    log(f"  {run_tag}: done in {(time.time() - t0) / 60:.1f} min.")
    return best_pt


def run_augment_arm():
    summary_path = os.path.join(MODEL_OUTPUT, f"kfold_honest_{ARM}_summary.json")
    if os.path.exists(summary_path):
        log(f"Arm '{ARM}': summary already exists. Skipping arm.")
        with open(summary_path) as f:
            return json.load(f)

    log(f"=== ARM '{ARM}' (baseline recipe + --spatial_aug, preprocessing=off) ===")
    fold_results = []
    for fi in range(N_FOLDS):
        log(f"--- fold{fi} ---")
        ckpt = train_augment_fold(fi)
        log(f"  evaluating fold{fi} ...")
        res = evaluate_fold(ckpt, fi, "off")  # off cache; honest protocol; sens<0.90 hard-aborts
        fold_results.append(res)
        log(f"  fold{fi}: T={res['T']:.3f} thr={res['frozen_threshold']:.3f} "
            f"sens={res['achieved_sensitivity']:.3f} spec={res['achieved_specificity']:.3f} "
            f"auc={res['auc']:.4f}")

    specs = [r["achieved_specificity"] for r in fold_results]
    senss = [r["achieved_sensitivity"] for r in fold_results]
    aucs  = [r["auc"] for r in fold_results]
    summary = {
        "arm": ARM, "preprocessing": "off", "folds": fold_results,
        "mean_sensitivity": float(np.mean(senss)), "std_sensitivity": float(np.std(senss)),
        "mean_specificity": float(np.mean(specs)), "std_specificity": float(np.std(specs)),
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"=== ARM '{ARM}' DONE  AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  "
        f"spec@95={np.mean(specs):.4f}  -> {summary_path} ===")
    return summary


# ── paired comparison (ΔAUC primary) ──────────────────────────────────────────

def compare(baseline_summary, augment_summary):
    log("=== Paired comparison (augment - baseline), AUC primary ===")
    b = {r["fold"]: r for r in baseline_summary["folds"]}
    t = {r["fold"]: r for r in augment_summary["folds"]}
    rows = []
    for fi in sorted(b):
        d_auc  = t[fi]["auc"] - b[fi]["auc"]
        d_spec = t[fi]["achieved_specificity"] - b[fi]["achieved_specificity"]
        d_sens = t[fi]["achieved_sensitivity"] - b[fi]["achieved_sensitivity"]
        drifted = abs(d_sens) > TAU_SENS
        rows.append({"fold": fi, "d_auc": d_auc, "d_spec": d_spec,
                     "d_sens": d_sens, "drifted": drifted})
        log(f"  fold{fi}: dAUC={d_auc:+.4f}  dspec={d_spec:+.4f}  dsens={d_sens:+.4f}"
            f"{'  [DRIFTED]' if drifted else ''}")

    auc_all   = [r["d_auc"] for r in rows]
    auc_clean = [r["d_auc"] for r in rows if not r["drifted"]]
    _t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(auc_all), 2.776)
    def _ci(v): return _t * (np.std(v, ddof=1) / len(v) ** 0.5) if len(v) > 1 else None
    m_all, sd_all, ci_all = float(np.mean(auc_all)), float(np.std(auc_all, ddof=1)), _ci(auc_all)

    result = {
        "metric_primary":   "d_auc",
        "tau_sens":         TAU_SENS,
        "excluded_folds":   [r["fold"] for r in rows if r["drifted"]],
        "per_fold":         rows,
        "mean_d_auc_all":   m_all,
        "std_d_auc_all":    sd_all,
        "ci_95_auc_all":    float(ci_all) if ci_all else None,
        "mean_d_auc_clean": float(np.mean(auc_clean)) if auc_clean else None,
        "mean_d_spec_all":  float(np.mean([r["d_spec"] for r in rows])),
        "mean_d_sens_all":  float(np.mean([r["d_sens"] for r in rows])),
    }
    path = os.path.join(MODEL_OUTPUT, "paired_augment_comparison.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  mean dAUC (all 5) = {m_all:+.4f} +/- {sd_all:.4f}  (pooled-SD gate)  -> {path}")
    return result


# ── Grad-CAM before/after (baseline model vs augment model, same off images) ───

def gradcam_before_after(baseline_summary, augment_summary):
    import torch
    from torchvision import transforms
    from gradcam import GradCAM, overlay_heatmap  # noqa: E402
    from model import build_model                  # noqa: E402
    from dataset import _BASE_CACHE_DIR            # noqa: E402

    log("=== Grad-CAM before/after (center-energy) ===")
    out_dir = os.path.join(ROOT, "gradcam_augment")
    os.makedirs(out_dir, exist_ok=True)
    image_dir = os.path.join(ROOT, _BASE_CACHE_DIR["off"])
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _cache = {}
    def net_for(arm, fold):
        key = (arm, fold)
        if key not in _cache:
            p = os.path.join(MODEL_OUTPUT, f"honest_{arm}_fold{fold}", "best_model.pt")
            if not os.path.exists(p):
                _cache[key] = None
            else:
                st = torch.load(p, map_location=device, weights_only=False)
                net, _ = build_model(st.get("arch", "efficientnet_b0"), num_classes=2, pretrained=False)
                net.load_state_dict(st["model_state"]); net.to(device).eval()
                _cache[key] = net
        return _cache[key]

    def cam_stats(arm, iid, fold, T):
        net = net_for(arm, fold)
        img_bgr = load_cached_bgr(iid, image_dir)
        if net is None or img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = tfm(img_rgb).unsqueeze(0).to(device)
        with torch.enable_grad():
            cam = GradCAM(net, net.features[-1]).generate(tensor, class_idx=1)
        with torch.no_grad():
            l_o = net(tensor); l_h = net(torch.flip(tensor, dims=[3])); l_v = net(torch.flip(tensor, dims=[2]))
            prob = float(torch.softmax(((l_o + l_h + l_v) / 3) / T, dim=1)[0, 1].cpu())
        h, w = cam.shape
        bd = max(1, int(0.15 * min(h, w)))
        inner = np.zeros_like(cam, dtype=bool); inner[bd:h - bd, bd:w - bd] = True
        ce = float(cam[inner].sum() / (cam.sum() + 1e-9))
        overlay = overlay_heatmap(img_rgb, cam)
        panel = np.concatenate([img_bgr, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)], axis=1)
        cv2.imwrite(os.path.join(out_dir, f"{iid}_{arm}.jpg"), panel)
        return prob, ce

    id_to_fold = {}
    for fi in range(N_FOLDS):
        with open(os.path.join(SPLIT_DIR, f"fold{fi}.json")) as f:
            for iid in json.load(f)["held_out"]:
                id_to_fold[iid] = fi
    b_by = {r["fold"]: r for r in baseline_summary["folds"]}
    a_by = {r["fold"]: r for r in augment_summary["folds"]}

    ce_rows = []
    log("  id            fold  ce: base->aug   prob: base->aug   outcome")
    for iid in FN_IDS + CONTROL_TPS:
        fold = id_to_fold.get(iid)
        if fold is None:
            log(f"  {iid}: not in any held_out. skip."); continue
        bstat = cam_stats("baseline", iid, fold, b_by[fold]["T"])
        astat = cam_stats("augment",  iid, fold, a_by[fold]["T"])
        if not bstat or not astat:
            log(f"  {iid}: missing ckpt/img. skip."); continue
        bp, bce = bstat; ap, ace = astat
        thr = a_by[fold]["frozen_threshold"]
        is_fn = iid in FN_IDS
        outcome = ("FN->correct" if is_fn and ap >= thr else
                   "FN->still-wrong" if is_fn else "TP-control")
        ce_rows.append({"id": iid, "fold": fold, "is_fn": is_fn,
                        "ce_base": bce, "ce_aug": ace, "prob_base": bp,
                        "prob_aug": ap, "thr": thr, "outcome": outcome})
        log(f"  {iid}  f{fold}   {bce:.3f}->{ace:.3f}    {bp:.3f}->{ap:.3f}    {outcome}")

    with open(os.path.join(out_dir, "center_energy.json"), "w") as f:
        json.dump(ce_rows, f, indent=2)
    return ce_rows


# ── report ────────────────────────────────────────────────────────────────────

def write_report(baseline_summary, augment_summary, cmp_result, ce_rows):
    log("=== Writing report_augment.md ===")
    b, t, p = baseline_summary, augment_summary, cmp_result
    m, sd = p["mean_d_auc_all"], p["std_d_auc_all"]
    ci = p.get("ci_95_auc_all")
    ci_str = f"[{m - ci:+.4f}, {m + ci:+.4f}]" if ci else "n/a"

    if abs(m) < sd:
        verdict = (f"**Neutral / no detectable effect.** Mean paired dAUC = {m:+.4f} "
                   f"(pooled SD {sd:.4f}; 95%CI {ci_str}) is within fold noise. At N=5 "
                   f"correlated folds this is a power ceiling -- NOT proof the hypothesis "
                   f"is false. Stronger spatial augmentation did not measurably change "
                   f"discrimination at this training scale.")
    elif m > 0:
        verdict = (f"**Improvement.** Mean paired dAUC = {m:+.4f} exceeds the pooled SD "
                   f"({sd:.4f}; 95%CI {ci_str}). Stronger spatial augmentation improved "
                   f"discrimination.")
    else:
        verdict = (f"**Regression.** Mean paired dAUC = {m:+.4f} is below -pooled SD "
                   f"({sd:.4f}; 95%CI {ci_str}). The augmentation hurt discrimination "
                   f"(likely lesion-crop label noise).")

    fn = [r for r in ce_rows if r["is_fn"]]
    tp = [r for r in ce_rows if not r["is_fn"]]
    n_flip = sum(1 for r in fn if r["outcome"] == "FN->correct")
    ce_move = np.mean([r["ce_aug"] - r["ce_base"] for r in fn]) if fn else float("nan")

    L = [
        "# Spatial-Augmentation Ablation — Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  "
        f"**Arm:** `augment` = baseline recipe + `--spatial_aug` "
        f"(RandomResizedCrop scale {CROP_SCALE} + rotation {ROT_DEG}, ColorJitter unchanged)",
        "",
        "Control = the existing `kfold_honest_baseline_summary.json` (same recipe, seed 42, "
        "byte-identical honest_splits, no extra spatial aug). Augmentation is train-time only; "
        "both arms are scored on the identical resize-only `off` cache.",
        "",
        "## 1. Honest 5-Fold Results (AUC primary)",
        "",
        "| Arm | AUC (primary) | Spec@95 | Sensitivity |",
        "|-----|---------------|---------|-------------|",
        f"| Baseline | {b['mean_auc']:.4f} +/- {b['std_auc']:.4f} "
        f"| {b['mean_specificity']:.4f} +/- {b['std_specificity']:.4f} "
        f"| {b['mean_sensitivity']:.4f} +/- {b['std_sensitivity']:.4f} |",
        f"| Augment  | {t['mean_auc']:.4f} +/- {t['std_auc']:.4f} "
        f"| {t['mean_specificity']:.4f} +/- {t['std_specificity']:.4f} "
        f"| {t['mean_sensitivity']:.4f} +/- {t['std_sensitivity']:.4f} |",
        "",
        "### Per-fold delta (augment − baseline)",
        "",
        "| Fold | dAUC | dSpec | dSens | Drifted? |",
        "|------|------|-------|-------|----------|",
    ]
    for r in p["per_fold"]:
        L.append(f"| {r['fold']} | {r['d_auc']:+.4f} | {r['d_spec']:+.4f} | "
                 f"{r['d_sens']:+.4f} | {'YES' if r['drifted'] else 'no'} |")
    L += [
        "",
        f"**Mean paired dAUC (all 5): {m:+.4f} +/- {sd:.4f}  95%CI {ci_str}**",
        "",
        "*Descriptive only — the 5 StratifiedGroupKFold folds share overlapping training "
        "data and are not independent. Baseline std_auc ~0.007, so a genuine gain below "
        "~0.01 AUC sits inside fold noise and reads neutral: a power ceiling at N=5.*",
        "",
        "### Verdict",
        "",
        verdict,
        "",
        "## 2. Mechanistic Grad-CAM (does attention move onto the lesion?)",
        "",
        f"Before/after overlays in `gradcam_augment/` (baseline vs augment checkpoint, "
        f"same off-cache images). FN cases flipped to correct: **{n_flip}/{len(fn)}**. "
        f"Mean center-energy change on FN cases: **{ce_move:+.3f}** "
        "(positive = attention moved toward the lesion).",
        "",
        "| Image ID | Fold | center-energy base→aug | prob base→aug | outcome |",
        "|----------|------|------------------------|----------------|---------|",
    ]
    for r in fn + tp:
        L.append(f"| {r['id']} | {r['fold']} | {r['ce_base']:.3f} → {r['ce_aug']:.3f} "
                 f"| {r['prob_base']:.3f} → {r['prob_aug']:.3f} | {r['outcome']} |")
    L += [
        "",
        "TP-control rows are high-confidence correct melanomas (on-lesion attention "
        "reference); FN rows are the tracked hard false negatives.",
        "",
        "---",
        f"*Generated by run_augment.py on {time.strftime('%Y-%m-%d %H:%M')}*",
    ]
    path = os.path.join(ROOT, "report_augment.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"  Wrote {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true",
                    help="Run the lesion-in-frame gate + QC grid, then STOP (no training).")
    args = ap.parse_args()

    log("=" * 60); log("run_augment.py  START"); log("=" * 60)
    try:
        preflight()
        if args.preflight_only:
            log("preflight-only: stopping before training as requested. "
                "Review augment_qc/preflight_grid.jpg, then rerun without --preflight-only.")
            return

        if not os.path.exists(BASELINE_SUMMARY):
            abort(f"missing {BASELINE_SUMMARY} -- the baseline arm must exist first (run_all.py).")
        with open(BASELINE_SUMMARY) as f:
            baseline_summary = json.load(f)
        assert_paired_integrity(baseline_summary)

        augment_summary = run_augment_arm()
        cmp_result = compare(baseline_summary, augment_summary)
        ce_rows = gradcam_before_after(baseline_summary, augment_summary)
        write_report(baseline_summary, augment_summary, cmp_result, ce_rows)

    except SystemExit:
        raise
    except Exception:
        log("[UNCAUGHT EXCEPTION]")
        log(traceback.format_exc())
        abort("Uncaught exception — see run_augment_log.txt")

    log("=" * 60); log("run_augment.py  COMPLETE"); log("=" * 60)


if __name__ == "__main__":
    main()
