"""
experiments/run_backbone.py -- EfficientNet-B3 backbone (capacity) experiment.

Trains ONE new arm ("backbone" = the exact baseline recipe, but --arch efficientnet_b3)
on the byte-identical honest_splits, then pairs it against the EXISTING B0 baseline summary.
No baseline retrain. Standalone / detached / idempotent / resumable. Reuses the honest
scoring protocol and the sens-floor / NaN-loss hard-abort guards from run_all.py.

Cloned from run_augment.py with three differences:
  - NO preflight gate (no augmentation/cropping change -> nothing to gate).
  - train arm passes --arch efficientnet_b3 (the ONLY change vs baseline) instead of
    --spatial_aug. arch is written into best_model.pt by train.py, so evaluate_fold and
    the Grad-CAM read it back automatically (build_model(ckpt["arch"], ...)).
  - per-fold train/val gap read from the existing training_log.csv (overfit guard).

Resolution: 224 only (the vetted path -- same as the B0 baseline, reuses the 224 `off`
cache and the 224-locked evaluate_fold unchanged). --img_size != 224 is aborted: a 300px
run needs a `_300` cache AND resolution plumbing into evaluate_fold, which this runner does
not do (see PLAN_backbone.md, decision 1).

Usage (from project root G:\\srip):
    python experiments/run_backbone.py

Idempotent skip guards:
  - each fold:  skipped if best_model.pt + >=15 CSV rows; partial -> delete best, --resume
  - arm:        skipped if model_output/kfold_honest_backbone_summary.json exists
  - compare/gradcam/report: always recomputed (cheap)
"""
import argparse
import csv
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

ARM              = "backbone"
ARCH             = "efficientnet_b3"          # THE variable vs the B0 baseline
LOG_FILE         = os.path.join(ROOT, "run_backbone_log.txt")
BASELINE_SUMMARY = os.path.join(MODEL_OUTPUT, "kfold_honest_baseline_summary.json")
CONTROL_TPS      = ["ISIC_0031784", "ISIC_0032699", "ISIC_0027261"]
IMG_SIZE         = 224
GAP_FLAG         = 0.15   # ponytail: heuristic; flag a fold if (val_loss - train_loss) at
                          # best epoch exceeds this. Tune against the B0 baseline gap if noisy.


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
        f.write(f"HARD ABORT (run_backbone): {reason}\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.exit(1)


# ── pairing integrity ─────────────────────────────────────────────────────────

def assert_paired_integrity(baseline_summary):
    """Prove the existing B0 baseline summary was scored on the CURRENT honest_splits:
    per-fold inner_val/held_out counts must match. Guards against pairing the B3 arm
    against a baseline computed on different splits/seed."""
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

def train_backbone_fold(fold_idx, img_size):
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
        "--arch",            ARCH,              # the ONLY difference vs the baseline arm
        "--preprocessing",   "off",
        "--epochs_stage1",   "5",
        "--epochs_stage2",   "10",
        "--lr_stage2",       "1e-4",
        "--unfreeze_blocks", "19",
        "--seed",            "42",
        "--img_size",        str(img_size),
        "--split_json",      split_json,
        "--run_name",        run_tag,
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


# ── overfit guard: train/val gap from the existing training_log.csv ────────────

def read_gap(fold_idx):
    """Read this fold's training_log.csv and summarise the train/val gap. No training
    change -- the CSV already logs per-epoch train_loss,val_loss,val_auc. Flags a fold if
    the gap at the best-val-AUC epoch is large, or if val_auc peaked then declined."""
    path = os.path.join(MODEL_OUTPUT, f"honest_{ARM}_fold{fold_idx}", "training_log.csv")
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            rows.append((float(d["train_loss"]), float(d["val_loss"]), float(d["val_auc"])))
    if not rows:
        return None
    best_i = int(np.argmax([r[2] for r in rows]))
    tr_b, vl_b, va_b = rows[best_i]
    va_final = rows[-1][2]
    va_peak = max(r[2] for r in rows)
    gap = vl_b - tr_b
    declined = (va_peak - va_final) > 0.01
    flagged = gap > GAP_FLAG or declined
    return {
        "fold": fold_idx, "best_epoch": best_i + 1, "n_epochs": len(rows),
        "train_loss_at_best": tr_b, "val_loss_at_best": vl_b, "val_auc_at_best": va_b,
        "val_auc_final": va_final, "val_auc_peak": va_peak,
        "gap": gap, "declined_after_peak": declined, "flagged": flagged,
    }


# ── arm runner ────────────────────────────────────────────────────────────────

def run_backbone_arm(img_size):
    summary_path = os.path.join(MODEL_OUTPUT, f"kfold_honest_{ARM}_summary.json")
    if os.path.exists(summary_path):
        log(f"Arm '{ARM}': summary already exists. Skipping arm.")
        with open(summary_path) as f:
            return json.load(f)

    log(f"=== ARM '{ARM}' ({ARCH}, baseline recipe, preprocessing=off, {img_size}px) ===")
    fold_results, gaps = [], []
    for fi in range(N_FOLDS):
        log(f"--- fold{fi} ---")
        ckpt = train_backbone_fold(fi, img_size)
        log(f"  evaluating fold{fi} ...")
        res = evaluate_fold(ckpt, fi, "off")  # off cache; honest protocol; sens<0.90 hard-aborts
        fold_results.append(res)
        g = read_gap(fi)
        if g:
            gaps.append(g)
            log(f"  fold{fi} gap: best_ep={g['best_epoch']} train={g['train_loss_at_best']:.3f} "
                f"val={g['val_loss_at_best']:.3f} gap={g['gap']:+.3f} "
                f"val_auc {g['val_auc_peak']:.4f}->{g['val_auc_final']:.4f}"
                f"{'  [OVERFIT-FLAG]' if g['flagged'] else ''}")
        log(f"  fold{fi}: T={res['T']:.3f} thr={res['frozen_threshold']:.3f} "
            f"sens={res['achieved_sensitivity']:.3f} spec={res['achieved_specificity']:.3f} "
            f"auc={res['auc']:.4f}")

    specs = [r["achieved_specificity"] for r in fold_results]
    senss = [r["achieved_sensitivity"] for r in fold_results]
    aucs  = [r["auc"] for r in fold_results]
    summary = {
        "arm": ARM, "arch": ARCH, "img_size": img_size, "preprocessing": "off",
        "folds": fold_results, "gaps": gaps,
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

def compare(baseline_summary, backbone_summary):
    log(f"=== Paired comparison ({ARM} - baseline), AUC primary ===")
    b = {r["fold"]: r for r in baseline_summary["folds"]}
    t = {r["fold"]: r for r in backbone_summary["folds"]}
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
    path = os.path.join(MODEL_OUTPUT, "paired_backbone_comparison.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  mean dAUC (all 5) = {m_all:+.4f} +/- {sd_all:.4f}  (pooled-SD gate)  -> {path}")
    return result


# ── Grad-CAM before/after (B0 baseline vs B3 backbone, same off images) ────────

def gradcam_before_after(baseline_summary, backbone_summary):
    import torch
    from torchvision import transforms
    from gradcam import GradCAM, overlay_heatmap  # noqa: E402
    from model import build_model                  # noqa: E402
    from dataset import _BASE_CACHE_DIR            # noqa: E402

    log("=== Grad-CAM before/after (center-energy) ===")
    out_dir = os.path.join(ROOT, "gradcam_backbone")
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
    a_by = {r["fold"]: r for r in backbone_summary["folds"]}

    ce_rows = []
    log("  id            fold  ce: base->b3   prob: base->b3   outcome")
    for iid in FN_IDS + CONTROL_TPS:
        fold = id_to_fold.get(iid)
        if fold is None:
            log(f"  {iid}: not in any held_out. skip."); continue
        bstat = cam_stats("baseline", iid, fold, b_by[fold]["T"])
        astat = cam_stats(ARM,        iid, fold, a_by[fold]["T"])
        if not bstat or not astat:
            log(f"  {iid}: missing ckpt/img. skip."); continue
        bp, bce = bstat; ap, ace = astat
        thr = a_by[fold]["frozen_threshold"]
        is_fn = iid in FN_IDS
        outcome = ("FN->correct" if is_fn and ap >= thr else
                   "FN->still-wrong" if is_fn else "TP-control")
        ce_rows.append({"id": iid, "fold": fold, "is_fn": is_fn,
                        "ce_base": bce, "ce_b3": ace, "prob_base": bp,
                        "prob_b3": ap, "thr": thr, "outcome": outcome})
        log(f"  {iid}  f{fold}   {bce:.3f}->{ace:.3f}    {bp:.3f}->{ap:.3f}    {outcome}")

    with open(os.path.join(out_dir, "center_energy.json"), "w") as f:
        json.dump(ce_rows, f, indent=2)
    return ce_rows


# ── report ────────────────────────────────────────────────────────────────────

def write_report(baseline_summary, backbone_summary, cmp_result, ce_rows):
    log("=== Writing report_backbone.md ===")
    b, t, p = baseline_summary, backbone_summary, cmp_result
    m, sd = p["mean_d_auc_all"], p["std_d_auc_all"]
    ci = p.get("ci_95_auc_all")
    ci_str = f"[{m - ci:+.4f}, {m + ci:+.4f}]" if ci else "n/a"
    img_size = t.get("img_size", IMG_SIZE)

    if abs(m) < sd:
        verdict = (f"**Neutral / no detectable effect.** Mean paired dAUC = {m:+.4f} "
                   f"(pooled SD {sd:.4f}; 95%CI {ci_str}) is within fold noise. At N=5 "
                   f"correlated folds this is a power ceiling -- NOT proof the capacity "
                   f"hypothesis is false. A larger (B3) backbone did not measurably change "
                   f"discrimination at this training scale/recipe.")
    elif m > 0:
        verdict = (f"**Improvement.** Mean paired dAUC = {m:+.4f} exceeds the pooled SD "
                   f"({sd:.4f}; 95%CI {ci_str}). The larger B3 backbone improved "
                   f"discrimination -- capacity was a real lever.")
    else:
        verdict = (f"**Regression.** Mean paired dAUC = {m:+.4f} is below -pooled SD "
                   f"({sd:.4f}; 95%CI {ci_str}). B3 underperformed B0 under this recipe "
                   f"(see the train/val-gap table -- likely overfitting, not a capacity ceiling).")

    fn = [r for r in ce_rows if r["is_fn"]]
    tp = [r for r in ce_rows if not r["is_fn"]]
    n_flip = sum(1 for r in fn if r["outcome"] == "FN->correct")
    ce_move = np.mean([r["ce_b3"] - r["ce_base"] for r in fn]) if fn else float("nan")
    gaps = t.get("gaps", [])
    n_flagged = sum(1 for g in gaps if g["flagged"])

    L = [
        "# EfficientNet-B3 Backbone (Capacity) Experiment — Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  "
        f"**Arm:** `backbone` = baseline recipe with `--arch efficientnet_b3` at {img_size}px",
        "",
        "Control = the existing `kfold_honest_baseline_summary.json` (EffNet-**B0**, same "
        "recipe, seed 42, byte-identical honest_splits). Both arms are scored on the identical "
        "resize-only `off` cache with the same honest protocol (T-calibration on inner_val, "
        "frozen 95%-sens threshold, TTA on held_out). The B3 checkpoint records its arch, so "
        "`evaluate_fold` and the Grad-CAM load B3 automatically.",
        "",
        "> **Not a strict single-variable ablation.** B0→B3 changes depth, width, and "
        "stochastic-depth internally, and the recipe (LR 1e-4, 5+10 epochs, all blocks "
        f"unfrozen) was tuned on B0. At {img_size}px the *resolution* is held equal to the "
        "baseline, so this isolates capacity-at-fixed-resolution -- but it is an "
        "**architecture comparison**, not a one-knob change.",
        "",
        "## 1. Honest 5-Fold Results (AUC primary)",
        "",
        "| Arm | AUC (primary) | Spec@95 | Sensitivity |",
        "|-----|---------------|---------|-------------|",
        f"| Baseline (B0) | {b['mean_auc']:.4f} +/- {b['std_auc']:.4f} "
        f"| {b['mean_specificity']:.4f} +/- {b['std_specificity']:.4f} "
        f"| {b['mean_sensitivity']:.4f} +/- {b['std_sensitivity']:.4f} |",
        f"| Backbone (B3) | {t['mean_auc']:.4f} +/- {t['std_auc']:.4f} "
        f"| {t['mean_specificity']:.4f} +/- {t['std_specificity']:.4f} "
        f"| {t['mean_sensitivity']:.4f} +/- {t['std_sensitivity']:.4f} |",
        "",
        "### Per-fold delta (B3 − baseline)",
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
        "## 2. Overfitting guard (train/val gap, from training_log.csv)",
        "",
        f"Folds flagged: **{n_flagged}/{len(gaps)}** "
        f"(flag if gap = val_loss − train_loss at best epoch > {GAP_FLAG}, "
        "or val_auc declined >0.01 from its peak).",
        "",
        "| Fold | best ep | train_loss | val_loss | gap | val_auc peak→final | flag |",
        "|------|---------|------------|----------|-----|--------------------|------|",
    ]
    for g in gaps:
        L.append(f"| {g['fold']} | {g['best_epoch']}/{g['n_epochs']} "
                 f"| {g['train_loss_at_best']:.3f} | {g['val_loss_at_best']:.3f} "
                 f"| {g['gap']:+.3f} | {g['val_auc_peak']:.4f}→{g['val_auc_final']:.4f} "
                 f"| {'OVERFIT' if g['flagged'] else 'ok'} |")
    L += [
        "",
        "## 3. Mechanistic Grad-CAM (does the bigger backbone localize better?)",
        "",
        f"Before/after overlays in `gradcam_backbone/` (B0 vs B3 checkpoint, same off-cache "
        f"images). FN cases flipped to correct: **{n_flip}/{len(fn)}**. "
        f"Mean center-energy change on FN cases: **{ce_move:+.3f}** "
        "(positive = attention moved toward the lesion).",
        "",
        "| Image ID | Fold | center-energy base→B3 | prob base→B3 | outcome |",
        "|----------|------|-----------------------|---------------|---------|",
    ]
    for r in fn + tp:
        L.append(f"| {r['id']} | {r['fold']} | {r['ce_base']:.3f} → {r['ce_b3']:.3f} "
                 f"| {r['prob_base']:.3f} → {r['prob_b3']:.3f} | {r['outcome']} |")
    L += [
        "",
        "TP-control rows are high-confidence correct melanomas (on-lesion attention "
        "reference); FN rows are the tracked hard false negatives.",
        "",
        "## 4. Next steps (conditional, NOT auto-run)",
        "",
        "- If B3 **regressed with flagged folds** → overfitting, not a capacity ceiling: "
        "re-run B3 with `--weight_decay 1e-4` (AdamW) + `--early_stop_patience 3` (or fewer "
        "stage-2 epochs).",
        "- If B3 was **neutral/positive at 224** → the native-resolution run "
        "(`--img_size 300`) disambiguates capacity from resolution. Requires building the "
        "`_300` `off` cache (`dataset.py --img_size 300`) AND resolution plumbing into "
        "`evaluate_fold` (not wired in this runner).",
        "",
        "---",
        f"*Generated by run_backbone.py on {time.strftime('%Y-%m-%d %H:%M')}*",
    ]
    path = os.path.join(ROOT, "report_backbone.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"  Wrote {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=IMG_SIZE,
                    help="Training resolution. Only 224 is supported (the vetted path, "
                         "matching the B0 baseline). 300 needs a _300 cache AND eval-resolution "
                         "plumbing -- see PLAN_backbone.md decision 1.")
    args = ap.parse_args()

    log("=" * 60); log("run_backbone.py  START"); log("=" * 60)
    try:
        if args.img_size != IMG_SIZE:
            abort(f"--img_size {args.img_size} not supported: evaluate_fold is 224-locked, so a "
                  f"{args.img_size}px train would be scored at 224 (train/serve skew). Run at 224, "
                  f"or wire resolution into evaluate_fold + build the _{args.img_size} cache first.")

        if not os.path.exists(BASELINE_SUMMARY):
            abort(f"missing {BASELINE_SUMMARY} -- the B0 baseline arm must exist first (run_all.py).")
        with open(BASELINE_SUMMARY) as f:
            baseline_summary = json.load(f)
        assert_paired_integrity(baseline_summary)

        backbone_summary = run_backbone_arm(args.img_size)
        cmp_result = compare(baseline_summary, backbone_summary)
        ce_rows = gradcam_before_after(baseline_summary, backbone_summary)
        write_report(baseline_summary, backbone_summary, cmp_result, ce_rows)

    except SystemExit:
        raise
    except Exception:
        log("[UNCAUGHT EXCEPTION]")
        log(traceback.format_exc())
        abort("Uncaught exception — see run_backbone_log.txt")

    log("=" * 60); log("run_backbone.py  COMPLETE"); log("=" * 60)


if __name__ == "__main__":
    main()
