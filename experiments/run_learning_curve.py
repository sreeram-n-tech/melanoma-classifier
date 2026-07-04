"""
experiments/run_learning_curve.py

Learning curve: AUC vs training-data size for the B0 baseline. Retrains the
baseline recipe on nested, prevalence-held, lesion-grouped subsets of each
fold's inner_train (25/50/75%) and reuses the existing 100% baseline as the
top anchor. Answers: is the ~0.904 AUC ceiling a data-QUANTITY limit?
  rising at 100% -> more data helps.  flat by 75% -> quantity isn't the ceiling.

Standalone / detached / idempotent / resumable. No model in the loop:
    python experiments/run_learning_curve.py            # full sweep
    python experiments/run_learning_curve.py --pilot    # fold0@50% only, print ETA, stop
    python experiments/run_learning_curve.py --selftest  # build+assert subsets, no GPU

Idempotent: each (fraction, model_seed, fold) writes one result JSON; on resume
finished results are skipped. Killing/relaunching never redoes finished work.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, ROOT)

from run_all import (  # noqa: E402
    MODEL_OUTPUT, SPLIT_DIR, N_FOLDS, TOTAL_EPOCHS,
    is_training_complete, _csv_epoch_count, _nan_in_log,
)

# ── config ────────────────────────────────────────────────────────────────────
FRACTIONS      = [25, 50, 75, 100]
SEED_BUDGET    = {25: [42, 43, 44], 50: [42, 43], 75: [42], 100: [42]}  # 3/2/1/1
SUBSAMPLE_SEED = 1234
PREVALENCE_TOL = 0.03          # allowed |image-level mel rate - fold's inner_train rate|
OUT            = os.path.join(ROOT, "results_learning_curve")
SUBSET_DIR     = os.path.join(OUT, "subsets")
RESULT_DIR     = os.path.join(OUT, "results")
LOG_FILE       = os.path.join(OUT, "run_learning_curve_log.txt")
BASELINE_SUMMARY = os.path.join(MODEL_OUTPUT, "kfold_honest_baseline_summary.json")


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def abort(reason):
    log(f"[HARD ABORT] {reason}")
    with open(os.path.join(OUT, "STOP_REPORT.txt"), "w") as f:
        f.write(f"HARD ABORT: {reason}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.exit(1)


# ── subset construction (nested, prevalence-held, lesion-grouped) ──────────────

def _fold_json(fold):
    with open(os.path.join(SPLIT_DIR, f"fold{fold}.json")) as f:
        return json.load(f)


def build_subsets():
    """For each fold, draw nested lesion-level subsets of inner_train at 25/50/75%
    holding mel prevalence, and write reduced split JSONs. Returns
    {(fold, frac): set(image_ids)} for the fractions actually trained (25/50/75)
    plus 100 (the full inner_train), for assertion / logging."""
    from dataset import load_metadata  # noqa: E402
    os.makedirs(SUBSET_DIR, exist_ok=True)
    df = load_metadata()
    sets = {}
    for fold in range(N_FOLDS):
        s = _fold_json(fold)
        train_ids = set(s["inner_train"])
        sub = df[df["image_id"].isin(train_ids)][["image_id", "lesion_id", "label"]]
        lesion_lab = sub.groupby("lesion_id")["label"].max()   # mel if any image is mel
        rng = np.random.RandomState(SUBSAMPLE_SEED)
        # shuffle each class's lesions ONCE -> cumulative prefixes are nested
        by_class = {}
        for cls in (0, 1):
            ids = lesion_lab.index[lesion_lab.values == cls].to_numpy()
            rng.shuffle(ids)
            by_class[cls] = ids
        full_rate = sub["label"].mean()

        sets[(fold, 100)] = set(sub["image_id"])
        for frac in (25, 50, 75):
            chosen = []
            for cls in (0, 1):
                ids = by_class[cls]
                n = int(round(frac / 100.0 * len(ids)))
                chosen.extend(ids[:n])
            chosen = set(chosen)
            img = sub[sub["lesion_id"].isin(chosen)]
            subset_ids = img["image_id"].tolist()
            sets[(fold, frac)] = set(subset_ids)
            rate = img["label"].mean()
            assert abs(rate - full_rate) <= PREVALENCE_TOL, (
                f"fold{fold} frac{frac}: mel rate {rate:.3f} drifts from {full_rate:.3f}")
            out = {"inner_train": subset_ids,
                   "inner_val": s["inner_val"], "held_out": s["held_out"]}
            with open(os.path.join(SUBSET_DIR, f"fold{fold}_frac{frac}.json"), "w") as f:
                json.dump(out, f, separators=(",", ":"))

        # nesting: 25 subset of 50 subset of 75 subset of 100
        prev = sets[(fold, 25)]
        for frac in (50, 75, 100):
            assert prev <= sets[(fold, frac)], f"fold{fold}: frac nesting broke at {frac}"
            prev = sets[(fold, frac)]
        # inner_val / held_out untouched
        for frac in (25, 50, 75):
            with open(os.path.join(SUBSET_DIR, f"fold{fold}_frac{frac}.json")) as f:
                w = json.load(f)
            assert w["inner_val"] == s["inner_val"], f"fold{fold} frac{frac}: inner_val changed"
            assert w["held_out"] == s["held_out"], f"fold{fold} frac{frac}: held_out changed"
        log(f"  fold{fold}: subsets built (inner_train full={len(train_ids)}, "
            f"mel={full_rate:.1%})")
    return sets, df


# ── training + AUC-only eval ───────────────────────────────────────────────────

def _run_tag(frac, seed, fold):
    return f"lc_frac{frac}_seed{seed}_fold{fold}"


def train_and_eval(frac, seed, fold):
    """Train one (frac,seed,fold) and write its result JSON. Skips if the result
    already exists. Returns the result dict."""
    res_path = os.path.join(RESULT_DIR, f"frac{frac}_seed{seed}_fold{fold}.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            return json.load(f)

    run_tag    = _run_tag(frac, seed, fold)
    out_dir    = os.path.join(MODEL_OUTPUT, run_tag)
    best_pt    = os.path.join(out_dir, "best_model.pt")
    subset_json = os.path.join(SUBSET_DIR, f"fold{fold}_frac{frac}.json")
    log_path   = os.path.join(out_dir, "train_stdout.log")
    os.makedirs(out_dir, exist_ok=True)

    if not is_training_complete(out_dir):
        n_done = _csv_epoch_count(out_dir)
        if os.path.exists(best_pt) and n_done < TOTAL_EPOCHS:
            os.remove(best_pt)  # partial -> resume from last_checkpoint
        log(f"  {run_tag}: training (done_so_far={n_done}) ...")
        cmd = [
            sys.executable, os.path.join(ROOT, "model", "train.py"),
            "--arch", "efficientnet_b0", "--preprocessing", "off",
            "--epochs_stage1", "5", "--epochs_stage2", "10",
            "--lr_stage2", "1e-4", "--unfreeze_blocks", "19",
            "--seed", str(seed), "--split_json", subset_json,
            "--run_name", run_tag, "--resume",
        ]
        with open(log_path, "a") as lf:
            r = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            abort(f"training failed for {run_tag} (exit {r.returncode}); see {log_path}")
        if _nan_in_log(log_path):
            abort(f"NaN in training output for {run_tag}; see {log_path}")
        if not os.path.exists(best_pt):
            abort(f"training finished but best_model.pt missing for {run_tag}")
    else:
        log(f"  {run_tag}: training already complete. Skipping.")

    auc = auc_only(best_pt, fold)
    with open(subset_json) as f:
        sub = json.load(f)
    from dataset import load_metadata  # noqa
    df = load_metadata()
    tr = df[df["image_id"].isin(sub["inner_train"])]
    result = {"fraction": frac, "model_seed": seed, "subsample_seed": SUBSAMPLE_SEED,
              "fold": fold, "n_train": int(len(tr)), "n_mel_train": int(tr["label"].sum()),
              "auc": auc}
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  {run_tag}: AUC={auc:.4f}  n_train={result['n_train']}  "
        f"n_mel={result['n_mel_train']}")
    return result


def auc_only(ckpt_path, fold):
    """TTA-AUC on the fold's (unchanged) held_out. AUC is invariant to temperature,
    so calibration/threshold/sens-abort are all skipped."""
    import torch
    from dataset import load_metadata, make_dataloaders, _BASE_CACHE_DIR, IMG_SIZE  # noqa
    from finalize import collect_logits, probs_pos                                  # noqa
    from model import build_model                                                   # noqa

    s = _fold_json(fold)
    df = load_metadata()
    ho_df = df[df["image_id"].isin(s["held_out"])].reset_index(drop=True)
    image_dir = _BASE_CACHE_DIR["off"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net, _ = build_model(ckpt.get("arch", "efficientnet_b0"), num_classes=2, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.to(device).eval()

    _, _, test_loader = make_dataloaders(ho_df, ho_df, ho_df, image_dir,
                                         batch_size=64, img_size=IMG_SIZE)
    logits, labels = collect_logits([net], test_loader, device, tta=True)
    probs = probs_pos(logits, 1.0)   # T=1: AUC is invariant to monotone T
    return float(roc_auc_score(labels, probs))


def reuse_baseline_100():
    """Write the 100% points from the existing baseline summary (no retrain)."""
    with open(BASELINE_SUMMARY) as f:
        b = json.load(f)
    os.makedirs(RESULT_DIR, exist_ok=True)
    for fr in b["folds"]:
        fold = fr["fold"]
        res_path = os.path.join(RESULT_DIR, f"frac100_seed42_fold{fold}.json")
        if os.path.exists(res_path):
            continue
        with open(res_path, "w") as f:
            json.dump({"fraction": 100, "model_seed": 42, "subsample_seed": SUBSAMPLE_SEED,
                       "fold": fold, "n_train": None, "n_mel_train": None,
                       "auc": float(fr["auc"]), "reused_baseline": True}, f, indent=2)
    log("  100% points reused from baseline summary.")


# ── aggregate + plot + report ──────────────────────────────────────────────────

def aggregate():
    results = []
    for fn in os.listdir(RESULT_DIR):
        if fn.endswith(".json"):
            with open(os.path.join(RESULT_DIR, fn)) as f:
                results.append(json.load(f))
    per_frac = {}
    for frac in FRACTIONS:
        aucs = [r["auc"] for r in results if r["fraction"] == frac]
        if aucs:
            per_frac[frac] = {"n": len(aucs), "mean_auc": float(np.mean(aucs)),
                              "std_auc": float(np.std(aucs))}
    summary = {"per_fraction": per_frac, "subsample_seed": SUBSAMPLE_SEED,
               "seed_budget": SEED_BUDGET}
    with open(os.path.join(OUT, "curve_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return per_frac


def plot(per_frac):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fracs = sorted(per_frac)
    means = [per_frac[p]["mean_auc"] for p in fracs]
    stds  = [per_frac[p]["std_auc"] for p in fracs]
    plt.figure(figsize=(6, 4))
    plt.errorbar(fracs, means, yerr=stds, marker="o", capsize=4, label="B0 learning curve")
    plt.axhline(0.904, ls="--", color="gray", label="baseline 0.904")
    plt.xlabel("Training-data fraction (%)"); plt.ylabel("Held-out AUC (TTA)")
    plt.title("Learning curve: AUC vs training-data size"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "learning_curve.png"), dpi=130)
    plt.close()


def write_report(per_frac):
    fracs = sorted(per_frac)
    top, prev = per_frac[100], per_frac[75] if 75 in per_frac else per_frac[max(f for f in fracs if f < 100)]
    delta = top["mean_auc"] - prev["mean_auc"]
    pooled_sd = float(np.mean([per_frac[p]["std_auc"] for p in fracs]))
    rising = delta > pooled_sd
    verdict = (
        f"**Rising.** AUC(100%)-AUC(75%) = {delta:+.4f} exceeds pooled per-fraction SD "
        f"({pooled_sd:.4f}) -> data quantity is still binding; more data plausibly helps."
        if rising else
        f"**Flat.** |AUC(100%)-AUC(75%)| = {delta:+.4f} is within pooled per-fraction SD "
        f"({pooled_sd:.4f}) -> data QUANTITY is not the binding constraint at this scale. "
        f"The ceiling is quality / intrinsic difficulty, not size."
    )
    lines = [
        "# Learning Curve — AUC vs Training-Data Size (B0 baseline)", "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  Nested lesion-grouped subsets, mel prevalence held, "
        f"subsample_seed={SUBSAMPLE_SEED}. 100% = existing baseline summary (TTA-AUC, not retrained).",
        "", "| Fraction | n runs | mean AUC | std |", "|---|---|---|---|",
    ]
    for p in fracs:
        d = per_frac[p]
        lines.append(f"| {p}% | {d['n']} | {d['mean_auc']:.4f} | {d['std_auc']:.4f} |")
    lines += [
        "", "## Verdict", "", verdict, "",
        "*Power limit: 5 StratifiedGroupKFold folds share training data (not independent); "
        "error bars fan out at 25% where mel count per fold is small. A flat read at N=5 is a "
        "power ceiling, not proof.*", "",
        f"![curve](learning_curve.png)", "",
        f"*Generated by run_learning_curve.py on {time.strftime('%Y-%m-%d %H:%M')}*",
    ]
    with open(os.path.join(OUT, "report_learning_curve.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  verdict: {'RISING' if rising else 'FLAT'} (delta={delta:+.4f}, pooled_sd={pooled_sd:.4f})")


# ── sweep driver ───────────────────────────────────────────────────────────────

def sweep(pilot=False):
    log("=" * 60); log("run_learning_curve START"); log("=" * 60)
    if not os.path.exists(BASELINE_SUMMARY):
        abort(f"baseline summary missing: {BASELINE_SUMMARY}")
    build_subsets()

    if pilot:
        t0 = time.time()
        train_and_eval(50, 42, 0)
        dt = time.time() - t0
        n_runs = sum(len(SEED_BUDGET[f]) * N_FOLDS for f in (25, 50, 75))  # 100 reused
        log(f"PILOT: fold0@50% took {dt/60:.1f} min. "
            f"ETA for {n_runs} GPU runs ~= {dt*n_runs/3600:.1f} h "
            f"(minus the 1 just run). Stopping (--pilot).")
        return

    for frac in (25, 50, 75):
        for seed in SEED_BUDGET[frac]:
            for fold in range(N_FOLDS):
                train_and_eval(frac, seed, fold)
    reuse_baseline_100()

    per_frac = aggregate()
    plot(per_frac)
    write_report(per_frac)
    log("=" * 60); log("run_learning_curve COMPLETE"); log("=" * 60)


def selftest():
    """No-GPU check: subset draw is nested, prevalence-held, and leaves
    inner_val/held_out untouched (assertions live in build_subsets)."""
    sets, df = build_subsets()
    for fold in range(N_FOLDS):
        sizes = {p: len(sets[(fold, p)]) for p in FRACTIONS}
        assert sizes[25] < sizes[50] < sizes[75] < sizes[100], f"fold{fold}: sizes not increasing {sizes}"
    print("SELFTEST OK: nested subsets, prevalence held, inner_val/held_out unchanged.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="fold0 at 50pct only, print ETA, stop")
    ap.add_argument("--selftest", action="store_true", help="build+assert subsets, no GPU")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        sweep(pilot=a.pilot)
