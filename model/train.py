"""
model/train.py

Two-stage transfer-learning training: Stage 1 trains only the classifier head
(backbone frozen), Stage 2 fine-tunes the last few backbone blocks at a lower
learning rate. Stages run automatically, in that order.

Configurable via CLI flags for architecture, loss function, and preprocessing
source -- this one script drives the main run AND all four comparison
experiments (preprocessing on/off, MobileNetV2 vs EfficientNetB0, K-fold,
focal vs weighted-CE), via the runner scripts in experiments/.

Recipe upgrades (all OFF by default, so existing runs reproduce bit-for-bit):
  --schedule cosine       cosine LR decay with linear warmup (per stage)
  --warmup_epochs N       warmup length for the cosine schedule (stage 2)
  --early_stop_patience N stop stage 2 once val AUC hasn't improved for N epochs
  --ema [--ema_decay D]   keep an EMA copy of the weights; validate + save THAT
  --label_smoothing S     label smoothing for the (weighted) CE loss
  --mixup / --cutmix      batch-level mixing for extra regularisation
  --strong_aug            add RandAugment + RandomErasing to the train pipeline
  --img_size N            train at resolution N (uses a separate `_N` cache)
These target the two things the logs showed: undertraining (cosine + more
epochs + EMA) and the val->test gap (label smoothing + mixup/cutmix + strong aug).

Checkpoint resume: saves progress every epoch (last_checkpoint.pt) as well as
the best-by-val-AUC model (best_model.pt). Re-running with --resume picks back
up from the exact stage/epoch it left off at (weights + EMA), instead of
restarting Stage 1 from scratch.

Usage (run from the project root, i.e. inside melanoma-core/):
    python model/train.py
    python model/train.py --arch efficientnet_b0 --loss focal
    python model/train.py --epochs_stage2 30 --schedule cosine --ema --mixup --resume
"""
import argparse
import json
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from dataset import (
    IMG_SIZE, cache_dir_for, get_class_weights, lesion_aware_kfold,
    lesion_aware_split, load_metadata, make_dataloaders,
)
from losses import FocalLoss
from model import build_model, freeze_backbone, unfreeze_last_n_blocks


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: no CUDA GPU detected -- falling back to CPU (will be slow).")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of model weights (params AND buffers, so BN
    running stats track too). Validate and deploy the EMA copy -- it's a small,
    reliable boost over the raw last-step weights."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(mv.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(mv)


def make_mixer(args):
    """Returns a per-batch mixing fn (mixup and/or cutmix) or None. The fn maps
    (imgs, labels) -> (mixed_imgs, y_a, y_b, lam); loss is then the lam-weighted
    blend of CE(.,y_a) and CE(.,y_b)."""
    if not (args.mixup or args.cutmix):
        return None

    def mixer(imgs, labels):
        use_cut = args.cutmix and (not args.mixup or np.random.rand() < 0.5)
        lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
        idx = torch.randperm(imgs.size(0), device=imgs.device)
        if use_cut:
            h, w = imgs.shape[2], imgs.shape[3]
            r = math.sqrt(1.0 - lam)
            cw, ch = int(w * r), int(h * r)
            cx, cy = np.random.randint(w), np.random.randint(h)
            x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
            y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
            imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))  # actual mixed area
        else:
            imgs = lam * imgs + (1.0 - lam) * imgs[idx]
        return imgs, labels, labels[idx], lam

    return mixer


def make_scheduler(optimizer, total_epochs, warmup_epochs, enabled):
    """Linear warmup -> cosine decay, stepped once per epoch. Returns None when
    --schedule is not cosine (constant LR, original behaviour)."""
    if not enabled:
        return None

    def factor(epoch):  # epoch = number of completed epochs in this stage
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / denom
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_epoch(model, loader, criterion, optimizer, device, train=True, mix=None, ema=None):
    model.train() if train else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
                if mix is not None:
                    imgs, y_a, y_b, lam = mix(imgs, labels)
                    logits = model(imgs)
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return avg_loss, probs, labels


def quick_auc(probs, labels):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, probs[:, 1])
    except ValueError:
        return float("nan")


def save_checkpoint(path, model, extra, ema=None):
    payload = {"model_state": model.state_dict()}
    if ema is not None:
        payload["ema_state"] = ema.shadow.state_dict()
    payload.update(extra)
    torch.save(payload, path)


def train_one_config(args):
    # Deterministic seeding so --seed is a real, reproducible knob (esp. in
    # --split_json mode, where args.seed otherwise feeds nothing). Seeds the
    # classifier-head init AND the shuffle=True train loader (RandomSampler
    # falls back to the global torch RNG). Logged via run_config.json.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    image_dir = cache_dir_for(args.preprocessing, args.img_size)

    df = load_metadata(task_mode=args.task_mode)

    unfreeze_suffix = "" if args.unfreeze_blocks == 3 else f"_unfreeze{args.unfreeze_blocks}"
    if args.split_json:
        with open(args.split_json) as _f:
            _splits = json.load(_f)
        train_df = df[df["image_id"].isin(_splits["inner_train"])].reset_index(drop=True)
        val_df   = df[df["image_id"].isin(_splits["inner_val"])].reset_index(drop=True)
        test_df  = val_df   # held_out evaluation is done externally by run_kfold_honest.py
        run_tag  = f"{args.arch}_{args.loss}_{args.preprocessing}{unfreeze_suffix}_splitjson"
    elif args.fold is not None:
        # K-fold mode: this fold's held-out split IS its evaluation set.
        splits = list(lesion_aware_kfold(df, n_splits=args.n_folds, seed=args.seed))
        train_df, val_df = splits[args.fold]
        test_df = val_df
        run_tag = f"{args.arch}_{args.loss}_{args.preprocessing}{unfreeze_suffix}_fold{args.fold}"
    else:
        train_df, val_df, test_df = lesion_aware_split(df, seed=args.seed)
        run_tag = f"{args.arch}_{args.loss}_{args.preprocessing}{unfreeze_suffix}_main"

    if args.run_name:  # explicit override -> never clobber an existing locked run
        run_tag = args.run_name

    train_loader, val_loader, _ = make_dataloaders(
        train_df, val_df, test_df, image_dir, batch_size=args.batch_size,
        img_size=args.img_size, strong_aug=args.strong_aug, spatial_aug=args.spatial_aug,
    )

    num_classes = 2 if args.task_mode == "binary" else 7
    net, backbone = build_model(args.arch, num_classes=num_classes)
    net.to(device)

    ema = EMA(net, args.ema_decay) if args.ema else None
    mixer = make_mixer(args)

    class_weights = get_class_weights(train_df, args.task_mode).to(device)
    if args.loss == "focal":
        if args.label_smoothing > 0:
            print("NOTE: --label_smoothing is ignored with --loss focal.")
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=args.label_smoothing)

    out_dir = os.path.join("model_output", run_tag)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log.csv")
    best_ckpt_path = os.path.join(out_dir, "best_model.pt")
    last_ckpt_path = os.path.join(out_dir, "last_checkpoint.pt")

    best_val_auc = -1.0
    global_epoch = 0
    stage1_start, stage2_start, skip_stage1 = 0, 0, False

    if os.path.exists(best_ckpt_path):
        best_val_auc = torch.load(best_ckpt_path, map_location=device).get("val_auc", -1.0)

    if args.resume and os.path.exists(last_ckpt_path):
        last = torch.load(last_ckpt_path, map_location=device)
        net.load_state_dict(last["model_state"])
        if ema is not None and "ema_state" in last:
            ema.shadow.load_state_dict(last["ema_state"])
        global_epoch = last["global_epoch"]
        if last["stage"] == "stage1":
            stage1_start = last["epoch_in_stage"]
        else:
            skip_stage1 = True
            stage2_start = last["epoch_in_stage"]
        print(f"Resuming from {last_ckpt_path}: stage={last['stage']}, "
              f"epoch_in_stage={last['epoch_in_stage']}")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("stage,global_epoch,train_loss,val_loss,val_auc\n")

    def log_row(stage, train_loss, val_loss, val_auc):
        with open(log_path, "a") as f:
            f.write(f"{stage},{global_epoch},{train_loss:.4f},{val_loss:.4f},{val_auc:.4f}\n")

    def maybe_save_best(val_auc, eval_model):
        nonlocal best_val_auc
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(best_ckpt_path, eval_model, {
                "arch": args.arch, "task_mode": args.task_mode,
                "val_auc": val_auc, "global_epoch": global_epoch,
            })
            return True
        return False

    cosine = args.schedule == "cosine"

    def run_stage(stage, optimizer, start_epoch, total_epochs, warmup):
        """Returns nothing; mutates global_epoch/best via closures. Early stop
        (patience) only applies in stage2. Stage 1 always uses constant LR --
        it's a short head warmup, not a place for cosine decay."""
        nonlocal global_epoch
        scheduler = make_scheduler(optimizer, total_epochs, warmup,
                                   cosine and stage == "stage2")
        for _ in range(start_epoch):  # fast-forward LR schedule on resume
            if scheduler:
                scheduler.step()
        no_improve = 0
        for epoch in range(start_epoch, total_epochs):
            t0 = time.time()
            train_loss, _, _ = run_epoch(net, train_loader, criterion, optimizer,
                                         device, train=True, mix=mixer, ema=ema)
            eval_model = ema.shadow if ema is not None else net
            val_loss, val_probs, val_labels = run_epoch(eval_model, val_loader, criterion,
                                                         optimizer, device, train=False)
            val_auc = quick_auc(val_probs, val_labels)
            global_epoch += 1
            log_row(stage, train_loss, val_loss, val_auc)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch + 1}/{total_epochs} | train_loss {train_loss:.4f} "
                  f"| val_loss {val_loss:.4f} | val_auc {val_auc:.4f} "
                  f"| lr {lr_now:.2e} | {time.time() - t0:.0f}s")
            improved = maybe_save_best(val_auc, eval_model)
            save_checkpoint(last_ckpt_path, net, {
                "stage": stage, "epoch_in_stage": epoch + 1, "global_epoch": global_epoch,
            }, ema=ema)
            if scheduler:
                scheduler.step()
            if stage == "stage2" and args.early_stop_patience > 0:
                no_improve = 0 if improved else no_improve + 1
                if no_improve >= args.early_stop_patience:
                    print(f"  early stop: no val-AUC improvement in "
                          f"{args.early_stop_patience} epochs.")
                    break

    # ---- Stage 1: frozen backbone, train classifier head only ----
    if not skip_stage1:
        freeze_backbone(backbone)
        Opt = AdamW if args.weight_decay > 0 else Adam
        optimizer = Opt([p for p in net.parameters() if p.requires_grad],
                        lr=args.lr_stage1, weight_decay=args.weight_decay)
        print(f"\n[{run_tag}] Stage 1: head-only, epochs {stage1_start}->{args.epochs_stage1}")
        run_stage("stage1", optimizer, stage1_start, args.epochs_stage1, warmup=0)

    # ---- Stage 2: unfreeze last few blocks, fine-tune at lower LR ----
    unfreeze_last_n_blocks(backbone, n=args.unfreeze_blocks)
    Opt = AdamW if args.weight_decay > 0 else Adam
    optimizer = Opt([p for p in net.parameters() if p.requires_grad],
                    lr=args.lr_stage2, weight_decay=args.weight_decay)
    print(f"\n[{run_tag}] Stage 2: fine-tune last {args.unfreeze_blocks} blocks, "
          f"epochs {stage2_start}->{args.epochs_stage2}")
    run_stage("stage2", optimizer, stage2_start, args.epochs_stage2, warmup=args.warmup_epochs)

    print(f"\nDone. Best val AUC: {best_val_auc:.4f}. Checkpoint: {best_ckpt_path}")
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["mobilenet_v2", "efficientnet_b0", "efficientnet_b3"], default="mobilenet_v2")
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--preprocessing", choices=["on", "off", "color_norm", "devig", "devig_ruler"], default="on")
    parser.add_argument("--task_mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--fold", type=int, default=None,
                         help="0-indexed fold for K-fold CV; omit for the main 70/15/15 run")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=10)
    parser.add_argument("--unfreeze_blocks", type=int, default=3)
    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default=None,
                         help="Override the output-dir name. Use this for new-recipe A/B runs "
                              "so they don't overwrite an existing locked run's checkpoints.")
    parser.add_argument("--split_json", default=None,
                        help="Path to an honest-split JSON (experiments/honest_splits/fold*.json). "
                             "Loads inner_train/inner_val/held_out id lists. Overrides --fold. "
                             "Held-out evaluation is done externally by run_kfold_honest.py.")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE,
                         help="Training resolution; uses the matching `_<size>` cache "
                              "(build it first with dataset.py --img_size).")
    # ---- recipe upgrades (all default to original behaviour) ----
    parser.add_argument("--schedule", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                         help="Stage-2 patience on val AUC; 0 disables early stopping.")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                         help="L2 weight decay for AdamW. 0 keeps original Adam (no decay). "
                              "1e-4 is a good starting point for fine-tuning.")
    parser.add_argument("--strong_aug", action="store_true",
                         help="Add RandAugment + RandomErasing to the train pipeline.")
    parser.add_argument("--spatial_aug", action="store_true",
                         help="Stronger spatial-only augmentation (RandomResizedCrop scale "
                              "0.55-1.0 + rotation 30) to break the positional bias. ColorJitter "
                              "unchanged; train-time only, never val/test.")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from last_checkpoint.pt for this exact run config, "
                              "if one exists -- picks up at the same stage/epoch.")
    args = parser.parse_args()
    train_one_config(args)
