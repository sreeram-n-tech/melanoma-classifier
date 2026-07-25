"""
webapp/inference.py

Single-image inference for the melanoma research demonstration.

This reuses the EXACT honest math from the research code — nothing here invents a
new pipeline:

  * `build_model` is imported from the research tree (`model/model.py`); it pulls
    in torch/torchvision only, no heavy side effects.
  * The eval transform, the 4 test-time-augmentation (TTA) flips, and the
    temperature-scaling step are vendored verbatim below from
    `model/dataset.py` (the `augment=False` transform) and `finalize.py`
    (`TTA_OPS`, `probs_pos`). They are kept here — rather than imported — so the
    web process does not drag in pandas/matplotlib (which `finalize.py` imports
    transitively). `parity_check.py` proves this vendored copy reproduces the
    numbers frozen in `deploy.json` bit-for-bit.

Calibration is NEVER recomputed here. Each model's temperature and decision
threshold are read from its `deploy.json` — the operating point that
`finalize.py` froze on the held-out validation split. Recomputing them would
require the HAM10000 validation set and would defeat the point of a frozen,
honest operating point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T

# --- import build_model from the research tree (read-only) --------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "model"))
from model import build_model  # noqa: E402  (research code, imported not copied)

IMG_SIZE = 224

# Mirrors model/dataset.py, HAM10000Dataset augment=False branch, verbatim.
_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRANSFORM = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), _NORM])

# Mirrors finalize.py TTA_OPS verbatim: identity, h-flip, v-flip, both.
# Dermoscopy images have no canonical orientation, so all four are
# label-preserving.
TTA_OPS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.flip(x, dims=[2, 3]),
]

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Registry of servable models. Phase 2 adds the MobileNetV2 honest baseline
# alongside the EfficientNet-B0 primary baseline. (The uncalibrated B3 is a
# separate open decision — not added here.) Each entry's temperature,
# threshold, TTA flag, and stats are read from its OWN deploy.json — never
# shared or borrowed across models.
_REGISTRY_SPEC = {
    "effb0": {
        "display_name": "EfficientNet-B0",
        "subtitle": "primary baseline",
        "dir": "effb0",
    },
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "subtitle": "honest baseline",
        "dir": "mobilenetv2",
    },
}


class ModelEntry:
    """A loaded model plus the frozen operating point read from its deploy.json."""

    def __init__(self, key: str, spec: dict):
        model_dir = MODELS_DIR / spec["dir"]

        with open(model_dir / "deploy.json") as f:
            deploy = json.load(f)

        ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu")
        arch = ckpt["arch"]
        net, _ = build_model(arch, num_classes=2, pretrained=False)
        net.load_state_dict(ckpt["model_state"])
        net.eval()

        self.key = key
        self.display_name = spec["display_name"]
        self.subtitle = spec["subtitle"]
        self.arch = arch
        self.net = net

        # Frozen, honest operating point — straight from deploy.json.
        self.temperature = float(deploy["temperature"])
        self.threshold = float(deploy["threshold"])
        self.tta = bool(deploy["tta"])
        self.calibrated = True  # models with a deploy.json are calibrated
        self.stats = {
            "auc": float(deploy["test_auc"]),
            "sensitivity": float(deploy["test_sensitivity"]),
            "specificity": float(deploy["test_specificity"]),
        }
        self.deploy_raw = deploy


_LOADED: dict[str, ModelEntry] = {}


def get_model(key: str) -> ModelEntry:
    if key not in _REGISTRY_SPEC:
        raise KeyError(key)
    if key not in _LOADED:
        _LOADED[key] = ModelEntry(key, _REGISTRY_SPEC[key])
    return _LOADED[key]


def available_models() -> list[dict]:
    return [
        {"key": k, "display_name": s["display_name"], "subtitle": s["subtitle"]}
        for k, s in _REGISTRY_SPEC.items()
    ]


@torch.no_grad()
def _mean_logits(net, x, tta: bool):
    """Mean raw logits over TTA views. Matches finalize.collect_logits with a
    single model (nets=[net]): sum over ops in logit space, then divide."""
    ops = TTA_OPS if tta else TTA_OPS[:1]
    acc = None
    for op in ops:
        out = net(op(x))
        acc = out if acc is None else acc + out
    return acc / len(ops)


def _prob_pos(logits, temperature: float) -> float:
    """P(melanoma) after temperature scaling. Mirrors finalize.probs_pos."""
    return float(torch.softmax(logits / temperature, dim=1)[0, 1])


@torch.no_grad()
def infer_pil(pil_img, model_key: str = "effb0") -> dict:
    """Run the full honest single-image pipeline on a PIL image and return the
    calibrated probability, the referral decision at the frozen threshold, and
    the model's known held-out performance for honest context."""
    entry = get_model(model_key)
    x = TRANSFORM(pil_img.convert("RGB")).unsqueeze(0)
    logits = _mean_logits(entry.net, x, entry.tta)
    prob = _prob_pos(logits, entry.temperature)
    return {
        "model_key": entry.key,
        "model_name": entry.display_name,
        "calibrated": entry.calibrated,
        "prob": prob,                                   # calibrated P(melanoma), [0,1]
        "percent": round(prob * 100, 1),
        "threshold": entry.threshold,                   # frozen referral threshold
        "above_threshold": bool(prob >= entry.threshold),
        "above_naive_05": bool(prob >= 0.5),            # shown only for contrast
        "temperature": entry.temperature,
        "tta": entry.tta,
        "stats": entry.stats,                           # AUC / sens / spec, held-out
    }
