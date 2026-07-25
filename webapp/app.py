"""
webapp/app.py

FastAPI server for the melanoma research demonstration.

Routes:
  GET  /         -> the upload page (Jinja template)
  POST /predict  -> multipart image upload -> honest JSON prediction (one model)
  POST /compare  -> multipart image upload -> honest JSON prediction from EVERY
                     registered model, run side by side (Phase 4)

Run locally (from the repo root, G:\\srip):
    uvicorn webapp.app:app --reload
Then open http://127.0.0.1:8000/

The prediction is the calibrated P(melanoma) plus the referral flag at the
selected model's own frozen threshold — the same operating point recorded in
that model's deploy.json. Optional preprocessing toggles (Phase 3) run BEFORE
inference via preprocessing_bridge.py; when any are on, the input is
off-distribution relative to what the model was trained/calibrated on, so the
response is flagged `preprocessing_applied` and the UI must caveat it as
illustrative.

/compare calls inference.infer_pil once per registered model on the SAME
preprocessed image — it never averages or ensembles the resulting
probabilities. Each model's number stays its own calibrated estimate; the
response just adds a `disagree` flag (the two models' own threshold decisions
differ) so the UI can disclose disagreement honestly instead of resolving it.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import List

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))  # so `import inference` works under `uvicorn webapp.app:app`

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

import inference
import preprocessing_bridge

app = FastAPI(title="Melanoma Classifier — Research Demonstration")
templates = Jinja2Templates(directory=str(BASE / "templates"))
# Disable Jinja's template LRU cache: it crashes on this Python 3.14 + Jinja2
# combo (unhashable cache key). Negligible cost for a small local demo.
templates.env.cache = None
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # inference.get_model() is the same public accessor /predict already uses —
    # reading its threshold/stats here is a display-only lookup for the
    # explainer section (§ "Dermoscopy & Models"), not a change to the honest
    # math or the registry itself.
    primary = inference.get_model("effb0")
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "models": inference.available_models(),
            "preprocess_steps": preprocessing_bridge.toggle_catalog(),
            "primary": primary,
        },
    )


async def _decode_and_preprocess(file: UploadFile, preprocess: List[str]):
    """Shared upload validation + preprocessing step for /predict and /compare.
    Both routes must reject the same bad input the same way and run the exact
    same preprocessing_bridge path, so a single-model reading and a compare
    reading are never subject to different rules."""
    unknown = set(preprocess) - set(preprocessing_bridge.TOGGLE_KEYS)
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"Unknown preprocessing step(s): {sorted(unknown)}"
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="That file isn't an image we can read — upload a JPG or PNG dermoscopy image.",
        )

    return preprocessing_bridge.apply_toggles(img, set(preprocess))


def _thumb_b64(processed_img) -> str:
    thumb = processed_img.copy()
    thumb.thumbnail((512, 512))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_key: str = Form("effb0"),
    preprocess: List[str] = Form([]),
):
    if model_key not in {m["key"] for m in inference.available_models()}:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")

    processed_img, steps = await _decode_and_preprocess(file, preprocess)

    result = inference.infer_pil(processed_img, model_key=model_key)
    result["preprocessing_applied"] = bool(preprocess)
    result["steps"] = steps

    if preprocess:
        result["processed_image_b64"] = _thumb_b64(processed_img)

    return JSONResponse(result)


@app.post("/compare")
async def compare(
    file: UploadFile = File(...),
    preprocess: List[str] = Form([]),
):
    """Run every registered model on the SAME preprocessed image and return
    each one's own honest reading side by side. No ensembling: each result is
    exactly what inference.infer_pil already returns for /predict, called once
    per model. The only thing added here is `disagree` — whether the models'
    own frozen-threshold decisions differ — so the UI can disclose that
    honestly instead of resolving it with a fake combined score."""
    processed_img, steps = await _decode_and_preprocess(file, preprocess)
    applied = bool(preprocess)

    results = []
    for m in inference.available_models():
        result = inference.infer_pil(processed_img, model_key=m["key"])
        result["preprocessing_applied"] = applied
        result["steps"] = steps
        results.append(result)

    payload = {
        "results": results,
        "disagree": results[0]["above_threshold"] != results[1]["above_threshold"],
        "preprocessing_applied": applied,
        "steps": steps,
    }

    if preprocess:
        payload["processed_image_b64"] = _thumb_b64(processed_img)

    return JSONResponse(payload)
