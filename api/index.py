"""
Wine Quality Prediction API — Vercel serverless version.

Vercel's Python runtime looks inside /api for files that expose an
ASGI app named `app`, and wraps each one as a serverless function.
This is the same FastAPI app as the Render version, just relocated
and made path-safe (serverless functions can run from any working
directory, so we load files relative to this script, not the cwd).

Endpoints:
  GET  /api/health
  GET  /api/metrics
  POST /api/predict
"""

import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Wine Quality API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load models once, at import time (Vercel keeps warm instances alive briefly) ----
red_model = joblib.load(BASE_DIR / "wine_red_model.joblib")
white_model = joblib.load(BASE_DIR / "wine_white_model.joblib")

with open(BASE_DIR / "feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

try:
    with open(BASE_DIR / "metrics.json") as f:
        METRICS = json.load(f)
except FileNotFoundError:
    METRICS = None


class WineFeatures(BaseModel):
    wine_type: str = Field(..., pattern="^(red|white)$")
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    if METRICS is None:
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return METRICS


@app.post("/api/predict")
def predict(payload: WineFeatures):
    model = red_model if payload.wine_type == "red" else white_model

    row = {
        "fixed acidity": payload.fixed_acidity,
        "volatile acidity": payload.volatile_acidity,
        "citric acid": payload.citric_acid,
        "residual sugar": payload.residual_sugar,
        "chlorides": payload.chlorides,
        "free sulfur dioxide": payload.free_sulfur_dioxide,
        "total sulfur dioxide": payload.total_sulfur_dioxide,
        "density": payload.density,
        "pH": payload.pH,
        "sulphates": payload.sulphates,
        "alcohol": payload.alcohol,
    }
    try:
        x = np.array([[row[name] for name in FEATURE_NAMES]])
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: {e}")

    pred_label = model.predict(x)[0]
    proba = dict(zip(model.classes_, model.predict_proba(x)[0].round(4).tolist()))

    return {
        "wine_type": payload.wine_type,
        "predicted_label": pred_label,
        "probabilities": proba,
    }
