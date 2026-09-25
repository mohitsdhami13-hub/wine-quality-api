"""
Wine Quality Prediction API — Vercel zero-config FastAPI version.

Vercel detects a FastAPI instance named `app` at this exact path
(api/index.py is one of its recognized entrypoints) and turns this
WHOLE file into a single Vercel Function. All routing below is handled
normally by FastAPI itself — no vercel.json or rewrites needed.

Routes (no /api prefix — call them directly on your domain):
  GET  /health
  GET  /metrics
  POST /predict
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

# ---- Load models once, at import time ----
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    if METRICS is None:
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return METRICS


@app.post("/predict")
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
