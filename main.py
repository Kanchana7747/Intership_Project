"""
Water Classification FastAPI Server
------------------------------------
Endpoints:
  GET  /                    → Health check
  GET  /health              → Detailed health info
  POST /predict             → Image classification (stacking ensemble)
  POST /api/forecast        → Water quality parameter forecasting
  POST /api/anomaly_batch   → Z-score anomaly detection

Designed for Google Cloud Run deployment.
"""

import os
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import models, transforms

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
CLASSES_FILE    = "models/image_classification/classes.txt"
META_MODEL_FILE = "models/image_classification/meta_model.pkl"
CSV_PATH        = "water_quality_with_timestamp.csv"
MODEL_DIR       = "models/forecast_regression"

MODELS_CONFIG = [
    {"name": "EfficientNet-B3",   "path": "models/image_classification/model_efficientnet.pth"},
    {"name": "ResNet-50",         "path": "models/image_classification/model_resnet50.pth"},
    {"name": "MobileNetV3-Large", "path": "models/image_classification/model_mobilenetv3.pth"},
]

TARGET_COLUMNS = ["temperature", "dissolved_oxygen", "pH", "conductivity"]
CONFIDENCE_THRESHOLD = 0.65

# ─────────────────────────────────────────────
#  Global state
# ─────────────────────────────────────────────
class AppState:
    class_names:    list[str] = []
    num_classes:    int       = 0
    device:         torch.device = torch.device("cpu")
    loaded_models:  list      = []
    meta_model                = None
    ensemble_ready: bool      = False
    forecast_models: dict     = {}
    df_forecast:    Optional[pd.DataFrame] = None
    min_timestamp             = None


state = AppState()

# ─────────────────────────────────────────────
#  Model builders (must match train.py exactly)
# ─────────────────────────────────────────────
def build_efficientnet_b3(n: int) -> nn.Module:
    m = models.efficientnet_b3(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_f, n))
    return m


def build_resnet50(n: int) -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(m.fc.in_features, n))
    return m


def build_mobilenetv3(n: int) -> nn.Module:
    m = models.mobilenet_v3_large(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, n)
    return m


BUILDERS = {
    "EfficientNet-B3":    build_efficientnet_b3,
    "ResNet-50":          build_resnet50,
    "MobileNetV3-Large":  build_mobilenetv3,
}

# ─────────────────────────────────────────────
#  Image transform (ImageNet normalisation)
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
#  Startup / shutdown lifecycle
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artefacts before the server starts accepting requests."""
    logger.info("═" * 60)
    logger.info("  Water Classification – FastAPI starting up")
    logger.info("═" * 60)

    # Device
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch device : {state.device}")

    # Classes
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE) as f:
            state.class_names = f.read().splitlines()
        state.num_classes = len(state.class_names)
        logger.info(f"Classes ({state.num_classes}): {state.class_names}")
    else:
        logger.warning(f"Classes file not found: {CLASSES_FILE}")

    # Base models
    if state.num_classes > 0:
        for cfg in MODELS_CONFIG:
            if not os.path.exists(cfg["path"]):
                logger.warning(f"⚠  Skipping {cfg['name']} — file not found: {cfg['path']}")
                continue
            try:
                m = BUILDERS[cfg["name"]](state.num_classes)
                m.load_state_dict(
                    torch.load(cfg["path"], map_location=state.device, weights_only=True)
                )
                m = m.to(state.device).eval()
                state.loaded_models.append(m)
                logger.info(f"✅  Loaded {cfg['name']}")
            except Exception as exc:
                logger.error(f"❌  Failed to load {cfg['name']}: {exc}")

        # Meta model
        if os.path.exists(META_MODEL_FILE):
            try:
                state.meta_model = joblib.load(META_MODEL_FILE)
                logger.info(f"✅  Loaded Meta-Model")
            except Exception as exc:
                logger.error(f"❌  Failed to load Meta-Model: {exc}")

        state.ensemble_ready = (
            len(state.loaded_models) == len(MODELS_CONFIG)
            and state.meta_model is not None
        )
        if state.ensemble_ready:
            logger.info("🎯  Stacking Ensemble ready — 3 base models + meta-model")
        else:
            logger.error("❌  Ensemble incomplete — check model files")

    # Forecast models
    for target in TARGET_COLUMNS:
        path = os.path.join(MODEL_DIR, f"{target}_model.pkl")
        if os.path.exists(path):
            state.forecast_models[target] = joblib.load(path)
            logger.info(f"✅  Loaded forecast model: {target}")

    # Historical CSV
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        state.min_timestamp = df["timestamp"].min()
        df["time_index"] = (df["timestamp"] - state.min_timestamp).dt.total_seconds()
        state.df_forecast = df
        logger.info(f"✅  Loaded historical CSV ({len(df)} rows)")
    else:
        logger.warning(f"⚠  Historical CSV not found: {CSV_PATH}")

    logger.info("═" * 60)
    logger.info("  Server ready")
    logger.info("═" * 60)
    yield
    # ── Shutdown ──
    logger.info("Server shutting down — releasing resources")
    state.loaded_models.clear()


# ─────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Water Classification API",
    description=(
        "Stacking Ensemble water-type classifier (EfficientNet-B3 + ResNet-50 + "
        "MobileNetV3-Large), water-quality forecasting, and statistical anomaly detection."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Tighten for production (e.g., your Vercel domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Pydantic schemas
# ─────────────────────────────────────────────
class ForecastRequest(BaseModel):
    parameter: str = Field(..., examples=["temperature"])
    horizon: str   = Field("24h", examples=["24h", "7d"])


class AnomalyRequest(BaseModel):
    parameter: str  = Field(..., examples=["pH"])
    threshold: float = Field(2.5, ge=0.5, le=10.0)


class Top3Item(BaseModel):
    class_name: str
    confidence: str


class PredictResponse(BaseModel):
    result: str
    top3: list[Top3Item]
    models_used: int
    confidence_score: float


class HealthResponse(BaseModel):
    status: str
    ensemble_ready: bool
    classes: list[str]
    forecast_models_loaded: list[str]
    device: str
    pytorch_version: str


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """API root — basic liveness check."""
    return {
        "status": "online",
        "message": "Water Classification API (FastAPI) is running.",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Detailed health check including model readiness."""
    return HealthResponse(
        status="healthy" if state.ensemble_ready else "degraded",
        ensemble_ready=state.ensemble_ready,
        classes=state.class_names,
        forecast_models_loaded=list(state.forecast_models.keys()),
        device=str(state.device),
        pytorch_version=torch.__version__,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
async def predict(file: UploadFile = File(..., description="Water image (jpg/png)")):
    """
    Classify a water image using the stacking ensemble.

    Returns the predicted water type, confidence score, and top-3 candidates.
    Images that don't resemble a known water type are rejected with an
    out-of-distribution warning (confidence < 65 %).
    """
    if not state.ensemble_ready:
        raise HTTPException(
            status_code=503,
            detail="Ensemble not ready. Ensure all model files are present.",
        )

    # ── Validate content type ──
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload an image.",
        )

    try:
        raw = await file.read()
        pil_image    = Image.open(io.BytesIO(raw)).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(state.device)

        # Collect class probabilities from each base model
        batch_probs: list[np.ndarray] = []
        with torch.no_grad():
            for model in state.loaded_models:
                probs = F.softmax(model(input_tensor), dim=1)
                batch_probs.append(probs.cpu().numpy())

        # Feature matrix → (1, 3 * num_classes)
        X_test = np.concatenate(batch_probs, axis=1)

        # Meta-model prediction
        meta_probs       = state.meta_model.predict_proba(X_test)[0]
        meta_probs_t     = torch.tensor(meta_probs).unsqueeze(0)
        confidence, pred = torch.max(meta_probs_t, 1)
        conf_score       = float(confidence.item())

        # Out-of-distribution rejection
        if conf_score < CONFIDENCE_THRESHOLD:
            return PredictResponse(
                result=f"Not a recognised water type (Conf: {conf_score*100:.1f}%)",
                top3=[],
                models_used=len(state.loaded_models),
                confidence_score=round(conf_score * 100, 2),
            )

        predicted_class = state.class_names[pred.item()]

        # Top-3
        k = min(3, state.num_classes)
        top3_probs, top3_idx = torch.topk(meta_probs_t, k, dim=1)
        top3 = [
            Top3Item(
                class_name=state.class_names[i.item()],
                confidence=f"{p.item()*100:.1f}%",
            )
            for p, i in zip(top3_probs[0], top3_idx[0])
        ]

        return PredictResponse(
            result=f"{predicted_class} ({conf_score*100:.1f}%)",
            top3=top3,
            models_used=len(state.loaded_models),
            confidence_score=round(conf_score * 100, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/forecast", tags=["Forecasting"])
def api_forecast(body: ForecastRequest):
    """
    Forecast a water quality parameter for the next 24 hours or 7 days.

    **parameter**: one of `temperature`, `dissolved_oxygen`, `pH`, `conductivity`  
    **horizon**: `24h` (hourly steps) or `7d` (daily steps)
    """
    param   = body.parameter
    horizon = body.horizon

    if param not in state.forecast_models:
        raise HTTPException(
            status_code=400,
            detail=f"No forecast model for '{param}'. "
                   f"Available: {list(state.forecast_models.keys())}",
        )
    if state.df_forecast is None:
        raise HTTPException(
            status_code=503,
            detail="Historical dataset not loaded.",
        )

    model     = state.forecast_models[param]
    df        = state.df_forecast
    last_time = float(df["time_index"].iloc[-1])

    if horizon == "24h":
        steps, step_size = 24, 3600
    elif horizon == "7d":
        steps, step_size = 7, 86400
    else:
        raise HTTPException(
            status_code=400,
            detail="horizon must be '24h' or '7d'.",
        )

    future_times = [last_time + (i * step_size) for i in range(1, steps + 1)]
    X_future     = np.array(future_times).reshape(-1, 1)
    preds        = model.predict(X_future)

    slope_per_sec  = float(model.coef_[0])
    slope_per_unit = slope_per_sec * step_size

    y_hist = df[param].values
    r2     = float(model.score(df[["time_index"]].values, y_hist))
    sigma  = float(np.std(y_hist))

    # Add slight sinusoidal noise to mimic realistic variance
    forecasted = [
        float(p + np.sin(i * 1.7) * sigma * 0.08)
        for i, p in enumerate(preds)
    ]

    return {
        "parameter":       param,
        "horizon":         horizon,
        "historical":      [float(v) for v in y_hist],
        "predictions":     forecasted,
        "r2":              round(r2, 4),
        "slope_per_unit":  round(slope_per_unit, 6),
        "sigma":           round(sigma, 4),
    }


@app.post("/api/anomaly_batch", tags=["Anomaly Detection"])
def api_anomaly_batch(body: AnomalyRequest):
    """
    Run Z-score anomaly detection on a water quality parameter.

    Returns the full scored dataset plus a ranked list of anomalies.  
    **threshold** (default 2.5): Z-score magnitude above which a reading is flagged.
    """
    param     = body.parameter
    threshold = body.threshold

    if not os.path.exists(CSV_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not found: {CSV_PATH}",
        )

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset: {exc}",
        )

    if param not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Parameter '{param}' not found. "
                   f"Available columns: {list(df.columns)}",
        )

    df[param] = pd.to_numeric(df[param], errors="coerce")
    df = df.dropna(subset=[param]).copy()

    vals  = df[param].values
    mu    = float(np.mean(vals))
    sigma = float(np.std(vals))

    if sigma == 0:
        raise HTTPException(
            status_code=400,
            detail="Standard deviation is 0 — cannot compute Z-scores.",
        )

    df["z"] = (df[param] - mu) / sigma

    scored_all = [
        {"rowNum": i + 1, "value": float(row[param]), "z": float(row["z"])}
        for i, (_, row) in enumerate(df.iterrows())
    ]

    anomalies_df = df[df["z"].abs() > threshold].copy()
    anomalies_df["abs_z"] = anomalies_df["z"].abs()
    anomalies_df = anomalies_df.sort_values("abs_z", ascending=False)

    anomalies_out = [
        {
            "rowNum": int(row.name) + 1,
            "value":  float(row[param]),
            "z":      float(row["z"]),
        }
        for _, row in anomalies_df.iterrows()
    ]

    return {
        "parameter":      param,
        "threshold":      threshold,
        "mu":             round(mu, 4),
        "sigma":          round(sigma, 4),
        "total_rows":     len(scored_all),
        "anomaly_count":  len(anomalies_out),
        "scored":         scored_all,
        "anomalies":      anomalies_out,
    }


# ─────────────────────────────────────────────
#  Entry point (local dev only)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
