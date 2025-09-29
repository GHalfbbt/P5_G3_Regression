# backend/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.model_utils import (
    get_available_models,
    get_model_info,
    predict_with_model,
)
from utils.data_utils import get_metadata, get_dataset_sample
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

app = FastAPI(
    title="Life Expectancy Predictor API",
    description="API para exponer modelos de predicción de esperanza de vida",
    version="1.0.0",
)

# ---------------------------
# CORS
# ---------------------------
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/metadata")
def metadata():
    """Devuelve la metadata del dataset."""
    return get_metadata()


@app.get("/models")
def models():
    """Lista los modelos disponibles en el backend."""
    return {"models": get_available_models()}


@app.get("/dataset")
def dataset():
    """Muestra una muestra limpia del dataset (para exploración)."""
    return get_dataset_sample()


@app.get("/model_info/{model}")
def model_info(model: str):
    """
    Devuelve información básica del modelo:
    - coeficientes o importancia de variables
    - métricas si están disponibles
    """
    info = get_model_info(model)
    if info is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return info


# ---------------------------
# Predict Endpoint
# ---------------------------

class PredictRequest(BaseModel):
    model: str
    input: dict


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Hace una predicción con el modelo elegido.
    """
    logger.info("Predict request: model=%s input_keys=%s", req.model, list(req.input.keys()))
    try:
        pred = predict_with_model(req.model, req.input)
        # devolver SIEMPRE prediction como float
        return {
            "prediction": float(pred) if not isinstance(pred, (list, tuple)) else float(pred[0]),
            "model": req.model,
            "input_received": req.input,
        }
    except FileNotFoundError as e:
        logger.error("Model not found: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error during prediction: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
