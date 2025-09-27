from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.model_utils import get_available_models, load_model, get_model_info, predict_with_model
from utils.data_utils import get_metadata, get_dataset_sample

app = FastAPI()

# Endpoint: /metadata
@app.get("/metadata")
def metadata():
    return get_metadata()

# Endpoint: /models
@app.get("/models")
def models():
    return {"models": get_available_models()}

# Endpoint: /dataset
@app.get("/dataset")
def dataset():
    return get_dataset_sample()

# Endpoint: /model_info/{model}
@app.get("/model_info/{model}")
def model_info(model: str):
    info = get_model_info(model)
    if info is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return info

# Modelo para la petición de predicción
class PredictRequest(BaseModel):
    model: str
    input: dict

# Endpoint: /predict
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        pred = predict_with_model(req.model, req.input)
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))