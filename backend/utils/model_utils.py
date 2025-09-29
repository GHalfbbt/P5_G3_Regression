from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from .data_utils import get_metadata

# Directorios base
BACKEND_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_MODEL_DIRS = [
    BACKEND_DIR / "models",
    BACKEND_DIR.parent / "models",
]

MODELS_DIR = next((p for p in CANDIDATE_MODEL_DIRS if p.exists()), CANDIDATE_MODEL_DIRS[0])


def get_available_models() -> List[str]:
    """Devuelve la lista de modelos disponibles (sin extensión)."""
    if not MODELS_DIR.exists():
        return []
    return sorted(
        p.stem for p in MODELS_DIR.iterdir() if p.suffix in (".pkl", ".joblib")
    )


def load_model(model_name: str):
    """Carga un modelo por nombre, probando .pkl y .joblib."""
    for ext in (".pkl", ".joblib"):
        path = MODELS_DIR / f"{model_name}{ext}"
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"No existe el modelo: '{model_name}' en {MODELS_DIR}")


def _to_dataframe_in_feature_order(input_data: dict, feature_names: list) -> pd.DataFrame:
    """Convierte el diccionario del frontend a un DataFrame con las mismas columnas que el modelo espera"""
    metadata = get_metadata()
    countries = metadata["feature_info"]["Country"]["categories"]

    # Separamos el valor de país del resto
    country_value = input_data.pop("Country", None)

    # One-hot encoding manual
    country_features = {f"Country_{c}": 0 for c in countries}
    if country_value in countries:
        country_features[f"Country_{country_value}"] = 1

    # Combinamos todo
    full_data = {**input_data, **country_features}

    # Crear DataFrame en el orden esperado
    df = pd.DataFrame([full_data])[feature_names]

    return df



def predict_with_model(model_name: str, input_dict: Dict[str, Any]) -> float:
    model = load_model(model_name)

    # Usar directamente el DataFrame con las features "crudas"
    X = pd.DataFrame([input_dict])

    y_pred = model.predict(X)
    return float(np.asarray(y_pred).ravel()[0])




def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Devuelve info básica del modelo si está disponible: coeficientes o importancias.
    Si es un Pipeline, intenta coger el estimador final (paso 'model' o último).
    """
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        return None

    info: Dict[str, Any] = {}
    estimator = model

    # Si es un Pipeline de sklearn, toma el último paso o el paso llamado "model"
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            estimator = model.named_steps.get("model", model.steps[-1][1])
    except Exception:
        pass

    feature_names = get_metadata()["feature_names"]

    # Coeficientes (modelos lineales)
    coefs = getattr(estimator, "coef_", None)
    if coefs is not None:
        coefs = np.asarray(coefs).ravel()
        info["coefficients"] = {
            name: float(val) for name, val in zip(feature_names, coefs[: len(feature_names)])
        }

    # Importancia de variables (árboles / ensambles)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        importances = np.asarray(importances).ravel()
        info["feature_importance"] = {
            name: float(val) for name, val in zip(feature_names, importances[: len(feature_names)])
        }

    return info