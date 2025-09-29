# backend/utils/model_utils.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data_utils import get_metadata
import logging

logger = logging.getLogger("model_utils")

# Directorios base (resolviendo relativo al paquete utils)
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
            logger.info("Loading model from %s", path)
            return joblib.load(path)
    raise FileNotFoundError(f"No existe el modelo: '{model_name}' en {MODELS_DIR}")


def _normalize_country_name(name: str) -> str:
    if name is None:
        return ""
    # quitar espacios al inicio/fin y reemplazar espacios por underscore (coincide con Country_<Name>)
    return str(name).strip().replace(" ", "_")


def _prepare_input(input_dict: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """
    Prepara los datos de entrada en el orden correcto.
    - Convierte Country en variables dummy si es necesario.
    - Asegura que las columnas estén alineadas con el dataset del modelo.
    """
    meta = get_metadata()

    # Selección de dataset según el modelo
    if any(x in model_name for x in ["linear", "ridge", "lasso", "elasticnet"]):
        df_ref = pd.read_csv("data/processed/features_scaled.csv")
    else:
        df_ref = pd.read_csv("data/processed/features_no_scaling.csv")

    # columnas de referencia (sin target)
    all_cols = df_ref.drop(columns=[meta["target"]], errors="ignore").columns.tolist()

    # Creamos df_input a partir del dict
    df_input = pd.DataFrame([input_dict])

    # Manejo de Country:
    if "Country" in df_input.columns:
        country_val = df_input.at[0, "Country"]
        country_norm = _normalize_country_name(country_val)

        # OneHot encoding consistente con columnas de referencia
        country_cols_ref = [c for c in all_cols if isinstance(c, str) and c.startswith("Country_")]
        if country_cols_ref:
            # asignar 1 a la columna coincidente si existe; si no existe, dejar todas 0
            matched_col = f"Country_{country_norm}"
            for col in country_cols_ref:
                df_input[col] = 1 if col == matched_col else 0
        # eliminar la columna 'Country' original antes de alinear
        df_input = df_input.drop(columns=["Country"], errors="ignore")

    # Asegurar que todas las columnas estén presentes en df_input
    for col in all_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    # Mantener orden exacto
    df_input = df_input[all_cols]

    # Convertir tipos numéricos (evitar strings en columnas numéricas)
    for c in df_input.columns:
        # intentar convertir columnas a numeric si se parecen numeric
        try:
            df_input[c] = pd.to_numeric(df_input[c], errors="ignore")
        except Exception:
            pass

    return df_input


def predict_with_model(model_name: str, input_dict: Dict[str, Any]) -> float:
    """
    Realiza una predicción garantizando el orden de columnas y usando DataFrame.
    Compatible con modelos de sklearn y Pipelines con ColumnTransformer.
    """
    model = load_model(model_name)
    X = _prepare_input(input_dict, model_name)
    y_pred = model.predict(X)
    return float(np.asarray(y_pred).ravel()[0])


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Devuelve info básica del modelo si está disponible: métricas, coeficientes o importancias.
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

    # ---------------------------
    # Selección de dataset según el modelo
    # ---------------------------
    try:
        if any(x in model_name for x in ["linear", "ridge", "lasso", "elasticnet"]):
            df = pd.read_csv("data/processed/features_scaled.csv")
        else:
            df = pd.read_csv("data/processed/features_no_scaling.csv")

        X = df.drop(columns=["Life expectancy"], errors="ignore")
        y = df["Life expectancy"]

        # Predicciones para calcular métricas
        y_pred = model.predict(X)

        info["metrics"] = {
            "MAE": float(mean_absolute_error(y, y_pred)),
            "RMSE": float(mean_squared_error(y, y_pred, squared=False)),
            "R2": float(r2_score(y, y_pred))
        }
    except Exception as e:
        info["metrics"] = {"error": str(e)}

    # ---------------------------
    # Coeficientes (modelos lineales)
    # ---------------------------
    coefs = getattr(estimator, "coef_", None)
    if coefs is not None:
        coefs = np.asarray(coefs).ravel()
        feature_names = X.columns.tolist()
        info["coefficients"] = {
            name: float(val) for name, val in zip(feature_names, coefs[: len(feature_names)])
        }

    # ---------------------------
    # Importancia de variables (árboles / ensambles)
    # ---------------------------
    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        importances = np.asarray(importances).ravel()
        feature_names = X.columns.tolist()
        info["feature_importance"] = {
            name: float(val) for name, val in zip(feature_names, importances[: len(feature_names)])
        }

    return info
