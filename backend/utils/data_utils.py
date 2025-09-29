# backend/utils/data_utils.py
import pandas as pd
from pathlib import Path
from typing import Dict, Any

DATA_DIR = Path("data/processed")

FILES = {
    "scaled": DATA_DIR / "features_scaled.csv",
    "no_scaling": DATA_DIR / "features_no_scaling.csv",
}

TARGET_COL = "Life expectancy"

# ---- Metadata fija con las 10 features que quieres exponer ----
DESIRED_FEATURES = [
   {"name": "HIV/AIDS", "type": "float"},
   {"name": "Income composition of resources", "type": "float"},
   {"name": "Country", "type": "string"},
   {"name": "Adult Mortality", "type": "int"},
   {"name": "under-five deaths", "type": "int"},
   {"name": "BMI", "type": "float"},
   {"name": "thinness 5-9 years", "type": "float"},
   {"name": "Diphtheria", "type": "float"},
   {"name": "infant deaths", "type": "int"},
   {"name": "Schooling", "type": "float"}
]

DESIRED_FEATURE_NAMES = [f["name"] for f in DESIRED_FEATURES]

def get_metadata(dataset_type: str = "no_scaling") -> Dict[str, Any]:
    """
    Devuelve metadata fija con las 10 features deseadas (evita exponer Country_*).
    """
    return {
        "features": DESIRED_FEATURES,
        "feature_names": DESIRED_FEATURE_NAMES,
        "target": TARGET_COL,
    }


def _recover_country_from_dummies_row(row: pd.Series) -> str:
    """
    Dado un row con columnas Country_XYZ (0/1), devuelve el nombre del país (XYZ)
    o None si no se detecta ninguno.
    """
    for col in row.index:
        if isinstance(col, str) and col.startswith("Country_"):
            try:
                val = row[col]
                # consideramos 1, True, '1' como positivo
                if pd.notna(val) and int(val) == 1:
                    return col.replace("Country_", "")
            except Exception:
                # skip conversion errors
                continue
    return None


def get_dataset_sample(dataset_type: str = "no_scaling", n: int = 5) -> Dict[str, Any]:
    """
    Devuelve una muestra de datos procesados (scaled o no_scaling).
    Si el CSV tiene dummies Country_X, construye una columna 'Country' detectando
    el dummy activo por fila para que el frontend pueda poblar el selectbox.
    """
    try:
        path = FILES.get(dataset_type)
        if path is None or not path.exists():
            return {"sample": [], "error": f"Archivo no encontrado para {dataset_type}"}

        df = pd.read_csv(path)

        # Reemplazar inf y NaN por None para serializar a JSON
        df = df.replace([float("inf"), float("-inf")], None)
        df = df.where(pd.notna(df), None)

        # Si 'Country' no existe, intentar reconstruirla a partir de dummies 'Country_...'
        if "Country" not in df.columns:
            country_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Country_")]
            if country_cols:
                # aplicar fila por fila para recuperar el país
                df["Country"] = df[country_cols].apply(_recover_country_from_dummies_row, axis=1)
                # si quedó todo None, dejamos None (el frontend hará fallback)
        
        records = df.sample(min(n, len(df))).to_dict(orient="records")

        return {
            "sample": records,
            "target_col": TARGET_COL if TARGET_COL in df.columns else None,
        }

    except Exception as e:
        return {"sample": [], "error": str(e)}
