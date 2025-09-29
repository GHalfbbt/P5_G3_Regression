import pandas as pd

DATA_PATH = "data/life_expectancy_data.csv"

# Definir las features que realmente quieres usar en el modelo
TOP_FEATURES = [
    "HIV/AIDS",
    "Income composition of resources",
    "Country",
    "Adult Mortality",
    "under-five deaths",
    "BMI",
    "thinness 5-9 years",
    "Diphtheria",
    "infant deaths",
    "Schooling"
]

def load_data():
    """Carga y limpia el dataset crudo, dejando solo las top_features y la variable objetivo"""
    df = pd.read_csv(DATA_PATH)

    # 1. Limpiar nombres de columnas (quitar espacios extra)
    df.columns = df.columns.str.strip()

    # 2. Limpiar valores string en columnas categóricas
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 3. Detectar columna objetivo (robusto: soporta 'Life expectancy ' o variantes)
    target_candidates = [c for c in df.columns if "Life" in c and "expectancy" in c]
    if not target_candidates:
        raise ValueError("No se encontró la columna de Life Expectancy en el dataset")
    target_col = target_candidates[0]

    # 4. Eliminar filas con NaN en target y en las features seleccionadas
    df = df.dropna(subset=[target_col])
    df = df.dropna(subset=TOP_FEATURES)

    # 5. Dividir en X e y
    X = df[TOP_FEATURES]
    y = df[target_col]

    return X, y, target_col


def get_metadata():
    """Devuelve la metadata para que el frontend genere el formulario"""
    X, _, target_col = load_data()

    # Obtener lista de países únicos del dataset
    countries = sorted(X["Country"].dropna().unique().tolist())

    metadata = {
        "feature_info": {
            "HIV/AIDS": {"type": "numeric", "min": float(X["HIV/AIDS"].min()), "max": float(X["HIV/AIDS"].max()), "default": float(X["HIV/AIDS"].median())},
            "Income composition of resources": {"type": "numeric", "min": float(X["Income composition of resources"].min()), "max": float(X["Income composition of resources"].max()), "default": float(X["Income composition of resources"].median())},
            "Country": {"type": "categorical", "categories": countries, "default": countries[0] if countries else None},
            "Adult Mortality": {"type": "numeric", "min": float(X["Adult Mortality"].min()), "max": float(X["Adult Mortality"].max()), "default": float(X["Adult Mortality"].median())},
            "under-five deaths": {"type": "numeric", "min": float(X["under-five deaths"].min()), "max": float(X["under-five deaths"].max()), "default": float(X["under-five deaths"].median())},
            "BMI": {"type": "numeric", "min": float(X["BMI"].min()), "max": float(X["BMI"].max()), "default": float(X["BMI"].median())},
            "thinness 5-9 years": {"type": "numeric", "min": float(X["thinness 5-9 years"].min()), "max": float(X["thinness 5-9 years"].max()), "default": float(X["thinness 5-9 years"].median())},
            "Diphtheria": {"type": "numeric", "min": float(X["Diphtheria"].min()), "max": float(X["Diphtheria"].max()), "default": float(X["Diphtheria"].median())},
            "infant deaths": {"type": "numeric", "min": float(X["infant deaths"].min()), "max": float(X["infant deaths"].max()), "default": float(X["infant deaths"].median())},
            "Schooling": {"type": "numeric", "min": float(X["Schooling"].min()), "max": float(X["Schooling"].max()), "default": float(X["Schooling"].median())},
        },
        "feature_names": TOP_FEATURES,
        "target": target_col
    }
    return metadata



def get_dataset_sample(n=5):
    """Devuelve una muestra del dataset limpio (sin NaNs) en JSON"""
    try:
        X, y, target_col = load_data()
        df = X.copy()
        df[target_col] = y
        return {"sample": df.sample(n).to_dict(orient="records")}
    except Exception as e:
        return {"sample": [], "error": str(e)}
