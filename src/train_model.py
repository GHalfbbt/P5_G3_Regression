"""
train_model.py
--------------
Entrena distintos modelos de regresión (Linear, DecisionTree, RandomForest,
XGBoost, Ridge, Lasso, ElasticNet, KNN) a partir del dataset original.

Cada modelo se guarda como .pkl en la carpeta "models/".
"""

"""
Entrenamiento de múltiples modelos de regresión
Usando solo las 10 variables más relevantes
"""

"""
Entrenamiento de múltiples modelos de regresión
Usando solo las 10 variables más relevantes
"""

import os
import time
import joblib
import pandas as pd
from pathlib import Path

# Modelos
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import optuna


# ======================
# Configuración
# ======================
# Carpeta donde se encuentra el dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "life_expectancy_data.csv")
# Carpeta donde se guardarán los modelos
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# Variables objetivo y features
TARGET = "Life expectancy"

# Top 10 features seleccionadas
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

# ===============================
# Diccionario con todos los modelos que entrenaremos
# ===============================

MODELS = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "elasticnet": ElasticNet(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "rf_baseline": RandomForestRegressor(random_state=42),
    "rf_gridsearchcv": RandomForestRegressor(random_state=42, max_depth=10, n_estimators=200),
    "xgb_baseline": XGBRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6
    ),
    "xgb_optuna": XGBRegressor(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8
    ),
    "knn": KNeighborsRegressor(n_neighbors=3) # el mejor k encontrado k=3
    
}

def build_pipeline(model):
    """Crea un pipeline con preprocesamiento + modelo"""
    numeric_features = [f for f in TOP_FEATURES if f != "Country"]
    categorical_features = ["Country"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline

def train_rf_gridsearch(X, y):
    """Entrena RandomForest con GridSearchCV sobre un pipeline"""
    rf = RandomForestRegressor(random_state=42)

    # Pipeline con preprocesador
    pipeline = build_pipeline(rf)

    # Definir la grid de hiperparámetros SOLO para el paso "model" del pipeline
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X, y)
    return grid.best_estimator_  # esto devuelve pipeline completo


# ======================
# Cargar dataset
# ======================
def load_data():
    """Carga y limpia los datos, dejando solo las top_features y la variable objetivo"""
    df = pd.read_csv(DATA_PATH)

    # 1. Limpiar nombres de columnas (quitar espacios extra)
    df.columns = df.columns.str.strip()

    # 2. Limpiar valores string (ej. ' Country ' → 'Country')
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 3. Detectar columna objetivo (robusto, por si hay espacio o guion bajo)
    target_candidates = [c for c in df.columns if "Life" in c and "expectancy" in c]
    if not target_candidates:
        raise ValueError("No se encontró la columna de Life Expectancy en el dataset")
    target_col = target_candidates[0]

    # 4. Eliminar filas con NaN en la variable objetivo
    df = df.dropna(subset=[target_col])

    # 5. Eliminar filas con NaN en las top_features seleccionadas
    df = df.dropna(subset=TOP_FEATURES)

    # 6. Dividir en X e y
    X = df[TOP_FEATURES]
    y = df[target_col]

    return X, y

# ======================
# Entrenamiento de modelos
# ======================
def train_and_save_models():
    X, y = load_data()

    for name, model in MODELS.items():
        print(f"\n🚀 Entrenando {name}...")
        start_time = time.time()

        if name == "rf_gridsearchcv":
            pipeline = train_rf_gridsearch(X, y)
        else:
            pipeline = build_pipeline(model)
            pipeline.fit(X, y)

        elapsed = time.time() - start_time
        joblib.dump(pipeline, os.path.join(MODELS_DIR, f"{name}.pkl"))
        print(f"✅ Modelo {name} guardado ({elapsed:.2f} segundos)")



# ======================
# Main
# ======================
if __name__ == "__main__":
    train_and_save_models()

