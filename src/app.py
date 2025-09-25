"""
app.py
------
Dashboard en Streamlit para predecir esperanza de vida
usando distintos modelos entrenados.


Aplicación Streamlit para que el usuario pueda:
- Introducir variables de entrada
- Seleccionar un modelo (más adelante)
- Obtener la predicción de esperanza de vida
"""

"""
Dashboard en Streamlit para predicción de esperanza de vida
Usando los modelos entrenados con las 10 variables top
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ======================
# Configuración
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "life_expectancy_data.csv"

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

st.set_page_config(page_title="Esperanza de Vida", page_icon="🌍", layout="wide")

# ======================
# Cargar dataset
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df

df = load_data()
countries = sorted(df["Country"].dropna().unique())

# ======================
# Sidebar
# ======================
st.sidebar.header("⚙️ Configuración")
algorithms = [
    "linear", "decision_tree",
    "rf_baseline", "rf_gridsearchcv",
    "xgb_baseline", "xgb_optuna",
    "ridge", "lasso", "elasticnet",
    "knn"
]
algorithm = st.sidebar.selectbox("🤖 Modelo de regresión:", algorithms)

# ======================
# Header
# ======================
st.markdown("<h1 style='text-align:center; font-size:3em;'>🌍 Predicción de Esperanza de Vida</h1>", unsafe_allow_html=True)
st.markdown("---")

# ======================
# Inputs principales
# ======================
st.subheader("📥 Variables de Entrada")
col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("🌐 País", countries, index=countries.index("Spain") if "Spain" in countries else 0)
    hiv = st.number_input("🧬 HIV/AIDS (muertes por 1000 habitantes)", min_value=0.0, value=0.1)
    income = st.number_input("💵 Índice de ingresos (0–1)", min_value=0.0, max_value=1.0, value=0.7)
    adult_mortality = st.number_input("⚰️ Mortalidad adulta", min_value=0.0, value=150.0)
    under5 = st.number_input("👶 Muertes menores de 5 años", min_value=0.0, value=20.0)

with col2:
    bmi = st.number_input("⚖️ BMI medio", min_value=0.0, value=22.0)
    thinness = st.number_input("📉 Thinness 5–9 años (%)", min_value=0.0, value=5.0)
    diphtheria = st.number_input("💉 Cobertura Diphtheria (%)", min_value=0.0, max_value=100.0, value=85.0)
    infant = st.number_input("🍼 Muertes infantiles (<1 año)", min_value=0.0, value=15.0)
    schooling = st.number_input("🎓 Escolarización (años)", min_value=0.0, value=12.0)

# Construir diccionario de entrada
input_data = {
    "Country": country,
    "HIV/AIDS": hiv,
    "Income composition of resources": income,
    "Adult Mortality": adult_mortality,
    "under-five deaths": under5,
    "BMI": bmi,
    "thinness 5-9 years": thinness,
    "Diphtheria": diphtheria,
    "infant deaths": infant,
    "Schooling": schooling
}

# ======================
# Predicción
# ======================
if st.button("🔮 Predecir Esperanza de Vida"):
    try:
        model_path = MODEL_DIR / f"{algorithm}.pkl"
        model = joblib.load(model_path)

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success(f"🌟 Predicción con {algorithm}: {prediction:.2f} años")

    except FileNotFoundError:
        st.error(f"❌ El modelo {algorithm} aún no está entrenado. Ejecuta train_model.py primero.")
