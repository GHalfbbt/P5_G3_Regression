# Streamlit frontend para predicciones (con backend FastAPI)
# Archivo: streamlit_frontend_predictor.py
# Uso: streamlit run streamlit_frontend_predictor.py

import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
from typing import Dict, Any

# Config página
st.set_page_config(page_title="Model Playground - Predict", layout="wide", initial_sidebar_state="expanded")

# --- Estilos mínimos para un look más "moderno" ---
st.markdown(
    """
    <style>
    .appview-container .main .block-container{padding-top:1rem}
    .stButton>button {border-radius: 8px}
    .stMetric > div {background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding: 12px; border-radius: 10px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 Model Playground — Predicciones (Streamlit)")
st.write("Interfaz para enviar una muestra de features al backend (FastAPI) y obtener predicciones de distintos modelos.")

# Sidebar: configuración
with st.sidebar:
    st.header("Configuración")
    backend_url = st.text_input("Backend base URL", value="http://localhost:8000")
    st.caption("El backend debe exponer /metadata, /models y /predict (ej.: POST /predict)")
    st.markdown("---")
    st.subheader("Opciones")
    compare_all = st.checkbox("Comparar todos los modelos disponibles (si el backend los soporta)", value=True)
    st.markdown("---")
    st.info("Asegúrate de que el backend permite CORS para este origen si está en distinto host/puerto.")

# Utilities
@st.cache_data(ttl=60)
def fetch_metadata(url: str):
    """Intenta obtener metadata desde el backend: /metadata
    Esperado: {"feature_names": [...], "feature_info": {feat: {"type":"numeric"/"categorical","min":..,"max":..,"categories":[..]}}}
    """
    try:
        r = requests.get(url.rstrip("/") + "/metadata", timeout=4)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

@st.cache_data(ttl=60)
def fetch_models(url: str):
    try:
        r = requests.get(url.rstrip("/") + "/models", timeout=4)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def build_input_widgets(feature_info: Dict[str, Any]):
    """Construye widgets dinámicos a partir de feature_info y devuelve un dict con valores.
    feature_info example:
    {"feat1": {"type":"numeric","min":0,"max":10,"default":1.2},
     "feat2": {"type":"categorical","categories":["a","b"]}}
    """
    inputs = {}
    st.subheader("Valores de entrada")
    cols = st.columns(2)
    i = 0
    for feat, info in feature_info.items():
        col = cols[i % 2]
        with col:
            if info.get("type") == "numeric":
                vmin = info.get("min", -1e6)
                vmax = info.get("max", 1e6)
                default = info.get("default", (vmin if vmin>-1e5 else 0.0))
                step = info.get("step", None)
                if step:
                    val = col.number_input(feat, min_value=float(vmin), max_value=float(vmax), value=float(default), step=float(step))
                else:
                    val = col.number_input(feat, min_value=float(vmin), max_value=float(vmax), value=float(default))
                inputs[feat] = float(val)

            elif info.get("type") == "categorical":
                cats = info.get("categories", [])
                if len(cats) > 0:
                    val = col.selectbox(feat, options=cats)
                    inputs[feat] = val
                else:
                    val = col.text_input(feat, value="")
                    inputs[feat] = val

            elif info.get("type") == "boolean":
                val = col.checkbox(feat, value=bool(info.get("default", False)))
                inputs[feat] = bool(val)

            else:
                # fallback: texto
                val = col.text_input(feat, value=str(info.get("default", "")))
                inputs[feat] = val

        i += 1

    return inputs


def predict_one(url: str, model: str, payload: dict):
    endpoint = url.rstrip("/") + "/predict"
    body = {"model": model, "input": payload}
    try:
        r = requests.post(endpoint, json=body, timeout=6)
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"Status {r.status_code}", "text": r.text}
    except Exception as e:
        return {"error": str(e)}


# --- Obtener metadata y modelos disponibles desde backend (si existe) ---
metadata = fetch_metadata(backend_url)
models_meta = fetch_models(backend_url)

if metadata is None:
    st.warning("No se pudo obtener metadata desde el backend. Puedes introducir manualmente la especificación de features en el panel de abajo.")

# Panel para entrada manual en caso de ausencia de metadata
if metadata is None:
    st.subheader("Especificación manual de features (JSON) — alternativa")
    st.caption('Introduce JSON con la forma {"feat1": {"type":"numeric","min":0,"max":10,"default":1}, ... }')
    manual_spec = st.text_area("Feature spec JSON", height=160, value='{}')
    try:
        feature_info = json.loads(manual_spec) if manual_spec.strip() else {}
    except Exception:
        st.error("JSON inválido")
        feature_info = {}
else:
    feature_info = metadata.get("feature_info", {})

# Si no hay feature_info pero metadata contiene feature_names, creamos spec simple
if not feature_info and metadata is not None:
    fnames = metadata.get("feature_names", [])
    feature_info = {f: {"type": "numeric", "min": 0.0, "max": 1.0, "default": 0.0} for f in fnames}

# Construir inputs
if feature_info:
    input_values = build_input_widgets(feature_info)
else:
    st.info("No hay features definidos. Especifica la metadata o usa la entrada manual JSON para enviar requests.")
    input_values = {}

# Selección de modelo
st.sidebar.markdown("---")
if models_meta and isinstance(models_meta, dict):
    available_models = models_meta.get("models", [])
    if compare_all:
        selected_models = available_models
        st.sidebar.write(f"Modelos detectados: {', '.join(available_models)}")
    else:
        sel = st.sidebar.selectbox("Modelo", options=available_models)
        selected_models = [sel]
else:
    # fallback a los 4 que tenemos en el notebook: linear, ridge, lasso, elasticnet
    default_models = ["linear", "ridge", "lasso", "elasticnet"]
    sel = st.sidebar.multiselect("Modelos (manual)", options=default_models, default=["linear"]) if compare_all else st.sidebar.selectbox("Modelo", options=default_models)
    if isinstance(sel, list):
        selected_models = sel
    else:
        selected_models = [sel]

# Button para predecir
if st.button("Enviar petición de predicción 🚀"):
    if not input_values:
        st.error("No hay valores de entrada. Revisa la metadata o introduce manualmente los inputs.")
    else:
        predictions = []
        for m in selected_models:
            resp = predict_one(backend_url, m, input_values)
            # Manejo de respuesta: se espera {'prediction': value, 'model':..., 'metrics': {...}}
            if resp is None:
                st.error(f"Respuesta vacía para el modelo {m}")
                continue
            if "error" in resp:
                st.error(f"Error al pedir {m}: {resp.get('error')} - {resp.get('text', '')}")
                continue
            pred = resp.get("prediction") if "prediction" in resp else resp
            metrics = resp.get("metrics") if isinstance(resp, dict) else None
            predictions.append({"model": m, "prediction": pred, "raw": resp, "metrics": metrics})

        if predictions:
            dfpred = pd.DataFrame([{"Modelo": p["model"], "Predicción": p["prediction"]} for p in predictions])
            st.subheader("Resultados")
            c1, c2 = st.columns([1,2])
            with c1:
                for p in predictions:
                    st.metric(label=f"{p['model']} → predicción", value=str(p['prediction']))
            with c2:
                fig = px.bar(dfpred, x="Modelo", y="Predicción", title="Comparativa de predicciones por modelo")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver respuestas crudas (JSON)"):
                st.json([p['raw'] for p in predictions])

            # permitir descarga
            csv_bytes = dfpred.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar predicciones CSV", csv_bytes, file_name="predicciones.csv")

# Footer: ejemplos y especificación mínima para el backend
with st.expander("Especificación mínima esperada del backend (ejemplos)"):
    st.markdown("""
    **GET /metadata**  -> devuelve:
    ```json
    {
      "feature_names": ["age","bmi",...],
      "feature_info": {"age": {"type": "numeric", "min": 0, "max": 120, "default": 30}, ...}
    }
    ```

    **GET /models** -> devuelve:
    ```json
    {"models": ["linear","ridge","lasso","elasticnet","decision_tree","random_forest"]}
    ```

    **POST /predict** -> payload:
    ```json
    {"model": "ridge", "input": {"age": 45, "bmi": 28.3, ...}}
    ```

    Respuesta esperada (ejemplo):
    ```json
    {"model": "ridge", "prediction": 12345.67, "metrics": {"RMSE": 100.2, "R2": 0.72}}
    ```
    """)

st.markdown("---")
st.caption("Frontend creado para integrarse con un backend FastAPI. Si quieres que adapte la app a un esquema de metadata o endpoints concreto que tu compañera vaya a implementar, dímelo y lo adapto.")
