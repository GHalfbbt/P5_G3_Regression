# client/streamlit_frontend_predictor.py
# Modern + colorful Streamlit frontend for model predictions (Esperanza de Vida)
# Run: streamlit run client/streamlit_frontend_predictor.py

import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
from typing import Dict, Any, Optional

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Esperanza de Vida — Model Playground",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Modern colorful CSS
# ---------------------------
st.markdown(
    """
<style>
/* Background gradient for the main app area */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0f1724 0%, #0b2545 40%, #07132a 100%);
    color: #e6eef8;
}

/* Sidebar gradient and styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061226 0%, #08263a 50%, #0b3a4d 100%);
    color: #f1f7ff;
    padding-top: 1rem;
}

/* Title inside sidebar */
section[data-testid="stSidebar"] .css-1d391kg { 
    color: #fff;
}

/* Button gradient in sidebar */
section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(90deg,#36d1dc,#5b86e5);
    color: white;
    border-radius: 12px;
    padding: 8px 10px;
    font-weight: 600;
    border: none;
    box-shadow: 0 6px 18px rgba(91,134,229,0.18);
}
section[data-testid="stSidebar"] .stButton>button:hover {
    transform: translateY(-2px);
}

/* Primary buttons in main area */
.stButton>button {
    background: linear-gradient(90deg,#ff6a00,#ee0979);
    color: white;
    border-radius: 12px;
    padding: 8px 12px;
    font-weight: 600;
    border: none;
    box-shadow: 0 6px 18px rgba(238,9,121,0.12);
}
.stButton>button:hover {
    transform: translateY(-2px);
}

/* Cards/metrics style */
.stMetric > div {
    background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    padding: 12px;
    border-radius: 12px;
    color: #f7fbff;
    box-shadow: 0 8px 24px rgba(2,6,23,0.6);
}

/* Headings color */
h1, h2, h3, h4, h5 {
    color: #f3f9ff;
}

/* Tables backgrounds */
.stDataFrame table {
    background: rgba(255,255,255,0.02);
}

/* Small helper tweaks */
.css-1lcbmhc { gap: .6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Utility functions (backend talk)
# ---------------------------
@st.cache_data(ttl=60)
def get_json(url: str, path: str, timeout: int = 4) -> Optional[dict]:
    try:
        r = requests.get(url.rstrip("/") + path, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def post_json(url: str, path: str, body: dict, timeout: int = 8) -> dict:
    try:
        r = requests.post(url.rstrip("/") + path, json=body, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        else:
            return {"_error": True, "status": r.status_code, "text": r.text}
    except Exception as e:
        return {"_error": True, "text": str(e)}

# ---------------------------
# Sidebar: menu and controls
# ---------------------------
with st.sidebar:
    st.title("🔬 Model Playground")
    st.markdown("Selecciona un algoritmo para ver información y realizar predicciones.")
    backend_url = st.text_input("Backend base URL", value="http://localhost:8000")
    st.caption("El backend debe exponer GET /metadata, GET /models, POST /predict. Opcional: /dataset, /model_info/{model}")
    st.markdown("---")

    model_menu = st.radio(
        "Algoritmos",
        options=[
            "Regresión Lineal",
            "Ridge / Lasso / ElasticNet",
            "Árbol de Decisión",
            "Random Forest",
            "XGBoost"
        ],
        index=0
    )

    st.markdown("---")
    compare_all = st.checkbox("Comparar todos los modelos detectados", value=False)
    show_details = st.checkbox("Mostrar coeficientes/importancias (si disponibles)", value=True)
    st.markdown("---")
    calc_button = st.button("🧾 Calcula tu esperanza de vida")

# ---------------------------
# Resolve available models
# ---------------------------
models_meta = get_json(backend_url, "/models")
if models_meta and isinstance(models_meta, dict):
    available_models = models_meta.get("models", [])
else:
    # fallback defaults
    available_models = ["linear", "ridge", "lasso", "elasticnet", "decision_tree", "random_forest", "xgboost"]

menu_map = {
    "Regresión Lineal": ["linear"],
    "Ridge / Lasso / ElasticNet": ["ridge", "lasso", "elasticnet"],
    "Árbol de Decisión": ["decision_tree"],
    "Random Forest": ["random_forest"],
    "XGBoost": ["xgboost"]
}

selected_models = menu_map.get(model_menu, ["linear"])
if compare_all:
    selected_models = available_models

# ---------------------------
# Main layout
# ---------------------------
st.header("🌍 Esperanza de Vida — Dashboard Interactivo")
st.write("Explora modelos, visualizaciones del dataset y calcula tu esperanza de vida con distintos algoritmos.")

left, right = st.columns([2, 1])

# ---------------------------
# Left column: dataset, model info, visuals
# ---------------------------
with left:
    st.subheader("Resumen del dataset")
    dataset_json = get_json(backend_url, "/dataset")  # optional endpoint
    if dataset_json:
        try:
            sample = pd.DataFrame(dataset_json.get("sample", []))
            stats = dataset_json.get("stats", {})
            target_col = dataset_json.get("target_col", None)
            if not sample.empty:
                st.markdown("**Muestra del dataset**")
                st.dataframe(sample.head(10), use_container_width=True)
            if stats:
                st.markdown("**Estadísticas resumidas**")
                st.json(stats)
        except Exception:
            st.info("No se pudo parsear /dataset. Asegúrate de que el backend retorna sample y stats correctamente.")
    else:
        meta = get_json(backend_url, "/metadata")
        if meta and meta.get("feature_names"):
            fnames = meta.get("feature_names", [])
            st.write(f"Metadata detectada con **{len(fnames)} features**.")
            st.dataframe(pd.DataFrame(meta.get("feature_info", {})).head(20), use_container_width=True)
        else:
            st.info("No se encontró /dataset ni /metadata. Pide a tu compañera que exponga esos endpoints para ver más info aquí.")

    st.markdown("---")

    # Model blocks
    for m in selected_models:
        st.subheader(f"Modelo: {m}")
        mi = get_json(backend_url, f"/model_info/{m}")
        if mi:
            # show simple metrics as cards
            metrics = mi.get("metrics", {})
            if metrics:
                cols = st.columns(len(metrics))
                for i, (k, v) in enumerate(metrics.items()):
                    try:
                        val = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                    except Exception:
                        val = str(v)
                    cols[i].metric(label=k, value=val)
            # feature importance or coef
            if show_details and mi.get("feature_importance"):
                fi = mi.get("feature_importance")
                fi_df = pd.DataFrame(list(fi.items()), columns=["feature", "importance"]).sort_values("importance", ascending=False)
                st.markdown("**Importancia de variables (top 20)**")
                fig = px.bar(fi_df.head(20), x="feature", y="importance", title=f"Importancia - {m}", color="importance", color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
            elif show_details and mi.get("coef"):
                coef = mi.get("coef")
                coef_df = pd.DataFrame(list(coef.items()), columns=["feature", "coef"]).assign(abscoef=lambda d: d.coef.abs()).sort_values("abscoef", ascending=False)
                st.markdown("**Coeficientes (top 20 por magnitud)**")
                fig = px.bar(coef_df.head(20), x="feature", y="coef", title=f"Coeficientes - {m}", color="coef", color_continuous_scale=px.colors.diverging.Picnic)
                st.plotly_chart(fig, use_container_width=True)
            # preds sample
            if mi.get("preds_sample"):
                ps = pd.DataFrame(mi.get("preds_sample"))
                if {"y_true", "y_pred"}.issubset(ps.columns):
                    st.markdown("**y_true vs y_pred (sample)**")
                    fig = px.scatter(ps, x="y_true", y="y_pred", trendline="ols", title=f"y_true vs y_pred - {m}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No hay `/model_info/{m}` disponible en el backend. Implementándolo obtendrás métricas, importancias y gráficos aquí.")

    st.markdown("---")
    st.subheader("Visualizaciones generales")
    if dataset_json and dataset_json.get("sample"):
        df = pd.DataFrame(dataset_json.get("sample"))
        tgt = dataset_json.get("target_col") or ("target" if "target" in df.columns else None)
        if tgt and tgt in df.columns:
            st.markdown(f"**Distribución de {tgt}**")
            fig = px.histogram(df, x=tgt, nbins=40, title=f"Distribución de {tgt}", color_discrete_sequence=["#ff6a00"])
            st.plotly_chart(fig, use_container_width=True)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            st.markdown("**Matriz de correlación (numéricas)**")
            fig = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Blues)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Para ver estadísticas y gráficas automáticas implementa GET /dataset que devuelva 'sample' y 'target_col'.")

# ---------------------------
# Right column: quick info & actions
# ---------------------------
with right:
    st.subheader("Acciones rápidas")
    st.markdown("- Ingresa la URL del backend en la sidebar.")
    st.markdown("- Selecciona el algoritmo y haz click en 'Calcula tu esperanza de vida'.")
    st.markdown("---")
    st.markdown("**Modelos detectados**")
    st.write(", ".join(available_models))
    st.markdown("---")
    st.info("Si el backend no responde, revisa CORS y que FastAPI esté corriendo. Este frontend espera JSONs concretos; pídeles la especificación a tu compañera si hay mismatches.")

# ---------------------------
# Prediction form (when user clicks)
# ---------------------------
# ---------------------------
# Prediction form (when user clicks)
# ---------------------------
if calc_button:
    st.markdown("---")
    st.header("🧾 Calcula tu esperanza de vida")
    st.write("Rellena el formulario con tus datos. Se enviarán al backend para obtener la predicción.")

    meta = get_json(backend_url, "/metadata")
    if meta and meta.get("features"):
        feature_info = meta["features"]  # lista de dicts [{name,type}, ...]
    else:
        feature_info = []
        st.error("No hay metadata disponible desde el backend.")

    if not feature_info:
        st.error("No hay features definidos. No puedo construir el formulario.")
    else:
        with st.form("predict_form"):
            st.markdown("**Introduce tus valores**")
            cols = st.columns(3)
            inputs = {}
            i = 0
            for f in feature_info:
                feat = f["name"]
                ftype = f.get("type", "float")
                col = cols[i % 3]

                if ftype in ("float", "int"):
                    val = col.number_input(feat, value=0.0 if ftype == "float" else 0)
                    inputs[feat] = float(val)
                elif ftype == "string":
                    inputs[feat] = col.text_input(feat, value="")
                else:
                    inputs[feat] = col.text_input(feat, value="")

                i += 1

            st.caption(f"Modelos a usar: {', '.join(selected_models)}")
            submit = st.form_submit_button("Predecir 🚀")

        # 👇 este bloque va fuera del `with st.form`
        if submit:
            results = []
            for mod in selected_models:
                body = {"model": mod, "input": inputs}
                resp = post_json(backend_url, "/predict", body)
                if resp is None or resp.get("_error"):
                    st.error(f"Error al predecir con {mod}: {resp.get('text') if isinstance(resp, dict) else 'No response'}")
                    continue
                prediction = resp.get("prediction")
                metrics = resp.get("metrics", {})
                results.append({"model": mod, "prediction": prediction, "metrics": metrics, "raw": resp})

            if results:
                dfres = pd.DataFrame([{"Modelo": r["model"], "Predicción": r["prediction"]} for r in results])
                st.success("Predicciones completadas")
                lc, rc = st.columns([1, 2])
                with lc:
                    for r in results:
                        st.metric(label=r["model"], value=str(r["prediction"]))
                with rc:
                    fig = px.bar(
                        dfres,
                        x="Modelo",
                        y="Predicción",
                        title="Comparación de predicciones",
                        color="Predicción",
                        color_continuous_scale=px.colors.sequential.OrRd
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with st.expander("Respuestas crudas (raw JSON)"):
                    st.json([r["raw"] for r in results])
                st.download_button(
                    "Descargar predicciones (CSV)",
                    dfres.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_vida.csv"
                )
