# client/streamlit_frontend_predictor.py
# Modern + colorful Streamlit frontend for model predictions (Esperanza de Vida)
# Run: streamlit run client/streamlit_frontend_predictor.py

import streamlit as st
import requests
import pandas as pd
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
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0f1724 0%, #0b2545 40%, #07132a 100%);
    color: #e6eef8;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061226 0%, #08263a 50%, #0b3a4d 100%);
    color: #f1f7ff;
    padding-top: 1rem;
}
section[data-testid="stSidebar"] .css-1d391kg { 
    color: #fff;
}
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
.stMetric > div {
    background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    padding: 12px;
    border-radius: 12px;
    color: #f7fbff;
    box-shadow: 0 8px 24px rgba(2,6,23,0.6);
}
h1, h2, h3, h4, h5 {
    color: #f3f9ff;
}
.stDataFrame table {
    background: rgba(255,255,255,0.02);
}
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
        else:
            return None
    except Exception:
        return None

def post_json(url: str, path: str, body: dict, timeout: int = 8) -> dict:
    try:
        r = requests.post(url.rstrip("/") + path, json=body, timeout=timeout)
        # Try to parse JSON even on non-200 to show useful debug info
        try:
            payload = r.json()
        except Exception:
            payload = {"_text": r.text, "_status": r.status_code}
        if r.status_code == 200:
            return payload
        else:
            # include status and text for debugging
            return {"_error": True, "status": r.status_code, "text": payload}
    except Exception as e:
        return {"_error": True, "text": str(e)}

# ---------------------------
# Sidebar: menu and controls
# ---------------------------
with st.sidebar:
    st.title("🔬 Model Playground")
    st.markdown("Selecciona un modelo para ver información y realizar predicciones.")
    backend_url = st.text_input("Backend base URL", value="http://localhost:8000")
    st.caption("El backend debe exponer GET /metadata, GET /models, POST /predict. Opcional: /dataset, /model_info/{model}")
    st.markdown("---")

    # Modelos dinámicos desde backend
    models_meta = get_json(backend_url, "/models")
    if models_meta and isinstance(models_meta, dict):
        available_models = models_meta.get("models", [])
    else:
        available_models = []

    if not available_models:
        st.warning("⚠️ No se detectaron modelos en el backend.")
        available_models = ["linear"]

    selected_models = st.multiselect(
        "Selecciona modelos",
        options=available_models,
        default=[available_models[0]] if available_models else []
    )

    st.markdown("---")
    show_details = st.checkbox("Mostrar coeficientes/importancias (si disponibles)", value=True)
    calc_button = st.button("🧾 Calcula tu esperanza de vida")

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
                fi = mi["feature_importance"]
                fi_df = pd.DataFrame(list(fi.items()), columns=["feature", "importance"]).sort_values("importance", ascending=False)
                st.markdown("**Importancia de variables (top 20)**")
                fig = px.bar(fi_df.head(20), x="feature", y="importance", title=f"Importancia - {m}", color="importance", color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
            elif show_details and mi.get("coefficients"):
                coef = mi["coefficients"]
                coef_df = pd.DataFrame(list(coef.items()), columns=["feature", "coef"]).assign(abscoef=lambda d: d.coef.abs()).sort_values("abscoef", ascending=False)
                st.markdown("**Coeficientes (top 20 por magnitud)**")
                fig = px.bar(coef_df.head(20), x="feature", y="coef", title=f"Coeficientes - {m}", color="coef", color_continuous_scale=px.colors.diverging.Picnic)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No hay `/model_info/{m}` disponible en el backend. Implementándolo obtendrás métricas, importancias y gráficos aquí.")

    st.markdown("---")
    st.subheader("Visualizaciones generales")
    if dataset_json and dataset_json.get("sample"):
        df = pd.DataFrame(dataset_json.get("sample"))
        num_cols = df.select_dtypes(include="number").columns.tolist()

        for m in selected_models:
            st.markdown(f"### Modelo: {m}")
            tgt = dataset_json.get("target_col") or ("Life expectancy" if "Life expectancy" in df.columns else None)
            if tgt and tgt in df.columns:
                st.markdown(f"**Distribución de {tgt} para {m}**")
                fig = px.histogram(df, x=tgt, nbins=40, title=f"Distribución de {tgt} — {m}", color_discrete_sequence=["#ff6a00"])
                st.plotly_chart(fig, use_container_width=True)

            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                st.markdown(f"**Matriz de correlación — {m}**")
                fig = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Blues, title=f"Correlación de variables — {m}")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Para ver estadísticas y gráficas automáticas implementa GET /dataset que devuelva 'sample' y 'target_col'.")

# ---------------------------
# Right column: quick info & actions
# ---------------------------
with right:
    st.subheader("Acciones rápidas")
    st.markdown("- Ingresa la URL del backend en la sidebar.")
    st.markdown("- Selecciona los modelos y haz click en 'Calcula tu esperanza de vida'.")
    st.markdown("---")
    st.markdown("**Modelos detectados**")
    st.write(", ".join(available_models))
    st.markdown("---")
    st.info("Si el backend no responde, revisa CORS y que FastAPI esté corriendo. Este frontend espera JSONs concretos.")

# ---------------------------
# Prediction form (when user clicks)
# ---------------------------
if calc_button:
    st.markdown("---")
    st.header("🧾 Calcula tu esperanza de vida")
    st.write("Rellena el formulario con tus datos. Se enviarán al backend para obtener la predicción.")

    # Pedimos metadata al backend (pero forzamos las 10 variables que quieres usar)
    meta = get_json(backend_url, "/metadata")
    if meta and meta.get("features"):
        feature_info = meta["features"]  # lista de dicts [{name,type}, ...]
    else:
        feature_info = []
        st.error("No hay metadata disponible desde el backend.")

    if not feature_info:
        st.error("No hay features definidos. No puedo construir el formulario.")
    else:
        # ---------------------------------------------------------
        # Lista fija de 10 features (coincide con backend/utils/data_utils.py)
        # ---------------------------------------------------------
        desired_features = [
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

        # Crear un map name->type a partir de feature_info recibida (por si hay tipos)
        type_map = {f.get("name"): f.get("type", "float") for f in feature_info}

        # Construir lista final de features a mostrar, respetando desired_features y tipos conocidos
        form_features = []
        for fname in desired_features:
            ftype = type_map.get(fname, "float" if fname != "Country" else "string")
            form_features.append({"name": fname, "type": ftype})

        # ---------------------------
        # FORM: un solo selectbox para Country + inputs para las demás 9 features
        # ---------------------------
        with st.form("predict_form"):
            st.markdown("**Introduce tus valores**")
            cols = st.columns(3)
            inputs: Dict[str, Any] = {}
            i = 0

            # Intentamos extraer lista de países desde /dataset
            dataset = get_json(backend_url, "/dataset")
            df_tmp = pd.DataFrame(dataset["sample"]) if dataset and dataset.get("sample") else pd.DataFrame()

            # 1) si existe 'Country' en sample -> usamos sus valores
            countries = []
            if "Country" in df_tmp.columns:
                try:
                    countries = sorted([c for c in df_tmp["Country"].dropna().unique().tolist() if c is not None])
                except Exception:
                    countries = []

            # 2) si no hay 'Country', pedimos /metadata y buscamos feature_names tipo Country_*
            if not countries:
                meta_for_countries = get_json(backend_url, "/metadata") or {}
                feature_names = meta_for_countries.get("feature_names", []) or []
                country_dummy_cols = [c for c in feature_names if isinstance(c, str) and c.startswith("Country_")]
                if country_dummy_cols:
                    countries = [c.replace("Country_", "") for c in country_dummy_cols]
                    countries = sorted(countries)

            # fallback final
            if not countries:
                countries = ["Unknown"]

            # Colocamos el selectbox de Country en la primera columna (único desplegable)
            country_selected = None
            if any(f["name"].lower() == "country" for f in form_features):
                country_selected = cols[0].selectbox("Country", countries)

            # Construimos los inputs para el resto de features (omitiendo Country_* dummies)
            for f in form_features:
                feat = f["name"]
                ftype = f.get("type", "float")

                if feat.lower() == "country":
                    inputs["Country"] = country_selected if country_selected is not None else countries[0]
                    continue

                col = cols[i % 3]

                # Números (int / float)
                if str(ftype).lower() in ("float", "int", "number"):
                    # heurística simple para enteros
                    if str(ftype).lower() == "int" or any(x in feat.lower() for x in ("death", "infant", "under-five", "adult mortality")):
                        val = col.number_input(feat, value=0, step=1)
                        inputs[feat] = int(val)
                    else:
                        val = col.number_input(feat, value=0.0, format="%.2f")
                        inputs[feat] = float(val)
                else:
                    # texto / string
                    inputs[feat] = col.text_input(feat, value="")

                i += 1

            st.caption(f"Modelos a usar: {', '.join(selected_models)}")
            submit = st.form_submit_button("Predecir 🚀")

        # ---------------------------
        # Envío y manejo de respuesta
        # ---------------------------
        if submit:
            # Validación mínima
            missing = [f["name"] for f in form_features if f["name"] not in inputs]
            if missing:
                st.error(f"Faltan valores: {missing}")
            else:
                results = []
                for mod in selected_models:
                    body = {"model": mod, "input": inputs}
                    resp = post_json(backend_url, "/predict", body)
                    # manejo de errores explícito y mostrar respuesta cruda para debug
                    if resp is None or resp.get("_error"):
                        err_text = ""
                        if isinstance(resp, dict):
                            err_text = resp.get("text") or str(resp)
                        else:
                            err_text = "No response from backend"
                        st.error(f"Error al predecir con {mod}: {err_text}")
                        with st.expander(f"Respuesta cruda de /predict para {mod}"):
                            st.write(resp)
                        continue

                    prediction = resp.get("prediction")
                    # show raw when prediction missing
                    if prediction is None:
                        st.warning(f"No se recibió 'prediction' desde el backend para {mod}")
                        with st.expander(f"Respuesta cruda de /predict para {mod}"):
                            st.write(resp)
                        continue

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
