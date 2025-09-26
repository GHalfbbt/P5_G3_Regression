# 📊 P5_G3_Regression
Proyecto grupal de Machine Learning dentro del bootcamp de IA de [Factoría F5 – Web Oficial](https://factoriaf5.org/)  
Nuestro objetivo es **predecir la esperanza de vida** a partir de indicadores socioeconómicos y sanitarios, comparando distintos algoritmos de regresión.


---

## 🚀 Objetivos
- Analizar el dataset de esperanza de vida mediante un **EDA riguroso**.
- Probar distintos algoritmos de regresión (lineales y no lineales).
- Evaluar con métricas de regresión: **RMSE, MAE, R²**.
- Comparar resultados entre compañeros y seleccionar el modelo con mejor rendimiento.


---

👥 Autores

    
- [Umit Gungor](https://github.com/GungorUmit) — Data Analyst & Python Developer  
- [Johi Ortiz Vallejos](https://github.com/johiortiz) — Data Analyst & Python Developer  
- [Yeder Pimentel](https://github.com/Yedpt) — Data Analyst & Python Developer  
- [Alfonso Bermúdez Torres](https://github.com/GHalfbbt) — Data Analyst & Python Developer

---

## 📂 Estructura del repositorio
```bash
P5_G3_Regression/
│── data/ # datasets (train/test)
│── notebooks/ # notebooks de cada algoritmo evaluado
│ ├── notebook_A.ipynb
│ ├── notebook_B.ipynb
│ └── ...
│── docs/ # documentos (PDF investigación, plan de trabajo, etc.)
├── src/                      # scripts del pipeline
│   ├── __init__.py
│   ├── preprocess.py         # limpieza + escalado + encoding
│   ├── train_model.py        # entrenamiento modelos - crea *.pkl
│   ├── predict.py            # cargar modelo y predecir
│   └── app.py                # PMV con Streamlit
│
├── models/                   # modelos guardados (son generados)
│   └── xgb_model.pkl
│
│── requirements.txt # dependencias del proyecto
│── README.md # descripción del proyecto
```

---

## 👥 Organización del trabajo
- Cada compañero trabaja en su **algoritmo asignado** en un notebook propio.
- Los notebooks se suben en la carpeta `notebooks/`.
- Se evaluarán los resultados de cada modelo según las métricas acordadas.
- Finalmente se fusionará el mejor modelo en la rama `main` (o se dejarán todos como referencia).

---

## 📊 Dataset
- **Fuente:** Life Expectancy Dataset (Kaggle / WHO).  
- **Variable objetivo:** Esperanza de vida (años).  
- **Características:** gasto en salud, mortalidad infantil, factores demográficos, socioeconómicos, etc.

🔗 [Dataset Life Expectancy Kaggle](https://www.kaggle.com/code/wrecked22/life-expectancy-regression)

---

## 🛠️ Tecnologías
- Python, Jupyter Lab
- Numpy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- GitHub para control de versiones y organización

---

## ⚙️ Instalación y entorno de ejecución

### 🔧 Requisitos previos
- Python 3.9+  
- git instalado  

### 💻 Clonar repositorio
```bash
git clone https://github.com/usuario/P5_G3_Regression.git
cd P5_G3_Regression
```


🖥️ Crear entorno virtual
Windows (PowerShell)

```bash
python -m venv venv
venv\Scripts\activate
```


Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

📦 Instalar dependencias
Este proyecto incluye un archivo requirements.txt con las librerías necesarias.
Ejecuta el siguiente comando dentro del entorno virtual:
```bash
pip install -r requirements.txt
```
Contenido sugerido de requirements.txt:
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyterlab
xgboost
lightgbm
optuna

▶️ Ejecutar Jupyter Lab
```bash
jupyter lab
```


---

## ✅ Tareas principales
1. Investigación y entrega en PDF.
2. Selección y análisis del dataset.
3. Exploratory Data Analysis (EDA).
4. Implementación de modelos baseline.
5. Implementación de algoritmos avanzados (Random Forest, XGBoost, etc.).
6. Comparación de resultados y elección del mejor modelo.
7. Documentación y entrega final.

---

## 🔄 Flujo de trabajo con Git
- `main`: rama estable con el proyecto consolidado.
- `feature/algoritmo_nombre`: ramas donde cada compañero desarrolla su notebook.  
- Pull requests para fusionar a `main`.

---

## 📌 Notas
Este proyecto **no es una competición oficial de Kaggle**, pero se inspira en su dinámica.  
Cada miembro probará un algoritmo distinto y evaluaremos cuál se comporta mejor con las métricas seleccionadas.
