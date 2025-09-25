"""
predict.py
----------
Carga un modelo entrenado (.pkl) y devuelve predicciones
a partir de un diccionario con valores de entrada.
"""

import joblib
import pandas as pd

# Ruta al modelo (se puede cambiar en función del seleccionado)
MODEL_PATH = "../models/xgb_model.pkl"

def predict(input_data: dict):
    """
    Realiza la predicción a partir de los datos de entrada.

    Args:
        input_data (dict): ejemplo {"Country": "Spain", "Year": 2020, "GDP": 10000}

    Returns:
        float: valor predicho (esperanza de vida en años)
    """

    # 1. Cargar pipeline entrenado (preprocesador + modelo)
    pipeline = joblib.load(MODEL_PATH)

    # 2. Convertir el input en DataFrame para compatibilidad con sklearn
    X_input = pd.DataFrame([input_data])

    # 3. Obtener la predicción
    prediction = pipeline.predict(X_input)

    return prediction[0]  # devolvemos un float
