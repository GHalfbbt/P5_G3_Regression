"""
preprocess.py
-------------
Define y devuelve el pipeline de preprocesamiento:
- Imputación de valores faltantes
- Escalado de variables numéricas
- Codificación one-hot de variables categóricas
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_preprocessor(numeric_features, categorical_features):
    """
    Crea un pipeline de preprocesamiento para numéricas y categóricas.
    
    Args:
        numeric_features (list): lista de columnas numéricas
        categorical_features (list): lista de columnas categóricas

    Returns:
        ColumnTransformer: pipeline de preprocesado
    """

    # Para variables numéricas → imputamos con la mediana y escalamos
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Para variables categóricas → imputamos con la moda y aplicamos OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Unimos ambos pipelines en un solo preprocesador
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor
