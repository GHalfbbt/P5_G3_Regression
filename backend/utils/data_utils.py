import pandas as pd

DATA_PATH = "data/life_expectancy_data.csv"

def get_metadata():
    # Ajusta según el dataset real
    metadata = {
        "features": [
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
        ],
        "feature_names": [
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
        ],
        "target": "Life expectancy"
    }
    return metadata

def get_dataset_sample(n=5):
    try:
        df = pd.read_csv(DATA_PATH)

        # Reemplazar inf y NaN por None
        df = df.replace([float("inf"), float("-inf")], None)
        df = df.where(pd.notna(df), None)

        # Convertir a lista de diccionarios limpia
        records = df.sample(n).to_dict(orient="records")

        # Asegurarse de que todos los valores NaN -> None
        clean_records = []
        for row in records:
            clean_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            clean_records.append(clean_row)

        return {"sample": clean_records}

    except Exception as e:
        return {"sample": [], "error": str(e)}

