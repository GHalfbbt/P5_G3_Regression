import pandas as pd

DATA_PATH = "data/dataset.csv"

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
        return {"sample": df.sample(n).to_dict(orient="records")}
    except Exception:
        return {"sample": []}