import pandas as pd

DATA_PATH = "data/life_expectancy_data.csv"

def get_metadata():
    metadata = {
        "feature_info": {
            "HIV/AIDS": {"type": "numeric", "min": 0, "max": 100, "default": 2.5},
            "Income composition of resources": {"type": "numeric", "min": 0, "max": 1, "default": 0.5},
            "Country": {"type": "categorical", "categories": ["Spain", "Argentina", "Brazil"], "default": "Spain"},
            "Adult Mortality": {"type": "numeric", "min": 0, "max": 500, "default": 100},
            "under-five deaths": {"type": "numeric", "min": 0, "max": 300, "default": 20},
            "BMI": {"type": "numeric", "min": 10, "max": 60, "default": 25},
            "thinness 5-9 years": {"type": "numeric", "min": 0, "max": 30, "default": 5},
            "Diphtheria": {"type": "numeric", "min": 0, "max": 100, "default": 80},
            "infant deaths": {"type": "numeric", "min": 0, "max": 300, "default": 30},
            "Schooling": {"type": "numeric", "min": 0, "max": 20, "default": 12}
        },
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