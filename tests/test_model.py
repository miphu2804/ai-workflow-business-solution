import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import model as model_module

def make_test_model(path):
    x = pd.DataFrame([
        {"country":"France","year":2026,"month":1,"marketing_spend":10000,"transactions":1000,"avg_order_value":50},
        {"country":"Germany","year":2026,"month":2,"marketing_spend":20000,"transactions":1400,"avg_order_value":65},
    ])
    y = [90000, 125000]
    pre = ColumnTransformer([("country", OneHotEncoder(handle_unknown="ignore"), ["country"])], remainder="passthrough")
    pipe = Pipeline([("preprocessor", pre), ("model", DummyRegressor(strategy="mean"))])
    pipe.fit(x, y)
    joblib.dump({"model": pipe, "countries": ["France","Germany"]}, path)

def test_single_country_prediction(isolated_env):
    make_test_model(isolated_env["model_path"])
    model_module._MODEL_CACHE = None
    result = model_module.predict("France", 2026, 5, 12000, 1100, 55)
    assert result["scope"] == "France"
    assert result["prediction"] > 0

def test_all_country_prediction(isolated_env):
    make_test_model(isolated_env["model_path"])
    model_module._MODEL_CACHE = None
    result = model_module.predict("all", 2026, 5, 12000, 1100, 55)
    assert result["scope"] == "all"
    assert set(result["by_country"]) == {"France", "Germany"}
    assert result["prediction"] == sum(result["by_country"].values())
