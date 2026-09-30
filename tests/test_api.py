import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import model as model_module
from src.api import app

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

def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_specific_country(isolated_env):
    make_test_model(isolated_env["model_path"])
    model_module._MODEL_CACHE = None
    client = TestClient(app)
    response = client.post("/predict", json={
        "country": "France",
        "year": 2026,
        "month": 8,
        "marketing_spend": 15000,
        "transactions": 1200,
        "avg_order_value": 60
    })
    assert response.status_code == 200
    assert response.json()["scope"] == "France"

def test_api_all_countries(isolated_env):
    make_test_model(isolated_env["model_path"])
    model_module._MODEL_CACHE = None
    client = TestClient(app)
    response = client.post("/predict", json={
        "country": "all",
        "year": 2026,
        "month": 8,
        "marketing_spend": 15000,
        "transactions": 1200,
        "avg_order_value": 60
    })
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "all"
    assert "by_country" in body
