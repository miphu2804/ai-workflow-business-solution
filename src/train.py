from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data_ingestion import generate_business_data, COUNTRIES

FEATURES = ["country", "year", "month", "marketing_spend", "transactions", "avg_order_value"]
TARGET = "revenue"

def build_pipeline(model):
    preprocessor = ColumnTransformer([
        ("country", OneHotEncoder(handle_unknown="ignore"), ["country"]),
    ], remainder="passthrough")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        "R2": float(r2_score(y_test, pred)),
    }

def train():
    data_path = Path("data/business_data.csv")
    if not data_path.exists():
        generate_business_data(str(data_path))

    df = pd.read_csv(data_path)
    train_df = df[df["year"] < 2026]
    test_df = df[df["year"] == 2026]

    x_train, y_train = train_df[FEATURES], train_df[TARGET]
    x_test, y_test = test_df[FEATURES], test_df[TARGET]

    candidates = {
        "baseline_dummy": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2
        ),
    }

    trained = {}
    metrics = {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(x_train, y_train)
        trained[name] = pipe
        metrics[name] = evaluate(pipe, x_test, y_test)

    best_name = min(
        (name for name in candidates if name != "baseline_dummy"),
        key=lambda name: metrics[name]["RMSE"]
    )
    best_model = trained[best_name]

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "countries": COUNTRIES,
        "features": FEATURES,
        "metrics": metrics,
    }, "artifacts/model_bundle.joblib")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/model_metrics.json", "w") as f:
        json.dump({"selected_model": best_name, "metrics": metrics}, f, indent=2)

    names = list(metrics)
    rmses = [metrics[n]["RMSE"] for n in names]
    plt.figure(figsize=(9, 5))
    plt.bar(names, rmses)
    plt.ylabel("RMSE (lower is better)")
    plt.title("Candidate Models vs Baseline")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("reports/model_comparison.png", dpi=160)
    plt.close()

    print(json.dumps({"selected_model": best_name, "metrics": metrics}, indent=2))

if __name__ == "__main__":
    train()
