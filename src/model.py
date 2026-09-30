from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = "artifacts/model_bundle.joblib"
_MODEL_CACHE = None

def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

def load_model(force_reload: bool = False):
    global _MODEL_CACHE
    if _MODEL_CACHE is None or force_reload:
        path = model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run `python -m src.train` first."
            )
        _MODEL_CACHE = joblib.load(path)
    return _MODEL_CACHE

def _make_row(country: str, year: int, month: int, marketing_spend: float,
              transactions: int, avg_order_value: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "country": country,
        "year": year,
        "month": month,
        "marketing_spend": marketing_spend,
        "transactions": transactions,
        "avg_order_value": avg_order_value,
    }])

def predict(country: str, year: int, month: int, marketing_spend: float,
            transactions: int, avg_order_value: float) -> Dict[str, Any]:
    bundle = load_model()
    model = bundle["model"]
    countries = bundle["countries"]

    if country.lower() == "all":
        predictions = {}
        for c in countries:
            row = _make_row(c, year, month, marketing_spend, transactions, avg_order_value)
            predictions[c] = float(model.predict(row)[0])
        return {
            "scope": "all",
            "prediction": float(sum(predictions.values())),
            "by_country": predictions,
        }

    if country not in countries:
        raise ValueError(f"Unknown country '{country}'. Valid values: {countries} or 'all'.")

    row = _make_row(country, year, month, marketing_spend, transactions, avg_order_value)
    value = float(model.predict(row)[0])
    return {"scope": country, "prediction": value}
