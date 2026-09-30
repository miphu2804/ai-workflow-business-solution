from time import perf_counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import predict
from .logging_utils import write_prediction_log
from .monitor import summarize

app = FastAPI(title="AI Workflow Business Solution", version="1.0.0")

class PredictionRequest(BaseModel):
    country: str = Field(examples=["France", "all"])
    year: int = Field(default=2026, ge=2020, le=2100)
    month: int = Field(default=10, ge=1, le=12)
    marketing_spend: float = Field(default=15000, ge=0)
    transactions: int = Field(default=1200, ge=0)
    avg_order_value: float = Field(default=60, ge=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/monitor")
def monitor():
    return summarize()

@app.post("/predict")
def predict_endpoint(req: PredictionRequest):
    started = perf_counter()
    try:
        result = predict(**req.model_dump())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = (perf_counter() - started) * 1000
    write_prediction_log({
        "country": req.country,
        "prediction": result["prediction"],
        "latency_ms": latency_ms,
    })
    return {**result, "latency_ms": round(latency_ms, 3)}
