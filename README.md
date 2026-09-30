# AI Workflow Business Solution

A compact end-to-end machine-learning project designed to satisfy the peer-review criteria for **Use the AI workflow to deploy a business solution**.

## Business problem

The service predicts monthly revenue for a selected country, or for all supported countries combined. It demonstrates the complete workflow:

**data ingestion → EDA → model comparison → model artifact → API → logging/monitoring → unit tests → Docker**

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the complete workflow

```bash
python -m src.data_ingestion
python -m src.eda
python -m src.train
```

This creates:

- `data/business_data.csv`
- `reports/eda_revenue_by_country.png`
- `reports/eda_monthly_revenue.png`
- `reports/eda_correlation.png`
- `reports/model_comparison.png`
- `reports/model_metrics.json`
- `artifacts/model_bundle.joblib`

## 3. Run all tests with one command

```bash
python run_tests.py
```

The unit tests cover API, model, logging, and test read/write isolation using temporary paths.

## 4. Run the API

```bash
uvicorn src.api:app --reload
```

Open interactive documentation at `http://127.0.0.1:8000/docs`.

### Specific country

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"France","year":2026,"month":10,"marketing_spend":15000,"transactions":1200,"avg_order_value":60}'
```

### All countries combined

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"all","year":2026,"month":10,"marketing_spend":15000,"transactions":1200,"avg_order_value":60}'
```

## 5. Monitoring

Each prediction writes a JSON line to `logs/predictions.jsonl`, including request scope, prediction, timestamp, and latency.

```bash
python -m src.monitor
```

or call `GET /monitor`. Metrics include request count, average latency, and p95 latency.

## 6. Docker

```bash
docker build -t ai-workflow-solution .
docker run --rm -p 8000:8000 ai-workflow-solution
```

## Peer-review checklist

| Criterion | Evidence |
|---|---|
| Unit tests for API | `tests/test_api.py` |
| Unit tests for model | `tests/test_model.py` |
| Unit tests for logging | `tests/test_logging.py` |
| Run all tests with one script | `python run_tests.py` |
| Performance monitoring | `src/monitor.py`, `/monitor` |
| Test isolation | `tests/conftest.py` uses temporary model/log paths |
| API predicts one country and all countries | `POST /predict` |
| Automated data ingestion | `src/data_ingestion.py` |
| Multiple models compared | Dummy, Linear Regression, Gradient Boosting, Random Forest |
| EDA visualizations | `src/eda.py` + generated `reports/eda_*.png` |
| Dockerized | `Dockerfile` |
| Visualization vs baseline | generated `reports/model_comparison.png` |

## Submission note

The included dataset is synthetic so the project is fully reproducible.