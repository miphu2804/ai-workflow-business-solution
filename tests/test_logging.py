import json
from src.logging_utils import write_prediction_log

def test_logging_isolated_from_production(isolated_env):
    write_prediction_log({"country": "France", "prediction": 123.4, "latency_ms": 4.2})
    assert isolated_env["log_path"].exists()
    event = json.loads(isolated_env["log_path"].read_text().strip())
    assert event["country"] == "France"
    assert event["prediction"] == 123.4
    assert "timestamp" in event
