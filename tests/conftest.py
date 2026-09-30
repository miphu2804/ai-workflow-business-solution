import pytest

@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    model_path = tmp_path / "model_bundle.joblib"
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("PREDICTION_LOG", str(log_path))
    return {"model_path": model_path, "log_path": log_path}
