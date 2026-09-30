from pathlib import Path
import json
import os
from datetime import datetime, timezone

def log_path() -> Path:
    return Path(os.getenv("PREDICTION_LOG", "logs/predictions.jsonl"))

def write_prediction_log(payload: dict) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
