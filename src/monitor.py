from pathlib import Path
import json
import os
import statistics

def summarize(log_file: str | None = None) -> dict:
    path = Path(log_file or os.getenv("PREDICTION_LOG", "logs/predictions.jsonl"))
    if not path.exists():
        return {"requests": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0.0}

    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    latencies = sorted(float(e["latency_ms"]) for e in events if "latency_ms" in e)
    if not latencies:
        return {"requests": len(events), "avg_latency_ms": 0.0, "p95_latency_ms": 0.0}

    idx = max(0, min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1)))))
    return {
        "requests": len(events),
        "avg_latency_ms": round(statistics.mean(latencies), 3),
        "p95_latency_ms": round(latencies[idx], 3),
    }

if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
