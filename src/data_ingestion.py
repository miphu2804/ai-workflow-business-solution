from pathlib import Path
import numpy as np
import pandas as pd

COUNTRIES = ["United Kingdom", "France", "Germany", "Spain", "Italy"]

def generate_business_data(output_path: str = "data/business_data.csv", seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible business dataset and save it for automated ingestion."""
    rng = np.random.default_rng(seed)
    rows = []
    country_effect = {
        "United Kingdom": 22000,
        "France": 17000,
        "Germany": 20000,
        "Spain": 12000,
        "Italy": 14000,
    }

    for year in [2024, 2025, 2026]:
        for month in range(1, 13):
            seasonality = 7000 * np.sin((month - 1) / 12 * 2 * np.pi) + (13000 if month == 12 else 0)
            for country in COUNTRIES:
                marketing_spend = float(rng.uniform(5000, 30000))
                transactions = int(rng.integers(600, 2500))
                avg_order_value = float(rng.uniform(25, 120))
                noise = float(rng.normal(0, 4500))
                revenue = (
                    country_effect[country]
                    + 1.55 * marketing_spend
                    + 0.58 * transactions * avg_order_value
                    + seasonality
                    + noise
                )
                rows.append({
                    "year": year,
                    "month": month,
                    "country": country,
                    "marketing_spend": round(marketing_spend, 2),
                    "transactions": transactions,
                    "avg_order_value": round(avg_order_value, 2),
                    "revenue": round(max(revenue, 0), 2),
                })

    df = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    df = generate_business_data()
    print(f"Ingested {len(df)} rows -> data/business_data.csv")
