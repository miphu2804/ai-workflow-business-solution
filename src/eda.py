from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from .data_ingestion import generate_business_data

def run_eda():
    data_path = Path("data/business_data.csv")
    if not data_path.exists():
        generate_business_data(str(data_path))

    df = pd.read_csv(data_path)
    Path("reports").mkdir(exist_ok=True)

    by_country = df.groupby("country")["revenue"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    by_country.plot(kind="bar")
    plt.ylabel("Average revenue")
    plt.title("Average Revenue by Country")
    plt.tight_layout()
    plt.savefig("reports/eda_revenue_by_country.png", dpi=160)
    plt.close()

    by_month = df.groupby("month")["revenue"].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(by_month.index, by_month.values, marker="o")
    plt.xlabel("Month")
    plt.ylabel("Average revenue")
    plt.title("Monthly Revenue Pattern")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/eda_monthly_revenue.png", dpi=160)
    plt.close()

    corr = df[["marketing_spend", "transactions", "avg_order_value", "revenue"]].corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Correlation")
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig("reports/eda_correlation.png", dpi=160)
    plt.close()

    print(df.describe(include="all").to_string())

if __name__ == "__main__":
    run_eda()
