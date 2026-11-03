
"""Rebuild the monthly modeling table from the supplied raw CSV files.

The original random-data generator was not included with the supplied CSVs. This
script deliberately preserves the raw data rather than inventing replacement
customers or transactions.
"""
from pathlib import Path
import argparse
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOMERS_PATH = PROJECT_ROOT / "data" / "raw" / "customers.csv"
TRANSACTIONS_PATH = PROJECT_ROOT / "data" / "raw" / "retail_transactions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "customer_monthly_ml.csv"


def _round_money(value: float) -> float:
    """Use ordinary financial half-up rounding rather than binary tie rounding."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build_monthly_table(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    snapshots = pd.date_range("2025-04-01", "2026-03-01", freq="MS")
    rows = []
    transactions = transactions.sort_values("purchase_date")
    for snapshot in snapshots:
        history_start = snapshot - pd.Timedelta(days=90)
        target_end = snapshot + pd.offsets.MonthBegin(1)
        for customer in customers.itertuples(index=False):
            customer_orders = transactions.loc[transactions["customer_id"].eq(customer.customer_id)]
            history = customer_orders.loc[
                customer_orders["purchase_date"].ge(history_start)
                & customer_orders["purchase_date"].lt(snapshot)
            ]
            all_previous = customer_orders.loc[customer_orders["purchase_date"].lt(snapshot)]
            target_orders = customer_orders.loc[
                customer_orders["purchase_date"].ge(snapshot)
                & customer_orders["purchase_date"].lt(target_end)
            ]
            amounts = history["purchase_amount_lkr"]
            target_spend = _round_money(float(target_orders["purchase_amount_lkr"].sum()))
            last_purchase = all_previous["purchase_date"].max()
            average = _round_money(float(amounts.mean())) if len(amounts) else 0.0
            # The supplied table uses sample standard deviation and zero for fewer than two orders.
            standard_deviation = _round_money(float(amounts.std(ddof=1))) if len(amounts) > 1 else 0.0
            rows.append({
                "customer_id": customer.customer_id,
                "snapshot_date": snapshot,
                "tenure_days": int((snapshot - customer.signup_date).days),
                "orders_last_90d": int(len(history)),
                "spend_last_90d_lkr": _round_money(float(amounts.sum())),
                "avg_order_value_last_90d_lkr": average,
                "order_value_std_last_90d_lkr": standard_deviation,
                "days_since_last_purchase": float((snapshot - last_purchase).days) if pd.notna(last_purchase) else np.nan,
                "has_previous_purchase": int(pd.notna(last_purchase)),
                "next_month_spend_lkr": target_spend,
                "will_purchase_next_month": int(target_spend > 0),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild monthly features from supplied raw CSVs")
    parser.add_argument("--force-rebuild", action="store_true", help="Replace the supplied processed table with a reconstruction from rounded raw amounts")
    arguments = parser.parse_args()
    for path in (CUSTOMERS_PATH, TRANSACTIONS_PATH):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Supplied raw file is required: {path}")
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0 and not arguments.force_rebuild:
        print(f"Preserved supplied processed table: {OUTPUT_PATH}")
        print("Use --force-rebuild only if you intentionally want to reconstruct it from rounded raw amounts.")
        return
    customers = pd.read_csv(CUSTOMERS_PATH, parse_dates=["signup_date"])
    transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["purchase_date"])
    monthly = build_monthly_table(customers, transactions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Preserved raw CSVs and wrote {len(monthly):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
