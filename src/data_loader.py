
"""CSV loading and validation helpers."""
from pathlib import Path

import pandas as pd

from src.config import (
    CLASSIFICATION_TARGET,
    CUSTOMERS_PATH,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MONTHLY_DATA_PATH,
    REGRESSION_TARGET,
    TRANSACTIONS_PATH,
)


def _require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required data file is missing or empty: {path}")


def load_monthly_data(path: Path = MONTHLY_DATA_PATH) -> pd.DataFrame:
    """Load and validate the customer-month modeling table."""
    _require_file(path)
    data = pd.read_csv(path, parse_dates=["snapshot_date"])
    validate_monthly_data(data)
    return data


def load_customers(path: Path = CUSTOMERS_PATH) -> pd.DataFrame:
    _require_file(path)
    data = pd.read_csv(path, parse_dates=["signup_date"])
    required = {"customer_id", "signup_date"}
    if not required.issubset(data.columns):
        raise ValueError(f"Customers file must contain {sorted(required)}")
    if data["customer_id"].duplicated().any():
        raise ValueError("customers.csv contains duplicate customer_id values")
    return data


def load_transactions(path: Path = TRANSACTIONS_PATH) -> pd.DataFrame:
    _require_file(path)
    data = pd.read_csv(path, parse_dates=["purchase_date"])
    required = {"order_id", "customer_id", "purchase_date", "purchase_amount_lkr"}
    if not required.issubset(data.columns):
        raise ValueError(f"Transactions file must contain {sorted(required)}")
    if data["order_id"].duplicated().any():
        raise ValueError("retail_transactions.csv contains duplicate order_id values")
    if data["purchase_amount_lkr"].isna().any() or (data["purchase_amount_lkr"] < 0).any():
        raise ValueError("Transaction amounts must be present and non-negative")
    return data


def validate_monthly_data(data: pd.DataFrame) -> None:
    """Raise useful errors instead of silently dropping invalid records."""
    required = set(METADATA_COLUMNS + FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Monthly data is missing required columns: {missing}")
    if data[METADATA_COLUMNS].isna().any().any():
        raise ValueError("Customer-month metadata cannot be missing")
    if data.duplicated(METADATA_COLUMNS).any():
        raise ValueError("customer_id + snapshot_date must be unique")
    non_negative = [column for column in FEATURE_COLUMNS + [REGRESSION_TARGET] if column != "days_since_last_purchase"]
    if data[non_negative].isna().any().any():
        raise ValueError("Features other than recency cannot be missing")
    if (data[non_negative] < 0).any().any():
        raise ValueError("Historical values and spend target cannot be negative")
    if data[CLASSIFICATION_TARGET].isna().any() or not set(data[CLASSIFICATION_TARGET].unique()).issubset({0, 1}):
        raise ValueError("Classification target must contain only 0 and 1")
    expected_purchase = (data[REGRESSION_TARGET] > 0).astype(int)
    if not expected_purchase.equals(data[CLASSIFICATION_TARGET].astype(int)):
        raise ValueError("Purchase target must equal whether next-month spend is positive")
    invalid_blank_recency = data["days_since_last_purchase"].isna() & data["has_previous_purchase"].eq(1)
    if invalid_blank_recency.any():
        raise ValueError("Blank recency is valid only when there is no previous purchase")
