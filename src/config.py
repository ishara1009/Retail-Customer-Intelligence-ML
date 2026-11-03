
"""Small, central configuration for the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONTHLY_DATA_PATH = PROCESSED_DATA_DIR / "customer_monthly_ml.csv"
CUSTOMERS_PATH = RAW_DATA_DIR / "customers.csv"
TRANSACTIONS_PATH = RAW_DATA_DIR / "retail_transactions.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

FEATURE_COLUMNS = [
    "tenure_days",
    "orders_last_90d",
    "spend_last_90d_lkr",
    "avg_order_value_last_90d_lkr",
    "order_value_std_last_90d_lkr",
    "days_since_last_purchase",
    "has_previous_purchase",
]
METADATA_COLUMNS = ["customer_id", "snapshot_date"]
REGRESSION_TARGET = "next_month_spend_lkr"
CLASSIFICATION_TARGET = "will_purchase_next_month"
TARGET_COLUMNS = [REGRESSION_TARGET, CLASSIFICATION_TARGET]

TRAIN_START = "2025-04-01"
TRAIN_END = "2025-11-01"
VALIDATION_START = "2025-12-01"
VALIDATION_END = "2026-01-01"
TEST_START = "2026-02-01"
TEST_END = "2026-03-01"
RANDOM_SEED = 42


def ensure_output_directories() -> None:
    """Create folders used by generated reports."""
    for directory in (FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
