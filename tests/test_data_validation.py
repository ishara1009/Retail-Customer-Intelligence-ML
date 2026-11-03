
"""Focused checks for data semantics, time splits, and preprocessing."""
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMNS
from src.data_loader import load_monthly_data, load_transactions
from src.preprocessing import chronological_split, numeric_preprocessor


def test_customer_month_key_is_unique() -> None:
    data = load_monthly_data()
    assert not data.duplicated(["customer_id", "snapshot_date"]).any()


def test_chronological_splits_do_not_overlap() -> None:
    train, validation, test = chronological_split(load_monthly_data())
    assert train["snapshot_date"].max() < validation["snapshot_date"].min()
    assert validation["snapshot_date"].max() < test["snapshot_date"].min()
    assert len(train) + len(validation) + len(test) == 12_000


def test_feature_allowlist_excludes_metadata_and_targets() -> None:
    assert set(FEATURE_COLUMNS).isdisjoint(METADATA_COLUMNS + TARGET_COLUMNS)


def test_binary_target_matches_positive_spend() -> None:
    data = load_monthly_data()
    expected = (data["next_month_spend_lkr"] > 0).astype(int)
    pd.testing.assert_series_equal(expected, data["will_purchase_next_month"], check_names=False)


def test_representative_snapshot_reconciles_with_transactions() -> None:
    data = load_monthly_data()
    transactions = load_transactions()
    row = data.loc[data["snapshot_date"].eq("2025-04-01") & data["customer_id"].eq("C0002")].iloc[0]
    customer_orders = transactions.loc[transactions["customer_id"].eq("C0002")]
    history = customer_orders.loc[customer_orders["purchase_date"].ge("2025-01-01") & customer_orders["purchase_date"].lt("2025-04-01")]
    target = customer_orders.loc[customer_orders["purchase_date"].ge("2025-04-01") & customer_orders["purchase_date"].lt("2025-05-01")]
    assert row["orders_last_90d"] == len(history)
    assert np.isclose(row["spend_last_90d_lkr"], history["purchase_amount_lkr"].sum())
    assert np.isclose(row["next_month_spend_lkr"], target["purchase_amount_lkr"].sum())


def test_preprocessing_statistics_use_training_data_only() -> None:
    training = pd.DataFrame({"feature": [1.0, np.nan, 3.0]})
    future_outlier = pd.DataFrame({"feature": [1000.0]})
    pipeline = numeric_preprocessor(scale=True)
    pipeline.fit(training)
    assert pipeline.named_steps["imputer"].statistics_[0] == 2.0
    assert pipeline.named_steps["scaler"].mean_[0] == 2.0
    assert pipeline.transform(future_outlier)[0, 0] > 100
