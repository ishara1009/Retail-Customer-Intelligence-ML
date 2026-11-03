
"""Chronological splitting and reusable model preprocessing."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)


def chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split customer-month rows with explicit, non-overlapping date masks."""
    dates = pd.to_datetime(data["snapshot_date"])
    train = data.loc[dates.between(TRAIN_START, TRAIN_END)].copy()
    validation = data.loc[dates.between(VALIDATION_START, VALIDATION_END)].copy()
    test = data.loc[dates.between(TEST_START, TEST_END)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("One or more chronological partitions are empty; check snapshot dates")
    return train, validation, test


def model_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Use an explicit allowlist so IDs and both targets cannot leak into X."""
    return data.loc[:, FEATURE_COLUMNS].copy()


def numeric_preprocessor(scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
