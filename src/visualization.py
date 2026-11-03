
"""Reusable plots saved with non-interactive Matplotlib."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(actual, predicted, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=actual, y=predicted, alpha=0.45)
    maximum = max(float(max(actual)), float(max(predicted)))
    plt.plot([0, maximum], [0, maximum], "--", color="black", label="Perfect prediction")
    plt.xlabel("Actual next-month spend (LKR)")
    plt.ylabel("Predicted next-month spend (LKR)")
    plt.title("Regression: actual versus predicted")
    plt.legend()
    _save(path)


def plot_residuals(predicted, residuals, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=predicted, y=residuals, alpha=0.45)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted next-month spend (LKR)")
    plt.ylabel("Residual: actual - predicted (LKR)")
    plt.title("Regression residuals")
    _save(path)


def plot_confusion(matrix, path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Classification confusion matrix")
    _save(path)


def plot_clusters(data: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=data, x="orders_last_90d", y="spend_last_90d_lkr", hue="cluster", palette="tab10")
    plt.ylabel("Spend in last 90 days (LKR)")
    plt.xlabel("Orders in last 90 days")
    plt.title("Customer clusters at 2025-11-01 snapshot")
    _save(path)


def plot_error_curves(values: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(values["complexity"], values["train_rmse"], marker="o", label="Training RMSE")
    plt.plot(values["complexity"], values["validation_rmse"], marker="o", label="Validation RMSE")
    plt.xlabel("Tree complexity (depth; None shown as 15)")
    plt.ylabel("RMSE (LKR)")
    plt.title("Underfitting and overfitting comparison")
    plt.legend()
    _save(path)


def plot_learning_curve(values: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(values["training_months"], values["train_rmse"], marker="o", label="Training RMSE")
    plt.plot(values["training_months"], values["validation_rmse"], marker="o", label="Validation RMSE")
    plt.xlabel("Number of chronological training months")
    plt.ylabel("RMSE (LKR)")
    plt.title("Chronological learning curve")
    plt.legend()
    _save(path)
