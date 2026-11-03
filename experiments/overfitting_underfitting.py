
"""Observe tree complexity and learning curves without touching test data."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import FIGURES_DIR, METRICS_DIR, RANDOM_SEED, REGRESSION_TARGET, ensure_output_directories
from src.data_loader import load_monthly_data
from src.evaluation import regression_metrics, save_json
from src.preprocessing import chronological_split, model_inputs, numeric_preprocessor
from src.visualization import plot_error_curves, plot_learning_curve


def _tree(depth: int | None = 5) -> Pipeline:
    return Pipeline([
        ("preprocess", numeric_preprocessor(False)),
        ("model", DecisionTreeRegressor(max_depth=depth, min_samples_leaf=20, random_state=RANDOM_SEED)),
    ])


def run_overfitting_experiment(data: pd.DataFrame | None = None) -> dict:
    ensure_output_directories()
    data = load_monthly_data() if data is None else data
    train, validation, _ = chronological_split(data)
    depth_rows = []
    for depth in [1, 2, 3, 5, 8, 12, None]:
        model = _tree(depth)
        model.fit(model_inputs(train), train[REGRESSION_TARGET])
        depth_rows.append({
            "depth": depth,
            "complexity": 15 if depth is None else depth,
            "train_rmse": regression_metrics(train[REGRESSION_TARGET], model.predict(model_inputs(train)))["rmse"],
            "validation_rmse": regression_metrics(validation[REGRESSION_TARGET], model.predict(model_inputs(validation)))["rmse"],
        })
    depth_results = pd.DataFrame(depth_rows)
    depth_results.to_csv(METRICS_DIR / "overfitting_tree_depths.csv", index=False)
    plot_error_curves(depth_results, FIGURES_DIR / "overfitting_tree_depths.png")

    months = sorted(train["snapshot_date"].unique())
    learning_rows = []
    for month_count in range(1, len(months) + 1):
        prefix = train.loc[train["snapshot_date"].isin(months[:month_count])]
        model = _tree(5)  # A fresh pipeline prevents preprocessing leakage between prefixes.
        model.fit(model_inputs(prefix), prefix[REGRESSION_TARGET])
        learning_rows.append({
            "training_months": month_count,
            "last_training_month": str(pd.Timestamp(months[month_count - 1]).date()),
            "train_rmse": regression_metrics(prefix[REGRESSION_TARGET], model.predict(model_inputs(prefix)))["rmse"],
            "validation_rmse": regression_metrics(validation[REGRESSION_TARGET], model.predict(model_inputs(validation)))["rmse"],
        })
    learning_results = pd.DataFrame(learning_rows)
    learning_results.to_csv(METRICS_DIR / "overfitting_learning_curve.csv", index=False)
    plot_learning_curve(learning_results, FIGURES_DIR / "overfitting_learning_curve.png")
    report = {
        "tree_depth_results": depth_rows,
        "learning_curve": learning_rows,
        "interpretation": "Compare the measured training and validation errors: shallow trees can underfit and deeper trees can overfit, but the observed values determine the conclusion.",
    }
    save_json(report, METRICS_DIR / "overfitting_underfitting_metrics.json")
    return report
