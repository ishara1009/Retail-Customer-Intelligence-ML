
"""Next-month spending regression workflow."""
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, RANDOM_SEED, REGRESSION_TARGET, ensure_output_directories
from src.data_loader import load_monthly_data
from src.evaluation import regression_metrics, save_json
from src.preprocessing import chronological_split, model_inputs, numeric_preprocessor
from src.visualization import plot_actual_vs_predicted, plot_residuals


def regression_candidates() -> dict[str, Pipeline]:
    candidates = {
        "dummy_mean": Pipeline([("preprocess", numeric_preprocessor(False)), ("model", DummyRegressor(strategy="mean"))]),
        "linear_regression": Pipeline([("preprocess", numeric_preprocessor(True)), ("model", LinearRegression())]),
    }
    for depth in [2, 4, 6, 8]:
        candidates[f"decision_tree_depth_{depth}"] = Pipeline([
            ("preprocess", numeric_preprocessor(False)),
            ("model", DecisionTreeRegressor(max_depth=depth, min_samples_leaf=20, random_state=RANDOM_SEED)),
        ])
    return candidates


def run_regression(data: pd.DataFrame | None = None) -> dict:
    ensure_output_directories()
    data = load_monthly_data() if data is None else data
    train, validation, test = chronological_split(data)
    validation_results = {}
    candidates = regression_candidates()
    for name, pipeline in candidates.items():
        pipeline.fit(model_inputs(train), train[REGRESSION_TARGET])
        predictions = pipeline.predict(model_inputs(validation))
        validation_results[name] = regression_metrics(validation[REGRESSION_TARGET], predictions)

    selected_name = min(validation_results, key=lambda name: validation_results[name]["mae"])
    combined = pd.concat([train, validation], ignore_index=True)
    final_model = clone(candidates[selected_name])
    final_model.fit(model_inputs(combined), combined[REGRESSION_TARGET])
    test_predictions = final_model.predict(model_inputs(test))
    test_metrics = regression_metrics(test[REGRESSION_TARGET], test_predictions)

    output = test[["customer_id", "snapshot_date"]].copy()
    output["actual_next_month_spend_lkr"] = test[REGRESSION_TARGET].to_numpy()
    output["predicted_next_month_spend_lkr"] = test_predictions
    output.to_csv(PREDICTIONS_DIR / "regression_test_predictions.csv", index=False)
    plot_actual_vs_predicted(output["actual_next_month_spend_lkr"], output["predicted_next_month_spend_lkr"], FIGURES_DIR / "regression_actual_vs_predicted.png")
    residuals = output["actual_next_month_spend_lkr"] - output["predicted_next_month_spend_lkr"]
    plot_residuals(output["predicted_next_month_spend_lkr"], residuals, FIGURES_DIR / "regression_residuals.png")

    report = {
        "selection_rule": "lowest validation MAE",
        "validation_candidates": validation_results,
        "selected_model": selected_name,
        "selected_parameters": final_model.named_steps["model"].get_params(),
        "test_metrics": test_metrics,
        "negative_predictions": int((test_predictions < 0).sum()),
        "negative_prediction_note": "Linear regression may predict negative spend; predictions are intentionally not clipped.",
    }
    save_json(report, METRICS_DIR / "regression_metrics.json")
    return report
