
"""Next-month purchase classification workflow."""
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import CLASSIFICATION_TARGET, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, RANDOM_SEED, ensure_output_directories
from src.data_loader import load_monthly_data
from src.evaluation import classification_metrics, save_json
from src.preprocessing import chronological_split, model_inputs, numeric_preprocessor
from src.visualization import plot_confusion


def classification_candidates() -> dict[str, Pipeline]:
    candidates = {
        "dummy_most_frequent": Pipeline([("preprocess", numeric_preprocessor(False)), ("model", DummyClassifier(strategy="most_frequent"))]),
        "logistic_regression": Pipeline([("preprocess", numeric_preprocessor(True)), ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))]),
    }
    for depth in [2, 4, 6, 8]:
        candidates[f"decision_tree_depth_{depth}"] = Pipeline([
            ("preprocess", numeric_preprocessor(False)),
            ("model", DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=RANDOM_SEED)),
        ])
    return candidates


def _positive_probability(pipeline: Pipeline, inputs: pd.DataFrame):
    probabilities = pipeline.predict_proba(inputs)
    classes = list(pipeline.named_steps["model"].classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else None


def run_classification(data: pd.DataFrame | None = None) -> dict:
    ensure_output_directories()
    data = load_monthly_data() if data is None else data
    train, validation, test = chronological_split(data)
    candidates = classification_candidates()
    validation_results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(model_inputs(train), train[CLASSIFICATION_TARGET])
        predictions = pipeline.predict(model_inputs(validation))
        probabilities = _positive_probability(pipeline, model_inputs(validation))
        validation_results[name] = classification_metrics(validation[CLASSIFICATION_TARGET], predictions, probabilities)

    selected_name = max(validation_results, key=lambda name: validation_results[name]["f1"])
    combined = pd.concat([train, validation], ignore_index=True)
    final_model = clone(candidates[selected_name])
    final_model.fit(model_inputs(combined), combined[CLASSIFICATION_TARGET])
    test_probabilities = _positive_probability(final_model, model_inputs(test))
    if test_probabilities is None:
        test_predictions = final_model.predict(model_inputs(test))
        probability_values = [None] * len(test)
    else:
        test_predictions = (test_probabilities >= 0.5).astype(int)
        probability_values = test_probabilities
    matrix = confusion_matrix(test[CLASSIFICATION_TARGET], test_predictions, labels=[0, 1])
    test_metrics = classification_metrics(test[CLASSIFICATION_TARGET], test_predictions, test_probabilities)

    output = test[["customer_id", "snapshot_date"]].copy()
    output["actual_will_purchase"] = test[CLASSIFICATION_TARGET].to_numpy()
    output["purchase_probability"] = probability_values
    output["predicted_will_purchase"] = test_predictions
    output.to_csv(PREDICTIONS_DIR / "classification_test_predictions.csv", index=False)
    plot_confusion(matrix, FIGURES_DIR / "classification_confusion_matrix.png")
    report = {
        "selection_rule": "highest validation F1 at threshold 0.5",
        "threshold": 0.5,
        "validation_candidates": validation_results,
        "selected_model": selected_name,
        "selected_parameters": final_model.named_steps["model"].get_params(),
        "test_metrics": test_metrics,
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "target_distribution_test": {str(key): int(value) for key, value in test[CLASSIFICATION_TARGET].value_counts().sort_index().items()},
        "roc_auc_note": "ROC-AUC is null when the evaluation data does not contain both classes.",
    }
    save_json(report, METRICS_DIR / "classification_metrics.json")
    return report
