
"""Compare all features with training-only univariate feature selection."""
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLASSIFICATION_TARGET, FEATURE_COLUMNS, METRICS_DIR, RANDOM_SEED, ensure_output_directories
from src.data_loader import load_monthly_data
from src.evaluation import classification_metrics, save_json
from src.preprocessing import chronological_split, model_inputs


def _pipeline(select: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if select:
        steps.append(("selection", SelectKBest(score_func=f_classif, k=4)))
    steps.append(("model", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)))
    return Pipeline(steps)


def run_feature_selection_experiment(data: pd.DataFrame | None = None) -> dict:
    ensure_output_directories()
    data = load_monthly_data() if data is None else data
    train, validation, _ = chronological_split(data)
    comparisons = {}
    fitted = {}
    for name, select in [("all_seven_features", False), ("select_k_best_4", True)]:
        pipeline = _pipeline(select)
        pipeline.fit(model_inputs(train), train[CLASSIFICATION_TARGET])
        predicted = pipeline.predict(model_inputs(validation))
        probabilities = pipeline.predict_proba(model_inputs(validation))[:, 1]
        comparisons[name] = classification_metrics(validation[CLASSIFICATION_TARGET], predicted, probabilities)
        fitted[name] = pipeline
    selector = fitted["select_k_best_4"].named_steps["selection"]
    scores = {feature: float(score) for feature, score in zip(FEATURE_COLUMNS, selector.scores_)}
    selected = [feature for feature, keep in zip(FEATURE_COLUMNS, selector.get_support()) if keep]
    report = {
        "validation_comparison": comparisons,
        "selected_features": selected,
        "selection_scores": scores,
        "outcome_note": "Use the measured validation results; selection can improve, preserve, or reduce performance.",
        "redundancy_note": "Total spend is approximately order count multiplied by mean order value, so some historical inputs are redundant.",
    }
    save_json(report, METRICS_DIR / "feature_selection_metrics.json")
    return report
