
"""Command-line entry point for every portfolio task."""
import argparse
from collections.abc import Callable

from experiments.bias_variance import run_bias_variance_experiment
from experiments.feature_selection import run_feature_selection_experiment
from experiments.overfitting_underfitting import run_overfitting_experiment
from reinforcement_learning.run_demo import run_rl_demo
from src.classification import run_classification
from src.clustering import run_clustering
from src.config import OUTPUT_DIR
from src.regression import run_regression


def run_experiments() -> dict:
    return {
        "overfitting_underfitting": run_overfitting_experiment(),
        "bias_variance": run_bias_variance_experiment(),
        "feature_selection": run_feature_selection_experiment(),
    }


def dispatch(task: str) -> None:
    tasks: dict[str, Callable[[], dict]] = {
        "regression": run_regression,
        "classification": run_classification,
        "clustering": run_clustering,
        "experiments": run_experiments,
        "rl": run_rl_demo,
    }
    selected_tasks = list(tasks) if task == "all" else [task]
    for selected in selected_tasks:
        print(f"\nRunning {selected}...")
        result = tasks[selected]()
        print(f"Finished {selected}. Summary: {result}")
    print(f"\nGenerated files are in: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Customer Intelligence ML portfolio")
    parser.add_argument("--task", required=True, choices=["regression", "classification", "clustering", "experiments", "rl", "all"])
    dispatch(parser.parse_args().task)


if __name__ == "__main__":
    main()
