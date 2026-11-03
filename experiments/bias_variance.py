
"""Educational bias-variance simulation separate from the retail data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.config import FIGURES_DIR, METRICS_DIR, RANDOM_SEED, ensure_output_directories
from src.evaluation import save_json


def run_bias_variance_experiment(dataset_count: int = 30, noise_std: float = 0.30) -> dict:
    ensure_output_directories()
    rng = np.random.default_rng(RANDOM_SEED)
    grid = np.linspace(0, 2 * np.pi, 200)
    true_values = np.sin(grid)
    results = {}
    plt.figure(figsize=(9, 5))
    colors = {"shallow_depth_2": "tab:blue", "deep_unlimited": "tab:orange"}
    for name, depth in [("shallow_depth_2", 2), ("deep_unlimited", None)]:
        predictions = []
        for _ in range(dataset_count):
            training_x = np.sort(rng.uniform(0, 2 * np.pi, size=40))
            training_y = np.sin(training_x) + rng.normal(0, noise_std, size=40)
            model = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_SEED)
            model.fit(training_x.reshape(-1, 1), training_y)
            predictions.append(model.predict(grid.reshape(-1, 1)))
        prediction_array = np.asarray(predictions)
        mean_prediction = prediction_array.mean(axis=0)
        prediction_std = prediction_array.std(axis=0)
        results[name] = {
            "average_squared_bias": float(np.mean((mean_prediction - true_values) ** 2)),
            "average_variance": float(np.mean(np.var(prediction_array, axis=0))),
            "specified_noise_variance": float(noise_std ** 2),
        }
        plt.plot(grid, mean_prediction, color=colors[name], label=f"{name} mean")
        plt.fill_between(grid, mean_prediction - prediction_std, mean_prediction + prediction_std, color=colors[name], alpha=0.18)
    plt.plot(grid, true_values, color="black", linestyle="--", label="Noise-free sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Bias-variance simulation across independent training sets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bias_variance_simulation.png", dpi=150, bbox_inches="tight")
    plt.close()
    report = {
        "simulation": "30 independent noisy samples from y = sin(x) + Gaussian noise",
        "dataset_count": dataset_count,
        "results": results,
        "note": "These finite-sample estimates need not exactly equal a population error decomposition. A single train/validation gap is not a direct estimate of statistical variance.",
    }
    save_json(report, METRICS_DIR / "bias_variance_metrics.json")
    return report
