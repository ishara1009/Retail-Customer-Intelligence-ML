
"""Train and evaluate the standalone GridWorld demo."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reinforcement_learning.q_learning import evaluate_policy, train_q_learning
from src.config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, RANDOM_SEED, ensure_output_directories
from src.evaluation import save_json


def run_rl_demo(episodes: int = 1000) -> dict:
    ensure_output_directories()
    q_table, rewards = train_q_learning(episodes=episodes, seed=RANDOM_SEED)
    learned_metrics, path = evaluate_policy(q_table, episodes=100, seed=RANDOM_SEED + 1)
    random_metrics, _ = evaluate_policy(None, episodes=100, seed=RANDOM_SEED + 1)
    pd.DataFrame({"episode": range(1, episodes + 1), "reward": rewards}).to_csv(PREDICTIONS_DIR / "rl_episode_rewards.csv", index=False)
    np.savetxt(PREDICTIONS_DIR / "rl_q_table.csv", q_table, delimiter=",", header="up,down,left,right", comments="")

    moving = pd.Series(rewards).rolling(window=min(50, episodes), min_periods=1).mean()
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, episodes + 1), moving)
    plt.xlabel("Training episode")
    plt.ylabel("Moving-average reward")
    plt.title("Q-learning training reward")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rl_training_rewards.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.xlim(-0.5, 3.5); plt.ylim(3.5, -0.5)
    plt.xticks(range(4)); plt.yticks(range(4)); plt.grid(True)
    for start, end in zip(path, path[1:]):
        plt.annotate("", xy=(end[1], end[0]), xytext=(start[1], start[0]), arrowprops={"arrowstyle": "->", "color": "tab:blue"})
    plt.scatter([0, 3], [0, 3], c=["green", "gold"], s=160)
    plt.title("Greedy learned path: start to goal")
    plt.xlabel("Column"); plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rl_learned_path.png", dpi=150)
    plt.close()
    report = {
        "training_episodes": episodes,
        "learned_policy": learned_metrics,
        "uniform_random_baseline": random_metrics,
        "representative_greedy_path": [list(position) for position in path],
        "truncation_note": "The 50-step limit stops an episode but is not the terminal goal; Q-learning may bootstrap on a truncated transition.",
        "scope_note": "This educational RL policy is not learned from retail transactions.",
    }
    save_json(report, METRICS_DIR / "rl_metrics.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GridWorld Q-learning demo")
    parser.add_argument("--episodes", type=int, default=1000)
    arguments = parser.parse_args()
    print(run_rl_demo(arguments.episodes))


if __name__ == "__main__":
    main()
