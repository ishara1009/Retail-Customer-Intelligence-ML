
"""Readable tabular Q-learning functions."""
import numpy as np

from reinforcement_learning.gridworld import GridWorld


def q_learning_target(reward: float, next_max_q: float, terminated: bool, discount_factor: float) -> float:
    """A terminal goal has no future value to bootstrap."""
    if terminated:
        return reward
    return reward + discount_factor * next_max_q


def update_q_value(current_q: float, reward: float, next_max_q: float, terminated: bool, learning_rate: float, discount_factor: float) -> float:
    target = q_learning_target(reward, next_max_q, terminated, discount_factor)
    return current_q + learning_rate * (target - current_q)


def _random_best_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best))


def train_q_learning(
    episodes: int = 1000,
    learning_rate: float = 0.15,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    minimum_epsilon: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    environment = GridWorld()
    rng = np.random.default_rng(seed)
    q_table = np.zeros((environment.number_of_states, environment.number_of_actions))
    rewards = []
    for _ in range(episodes):
        state = environment.reset()
        total_reward = 0.0
        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(environment.number_of_actions))
            else:
                action = _random_best_action(q_table[state], rng)
            next_state, reward, terminated, truncated = environment.step(action)
            q_table[state, action] = update_q_value(
                q_table[state, action], reward, q_table[next_state].max(), terminated,
                learning_rate, discount_factor,
            )
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        rewards.append(total_reward)
        epsilon = max(minimum_epsilon, epsilon * epsilon_decay)
    return q_table, rewards


def evaluate_policy(q_table: np.ndarray | None, episodes: int = 100, seed: int = 42) -> tuple[dict, list[tuple[int, int]]]:
    environment = GridWorld()
    rng = np.random.default_rng(seed)
    successes, rewards, lengths = 0, [], []
    representative_path = []
    for episode in range(episodes):
        state = environment.reset()
        path = [environment.position]
        total_reward = 0.0
        while True:
            action = int(rng.integers(environment.number_of_actions)) if q_table is None else _random_best_action(q_table[state], rng)
            state, reward, terminated, truncated = environment.step(action)
            path.append(environment.position)
            total_reward += reward
            if terminated or truncated:
                successes += int(terminated)
                break
        rewards.append(total_reward)
        lengths.append(environment.steps)
        if episode == 0:
            representative_path = path
    metrics = {
        "success_rate": float(successes / episodes),
        "mean_episode_reward": float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "episodes": episodes,
    }
    return metrics, representative_path
