
"""Environment behavior and terminal Q-learning checks."""
from reinforcement_learning.gridworld import GridWorld
from reinforcement_learning.q_learning import update_q_value


def test_boundary_movement_keeps_agent_in_grid() -> None:
    environment = GridWorld()
    environment.reset()
    state, reward, terminated, truncated = environment.step(0)
    assert environment.position == (0, 0)
    assert state == 0 and reward == -1 and not terminated and not truncated


def test_normal_movement_and_reward() -> None:
    environment = GridWorld()
    environment.reset()
    state, reward, terminated, _ = environment.step(1)
    assert environment.position == (1, 0)
    assert state == 4 and reward == -1 and not terminated


def test_goal_reward_and_termination() -> None:
    environment = GridWorld()
    environment.position = (3, 2)
    state, reward, terminated, truncated = environment.step(3)
    assert state == 15 and reward == 10 and terminated and not truncated


def test_reset_restores_start_and_step_counter() -> None:
    environment = GridWorld()
    environment.step(1)
    assert environment.reset() == 0
    assert environment.position == (0, 0) and environment.steps == 0


def test_terminal_update_does_not_bootstrap() -> None:
    updated = update_q_value(current_q=2.0, reward=10.0, next_max_q=999.0, terminated=True, learning_rate=0.5, discount_factor=0.95)
    assert updated == 6.0
