
"""A minimal 4 x 4 GridWorld for learning RL fundamentals."""
from dataclasses import dataclass


@dataclass
class GridWorld:
    size: int = 4
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (3, 3)
    max_steps: int = 50

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ["up", "down", "left", "right"]

    def __post_init__(self) -> None:
        self.position = self.start
        self.steps = 0

    @property
    def number_of_states(self) -> int:
        return self.size * self.size

    @property
    def number_of_actions(self) -> int:
        return len(self.ACTIONS)

    def state_index(self, position: tuple[int, int] | None = None) -> int:
        row, column = self.position if position is None else position
        return row * self.size + column

    def reset(self) -> int:
        self.position = self.start
        self.steps = 0
        return self.state_index()

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        if action not in self.ACTIONS:
            raise ValueError(f"Action must be one of {list(self.ACTIONS)}")
        row_change, column_change = self.ACTIONS[action]
        row = min(max(self.position[0] + row_change, 0), self.size - 1)
        column = min(max(self.position[1] + column_change, 0), self.size - 1)
        self.position = (row, column)
        self.steps += 1
        terminated = self.position == self.goal
        truncated = self.steps >= self.max_steps and not terminated
        reward = 10.0 if terminated else -1.0
        return self.state_index(), reward, terminated, truncated
