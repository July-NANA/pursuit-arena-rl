from __future__ import annotations

from pathlib import Path

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        algorithm: str = "double",
    ) -> None:
        if algorithm not in {"baseline", "double"}:
            raise ValueError("algorithm must be 'baseline' or 'double'")

        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.algorithm = algorithm

        self.q1 = np.zeros((n_states, n_actions), dtype=np.float32)
        self.q2 = np.zeros((n_states, n_actions), dtype=np.float32)

    def reset_exploration(self) -> None:
        self.epsilon = self.epsilon_start

    def _q_for_policy(self) -> np.ndarray:
        if self.algorithm == "double":
            return self.q1 + self.q2
        return self.q1

    def select_action(self, state_id: int, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        qvals = self._q_for_policy()[state_id]
        max_q = np.max(qvals)
        candidates = np.flatnonzero(np.isclose(qvals, max_q))
        return int(np.random.choice(candidates))

    def update_q(self, state_id: int, action: int, reward: float, next_state_id: int, done: bool) -> None:
        if self.algorithm == "baseline":
            best_next = 0.0 if done else float(np.max(self.q1[next_state_id]))
            target = reward + self.gamma * best_next
            self.q1[state_id, action] += self.alpha * (target - self.q1[state_id, action])
            return

        if np.random.rand() < 0.5:
            best_a = int(np.argmax(self.q1[next_state_id])) if not done else 0
            best_next = 0.0 if done else float(self.q2[next_state_id, best_a])
            target = reward + self.gamma * best_next
            self.q1[state_id, action] += self.alpha * (target - self.q1[state_id, action])
        else:
            best_a = int(np.argmax(self.q2[next_state_id])) if not done else 0
            best_next = 0.0 if done else float(self.q1[next_state_id, best_a])
            target = reward + self.gamma * best_next
            self.q2[state_id, action] += self.alpha * (target - self.q2[state_id, action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            q1=self.q1,
            q2=self.q2,
            algorithm=self.algorithm,
            epsilon=np.array([self.epsilon], dtype=np.float32),
        )

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.q1 = data["q1"]
        self.q2 = data["q2"]
        self.algorithm = str(data["algorithm"])
        self.epsilon = float(data["epsilon"][0])
