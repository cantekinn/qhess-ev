from __future__ import annotations

import pickle
import random
from collections import deque
from typing import Sequence

import numpy as np

from src.agents.base_agent import BaseAgent


def _bin_index(value: float, edges: Sequence[float]) -> int:
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


class QLearningAgent(BaseAgent):
    name = "q_learning"

    def __init__(
        self,
        n_actions: int,
        bins_p: Sequence[float],
        bins_soc_bat: Sequence[float],
        bins_soc_sc: Sequence[float],
        learning_rate: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985,
        init_battery_bias: float = 0.5,
        replay_last_n: int = 100,
        rng: np.random.Generator | None = None,
    ):
        self.n_actions = n_actions
        self.bins_p = list(bins_p)
        self.bins_b = list(bins_soc_bat)
        self.bins_s = list(bins_soc_sc)
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.init_battery_bias = init_battery_bias
        self.replay_last_n = int(replay_last_n)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.q: dict[tuple, np.ndarray] = {}
        self.rng_py = random.Random(int(self.rng.integers(0, 1_000_000)))
        # Episode buffer (replay için)
        self._ep_buffer: deque = deque(maxlen=max(self.replay_last_n, 1))

    def _key(self, state: np.ndarray) -> tuple:
        p, sb, ss = state[0], state[1], state[2]
        d = int(state[7])
        return (_bin_index(p, self.bins_p), _bin_index(sb, self.bins_b), _bin_index(ss, self.bins_s), d)

    def _q_row(self, key: tuple) -> np.ndarray:
        if key not in self.q:
            row = np.zeros(self.n_actions, dtype=np.float32)
            # Bias: a=0 (sadece batarya) düşük talepte mantıklı → küçük + init
            row[0] = self.init_battery_bias
            self.q[key] = row
        return self.q[key]

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and self.rng_py.random() < self.epsilon:
            return self.rng_py.randint(0, self.n_actions - 1)
        return int(np.argmax(self._q_row(self._key(state))))

    def _update_one(self, key: tuple, a: int, r: float, key_next: tuple | None, done: bool):
        old = self._q_row(key)[a]
        if done or key_next is None:
            target = r
        else:
            target = r + self.gamma * float(np.max(self._q_row(key_next)))
        self.q[key][a] = old + self.alpha * (target - old)

    def update(self, s, a, r, s_next, done: bool):
        key = self._key(s)
        key_next = self._key(s_next) if s_next is not None else None
        self._update_one(key, a, r, key_next, done)
        self._ep_buffer.append((key, a, r, key_next, done))
        return 0.0

    def end_episode(self):
        # Replay: son N adımı bir kez daha update et (off-policy benzeri pekiştirme)
        for key, a, r, key_next, done in list(self._ep_buffer):
            self._update_one(key, a, r, key_next, done)
        self._ep_buffer.clear()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def coverage(self) -> float:
        """Q-tablosu doluluk oranı (yaklaşık)."""
        max_states = (len(self.bins_p) + 1) * (len(self.bins_b) + 1) * (len(self.bins_s) + 1) * 5
        return len(self.q) / max(max_states, 1)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q": self.q,
                    "epsilon": self.epsilon,
                    "config": {
                        "n_actions": self.n_actions,
                        "bins_p": self.bins_p,
                        "bins_b": self.bins_b,
                        "bins_s": self.bins_s,
                        "alpha": self.alpha,
                        "gamma": self.gamma,
                    },
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.q = d["q"]
        self.epsilon = d["epsilon"]
