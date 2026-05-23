"""Ortak RL ajan arayüzü — act / learn / save / load."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        ...

    @abstractmethod
    def update(self, *args, **kwargs):
        ...

    @abstractmethod
    def save(self, path: str):
        ...

    @abstractmethod
    def load(self, path: str):
        ...
