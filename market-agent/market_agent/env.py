"""Gymnasium-style market environment scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
from datasets import Dataset, Value
from gymnasium import spaces


@dataclass
class MarketEnvConfig:
    """Configuration for the market environment."""

    window_size: int = 30
    initial_cash: float = 10000.0
    max_position: int = 1
    price_column: str = "close"
    reward_scale: float = 1.0
    feature_columns: Optional[Sequence[str]] = None


class Action(IntEnum):
    SELL = 0
    HOLD = 1
    BUY = 2


class MarketEnv(gym.Env):
    """Minimal Gymnasium-style market environment."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        dataset: Dataset,
        *,
        config: Optional[MarketEnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config or MarketEnvConfig()
        self.render_mode = render_mode

        self._feature_columns = self._resolve_feature_columns()

        self._index = 0
        self._position = 0
        self._cash = float(self.config.initial_cash)
        self._prev_value = float(self.config.initial_cash)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.window_size, len(self._feature_columns)),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(Action))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._index = 0
        self._position = 0
        self._cash = float(self.config.initial_cash)
        self._prev_value = float(self.config.initial_cash)
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._apply_action(Action(action))
        self._index += 1
        terminated = self._index >= len(self.dataset) - 1
        observation = self._get_observation()
        reward = self._compute_reward()
        info = self._get_info()
        truncated = False
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        start = max(0, self._index - self.config.window_size + 1)
        end = self._index + 1
        window = self.dataset.select(range(start, end))
        frame = window.to_pandas()[self._feature_columns]
        array = frame.to_numpy(dtype=np.float32)
        if array.shape[0] < self.config.window_size:
            pad = np.zeros(
                (self.config.window_size - array.shape[0], array.shape[1]),
                dtype=np.float32,
            )
            array = np.vstack([pad, array])
        return array

    def _resolve_feature_columns(self) -> list[str]:
        if self.config.feature_columns:
            return list(self.config.feature_columns)
        numeric_columns: list[str] = []
        for name, feature in self.dataset.features.items():
            if isinstance(feature, Value):
                if feature.dtype.startswith(("int", "uint", "float", "double")):
                    numeric_columns.append(name)
        if not numeric_columns:
            raise ValueError(
                "No numeric columns found. Provide feature_columns in MarketEnvConfig."
            )
        return numeric_columns

    def _apply_action(self, action: Action) -> None:
        price = self._current_price()
        if action == Action.SELL:
            if self._position > 0:
                self._position -= 1
                self._cash += price
        elif action == Action.BUY:
            if self._position < self.config.max_position and self._cash >= price:
                self._position += 1
                self._cash -= price

    def _compute_reward(self) -> float:
        price = self._current_price()
        value = self._cash + self._position * price
        reward = (value - self._prev_value) * self.config.reward_scale
        self._prev_value = value
        return reward

    def _current_price(self) -> float:
        return float(self.dataset[self._index][self.config.price_column])

    def _get_info(self) -> Dict[str, Any]:
        return {
            "index": self._index,
            "position": self._position,
            "cash": self._cash,
        }
