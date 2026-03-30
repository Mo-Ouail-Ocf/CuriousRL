"""
curiosity/base.py

Every curiosity module exposes two methods to train.py:

    rewards_int, metrics = module.step(obs_t, action, obs_t1, infos, dones)
    metrics              = module.update(obs_t, action, obs_t1)

train.py calls these blindly — no isinstance, no branching.
Each module owns: obs normalization, reward normalization, episodic counts, everything.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import Tensor


class CuriosityModule(ABC):

    @abstractmethod
    def step(
        self,
        obs_t:  Tensor,       # (B, 7, 7, 3) uint8 — current obs
        action: Tensor,       # (B,)          int64
        obs_t1: Tensor,       # (B, 7, 7, 3) uint8 — next obs
        infos:  dict,         # vectorenv info dict (keys -> arrays of length B)
        dones:  np.ndarray,   # (B,) bool
    ) -> tuple:
        """
        Returns:
          rewards_int  (B,) float32 — normalized, ready for buffer
          metrics      dict of scalar floats for logging
        """

    @abstractmethod
    def update(
        self,
        obs_t:  Tensor,
        action: Tensor,
        obs_t1: Tensor,
    ) -> dict:
        """Train module parameters. Returns dict of loss scalars."""