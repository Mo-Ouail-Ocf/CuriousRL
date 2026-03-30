"""
curiosity/base.py

Abstract interface for ICM and RND curiosity modules.
NovelDModule is NOT a subclass — it wraps RNDModule with additional logic.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor


class CuriosityModule(ABC):

    @torch.no_grad()
    @abstractmethod
    def compute_surprise(
        self,
        obs_t: Tensor,
        action_t: Tensor = None,
        obs_t1: Tensor = None,
    ) -> Tensor:
        """
        Returns per-env surprise: (n_envs,) float32.
        - RND:  uses obs_t only.
        - ICM:  uses obs_t, action_t, obs_t1.
        """

    @abstractmethod
    def update(
        self,
        obs_t: Tensor,
        action_t: Tensor = None,
        obs_t1: Tensor = None,
    ) -> dict:
        """
        Trains curiosity module parameters.
        Returns dict of scalar metrics for logging.
        """
