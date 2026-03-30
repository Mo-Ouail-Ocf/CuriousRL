"""
utils/running_stats.py

Welford online algorithm for running mean and variance.
Used by:
  - RewardNormalizer  (divide by std only — never subtract mean)
  - ObsNormalizer     (RND-specific: clip to [-5, 5] after normalization)
"""

import numpy as np
import torch


class RunningMeanStd:
    """
    Welford online mean/variance estimator.
    shape: shape of a single sample (tuple).
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # small init to avoid div-by-zero on first update

    def update(self, x: np.ndarray):
        """x: array of shape (batch, *shape) or (*shape,)."""
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis]  # add batch dim
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"].copy()
        self.var = d["var"].copy()
        self.count = d["count"]


class RewardNormalizer:
    """
    Normalizes a stream of intrinsic rewards.
    Divides by running std only — never subtract mean (non-stationary rewards).
    Shape: () for scalar rewards per env.
    """

    def __init__(self):
        self.rms = RunningMeanStd(shape=())

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """rewards: (n_envs,) float32 → normalized float32."""
        self.rms.update(rewards)
        return (rewards / self.rms.std).astype(np.float32)

    def state_dict(self):
        return self.rms.state_dict()

    def load_state_dict(self, d):
        self.rms.load_state_dict(d)


class ObsNormalizer:
    """
    RND-specific observation normalizer.
    Normalizes obs to zero mean / unit variance, then clips to [-5, 5].
    shape: shape of a single obs, e.g. (3, 7, 7) (channel-first).
    """

    def __init__(self, shape):
        self.rms = RunningMeanStd(shape=shape)

    def update(self, obs: np.ndarray):
        """obs: (batch, *shape) float32 in [0.0, 1.0]."""
        self.rms.update(obs.astype(np.float64))

    def normalize(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        obs_tensor: (B, *shape) float32 on any device.
        Returns normalized tensor on the same device.
        """
        mean = torch.tensor(self.rms.mean, dtype=torch.float32, device=obs_tensor.device)
        std = torch.tensor(self.rms.std, dtype=torch.float32, device=obs_tensor.device)
        normed = (obs_tensor - mean) / std
        return normed.clamp(-5.0, 5.0)

    def state_dict(self):
        return self.rms.state_dict()

    def load_state_dict(self, d):
        self.rms.load_state_dict(d)
