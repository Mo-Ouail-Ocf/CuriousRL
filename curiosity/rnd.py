"""
curiosity/rnd.py  —  Random Network Distillation (Burda et al., 2018)

Verified against CleanRL ppo_rnd_envpool.py:
  - Predictor trains on obs_t1 (next obs), NOT obs_t
  - update_proportion applied as random mask per minibatch (not randperm slice)
  - obs normalizer updated on full batch before minibatch loop
  - reward_normalizer included in checkpoint (state_dict_extra)
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base import CuriosityModule
from agent.agent import MiniGridCNN
from utils.running_stats import RewardNormalizer, ObsNormalizer


def _to_chw(obs: Tensor) -> Tensor:
    """uint8 (B,7,7,3) -> float32 (B,3,7,7) in [0,1]."""
    return obs.float().div(10.0).permute(0, 3, 1, 2)


class RNDModule(CuriosityModule):

    def __init__(self, cfg, device: torch.device):
        self.cfg    = cfg
        self.device = device
        out_dim     = cfg.rnd.output_dim

        # Frozen target
        self.target_net = MiniGridCNN(embed_dim=out_dim).to(device)
        self.target_net.requires_grad_(False)
        self.target_net.eval()

        # Predictor — two-layer MLP head on top of shared CNN,
        # matching the depth used in CleanRL's RNDModel.predictor
        self.predictor_net = nn.Sequential(
            MiniGridCNN(embed_dim=128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        ).to(device)
        nn.init.orthogonal_(self.predictor_net[2].weight, gain=1.0)
        nn.init.constant_(self.predictor_net[2].bias, 0)

        self.optimizer         = torch.optim.Adam(self.predictor_net.parameters(), lr=cfg.rnd.lr)
        self.obs_normalizer    = ObsNormalizer(shape=(3, 7, 7))
        self.reward_normalizer = RewardNormalizer()

    # ------------------------------------------------------------------ #

    def _normalize_obs(self, obs: Tensor) -> Tensor:
        """obs: (B,3,7,7) float [0,1] -> normalized, clipped [-5,5]."""
        return self.obs_normalizer.normalize(obs)

    def _raw_surprise(self, obs_t1: Tensor) -> np.ndarray:
        """Compute per-env surprise on NEXT obs (obs_t1)."""
        obs_f   = _to_chw(obs_t1.to(self.device))
        obs_rnd = self._normalize_obs(obs_f)
        with torch.no_grad():
            target = self.target_net(obs_rnd)
            pred   = self.predictor_net(obs_rnd)
        return ((pred - target) ** 2).mean(dim=-1).cpu().numpy()

    # ------------------------------------------------------------------ #

    def step(self, obs_t, action, obs_t1, infos, dones):
        # Update obs normalizer on next obs BEFORE computing surprise
        obs_f_np = obs_t1.cpu().numpy().astype(np.float32) / 10.0
        obs_f_np = obs_f_np.transpose(0, 3, 1, 2)          # (B,3,7,7)
        self.obs_normalizer.update(obs_f_np)

        surprise    = self._raw_surprise(obs_t1)            # uses obs_t1 ✓
        rewards_int = self.reward_normalizer.normalize(surprise)
        metrics = {
            "curiosity/E_mean":       float(surprise.mean()),
            "curiosity/E_std":        float(surprise.std()),
            "rewards/r_int_raw_mean": float(surprise.mean()),
        }
        return rewards_int, metrics

    def update(self, obs_t, action, obs_t1):
        # ── Normalizer update on full batch first (CleanRL pattern) ──────
        obs_f_np = obs_t1.cpu().numpy().astype(np.float32) / 10.0
        obs_f_np = obs_f_np.transpose(0, 3, 1, 2)
        self.obs_normalizer.update(obs_f_np)

        # ── Train predictor on obs_t1, NOT obs_t ─────────────────────────
        obs_f   = _to_chw(obs_t1.to(self.device))          # ← fixed: was obs_t
        obs_rnd = self._normalize_obs(obs_f)

        B = obs_rnd.shape[0]

        # CleanRL-style random mask: each sample independently kept with
        # probability update_proportion, rather than a hard randperm slice.
        # This avoids correlation between which samples are selected across
        # consecutive update calls.
        with torch.no_grad():
            target = self.target_net(obs_rnd)
        pred        = self.predictor_net(obs_rnd)
        per_sample  = nn.functional.mse_loss(pred, target, reduction="none").mean(dim=-1)

        mask = (torch.rand(B, device=self.device) < self.cfg.rnd.update_proportion).float()
        # Guard: if mask is all-zero (extremely unlikely), fall back to full batch
        denom = mask.sum().clamp(min=1.0)
        loss  = (per_sample * mask).sum() / denom

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"curiosity/rnd_loss": loss.item()}

    # ------------------------------------------------------------------ #

    def state_dict_extra(self):
        return {
            "obs_normalizer":    self.obs_normalizer.state_dict(),
            "reward_normalizer": self.reward_normalizer.state_dict(),  # ← added
        }

    def load_state_dict_extra(self, d):
        self.obs_normalizer.load_state_dict(d["obs_normalizer"])
        if "reward_normalizer" in d:                                   # back-compat
            self.reward_normalizer.load_state_dict(d["reward_normalizer"])

    def update_obs_normalizer(self, obs: np.ndarray):
        obs_f_np = obs.astype(np.float32) / 10.0
        obs_f_np = obs_f_np.transpose(0, 3, 1, 2)
        self.obs_normalizer.update(obs_f_np)