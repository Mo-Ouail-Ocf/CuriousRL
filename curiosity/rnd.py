"""
curiosity/rnd.py

Random Network Distillation (Burda et al., 2018).

Modules:
  target_net:    MiniGridCNN(output_dim) → FROZEN immediately after init
  predictor_net: MiniGridCNN(128) → Linear(128, output_dim)
  obs_normalizer: ObsNormalizer(shape=(3,7,7))  ← RND-specific, not shared

Key design choices (Section 6.2):
  - compute_surprise takes obs_t ONLY (state novelty, not transition novelty)
  - obs normalizer updated BEFORE compute_surprise during rollout
  - predictor trained on random 25% subset to avoid overfitting on-policy dist
  - warmup_steps random-policy rollout to pre-populate Welford stats
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base import CuriosityModule
from agent.agent import MiniGridCNN
from utils.running_stats import ObsNormalizer


def _to_float_chw(obs: Tensor) -> Tensor:
    """uint8 (B,7,7,3) → float32 (B,3,7,7) in [0,1]."""
    return obs.float().div(10.0).permute(0, 3, 1, 2)


class RNDModule(CuriosityModule):

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        output_dim = cfg.rnd.output_dim

        # Target network — frozen immediately (Constraint 18, Section 6.2)
        self.target_net = MiniGridCNN(embed_dim=output_dim).to(device)
        self.target_net.requires_grad_(False)

        # Predictor: CNN(128) → Linear(128, output_dim)
        self.predictor_net = nn.Sequential(
            MiniGridCNN(embed_dim=128),
            nn.Linear(128, output_dim),
        ).to(device)
        nn.init.orthogonal_(self.predictor_net[1].weight, gain=1.0)
        nn.init.constant_(self.predictor_net[1].bias, 0)

        # RND-specific obs normalizer — (3,7,7) channel-first
        self.obs_normalizer = ObsNormalizer(shape=(3, 7, 7))

        self.optimizer = torch.optim.Adam(
            self.predictor_net.parameters(), lr=cfg.rnd.lr
        )

    # ------------------------------------------------------------------
    # Obs normalizer
    # ------------------------------------------------------------------

    def update_obs_normalizer(self, obs: np.ndarray):
        """
        obs: (B, 7, 7, 3) uint8 numpy.
        Updates Welford stats. Call during rollout BEFORE compute_surprise.
        """
        obs_f = obs.astype(np.float32) / 10.0         # [0, 1]
        obs_f = obs_f.transpose(0, 3, 1, 2)            # (B, 3, 7, 7)
        self.obs_normalizer.update(obs_f)

    def _normalize_obs(self, obs: Tensor) -> Tensor:
        """obs: (B, 3, 7, 7) float32 [0,1] → normalized, clipped to [-5, 5]."""
        return self.obs_normalizer.normalize(obs)

    # ------------------------------------------------------------------
    # CuriosityModule interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_surprise(
        self,
        obs_t: Tensor,
        action_t: Tensor = None,    # ignored
        obs_t1: Tensor = None,      # ignored — RND is state-only
    ) -> Tensor:
        """
        Returns per-sample RND error: (B,) float32.
        Uses obs_t ONLY (Constraint 10).
        """
        obs_f = _to_float_chw(obs_t.to(self.device))
        obs_rnd = self._normalize_obs(obs_f)
        target_feat = self.target_net(obs_rnd)
        pred_feat = self.predictor_net(obs_rnd)
        return ((pred_feat - target_feat) ** 2).mean(dim=-1)

    def update(
        self,
        obs_t: Tensor,
        action_t: Tensor = None,
        obs_t1: Tensor = None,
    ) -> dict:
        """
        Trains predictor on a random 25% subset of the rollout batch
        to avoid overfitting to the on-policy distribution.
        """
        obs_f = _to_float_chw(obs_t.to(self.device))
        B = obs_f.shape[0]
        n_update = max(1, int(B * self.cfg.rnd.update_proportion))
        idx = torch.randperm(B, device=self.device)[:n_update]
        obs_sub = obs_f[idx]
        obs_rnd = self._normalize_obs(obs_sub)

        with torch.no_grad():
            target_feat = self.target_net(obs_rnd)
        pred_feat = self.predictor_net(obs_rnd)
        loss = nn.functional.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"curiosity/rnd_loss": loss.item()}

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict_extra(self):
        """Returns obs normalizer stats for checkpointing (Constraint 11)."""
        return {"obs_normalizer": self.obs_normalizer.state_dict()}

    def load_state_dict_extra(self, d):
        self.obs_normalizer.load_state_dict(d["obs_normalizer"])
