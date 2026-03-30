"""
curiosity/icm.py  —  Intrinsic Curiosity Module (Pathak et al., 2017)

  - inverse_model and forward_model are  proper MLPs with hidden layer + ReLU,
    matching the architecture in Pathak et al. and SB3-contrib CuriosityModule.
    Single Linear layers gave near-zero intrinsic reward after a few updates because
    a linear map cannot approximate nonlinear next-state dynamics.
  - forward_model target detach was already correct (phi_t1.detach()).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import CuriosityModule
from agent.agent import MiniGridCNN
from utils.running_stats import RewardNormalizer


def _preprocess(obs: Tensor) -> Tensor:
    """uint8 (B,7,7,3) -> float32 (B,3,7,7) in [0,1]."""
    return obs.float().div(10.0).permute(0, 3, 1, 2)


class ICMModule(CuriosityModule):

    def __init__(self, cfg, n_actions: int, device: torch.device):
        self.cfg       = cfg
        self.n_actions = n_actions
        self.device    = device
        feat_dim       = cfg.icm.feature_dim

        self.encoder = MiniGridCNN(embed_dim=feat_dim).to(device)

        # ── Inverse model: (phi_t || phi_t1) -> action logits ─────────────
        # Needs hidden layer — a single Linear collapses to a rank-limited
        # linear classifier and gives trivial gradients after warmup.
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        ).to(device)

        # ── Forward model: (phi_t || a_onehot) -> phi_t1_hat ──────────────
        # Same reasoning: must be nonlinear to approximate dynamics.
        self.forward_model = nn.Sequential(
            nn.Linear(feat_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, feat_dim),
        ).to(device)

        # Orthogonal init on output layers only
        for seq in (self.inverse_model, self.forward_model):
            nn.init.orthogonal_(seq[-1].weight, gain=1.0)
            nn.init.constant_(seq[-1].bias, 0)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.inverse_model.parameters())
            + list(self.forward_model.parameters()),
            lr=cfg.icm.lr,
        )
        self.reward_normalizer = RewardNormalizer()

    # ------------------------------------------------------------------ #

    def step(self, obs_t, action, obs_t1, infos, dones):
        with torch.no_grad():
            phi_t  = self.encoder(_preprocess(obs_t.to(self.device)))
            phi_t1 = self.encoder(_preprocess(obs_t1.to(self.device)))
            act_oh = F.one_hot(action.to(self.device), self.n_actions).float()
            phi_pred = self.forward_model(torch.cat([phi_t, act_oh], dim=-1))
            surprise = ((phi_pred - phi_t1) ** 2).mean(dim=-1).cpu().numpy()

        rewards_int = self.reward_normalizer.normalize(surprise)
        metrics = {
            "rewards/r_int_raw_mean": float(surprise.mean()),
        }
        return rewards_int, metrics

    def update(self, obs_t, action, obs_t1):
        phi_t  = self.encoder(_preprocess(obs_t.to(self.device)))
        phi_t1 = self.encoder(_preprocess(obs_t1.to(self.device)))
        act    = action.to(self.device)
        act_oh = F.one_hot(act, self.n_actions).float()

        # Inverse loss
        inv_logits = self.inverse_model(torch.cat([phi_t, phi_t1], dim=-1))
        loss_inv   = F.cross_entropy(inv_logits, act)

        # Forward loss — detach phi_t1 so encoder only gets gradient from inverse
        phi_pred = self.forward_model(torch.cat([phi_t, act_oh], dim=-1))
        loss_fwd = F.mse_loss(phi_pred, phi_t1.detach())

        loss = self.cfg.icm.inverse_weight * loss_inv + self.cfg.icm.forward_weight * loss_fwd

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "curiosity/icm_loss":     loss.item(),
            "curiosity/icm_inv_loss": loss_inv.item(),
            "curiosity/icm_fwd_loss": loss_fwd.item(),
        }