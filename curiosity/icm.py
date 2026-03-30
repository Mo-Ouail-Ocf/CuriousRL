"""
curiosity/icm.py

Intrinsic Curiosity Module (Pathak et al., 2017).

Modules (all own independent weights — Constraint 18):
  encoder:       MiniGridCNN(embed_dim=feature_dim)   owns weights
  inverse_model: Linear(2*feature_dim, n_actions)
  forward_model: Linear(feature_dim + n_actions, feature_dim)

compute_surprise uses obs_t, action_t, obs_t1 (transition-based).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import CuriosityModule
from agent.agent import MiniGridCNN


def _preprocess(obs: Tensor) -> Tensor:
    """uint8 (B,7,7,3) → float32 (B,3,7,7) in [0,1]."""
    return obs.float().div(10.0).permute(0, 3, 1, 2)


class ICMModule(CuriosityModule):

    def __init__(self, cfg, n_actions: int, device: torch.device):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device

        feature_dim = cfg.icm.feature_dim

        # Own encoder — never shared
        self.encoder = MiniGridCNN(embed_dim=feature_dim).to(device)

        # Inverse model: [φ_t || φ_t1] → action logits
        self.inverse_model = nn.Linear(2 * feature_dim, n_actions).to(device)
        nn.init.orthogonal_(self.inverse_model.weight, gain=1.0)
        nn.init.constant_(self.inverse_model.bias, 0)

        # Forward model: [φ_t || one_hot(a)] → φ_t1 prediction
        self.forward_model = nn.Linear(feature_dim + n_actions, feature_dim).to(device)
        nn.init.orthogonal_(self.forward_model.weight, gain=1.0)
        nn.init.constant_(self.forward_model.bias, 0)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.inverse_model.parameters())
            + list(self.forward_model.parameters()),
            lr=cfg.icm.lr,
        )

    @torch.no_grad()
    def compute_surprise(
        self,
        obs_t: Tensor,
        action_t: Tensor,
        obs_t1: Tensor,
    ) -> Tensor:
        """Returns forward-model prediction error per sample: (B,) float32."""
        phi_t = self.encoder(_preprocess(obs_t.to(self.device)))
        phi_t1 = self.encoder(_preprocess(obs_t1.to(self.device)))
        act_oh = F.one_hot(action_t.to(self.device), self.n_actions).float()
        phi_pred = self.forward_model(torch.cat([phi_t, act_oh], dim=-1))
        # Per-sample MSE
        return ((phi_pred - phi_t1) ** 2).mean(dim=-1)

    def update(
        self,
        obs_t: Tensor,
        action_t: Tensor,
        obs_t1: Tensor,
    ) -> dict:
        """Trains encoder (via inverse loss) and forward model."""
        phi_t = self.encoder(_preprocess(obs_t.to(self.device)))
        phi_t1 = self.encoder(_preprocess(obs_t1.to(self.device)))
        act = action_t.to(self.device)
        act_oh = F.one_hot(act, self.n_actions).float()

        # Inverse loss — trains encoder
        inv_logits = self.inverse_model(torch.cat([phi_t, phi_t1], dim=-1))
        loss_inv = F.cross_entropy(inv_logits, act)

        # Forward loss — detach target to avoid gradient into encoder from here
        phi_pred = self.forward_model(torch.cat([phi_t, act_oh], dim=-1))
        loss_fwd = F.mse_loss(phi_pred, phi_t1.detach())

        loss = (
            self.cfg.icm.inverse_weight * loss_inv
            + self.cfg.icm.forward_weight * loss_fwd
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "curiosity/icm_loss": loss.item(),
            "curiosity/icm_inv_loss": loss_inv.item(),
            "curiosity/icm_fwd_loss": loss_fwd.item(),
        }
