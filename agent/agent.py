"""
agent/agent.py

MiniGridCNN  — shared architecture, instantiated independently per module.
Agent        — CNN + LSTM + actor + dual value heads.

All 5 LSTM implementation details from Section 4:
  1. Orthogonal init on LSTM weights, zero biases.
  2. Zero-init hidden state, shape (1, B, hidden).
  3. Reset h,c on done BEFORE LSTM call.
  4. Store h,c BEFORE each step (state going INTO the step).
  5. During PPO update, replay full sequences from stored h0,c0.
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MiniGridCNN(nn.Module):
    """
    Input:  (B, 3, 7, 7) float32, values in [0.0, 1.0]
    Output: (B, embed_dim)

    Three stride-2 conv layers: 7→4→2→1, so 32×1×1=32 before Linear.
    BatchNorm2d on every conv. NO BatchNorm on Linear.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),               # (B, 32*1*1) = (B, 32)
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent(nn.Module):
    """
    CNN + LSTM backbone with actor + dual critic heads.
    critic_int is only instantiated when use_dual_heads=True.
    """

    def __init__(self, cfg, n_actions: int, use_dual_heads: bool = False):
        super().__init__()
        self.cfg = cfg
        self.n_actions = n_actions
        self.use_dual_heads = use_dual_heads

        embed_dim = cfg.agent.embed_dim          # 64
        hidden = cfg.agent.lstm_hidden_size      # 128

        # Each module owns its own CNN — never shared (Constraint 18)
        self.cnn = MiniGridCNN(embed_dim=embed_dim)

        # LSTM: num_layers=1, batch_first=True
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )

        # Heads
        self.actor = nn.Linear(hidden, n_actions)
        self.critic_ext = nn.Linear(hidden, 1)
        if use_dual_heads:
            self.critic_int = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        # Conv + hidden Linear: orthogonal(sqrt(2)) — handled in MiniGridCNN
        # Actor head: orthogonal(0.01)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        # Critic heads: orthogonal(1.0)
        nn.init.orthogonal_(self.critic_ext.weight, gain=1.0)
        nn.init.constant_(self.critic_ext.bias, 0)
        if self.use_dual_heads:
            nn.init.orthogonal_(self.critic_int.weight, gain=1.0)
            nn.init.constant_(self.critic_int.bias, 0)

        # DETAIL 1 — LSTM weights: orthogonal(1.0), biases: 0
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)

    def initial_lstm_state(self, n_envs: int, device: torch.device):
        """
        DETAIL 2 — Zero-init hidden state.
        Shape: (num_layers=1, B, hidden).
        """
        hidden = self.cfg.agent.lstm_hidden_size
        h = torch.zeros(1, n_envs, hidden, device=device)
        c = torch.zeros(1, n_envs, hidden, device=device)
        return (h, c)

    def forward(
        self,
        obs: torch.Tensor,
        lstm_state: tuple,
        done: torch.Tensor,
    ):
        """
        obs:        (B, 7, 7, 3) uint8
        lstm_state: ((1, B, hidden), (1, B, hidden))
        done:       (B,) bool  — terminated | truncated

        Returns: feat (B, hidden), new_lstm_state
        """
        # Normalize + reorder channels (Constraint 2)
        x = obs.float() / 10.0                  # [0, 1]
        x = x.permute(0, 3, 1, 2)               # (B, 3, 7, 7)
        feat = self.cnn(x)                       # (B, embed_dim)
        feat = feat.unsqueeze(1)                 # (B, 1, embed_dim) for LSTM

        # DETAIL 3 — reset hidden state on done BEFORE LSTM
        h, c = lstm_state
        mask = (1.0 - done.float()).view(1, -1, 1)
        h = h * mask
        c = c * mask

        out, (h_new, c_new) = self.lstm(feat, (h, c))
        out = out.squeeze(1)                     # (B, hidden)
        return out, (h_new, c_new)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        lstm_state: tuple,
        done: torch.Tensor,
        action: torch.Tensor = None,
    ):
        """
        Full forward pass returning everything needed for rollout + update.
        Returns:
          action, log_prob, entropy, value_ext, value_int (or None), new_lstm_state
        """
        feat, lstm_state_new = self.forward(obs, lstm_state, done)
        logits = self.actor(feat)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value_ext = self.critic_ext(feat)
        value_int = self.critic_int(feat) if self.use_dual_heads else None
        return action, log_prob, entropy, value_ext, value_int, lstm_state_new

    def get_value(self, obs: torch.Tensor, lstm_state: tuple, done: torch.Tensor):
        """Returns (value_ext, value_int or None) without sampling action."""
        feat, _ = self.forward(obs, lstm_state, done)
        value_ext = self.critic_ext(feat)
        value_int = self.critic_int(feat) if self.use_dual_heads else None
        return value_ext, value_int
