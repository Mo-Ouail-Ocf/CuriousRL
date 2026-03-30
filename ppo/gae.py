"""
ppo/gae.py

Generalized Advantage Estimation (Schulman et al., 2016).

Dual-stream GAE (Section 5):
  Extrinsic: mask on `terminated` only — NOT `done`.
    When truncated=True but terminated=False, we bootstrap from V(s').
    When terminated=True, we do NOT bootstrap (true terminal).

  Intrinsic: NO masking at all (non-episodic curiosity).

Combined advantage:
  A = A_ext + lambda_int * A_int
"""

import numpy as np
import torch


def compute_gae(buffer, agent, next_obs, next_lstm_state, next_done, cfg, device):
    """
    Fills buffer.advantages_ext, advantages_int, returns_ext, returns_int.

    next_obs:         (n_envs, 7, 7, 3) uint8 — obs AFTER last rollout step
    next_lstm_state:  final LSTM state
    next_done:        (n_envs,) bool
    """
    n_steps = cfg.ppo.n_steps
    gamma_ext = cfg.curiosity.gamma_ext
    gamma_int = cfg.curiosity.gamma_int
    gae_lambda = cfg.ppo.gae_lambda
    lambda_int = cfg.curiosity.lambda_int
    use_dual = cfg.curiosity.method != "none"

    with torch.no_grad():
        next_obs_t = torch.tensor(next_obs, device=device)
        next_done_t = torch.tensor(next_done, device=device, dtype=torch.float32)
        next_val_ext, next_val_int = agent.get_value(
            next_obs_t, next_lstm_state, next_done_t
        )
        next_val_ext = next_val_ext.squeeze(-1).cpu().numpy()
        next_val_int = (
            next_val_int.squeeze(-1).cpu().numpy()
            if next_val_int is not None
            else np.zeros_like(next_val_ext)
        )

    gae_ext = np.zeros(cfg.env.n_envs, dtype=np.float32)
    gae_int = np.zeros(cfg.env.n_envs, dtype=np.float32)

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            v_ext_next = next_val_ext
            v_int_next = next_val_int
            terminated_next = np.zeros(cfg.env.n_envs, dtype=np.float32)
        else:
            v_ext_next = buffer.values_ext[t + 1]
            v_int_next = buffer.values_int[t + 1]
            terminated_next = buffer.terminated[t + 1].astype(np.float32)

        # Extrinsic: mask on terminated, NOT done (Section 5)
        not_terminated = 1.0 - buffer.terminated[t].astype(np.float32)
        delta_ext = (
            buffer.rewards_ext[t]
            + gamma_ext * v_ext_next * not_terminated
            - buffer.values_ext[t]
        )
        gae_ext = delta_ext + gamma_ext * gae_lambda * not_terminated * gae_ext

        # Intrinsic: NO masking (non-episodic, Section 5)
        delta_int = (
            buffer.rewards_int[t]
            + gamma_int * v_int_next
            - buffer.values_int[t]
        )
        gae_int = delta_int + gamma_int * gae_lambda * gae_int

        buffer.advantages_ext[t] = gae_ext
        buffer.advantages_int[t] = gae_int
        buffer.returns_ext[t] = gae_ext + buffer.values_ext[t]
        buffer.returns_int[t] = gae_int + buffer.values_int[t]

    # Combined advantage (used if dual heads)
    if use_dual:
        buffer.advantages_combined = (
            buffer.advantages_ext + lambda_int * buffer.advantages_int
        )
    else:
        buffer.advantages_combined = buffer.advantages_ext
