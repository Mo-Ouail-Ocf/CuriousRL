"""
curiosity/noveld.py  —  NovelD (Zhang et al., 2021)

Wraps RNDModule. Owns episodic counts, prev_E, reward normalization.

  1. prev_E reset ordering: zero done envs BEFORE the subtraction so that
     terminal transitions don't compute (E_new_ep_start - alpha*E_terminal).
     The NovelD paper defines r_lifelong = max(E(s_{t+1}) - alpha*E(s_t), 0);
     on done steps s_t is terminal and s_{t+1} is the reset state. Setting
     prev_E[done]=0 first makes the subtraction alpha*0=0, giving the pure
     E(reset_state) as the bonus — a clean fresh-episode signal.
  2. Config flags use_episodic_count and use_global_pos_hash are now wired up.
     - use_episodic_count=False: r_episodic = 1 (lifelong bonus only)
     - use_global_pos_hash=False: key = (pos_x, pos_y, dir) [default, with dir]
     - use_global_pos_hash=True:  key = (pos_x, pos_y)      [ignores direction]
       (despite the name, "global_pos_hash" in the original code means
        position-only hashing without direction — matches NovelD paper ablation)
"""

import math
import numpy as np
import torch
from torch import Tensor

from .base import CuriosityModule
from .rnd import RNDModule
from utils.running_stats import RewardNormalizer


class NovelDModule(CuriosityModule):

    def __init__(self, cfg, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.alpha  = cfg.noveld.alpha
        self.n_envs = cfg.env.n_envs

        # Config flags (were dead knobs before — now wired)
        self.use_episodic_count  = cfg.noveld.use_episodic_count
        self.use_global_pos_hash = cfg.noveld.use_global_pos_hash

        self.rnd               = RNDModule(cfg, device)
        self.prev_E            = None   # initialized lazily on first step
        self.episodic_counts   = None   # initialized lazily on first step
        self.reward_normalizer = RewardNormalizer()

    # ------------------------------------------------------------------ #

    def step(self, obs_t, action, obs_t1, infos, dones):
        n = obs_t1.shape[0]

        # Lazy init
        if self.prev_E is None:
            self.prev_E = np.zeros(n, dtype=np.float32)
        if self.episodic_counts is None:
            self.episodic_counts = [{} for _ in range(n)]

        # ── Obs normalizer update (delegates to inner RND) ────────────────
        obs_f_np = obs_t1.cpu().numpy().astype(np.float32) / 10.0
        obs_f_np = obs_f_np.transpose(0, 3, 1, 2)
        self.rnd.obs_normalizer.update(obs_f_np)

        # ── Raw RND surprise on obs_t1 ────────────────────────────────────
        E_next = self.rnd._raw_surprise(obs_t1)             # (n,)

        # ── Fix 1: reset prev_E for done envs BEFORE subtraction ──────────
        # On done steps obs_t1 is already the reset obs (vectorenv auto-reset).
        # Zero out prev_E so r_lifelong = max(E(s0_new_ep) - alpha*0, 0)
        # instead of measuring delta against the terminal state's surprise.
        self.prev_E[dones] = 0.0

        # ── Lifelong reward (NovelD step-difference) ──────────────────────
        r_lifelong = np.maximum(E_next - self.alpha * self.prev_E, 0.0)
        self.prev_E = E_next.copy()

        # ── Episodic count ────────────────────────────────────────────────
        if self.use_episodic_count:
            agent_positions = infos["agent_pos"]
            agent_dirs      = infos["agent_dir"]

            r_episodic = np.ones(n, dtype=np.float32)
            for i in range(n):
                if dones[i]:
                    self.episodic_counts[i] = {}

                # Fix 2: use_global_pos_hash controls whether direction is hashed
                if self.use_global_pos_hash:
                    # position-only key (direction-agnostic)
                    key = (int(agent_positions[i][0]), int(agent_positions[i][1]))
                else:
                    # full key including direction (default)
                    key = (
                        int(agent_positions[i][0]),
                        int(agent_positions[i][1]),
                        int(agent_dirs[i]),
                    )

                cnt = self.episodic_counts[i].get(key, 0) + 1
                self.episodic_counts[i][key] = cnt
                r_episodic[i] = 1.0 / math.sqrt(float(cnt))
        else:
            # Episodic count disabled — use lifelong bonus alone
            r_episodic = np.ones(n, dtype=np.float32)

        # ── Combine + normalize ───────────────────────────────────────────
        r_int       = r_lifelong * r_episodic
        rewards_int = self.reward_normalizer.normalize(r_int)

        all_counts = [c for ec in self.episodic_counts for c in ec.values()] \
                     if self.use_episodic_count else []
        metrics = {
            "curiosity/E_mean":              float(E_next.mean()),
            "curiosity/E_std":               float(E_next.std()),
            "curiosity/r_lifelong_mean":     float(r_lifelong.mean()),
            "curiosity/r_episodic_mean":     float(r_episodic.mean()),
            "curiosity/episodic_count_mean": float(np.mean(all_counts)) if all_counts else 0.0,
            "rewards/r_int_raw_mean":        float(r_int.mean()),
        }
        return rewards_int, metrics

    def update(self, obs_t, action, obs_t1):
        # Delegates fully to RND (which now correctly trains on obs_t1)
        return self.rnd.update(obs_t, action, obs_t1)

    # ------------------------------------------------------------------ #

    def state_dict_extra(self):
        return {
            "rnd":               self.rnd.state_dict_extra(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
            "prev_E":            self.prev_E.copy() if self.prev_E is not None
                                 else np.zeros(self.n_envs, dtype=np.float32),
        }

    def load_state_dict_extra(self, d):
        self.rnd.load_state_dict_extra(d["rnd"])
        self.reward_normalizer.load_state_dict(d["reward_normalizer"])
        self.prev_E = d["prev_E"].copy()

    def update_obs_normalizer(self, obs: np.ndarray):
        """Delegates to the inner RND (NovelD has no separate normalizer)."""
        self.rnd.update_obs_normalizer(obs)