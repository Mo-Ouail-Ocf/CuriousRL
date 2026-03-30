"""
curiosity/noveld.py

NovelD (Zhang et al., 2021) — wraps RNDModule with step-difference shaping
and episodic visit counts keyed on global (agent_pos, agent_dir).

NOT a CuriosityModule subclass — its compute_surprise signature is different.

Key design (Section 6.3):
  r_int = r_lifelong * r_episodic
  r_lifelong  = max(E_{t+1} - alpha * E_t, 0)     (step-difference)
  r_episodic  = 1 / sqrt(count(pos, dir))
  prev_E reset to 0.0 on episode done (not copied from E_next).

Global position hashing (WHY, Section 6.3):
  The 7×7 egocentric obs aliases across rooms — identical-looking grey walls.
  (x, y, dir) is unique, free, and injected by GlobalPosWrapper.
"""

import math
import numpy as np
import torch
from torch import Tensor
from typing import List

from .rnd import RNDModule
from utils.running_stats import RewardNormalizer


class NovelDModule:

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.alpha = cfg.noveld.alpha
        self.n_envs = cfg.env.n_envs

        # Owned RND instance
        self.rnd = RNDModule(cfg, device)

        # Per-env running state
        self.prev_E = np.zeros(self.n_envs, dtype=np.float32)
        self.episodic_counts: List[dict] = [{} for _ in range(self.n_envs)]

        # Intrinsic reward normalizer
        self.reward_normalizer = RewardNormalizer()

    # ------------------------------------------------------------------
    # Obs normalizer passthrough (called by train.py during rollout)
    # ------------------------------------------------------------------

    def update_obs_normalizer(self, obs: np.ndarray):
        self.rnd.update_obs_normalizer(obs)

    # ------------------------------------------------------------------
    # Intrinsic reward computation
    # ------------------------------------------------------------------

    def compute_surprise(
        self,
        obs_t1: Tensor,
        infos: list,
        dones: np.ndarray,
    ) -> np.ndarray:
        """
        Full NovelD intrinsic reward per env: (n_envs,) float32.

        Steps (Section 6.3):
          1. E_next = rnd.compute_surprise(obs_t1)
          2. r_lifelong = max(E_next - alpha * prev_E, 0)
          3. prev_E = E_next; prev_E[dones] = 0
          4. r_episodic = 1 / sqrt(count(pos, dir))
          5. r_int = r_lifelong * r_episodic
          6. normalize by std
        """
        # Step 1 — raw RND surprise on next state
        E_next = self.rnd.compute_surprise(obs_t1).cpu().numpy()  # (n_envs,)

        # Step 2 — lifelong reward (step-difference)
        r_lifelong = np.maximum(E_next - self.alpha * self.prev_E, 0.0)

        # Step 3 — advance prev_E; reset to 0 on done (NOT to E_next)
        self.prev_E = E_next.copy()
        self.prev_E[dones] = 0.0

        # Step 4 — episodic count
        r_episodic = np.ones(self.n_envs, dtype=np.float32)
        for i in range(self.n_envs):
            if dones[i]:
                self.episodic_counts[i] = {}

            pos = infos[i]["agent_pos"]    # (x, y)
            dir_ = infos[i]["agent_dir"]   # int
            key = (int(pos[0]), int(pos[1]), int(dir_))
            cnt = self.episodic_counts[i].get(key, 0) + 1
            self.episodic_counts[i][key] = cnt
            r_episodic[i] = 1.0 / math.sqrt(float(cnt))

        # Step 5 — combine
        r_int = r_lifelong * r_episodic

        # Step 6 — normalize by std only
        r_int_norm = self.reward_normalizer.normalize(r_int)
        return r_int_norm, {
            "curiosity/E_mean": float(E_next.mean()),
            "curiosity/E_std": float(E_next.std()),
            "curiosity/r_lifelong_mean": float(r_lifelong.mean()),
            "curiosity/r_episodic_mean": float(r_episodic.mean()),
            "curiosity/episodic_count_mean": float(
                np.mean([cnt for ec in self.episodic_counts for cnt in ec.values()])
                if any(ec for ec in self.episodic_counts)
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Update — delegates to RND
    # ------------------------------------------------------------------

    def update(self, obs_t: Tensor, action_t=None, obs_t1=None) -> dict:
        return self.rnd.update(obs_t)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict_extra(self):
        return {
            "rnd": self.rnd.state_dict_extra(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
            "prev_E": self.prev_E.copy(),
        }

    def load_state_dict_extra(self, d):
        self.rnd.load_state_dict_extra(d["rnd"])
        self.reward_normalizer.load_state_dict(d["reward_normalizer"])
        self.prev_E = d["prev_E"].copy()
