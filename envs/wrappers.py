"""
envs/wrappers.py

Wrapper stack (applied in order):
  1. minigrid.wrappers.ImgObsWrapper   — extracts "image" key → (7,7,3) uint8
  2. GlobalPosWrapper                  — injects agent_pos, agent_dir into info
  3. gymnasium.wrappers.TimeLimit      — per-env step budget

Vectorization: SyncVectorEnv (IPC overhead > parallelism gain for MiniGrid).
"""

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper


class GlobalPosWrapper(gym.Wrapper):
    """
    Injects agent_pos and agent_dir into the info dict on every step and reset.
    WHY: NovelD episodic counts must use global (x, y, dir), not the egocentric
    7×7 view. The egocentric view aliases heavily — the same grey wall looks
    identical in every room. Global position is unique, unambiguous, and free.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["agent_pos"] = tuple(self.env.unwrapped.agent_pos)
        info["agent_dir"] = int(self.env.unwrapped.agent_dir)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["agent_pos"] = tuple(self.env.unwrapped.agent_pos)
        info["agent_dir"] = int(self.env.unwrapped.agent_dir)
        return obs, info


def make_env(env_name: str, seed: int, max_steps: int):
    """Factory returning a thunk (callable) for SyncVectorEnv."""

    def thunk():
        env = gym.make(env_name)
        env = ImgObsWrapper(env)          # (7,7,3) uint8, values in [0,10]
        env = GlobalPosWrapper(env)       # inject global pos into info
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env.reset(seed=seed)
        return env

    return thunk


def make_envs(cfg) -> gym.vector.SyncVectorEnv:
    """
    Build a SyncVectorEnv of n_envs instances.
    Each sub-env gets seed = cfg.run.seed + rank.
    """
    env_fns = [
        make_env(
            env_name=cfg.env.name,
            seed=cfg.run.seed + i,
            max_steps=cfg.env.max_steps,
        )
        for i in range(cfg.env.n_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)
