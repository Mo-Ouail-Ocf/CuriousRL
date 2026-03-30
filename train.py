"""
train.py

Main training entry point for the MiniGrid Curiosity Benchmark.

— Vanilla PPO-LSTM | PPO-LSTM+ICM | PPO-LSTM+RND | PPO-LSTM+NovelD

Usage:
  python train.py                             # uses configs/config.yaml
  python train.py env=doorkey                 # merges doorkey.yaml
  python train.py curiosity.method=noveld     # override single key
  python train.py env=multiroom curiosity.method=icm
"""

import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm

# Hydra for config management
import hydra
from omegaconf import DictConfig, OmegaConf

from envs.wrappers import make_envs
from agent.agent import Agent
from curiosity.factory import build_curiosity
from ppo.rollout import RolloutBuffer
from ppo.gae import compute_gae
from ppo.update import ppo_update
from utils.logger import Logger
from curiosity.noveld import NovelDModule, RNDModule


# ─────────────────────────────────────────────────────────────────────────────
# Seeding (Section 9)
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, agent, optimizer, curiosity_module, global_step, cfg):
    payload = {
        "global_step": global_step,
        "model_weights": agent.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    # Save normalizer stats (Constraint 14 / Constraint 11)
    if isinstance(curiosity_module, NovelDModule):
        payload["normalizer_stats"] = curiosity_module.state_dict_extra()
    elif isinstance(curiosity_module, RNDModule):
        payload["normalizer_stats"] = curiosity_module.state_dict_extra()
    torch.save(payload, path)


def load_checkpoint(path, agent, optimizer, curiosity_module):
    ckpt = torch.load(path, map_location="cpu")
    agent.load_state_dict(ckpt["model_weights"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "normalizer_stats" in ckpt:
        if isinstance(curiosity_module, NovelDModule):
            curiosity_module.load_state_dict_extra(ckpt["normalizer_stats"])
        elif isinstance(curiosity_module, RNDModule):
            curiosity_module.load_state_dict_extra(ckpt["normalizer_stats"])
    return ckpt.get("global_step", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup loop
# ─────────────────────────────────────────────────────────────────────────────

def run_warmup(envs, curiosity_module, cfg, device):
    """
    Random-policy warmup to pre-populate RND obs normalizer (Constraint 12).
    No network parameter updates during warmup.
    [PHASE3-HOOK] also runs LeWM warmup if lewm.enabled
    """
    method = cfg.curiosity.method
    if method not in ("rnd", "noveld"):
        return  # warmup only needed for RND-based methods

    warmup_steps = cfg.rnd.warmup_steps
    n_envs = cfg.env.n_envs
    obs, _ = envs.reset()

    print(f"[Warmup] Running {warmup_steps} random steps to pre-populate obs normalizer...")
    for step in tqdm(range(warmup_steps), desc="Warmup"):
        actions = np.array([envs.single_action_space.sample() for _ in range(n_envs)])
        obs_next, _, terminated, truncated, _ = envs.step(actions)

        # Update obs normalizer only — no network update
        if isinstance(curiosity_module, NovelDModule):
            curiosity_module.update_obs_normalizer(obs)
        elif isinstance(curiosity_module, RNDModule):
            curiosity_module.update_obs_normalizer(obs)

        dones = terminated | truncated
        obs = obs_next
        if dones.any():
            obs_reset, _ = envs.reset()
            obs[dones] = obs_reset[dones]

        # [PHASE3-HOOK] if lewm.enabled: lewm.warmup_step(obs, actions, obs_next)

    print("[Warmup] Done.")
    envs.reset()  # fresh reset before main loop


# ─────────────────────────────────────────────────────────────────────────────
# Episode stats tracker
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeTracker:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.ep_returns = np.zeros(n_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(n_envs, dtype=np.int32)
        self.completed_returns = []
        self.completed_lengths = []
        self.completed_successes = []  # reward > 0

    def update(self, rewards_ext, dones):
        self.ep_returns += rewards_ext
        self.ep_lengths += 1
        for i in range(self.n_envs):
            if dones[i]:
                self.completed_returns.append(self.ep_returns[i])
                self.completed_lengths.append(self.ep_lengths[i])
                self.completed_successes.append(float(self.ep_returns[i] > 0))
                self.ep_returns[i] = 0.0
                self.ep_lengths[i] = 0

    def flush(self) -> dict:
        if not self.completed_returns:
            return {}
        stats = {
            "charts/episodic_return": np.mean(self.completed_returns),
            "charts/episodic_length": np.mean(self.completed_lengths),
            "charts/success_rate": np.mean(self.completed_successes),
        }
        self.completed_returns.clear()
        self.completed_lengths.clear()
        self.completed_successes.clear()
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Step-metric accumulator
# ─────────────────────────────────────────────────────────────────────────────

class StepMetricAccumulator:
    """
    Accumulates scalar dicts emitted by curiosity_module.step() across all
    rollout steps, then returns the per-key mean when flush() is called.

    Previously curiosity_step_metrics was overwritten each step, so only the
    last step's values (n_envs=16 samples out of n_steps*n_envs=2048) were
    ever logged — making E_mean, r_int_raw_mean, etc. unreliable.
    """

    def __init__(self):
        self._sums: dict = {}
        self._count: int = 0

    def update(self, metrics: dict):
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v)
        self._count += 1

    def flush(self) -> dict:
        if self._count == 0:
            return {}
        out = {k: v / self._count for k, v in self._sums.items()}
        self._sums.clear()
        self._count = 0
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.run.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Envs ──────────────────────────────────────────────────────────────
    envs = make_envs(cfg)
    n_actions = envs.single_action_space.n

    # ── Run name & logger ─────────────────────────────────────────────────
    run_name = (
        f"{cfg.env.name}__{cfg.curiosity.method}"
        f"__seed{cfg.run.seed}__{int(time.time())}"
    )
    logger = Logger(cfg, run_name)
    ckpt_dir = os.path.join(cfg.run.log_dir, run_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Agent ─────────────────────────────────────────────────────────────
    use_dual = cfg.curiosity.method != "none"
    agent = Agent(cfg, n_actions, use_dual_heads=use_dual).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.ppo.lr, eps=1e-5)

    # ── Curiosity module ──────────────────────────────────────────────────
    curiosity_module = build_curiosity(cfg, n_actions, device)

    # ── Rollout buffer ─────────────────────────────────────────────────────
    buffer = RolloutBuffer(cfg, cfg.agent.lstm_hidden_size, device)

    # ── Warmup (RND obs normalizer pre-population) ─────────────────────────
    run_warmup(envs, curiosity_module, cfg, device)

    # ── Training state ────────────────────────────────────────────────────
    total_timesteps = cfg.ppo.total_timesteps
    n_steps = cfg.ppo.n_steps
    n_envs = cfg.env.n_envs
    batch_size = n_steps * n_envs
    num_updates = total_timesteps // batch_size

    obs, _ = envs.reset()
    lstm_state = agent.initial_lstm_state(n_envs, device)
    done = np.zeros(n_envs, dtype=bool)

    ep_tracker = EpisodeTracker(n_envs)
    step_metric_acc = StepMetricAccumulator()   # ← accumulate across steps
    global_step = 0
    start_time = time.time()

    print(f"\n[Train] Starting: {num_updates} updates × {batch_size} steps = {total_timesteps} total")

    for update in tqdm(range(1, num_updates + 1), desc="Training"):
        # ── Rollout ───────────────────────────────────────────────────────
        for step in range(n_steps):
            global_step += n_envs

            obs_t = torch.tensor(obs, device=device)
            done_t = torch.tensor(done, device=device, dtype=torch.bool)

            # Store LSTM state BEFORE step (Detail 4)
            h_store = lstm_state[0].squeeze(0).detach().cpu().numpy()  # (n_envs, hidden)
            c_store = lstm_state[1].squeeze(0).detach().cpu().numpy()

            with torch.no_grad():
                action, log_prob, _, val_ext, val_int, lstm_state_new = \
                    agent.get_action_and_value(obs_t, lstm_state, done_t)

            actions_np = action.cpu().numpy()
            obs_next, rewards_ext, terminated, truncated, infos = envs.step(actions_np)
            dones = terminated | truncated

            # ── Curiosity ─────────────────────────────────────────────────
            # Unified interface: train.py never branches on method.
            if curiosity_module is None:
                rewards_int = np.zeros(n_envs, dtype=np.float32)
                step_metrics = {}
            else:
                obs_t1 = torch.tensor(obs_next, device=device)
                rewards_int, step_metrics = curiosity_module.step(
                    obs_t, action, obs_t1, infos, dones
                )

            # Accumulate step metrics across the whole rollout (fix: was
            # overwriting so only the last step was ever logged)
            step_metric_acc.update(step_metrics)

            # ── Buffer ────────────────────────────────────────────────────
            buffer.add(
                step=step,
                obs=obs,
                actions=actions_np,
                log_probs=log_prob.cpu().numpy(),
                rewards_ext=rewards_ext.astype(np.float32),
                rewards_int=rewards_int,
                terminated=terminated,
                truncated=truncated,
                values_ext=val_ext,
                values_int=val_int,
                lstm_h=h_store,
                lstm_c=c_store,
            )

            ep_tracker.update(rewards_ext, dones)

            # DETAIL 3: LSTM reset on done (handled inside agent.forward)
            lstm_state = lstm_state_new

            obs = obs_next
            done = dones

        # ── GAE ───────────────────────────────────────────────────────────
        compute_gae(
            buffer, agent,
            next_obs=obs,
            next_lstm_state=lstm_state,
            next_done=done,
            cfg=cfg,
            device=device,
        )

        # ── Curiosity update ──────────────────────────────────────────────
        curiosity_metrics = {}
        if curiosity_module is not None:
            obs_all = torch.tensor(
                buffer.obs.reshape(-1, 7, 7, 3), device=device
            )
            act_all = torch.tensor(
                buffer.actions.reshape(-1), device=device, dtype=torch.long
            )
            # obs_t1: shift buffer.obs by 1 step, use current obs as last t+1.
            # This is an approximation (exact t+1 not stored separately in buffer).
            obs_t1_np = np.concatenate(
                [buffer.obs[1:], obs[np.newaxis]], axis=0
            ).reshape(-1, 7, 7, 3)
            obs_t1_all = torch.tensor(obs_t1_np, device=device)

            curiosity_metrics = curiosity_module.update(obs_all, act_all, obs_t1_all)

            # [PHASE3-HOOK] lewm.update(replay_buffer.sample())

        # ── PPO update ────────────────────────────────────────────────────
        ppo_metrics = ppo_update(
            agent, optimizer, buffer, cfg, device,
            global_step=global_step,
            total_steps=total_timesteps,
        )

        # ── Logging ───────────────────────────────────────────────────────
        if update % cfg.logging.log_freq == 0:
            sps = int(global_step / (time.time() - start_time))
            log_metrics = {
                "charts/steps_per_second": sps,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "rewards/r_ext_mean": float(buffer.rewards_ext.mean()),
                "rewards/r_ext_std": float(buffer.rewards_ext.std()),
                "rewards/r_int_norm_mean": float(buffer.rewards_int.mean()),
                "rewards/r_int_norm_std": float(buffer.rewards_int.std()),
            }
            log_metrics.update(ppo_metrics)
            log_metrics.update(curiosity_metrics)
            # Flush the accumulated per-step metrics (mean over full rollout)
            log_metrics.update(step_metric_acc.flush())

            ep_stats = ep_tracker.flush()
            log_metrics.update(ep_stats)

            logger.log(log_metrics, global_step)

            if update % cfg.logging.print_freq == 0:
                sr = ep_stats.get("charts/success_rate", float("nan"))
                ret = ep_stats.get("charts/episodic_return", float("nan"))
                print(
                    f"[{global_step:>9d}] update={update:>5d} | "
                    f"ret={ret:.3f} | success={sr:.3f} | "
                    f"sps={sps} | "
                    f"kl={ppo_metrics.get('losses/approx_kl', 0):.4f}"
                )

        # ── Checkpoint ────────────────────────────────────────────────────
        if update % cfg.checkpoint.save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{global_step}.pt")
            save_checkpoint(ckpt_path, agent, optimizer, curiosity_module, global_step, cfg)

    # ── Final checkpoint ──────────────────────────────────────────────────
    save_checkpoint(
        os.path.join(ckpt_dir, "final.pt"),
        agent, optimizer, curiosity_module, global_step, cfg
    )

    logger.close()
    envs.close()
    print(f"\n[Done] {global_step} steps in {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()