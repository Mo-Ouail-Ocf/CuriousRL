"""
utils/logger.py

Thin wrapper over TensorBoard SummaryWriter with optional W&B integration.
"""

import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, cfg, run_name: str):
        log_path = os.path.join(cfg.logging.get("log_dir", cfg.run.log_dir), run_name)
        self.writer = SummaryWriter(log_path)
        self.use_wandb = cfg.run.use_wandb
        self._step = 0

        if self.use_wandb:
            import wandb
            wandb.init(
                project=cfg.run.wandb_project,
                name=run_name,
                config=dict(cfg),
                sync_tensorboard=True,
            )

    def log(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self):
        self.writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()
