"""
curiosity/factory.py

build_curiosity(cfg, n_actions, device) → module or None.

"""

import torch
from .icm import ICMModule
from .rnd import RNDModule
from .noveld import NovelDModule


def build_curiosity(cfg, n_actions: int, device: torch.device):
    """
    Returns:
      ICMModule   | method == "icm"
      RNDModule   | method == "rnd"
      NovelDModule| method == "noveld"
      None        | method == "none"
    """
    method = cfg.curiosity.method
    if method == "none":
        return None
    elif method == "icm":
        return ICMModule(cfg, n_actions, device)
    elif method == "rnd":
        return RNDModule(cfg, device)
    elif method == "noveld":
        return NovelDModule(cfg, device)
    # [PHASE3-HOOK] elif method == "lewm_noveld": ...
    else:
        raise ValueError(f"Unknown curiosity method: {method!r}")
