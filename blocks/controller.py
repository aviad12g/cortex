"""
Neuromodulator controller that drives fast write gates and sleep scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class ControllerConfig:
    d_model: int
    hidden: int = 128


class CortexController(nn.Module):
    """Compute neuromodulator signals from surprise, uncertainty, and rhythm inputs."""

    is_cortex_param = True

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(4, cfg.hidden),
            nn.Tanh(),
            nn.Linear(cfg.hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: dict with keys surprise, uncertainty, reward, phase each [B, 1].
        Returns:
            dict with keys m_gate (global plasticity) and write_scale.
        """
        stacked = torch.cat(
            [inputs[k] for k in ("surprise", "uncertainty", "reward", "phase")], dim=-1
        )
        out = self.net(stacked)
        return {"m_gate": out[..., :1], "write_scale": out[..., 1:]}

