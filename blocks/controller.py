"""Neuromodulator - controls when/how much to write to fast weights."""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class ControllerConfig:
    d_model: int
    hidden: int = 128


class CortexController(nn.Module):

    is_cortex_param = True

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg
        # simple 2-layer MLP: 4 inputs -> 2 outputs
        self.net = nn.Sequential(
            nn.Linear(4, cfg.hidden),
            nn.Tanh(),
            nn.Linear(cfg.hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # inputs: surprise, uncertainty, reward, phase
        stacked = torch.cat(
            [inputs[k] for k in ("surprise", "uncertainty", "reward", "phase")], dim=-1
        )
        out = self.net(stacked)
        return {"m_gate": out[..., :1], "write_scale": out[..., 1:]}

