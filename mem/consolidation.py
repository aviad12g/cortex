"""
Micro-sleep consolidation loop operating on Cortex sidecar parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from mem import syn_scaling


@dataclass
class ConsolidationConfig:
    sleep_steps: int = 8
    fisher_lambda: float = 0.1
    scaling_rate: float = 1e-4


class Consolidator:
    """Coordinates replay, EWC-like penalties, and Hebbian capture."""

    def __init__(self, cfg: ConsolidationConfig, model: nn.Module):
        self.cfg = cfg
        self.model = model

    def step(self, replay_loss: torch.Tensor, fisher_penalty: torch.Tensor) -> None:
        loss = replay_loss + self.cfg.fisher_lambda * fisher_penalty
        loss.backward()

    def post_update(self) -> None:
        syn_scaling.apply_synaptic_scaling(self.model, target=1.0, rate=self.cfg.scaling_rate)

