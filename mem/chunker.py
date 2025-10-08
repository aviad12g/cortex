"""
Surprise-driven segmenter that emits boundaries and caches segment statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class ChunkerConfig:
    ema_decay: float = 0.99
    tau_up_multiplier: float = 1.2
    tau_down_multiplier: float = 1.05
    min_steps_between_boundaries: int = 16


class SurpriseChunker:
    """Track running surprise and emit boundary flags with hysteresis."""

    def __init__(self, cfg: ChunkerConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.ema: Optional[torch.Tensor] = None
        self.active = False
        self.step_count = 0
        self.last_boundary_step = -self.cfg.min_steps_between_boundaries

    def step(self, nll: torch.Tensor) -> Optional[Literal["open", "close"]]:
        """
        Args:
            nll: scalar tensor with token negative log-likelihood.
        Returns:
            "open", "close", or None depending on boundary events.
        """
        self.step_count += 1
        if self.ema is None:
            self.ema = nll.detach()
        else:
            self.ema = self.cfg.ema_decay * self.ema + (1 - self.cfg.ema_decay) * nll.detach()

        if self.ema is None or self.ema.item() <= 0:
            return None

        if self.step_count - self.last_boundary_step < self.cfg.min_steps_between_boundaries:
            return None

        surprise = (nll - self.ema).item()
        tau_up = self.cfg.tau_up_multiplier * self.ema.item()
        tau_down = self.cfg.tau_down_multiplier * self.ema.item()

        if not self.active and surprise > tau_up:
            self.active = True
            self.last_boundary_step = self.step_count
            return "open"

        if self.active and surprise < tau_down:
            self.active = False
            self.last_boundary_step = self.step_count
            return "close"

        return None
