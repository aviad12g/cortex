"""
Approximate diagonal Fisher information estimates for Cortex parameters.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn


class FisherDiagonal:
    """Track running second moments of gradients for EWC-style regularisation."""

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.stats: Dict[str, torch.Tensor] = {}

    def update(self, params: Iterable[nn.Parameter]) -> None:
        for param in params:
            if param.grad is None:
                continue
            name = str(id(param))
            value = param.grad.detach() ** 2
            if name not in self.stats:
                self.stats[name] = value
            else:
                self.stats[name] = self.decay * self.stats[name] + (1 - self.decay) * value

    def penalty(self, params: Iterable[nn.Parameter]) -> torch.Tensor:
        loss = 0.0
        for param in params:
            name = str(id(param))
            if name not in self.stats:
                continue
            loss = loss + torch.sum(self.stats[name] * (param.detach() - param) ** 2)
        return loss

