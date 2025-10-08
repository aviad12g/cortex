"""
Utility functions for managing Cortex fast-weight buffers.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def allocate_fast_buffers(
    module: nn.Module,
    batch_size: int,
    device: torch.device,
) -> None:
    """
    Recursively reset fast buffers on all Cortex blocks within `module`.
    """
    for child in module.modules():
        if hasattr(child, "reset_fast"):
            child.reset_fast(batch_size, device=device)


class FastWeightTraces:
    """Container for retaining delta snapshots per layer within a segment."""

    def __init__(self):
        self.traces: Dict[str, torch.Tensor] = {}

    def record(self, name: str, delta: torch.Tensor) -> None:
        self.traces[name] = delta.detach().cpu()

    def clear(self) -> None:
        self.traces.clear()

