"""Helpers for managing fast-weight buffers."""

from typing import Dict

import torch
import torch.nn as nn


def allocate_fast_buffers(
    module: nn.Module,
    batch_size: int,
    device: torch.device,
) -> None:
    # recursively find all Cortex blocks and reset their U, V
    for child in module.modules():
        if hasattr(child, "reset_fast"):
            child.reset_fast(batch_size, device=device)


class FastWeightTraces:
    """Track deltas across a segment for later replay."""

    def __init__(self):
        self.traces: Dict[str, torch.Tensor] = {}

    def record(self, name: str, delta: torch.Tensor) -> None:
        self.traces[name] = delta.detach().cpu()

    def clear(self) -> None:
        self.traces.clear()

