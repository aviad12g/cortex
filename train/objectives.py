"""
Loss functions and auxiliary probes for Cortex Stage A training.
"""

from __future__ import annotations

import torch


def recall_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple L2 recall loss placeholder."""
    return torch.mean((pred - target) ** 2)

