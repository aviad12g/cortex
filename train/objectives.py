"""Loss functions for training."""

import torch


def recall_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # simple MSE for now
    return torch.mean((pred - target) ** 2)

