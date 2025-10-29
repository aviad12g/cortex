"""Synaptic scaling to keep norms stable."""

import torch
import torch.nn as nn


def apply_synaptic_scaling(module: nn.Module, target: float = 1.0, rate: float = 1e-4) -> None:
    # gently push weights toward target norm
    for submodule in module.modules():
        if isinstance(submodule, nn.LayerNorm):
            with torch.no_grad():
                delta = target - submodule.weight.data.norm(p=2)
                submodule.weight.add_(rate * delta)
        elif isinstance(submodule, nn.Linear):
            with torch.no_grad():
                post_norm = submodule.weight.data.norm(p=2)
                if post_norm > 0:
                    scale = target / post_norm
                    submodule.weight.mul_(1 + rate * (scale - 1))

