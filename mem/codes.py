"""Compress segment statistics into fixed-size latent codes."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class CodeMakerConfig:
    d_model: int
    n_layers: int
    n_heads: int
    rank_fast: int
    d_code: int = 128

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def fast_delta_dim(self) -> int:
        # flattened U and V deltas
        return 2 * self.n_heads * self.d_head * self.rank_fast


class CodeMaker(nn.Module):

    is_cortex_param = True

    def __init__(self, cfg: CodeMakerConfig):
        super().__init__()
        self.cfg = cfg
        # project fast deltas down to d_model
        self.delta_proj = nn.Linear(cfg.fast_delta_dim, cfg.d_model)
        # aggregate across layers
        self.layer_proj = nn.Linear(cfg.n_layers * 3 * cfg.d_model, cfg.d_model)
        self.encoder = nn.Sequential(
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.d_code),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_code, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
        )

    def encode(
        self,
        h_mean: torch.Tensor,
        h_second: torch.Tensor,
        delta_fast: torch.Tensor,
    ) -> torch.Tensor:
        # h_mean, h_second: [B, L, D]
        # delta_fast: [B, L, fast_delta_dim]
        B = h_mean.size(0)
        delta_proj = self.delta_proj(delta_fast.view(B * self.cfg.n_layers, -1))
        delta_proj = delta_proj.view(B, self.cfg.n_layers, self.cfg.d_model)

        features = torch.cat([h_mean, h_second, delta_proj], dim=-1)
        features = features.view(B, -1)
        summary = self.layer_proj(features)
        return self.encoder(summary)

    def reconstruct(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(code)


class CodeQueue:
    """Fixed-size queue of recent codes for replay and contrastive negatives."""

    def __init__(self, maxlen: int = 1024):
        self.buffer: Deque[torch.Tensor] = deque(maxlen=maxlen)

    def push(self, code: torch.Tensor) -> None:
        if code.ndim == 1:
            code = code.unsqueeze(0)
        for item in code.detach():
            self.buffer.append(item.cpu())

    def sample(self, k: int) -> Optional[torch.Tensor]:
        if not self.buffer:
            return None
        k = min(k, len(self.buffer))
        indices = torch.randperm(len(self.buffer))[:k]
        stacked = torch.stack([self.buffer[i] for i in indices], dim=0)
        return stacked

    def clear(self) -> None:
        self.buffer.clear()


class SegmentAccumulator:
    """Collect mean/variance of hiddens + fast-weight deltas across a segment."""

    def __init__(self, cfg: CodeMakerConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._h_sum = None
        self._h_sq_sum = None
        self._fast_start: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def start(self, fast_buffers: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.reset()
        for idx, (U, V) in fast_buffers.items():
            self._fast_start[idx] = (U.detach().clone(), V.detach().clone())

    def accumulate(self, layer_hiddens: Sequence[torch.Tensor]) -> None:
        if not layer_hiddens:
            return
        B = layer_hiddens[0].shape[0]
        device = layer_hiddens[0].device
        if self._h_sum is None:
            self._h_sum = torch.zeros(B, self.cfg.n_layers, self.cfg.d_model, device=device)
            self._h_sq_sum = torch.zeros_like(self._h_sum)
        for idx, hidden in enumerate(layer_hiddens):
            self._h_sum[:, idx, :] += hidden
            self._h_sq_sum[:, idx, :] += hidden * hidden
        self._count += 1

    def finalize(
        self,
        fast_buffers_end: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._h_sum is not None and self._h_sq_sum is not None, "No tokens accumulated for segment."
        B = self._h_sum.size(0)
        count = max(self._count, 1)
        h_mean = self._h_sum / count
        h_second = self._h_sq_sum / count

        delta_fast = []
        for layer_idx in range(self.cfg.n_layers):
            start_snapshot = self._fast_start.get(layer_idx)
            end_snapshot = fast_buffers_end.get(layer_idx)
            if start_snapshot is None or end_snapshot is None:
                delta_fast.append(torch.zeros(B, self.cfg.fast_delta_dim, device=self._h_sum.device))
                continue
            start_U, start_V = start_snapshot
            end_U, end_V = end_snapshot
            delta_U = (end_U - start_U).reshape(B, -1)
            delta_V = (end_V - start_V).reshape(B, -1)
            delta = torch.cat([delta_U, delta_V], dim=-1)
            delta_fast.append(delta)

        delta_fast_tensor = torch.stack(delta_fast, dim=1)
        return h_mean, h_second, delta_fast_tensor

