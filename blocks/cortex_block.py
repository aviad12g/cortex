"""
CortexBlock: attention sidecar with low-rank fast weights and dual-path mixing.

This module will be attached to each attention layer of the base model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CortexBlockConfig:
    d_model: int
    n_heads: int
    rank_fast: int = 16
    decay: float = 0.95
    alpha_max: float = 0.05
    beta: float = 0.01
    eps: float = 1e-5


class CortexBlock(nn.Module):
    """
    Fast-weight sidecar that augments a standard self-attention block.

    The block stores low-rank fast weights (U, V) per head and exposes a forward
    method that iterates over sequence positions, updating the fast buffers online.
    """

    is_cortex_param = True

    def __init__(self, cfg: CortexBlockConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = cfg.d_model // cfg.n_heads

        # Fast weight buffers initialised lazily per batch via reset_fast.
        self.register_buffer(
            "U",
            torch.zeros(1, cfg.n_heads, self.d_head, cfg.rank_fast),
            persistent=False,
        )
        self.register_buffer(
            "V",
            torch.zeros(1, cfg.n_heads, cfg.rank_fast, self.d_head),
            persistent=False,
        )

        # Plasticity controller projections.
        self.alpha_proj = nn.Linear(cfg.d_model, cfg.n_heads)
        self.mix_logit = nn.Parameter(torch.zeros(cfg.n_heads))

        # Projections default to standalone copies; set via tie_projections.
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.last_fast_share: Optional[torch.Tensor] = None
        self.last_alpha: Optional[torch.Tensor] = None

    def reset_fast(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """Reset fast weights for a new sequence batch."""
        device = device or self.U.device
        self.U.resize_(batch_size, self.cfg.n_heads, self.d_head, self.cfg.rank_fast).zero_()
        self.V.resize_(batch_size, self.cfg.n_heads, self.cfg.rank_fast, self.d_head).zero_()

    def load_fast(self, U: torch.Tensor, V: torch.Tensor) -> None:
        """Load fast weights from a stored buffer."""
        self.U.resize_as_(U).copy_(U)
        self.V.resize_as_(V).copy_(V)

    def _clamp_update(self, tensor: torch.Tensor) -> torch.Tensor:
        """Anti-Hebbian stabilisation term."""
        return torch.clamp(tensor, min=-1.0, max=1.0)

    def tie_projections(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear) -> None:
        """Copy weights from a base attention block into the Cortex projections."""
        with torch.no_grad():
            self.q_proj.weight.copy_(q_proj.weight)
            self.k_proj.weight.copy_(k_proj.weight)
            self.v_proj.weight.copy_(v_proj.weight)
            self.o_proj.weight.copy_(o_proj.weight)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            layer.weight.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        m_gate: torch.Tensor,
        alpha_scale: torch.Tensor,
        mix_mode: str = "dual",
    ) -> torch.Tensor:
        """
        Process a sequence of hidden states using Cortex fast weights.

        Args:
            hidden_states: [B, T, D] residual stream entering the block.
            m_gate: [B, T] plasticity control values in [0, 1].
            alpha_scale: [B, T, H] per-head plasticity scale in [0, 1].
            mix_mode: mode controlling slow/fast blending.
        Returns:
            Updated hidden states with sidecar contribution applied.
        """
        B, T, _ = hidden_states.shape
        if self.U.shape[0] != B:
            self.reset_fast(B, device=hidden_states.device)

        deltas = []
        fast_share_accum: list[torch.Tensor] = []
        alpha_accum: list[torch.Tensor] = []
        self.last_fast_share = None
        self.last_alpha = None
        for t in range(T):
            h_t = hidden_states[:, t, :]
            q = self.q_proj(h_t).view(B, self.cfg.n_heads, self.d_head)
            k = self.k_proj(h_t).view(B, self.cfg.n_heads, self.d_head)
            v = self.v_proj(h_t).view(B, self.cfg.n_heads, self.d_head)

            with torch.no_grad():
                alpha = torch.sigmoid(self.alpha_proj(h_t))  # [B, H]
                alpha = alpha * m_gate[:, t].unsqueeze(-1)
                alpha = alpha * alpha_scale[:, t, :]
                if self.cfg.alpha_max is not None:
                    alpha = torch.clamp(alpha, max=self.cfg.alpha_max)

                self.U.mul_(self.cfg.decay)
                self.V.mul_(self.cfg.decay)

                ku = torch.einsum("bhd,bhdr->bhr", k, self.U)
                self.U.add_(
                    alpha.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhd,bhr->bhdr", k, ku)
                )
                self.V.add_(
                    alpha.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhr,bhd->bhrd", ku, v)
                )
                self.U.add_(-self.cfg.beta * self._clamp_update(self.U))
                self.V.add_(-self.cfg.beta * self._clamp_update(self.V))

            q_norm = q / (self.d_head**0.5)
            k_fast = torch.einsum("bhdr,bhrd->bhd", self.U, self.V)
            v_fast = k_fast

            score_slow = torch.einsum("bhd,bhd->bh", q_norm, k)
            score_fast = torch.einsum("bhd,bhd->bh", q_norm, k_fast)

            mix_logits = torch.stack([score_slow, score_fast], dim=-1) + self.mix_logit.view(1, -1, 1)
            mix = torch.softmax(mix_logits, dim=-1)
            if mix_mode == "slow_only":
                mix = mix.clone()
                mix[..., 0] = 1.0
                mix[..., 1] = 0.0

            y = mix[..., 0].unsqueeze(-1) * v + mix[..., 1].unsqueeze(-1) * v_fast
            y = y.reshape(B, -1)
            delta = self.o_proj(y)
            deltas.append(delta.unsqueeze(1))
            fast_share_accum.append(mix[..., 1].unsqueeze(1))
            alpha_accum.append(alpha.unsqueeze(1))

        if fast_share_accum:
            self.last_fast_share = torch.cat(fast_share_accum, dim=1).detach().cpu()
        else:
            self.last_fast_share = None
        if alpha_accum:
            self.last_alpha = torch.cat(alpha_accum, dim=1).detach().cpu()
        else:
            self.last_alpha = None
        return torch.cat(deltas, dim=1)
