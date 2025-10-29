"""Fast-weight sidecar that sits alongside base attention."""

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
    eps: float = 1e-5  # unused for now


class CortexBlock(nn.Module):
    """Low-rank fast weights (U, V) that get updated online per token."""

    is_cortex_param = True

    def __init__(self, cfg: CortexBlockConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.d_model % cfg.n_heads == 0
        self.d_head = cfg.d_model // cfg.n_heads

        # U, V buffers - not persistent, reset each batch
        self.register_buffer(
            "U",
            torch.zeros(1, cfg.n_heads, self.d_head, cfg.rank_fast),
            persistent=False,
        )
        self.register_buffer(
            "V",
            torch.zeros(1, cfg.n_heads, cfg.rank_fast, self.d_head),
            persistent=False
        )

        self.alpha_proj = nn.Linear(cfg.d_model, cfg.n_heads)
        self.mix_logit = nn.Parameter(torch.zeros(cfg.n_heads))

        # QKV projections - will be tied to base model later
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # tracking for analysis
        self.last_fast_share = None
        self.last_alpha = None

    def reset_fast(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        device = device or self.U.device
        self.U.resize_(batch_size, self.cfg.n_heads, self.d_head, self.cfg.rank_fast).zero_()
        self.V.resize_(batch_size, self.cfg.n_heads, self.cfg.rank_fast, self.d_head).zero_()

    def load_fast(self, U: torch.Tensor, V: torch.Tensor) -> None:
        self.U.resize_as_(U).copy_(U)
        self.V.resize_as_(V).copy_(V)

    def _clamp_update(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor, min=-1.0, max=1.0)

    def tie_projections(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear) -> None:
        # share QKV weights with base model, freeze them
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
        # hidden_states: [B, T, D]
        # m_gate: [B, T] plasticity
        # alpha_scale: [B, T, H] per-head scale
        B, T, _ = hidden_states.shape
        if self.U.shape[0] != B:
            self.reset_fast(B, device=hidden_states.device)

        deltas = []
        fast_share_accum = []
        alpha_accum = []
        self.last_fast_share = None
        self.last_alpha = None
        
        for t in range(T):
            h_t = hidden_states[:, t, :]
            q = self.q_proj(h_t).view(B, self.cfg.n_heads, self.d_head)
            k = self.k_proj(h_t).view(B, self.cfg.n_heads, self.d_head)
            v = self.v_proj(h_t).view(B, self.cfg.n_heads, self.d_head)

            with torch.no_grad():
                # compute effective plasticity
                alpha = torch.sigmoid(self.alpha_proj(h_t))
                alpha = alpha * m_gate[:, t].unsqueeze(-1)
                alpha = alpha * alpha_scale[:, t, :]
                if self.cfg.alpha_max is not None:
                    alpha = torch.clamp(alpha, max=self.cfg.alpha_max)

                # decay
                self.U.mul_(self.cfg.decay)
                self.V.mul_(self.cfg.decay)

                # Hebbian update: project k through U, then outer product back
                ku = torch.einsum("bhd,bhdr->bhr", k, self.U)
                self.U.add_(
                    alpha.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhd,bhr->bhdr", k, ku)
                )
                self.V.add_(
                    alpha.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhr,bhd->bhrd", ku, v)
                )
                # anti-Hebbian to prevent saturation
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
