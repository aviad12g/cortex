"""Fast-weight sidecar implementing Gated Delta Rule (Yang et al., 2025)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CortexBlockConfig:
    d_model: int
    n_heads: int
    d_head: int = 64  # Default from Gated DeltaNet paper
    conv_kernel: int = 4
    
    # Hyperparameters for Gated Delta Rule
    # These can be learned or fixed
    use_short_conv: bool = True
    use_output_gate: bool = True


class CortexBlock(nn.Module):
    """
    Gated DeltaNet Block.
    
    Implements the update rule:
    S_t = S_{t-1} * alpha_t * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T
    
    Simplified for efficient computation:
    v_old = S_{t-1} @ k_t
    S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * v_old) @ k_t^T
    """

    is_cortex_param = True

    def __init__(self, cfg: CortexBlockConfig):
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_head
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        
        # Projections
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        
        # Gating projections (alpha = forget, beta = write strength)
        # Projecting from d_model to n_heads (scalar per head)
        self.alpha_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        self.beta_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        
        # Short Convolution (Depthwise)
        if cfg.use_short_conv:
            self.conv_q = nn.Conv1d(cfg.n_heads * cfg.d_head, cfg.n_heads * cfg.d_head, 
                                    kernel_size=cfg.conv_kernel, groups=cfg.n_heads * cfg.d_head, padding=cfg.conv_kernel-1)
            self.conv_k = nn.Conv1d(cfg.n_heads * cfg.d_head, cfg.n_heads * cfg.d_head,
                                    kernel_size=cfg.conv_kernel, groups=cfg.n_heads * cfg.d_head, padding=cfg.conv_kernel-1)
            self.conv_v = nn.Conv1d(cfg.n_heads * cfg.d_head, cfg.n_heads * cfg.d_head,
                                    kernel_size=cfg.conv_kernel, groups=cfg.n_heads * cfg.d_head, padding=cfg.conv_kernel-1)
        
        # Output projection
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        
        if cfg.use_output_gate:
            self.g_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=True)
            
        # Layer Norms / Group Norms could be added here as per paper, 
        # but for sidecar usage, we stick to simple structures first.
        
        # Fast Weight State (S)
        # Buffer shape: [1, H, D, D]
        self.register_buffer(
            "S",
            torch.zeros(1, cfg.n_heads, cfg.d_head, cfg.d_head),
            persistent=False
        )

    def load_fast(self, U: torch.Tensor, V: Optional[torch.Tensor] = None) -> None:
        # We only use one buffer 'S' now (combining U and V in the new architecture)
        # If V is passed (legacy), ignore it or assume U contains S
        self.S = U
    def reset_fast(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        device = device or self.S.device
        if self.S.shape[0] != batch_size or self.S.device != device:
             self.S = torch.zeros(batch_size, self.cfg.n_heads, self.d_head, self.d_head, device=device)
        else:
            self.S.zero_()

    def tie_projections(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear) -> None:
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        m_gate: Optional[torch.Tensor] = None, 
        alpha_scale: Optional[torch.Tensor] = None, 
        mix_mode: str = "dual",
    ) -> torch.Tensor:
        B, T, D = hidden_states.shape
        if self.S.shape[0] != B:
            self.reset_fast(B, device=hidden_states.device)

        # 1. Projections
        q = self.q_proj(hidden_states) # [B, T, H*D_head]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 2. Transpose for Conv1d: [B, C, T]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 3. Short Conv (Causal)
        if self.cfg.use_short_conv:
            q = self.conv_q(q)[:, :, :T]
            k = self.conv_k(k)[:, :, :T]
            v = self.conv_v(v)[:, :, :T]
            
        # 4. Activations & Norms
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v) 
        
        # L2 Norm on k (crucial for DeltaNet stability)
        k = k.transpose(1, 2).view(B, T, self.n_heads, self.d_head)
        k = F.normalize(k, p=2, dim=-1)
        
        q = q.transpose(1, 2).view(B, T, self.n_heads, self.d_head)
        v = v.transpose(1, 2).view(B, T, self.n_heads, self.d_head)
        
        # 5. Compute Gating Factors
        alpha_logits = self.alpha_proj(hidden_states) # [B, T, H]
        beta_logits = self.beta_proj(hidden_states)   # [B, T, H]
        
        alpha = torch.sigmoid(alpha_logits)
        beta = torch.sigmoid(beta_logits)
        
        if m_gate is not None:
             alpha = alpha * m_gate.unsqueeze(-1)
             beta = beta * m_gate.unsqueeze(-1)

        outputs = []
        
        for t in range(T):
            kt = k[:, t] # [B, H, D]
            vt = v[:, t] # [B, H, D]
            qt = q[:, t] # [B, H, D]
            at = alpha[:, t, :, None, None] # [B, H, 1, 1]
            bt = beta[:, t, :, None, None]  # [B, H, 1, 1]
            
            # v_old = S_{t-1} @ k_t
            # S: [B, H, D, D]
            # kt: [B, H, D] -> [B, H, D, 1]
            v_old = torch.matmul(self.S, kt.unsqueeze(-1)).squeeze(-1) # [B, H, D]
            
            # Update term
            # at is [B, H, 1, 1], we need [B, H, 1] for broadcasting with v_old [B, H, D]
            update_val = vt - at.squeeze(-1) * v_old
            
            # delta_S
            delta_S = torch.matmul(update_val.unsqueeze(-1), kt.unsqueeze(-2))
            
            # In-place update might cause issues with autograd if not careful, 
            # but here we overwrite self.S. 
            # For backprop through time, this sequential overwrite is fine in eager mode 
            # (PyTorch tracks versions).
            # To avoid 'in-place' error on S during backward if needed:
            S_next = at * self.S.clone() + bt * delta_S
            
            # STABILITY FIX: Clamp state to prevent explosion
            self.S = torch.clamp(S_next, -5.0, 5.0)
            
            # Read: o_t = S_t @ q_t
            ot = torch.matmul(self.S, qt.unsqueeze(-1)).squeeze(-1) # [B, H, D]
            outputs.append(ot)
            
        y = torch.stack(outputs, dim=1) # [B, T, H, D]
        
        # 7. Output Gate & Projection
        y = y.view(B, T, self.n_heads * self.d_head)
        
        if self.cfg.use_output_gate:
            g = F.silu(self.g_proj(hidden_states))
            y = y * g
            
        out = self.o_proj(y)
        
        # Telemetry for logs
        self.last_fast_share = y.detach().mean(dim=-1) # Proxy for "how much fast weight contributed"
        self.last_alpha = alpha.detach()
        
        return out
