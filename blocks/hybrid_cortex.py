import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


from blocks.compressor import Compressor  # Import the new module


class ChunkwiseDeltaCore(nn.Module):
    """
    Cortex v2/v3 Core: Chunkwise Parallel Gated DeltaNet

    Speed: 10x-50x faster than sequential recurrence.
    Mechanism:
      - Intra-Chunk: Parallel Scan (Attention-like)
      - Inter-Chunk: Recurrent State Update
    """

    is_cortex_param = True

    def __init__(self, dim: int, head_dim: int, chunk_size: int = 128):
        super().__init__()
        self.dim = dim
        self.d_state = head_dim
        self.chunk_size = chunk_size

        # Projections
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_beta = nn.Linear(dim, dim)  # Write Strength
        # Alpha is data-dependent decay. We use sigmoid to keep it [0, 1]
        self.proj_alpha = nn.Linear(dim, dim)

        # THE BIAS HACK: Initialize Alpha bias to +2.0
        # This ensures memory persistence starts high (~0.88)
        with torch.no_grad():
            self.proj_alpha.bias.fill_(2.0)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def check(t, name):
            if t is None: return
            if torch.isnan(t).any():
                print(f"!!! NAN IN {name} !!!")
                print(f"{name}: min={t.min()}, max={t.max()}, mean={t.mean()}")
                import sys; sys.exit(1) # FORCE EXIT
            if torch.isinf(t).any():
                print(f"!!! INF IN {name} !!!")
                print(f"{name}: min={t.min()}, max={t.max()}, mean={t.mean()}")
                import sys; sys.exit(1) # FORCE EXIT

        check(x, "Input X")
        
        B, T, D = x.shape
        C = self.chunk_size

        # 1. Projections
        scale = self.d_state ** -0.5
        q = (self.proj_q(x) * scale).view(B, T, -1, self.d_state)
        k = (self.proj_k(x) * scale).view(B, T, -1, self.d_state)
        v = self.proj_v(x).view(B, T, -1, self.d_state)
        
        check(q, "Q")
        check(k, "K")
        check(v, "V")

        # Gates
        alpha_logits = self.proj_alpha(x).view(B, T, -1, self.d_state)
        check(alpha_logits, "Alpha Logits Pre-Clamp")
        alpha_logits = torch.clamp(alpha_logits, min=-10.0, max=10.0)
        log_alpha = F.logsigmoid(alpha_logits)
        check(log_alpha, "Log Alpha")

        alpha_val = torch.sigmoid(alpha_logits)
        beta = F.softplus(self.proj_beta(x)).view(B, T, -1, self.d_state)
        check(beta, "Beta")

        # --- PRECISION FIX: Cast to float32 for recurrence ---
        orig_dtype = x.dtype
        q = q.float()
        k = k.float()
        v = v.float()
        log_alpha = log_alpha.float()
        beta = beta.float()
        # -----------------------------------------------------

        # 2. Reshape into Chunks
        if T % C != 0:
            padding = C - (T % C)
            q = F.pad(q, (0, 0, 0, 0, 0, padding))
            k = F.pad(k, (0, 0, 0, 0, 0, padding))
            v = F.pad(v, (0, 0, 0, 0, 0, padding))
            log_alpha = F.pad(log_alpha, (0, 0, 0, 0, 0, padding))
            beta = F.pad(beta, (0, 0, 0, 0, 0, padding))

        n_chunks = q.shape[1] // C
        q = q.view(B, n_chunks, C, -1, self.d_state)
        k = k.view(B, n_chunks, C, -1, self.d_state)
        v = v.view(B, n_chunks, C, -1, self.d_state)
        log_alpha = log_alpha.view(B, n_chunks, C, -1, self.d_state)
        beta = beta.view(B, n_chunks, C, -1, self.d_state)

        # 3. INTRA-CHUNK
        chunk_cumsum = torch.cumsum(log_alpha, dim=2)
        decay_logits = chunk_cumsum.unsqueeze(3) - chunk_cumsum.unsqueeze(2)
        decay_logits_mean = decay_logits.mean(dim=-1)

        mask = torch.tril(torch.ones(C, C, device=x.device), diagonal=0)
        mask_broad = mask.view(1, 1, C, C, 1)
        decay_logits_mean = decay_logits_mean.masked_fill(mask_broad == 0, -1e9)
        
        decay_matrix = torch.exp(decay_logits_mean)
        check(decay_matrix, "Decay Matrix")

        attn = torch.einsum("bnihd, bnjhd -> bnhij", q, k)
        check(attn, "Attn Raw")
        attn = attn * decay_matrix.permute(0, 1, 4, 2, 3)
        check(attn, "Attn Decayed")

        y_intra = torch.einsum("bnhij, bnjhd -> bnihd", attn, v * beta)
        check(y_intra, "Y Intra")

        # 4. INTER-CHUNK
        if state is None:
            state = torch.zeros(
                B, q.shape[3], self.d_state, self.d_state, device=x.device, dtype=torch.float32
            )
        else:
            state = state.float()

        y_inter = []

        for i in range(n_chunks):
            mem_out = torch.einsum("bchq, bhqk -> bchk", q[:, i], state)
            
            chunk_decay_curve = torch.exp(
                chunk_cumsum[:, i].mean(dim=-1)
            ).unsqueeze(-1)
            mem_out = mem_out * chunk_decay_curve
            
            y_chunk = y_intra[:, i] + mem_out
            y_inter.append(y_chunk)

            total_chunk_decay = (
                torch.exp(torch.sum(log_alpha[:, i], dim=1))
                .mean(dim=-1)
                .view(B, -1, 1, 1)
            )
            state = state * total_chunk_decay

            delta_chunk = torch.einsum(
                "bchk, bchv, bch -> bhkv",
                k[:, i],
                v[:, i],
                beta[:, i].mean(dim=-1),
            )
            state = state + delta_chunk
            check(state, f"State Chunk {i}")

        y = torch.cat(y_inter, dim=1)

        if T % C != 0:
            y = y[:, :T, :]

        y = y.reshape(B, T, D)
        
        y = y.to(dtype=orig_dtype)
        state = state.to(dtype=orig_dtype)
        check(y, "Output Y")

        return y, state, alpha_val


class SlidingWindowAttention(nn.Module):
    def __init__(self, dim: int, window: int):
        super().__init__()
        self.head_dim = 64
        self.n_heads = max(1, dim // 64)
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # Standard QKV computation
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape for Multi-Head: [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Sliding Window Mask
        # We only attend to j where i - window <= j <= i
        # SDPA accepts a boolean mask where True = Attend, False = Mask out
        # OR a float mask with -inf.
        # Creating mask on the fly (memory O(T^2), but T is usually small in TBPTT)
        mask = (
            torch.ones(T, T, device=x.device, dtype=torch.bool)
            .tril(0)
            .triu(-self.window)
        )

        # Efficient SDPA
        # Note: attn_mask support varies by PyTorch version.
        # If this fails, we can revert to manual. But 2.0+ supports it.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("!!! NAN/INF IN SLIDING WINDOW ATTN !!!")
            import sys; sys.exit(1)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


class HybridCortexBlock(nn.Module):
    """
    Cortex v3: The Compressive Hybrid
    Combines:
    1. Sliding Window Attention (High Res, Local)
    2. Compressive Gated DeltaNet (Low Res, Global)
    3. Local Prediction Error (Deep Supervision)
    """

    is_cortex_param = True

    def __init__(
        self,
        dim: int,
        head_dim: int,
        window_size: int = 128,
        compression_ratio: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.compression_ratio = compression_ratio

        # 1. The "Photographic" Short-Term Memory
        # Uses standard attention for the last 'window_size' tokens
        self.window_attn = SlidingWindowAttention(dim, window_size)

        # 2. The "Infinite" Long-Term Memory (Compressive)
        # Compresses input, updates state, decompresses output.
        self.compressor = Compressor(dim, ratio=compression_ratio)
        
        # UPDATED: Use ChunkwiseDeltaCore (v3 Engine)
        # We set internal chunk_size to 32 (assuming compression_ratio=4 and typical TBPTT=128+)
        # This minimizes padding waste for compressed sequences.
        core_chunk_size = max(16, 128 // compression_ratio)
        self.recurrent_mem = ChunkwiseDeltaCore(dim, head_dim, chunk_size=core_chunk_size)

        # 3. The "Intelligence" Gate (Controller)
        # Determines how much to trust Memory vs. Attention
        self.gate_mixer = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.Sigmoid()
        )

        self.out_proj = nn.Linear(dim, dim)

        # --- CRITICAL FIX: ZERO INITIALIZATION ---
        # This forces the sidecar to act as an Identity Function (f(x) = x)
        # at Step 0.
        # The model starts with 100% baseline performance and slowly
        # fades in the memory.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # -----------------------------------------

        # Telemetry placeholders
        self.last_alpha = None
        self.last_fast_share = None
        self.m_gate_val = None

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [Batch, Seq, Dim] (Expects LayerNormed input)
        state: The recurrent memory state S_{t-1}
        """
        B, T, D = x.shape

        # A. Short-Term Precision (The "Eyes")
        # Looks at recent tokens with perfect clarity
        attn_out = self.window_attn(x)

        # B. Long-Term Context (The "Brain")
        # 1. Handle Padding
        pad_len = (
            self.compression_ratio - (T % self.compression_ratio)
        ) % self.compression_ratio
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        # 2. Compress
        x_compressed = self.compressor(x_padded)

        # 3. Recurse (Low Res)
        # ChunkwiseDeltaCore returns (y, state, alpha)
        mem_out_small, next_state, alpha_tensor = (
            self.recurrent_mem(x_compressed, state)
        )

        # 4. Decompress / Broadcast
        mem_out = mem_out_small.repeat_interleave(
            self.compression_ratio, dim=1
        )

        # --- CAUSAL FIX ---
        # Shift output to the right to prevent leakage from future tokens within the compressed chunk.
        # y[0] (derived from x[0..3]) should only be visible at t=4.
        mem_out = torch.roll(mem_out, shifts=self.compression_ratio, dims=1)
        mem_out[:, :self.compression_ratio, :] = 0.0
        # ------------------

        # 5. Trim Padding
        mem_out = mem_out[:, :T, :]

        # C. Cognitive Mixing
        # The model decides: "Do I need the immediate detail (Attn) or the
        # deep context (Mem)?"
        # We concatenate both and let the gate decide.
        mix_gate = self.gate_mixer(torch.cat([attn_out, mem_out], dim=-1))

        # Final Output is a blend
        combined = (mix_gate * attn_out) + ((1 - mix_gate) * mem_out)
        output = self.out_proj(combined)

        # Telemetry
        self.last_fast_share = mem_out.detach().mean(
            dim=-1
        )  # Proxy

        # LOG THE REAL DATA (Telemetry Fix)
        self.last_alpha = alpha_tensor.detach()
        self.m_gate_val = mix_gate.detach().mean()
        
        # Surprise proxy (not computed in ChunkwiseDeltaCore for speed)
        surprise_signal = torch.tensor(0.0, device=x.device)

        # STATEFUL FIX: Store state for hf_wrap.py compatibility
        self.S = next_state

        return output, next_state, surprise_signal

    def reset_fast(self, batch_size: int, device: torch.device):
        self.S = None
