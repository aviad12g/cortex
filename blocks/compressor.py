# File: blocks/compressor.py
import torch
import torch.nn as nn
import math

class Compressor(nn.Module):
    """
    Cortex v3 Compressor Module.
    Reduces sequence length by a factor of 'ratio' using a strided convolution.
    This acts as a learned pooling layer, converting 'tokens' into 'concept vectors'.
    """
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        self.ratio = ratio
        
        # 1. The Convolution (Depthwise)
        # Acts as a learned pooling layer per channel.
        # Kernel = Stride = Ratio -> Non-overlapping windows (Hard Compression).
        # Groups = Dim -> Depthwise Convolution (Independent per channel).
        # Params: Dim * Kernel (Tiny compared to Dense Conv)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=ratio,
            stride=ratio,
            padding=0,     # We handle padding externally in the parent block
            groups=dim,    # DEPTHWISE: Massive parameter reduction
            bias=True
        )
        
        # 2. Normalization & Activation
        # We use LayerNorm and GELU to match standard Transformer blocks.
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        
        # 3. Initialization
        # We initialize to keep variance controlled at the start.
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights to avoid signal explosion at start of training.
        Standard Kaiming Normal is good, but we scale it slightly to ensure
        the compressed output doesn't dominate the recurrent state immediately.
        """
        with torch.no_grad():
            nn.init.kaiming_normal_(self.conv.weight, a=math.sqrt(5))
            # Slight scaling down to smooth early training
            self.conv.weight.data *= 0.5
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        # Transpose for Conv1d: [B, D, T]
        x_t = x.transpose(1, 2)
        
        # Conv1d
        out = self.conv(x_t)
        
        # GELU
        out = self.act(out)
        
        # Transpose back: [B, T_new, D]
        out = out.transpose(1, 2)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("!!! NAN/INF IN COMPRESSOR !!!")
            import sys; sys.exit(1)
            
        return out
