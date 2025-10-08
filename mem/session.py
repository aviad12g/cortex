"""
Session management for Cortex fast-weight persistence and controller signals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import torch


@dataclass
class FastBufferSnapshot:
    U: torch.Tensor
    V: torch.Tensor


@dataclass
class SessionState:
    """Holds per-session fast buffers, controller context, and code queues."""

    session_id: str
    phase_period: int = 256
    code_queue_size: int = 1024
    step_count: int = 0
    surprise_level: float = 0.0
    uncertainty_level: float = 0.0
    reward_level: float = 0.0
    fast_buffers: Dict[int, FastBufferSnapshot] = field(default_factory=dict)
    code_queue: Deque[torch.Tensor] = field(default_factory=deque)

    def reset(self) -> None:
        self.fast_buffers.clear()
        self.code_queue.clear()
        self.step_count = 0
        self.surprise_level = 0.0
        self.uncertainty_level = 0.0
        self.reward_level = 0.0

    def store_fast_buffer(self, layer_idx: int, U: torch.Tensor, V: torch.Tensor) -> None:
        self.fast_buffers[layer_idx] = FastBufferSnapshot(U.detach().clone(), V.detach().clone())

    def get_fast_buffer(self, layer_idx: int) -> Optional[FastBufferSnapshot]:
        return self.fast_buffers.get(layer_idx)

    def controller_inputs(self, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Produce controller inputs for the upcoming sequence portion.
        Currently uses stored scalar levels and a simple sinusoidal phase.
        """
        ticks = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        absolute_step = self.step_count + ticks
        phase = torch.sin(2 * torch.pi * absolute_step / max(self.phase_period, 1))

        surprise = torch.full((batch_size, seq_len, 1), self.surprise_level, device=device)
        uncertainty = torch.full((batch_size, seq_len, 1), self.uncertainty_level, device=device)
        reward = torch.full((batch_size, seq_len, 1), self.reward_level, device=device)
        phase = phase.unsqueeze(-1)

        return {
            "surprise": surprise,
            "uncertainty": uncertainty,
            "reward": reward,
            "phase": phase,
        }

    def advance(self, tokens: int) -> None:
        self.step_count += tokens

    def push_code(self, code: torch.Tensor) -> None:
        if len(self.code_queue) >= self.code_queue_size:
            self.code_queue.popleft()
        self.code_queue.append(code.detach())

