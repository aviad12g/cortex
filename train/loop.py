"""
Training loop utilities covering live tokens and micro-sleep phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class LoopHooks:
    on_live_step: Callable[[Dict], None]
    on_sleep_step: Callable[[Dict], None]

