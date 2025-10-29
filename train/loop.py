"""Hooks for live and sleep training phases."""

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class LoopHooks:
    on_live_step: Callable[[Dict], None]
    on_sleep_step: Callable[[Dict], None]

