from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExitContext:
    """Everything the exit engine needs to evaluate exits for one bar."""

    position: Any
    bar: Any
    session: Any
    current_bar_index: int
    flow_metrics: Dict[str, Any] = field(default_factory=dict)
    momentum_config: Dict[str, Any] = field(default_factory=dict)
    momentum_sleeve_id: str = ""
    # Context events from PositionContextMonitor (Phase 2/3)
    context_events: List[Any] = field(default_factory=list)
    entry_snapshot: Any = None


@dataclass
class ExitDecision:
    """Result of an exit policy evaluation."""

    should_exit: bool
    reason: str
    exit_price: float
    metrics: Dict[str, Any] = field(default_factory=dict)


__all__ = ["ExitContext", "ExitDecision"]
