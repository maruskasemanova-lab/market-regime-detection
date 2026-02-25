"""Evidence/momentum decision-gate helper facade for runtime trading-bar processing."""

from __future__ import annotations

from .decision_gates_confirming_sources import _apply_confirming_sources_gate
from .decision_gates_momentum import (
    _apply_momentum_diversification_gate,
    _apply_momentum_flow_confirmation_gate,
)

__all__ = [
    "_apply_confirming_sources_gate",
    "_apply_momentum_flow_confirmation_gate",
    "_apply_momentum_diversification_gate",
]
