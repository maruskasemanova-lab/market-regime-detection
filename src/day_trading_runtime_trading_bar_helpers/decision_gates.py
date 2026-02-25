"""Decision-gate helper facade for runtime trading-bar processing."""

from __future__ import annotations

from .decision_gates_base import (
    _apply_decision_slice_result_payload,
    _apply_golden_setup_signal_adjustments,
    _apply_pre_entry_guards,
    _build_decision_proxy_from_slice,
    _normalize_direction_token,
    _signal_direction,
)
from .decision_gates_filters import (
    _apply_confirming_sources_gate,
    _apply_l2_confirmation_gate,
    _apply_momentum_diversification_gate,
    _apply_momentum_flow_confirmation_gate,
    _apply_mu_choppy_regime_gate,
    _apply_tcbbo_confirmation_gate,
)

__all__ = [
    "_normalize_direction_token",
    "_signal_direction",
    "_apply_pre_entry_guards",
    "_apply_decision_slice_result_payload",
    "_build_decision_proxy_from_slice",
    "_apply_mu_choppy_regime_gate",
    "_apply_l2_confirmation_gate",
    "_apply_tcbbo_confirmation_gate",
    "_apply_golden_setup_signal_adjustments",
    "_apply_confirming_sources_gate",
    "_apply_momentum_flow_confirmation_gate",
    "_apply_momentum_diversification_gate",
]
