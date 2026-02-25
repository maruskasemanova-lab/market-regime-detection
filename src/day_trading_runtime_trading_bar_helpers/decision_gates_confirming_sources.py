"""Decision-gate helpers for runtime trading-bar processing."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ..day_trading_models import TradingSession
from ..day_trading_runtime_portfolio import (
    cooldown_bars_remaining as _cooldown_bars_remaining_impl,
)
from ..strategies.base_strategy import Regime, Signal, SignalType


def _apply_confirming_sources_gate(
    self,
    *,
    session: TradingSession,
    signal: Signal,
    decision: SimpleNamespace,
    effective_trade_threshold: float,
    regime: Optional[Regime],
    tod_boost: float,
    timestamp: datetime,
    required_confirming_sources: int,
    result: Dict[str, Any],
) -> bool:
    """Apply confirming-source threshold gate and populate diagnostics."""

    confirming_stats = self._confirming_source_stats(signal)
    confirming_sources = confirming_stats["confirming_sources"]

    layer_scores = result["layer_scores"]
    layer_scores["confirming_sources"] = confirming_sources
    layer_scores["confirming_sources_source"] = confirming_stats["count_source"]
    layer_scores["aligned_evidence_sources"] = confirming_stats["aligned_evidence_sources"]
    layer_scores["aligned_source_keys"] = confirming_stats["aligned_source_keys"]

    strategy_key = str(signal.strategy_name or "").strip().lower()
    level_strategies = {"mean_reversion", "rotation", "vwap_magnet", "volumeprofile"}
    if strategy_key in level_strategies:
        required_confirming_sources = min(required_confirming_sources, 2)
    elif strategy_key == "pullback":
        required_confirming_sources = min(
            required_confirming_sources,
            2
            + (
                1
                if (session.detected_regime or Regime.MIXED) in {Regime.CHOPPY, Regime.MIXED}
                else 0
            ),
        )
    layer_scores["required_confirming_sources"] = required_confirming_sources

    if confirming_sources >= required_confirming_sources:
        return False

    result["action"] = "confirming_sources_filtered"
    result["reason"] = (
        f"Confirming sources {confirming_sources} below required "
        f"{required_confirming_sources}"
    )
    aligned_set = set(confirming_stats["aligned_source_keys"])
    non_aligned_keys = []
    signal_direction = (
        self.evidence_service.signal_direction(signal) if hasattr(self, "evidence_service") else None
    )
    raw_evidence = signal.metadata.get("evidence_sources", []) if isinstance(signal.metadata, dict) else []
    for src in (raw_evidence if isinstance(raw_evidence, list) else []):
        if not isinstance(src, dict):
            continue
        src_type = str(src.get("type", "")).strip().lower()
        src_name = str(src.get("name", "")).strip().lower()
        if not src_type or not src_name:
            continue
        src_key = f"{src_type}:{src_name}"
        if src_key not in aligned_set:
            src_dir = str(src.get("direction", "")).strip().lower()
            non_aligned_keys.append({"key": src_key, "direction": src_dir})

    seen_na = set()
    unique_non_aligned = []
    for item in non_aligned_keys:
        if item["key"] not in seen_na:
            seen_na.add(item["key"])
            unique_non_aligned.append(item)

    result["signal_rejected"] = {
        "gate": "confirming_sources",
        "strategy": signal.strategy_name,
        "signal_type": signal.signal_type.value if signal.signal_type else None,
        "confidence": round(signal.confidence, 1),
        "combined_score": round(decision.combined_score, 1),
        "threshold_used": effective_trade_threshold,
        "regime": regime.value if regime else None,
        "micro_regime": session.micro_regime,
        "actual_confirming_sources": confirming_sources,
        "required_confirming_sources": required_confirming_sources,
        "aligned_evidence_sources": confirming_stats["aligned_evidence_sources"],
        "count_source": confirming_stats["count_source"],
        "aligned_source_keys": confirming_stats["aligned_source_keys"],
        "non_aligned_source_keys": unique_non_aligned,
        "signal_direction": signal_direction,
        "tod_threshold_boost": tod_boost,
        "timestamp": timestamp.isoformat(),
    }
    return True

