"""Evidence and source-confirmation helpers for DayTradingManager."""

from __future__ import annotations

from datetime import time
from typing import Any, Callable, Dict, List, Optional

from .strategies.base_strategy import Regime, Signal, SignalType


class DayTradingEvidenceService:
    """Encapsulates evidence/source parsing and confidence helper logic."""

    def __init__(
        self,
        canonical_strategy_key: Callable[[str], str],
        safe_float: Callable[[Any, Optional[float]], Optional[float]],
    ):
        self._canonical_strategy_key = canonical_strategy_key
        self._safe_float = safe_float

    def safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        return self._safe_float(value, default)

    @staticmethod
    def bars_held(position: Any, current_bar_index: int) -> int:
        entry_index = getattr(position, "entry_bar_index", None)
        if entry_index is None:
            entry_index = current_bar_index
        return max(0, int(current_bar_index) - int(entry_index))

    def latest_indicator_value(
        self,
        indicators: Dict[str, Any],
        key: str,
        bars: Optional[List[Any]] = None,
    ) -> float:
        raw = indicators.get(key) if isinstance(indicators, dict) else None
        value: Optional[float] = None

        if isinstance(raw, list):
            if raw:
                value = self._safe_float(raw[-1], None)
        elif raw is not None:
            value = self._safe_float(raw, None)

        if value is None and key == "atr" and bars and len(bars) >= 2:
            tr_values: List[float] = []
            for i in range(1, len(bars)):
                prev_close = float(getattr(bars[i - 1], "close", 0.0) or 0.0)
                high = float(getattr(bars[i], "high", 0.0) or 0.0)
                low = float(getattr(bars[i], "low", 0.0) or 0.0)
                hl = high - low
                hc = abs(high - prev_close)
                lc = abs(low - prev_close)
                tr_values.append(max(hl, hc, lc))
            if tr_values:
                window = min(14, len(tr_values))
                value = sum(tr_values[-window:]) / window

        return float(value) if value is not None else 0.0

    @staticmethod
    def to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def resolve_evidence_weight_context(flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute lightweight strategy-weight context for diagnostics."""
        base_strategy_weight = 0.95
        bars_with_l2 = max(0.0, float(flow_metrics.get("bars_with_l2", 0.0) or 0.0))
        lookback_bars = max(1.0, float(flow_metrics.get("lookback_bars", 1.0) or 1.0))
        l2_coverage_ratio = min(1.0, bars_with_l2 / lookback_bars)
        l2_has_coverage = bool(flow_metrics.get("has_l2_coverage", False))
        l2_quality_ok = bool(l2_has_coverage)

        strategy_weight_source = "l2" if l2_quality_ok else "fallback"
        effective_strategy_weight = 1.0 if l2_quality_ok else base_strategy_weight

        return {
            "strategy_weight": float(effective_strategy_weight),
            "strategy_weight_source": strategy_weight_source,
            "base_strategy_weight": base_strategy_weight,
            "l2_has_coverage": l2_has_coverage,
            "l2_quality_ok": l2_quality_ok,
            "l2_coverage_ratio": l2_coverage_ratio,
            "l2_bars_with_coverage": bars_with_l2,
            "l2_lookback_bars": lookback_bars,
        }

    @staticmethod
    def is_midday_window(bar_time: time) -> bool:
        """Return True for lower-conviction midday window."""
        midday_start = time(10, 30)
        midday_end = time(14, 0)
        return midday_start <= bar_time < midday_end

    @staticmethod
    def required_confirming_sources(session: Any, bar_time: time, strategy_key: str = "") -> int:
        """Dynamic confirmation guard to reduce low-quality trades in noisy states.

        MR/rotation strategies are not penalised by the midday +1 because
        quiet midday ranges are their ideal environment.  The choppy +1
        still applies (they already get a cap to 2 downstream for level
        strategies, so the net effect is capped).
        """
        required = 2
        macro_regime = session.detected_regime or Regime.MIXED
        micro_regime = (session.micro_regime or "MIXED").upper()

        if macro_regime in {Regime.CHOPPY, Regime.MIXED} or micro_regime in {"CHOPPY", "MIXED"}:
            required = 3

        _MIDDAY_RELAXED = {"mean_reversion", "rotation", "vwap_magnet", "volumeprofile"}
        if DayTradingEvidenceService.is_midday_window(bar_time):
            if str(strategy_key or "").strip().lower() not in _MIDDAY_RELAXED:
                required += 1

        return required

    @staticmethod
    def is_mu_choppy_blocked(session: Any, regime: Regime) -> bool:
        """Block MU CHOPPY entries unless intrabar scalp module is active."""
        if session.ticker.upper() != "MU":
            return False

        micro_regime = (session.micro_regime or "MIXED").upper()
        if regime != Regime.CHOPPY and micro_regime != "CHOPPY":
            return False

        active_strategies = {
            str(name or "").strip().lower()
            for name in getattr(session, "active_strategies", []) or []
            if str(name or "").strip()
        }
        if "scalp_l2_intrabar" in active_strategies:
            return False
        return True

    @staticmethod
    def signal_direction(signal: Signal) -> Optional[str]:
        if signal.signal_type == SignalType.BUY:
            return "bullish"
        if signal.signal_type == SignalType.SELL:
            return "bearish"
        return None

    def aligned_evidence_source_keys(self, signal: Signal) -> List[str]:
        """List unique aligned evidence source keys from signal metadata."""
        if not isinstance(signal.metadata, dict):
            return []

        evidence = signal.metadata.get("evidence_sources")
        if not isinstance(evidence, list):
            return []

        direction = self.signal_direction(signal)
        if direction is None:
            return []

        aligned = set()
        for source in evidence:
            if not isinstance(source, dict):
                continue
            source_dir = str(source.get("direction", "")).strip().lower()
            if source_dir != direction:
                continue
            source_type = str(source.get("type", "")).strip().lower()
            source_name = str(source.get("name", "")).strip().lower()
            if not source_type or not source_name:
                continue
            aligned.add(f"{source_type}:{source_name}")

        return sorted(aligned)

    def confirming_source_stats(self, signal: Signal) -> Dict[str, Any]:
        """Resolve confirming-source count from ensemble metadata with evidence fallback."""
        aligned_keys = self.aligned_evidence_source_keys(signal)
        aligned_count = len(aligned_keys)

        confirming_count = aligned_count
        count_source = "aligned_evidence_sources"

        if isinstance(signal.metadata, dict):
            layer_scores = signal.metadata.get("layer_scores")
            if isinstance(layer_scores, dict):
                maybe_count = self._safe_float(layer_scores.get("confirming_sources"), None)
                if maybe_count is not None:
                    confirming_count = max(0, int(round(maybe_count)))
                    count_source = "ensemble_layer_scores"

        return {
            "confirming_sources": confirming_count,
            "aligned_evidence_sources": aligned_count,
            "aligned_source_keys": aligned_keys,
            "count_source": count_source,
        }

    @staticmethod
    def normalize_source_key(value: Any) -> Optional[str]:
        """Normalize a source key to `type:name` lower-case form."""
        if not isinstance(value, str):
            return None
        raw = value.strip().lower()
        if not raw or ":" not in raw:
            return None
        source_type, source_name = raw.split(":", 1)
        source_type = source_type.strip()
        source_name = source_name.strip()
        if not source_type or not source_name:
            return None
        return f"{source_type}:{source_name}"

    def extract_confirming_source_keys_from_metadata(
        self,
        signal_metadata: Dict[str, Any],
        side: str,
        strategy_name: str,
    ) -> List[str]:
        """Extract aligned evidence source keys from stored signal metadata."""
        if not isinstance(signal_metadata, dict):
            return []

        candidates: List[str] = []

        layer_scores = signal_metadata.get("layer_scores")
        if isinstance(layer_scores, dict):
            raw_keys = layer_scores.get("aligned_source_keys")
            if isinstance(raw_keys, list):
                for raw_key in raw_keys:
                    normalized = self.normalize_source_key(raw_key)
                    if normalized:
                        candidates.append(normalized)

        if not candidates:
            evidence = signal_metadata.get("evidence_sources")
            if isinstance(evidence, list):
                direction = "bullish" if side == "long" else "bearish" if side == "short" else None
                for source in evidence:
                    if not isinstance(source, dict):
                        continue
                    if direction:
                        source_dir = str(source.get("direction", "")).strip().lower()
                        if source_dir != direction:
                            continue
                    source_type = str(source.get("type", "")).strip().lower()
                    source_name = str(source.get("name", "")).strip().lower()
                    normalized = self.normalize_source_key(f"{source_type}:{source_name}")
                    if normalized:
                        candidates.append(normalized)

        primary_strategy_key = self.normalize_source_key(
            f"strategy:{self._canonical_strategy_key(strategy_name)}"
        )
        deduped: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if primary_strategy_key and candidate == primary_strategy_key:
                continue
            if candidate.startswith("strategy:"):
                continue
            deduped.append(candidate)

        return deduped

    def extract_raw_confidence_from_metadata(self, signal_metadata: Dict[str, Any]) -> float:
        """Resolve best available confidence proxy from stored signal metadata."""
        if not isinstance(signal_metadata, dict):
            return 50.0

        confidence = self._safe_float(signal_metadata.get("confidence"), None)
        if confidence is not None:
            return max(1.0, min(100.0, float(confidence)))

        adjustment = signal_metadata.get("confidence_adjustment")
        if isinstance(adjustment, dict):
            adjusted = self._safe_float(adjustment.get("adjusted_confidence"), None)
            if adjusted is not None:
                return max(1.0, min(100.0, float(adjusted)))

        layer_scores = signal_metadata.get("layer_scores")
        if isinstance(layer_scores, dict):
            strategy_score = self._safe_float(layer_scores.get("strategy_score"), None)
            if strategy_score is not None:
                return max(1.0, min(100.0, float(strategy_score)))

        return 50.0

    def passes_momentum_flow_delta_confirmation(self, signal: Signal) -> tuple[bool, Dict[str, Any]]:
        """Delta-divergence co-confirmation is disabled to avoid overfit gating."""
        strategy_key = self._canonical_strategy_key(signal.strategy_name or "")
        metrics: Dict[str, Any] = {
            "enabled": False,
            "strategy_key": strategy_key,
            "required_source": None,
            "reason": "momentum_flow_delta_divergence_filter_disabled",
            "passed": True,
        }
        return True, metrics

    @staticmethod
    def extract_confirming_sources(signal: Signal) -> Optional[int]:
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        if not isinstance(metadata, dict):
            return None

        layer_scores = metadata.get("layer_scores")
        if isinstance(layer_scores, dict):
            raw = layer_scores.get("confirming_sources")
            try:
                value = int(raw)
                return value if value > 0 else None
            except (TypeError, ValueError):
                pass

        ensemble = metadata.get("ensemble")
        if isinstance(ensemble, dict):
            raw = ensemble.get("confirming_sources")
            try:
                value = int(raw)
                return value if value > 0 else None
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def agreement_risk_multiplier(confirming_sources: Optional[int]) -> float:
        """Conservative risk scaling by source agreement."""
        if confirming_sources is None:
            return 1.0
        if confirming_sources <= 1:
            return 0.55
        if confirming_sources == 2:
            return 0.75
        return 1.0

    @staticmethod
    def trailing_multiplier(confirming_sources: Optional[int]) -> float:
        """Monotonic widening for higher-conviction entries."""
        if confirming_sources is None:
            return 1.0
        if confirming_sources >= 5:
            return 1.30
        if confirming_sources == 4:
            return 1.20
        if confirming_sources == 3:
            return 1.10
        return 1.0
