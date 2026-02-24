"""Intrabar slice evaluation helper extracted from runtime implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..day_trading_models import BarData, TradingSession
from ..day_trading_runtime_sweep import feature_vector_value
from ..strategies.base_strategy import Regime, SignalType


def _normalize_direction_token(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if token in {"bullish", "long", "buy", "up", "positive"}:
        return "bullish"
    if token in {"bearish", "short", "sell", "down", "negative"}:
        return "bearish"
    return ""


def _decision_direction(decision: Any) -> str:
    direction = _normalize_direction_token(getattr(decision, "direction", None))
    if direction:
        return direction
    signal = getattr(decision, "signal", None)
    signal_type = getattr(signal, "signal_type", None)
    if signal_type == SignalType.BUY:
        return "bullish"
    if signal_type == SignalType.SELL:
        return "bearish"
    return ""


def runtime_evaluate_intrabar_slice(
    self,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
) -> Dict[str, Any]:
    """
    Side-effect-free evaluation of a single intrabar slice.
    Used by the Strategy Analyzer to view full evidence tracing for any 5s checkpoint.
    """

    bars_data = list(session.bars[-100:]) if len(session.bars) >= 100 else list(session.bars)
    if bars_data and getattr(bars_data[-1], "timestamp", None) == bar.timestamp:
        bars_data[-1] = bar
    else:
        bars_data.append(bar)

    ohlcv = {
        "open": [b.open for b in bars_data],
        "high": [b.high for b in bars_data],
        "low": [b.low for b in bars_data],
        "close": [b.close for b in bars_data],
        "volume": [b.volume for b in bars_data],
    }

    indicators = self._calculate_indicators(bars_data, session=session)
    regime = session.detected_regime or Regime.MIXED
    flow_metrics = dict(indicators.get("order_flow") or {})

    orch = session.orchestrator
    if orch is None:
        return {"error": "Orchestrator not initialized"}

    # Per-checkpoint feature vector: recomputes price-sensitive indicators
    # (RSI, ROC, VWAP dist) from the checkpoint's OHLCV without mutating
    # rolling state.  Falls back to the minute-bar FV only if orchestrator
    # hasn't been warmed up yet.
    cp_bar_dict = {
        "open": bar.open, "high": bar.high, "low": bar.low,
        "close": bar.close, "volume": bar.volume,
        "vwap": getattr(bar, "vwap", None),
    }
    fv = orch.checkpoint_feature_vector(
        cp_bar_dict,
        pre_bar_fv=getattr(session, '_pre_bar_fv', None),
    ) or orch.current_feature_vector
    if fv is not None:
        l2_aggression_z = feature_vector_value(fv, "l2_aggression_z", 0.0)
        l2_book_pressure_z = feature_vector_value(fv, "l2_book_pressure_z", 0.0)
        flow_metrics["l2_aggression_z"] = l2_aggression_z
        flow_metrics["l2_book_pressure_z"] = l2_book_pressure_z

    ticker_cfg = getattr(self, "ticker_params", {}).get(session.ticker.upper(), {})
    is_long_only = bool(ticker_cfg.get("long_only", False))

    current_price = bar.close
    market_ts = self._to_market_time(timestamp)
    bar_time = market_ts.time()

    tod_boost = self.gate_engine.time_of_day_threshold_boost(bar_time)
    required_confirming_sources = self._required_confirming_sources(session, bar_time)

    def gen_signal_fn():
        original_bars = session.bars
        session.bars = bars_data
        try:
            return self._generate_signal(session, bar, timestamp)
        finally:
            session.bars = original_bars

    decision = orch.evidence_engine.evaluate(
        ohlcv=ohlcv,
        indicators=indicators,
        regime=regime,
        strategies=self.strategies,
        active_strategy_names=session.active_strategies,
        current_price=current_price,
        timestamp=timestamp,
        ticker=session.ticker,
        generate_signal_fn=gen_signal_fn,
        is_long_only=is_long_only,
        feature_vector=fv,
        regime_state=orch.current_regime_state,
        cross_asset_state=orch.current_cross_asset_state,
        time_of_day_boost=tod_boost,
    )

    effective_weights = self._resolve_evidence_weight_context(flow_metrics)
    headwind_boost, headwind_metrics = self.gate_engine.cross_asset_headwind_threshold_boost(
        cross_asset_state=orch.current_cross_asset_state,
        decision_direction=getattr(decision, "direction", None),
    )
    effective_trade_threshold = float(decision.threshold) + float(headwind_boost)
    threshold_used_reason = str(getattr(decision, "threshold_used_reason", "base_threshold"))
    if headwind_boost > 0.0:
        threshold_used_reason = f"{threshold_used_reason}+headwind({headwind_boost:.1f})"

    tcbbo_regime_override = self.gate_engine.tcbbo_directional_override(session)
    tcbbo_threshold_relief = 0.0
    tcbbo_override_direction = str(tcbbo_regime_override.get("direction", "neutral") or "neutral").lower()
    decision_direction = str(getattr(decision, "direction", "") or "").lower()
    if bool(tcbbo_regime_override.get("applied", False)) and decision_direction == tcbbo_override_direction:
        tcbbo_threshold_relief = max(
            2.0,
            min(8.0, float(getattr(session, "tcbbo_sweep_boost", 5.0) or 5.0)),
        )
        effective_trade_threshold = max(0.0, effective_trade_threshold - tcbbo_threshold_relief)
        threshold_used_reason = f"{threshold_used_reason}-tcbbo({tcbbo_threshold_relief:.1f})"

    golden_setup_payload = (
        dict(session.golden_setup_result)
        if isinstance(getattr(session, "golden_setup_result", None), dict)
        else {}
    )
    golden_setup_relief = 0.0
    golden_setup_direction = _normalize_direction_token(golden_setup_payload.get("best_direction"))
    gate_direction = _decision_direction(decision)
    if (
        bool(golden_setup_payload.get("active", False))
        and bool(golden_setup_direction)
        and (not gate_direction or gate_direction == golden_setup_direction)
    ):
        golden_setup_relief = max(
            0.0,
            float(golden_setup_payload.get("threshold_reduction", 0.0) or 0.0),
        )
    if golden_setup_relief > 0.0:
        effective_trade_threshold = max(
            0.0,
            effective_trade_threshold - golden_setup_relief,
        )
        threshold_used_reason = f"{threshold_used_reason}-golden({golden_setup_relief:.1f})"

    passed_trade_threshold = float(decision.combined_score) >= effective_trade_threshold

    # Strategy-aware threshold corrections (second code path)
    if decision.signal:
        _sig_strat2 = str(decision.signal.strategy_name or "").strip().lower()
        _midday_relaxed_strats2 = {"mean_reversion", "rotation", "vwap_magnet", "volumeprofile"}
        if tod_boost > 0 and _sig_strat2 in _midday_relaxed_strats2:
            _strategy_tod2 = self.gate_engine.time_of_day_threshold_boost(
                bar_time,
                strategy_key=_sig_strat2,
            )
            _tod_relief2 = tod_boost - _strategy_tod2
            if _tod_relief2 > 0:
                effective_trade_threshold = max(0.0, effective_trade_threshold - _tod_relief2)
                tod_boost = _strategy_tod2
                threshold_used_reason = f"{threshold_used_reason}-midday_relief({_tod_relief2:.0f})"
        _headwind_contrarian2 = {"mean_reversion", "absorption_reversal", "rotation"}
        if headwind_boost > 2.0 and _sig_strat2 in _headwind_contrarian2:
            _hw_relief2 = headwind_boost - 2.0
            headwind_boost = 2.0
            effective_trade_threshold = max(0.0, effective_trade_threshold - _hw_relief2)
            threshold_used_reason = f"{threshold_used_reason}-hw_relief({_hw_relief2:.1f})"
        passed_trade_threshold = float(decision.combined_score) >= effective_trade_threshold

    combined_raw = float(getattr(decision, "combined_raw", decision.combined_score) or 0.0)
    combined_norm = getattr(decision, "combined_norm_0_100", None)

    layer_scores = {
        "schema_version": 2,
        "strategy_score": round(decision.strategy_score, 1),
        "combined_score": round(decision.combined_score, 1),
        "combined_raw": round(combined_raw, 1),
        "combined_norm_0_100": round(float(combined_norm), 1) if combined_norm is not None else None,
        "threshold": decision.threshold,
        "trade_gate_threshold": float(effective_trade_threshold),
        "threshold_used": float(effective_trade_threshold),
        "threshold_used_reason": threshold_used_reason,
        "strategy_weight": round(float(effective_weights["strategy_weight"]), 3),
        "strategy_weight_source": effective_weights["strategy_weight_source"],
        "l2_has_coverage": bool(effective_weights["l2_has_coverage"]),
        "l2_quality_ok": bool(effective_weights["l2_quality_ok"]),
        "l2_coverage_ratio": round(float(effective_weights["l2_coverage_ratio"]), 3),
        "flow_score": round(float(flow_metrics.get("flow_score", 0.0) or 0.0), 1),
        "l2_aggression_z": round(float(flow_metrics.get("l2_aggression_z", 0.0) or 0.0), 3),
        "l2_book_pressure_z": round(float(flow_metrics.get("l2_book_pressure_z", 0.0) or 0.0), 3),
        "book_pressure_avg": round(float(flow_metrics.get("book_pressure_avg", 0.0) or 0.0), 3),
        "book_pressure_trend": round(float(flow_metrics.get("book_pressure_trend", 0.0) or 0.0), 3),
        "large_trader_activity": round(float(flow_metrics.get("large_trader_activity", 0.0) or 0.0), 3),
        "vwap_execution_flow": round(float(flow_metrics.get("vwap_execution_flow", 0.0) or 0.0), 3),
        "micro_regime": session.micro_regime,
        "passed": bool(decision.execute and passed_trade_threshold),
        "tod_threshold_boost": tod_boost,
        "headwind_threshold_boost": round(float(headwind_boost), 4),
        "tcbbo_threshold_relief": round(float(tcbbo_threshold_relief), 4),
        "golden_setup_relief": round(float(golden_setup_relief), 4),
        "headwind_activation_score": float(getattr(self, "headwind_activation_score", 0.5)),
        "cross_asset_headwind": headwind_metrics,
        "tcbbo_regime_override": dict(tcbbo_regime_override),
        "golden_setup": dict(golden_setup_payload) if golden_setup_payload else {},
        "required_confirming_sources": required_confirming_sources,
        "engine": "evidence_v1",
    }

    result = {
        "timestamp": timestamp.isoformat(),
        "layer_scores": layer_scores,
        "_decision_execute": bool(getattr(decision, "execute", False)),
        "_passed_trade_threshold": bool(passed_trade_threshold),
        "_decision_reasoning": str(getattr(decision, "reasoning", "") or ""),
        "_decision_direction": getattr(decision, "direction", None),
        "_combined_score_raw": float(getattr(decision, "combined_score", 0.0) or 0.0),
    }

    if decision.signal:
        signal = decision.signal
        confirming_stats = self._confirming_source_stats(signal)
        layer_scores["confirming_sources"] = confirming_stats["confirming_sources"]
        layer_scores["confirming_sources_source"] = confirming_stats["count_source"]
        layer_scores["aligned_evidence_sources"] = confirming_stats["aligned_evidence_sources"]
        layer_scores["aligned_source_keys"] = confirming_stats["aligned_source_keys"]
        result["signal"] = signal.to_dict()
        result["_raw_signal"] = signal

        cand_diag = signal.metadata.get("candidate_diagnostics")
        if cand_diag:
            result["candidate_diagnostics"] = cand_diag

    if decision.combined_score > 0 and (not decision.execute or not passed_trade_threshold):
        result["signal_rejected"] = {
            "gate": "cross_asset_headwind" if (decision.execute and not passed_trade_threshold) else "threshold",
            "schema_version": 2,
            "combined_score": round(decision.combined_score, 1),
            "combined_raw": round(float(getattr(decision, "combined_raw", decision.combined_score) or 0.0), 1),
            "combined_norm_0_100": round(float(getattr(decision, "combined_norm_0_100", 0.0) or 0.0), 1),
            "threshold_used": effective_trade_threshold,
            "trade_gate_threshold": effective_trade_threshold,
            "threshold_used_reason": threshold_used_reason,
            "strategy_score": round(decision.strategy_score, 1),
            "regime": regime.value if regime else None,
            "micro_regime": session.micro_regime,
            "reasoning": decision.reasoning,
            "tod_threshold_boost": tod_boost,
            "headwind_threshold_boost": round(float(headwind_boost), 4),
            "cross_asset_headwind": headwind_metrics,
            "timestamp": timestamp.isoformat(),
        }

    if "candidate_diagnostics" not in result and hasattr(decision, "confirming_signals") and decision.confirming_signals:
        for sig in decision.confirming_signals:
            if isinstance(sig.metadata, dict) and "candidate_diagnostics" in sig.metadata:
                result["candidate_diagnostics"] = sig.metadata["candidate_diagnostics"]
                break

    return result
