"""Signal confirmation and momentum gate implementations extracted from DayTradingManager."""

from __future__ import annotations

from datetime import time
from typing import Any, Dict, List, Optional, Tuple

from .day_trading_models import BarData, DayTrade, TradingSession
from .exit_policy_engine import ExitPolicyEngine
from .strategies.base_strategy import Position, Signal, SignalType
from .day_trading_gate.thresholds import (
    cross_asset_headwind_threshold_boost as _cross_asset_headwind_threshold_boost_impl,
    time_of_day_threshold_boost as _time_of_day_threshold_boost_impl,
)

class GateEvaluationEngine:
    def __init__(
        self,
        exit_engine: ExitPolicyEngine,
        config_service: Any,
        evidence_service: Any,
        default_momentum_strategies: Tuple[str, ...],
        *,
        manager: Any = None,
        ticker_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.exit_engine = exit_engine
        self.config_service = config_service
        self.evidence_service = evidence_service
        self.default_momentum_strategies = tuple(default_momentum_strategies)
        # Keep legacy attribute name for refactored callers.
        self.DEFAULT_MOMENTUM_STRATEGIES = tuple(default_momentum_strategies)
        self.manager = manager
        self._ticker_params = ticker_params if isinstance(ticker_params, dict) else {}

    def __getattr__(self, name: str) -> Any:
        if name == "ticker_params":
            manager = self.__dict__.get("manager")
            if manager is not None and hasattr(manager, "ticker_params"):
                return getattr(manager, "ticker_params")
            return self.__dict__.get("_ticker_params", {})

        manager = self.__dict__.get("manager")
        if manager is not None and hasattr(manager, name):
            return getattr(manager, name)
        raise AttributeError(f"{self.__class__.__name__!s} object has no attribute {name!r}")

    def evaluate_momentum_diversification_gate_candidate(
            self,
            *,
        momentum_cfg: Dict[str, Any],
        strategy_key: str,
        micro_regime: str,
        direction: float,
        flow_metrics: Dict[str, Any],
        selected_sleeve_id: Optional[str] = None,
    ) -> tuple[bool, Dict[str, Any]]:
        apply_to = self._normalize_strategy_list(
            momentum_cfg.get("apply_to_strategies"),
            fallback=list(self.DEFAULT_MOMENTUM_STRATEGIES),
        )
        metrics: Dict[str, Any] = {
            "enabled": bool(momentum_cfg.get("enabled", False)),
            "strategy_key": strategy_key,
            "apply_to_strategies": apply_to,
            "micro_regime": micro_regime,
            "selected_sleeve_id": selected_sleeve_id,
            "passed": True,
            "applicable": True,
        }
        if not metrics["enabled"] or strategy_key not in apply_to:
            metrics["applicable"] = False
            return True, metrics

        blocked_micro = set(momentum_cfg.get("blocked_micro_regimes", []))
        allowed_micro = set(momentum_cfg.get("allowed_micro_regimes", []))
        if micro_regime in blocked_micro:
            metrics.update(
                {
                    "passed": False,
                    "reason": "momentum_diversification_blocked_micro_regime",
                    "blocked_micro_regimes": sorted(blocked_micro),
                }
            )
            return False, metrics
        if allowed_micro and micro_regime not in allowed_micro:
            metrics.update(
                {
                    "passed": False,
                    "reason": "momentum_diversification_micro_regime_not_allowed",
                    "allowed_micro_regimes": sorted(allowed_micro),
                }
            )
            return False, metrics

        has_l2 = bool(flow_metrics.get("has_l2_coverage", False))
        if bool(momentum_cfg.get("require_l2_coverage", True)) and not has_l2:
            metrics.update(
                {
                    "passed": False,
                    "reason": "momentum_diversification_l2_required",
                    "has_l2_coverage": has_l2,
                }
            )
            return False, metrics

        directional_signed_aggression = float(flow_metrics.get("signed_aggression", 0.0) or 0.0) * direction
        directional_imbalance = float(flow_metrics.get("imbalance_avg", 0.0) or 0.0) * direction
        directional_cvd = float(flow_metrics.get("cumulative_delta", 0.0) or 0.0) * direction
        directional_delta_acceleration = float(flow_metrics.get("delta_acceleration", 0.0) or 0.0) * direction
        directional_price_change_pct = float(flow_metrics.get("price_change_pct", 0.0) or 0.0) * direction
        directional_consistency = float(flow_metrics.get("directional_consistency", 0.0) or 0.0)
        price_trend_efficiency = float(flow_metrics.get("price_trend_efficiency", 0.0) or 0.0)
        last_bar_body_ratio = float(flow_metrics.get("latest_bar_body_ratio", 0.0) or 0.0)
        last_bar_close_location_raw = float(flow_metrics.get("latest_bar_close_location", 0.5) or 0.5)
        directional_last_bar_close_location = (
            last_bar_close_location_raw if direction > 0 else (1.0 - last_bar_close_location_raw)
        )
        flow_score = float(flow_metrics.get("flow_score", 0.0) or 0.0)

        min_flow_score = max(0.0, min(100.0, float(momentum_cfg.get("min_flow_score", 55.0))))
        min_consistency = max(0.0, min(1.0, float(momentum_cfg.get("min_directional_consistency", 0.35))))
        min_signed = max(0.0, min(1.0, float(momentum_cfg.get("min_signed_aggression", 0.03))))
        min_imbalance = max(0.0, min(1.0, float(momentum_cfg.get("min_imbalance", 0.02))))
        min_cvd = float(momentum_cfg.get("min_cvd", 0.0))
        min_directional_price_change_pct = float(
            momentum_cfg.get("min_directional_price_change_pct", 0.0)
        )
        min_price_trend_efficiency = max(
            0.0, min(1.0, float(momentum_cfg.get("min_price_trend_efficiency", 0.0)))
        )
        min_last_bar_body_ratio = max(
            0.0, min(1.0, float(momentum_cfg.get("min_last_bar_body_ratio", 0.0)))
        )
        min_last_bar_close_location = max(
            0.0, min(1.0, float(momentum_cfg.get("min_last_bar_close_location", 0.0)))
        )
        min_delta_accel = float(momentum_cfg.get("min_delta_acceleration", 0.0))

        if has_l2:
            passes_flow = flow_score >= min_flow_score
            passes_consistency = directional_consistency >= min_consistency
            passes_signed = directional_signed_aggression >= min_signed
            passes_imbalance = directional_imbalance >= min_imbalance
            passes_cvd = directional_cvd >= min_cvd
            passes_delta_accel = directional_delta_acceleration >= min_delta_accel
        else:
            # Preserve backward compatibility when L2 is absent and L2 coverage is optional.
            passes_flow = True
            passes_consistency = True
            passes_signed = True
            passes_imbalance = True
            passes_cvd = True
            passes_delta_accel = True

        passes_directional_price_change = (
            directional_price_change_pct >= min_directional_price_change_pct
        )
        passes_price_trend_efficiency = price_trend_efficiency >= min_price_trend_efficiency
        passes_last_bar_body_ratio = last_bar_body_ratio >= min_last_bar_body_ratio
        passes_last_bar_close_location = (
            directional_last_bar_close_location >= min_last_bar_close_location
        )

        gate_mode = str(momentum_cfg.get("gate_mode", "weighted")).lower()

        if gate_mode == "weighted" and has_l2:
            # Weighted scoring: partial credit instead of all-or-nothing AND gate.
            # Each metric contributes proportionally; one weak dimension doesn't kill the signal.
            def _norm(value: float, threshold: float, scale: float = 2.0) -> float:
                """Normalize value relative to threshold. 1.0 = at threshold, capped at [0, 1]."""
                if threshold <= 0:
                    return 1.0 if value >= 0 else max(0.0, min(1.0, 0.5 + value))
                return max(0.0, min(1.0, value / (threshold * scale)))

            gate_score = (
                0.20 * _norm(flow_score, min_flow_score, 2.0)
                + 0.18 * _norm(directional_consistency, min_consistency, 2.0)
                + 0.15 * _norm(directional_signed_aggression, min_signed, 2.0)
                + 0.12 * _norm(directional_imbalance, min_imbalance, 2.0)
                + 0.10 * _norm(directional_cvd, max(min_cvd, 1.0), 2.0)
                + 0.08 * _norm(directional_delta_acceleration, max(min_delta_accel, 0.01), 2.0)
                + 0.09 * _norm(directional_price_change_pct, max(min_directional_price_change_pct, 0.01), 2.0)
                + 0.07 * _norm(price_trend_efficiency, max(min_price_trend_efficiency, 0.1), 2.0)
                + 0.05 * _norm(last_bar_body_ratio, max(min_last_bar_body_ratio, 0.1), 2.0)
                + 0.04 * _norm(directional_last_bar_close_location, max(min_last_bar_close_location, 0.1), 2.0)
            )
            gate_threshold = max(0.0, min(1.0, float(momentum_cfg.get("gate_threshold", 0.55))))
            # Hard floor: flow_score must be minimally present to prevent zero-flow signals
            flow_floor = max(0.0, float(momentum_cfg.get("gate_flow_floor", 40.0)))
            passes_flow_floor = flow_score >= flow_floor

            passed = bool(gate_score >= gate_threshold and passes_flow_floor)
        else:
            # Legacy all-pass mode
            gate_score = -1.0
            gate_threshold = -1.0
            passes_flow_floor = True
            passed = bool(
                passes_flow
                and passes_consistency
                and passes_signed
                and passes_imbalance
                and passes_cvd
                and passes_delta_accel
                and passes_directional_price_change
                and passes_price_trend_efficiency
                and passes_last_bar_body_ratio
                and passes_last_bar_close_location
            )
        metrics.update(
            {
                "has_l2_coverage": has_l2,
                "gate_mode": gate_mode,
                "gate_score": round(gate_score, 4) if gate_score >= 0 else None,
                "gate_threshold": round(gate_threshold, 4) if gate_threshold >= 0 else None,
                "passes_flow_floor": passes_flow_floor,
                "flow_score": flow_score,
                "min_flow_score": min_flow_score,
                "directional_consistency": directional_consistency,
                "min_directional_consistency": min_consistency,
                "directional_signed_aggression": directional_signed_aggression,
                "min_signed_aggression": min_signed,
                "directional_imbalance": directional_imbalance,
                "min_imbalance": min_imbalance,
                "directional_cvd": directional_cvd,
                "min_cvd": min_cvd,
                "directional_delta_acceleration": directional_delta_acceleration,
                "min_delta_acceleration": min_delta_accel,
                "directional_price_change_pct": directional_price_change_pct,
                "min_directional_price_change_pct": min_directional_price_change_pct,
                "price_trend_efficiency": price_trend_efficiency,
                "min_price_trend_efficiency": min_price_trend_efficiency,
                "last_bar_body_ratio": last_bar_body_ratio,
                "min_last_bar_body_ratio": min_last_bar_body_ratio,
                "directional_last_bar_close_location": directional_last_bar_close_location,
                "min_last_bar_close_location": min_last_bar_close_location,
                "passes_flow_score": passes_flow,
                "passes_directional_consistency": passes_consistency,
                "passes_signed_aggression": passes_signed,
                "passes_imbalance": passes_imbalance,
                "passes_cvd": passes_cvd,
                "passes_delta_acceleration": passes_delta_accel,
                "passes_directional_price_change_pct": passes_directional_price_change,
                "passes_price_trend_efficiency": passes_price_trend_efficiency,
                "passes_last_bar_body_ratio": passes_last_bar_body_ratio,
                "passes_last_bar_close_location": passes_last_bar_close_location,
                "passed": passed,
            }
        )
        if not passed:
            metrics["reason"] = "momentum_diversification_gate_failed"
        return passed, metrics

    def passes_momentum_diversification_gate(
        self,
        session: TradingSession,
        signal: Signal,
        flow_metrics: Dict[str, Any],
    ) -> tuple[bool, Dict[str, Any]]:
        ticker_cfg = self.ticker_params.get(session.ticker.upper(), {})
        adaptive_cfg = self._normalize_adaptive_config(ticker_cfg.get("adaptive"))
        momentum_cfg = self._resolve_momentum_diversification(session, adaptive_cfg)
        strategy_key = self._canonical_strategy_key(signal.strategy_name or "")
        micro_regime = (session.micro_regime or "MIXED").upper()
        direction = 1.0 if signal.signal_type == SignalType.BUY else -1.0

        sleeves_raw = momentum_cfg.get("sleeves")
        sleeves = [item for item in sleeves_raw if isinstance(item, dict)] if isinstance(sleeves_raw, list) else []
        candidate_cfgs: List[Tuple[Dict[str, Any], Optional[str]]] = []
        if sleeves:
            for idx, sleeve in enumerate(sleeves):
                sleeve_id = self._normalize_momentum_sleeve_id(
                    sleeve.get("sleeve_id"),
                    fallback=f"sleeve_{idx + 1}",
                )
                candidate_cfgs.append((sleeve, sleeve_id))
        else:
            candidate_cfgs.append(
                (self._normalize_momentum_diversification_config(momentum_cfg, include_sleeves=False), None)
            )

        applicable_failures: List[Dict[str, Any]] = []
        for candidate_cfg, sleeve_id in candidate_cfgs:
            passed, metrics = self.evaluate_momentum_diversification_gate_candidate(
                momentum_cfg=candidate_cfg,
                strategy_key=strategy_key,
                micro_regime=micro_regime,
                direction=direction,
                flow_metrics=flow_metrics,
                selected_sleeve_id=sleeve_id,
            )
            if metrics.get("applicable") and passed:
                if sleeve_id:
                    metrics["sleeve_mode"] = "multi"
                return True, metrics
            if metrics.get("applicable") and not passed:
                applicable_failures.append(metrics)

        if not applicable_failures:
            # No sleeve applied to this strategy/context -> keep backward-compatible pass-through.
            fallback_apply_to = self._normalize_strategy_list(
                momentum_cfg.get("apply_to_strategies"),
                fallback=list(self.DEFAULT_MOMENTUM_STRATEGIES),
            )
            return True, {
                "enabled": bool(momentum_cfg.get("enabled", False)),
                "strategy_key": strategy_key,
                "apply_to_strategies": fallback_apply_to,
                "micro_regime": micro_regime,
                "passed": True,
                "applicable": False,
                "sleeve_mode": "multi" if sleeves else "single",
            }

        def _failure_rank(item: Dict[str, Any]) -> int:
            score = 0
            for key in (
                "passes_flow_score",
                "passes_directional_consistency",
                "passes_signed_aggression",
                "passes_imbalance",
                "passes_cvd",
                "passes_delta_acceleration",
                "passes_directional_price_change_pct",
                "passes_price_trend_efficiency",
                "passes_last_bar_body_ratio",
                "passes_last_bar_close_location",
            ):
                if bool(item.get(key)):
                    score += 1
            return score

        best_failure = max(applicable_failures, key=_failure_rank)
        if sleeves:
            best_failure["sleeve_mode"] = "multi"
        return False, best_failure

    def should_momentum_fail_fast_exit(
        self,
        session: TradingSession,
        pos: Position,
        current_bar_index: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        ticker_cfg = self.ticker_params.get(session.ticker.upper(), {})
        adaptive_cfg = self._normalize_adaptive_config(ticker_cfg.get("adaptive"))
        momentum_cfg_all = self._resolve_momentum_diversification(session, adaptive_cfg)
        strategy_key = self._canonical_strategy_key(pos.strategy_name or "")
        signal_md = pos.signal_metadata if isinstance(pos.signal_metadata, dict) else {}
        momentum_md = signal_md.get("momentum_diversification") if isinstance(signal_md, dict) else {}
        preferred_sleeve_id = ""
        if isinstance(momentum_md, dict):
            preferred_sleeve_id = str(
                momentum_md.get("selected_sleeve_id")
                or momentum_md.get("sleeve_id")
                or ""
            ).strip()

        flow = self._calculate_order_flow_metrics(session.bars, lookback=min(8, len(session.bars)))
        selected_cfg, selected_sleeve_id, _ = self._select_momentum_sleeve(
            momentum_cfg_all,
            strategy_key=strategy_key,
            micro_regime=(session.micro_regime or "MIXED").upper(),
            has_l2_coverage=bool(flow.get("has_l2_coverage", False)),
            preferred_sleeve_id=preferred_sleeve_id,
        )
        apply_to = self._normalize_strategy_list(
            selected_cfg.get("apply_to_strategies"),
            fallback=list(self.DEFAULT_MOMENTUM_STRATEGIES),
        )
        should_exit, metrics = self.exit_engine.should_momentum_fail_fast_exit(
            pos=pos,
            current_bar_index=current_bar_index,
            flow_metrics=flow,
            momentum_config=selected_cfg,
            momentum_sleeve_id=selected_sleeve_id or "",
            apply_to_strategies=apply_to,
        )
        metrics["strategy_key"] = strategy_key
        metrics["selected_sleeve_id"] = selected_sleeve_id
        return should_exit, metrics

    @staticmethod
    def time_of_day_threshold_boost(bar_time: time, strategy_key: str = "") -> float:
        return _time_of_day_threshold_boost_impl(bar_time, strategy_key=strategy_key)

    @staticmethod
    def cross_asset_headwind_threshold_boost(
        cross_asset_state: Any,
        decision_direction: Optional[str],
        *,
        activation_score: float = 0.5,
        min_boost: float = 5.0,
        max_boost: float = 10.0,
        strategy_key: str = "",
    ) -> tuple[float, Dict[str, Any]]:
        return _cross_asset_headwind_threshold_boost_impl(
            cross_asset_state=cross_asset_state,
            decision_direction=decision_direction,
            activation_score=activation_score,
            min_boost=min_boost,
            max_boost=max_boost,
            strategy_key=strategy_key,
        )

    def build_position_closed_payload(
        self,
        trade: DayTrade,
        exit_reason: str,
        bars_held: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.exit_engine.build_position_closed_payload(
            trade=trade,
            exit_reason=exit_reason,
            bars_held=bars_held,
        )

    @staticmethod
    def bar_has_l2_data(bar: BarData) -> bool:
        return (
            bar.l2_delta is not None
            or bar.l2_imbalance is not None
            or bar.l2_volume is not None
            or bar.l2_book_pressure is not None
            or bar.l2_bid_depth_total is not None
            or bar.l2_ask_depth_total is not None
            or bar.l2_iceberg_bias is not None
        )

    def _canonical_signal_strategy_key(self, signal: Signal) -> str:
        raw = str(getattr(signal, "strategy_name", "") or "")
        canonicalizer = getattr(self, "_canonical_strategy_key", None)
        if callable(canonicalizer):
            try:
                canonical = canonicalizer(raw)
                return str(canonical or "").strip().lower()
            except Exception:
                pass
        return raw.strip().lower().replace(" ", "_")

    def passes_l2_confirmation(
        self,
        session: TradingSession,
        signal: Signal,
        flow_metrics: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Optional order-flow confirmation gate.

        Uses only current/past bars (no look-ahead):
        - summed delta over lookback
        - mean imbalance over lookback
        - summed iceberg bias over lookback
        """
        flow_ctx = flow_metrics if isinstance(flow_metrics, dict) else {}
        strategy_key = self._canonical_signal_strategy_key(signal)
        l2_book_pressure_z = self._to_float(flow_ctx.get("l2_book_pressure_z"), 0.0)
        book_pressure_block_z_threshold = 2.5

        metrics: Dict[str, Any] = {
            "enabled": bool(session.l2_confirm_enabled),
            "strategy_key": strategy_key,
            "lookback_bars": max(1, int(session.l2_lookback_bars)),
            "min_delta": float(session.l2_min_delta),
            "min_imbalance": float(session.l2_min_imbalance),
            "min_iceberg_bias": float(session.l2_min_iceberg_bias),
            "min_participation_ratio": float(session.l2_min_participation_ratio),
            "min_directional_consistency": float(session.l2_min_directional_consistency),
            "min_signed_aggression": float(session.l2_min_signed_aggression),
            "l2_book_pressure_z": l2_book_pressure_z,
            "book_pressure_block_z_threshold": book_pressure_block_z_threshold,
            "passed": True,
        }

        if not session.l2_confirm_enabled:
            return True, metrics

        lookback = metrics["lookback_bars"]
        window = session.bars[-lookback:] if session.bars else []
        if not window:
            metrics.update({"passed": False, "reason": "l2_window_empty"})
            return False, metrics

        deltas = [self._to_float(b.l2_delta, 0.0) for b in window]
        imbalances = [self._to_float(b.l2_imbalance, 0.0) for b in window]
        iceberg_biases = [self._to_float(b.l2_iceberg_bias, 0.0) for b in window]

        bars_with_l2 = sum(
            1 for b in window
            if (b.l2_delta is not None)
            or (b.l2_imbalance is not None)
            or (b.l2_iceberg_bias is not None)
        )
        has_any_l2 = bars_with_l2 > 0
        has_l2_coverage = bars_with_l2 >= max(3, len(window) // 2)
        metrics["has_l2_coverage"] = bool(has_l2_coverage)
        metrics["bars_with_l2"] = bars_with_l2
        if not has_any_l2:
            metrics.update({"passed": False, "reason": "l2_data_missing"})
            return False, metrics

        delta_sum = float(sum(deltas))
        imbalance_avg = float(sum(imbalances) / len(imbalances)) if imbalances else 0.0
        iceberg_bias_sum = float(sum(iceberg_biases))
        l2_volumes = [max(0.0, self._to_float(b.l2_volume, 0.0)) for b in window]
        bar_volumes = [max(0.0, self._to_float(b.volume, 0.0)) for b in window]

        direction = 1.0 if signal.signal_type == SignalType.BUY else -1.0
        directional_delta = delta_sum * direction
        directional_imbalance = imbalance_avg * direction
        directional_iceberg_bias = iceberg_bias_sum * direction

        participation_samples: List[float] = []
        signed_aggression_samples: List[float] = []
        directional_consistency_base = 0
        directional_consistency_hits = 0

        for b, l2_vol, bar_vol in zip(window, l2_volumes, bar_volumes):
            if l2_vol > 0 and bar_vol > 0:
                participation_samples.append(l2_vol / bar_vol)

            delta_val = self._to_float(b.l2_delta, 0.0)
            if l2_vol > 0:
                signed_aggression_samples.append((delta_val / l2_vol) * direction)

            # Consistency check uses delta sign first, then imbalance sign fallback.
            if abs(delta_val) > 1e-9:
                directional_consistency_base += 1
                if (delta_val * direction) > 0:
                    directional_consistency_hits += 1
                continue

            if b.l2_imbalance is not None:
                imb_val = self._to_float(b.l2_imbalance, 0.0)
                if abs(imb_val) > 1e-9:
                    directional_consistency_base += 1
                    if (imb_val * direction) > 0:
                        directional_consistency_hits += 1

        participation_avg = (
            float(sum(participation_samples) / len(participation_samples))
            if participation_samples else 0.0
        )
        signed_aggression_avg = (
            float(sum(signed_aggression_samples) / len(signed_aggression_samples))
            if signed_aggression_samples else 0.0
        )
        directional_consistency = (
            float(directional_consistency_hits / directional_consistency_base)
            if directional_consistency_base > 0 else 0.0
        )

        min_delta_val = float(session.l2_min_delta)
        min_imbalance_val = float(session.l2_min_imbalance)
        min_iceberg_val = float(session.l2_min_iceberg_bias)
        min_participation_val = float(session.l2_min_participation_ratio)
        min_consistency_val = float(session.l2_min_directional_consistency)
        min_signed_agg_val = float(session.l2_min_signed_aggression)

        passes_delta = directional_delta >= min_delta_val
        passes_imbalance = directional_imbalance >= min_imbalance_val
        passes_iceberg = directional_iceberg_bias >= min_iceberg_val
        passes_participation = participation_avg >= min_participation_val
        passes_consistency = directional_consistency >= min_consistency_val
        passes_signed_aggression = signed_aggression_avg >= min_signed_agg_val

        l2_gate_mode = str(getattr(session, "l2_gate_mode", "weighted") or "weighted").lower()

        if l2_gate_mode == "weighted":
            def _l2_norm(value: float, threshold: float, scale: float = 2.0) -> float:
                if threshold <= 0:
                    return 1.0 if value >= 0 else max(0.0, min(1.0, 0.5 + value))
                return max(0.0, min(1.0, value / (threshold * scale)))

            l2_gate_score = (
                0.25 * _l2_norm(directional_delta, max(min_delta_val, 1.0), 2.0)
                + 0.22 * _l2_norm(signed_aggression_avg, max(min_signed_agg_val, 0.01), 2.0)
                + 0.20 * _l2_norm(directional_consistency, max(min_consistency_val, 0.1), 2.0)
                + 0.20 * _l2_norm(directional_imbalance, max(min_imbalance_val, 0.01), 2.0)
                + 0.08 * _l2_norm(participation_avg, max(min_participation_val, 0.01), 2.0)
                + 0.05 * _l2_norm(directional_iceberg_bias, max(min_iceberg_val, 0.1), 2.0)
            )
            l2_gate_threshold = max(0.0, min(1.0, float(
                getattr(session, "l2_gate_threshold", 0.50) or 0.50
            )))
            passed = bool(l2_gate_score >= l2_gate_threshold)
        else:
            l2_gate_score = -1.0
            l2_gate_threshold = -1.0
            passed = bool(
                passes_delta
                and passes_imbalance
                and passes_iceberg
                and passes_participation
                and passes_consistency
                and passes_signed_aggression
            )

        metrics.update({
            "window_size": len(window),
            "l2_gate_mode": l2_gate_mode,
            "l2_gate_score": round(l2_gate_score, 4) if l2_gate_score >= 0 else None,
            "l2_gate_threshold": round(l2_gate_threshold, 4) if l2_gate_threshold >= 0 else None,
            "delta_sum": delta_sum,
            "imbalance_avg": imbalance_avg,
            "iceberg_bias_sum": iceberg_bias_sum,
            "participation_avg": participation_avg,
            "directional_consistency": directional_consistency,
            "signed_aggression_avg": signed_aggression_avg,
            "directional_delta": directional_delta,
            "directional_imbalance": directional_imbalance,
            "directional_iceberg_bias": directional_iceberg_bias,
            "passes_delta": passes_delta,
            "passes_imbalance": passes_imbalance,
            "passes_iceberg": passes_iceberg,
            "passes_participation": passes_participation,
            "passes_consistency": passes_consistency,
            "passes_signed_aggression": passes_signed_aggression,
        })

        soft_gate_passed = bool(passed)
        is_long_signal = signal.signal_type == SignalType.BUY
        is_short_signal = signal.signal_type == SignalType.SELL

        book_pressure_block_long = (
            is_long_signal and l2_book_pressure_z < -book_pressure_block_z_threshold
        )
        book_pressure_block_short = (
            is_short_signal and l2_book_pressure_z > book_pressure_block_z_threshold
        )

        hard_block_reason: Optional[str] = None
        if book_pressure_block_long:
            hard_block_reason = "book_pressure_block_long"
        elif book_pressure_block_short:
            hard_block_reason = "book_pressure_block_short"
        # L2 acts as an emergency brake only; soft score diagnostics stay informational.
        passed = bool(hard_block_reason is None)
        metrics.update(
            {
                "l2_effective_mode": "hard_block_only",
                "l2_soft_gate_passed": soft_gate_passed,
                "book_pressure_block_long": bool(book_pressure_block_long),
                "book_pressure_block_short": bool(book_pressure_block_short),
                "passes_book_pressure_block": bool(
                    not book_pressure_block_long and not book_pressure_block_short
                ),
                "hard_block": bool(hard_block_reason),
                "passed": passed,
            }
        )

        if hard_block_reason:
            metrics["reason"] = hard_block_reason
        return passed, metrics

    def tcbbo_directional_override(self, session: TradingSession) -> Dict[str, Any]:
        """Strong options flow can override conservative regime blocks."""
        lookback = max(1, int(getattr(session, "tcbbo_lookback_bars", 5) or 5))
        metrics: Dict[str, Any] = {
            "enabled": bool(getattr(session, "tcbbo_gate_enabled", False)),
            "lookback_bars": lookback,
            "applied": False,
            "direction": "neutral",
            "window_net_premium": 0.0,
            "cumulative_net_premium": 0.0,
            "override_threshold": 0.0,
            "override_multiplier": 4.0,
            "override_floor": 500_000.0,
        }
        if not metrics["enabled"]:
            metrics["reason"] = "tcbbo_disabled"
            return metrics

        window = session.bars[-lookback:] if session.bars else []
        if not window:
            metrics["reason"] = "tcbbo_window_empty"
            return metrics

        covered = [b for b in window if bool(getattr(b, "tcbbo_has_data", False))]
        if not covered:
            metrics["reason"] = "tcbbo_data_missing"
            return metrics

        window_net = sum(
            self._to_float(getattr(b, "tcbbo_net_premium", None), 0.0)
            for b in covered
        )
        cumulative_net = self._to_float(
            getattr(covered[-1], "tcbbo_cumulative_net_premium", None),
            0.0,
        )
        direction = "neutral"
        if window_net > 0:
            direction = "bullish"
        elif window_net < 0:
            direction = "bearish"
        elif cumulative_net > 0:
            direction = "bullish"
        elif cumulative_net < 0:
            direction = "bearish"

        base_min = max(0.0, float(getattr(session, "tcbbo_min_net_premium", 0.0) or 0.0))
        override_multiplier = float(metrics["override_multiplier"])
        override_floor = float(metrics["override_floor"])
        override_threshold = max(base_min * override_multiplier, override_floor)
        effective_premium = max(abs(window_net), abs(cumulative_net))
        applied = bool(direction != "neutral" and effective_premium >= override_threshold)

        metrics.update(
            {
                "window_size": len(window),
                "covered_bars": len(covered),
                "window_net_premium": round(float(window_net), 2),
                "cumulative_net_premium": round(float(cumulative_net), 2),
                "override_threshold": round(float(override_threshold), 2),
                "effective_abs_premium": round(float(effective_premium), 2),
                "direction": direction,
                "applied": applied,
            }
        )
        if not applied:
            metrics["reason"] = (
                "tcbbo_flow_not_strong_enough"
                if direction != "neutral"
                else "tcbbo_neutral_flow"
            )
        return metrics

    _TCBBO_CONTRARIAN_STRATEGIES = frozenset({
        "mean_reversion", "absorption_reversal", "rotation",
    })

    def passes_tcbbo_confirmation(
        self, session: TradingSession, signal: Signal
    ) -> tuple[bool, Dict[str, Any]]:
        """Optional TCBBO options flow confirmation gate.

        Checks that options net premium flow aligns with signal direction
        over a lookback window. Whale sweeps aligned with direction provide
        a confidence boost.

        Contrarian strategies (MR, absorption, rotation) bypass the alignment
        check because they intentionally trade against panicked retail flow.
        """
        metrics: Dict[str, Any] = {
            "enabled": bool(session.tcbbo_gate_enabled),
            "lookback_bars": max(1, int(session.tcbbo_lookback_bars)),
            "min_net_premium": float(session.tcbbo_min_net_premium),
            "sweep_boost": float(session.tcbbo_sweep_boost),
            "passed": True,
        }

        if not session.tcbbo_gate_enabled:
            return True, metrics

        lookback = metrics["lookback_bars"]
        window = session.bars[-lookback:] if session.bars else []
        if not window:
            metrics.update({"passed": False, "reason": "tcbbo_window_empty"})
            return False, metrics

        # Check TCBBO data coverage
        has_tcbbo = any(
            bool(getattr(b, "tcbbo_has_data", None))
            for b in window
        )
        if not has_tcbbo:
            # No TCBBO data available — pass through (don't block)
            metrics.update({"passed": True, "reason": "tcbbo_data_missing_passthrough"})
            return True, metrics

        direction = 1.0 if signal.signal_type == SignalType.BUY else -1.0

        # Aggregate net premium over lookback
        net_premiums = [
            self._to_float(getattr(b, "tcbbo_net_premium", None), 0.0)
            for b in window
            if getattr(b, "tcbbo_has_data", False)
        ]
        window_net_premium = sum(net_premiums)
        directional_premium = window_net_premium * direction

        # Cumulative premium from latest bar (overall trend)
        latest_cum = self._to_float(
            getattr(window[-1], "tcbbo_cumulative_net_premium", None), 0.0
        )
        directional_cumulative = latest_cum * direction

        # Sweep analysis
        sweep_counts = [
            int(self._to_float(getattr(b, "tcbbo_sweep_count", None), 0))
            for b in window
            if getattr(b, "tcbbo_has_data", False)
        ]
        total_sweeps = sum(sweep_counts)
        sweep_premium = sum(
            self._to_float(getattr(b, "tcbbo_sweep_premium", None), 0.0)
            for b in window
            if getattr(b, "tcbbo_has_data", False)
        )

        # Premium threshold check
        min_prem = float(session.tcbbo_min_net_premium)
        passes_premium = directional_premium >= min_prem

        # Contrarian strategies (MR, rotation, absorption) trade against retail
        # panic flow.  Requiring aligned premium is counterproductive — bypass.
        _strategy_key = self._canonical_signal_strategy_key(signal)
        is_contrarian = _strategy_key in self._TCBBO_CONTRARIAN_STRATEGIES

        # Sweep alignment bonus (informational — doesn't block)
        sweep_aligned = total_sweeps > 0 and directional_premium > 0
        confidence_boost = float(session.tcbbo_sweep_boost) if sweep_aligned else 0.0

        passed = passes_premium or is_contrarian

        metrics.update({
            "window_size": len(window),
            "tcbbo_covered_bars": sum(
                1 for b in window if getattr(b, "tcbbo_has_data", False)
            ),
            "window_net_premium": round(window_net_premium, 2),
            "directional_premium": round(directional_premium, 2),
            "cumulative_net_premium": round(latest_cum, 2),
            "directional_cumulative": round(directional_cumulative, 2),
            "total_sweeps": total_sweeps,
            "sweep_premium": round(sweep_premium, 2),
            "passes_premium": passes_premium,
            "is_contrarian_bypass": is_contrarian,
            "sweep_aligned": sweep_aligned,
            "confidence_boost": confidence_boost,
            "passed": passed,
        })

        if not passed:
            metrics["reason"] = "tcbbo_confirmation_failed"
        elif is_contrarian and not passes_premium:
            metrics["reason"] = "tcbbo_contrarian_bypass"
        return passed, metrics
