from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .day_trading_models import BarData, TradingSession, Regime
from .strategies.base_strategy import BaseStrategy, Signal, Position
from .strategy_formula_engine import evaluate_strategy_formula, StrategyFormulaEvaluationError

class StrategyEvaluatorEngine:
    def __init__(self, evidence_service):
        self.evidence_service = evidence_service

    def build_strategy_formula_context(
        self,
        session: TradingSession,
        bar: BarData,
        indicators: Dict[str, Any],
        flow: Dict[str, Any],
        current_bar_index: int,
        signal: Optional[Signal] = None,
        position: Optional[Position] = None,
    ) -> Dict[str, Any]:
        intrabar = {}
        if hasattr(bar, "intrabar") and getattr(bar, "intrabar"):
            intrabar = getattr(bar, "intrabar")
        if getattr(session, "simulate_intrabar_metadata", False):
            # Mock intrabar data for non-L2 sessions
            intrabar = {}

        vwap_series = indicators.get("vwap") if isinstance(indicators, dict) else None
        if isinstance(vwap_series, list):
            vwap_ind = self.evidence_service.safe_float(vwap_series[-1] if vwap_series else None, 0.0) or 0.0
        else:
            vwap_ind = self.evidence_service.safe_float(vwap_series, 0.0) or 0.0
        vwap_value = self.evidence_service.safe_float(getattr(bar, "vwap", None), None)
        if vwap_value is None or vwap_value <= 0:
            vwap_value = vwap_ind

        signal_side = ""
        confidence = 0.0
        if signal is not None:
            signal_type = getattr(signal, "signal_type", None)
            signal_side = str(getattr(signal_type, "value", signal_type) or "").strip().lower()
            confidence = self.evidence_service.safe_float(getattr(signal, "confidence", None), 0.0) or 0.0

        position_side = ""
        bars_held = 0
        position_pnl_pct = 0.0
        position_pnl_dollars = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        trailing_stop_price = 0.0
        if position is not None:
            position_side = str(getattr(position, "side", "") or "").strip().lower()
            bars_held = max(0, int(self.evidence_service.bars_held(position, current_bar_index)))
            pos_entry = self.evidence_service.safe_float(getattr(position, "entry_price", None), 0.0) or 0.0
            position_size = max(0.0, self.evidence_service.safe_float(getattr(position, "size", 0.0), 0.0) or 0.0)
            if pos_entry > 0:
                if position_side == "short":
                    position_pnl_pct = ((pos_entry - bar.close) / pos_entry) * 100.0
                    position_pnl_dollars = (pos_entry - bar.close) * position_size
                else:
                    position_pnl_pct = ((bar.close - pos_entry) / pos_entry) * 100.0
                    position_pnl_dollars = (bar.close - pos_entry) * position_size
            entry_price = pos_entry
            stop_loss = self.evidence_service.safe_float(getattr(position, "stop_loss", None), 0.0) or 0.0
            take_profit = self.evidence_service.safe_float(getattr(position, "take_profit", None), 0.0) or 0.0
            trailing_stop_price = self.evidence_service.safe_float(
                getattr(position, "trailing_stop_price", None),
                0.0,
            ) or 0.0

        context = {
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "price": float(bar.close),
            "volume": float(bar.volume),
            "vwap": float(vwap_value or 0.0),
            "regime": str(
                getattr(getattr(session, "detected_regime", None), "value", "MIXED") or "MIXED"
            ),
            "micro_regime": str(getattr(session, "micro_regime", "") or ""),
            "bar_index": int(current_bar_index),
            "trade_count": int(len(getattr(session, "trades", []) or [])),
            "open_positions": int(1 if getattr(session, "active_position", None) else 0),
            "confidence": float(confidence),
            "signal_side": signal_side,
            "position_side": position_side,
            "bars_held": int(bars_held),
            "position_pnl_pct": float(position_pnl_pct),
            "position_pnl_dollars": float(position_pnl_dollars),
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "trailing_stop_price": float(trailing_stop_price),
            "atr": float(self.evidence_service.latest_indicator_value(indicators, "atr", session.bars)),
            "adx": float(self.evidence_service.latest_indicator_value(indicators, "adx", session.bars)),
            "rsi": float(self.evidence_service.latest_indicator_value(indicators, "rsi", session.bars)),
            "ema_fast": float(self.evidence_service.latest_indicator_value(indicators, "ema_fast", session.bars)),
            "ema_slow": float(self.evidence_service.latest_indicator_value(indicators, "ema_slow", session.bars)),
            "sma": float(self.evidence_service.latest_indicator_value(indicators, "sma", session.bars)),
            "flow_score": float(flow.get("flow_score", 0.0) or 0.0),
            "signed_aggression": float(flow.get("signed_aggression", 0.0) or 0.0),
            "directional_consistency": float(flow.get("directional_consistency", 0.0) or 0.0),
            "imbalance": float(flow.get("imbalance_avg", 0.0) or 0.0),
            "absorption_rate": float(flow.get("absorption_rate", 0.0) or 0.0),
            "sweep_intensity": float(flow.get("sweep_intensity", 0.0) or 0.0),
            "book_pressure": float(flow.get("book_pressure_avg", 0.0) or 0.0),
            "book_pressure_trend": float(flow.get("book_pressure_trend", 0.0) or 0.0),
            "participation_ratio": float(flow.get("participation_ratio", 0.0) or 0.0),
            "delta_zscore": float(flow.get("delta_zscore", 0.0) or 0.0),
            "large_trader_activity": float(flow.get("large_trader_activity", 0.0) or 0.0),
            "vwap_execution_flow": float(flow.get("vwap_execution_flow", 0.0) or 0.0),
            "has_l2_coverage": bool(flow.get("has_l2_coverage", False)),
            "intrabar_move_pct": float(intrabar.get("mid_move_pct", 0.0) or 0.0),
            "intrabar_push_ratio": float(intrabar.get("push_ratio", 0.0) or 0.0),
            "intrabar_directional_consistency": float(
                intrabar.get("directional_consistency", 0.0) or 0.0
            ),
            "intrabar_spread_bps": float(intrabar.get("spread_bps_avg", 0.0) or 0.0),
            "intrabar_micro_volatility_bps": float(
                intrabar.get("micro_volatility_bps", 0.0) or 0.0
            ),
        }
        return context

    @staticmethod
    def strategy_formula_fields(formula_type: str) -> Tuple[str, str]:
        if str(formula_type).strip().lower() == "exit":
            return "custom_exit_formula_enabled", "custom_exit_formula"
        return "custom_entry_formula_enabled", "custom_entry_formula"

    def evaluate_strategy_custom_formula(
        self,
        *,
        strategy: BaseStrategy,
        formula_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        enabled_field, formula_field = self.strategy_formula_fields(formula_type)
        enabled = bool(getattr(strategy, enabled_field, False))
        formula_text = str(getattr(strategy, formula_field, "") or "").strip()
        active = bool(enabled and formula_text)
        base = {
            "type": str(formula_type).strip().lower() or "entry",
            "enabled": active,
            "formula": formula_text,
            "passed": True,
            "error": None,
        }
        if not active:
            return base
        try:
            passed = bool(evaluate_strategy_formula(formula_text, context))
            base["passed"] = passed
            return base
        except StrategyFormulaEvaluationError as exc:
            base["passed"] = False
            base["error"] = str(exc)
            return base

    def generate_signal_with_overrides(
        self,
        strategy: BaseStrategy,
        overrides: Dict[str, Any],
        current_price: float,
        ohlcv: Dict[str, List[float]],
        indicators: Dict[str, Any],
        regime: Regime,
        timestamp: datetime
    ) -> Optional[Signal]:
        if not overrides:
            return strategy.generate_signal(
                current_price=current_price,
                ohlcv=ohlcv,
                indicators=indicators,
                regime=regime,
                timestamp=timestamp
            )

        original_values = {}
        for k, v in overrides.items():
            if hasattr(strategy, k):
                original_values[k] = getattr(strategy, k)
                setattr(strategy, k, v)

        try:
            return strategy.generate_signal(
                current_price=current_price,
                ohlcv=ohlcv,
                indicators=indicators,
                regime=regime,
                timestamp=timestamp
            )
        finally:
            for k, v in original_values.items():
                setattr(strategy, k, v)
