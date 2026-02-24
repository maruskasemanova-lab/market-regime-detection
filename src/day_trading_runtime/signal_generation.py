"""Signal generation and indicator calculation helpers extracted from runtime impl."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..day_trading_models import BarData, TradingSession
from ..strategies.base_strategy import Regime, Signal, SignalType
from ..day_trading_runtime_intrabar import (
    calculate_intrabar_1s_snapshot as _calculate_intrabar_1s_snapshot_impl,
)


def runtime_generate_signal(
    self,
    session: TradingSession,
    bar: BarData,
    timestamp: datetime,
) -> Optional[Signal]:
    """
    Generate trading signal using ALL active strategies.
    Returns the signal with highest confidence.
    """

    active_strategies = session.active_strategies
    if not active_strategies:
        if session.selected_strategy:
            active_strategies = [session.selected_strategy]
        else:
            return None

    bars = session.bars[-100:] if len(session.bars) >= 100 else session.bars
    ohlcv = {
        "open": [b.open for b in bars],
        "high": [b.high for b in bars],
        "low": [b.low for b in bars],
        "close": [b.close for b in bars],
        "volume": [b.volume for b in bars],
    }
    indicators = self._calculate_indicators(bars, session=session)
    regime = session.detected_regime or Regime.MIXED

    candidate_signals = []
    ticker_cfg = self.ticker_params.get(session.ticker.upper(), {})
    is_long_only = bool(ticker_cfg.get("long_only", False))
    flow_metrics = indicators.get("order_flow") or {}
    flow_available = bool(flow_metrics.get("has_l2_coverage", False))
    flow_strategy_keys = {
        "absorption_reversal",
        "momentum_flow",
        "exhaustion_fade",
        "scalp_l2_intrabar",
        "evidence_scalp",
    }

    for strategy_name in active_strategies:
        if strategy_name not in self.strategies:
            continue

        strategy = self.strategies[strategy_name]

        signal = self.strategy_evaluator.generate_signal_with_overrides(
            strategy=strategy,
            overrides=self._get_ticker_strategy_overrides(session.ticker, strategy_name),
            current_price=bar.close,
            ohlcv=ohlcv,
            indicators=indicators,
            regime=regime,
            timestamp=timestamp,
        )

        if signal:
            if is_long_only and signal.signal_type == SignalType.SELL:
                continue
            strategy_key = self._canonical_strategy_key(signal.strategy_name)
            edge_adjustment = self._strategy_edge_adjustment(session, strategy_key)
            original_confidence = float(signal.confidence)
            signal.confidence = max(1.0, min(100.0, original_confidence + edge_adjustment))
            signal.metadata.setdefault(
                "confidence_adjustment",
                {
                    "base_confidence": original_confidence,
                    "edge_adjustment": edge_adjustment,
                    "adjusted_confidence": signal.confidence,
                },
            )
            candidate_signals.append(signal)

    if not candidate_signals:
        return None

    rank = {name: idx for idx, name in enumerate(active_strategies)}

    strategy_trade_history: Dict[str, List[float]] = {}
    for trade in session.trades:
        trade_key = self._canonical_strategy_key(trade.strategy)
        strategy_trade_history.setdefault(trade_key, []).append(trade.gross_pnl_pct)

    scored = []
    for sig in candidate_signals:
        key = self._canonical_strategy_key(sig.strategy_name)
        preference_bonus = max(0.0, 6.0 - (rank.get(key, 99) * 3.0))
        recent = strategy_trade_history.get(key, [])[-3:]
        perf_bonus = 0.0
        if recent:
            perf_bonus = max(-8.0, min(8.0, (sum(recent) / len(recent)) * 2.0))

        flow_bonus = 0.0
        if flow_available:
            flow_bonus = 3.0 if key in flow_strategy_keys else 0.0

        score = float(sig.confidence) + preference_bonus + perf_bonus + flow_bonus
        scored.append((score, sig))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_signal = scored[0][1]

    best_score = scored[0][0]
    top3 = []
    for s_score, s_sig in scored[:3]:
        top3.append(
            {
                "name": self._canonical_strategy_key(s_sig.strategy_name),
                "score": round(s_score, 1),
                "confidence": round(float(s_sig.confidence), 1),
                "margin": round(best_score - s_score, 1),
            }
        )
    best_signal.metadata["candidate_diagnostics"] = {
        "strategy_name": self._canonical_strategy_key(best_signal.strategy_name),
        "candidate_strategies_count": len(candidate_signals),
        "active_strategies_count": len(active_strategies),
        "active_strategies": list(active_strategies),
        "top3": top3,
    }

    return best_signal


def runtime_calculate_indicators(
    self,
    bars: List[BarData],
    session: Optional[TradingSession] = None,
) -> Dict[str, Any]:
    """Calculate indicators from bars."""

    latest_bar = bars[-1] if bars else None
    intraday_levels = (
        self._intraday_levels_indicator_payload(session)
        if isinstance(session, TradingSession)
        else {}
    )
    if len(bars) < 5:
        return {
            "order_flow": self._calculate_order_flow_metrics(bars, lookback=len(bars) or 1),
            "intrabar_1s": _calculate_intrabar_1s_snapshot_impl(latest_bar),
            "intraday_levels": intraday_levels,
        }

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]

    indicators = {}

    if len(closes) >= 20:
        indicators["sma"] = [
            sum(closes[max(0, i - 19) : i + 1]) / min(i + 1, 20)
            for i in range(len(closes))
        ]

    if len(closes) >= 10:
        ema = []
        multiplier = 2 / (10 + 1)
        for i, c in enumerate(closes):
            if i == 0:
                ema.append(c)
            else:
                ema.append((c - ema[-1]) * multiplier + ema[-1])
        indicators["ema"] = ema
        indicators["ema_fast"] = ema

    if len(closes) >= 20:
        ema_slow = []
        multiplier = 2 / (20 + 1)
        for i, c in enumerate(closes):
            if i == 0:
                ema_slow.append(c)
            else:
                ema_slow.append((c - ema_slow[-1]) * multiplier + ema_slow[-1])
        indicators["ema_slow"] = ema_slow

    if len(closes) >= 15:
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0))
            losses.append(abs(min(change, 0)))

        rsi = []
        for i in range(13, len(gains)):
            avg_gain = sum(gains[max(0, i - 13) : i + 1]) / 14
            avg_loss = sum(losses[max(0, i - 13) : i + 1]) / 14
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))

        indicators["rsi"] = [50] * (len(closes) - len(rsi)) + rsi

    if len(closes) >= 2:
        tr = []
        for i in range(1, len(bars)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr.append(max(hl, hc, lc))

        if tr:
            period = min(14, len(tr))
            atr = []
            for i in range(len(tr)):
                window = tr[max(0, i - period + 1) : i + 1]
                atr.append(sum(window) / len(window))
            indicators["atr"] = [atr[0]] + atr

    vwaps = [b.vwap for b in bars if b.vwap is not None]
    if vwaps:
        indicators["vwap"] = vwaps
    else:
        cum_vol = 0
        cum_pv = 0
        vwap = []
        for b in bars:
            typical_price = (b.high + b.low + b.close) / 3
            cum_vol += b.volume
            cum_pv += typical_price * b.volume
            vwap.append(cum_pv / cum_vol if cum_vol > 0 else typical_price)
        indicators["vwap"] = vwap

    adx_series = self._calc_adx_series(bars, 14)
    indicators["adx"] = adx_series if adx_series else [None] * len(closes)

    indicators["order_flow"] = self._calculate_order_flow_metrics(
        bars,
        lookback=min(24, len(bars)),
    )
    indicators["intrabar_1s"] = _calculate_intrabar_1s_snapshot_impl(latest_bar)
    indicators["intraday_levels"] = intraday_levels

    return indicators
