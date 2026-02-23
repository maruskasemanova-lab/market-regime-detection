"""Macro regime detection helper extracted from day_trading_regime_impl."""

from __future__ import annotations

from ..day_trading_models import TradingSession
from ..strategies.base_strategy import Regime


def regime_detect_regime(self, session: TradingSession) -> Regime:
    """Detect macro regime and update session micro-regime."""

    orch = session.orchestrator
    adaptive_enabled = bool(
        getattr(getattr(orch, "config", None), "use_adaptive_regime", False)
    )
    if orch and adaptive_enabled and orch.current_feature_vector:
        regime_state = orch.detect_regime()
        session.micro_regime = regime_state.micro_regime
        return self._map_adaptive_regime(regime_state.primary)

    all_bars = session.bars
    if len(all_bars) < 20:
        session.micro_regime = "MIXED"
        return Regime.MIXED

    closes = [b.close for b in all_bars[-min(len(all_bars), 30):]]
    if len(closes) < 10:
        session.micro_regime = "MIXED"
        return Regime.MIXED

    net_move = abs(closes[-1] - closes[0])
    total_move = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    if total_move == 0:
        session.micro_regime = "MIXED"
        return Regime.MIXED
    trend_efficiency = net_move / total_move

    adx = self._calc_adx(all_bars)
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1] != 0
    ]
    avg_return = (sum(returns) / len(returns)) if returns else 0.0
    variance = (
        (sum((r - avg_return) ** 2 for r in returns) / len(returns))
        if returns
        else 0.0
    )
    volatility = variance**0.5
    price_change_pct = self._safe_div((closes[-1] - closes[0]) * 100.0, closes[0], 0.0)

    gap_factor = 0.0
    if session.pre_market_bars and session.bars:
        pre_close = session.pre_market_bars[-1].close
        open_price = session.bars[0].open
        if pre_close > 0:
            gap_pct = abs(open_price - pre_close) / pre_close * 100.0
            if gap_pct > 1.0:
                gap_factor = 0.1

    flow = self._calculate_order_flow_metrics(all_bars, lookback=min(24, len(all_bars)))
    micro_regime = self._classify_micro_regime(
        trend_efficiency=trend_efficiency + gap_factor,
        adx=adx,
        volatility=volatility,
        price_change_pct=price_change_pct,
        flow=flow,
    )
    session.micro_regime = micro_regime
    return self._map_micro_to_regime(micro_regime)
