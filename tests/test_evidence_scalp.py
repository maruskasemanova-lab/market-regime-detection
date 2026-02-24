"""Tests for EvidenceScalpStrategy v2 (Micro-Scalper)."""
from datetime import datetime

import pytest

from src.strategies.base_strategy import Regime, SignalType
from src.strategies.evidence_scalp import EvidenceScalpStrategy


def _make_ohlcv():
    return {
        "open": [100.0] * 5,
        "high": [100.5] * 5,
        "low": [99.5] * 5,
        "close": [100.0] * 5,
        "volume": [1000] * 5,
    }


def _base_indicators(
    move_pct=0.0,
    push_ratio=0.0,
    aggr=0.0,
    book=0.0,
    delta_div=0.0,
    options_net=0.0,
    has_l2=True,
    has_options=True,
    spread_bps=1.0,
):
    """Build a minimal indicators dict."""
    return {
        "intrabar_1s": {
            "has_intrabar_coverage": True,
            "mid_move_pct": move_pct,
            "push_ratio": push_ratio,
            "spread_bps_avg": spread_bps,
        },
        "order_flow": {
            "has_l2_coverage": has_l2,
            "signed_aggression": aggr,
            "book_pressure_avg": book,
            "delta_price_divergence": delta_div,
        },
        "tcbbo": {
            "is_valid": has_options,
            "net_premium": options_net,
        },
    }


TS = datetime(2026, 2, 23, 10, 30, 0)


class TestEvidenceMicroScalpBasics:
    def test_no_signal_when_insufficient_evidence(self):
        strat = EvidenceScalpStrategy()
        ohlcv = _make_ohlcv()
        indicators = _base_indicators(move_pct=0.01, push_ratio=0.5)
        sig = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, TS)
        assert sig is None

    def test_bullish_momentum_signal(self):
        strat = EvidenceScalpStrategy(min_5s_move_pct=0.02, min_5s_push_ratio=0.6)
        ohlcv = _make_ohlcv()
        indicators = _base_indicators(
            move_pct=0.03,        # strong move up
            push_ratio=0.8,       # strong push up
            aggr=0.05,            # strong L2 buying
        )
        sig = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, TS)
        assert sig is not None
        assert sig.signal_type == SignalType.BUY
        assert "momentum_long" in sig.metadata["scalp_trigger"]

    def test_bearish_momentum_signal_from_options(self):
        strat = EvidenceScalpStrategy(min_5s_move_pct=0.02, min_tcbbo_premium=25000.0)
        ohlcv = _make_ohlcv()
        indicators = _base_indicators(
            move_pct=-0.03,       # strong move down
            push_ratio=-0.8,      # strong push down
            options_net=-30000.0, # heavy puts
        )
        sig = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, TS)
        assert sig is not None
        assert sig.signal_type == SignalType.SELL
        assert "momentum_short" in sig.metadata["scalp_trigger"]

    def test_bearish_fade_absorption_signal(self):
        strat = EvidenceScalpStrategy()
        ohlcv = _make_ohlcv()
        indicators = _base_indicators(
            move_pct=0.03,        # price pushed UP
            push_ratio=0.8,       # push up
            aggr=-0.06,           # L2 aggression heavily negative (selling into the rally)
            delta_div=-0.10,      # heavy divergence
        )
        sig = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, TS)
        assert sig is not None
        assert sig.signal_type == SignalType.SELL
        assert "fade_absorption_short" in sig.metadata["scalp_trigger"]

    def test_bullish_fade_absorption_signal(self):
        strat = EvidenceScalpStrategy()
        ohlcv = _make_ohlcv()
        indicators = _base_indicators(
            move_pct=-0.03,       # price pushed DOWN
            push_ratio=-0.8,      # push down
            aggr=0.06,            # L2 aggression heavily positive (buying the dip)
            delta_div=0.10,       # heavy divergence
        )
        sig = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, TS)
        assert sig is not None
        assert sig.signal_type == SignalType.BUY
        assert "fade_absorption_long" in sig.metadata["scalp_trigger"]

    def test_regime_filtering(self):
        strat = EvidenceScalpStrategy()
        strat.allowed_regimes = [Regime.TRENDING]
        ohlcv = _make_ohlcv()
        indicators = _base_indicators(move_pct=0.03, push_ratio=0.8, aggr=0.06)
        sig = strat.generate_signal(100.0, ohlcv, indicators, Regime.CHOPPY, TS)
        assert sig is None


class TestEvidenceMicroScalpCooldown:
    def test_cooldown_blocks_rapid_signals(self):
        strat = EvidenceScalpStrategy(min_signal_interval_seconds=5)
        ohlcv = _make_ohlcv()
        indicators = _base_indicators(move_pct=0.03, push_ratio=0.8, aggr=0.06)

        ts1 = datetime(2026, 2, 23, 10, 30, 0)
        sig1 = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, ts1)
        assert sig1 is not None

        ts2 = datetime(2026, 2, 23, 10, 30, 2)  # 2s later
        sig2 = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, ts2)
        assert sig2 is None  # blocked by cooldown

        ts3 = datetime(2026, 2, 23, 10, 30, 6)  # 6s later
        sig3 = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, ts3)
        assert sig3 is not None  # passed cooldown


class TestEvidenceMicroScalpCostGuard:
    def test_cost_guard_blocks_unprofitable(self):
        strat = EvidenceScalpStrategy(
            base_stop_loss_pct=0.05,  # tiny stop -> tiny reward
            target_rr_ratio=1.0,
            min_round_trip_cost_bps=5.0,  # reward = 5 bps, cost = 5 bps
        )
        ohlcv = _make_ohlcv()
        indicators = _base_indicators(
            move_pct=0.03, push_ratio=0.8, aggr=0.06, spread_bps=10.0
        )
        # est_cost_bps = max(5.0, 10.0 * 1.2) = 12.0 bps.
        # target reward = 0.05% * 1.0 = 5 bps. 5 < 12.0
        sig = strat.generate_signal(100.0, ohlcv, indicators, Regime.TRENDING, TS)
        assert sig is None  # blocked by cost guard


class TestEvidenceMicroScalpSerialization:
    def test_to_dict_includes_all_params(self):
        strat = EvidenceScalpStrategy()
        d = strat.to_dict()
        assert d["min_5s_move_pct"] == 0.02
        assert d["min_l2_aggression"] == 0.04
        assert d["base_stop_loss_pct"] == 0.15
        assert d["min_signal_interval_seconds"] == 5
