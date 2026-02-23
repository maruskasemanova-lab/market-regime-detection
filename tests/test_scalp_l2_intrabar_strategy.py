"""Unit tests for the ScalpL2Intrabar strategy."""

from datetime import datetime, timezone
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.strategies.base_strategy import Regime, SignalType
from src.strategies.scalp_l2_intrabar import ScalpL2IntrabarStrategy


def _base_ohlcv() -> dict:
    return {
        "open": [100.0, 100.2, 100.4],
        "high": [100.3, 100.5, 100.7],
        "low": [99.8, 100.0, 100.2],
        "close": [100.2, 100.4, 100.6],
        "volume": [20_000.0, 22_000.0, 21_500.0],
    }


def _flow(**overrides) -> dict:
    payload = {
        "has_l2_coverage": True,
        "flow_score": 68.0,
        "signed_aggression": 0.09,
        "directional_consistency": 0.62,
        "imbalance_avg": 0.07,
        "book_pressure_avg": 0.06,
        "participation_ratio": 0.24,
        "flow_score_trend_3bar": 1.8,
        "sweep_intensity": 0.2,
        "price_change_pct": 0.19,
    }
    payload.update(overrides)
    return payload


def _intrabar(**overrides) -> dict:
    payload = {
        "has_intrabar_coverage": True,
        "coverage_points": 12,
        "mid_move_pct": 0.08,
        "push_ratio": 0.36,
        "directional_consistency": 0.74,
        "micro_volatility_bps": 6.5,
        "spread_bps_avg": 4.8,
        "window_eval_seconds": 5,
        "window_long_move_pct": 0.022,
        "window_long_push_ratio": 0.20,
        "window_long_directional_consistency": 0.44,
        "window_short_move_pct": -0.018,
        "window_short_push_ratio": -0.18,
        "window_short_directional_consistency": 0.41,
    }
    payload.update(overrides)
    return payload


def test_generates_buy_signal_with_flow_and_intrabar_confirmation() -> None:
    strategy = ScalpL2IntrabarStrategy(min_confidence=40.0)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(),
            "intrabar_1s": _intrabar(),
            "atr": [0.32],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.stop_loss is not None and signal.stop_loss < signal.price
    assert signal.take_profit is not None and signal.take_profit > signal.price
    assert signal.metadata["intrabar_1s"]["has_intrabar_coverage"] is True


def test_generates_short_signal_when_flow_and_intrabar_are_bearish() -> None:
    strategy = ScalpL2IntrabarStrategy(min_confidence=40.0)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(
                signed_aggression=-0.10,
                imbalance_avg=-0.08,
                book_pressure_avg=-0.07,
                price_change_pct=-0.22,
            ),
            "intrabar_1s": _intrabar(mid_move_pct=-0.09, push_ratio=-0.40),
            "atr": [0.33],
        },
        regime=Regime.MIXED,
        timestamp=datetime(2026, 2, 3, 14, 33, tzinfo=timezone.utc),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.SELL
    assert signal.stop_loss is not None and signal.stop_loss > signal.price
    assert signal.take_profit is not None and signal.take_profit < signal.price


def test_window_5s_confirmation_can_trigger_long_entry() -> None:
    strategy = ScalpL2IntrabarStrategy(min_confidence=35.0)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(),
            "intrabar_1s": _intrabar(
                mid_move_pct=0.01,
                push_ratio=0.03,
                directional_consistency=0.09,
                window_long_move_pct=0.028,
                window_long_push_ratio=0.22,
                window_long_directional_consistency=0.36,
            ),
            "atr": [0.32],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 34, tzinfo=timezone.utc),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.metadata["intrabar_1s"]["window_long_ok"] is True


def test_rejects_signal_without_l2_coverage() -> None:
    strategy = ScalpL2IntrabarStrategy()
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(has_l2_coverage=False),
            "intrabar_1s": _intrabar(),
            "atr": [0.3],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 35, tzinfo=timezone.utc),
    )
    assert signal is None


def test_can_trade_without_intrabar_when_not_required_and_flow_is_very_strong() -> None:
    strategy = ScalpL2IntrabarStrategy(
        require_intrabar_confirmation=False,
        min_flow_score=58.0,
        no_intrabar_flow_buffer=8.0,
        min_confidence=40.0,
    )
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(flow_score=70.0),
            "intrabar_1s": {"has_intrabar_coverage": False},
            "atr": [0.28],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 36, tzinfo=timezone.utc),
    )
    assert signal is not None


def test_requires_intrabar_when_flag_enabled() -> None:
    strategy = ScalpL2IntrabarStrategy(require_intrabar_confirmation=True)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(flow_score=75.0),
            "intrabar_1s": {"has_intrabar_coverage": False},
            "atr": [0.28],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 37, tzinfo=timezone.utc),
    )
    assert signal is None


def test_rejects_signal_when_participation_ratio_is_too_low() -> None:
    strategy = ScalpL2IntrabarStrategy(min_participation_ratio=0.15)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(participation_ratio=0.03),
            "intrabar_1s": _intrabar(),
            "atr": [0.30],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 40, tzinfo=timezone.utc),
    )
    assert signal is None


def test_rejects_signal_when_intrabar_micro_volatility_is_too_high() -> None:
    strategy = ScalpL2IntrabarStrategy(max_intrabar_micro_volatility_bps=8.0)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(),
            "intrabar_1s": _intrabar(micro_volatility_bps=16.0),
            "atr": [0.30],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 41, tzinfo=timezone.utc),
    )
    assert signal is None


def test_rejects_signal_when_intrabar_coverage_is_too_sparse() -> None:
    strategy = ScalpL2IntrabarStrategy(min_intrabar_coverage_points=8)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(),
            "intrabar_1s": _intrabar(coverage_points=3),
            "atr": [0.30],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 42, tzinfo=timezone.utc),
    )
    assert signal is None


def test_spread_penalty_can_block_borderline_flow_setup() -> None:
    strategy = ScalpL2IntrabarStrategy(
        min_flow_score=60.0,
        spread_penalty_floor_bps=4.0,
        spread_flow_score_penalty_per_bps=1.0,
        min_confidence=35.0,
    )
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(flow_score=63.0),
            "intrabar_1s": _intrabar(spread_bps_avg=8.0),
            "atr": [0.30],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 43, tzinfo=timezone.utc),
    )
    assert signal is None


def test_cost_guard_rejects_low_reward_relative_to_estimated_cost() -> None:
    strategy = ScalpL2IntrabarStrategy(
        min_confidence=35.0,
        atr_stop_multiplier=0.20,
        rr_ratio=1.0,
        min_round_trip_cost_bps=12.0,
        spread_cost_multiplier=1.5,
        min_reward_to_cost_ratio=2.0,
    )
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_base_ohlcv(),
        indicators={
            "order_flow": _flow(flow_score=80.0),
            "intrabar_1s": _intrabar(spread_bps_avg=10.0),
            "atr": [0.10],
        },
        regime=Regime.TRENDING,
        timestamp=datetime(2026, 2, 3, 14, 44, tzinfo=timezone.utc),
    )
    assert signal is None
