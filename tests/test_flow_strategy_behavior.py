"""Behavior tests for flow-first strategy modules and thresholds."""

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.strategy_factory import build_strategy_registry
from src.strategies.absorption_reversal import AbsorptionReversalStrategy
from src.strategies.base_strategy import Regime, Signal, SignalType
from src.strategies.exhaustion_fade import ExhaustionFadeStrategy
from src.strategies.momentum_flow import MomentumFlowStrategy


def _ts() -> datetime:
    return datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc)


def _ohlcv() -> dict:
    return {
        "open": [100.0, 100.2, 100.4],
        "high": [100.3, 100.5, 100.7],
        "low": [99.8, 100.0, 100.2],
        "close": [100.2, 100.4, 100.6],
        "volume": [20_000.0, 21_500.0, 22_000.0],
    }


def _indicators(order_flow: dict, atr: float = 0.35) -> dict:
    return {
        "order_flow": order_flow,
        "atr": [atr],
    }


def _open_long_position(strategy, *, price: float = 100.0) -> None:
    strategy.open_position(
        Signal(
            strategy_name=strategy.name,
            signal_type=SignalType.BUY,
            price=price,
            timestamp=_ts(),
            confidence=75.0,
            stop_loss=price - 0.6,
            take_profit=price + 1.2,
            trailing_stop=True,
            trailing_stop_pct=0.8,
        )
    )


def _absorption_flow(**overrides) -> dict:
    payload = {
        "has_l2_coverage": True,
        "absorption_rate": 0.80,
        "delta_price_divergence": 0.35,
        "signed_aggression": -0.25,
        "book_pressure_avg": 0.15,
        "price_change_pct": -0.25,
        "imbalance_avg": 0.0,
        "directional_consistency": 0.80,
    }
    payload.update(overrides)
    return payload


def _momentum_flow(**overrides) -> dict:
    payload = {
        "has_l2_coverage": True,
        "signed_aggression": 0.18,
        "directional_consistency": 0.82,
        "imbalance_avg": 0.10,
        "sweep_intensity": 0.30,
        "book_pressure_avg": 0.08,
        "book_pressure_trend": 0.02,
        "price_change_pct": 0.22,
        "delta_acceleration": 0.05,
    }
    payload.update(overrides)
    return payload


def _exhaustion_flow(**overrides) -> dict:
    payload = {
        "has_l2_coverage": True,
        "absorption_rate": 0.65,
        "delta_price_divergence": 0.32,
        "delta_zscore": -2.2,
        "sweep_intensity": 0.25,
        "signed_aggression": -0.12,
        "book_pressure_avg": 0.07,
        "price_change_pct": -0.12,
    }
    payload.update(overrides)
    return payload


def test_absorption_reversal_generates_buy_signal_when_thresholds_are_met() -> None:
    strategy = AbsorptionReversalStrategy(min_confidence=60.0)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_absorption_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.stop_loss is not None and signal.stop_loss < signal.price
    assert signal.take_profit is not None and signal.take_profit > signal.price
    assert signal.metadata["order_flow"]["absorption_rate"] == pytest.approx(0.80)


def test_absorption_reversal_generates_sell_signal_when_bearish_absorption_forms() -> None:
    strategy = AbsorptionReversalStrategy(min_confidence=60.0)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(
            _absorption_flow(
                delta_price_divergence=-0.35,
                signed_aggression=0.22,
                book_pressure_avg=-0.14,
                price_change_pct=0.26,
                imbalance_avg=0.0,
            )
        ),
        regime=Regime.MIXED,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.SELL
    assert signal.stop_loss is not None and signal.stop_loss > signal.price
    assert signal.take_profit is not None and signal.take_profit < signal.price


def test_absorption_reversal_enforces_min_absorption_rate_threshold() -> None:
    strategy = AbsorptionReversalStrategy(min_absorption_rate=0.70, min_confidence=20.0)
    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_absorption_flow(absorption_rate=0.69)),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is None


def test_absorption_reversal_skips_new_entries_when_position_is_open() -> None:
    strategy = AbsorptionReversalStrategy(min_confidence=50.0)
    _open_long_position(strategy)

    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_absorption_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is None


def test_momentum_flow_generates_buy_signal_for_confirmed_upflow() -> None:
    strategy = MomentumFlowStrategy(min_confidence=50.0)
    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_momentum_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.metadata["order_flow"]["signed_aggression"] == pytest.approx(0.18)


def test_momentum_flow_generates_sell_signal_for_confirmed_downflow() -> None:
    strategy = MomentumFlowStrategy(min_confidence=50.0)
    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(
            _momentum_flow(
                signed_aggression=-0.18,
                imbalance_avg=-0.10,
                book_pressure_avg=-0.08,
                book_pressure_trend=-0.02,
                price_change_pct=-0.22,
            )
        ),
        regime=Regime.CHOPPY,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.SELL


def test_momentum_flow_enforces_min_signed_aggression_threshold() -> None:
    strategy = MomentumFlowStrategy(min_signed_aggression=0.20, min_confidence=20.0)
    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_momentum_flow(signed_aggression=0.19)),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is None


def test_momentum_flow_limits_entries_to_two_open_positions() -> None:
    strategy = MomentumFlowStrategy(min_confidence=40.0)
    _open_long_position(strategy, price=100.0)
    _open_long_position(strategy, price=100.5)

    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_momentum_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is None


def test_momentum_flow_requires_l2_coverage() -> None:
    strategy = MomentumFlowStrategy()
    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_momentum_flow(has_l2_coverage=False)),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is None


def test_exhaustion_fade_generates_buy_signal_on_bullish_exhaustion() -> None:
    strategy = ExhaustionFadeStrategy(min_confidence=55.0)
    signal = strategy.generate_signal(
        current_price=99.8,
        ohlcv=_ohlcv(),
        indicators=_indicators(_exhaustion_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.stop_loss is not None and signal.stop_loss < signal.price
    assert signal.take_profit is not None and signal.take_profit > signal.price


def test_exhaustion_fade_generates_sell_signal_on_bearish_exhaustion() -> None:
    strategy = ExhaustionFadeStrategy(min_confidence=55.0)
    signal = strategy.generate_signal(
        current_price=99.8,
        ohlcv=_ohlcv(),
        indicators=_indicators(
            _exhaustion_flow(
                delta_price_divergence=-0.32,
                delta_zscore=2.2,
                signed_aggression=0.12,
                book_pressure_avg=-0.07,
                price_change_pct=0.12,
            )
        ),
        regime=Regime.MIXED,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.SELL


def test_exhaustion_fade_enforces_min_absorption_rate() -> None:
    strategy = ExhaustionFadeStrategy(min_absorption_rate=0.70, min_confidence=20.0)
    signal = strategy.generate_signal(
        current_price=99.8,
        ohlcv=_ohlcv(),
        indicators=_indicators(_exhaustion_flow(absorption_rate=0.69)),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is None


def test_exhaustion_fade_blocks_when_sweep_intensity_exceeds_max() -> None:
    strategy = ExhaustionFadeStrategy(max_sweep_intensity=0.30, min_confidence=20.0)
    signal = strategy.generate_signal(
        current_price=99.8,
        ohlcv=_ohlcv(),
        indicators=_indicators(_exhaustion_flow(sweep_intensity=0.31)),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is None


def test_exhaustion_fade_confidence_floor_blocks_borderline_setup() -> None:
    strategy = ExhaustionFadeStrategy(min_confidence=90.0)
    signal = strategy.generate_signal(
        current_price=99.8,
        ohlcv=_ohlcv(),
        indicators=_indicators(
            _exhaustion_flow(
                absorption_rate=0.55,
                delta_price_divergence=0.20,
                delta_zscore=-1.4,
                sweep_intensity=0.40,
                signed_aggression=-0.05,
                book_pressure_avg=0.03,
                price_change_pct=-0.01,
            )
        ),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is None


def test_strategy_factory_registers_flow_strategies_with_expected_defaults() -> None:
    registry = build_strategy_registry()

    assert "absorption_reversal" in registry
    assert "momentum_flow" in registry
    assert "exhaustion_fade" in registry

    assert isinstance(registry["absorption_reversal"], AbsorptionReversalStrategy)
    assert isinstance(registry["momentum_flow"], MomentumFlowStrategy)
    assert isinstance(registry["exhaustion_fade"], ExhaustionFadeStrategy)

    assert registry["absorption_reversal"].min_absorption_rate == pytest.approx(0.55)
    assert registry["momentum_flow"].min_signed_aggression == pytest.approx(0.08)
    assert registry["exhaustion_fade"].min_absorption_rate == pytest.approx(0.50)


def test_strategy_factory_returns_fresh_instances_per_call() -> None:
    first_registry = build_strategy_registry()
    second_registry = build_strategy_registry()

    assert first_registry["absorption_reversal"] is not second_registry["absorption_reversal"]
    assert first_registry["momentum_flow"] is not second_registry["momentum_flow"]
    assert first_registry["exhaustion_fade"] is not second_registry["exhaustion_fade"]


def test_momentum_flow_uses_global_trailing_when_mode_is_global() -> None:
    strategy = MomentumFlowStrategy(min_confidence=50.0, trailing_stop_pct=1.4)
    strategy.exit_mode = "global"
    strategy.trailing_stop_mode = "global"
    strategy.global_trailing_stop_pct = 0.42

    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_momentum_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.trailing_stop_pct == pytest.approx(0.42)


def test_momentum_flow_keeps_custom_trailing_when_mode_is_custom() -> None:
    strategy = MomentumFlowStrategy(min_confidence=50.0, trailing_stop_pct=1.15)
    strategy.exit_mode = "custom"
    strategy.trailing_stop_mode = "custom"
    strategy.global_trailing_stop_pct = 0.33

    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_momentum_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.trailing_stop_pct == pytest.approx(1.15)


def test_momentum_flow_uses_global_exit_rr_ratio_when_mode_is_global() -> None:
    strategy = MomentumFlowStrategy(min_confidence=50.0, rr_ratio=2.4)
    strategy.exit_mode = "global"
    strategy.global_rr_ratio = 1.3

    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_momentum_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    risk = abs(float(signal.price) - float(signal.stop_loss or signal.price))
    expected_take_profit = float(signal.price) + risk * 1.3
    assert signal.take_profit == pytest.approx(expected_take_profit, rel=1e-6)


def test_momentum_flow_keeps_custom_rr_ratio_when_exit_mode_is_custom() -> None:
    strategy = MomentumFlowStrategy(min_confidence=50.0, rr_ratio=2.05)
    strategy.exit_mode = "custom"
    strategy.global_rr_ratio = 1.2

    signal = strategy.generate_signal(
        current_price=102.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_momentum_flow()),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    risk = abs(float(signal.price) - float(signal.stop_loss or signal.price))
    expected_take_profit = float(signal.price) + risk * 2.05
    assert signal.take_profit == pytest.approx(expected_take_profit, rel=1e-6)


def test_absorption_reversal_uses_global_risk_multiplier_when_mode_is_global() -> None:
    strategy = AbsorptionReversalStrategy(min_confidence=60.0, atr_stop_multiplier=1.1)
    strategy.risk_mode = "global"
    strategy.global_atr_stop_multiplier = 0.5

    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_absorption_flow(), atr=0.4),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.stop_loss == pytest.approx(101.0 - (0.4 * 0.5), rel=1e-6)


def test_absorption_reversal_keeps_custom_risk_multiplier_when_mode_is_custom() -> None:
    strategy = AbsorptionReversalStrategy(min_confidence=60.0, atr_stop_multiplier=1.25)
    strategy.risk_mode = "custom"
    strategy.global_atr_stop_multiplier = 0.5

    signal = strategy.generate_signal(
        current_price=101.0,
        ohlcv=_ohlcv(),
        indicators=_indicators(_absorption_flow(), atr=0.4),
        regime=Regime.TRENDING,
        timestamp=_ts(),
    )

    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.stop_loss == pytest.approx(101.0 - (0.4 * 1.25), rel=1e-6)


def test_strategy_to_dict_exposes_exit_and_risk_source_fields() -> None:
    strategy = MomentumFlowStrategy(trailing_stop_pct=0.9)
    strategy.exit_mode = "global"
    strategy.risk_mode = "global"
    strategy.trailing_stop_mode = "global"
    strategy.global_trailing_stop_pct = 0.55
    strategy.global_rr_ratio = 1.75
    strategy.global_atr_stop_multiplier = 0.7
    payload = strategy.to_dict()

    assert payload["exit_mode"] == "global"
    assert payload["risk_mode"] == "global"
    assert payload["trailing_stop_mode"] == "global"
    assert payload["global_trailing_stop_pct"] == pytest.approx(0.55)
    assert payload["effective_trailing_stop_pct"] == pytest.approx(0.55)
    assert payload["global_rr_ratio"] == pytest.approx(1.75)
    assert payload["effective_rr_ratio"] == pytest.approx(1.75)
    assert payload["global_atr_stop_multiplier"] == pytest.approx(0.7)
    assert payload["effective_atr_stop_multiplier"] == pytest.approx(0.7)
