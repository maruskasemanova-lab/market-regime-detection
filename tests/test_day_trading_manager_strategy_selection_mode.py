from typing import Any, Dict, List, Optional, Tuple
"""Tests for strategy-selection modes in DayTradingManager."""

from datetime import datetime, timedelta
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_manager import BarData, DayTradingManager
from src.strategies.base_strategy import Regime


def _configure_enabled_set(manager: DayTradingManager, enabled_names: set[str]) -> None:
    for name, strategy in manager.strategies.items():
        strategy.enabled = name in enabled_names
        if name in enabled_names and hasattr(strategy, "allowed_regimes"):
            strategy.allowed_regimes = [Regime.TRENDING]


def _make_l2_bars(count: int = 10) -> list[BarData]:
    start = datetime(2026, 2, 3, 14, 30)
    bars: list[BarData] = []
    for idx in range(count):
        bars.append(
            BarData(
                timestamp=start + timedelta(minutes=idx),
                open=100.0 + idx * 0.1,
                high=100.2 + idx * 0.1,
                low=99.8 + idx * 0.1,
                close=100.1 + idx * 0.1,
                volume=20000.0 + idx * 500.0,
                l2_delta=1200.0 + idx * 50.0,
                l2_buy_volume=8000.0 + idx * 120.0,
                l2_sell_volume=6800.0 + idx * 80.0,
                l2_volume=14800.0 + idx * 200.0,
                l2_imbalance=0.16,
                l2_bid_depth_total=4200.0 + idx * 20.0,
                l2_ask_depth_total=3900.0 + idx * 15.0,
                l2_book_pressure=0.08,
                l2_book_pressure_change=0.01,
                l2_iceberg_buy_count=1.0,
                l2_iceberg_sell_count=0.0,
                l2_iceberg_bias=0.12,
            )
        )
    return bars


def _adaptive_pref_config(
    *,
    flow_bias_enabled: bool,
    use_ohlcv_fallbacks: bool,
    min_active_bars_before_switch: int = 0,
    switch_cooldown_bars: int = 0,
    momentum_diversification: Optional[dict] = None,
) -> dict:
    payload = {
        "regime_preferences": {
            "TRENDING": ["mean_reversion"],
            "CHOPPY": ["mean_reversion"],
            "MIXED": ["mean_reversion"],
        },
        "micro_regime_preferences": {
            "TRENDING_UP": [],
            "TRENDING_DOWN": [],
            "CHOPPY": [],
            "ABSORPTION": [],
            "BREAKOUT": [],
            "MIXED": [],
            "TRANSITION": [],
            "UNKNOWN": [],
        },
        "flow_bias_enabled": flow_bias_enabled,
        "use_ohlcv_fallbacks": use_ohlcv_fallbacks,
        "flow_bias_strategies": ["momentum_flow"],
        "min_active_bars_before_switch": min_active_bars_before_switch,
        "switch_cooldown_bars": switch_cooldown_bars,
    }
    if momentum_diversification is not None:
        payload["momentum_diversification"] = momentum_diversification
    return payload


def _prepare_switch_guard_session(manager: DayTradingManager):
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "MIXED"
    session.strategy_selection_mode = "adaptive_top_n"
    session.max_active_strategies = 1
    session.regime_refresh_bars = 3
    session.bars = _make_l2_bars(32)
    session.active_strategies = ["momentum_flow"]
    session.selected_strategy = "momentum_flow"
    session.last_strategy_switch_bar_index = 20
    session.last_regime_refresh_bar_index = 20
    manager._detect_regime = lambda _session: Regime.TRENDING
    return session


def test_adaptive_top_n_respects_max_active_strategies() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"momentum_flow", "absorption_reversal", "exhaustion_fade", "mean_reversion"}
    _configure_enabled_set(manager, enabled)

    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "TRENDING_UP"
    session.strategy_selection_mode = "adaptive_top_n"
    session.max_active_strategies = 2
    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=False,
            use_ohlcv_fallbacks=False,
        ),
    }
    manager.ticker_params["MU"]["adaptive"]["regime_preferences"]["TRENDING"] = [
        "momentum_flow",
        "mean_reversion",
    ]

    selected = manager._select_strategies(session)
    assert len(selected) == 2
    assert set(selected).issubset(enabled)


def test_all_enabled_mode_returns_all_enabled_regime_compatible_strategies() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"momentum_flow", "absorption_reversal", "exhaustion_fade", "mean_reversion"}
    _configure_enabled_set(manager, enabled)

    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "TRENDING_UP"
    session.strategy_selection_mode = "all_enabled"
    session.max_active_strategies = 1
    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=False,
            use_ohlcv_fallbacks=False,
        ),
    }

    selected = manager._select_strategies(session)
    assert set(selected) == enabled
    assert len(selected) == len(enabled)


def test_adaptive_regime_preferences_override_default_priority_order() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"mean_reversion", "gap_liquidity"}
    _configure_enabled_set(manager, enabled)

    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "MIXED"
    session.strategy_selection_mode = "adaptive_top_n"
    session.max_active_strategies = 1

    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=False,
            use_ohlcv_fallbacks=False,
        ),
    }

    selected = manager._select_strategies(session)
    assert selected == ["mean_reversion"]


def test_flow_bias_toggle_switches_l2_priority_source() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"momentum_flow", "mean_reversion"}
    _configure_enabled_set(manager, enabled)

    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "MIXED"
    session.strategy_selection_mode = "adaptive_top_n"
    session.max_active_strategies = 1
    session.bars = _make_l2_bars(12)

    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=True,
            use_ohlcv_fallbacks=True,
        ),
    }

    selected_with_flow_bias = manager._select_strategies(session)
    assert selected_with_flow_bias == ["momentum_flow"]

    manager.ticker_params["MU"]["adaptive"]["flow_bias_enabled"] = False
    selected_without_flow_bias = manager._select_strategies(session)
    assert selected_without_flow_bias == ["mean_reversion"]


def test_min_active_bars_holds_strategy_switch_until_minimum_age() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"momentum_flow", "mean_reversion"}
    _configure_enabled_set(manager, enabled)
    session = _prepare_switch_guard_session(manager)
    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=False,
            use_ohlcv_fallbacks=False,
            min_active_bars_before_switch=8,
            switch_cooldown_bars=0,
        ),
    }

    payload = manager._maybe_refresh_regime(
        session=session,
        current_bar_index=25,
        timestamp=session.bars[-1].timestamp,
    )

    assert session.active_strategies == ["momentum_flow"]
    assert session.last_strategy_switch_bar_index == 20
    assert payload is None


def test_switch_cooldown_blocks_rapid_strategy_flip() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"momentum_flow", "mean_reversion"}
    _configure_enabled_set(manager, enabled)
    session = _prepare_switch_guard_session(manager)
    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=False,
            use_ohlcv_fallbacks=False,
            min_active_bars_before_switch=0,
            switch_cooldown_bars=7,
        ),
    }

    payload = manager._maybe_refresh_regime(
        session=session,
        current_bar_index=25,
        timestamp=session.bars[-1].timestamp,
    )

    assert session.active_strategies == ["momentum_flow"]
    assert session.last_strategy_switch_bar_index == 20
    assert payload is None


def test_switch_guard_allows_strategy_change_after_thresholds() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"momentum_flow", "mean_reversion"}
    _configure_enabled_set(manager, enabled)
    session = _prepare_switch_guard_session(manager)
    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=False,
            use_ohlcv_fallbacks=False,
            min_active_bars_before_switch=3,
            switch_cooldown_bars=2,
        ),
    }

    payload = manager._maybe_refresh_regime(
        session=session,
        current_bar_index=25,
        timestamp=session.bars[-1].timestamp,
    )

    assert payload is not None
    assert session.active_strategies == ["mean_reversion"]
    assert session.last_strategy_switch_bar_index == 25
    assert payload["switch_guard"]["blocked"] is False


def test_momentum_diversification_route_prioritizes_defensive_stack() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"momentum_flow", "momentum", "pullback", "absorption_reversal", "mean_reversion"}
    _configure_enabled_set(manager, enabled)

    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "CHOPPY"
    session.strategy_selection_mode = "adaptive_top_n"
    session.max_active_strategies = 2
    session.bars = _make_l2_bars(16)

    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=False,
            use_ohlcv_fallbacks=False,
            momentum_diversification={
                "enabled": True,
                "route_enabled": True,
                "route_strategy_map": {
                    "impulse": ["momentum_flow"],
                    "continuation": ["pullback", "momentum_flow"],
                    "defensive": ["absorption_reversal", "mean_reversion"],
                },
                "micro_regime_routes": {
                    "CHOPPY": "defensive",
                    "TRENDING_UP": "impulse",
                    "TRENDING_DOWN": "impulse",
                    "MIXED": "continuation",
                    "ABSORPTION": "defensive",
                    "BREAKOUT": "impulse",
                },
            },
        ),
    }

    selected = manager._select_strategies(session)
    assert selected[:2] == ["absorption_reversal", "mean_reversion"]


def test_momentum_diversification_route_supports_multi_sleeve_map() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    enabled = {"momentum_flow", "pullback", "absorption_reversal", "mean_reversion"}
    _configure_enabled_set(manager, enabled)

    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "CHOPPY"
    session.strategy_selection_mode = "adaptive_top_n"
    session.max_active_strategies = 2
    session.bars = _make_l2_bars(16)

    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            flow_bias_enabled=False,
            use_ohlcv_fallbacks=False,
            momentum_diversification={
                "enabled": True,
                "sleeves": [
                    {
                        "sleeve_id": "impulse",
                        "enabled": True,
                        "route_enabled": True,
                        "apply_to_strategies": ["momentum_flow"],
                        "route_strategy_map": {
                            "impulse": ["momentum_flow"],
                            "continuation": ["pullback"],
                            "defensive": ["momentum_flow"],
                        },
                        "micro_regime_routes": {
                            "CHOPPY": "impulse",
                            "TRENDING_UP": "impulse",
                            "TRENDING_DOWN": "impulse",
                            "MIXED": "continuation",
                            "ABSORPTION": "defensive",
                            "BREAKOUT": "impulse",
                        },
                    },
                    {
                        "sleeve_id": "defensive",
                        "enabled": True,
                        "route_enabled": True,
                        "apply_to_strategies": ["absorption_reversal"],
                        "route_strategy_map": {
                            "impulse": ["pullback"],
                            "continuation": ["pullback"],
                            "defensive": ["absorption_reversal", "mean_reversion"],
                        },
                        "micro_regime_routes": {
                            "CHOPPY": "defensive",
                            "TRENDING_UP": "impulse",
                            "TRENDING_DOWN": "impulse",
                            "MIXED": "continuation",
                            "ABSORPTION": "defensive",
                            "BREAKOUT": "impulse",
                        },
                    },
                ],
            },
        ),
    }

    selected = manager._select_strategies(session)
    assert selected[:2] == ["absorption_reversal", "mean_reversion"]
