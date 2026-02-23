"""Regime/flow metric regression tests for day_trading_regime_impl."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from src.day_trading_manager import BarData, DayTradingManager
from src.strategies.base_strategy import Regime


def _build_l2_bar(
    ts: datetime,
    close: float,
    *,
    delta: float,
    l2_volume: float,
    imbalance: float,
    book_pressure: float,
    volume: float,
) -> BarData:
    return BarData(
        timestamp=ts,
        open=close - 0.05,
        high=close + 0.12,
        low=close - 0.12,
        close=close,
        volume=volume,
        l2_delta=delta,
        l2_volume=l2_volume,
        l2_imbalance=imbalance,
        l2_book_pressure=book_pressure,
        l2_bid_depth_total=max(100.0, l2_volume * 0.55),
        l2_ask_depth_total=max(100.0, l2_volume * 0.45),
    )


def test_no_l2_trending_requires_efficiency_and_adx() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    regime = manager._classify_micro_regime(
        trend_efficiency=0.30,  # below no-L2 trend threshold
        adx=38.0,               # high ADX alone must not force TRENDING
        volatility=0.008,
        price_change_pct=0.40,
        flow={"has_l2_coverage": False, "price_change_pct": 0.40},
    )
    assert regime != "TRENDING_UP"


def test_flow_score_trend_is_neutral_for_stable_flow_profile() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    start = datetime(2026, 2, 4, 14, 30)
    deltas = [120.0, 90.0, 150.0]
    l2_volumes = [3100.0, 3300.0, 3200.0]
    bars: list[BarData] = []
    for idx in range(50):
        bars.append(
            _build_l2_bar(
                start + timedelta(minutes=idx),
                close=100.0 + idx * 0.08,
                delta=deltas[idx % len(deltas)],
                l2_volume=l2_volumes[idx % len(l2_volumes)],
                imbalance=0.12,
                book_pressure=0.06,
                volume=20_000.0,
            )
        )

    flow = manager._calculate_order_flow_metrics(bars, lookback=20)
    assert abs(float(flow.get("flow_score_trend_3bar", 0.0))) < 1.0


def test_flow_score_is_comparable_across_l2_scales() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    start = datetime(2026, 2, 4, 14, 30)
    bars_base: list[BarData] = []
    bars_scaled: list[BarData] = []
    for idx in range(35):
        close = 100.0 + idx * 0.10
        delta = 130.0 + (idx % 4) * 18.0
        l2_volume = 2900.0 + (idx % 5) * 140.0
        bars_base.append(
            _build_l2_bar(
                start + timedelta(minutes=idx),
                close=close,
                delta=delta,
                l2_volume=l2_volume,
                imbalance=0.10,
                book_pressure=0.07,
                volume=18_000.0,
            )
        )
        bars_scaled.append(
            _build_l2_bar(
                start + timedelta(minutes=idx),
                close=close,
                delta=delta * 100.0,
                l2_volume=l2_volume * 100.0,
                imbalance=0.10,
                book_pressure=0.07,
                volume=1_800_000.0,
            )
        )

    flow_base = manager._calculate_order_flow_metrics(bars_base, lookback=20)
    flow_scaled = manager._calculate_order_flow_metrics(bars_scaled, lookback=20)
    diff = abs(float(flow_base["flow_score"]) - float(flow_scaled["flow_score"]))
    assert diff < 6.0


def test_transition_is_listed_as_valid_micro_regime_key() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    assert "TRANSITION" in manager.MICRO_REGIME_KEYS


def test_regime_refresh_bars_supports_sub_three_values() -> None:
    manager = DayTradingManager(regime_detection_minutes=0, regime_refresh_bars=1)
    assert manager.regime_refresh_bars == 1


def test_adaptive_refresh_does_not_require_double_confirmation() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-04")
    start = datetime(2026, 2, 4, 14, 30)
    session.bars = [
        _build_l2_bar(
            start + timedelta(minutes=i),
            close=100.0 + i * 0.1,
            delta=120.0,
            l2_volume=3200.0,
            imbalance=0.10,
            book_pressure=0.05,
            volume=20_000.0,
        )
        for i in range(25)
    ]
    session.detected_regime = Regime.CHOPPY
    session.micro_regime = "MIXED"
    session.active_strategies = ["mean_reversion"]
    session.last_regime_refresh_bar_index = 0
    session.regime_refresh_bars = 3
    session.orchestrator = SimpleNamespace(
        config=SimpleNamespace(use_adaptive_regime=True),
        current_feature_vector=object(),
        current_regime_state=SimpleNamespace(transition_velocity=0.0, micro_regime="MIXED"),
    )

    manager._detect_regime = lambda _session: Regime.TRENDING
    manager._select_strategies = lambda _session: ["momentum_flow"]

    payload = manager._maybe_refresh_regime(
        session=session,
        current_bar_index=3,
        timestamp=session.bars[-1].timestamp,
    )

    assert payload is not None
    assert session.detected_regime == Regime.TRENDING
