"""Tests for evidence guardrails in DayTradingManager."""
from datetime import datetime, time, timezone

from src.day_trading_manager import DayTradingManager
from src.strategies.base_strategy import Position, Regime, Signal, SignalType


def _signal(
    strategy_name: str = "MomentumFlow",
    signal_type: SignalType = SignalType.BUY,
    metadata: dict | None = None,
) -> Signal:
    return Signal(
        strategy_name=strategy_name,
        signal_type=signal_type,
        price=100.0,
        timestamp=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        confidence=80.0,
        stop_loss=99.0,
        take_profit=103.0,
        trailing_stop=False,
        reasoning="unit-test",
        metadata=metadata or {},
    )


def test_required_confirming_sources_is_dynamic() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03", 100.0)

    session.detected_regime = Regime.TRENDING
    session.micro_regime = "TRENDING_UP"
    assert manager._required_confirming_sources(session, time(9, 45)) == 2
    assert manager._required_confirming_sources(session, time(11, 0)) == 3

    session.detected_regime = Regime.CHOPPY
    session.micro_regime = "CHOPPY"
    assert manager._required_confirming_sources(session, time(11, 0)) == 4


def test_mu_choppy_filter_blocks_only_mu() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    mu_session = manager.get_or_create_session("run", "MU", "2026-02-03", 100.0)
    nvda_session = manager.get_or_create_session("run", "NVDA", "2026-02-03", 100.0)

    mu_session.micro_regime = "CHOPPY"
    assert manager._is_mu_choppy_blocked(mu_session, Regime.TRENDING) is True

    mu_session.micro_regime = "TRENDING_UP"
    assert manager._is_mu_choppy_blocked(mu_session, Regime.CHOPPY) is True

    nvda_session.micro_regime = "CHOPPY"
    assert manager._is_mu_choppy_blocked(nvda_session, Regime.CHOPPY) is False


def test_momentum_flow_requires_delta_divergence_confirmation() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)

    no_delta_signal = _signal(
        metadata={
            "evidence_sources": [
                {"type": "feature", "name": "rsi_oversold", "direction": "bullish"},
                {"type": "strategy_signal", "name": "momentum_flow", "direction": "bullish"},
            ],
            "layer_scores": {"source_weights": {"strategy_signal:momentum_flow": 0.6}},
        }
    )
    passed, metrics = manager._passes_momentum_flow_delta_confirmation(no_delta_signal)
    assert passed is False
    assert metrics["reason"] == "momentum_flow_missing_delta_divergence"

    with_delta_signal = _signal(
        metadata={
            "evidence_sources": [
                {"type": "l2_flow", "name": "delta_divergence", "direction": "bullish"},
                {"type": "strategy_signal", "name": "momentum_flow", "direction": "bullish"},
            ],
            "layer_scores": {
                "source_weights": {
                    "l2_flow:delta_divergence": 0.35,
                    "strategy_signal:momentum_flow": 0.65,
                }
            },
        }
    )
    passed, _ = manager._passes_momentum_flow_delta_confirmation(with_delta_signal)
    assert passed is True


def test_confirming_source_stats_prefers_ensemble_count() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    signal = _signal(
        metadata={
            "evidence_sources": [
                {"type": "feature", "name": "rsi_oversold", "direction": "bullish"},
                {"type": "l2_flow", "name": "aggression", "direction": "bullish"},
                {"type": "strategy_signal", "name": "momentum_flow", "direction": "bullish"},
            ],
            "layer_scores": {"confirming_sources": 1},
        }
    )

    stats = manager._confirming_source_stats(signal)
    assert stats["confirming_sources"] == 1
    assert stats["aligned_evidence_sources"] == 3
    assert stats["count_source"] == "ensemble_layer_scores"


def test_extract_confirming_source_keys_excludes_primary_strategy() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    signal_metadata = {
        "layer_scores": {
            "aligned_source_keys": [
                "strategy:momentum_flow",
                "l2_flow:delta_divergence",
                "feature:rsi_oversold",
                "feature:rsi_oversold",
            ]
        }
    }

    keys = manager._extract_confirming_source_keys_from_metadata(
        signal_metadata=signal_metadata,
        side="long",
        strategy_name="MomentumFlow",
    )
    assert keys == ["l2_flow:delta_divergence", "feature:rsi_oversold"]


def test_extract_confirming_source_keys_falls_back_to_evidence_sources() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    signal_metadata = {
        "evidence_sources": [
            {"type": "strategy", "name": "momentum_flow", "direction": "bullish"},
            {"type": "feature", "name": "rsi_oversold", "direction": "bullish"},
            {"type": "l2_flow", "name": "aggression", "direction": "bearish"},
        ]
    }

    keys = manager._extract_confirming_source_keys_from_metadata(
        signal_metadata=signal_metadata,
        side="long",
        strategy_name="momentum_flow",
    )
    assert keys == ["feature:rsi_oversold"]


def test_extract_raw_confidence_prefers_adjusted_then_strategy_score() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    adjusted = manager._extract_raw_confidence_from_metadata(
        {
            "confidence_adjustment": {"adjusted_confidence": 77.5},
            "layer_scores": {"strategy_score": 88.0},
        }
    )
    assert adjusted == 77.5

    from_strategy = manager._extract_raw_confidence_from_metadata(
        {"layer_scores": {"strategy_score": 88.0}}
    )
    assert from_strategy == 88.0


def test_close_position_forwards_confirming_sources_and_confidence() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING

    class _StubOrchestrator:
        def __init__(self):
            self.calls = []

        def record_trade_outcome(self, **kwargs):
            self.calls.append(kwargs)

    stub = _StubOrchestrator()
    session.orchestrator = stub

    position = Position(
        strategy_name="momentum_flow",
        entry_price=100.0,
        entry_time=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        side="long",
        size=10.0,
        stop_loss=99.0,
    )
    position.signal_metadata = {
        "confidence_adjustment": {"adjusted_confidence": 74.0},
        "layer_scores": {
            "strategy_score": 80.0,
            "aligned_source_keys": [
                "strategy:momentum_flow",
                "feature:rsi_oversold",
                "l2_flow:delta_divergence",
            ],
        },
    }
    session.active_position = position

    manager._close_position(
        session=session,
        exit_price=101.0,
        exit_time=datetime(2026, 2, 3, 14, 35, tzinfo=timezone.utc),
        reason="take_profit",
        bar_volume=100_000.0,
    )

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["raw_confidence"] == 74.0
    assert call["confirming_sources"] == [
        "feature:rsi_oversold",
        "l2_flow:delta_divergence",
    ]
