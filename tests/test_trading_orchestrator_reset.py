"""Tests for orchestrator reset behavior in deterministic backtests."""
from src.trading_orchestrator import TradingOrchestrator


def _bar(close: float = 100.0) -> dict:
    return {
        "open": close - 0.2,
        "high": close + 0.4,
        "low": close - 0.5,
        "close": close,
        "volume": 10_000.0,
        "vwap": close - 0.1,
    }


def _seed_learning(orch: TradingOrchestrator) -> None:
    orch.record_trade_outcome(
        strategy="momentum",
        regime="TRENDING",
        raw_confidence=72.0,
        was_profitable=True,
        pnl_r=1.4,
        bar_index=10,
    )
    orch.record_trade_outcome(
        strategy="momentum",
        regime="TRENDING",
        raw_confidence=48.0,
        was_profitable=False,
        pnl_r=-1.0,
        bar_index=11,
    )


def test_new_session_keeps_learning_state() -> None:
    orch = TradingOrchestrator()
    _seed_learning(orch)

    before = orch.get_system_health()
    assert before["calibrator"]["global"]["n_trades"] == 2
    assert before["edge_monitor"]["global"]["total_trades"] == 2
    assert before["combiner"]

    orch.new_session()
    after = orch.get_system_health()

    # Per-session pipeline reset, learned memory remains.
    assert after["bar_count"] == 0
    assert after["calibrator"]["global"]["n_trades"] == 2
    assert after["edge_monitor"]["global"]["total_trades"] == 2
    assert after["combiner"]


def test_full_reset_clears_learning_and_runtime() -> None:
    orch = TradingOrchestrator()
    orch.update_bar(_bar(101.0))
    orch.detect_regime()
    _seed_learning(orch)

    before = orch.get_system_health()
    assert before["bar_count"] == 1
    assert before["calibrator"]["global"]["n_trades"] == 2
    assert before["edge_monitor"]["global"]["total_trades"] == 2
    assert before["combiner"]

    orch.full_reset()
    after = orch.get_system_health()

    assert after["bar_count"] == 0
    assert after["regime"] is None
    assert after["cross_asset"] is None
    assert after["calibrator"]["global"]["n_trades"] == 0
    assert after["edge_monitor"]["global"]["total_trades"] == 0
    assert after["combiner"] == {}
