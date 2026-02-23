from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_manager import DayTradingManager
from src.day_trading_models import BarData, TradingSession
from src.strategies.base_strategy import Signal, SignalType


def test_open_position_populates_context_risk_snapshot_from_effective_levels() -> None:
    manager = DayTradingManager(regime_detection_minutes=5)
    timestamp = datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc)

    session = TradingSession(run_id="run-entry-risk", ticker="MU", date="2026-02-03")
    session.stop_loss_mode = "capped"
    session.fixed_stop_loss_pct = 0.30
    session.bars.append(
        BarData(
            timestamp=timestamp,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=100_000.0,
        )
    )

    signal = Signal(
        strategy_name="VWAPMagnet",
        signal_type=SignalType.BUY,
        price=100.0,
        timestamp=timestamp,
        confidence=92.0,
        stop_loss=99.60,  # looser than capped 0.30% floor -> effective stop should be 99.70
        take_profit=101.20,
        reasoning="unit-test entry context",
        metadata={},
    )

    position = manager._open_position(
        session=session,
        signal=signal,
        entry_price=100.0,
        entry_time=timestamp,
        signal_bar_index=4,
        entry_bar_index=5,
        entry_bar_volume=100_000.0,
    )

    metadata = position.signal_metadata if isinstance(position.signal_metadata, dict) else {}
    risk_controls = metadata.get("risk_controls") if isinstance(metadata.get("risk_controls"), dict) else {}
    context_risk = metadata.get("context_risk") if isinstance(metadata.get("context_risk"), dict) else {}

    assert risk_controls.get("stop_loss_mode") == "capped"
    assert risk_controls.get("effective_stop_loss") == pytest.approx(99.70, abs=1e-6)
    assert risk_controls.get("strategy_stop_loss") == pytest.approx(99.60, abs=1e-6)

    assert context_risk.get("sl_reason") == "capped_fixed_floor:0.3000"
    assert context_risk.get("tp_reason") == "strategy_take_profit"
    assert context_risk.get("risk_pct") == pytest.approx(0.30, abs=1e-6)
    assert context_risk.get("effective_rr") == pytest.approx(4.0, abs=1e-6)
    assert context_risk.get("skip") is False
    assert context_risk.get("skip_reason") == "ok"

    # Runtime uses signal metadata for position_opened payload; keep it enriched too.
    assert isinstance(signal.metadata, dict)
    assert signal.metadata.get("context_risk", {}).get("risk_pct") == pytest.approx(
        0.30, abs=1e-6
    )
