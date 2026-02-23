from src.day_trading_manager import DayTradingManager
from src.feature_store import FeatureVector
from src.intraday_levels import ensure_intraday_levels_state
from src.trading_config import TradingConfig


def _session_with_sweep_config():
    manager = DayTradingManager()
    session = manager.get_or_create_session(
        run_id="sweep-test",
        ticker="MU",
        date="2026-02-10",
    )
    session.apply_trading_config(
        TradingConfig.from_dict(
            {
                "liquidity_sweep_detection_enabled": True,
                "sweep_min_aggression_z": -2.0,
                "sweep_min_book_pressure_z": 1.5,
                "sweep_max_price_change_pct": 0.05,
                "intraday_levels_entry_tolerance_pct": 0.10,
            }
        )
    )
    session.intraday_levels_state = ensure_intraday_levels_state(
        {
            "levels": [
                {
                    "id": 1,
                    "kind": "support",
                    "price": 100.0,
                    "tests": 2,
                    "source": "swing",
                    "broken": False,
                }
            ]
        }
    )
    return manager, session


def test_detect_liquidity_sweep_sets_pending_state_when_divergence_matches() -> None:
    manager, session = _session_with_sweep_config()
    fv = FeatureVector(
        l2_aggression_z=-2.6,
        l2_book_pressure_z=2.1,
        tf5_trend_slope=0.01,
    )

    payload = manager._detect_liquidity_sweep(
        session=session,
        current_price=100.03,
        fv=fv,
    )

    assert payload["sweep_detected"] is True
    assert payload["direction"] == "long"
    assert session.potential_sweep_active is True
    assert session.potential_sweep_context.get("direction") == "long"
    assert session.potential_sweep_context.get("level_kind") == "support"


def test_detect_liquidity_sweep_rejects_when_divergence_does_not_match() -> None:
    manager, session = _session_with_sweep_config()
    fv = FeatureVector(
        l2_aggression_z=-1.0,
        l2_book_pressure_z=0.2,
        tf5_trend_slope=0.01,
    )

    payload = manager._detect_liquidity_sweep(
        session=session,
        current_price=100.03,
        fv=fv,
    )

    assert payload["sweep_detected"] is False
    assert payload["reason"] == "divergence_not_met"
    assert session.potential_sweep_active is False


def test_detect_liquidity_sweep_rejects_without_l2_coverage() -> None:
    manager, session = _session_with_sweep_config()
    fv = FeatureVector(
        l2_aggression_z=-2.8,
        l2_book_pressure_z=2.2,
        tf5_trend_slope=0.01,
    )

    payload = manager._detect_liquidity_sweep(
        session=session,
        current_price=100.03,
        fv=fv,
        flow_metrics={"has_l2_coverage": False},
    )

    assert payload["sweep_detected"] is False
    assert payload["reason"] == "insufficient_l2_coverage"
    assert session.potential_sweep_active is False
