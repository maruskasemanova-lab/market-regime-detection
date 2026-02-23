"""Fixture-backed tests for macro/micro regime preference selection on MU data."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, Any, Dict, List

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.day_trading_manager import BarData, DayTradingManager
from src.strategies.base_strategy import Regime


_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mu_regime_preference_cases.json"


def _load_fixture_cases() -> List[Dict[str, Any]]:
    payload = json.loads(_FIXTURE_PATH.read_text())
    cases = payload.get("cases", [])
    assert isinstance(cases, list) and cases, "MU fixture cases must be non-empty"
    return cases


def _new_manager() -> DayTradingManager:
    manager = DayTradingManager(regime_detection_minutes=0)
    manager._load_aos_config = lambda *_args, **_kwargs: None
    manager.ticker_params = {}
    return manager


def _to_bars(rows: List[Dict[str, Any]]) -> List[BarData]:
    bars: List[BarData] = []
    for row in rows:
        bars.append(
            BarData(
                timestamp=datetime.fromisoformat(str(row["timestamp"])),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return bars


def _configure_enabled_set(manager: DayTradingManager, enabled_names: set[str]) -> None:
    all_regimes = [Regime.TRENDING, Regime.CHOPPY, Regime.MIXED]
    for name, strategy in manager.strategies.items():
        strategy.enabled = name in enabled_names
        if name in enabled_names and hasattr(strategy, "allowed_regimes"):
            strategy.allowed_regimes = list(all_regimes)


def _adaptive_pref_config(*, micro_preferences: Dict[str, List[str]]) -> Dict[str, Any]:
    return {
        "regime_preferences": {
            "TRENDING": ["gap_liquidity"],
            "CHOPPY": ["volume_profile"],
            "MIXED": ["rotation"],
        },
        "micro_regime_preferences": micro_preferences,
        "flow_bias_enabled": False,
        "use_ohlcv_fallbacks": False,
        "strict_preference_selection": True,
    }


def _case_by_id(cases: List[Dict[str, Any]], case_id: str) -> Dict[str, Any]:
    for case in cases:
        if case.get("case_id") == case_id:
            return case
    raise AssertionError(f"Fixture case not found: {case_id}")


def _run_case(manager: DayTradingManager, case: Dict[str, Any]) -> tuple[str, str, List[str], List[str]]:
    session = manager.get_or_create_session("fixture-run", "MU", case["asof"])
    session.bars = _to_bars(case["bars"])
    detected_macro = manager._detect_regime(session)
    session.detected_regime = detected_macro
    selected = manager._select_strategies(session)
    return (
        str(detected_macro.value),
        str(session.micro_regime),
        list(selected),
        list(getattr(session, "selection_warnings", [])),
    )


def test_mu_fixture_regime_detection_stays_stable_for_reference_windows() -> None:
    manager = _new_manager()
    cases = _load_fixture_cases()

    for case in cases:
        macro, micro, _selected, _warnings = _run_case(manager, case)
        assert macro == case["expected_macro"], (
            f"macro mismatch for {case['case_id']} @ {case['asof']}: "
            f"expected {case['expected_macro']}, got {macro}"
        )
        assert micro == case["expected_micro"], (
            f"micro mismatch for {case['case_id']} @ {case['asof']}: "
            f"expected {case['expected_micro']}, got {micro}"
        )


def test_mu_fixture_micro_preferences_override_macro_preferences_when_present() -> None:
    manager = _new_manager()
    enabled = {
        "vwap_magnet",
        "momentum",
        "exhaustion_fade",
        "absorption_reversal",
        "momentum_flow",
        "gap_liquidity",
        "mean_reversion",
        "volume_profile",
        "rotation",
    }
    _configure_enabled_set(manager, enabled)
    manager.ticker_params["MU"] = {
        "adaptive": _adaptive_pref_config(
            micro_preferences={
                "TRENDING_UP": ["momentum_flow"],
                "TRENDING_DOWN": ["momentum"],
                "CHOPPY": ["absorption_reversal"],
                "ABSORPTION": ["mean_reversion"],
                "BREAKOUT": ["gap_liquidity"],
                "MIXED": ["exhaustion_fade"],
                "TRANSITION": ["rotation"],
                "UNKNOWN": ["vwap_magnet"],
            }
        )
    }

    expected_first_by_case = {
        "unknown_opening": "exhaustion_fade",
        "trending_down_morning": "momentum",
        "mixed_after_open": "exhaustion_fade",
        "choppy_late_morning": "absorption_reversal",
        "trending_up_noon": "momentum_flow",
    }
    cases = _load_fixture_cases()

    hits = 0
    for case in cases:
        _macro, _micro, selected, warnings = _run_case(manager, case)
        assert warnings == [], f"Unexpected strict-selection warning for {case['case_id']}: {warnings}"
        assert selected, f"Expected non-empty strategy selection for {case['case_id']}"
        expected_first = expected_first_by_case[case["case_id"]]
        if selected[0] == expected_first:
            hits += 1
        assert selected[0] == expected_first, (
            f"first strategy mismatch for {case['case_id']}: "
            f"expected {expected_first}, got {selected[0]}"
        )

    assert hits == len(expected_first_by_case)


def test_mu_fixture_missing_micro_preferences_do_not_fallback_and_emit_warning() -> None:
    manager = _new_manager()
    enabled = {"absorption_reversal", "volume_profile", "rotation", "gap_liquidity"}
    _configure_enabled_set(manager, enabled)
    adaptive_cfg = _adaptive_pref_config(
        micro_preferences={
            "TRENDING_UP": [],
            "TRENDING_DOWN": [],
            "CHOPPY": [],
            "ABSORPTION": [],
            "BREAKOUT": [],
            "MIXED": [],
            "TRANSITION": [],
            "UNKNOWN": [],
        }
    )
    adaptive_cfg["regime_preferences"]["CHOPPY"] = []
    manager.ticker_params["MU"] = {
        "adaptive": adaptive_cfg
    }

    cases = _load_fixture_cases()
    choppy_case = _case_by_id(cases, "choppy_late_morning")
    macro, micro, selected, warnings = _run_case(manager, choppy_case)

    assert macro == "CHOPPY"
    assert micro == "CHOPPY"
    assert selected == []
    assert warnings
    warning_blob = " | ".join(warnings)
    assert "missing micro_regime_preferences" in warning_blob
    assert "missing regime_preferences" in warning_blob


def test_mu_fixture_strict_warnings_surface_in_regime_refresh_payload() -> None:
    manager = _new_manager()
    enabled = {"absorption_reversal", "volume_profile", "rotation", "gap_liquidity"}
    _configure_enabled_set(manager, enabled)
    adaptive_cfg = _adaptive_pref_config(
        micro_preferences={
            "TRENDING_UP": [],
            "TRENDING_DOWN": [],
            "CHOPPY": [],
            "ABSORPTION": [],
            "BREAKOUT": [],
            "MIXED": [],
            "TRANSITION": [],
            "UNKNOWN": [],
        }
    )
    adaptive_cfg["regime_preferences"]["CHOPPY"] = []
    manager.ticker_params["MU"] = {"adaptive": adaptive_cfg}

    cases = _load_fixture_cases()
    choppy_case = _case_by_id(cases, "choppy_late_morning")
    session = manager.get_or_create_session("fixture-refresh", "MU", choppy_case["asof"])
    session.bars = _to_bars(choppy_case["bars"])
    session.detected_regime = Regime.CHOPPY
    session.micro_regime = "CHOPPY"
    session.active_strategies = ["volume_profile"]
    session.last_regime_refresh_bar_index = -1
    session.regime_refresh_bars = 3

    payload = manager._maybe_refresh_regime(
        session=session,
        current_bar_index=len(session.bars) - 1,
        timestamp=session.bars[-1].timestamp,
    )

    assert payload is not None
    assert payload["strategies"] == []
    warning_blob = " | ".join(payload.get("selection_warnings", []))
    assert "missing micro_regime_preferences" in warning_blob
    assert "missing regime_preferences" in warning_blob
