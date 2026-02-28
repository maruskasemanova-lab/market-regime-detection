from src.api_models import BarInput
from src.api_server_helpers.bar_payload import (
    build_day_trading_bar_payload,
    parse_bar_timestamp,
    sanitize_non_finite_numbers,
)


def _bar_input(**overrides):
    payload = {
        "run_id": "run-1",
        "ticker": "MU",
        "timestamp": "2026-02-27T14:30:00Z",
        "open": 100.0,
        "high": 101.0,
        "low": 99.5,
        "close": 100.5,
        "volume": 12345.0,
        "vwap": 100.2,
        "l2_delta": 15.0,
        "l2_book_pressure": 0.2,
        "l2_quality_flags": ["ok"],
        "l2_quality": {"coverage": 0.9},
        "intrabar_quotes_1s": [{"s": 0, "bid": 100.4, "ask": 100.6}],
        "tcbbo_net_premium": 2500.0,
        "tcbbo_has_data": True,
    }
    payload.update(overrides)
    return BarInput(**payload)


def test_build_day_trading_bar_payload_respects_l2_quality_flag_toggle():
    bar = _bar_input()

    intrabar_payload = build_day_trading_bar_payload(
        bar,
        include_l2_quality_flags=False,
    )
    session_payload = build_day_trading_bar_payload(
        bar,
        include_l2_quality_flags=True,
    )

    assert "l2_quality_flags" not in intrabar_payload
    assert session_payload["l2_quality_flags"] == ["ok"]
    assert intrabar_payload["tcbbo_net_premium"] == 2500.0
    assert session_payload["intrabar_quotes_1s"][0]["s"] == 0


def test_parse_bar_timestamp_accepts_zulu_suffix():
    parsed = parse_bar_timestamp("2026-02-27T14:30:00Z")
    assert parsed.isoformat() == "2026-02-27T14:30:00+00:00"


def test_sanitize_non_finite_numbers_recursive():
    payload = {
        "a": float("nan"),
        "b": [1.0, float("inf"), {"c": float("-inf")}],
    }
    sanitized = sanitize_non_finite_numbers(payload)
    assert sanitized == {"a": None, "b": [1.0, None, {"c": None}]}
