from __future__ import annotations

import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List

import polars as pl
from fastapi.testclient import TestClient

import api_server


def _internal_headers() -> dict[str, str]:
    token = str(os.getenv("STRATEGY_INTERNAL_API_TOKEN") or "").strip()
    if not token:
        return {}
    return {"x-internal-token": token}


def test_api_server_seamlessly_serializes_nan_in_response(monkeypatch) -> None:
    client = TestClient(api_server.app)

    def _fake_process_bar(
        *,
        run_id: str,
        ticker: str,
        timestamp: datetime,
        bar_data: Dict[str, Any],
        warmup_only: bool = False,
    ) -> Dict[str, Any]:
        """Returns a payload containing NaNs and Infinities, both as keys and values."""
        import math
        return {
            "phase": "TRADING",
            "action": "hold",
            "signals": [],
            "warmup_only": False,
            "metrics": {
                "some_nan": float("nan"),
                "some_inf": float("inf"),
                "some_-inf": float("-inf"),
                float("nan"): "Value at NaN key",
                float("inf"): "Value at Inf key"
            }
        }

    monkeypatch.setattr(api_server.day_trading_manager, "process_bar", _fake_process_bar)

    rows = [
        {
            "run_id": "batch-api-2",
            "ticker": "MU",
            "timestamp": datetime(2026, 2, 27, 14, 30, tzinfo=timezone.utc),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
            "volume": 1_000.0,
            "warmup_only": False,
        }
    ]
    buffer = BytesIO()
    pl.DataFrame(rows).write_ipc(buffer)

    response = client.post(
        "/api/session/bars",
        content=buffer.getvalue(),
        headers={
            **_internal_headers(),
            "content-type": "application/vnd.apache.arrow.stream",
        },
    )

    # FastAPI jsonable_encoder would crash with ValueError if dictionaries contained NaN 
    # and were not sanitized correctly by our middleware.
    assert response.status_code == 200, response.text
    payload = response.json()
    
    assert payload["bars_processed"] == 1
    assert len(payload["results"]) == 1
    
    res = payload["results"][0]
    assert res["metrics"]["some_nan"] is None
    assert res["metrics"]["some_inf"] is None
    assert res["metrics"]["some_-inf"] is None
    
    # Check that keys got stringified correctly
    assert res["metrics"]["NaN"] == "Value at NaN key"
    assert res["metrics"]["Infinity"] == "Value at Inf key"
