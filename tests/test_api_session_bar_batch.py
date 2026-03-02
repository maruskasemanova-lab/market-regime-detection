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


def test_process_bars_accepts_arrow_ipc_batch(monkeypatch) -> None:
    client = TestClient(api_server.app)
    captured: List[Dict[str, Any]] = []

    def _fake_process_bar(
        *,
        run_id: str,
        ticker: str,
        timestamp: datetime,
        bar_data: Dict[str, Any],
        warmup_only: bool = False,
    ) -> Dict[str, Any]:
        captured.append(
            {
                "run_id": run_id,
                "ticker": ticker,
                "timestamp": timestamp,
                "bar_data": dict(bar_data),
                "warmup_only": bool(warmup_only),
            }
        )
        return {
            "phase": "TRADING",
            "action": "hold",
            "signals": [],
            "warmup_only": bool(warmup_only),
        }

    monkeypatch.setattr(api_server.day_trading_manager, "process_bar", _fake_process_bar)

    rows = [
        {
            "run_id": "batch-api-1",
            "ticker": "MU",
            "timestamp": datetime(2026, 2, 27, 14, 30, tzinfo=timezone.utc),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
            "volume": 1_000.0,
            "warmup_only": True,
            "l2_book_pressure_delta": 0.2,
        },
        {
            "run_id": "batch-api-1",
            "ticker": "MU",
            "timestamp": datetime(2026, 2, 27, 14, 31, tzinfo=timezone.utc),
            "open": 100.2,
            "high": 100.8,
            "low": 100.0,
            "close": 100.6,
            "volume": 1_200.0,
            "l2_quality_flags": ["good"],
        },
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

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["bars_processed"] == 2
    assert len(payload["results"]) == 2
    assert len(captured) == 2
    assert captured[0]["warmup_only"] is True
    assert captured[0]["bar_data"]["l2_book_pressure_change"] == 0.2
    assert captured[1]["bar_data"]["l2_quality_flags"] == ["good"]
    assert captured[1]["timestamp"] == datetime(2026, 2, 27, 14, 31, tzinfo=timezone.utc)
