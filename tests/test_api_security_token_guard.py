from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _load_app(monkeypatch, token: str):
    monkeypatch.setenv("STRATEGY_INTERNAL_API_TOKEN", token)
    import api_server  # noqa: WPS433

    module = importlib.reload(api_server)
    return module.app


def test_protected_endpoint_requires_internal_token(monkeypatch):
    app = _load_app(monkeypatch, "internal-secret")
    client = TestClient(app)

    resp = client.post("/api/orchestrator/reset")
    assert resp.status_code == 403


def test_public_endpoint_still_accessible_without_internal_token(monkeypatch):
    app = _load_app(monkeypatch, "internal-secret")
    client = TestClient(app)

    resp = client.get("/api/strategies")
    assert resp.status_code == 200


def test_orchestrator_checkpoints_readonly_endpoint_is_public(monkeypatch):
    app = _load_app(monkeypatch, "internal-secret")
    client = TestClient(app)

    resp = client.get("/api/orchestrator/checkpoints")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_protected_endpoint_accepts_correct_internal_token(monkeypatch):
    app = _load_app(monkeypatch, "internal-secret")
    client = TestClient(app)

    resp = client.post(
        "/api/orchestrator/reset",
        headers={"x-internal-token": "internal-secret"},
    )
    assert resp.status_code == 200


def test_protected_endpoint_403_contains_cors_headers_for_allowed_origin(monkeypatch):
    app = _load_app(monkeypatch, "internal-secret")
    client = TestClient(app)

    resp = client.post(
        "/api/orchestrator/reset",
        headers={"origin": "http://localhost:5173"},
    )
    assert resp.status_code == 403
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
