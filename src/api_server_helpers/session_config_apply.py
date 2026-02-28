"""Application-layer helpers for session config endpoint flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set

from ..runtime_exit_formulas import normalize_runtime_exit_formula_fields
from ..trading_config import TradingConfig
from .session_config import (
    build_session_config_payload,
    merge_body_payload,
    parse_momentum_diversification_payload,
    parse_regime_filter_payload,
    resolve_json_string_override,
    resolve_optional_override,
)


@dataclass
class SessionConfigRequestResolution:
    config_payload: Dict[str, Any]
    momentum_diversification_payload: Optional[Dict[str, Any]]
    max_daily_trades: Any
    mu_choppy_hard_block_enabled: Any


def resolve_session_config_request(
    *,
    local_values: Mapping[str, Any],
    body_payload: Mapping[str, Any],
    query_param_keys: Set[str],
    momentum_diversification_json: str,
    regime_filter_json: str,
    max_daily_trades: Any,
    mu_choppy_hard_block_enabled: Any,
) -> SessionConfigRequestResolution:
    """Resolve query/body overrides into one canonical config payload."""
    config_payload = build_session_config_payload(local_values)
    body_payload = dict(body_payload) if body_payload else {}

    if body_payload:
        merge_body_payload(
            config_payload=config_payload,
            body_payload=body_payload,
            query_param_keys=query_param_keys,
        )
        momentum_diversification_json = resolve_json_string_override(
            override_key="momentum_diversification_json",
            override_value=momentum_diversification_json,
            body_payload=body_payload,
            query_param_keys=query_param_keys,
        )
        regime_filter_json = resolve_json_string_override(
            override_key="regime_filter_json",
            override_value=regime_filter_json,
            body_payload=body_payload,
            query_param_keys=query_param_keys,
        )
        max_daily_trades = resolve_optional_override(
            override_key="max_daily_trades",
            override_value=max_daily_trades,
            body_payload=body_payload,
            query_param_keys=query_param_keys,
        )
        mu_choppy_hard_block_enabled = resolve_optional_override(
            override_key="mu_choppy_hard_block_enabled",
            override_value=mu_choppy_hard_block_enabled,
            body_payload=body_payload,
            query_param_keys=query_param_keys,
            copy_null=True,
        )

    momentum_diversification_payload = parse_momentum_diversification_payload(
        momentum_diversification_json=momentum_diversification_json,
        body_payload=body_payload,
        query_param_keys=query_param_keys,
    )
    if momentum_diversification_payload is not None:
        config_payload["momentum_diversification"] = momentum_diversification_payload

    regime_filter_payload = parse_regime_filter_payload(
        regime_filter_json=regime_filter_json,
        body_payload=body_payload,
        query_param_keys=query_param_keys,
    )
    if regime_filter_payload is not None:
        config_payload["regime_filter"] = list(regime_filter_payload)

    return SessionConfigRequestResolution(
        config_payload=config_payload,
        momentum_diversification_payload=momentum_diversification_payload,
        max_daily_trades=max_daily_trades,
        mu_choppy_hard_block_enabled=mu_choppy_hard_block_enabled,
    )


def apply_session_config_resolution(
    *,
    manager: Any,
    session: Any,
    run_id: str,
    ticker: str,
    resolution: SessionConfigRequestResolution,
) -> Dict[str, Any]:
    """Apply resolved config to manager/session and build endpoint response."""
    config_payload = dict(resolution.config_payload)
    config_payload.update(normalize_runtime_exit_formula_fields(config_payload))
    canonical = TradingConfig.from_dict(config_payload)

    manager._apply_trading_config_to_session(
        session,
        canonical,
        normalize_momentum=(resolution.momentum_diversification_payload is not None),
    )
    manager.regime_refresh_bars = canonical.regime_refresh_bars
    manager.max_fill_participation_rate = canonical.max_fill_participation_rate
    manager.min_fill_ratio = canonical.min_fill_ratio

    manager.set_run_defaults(
        run_id=run_id,
        ticker=ticker,
        trading_config=canonical,
    )
    session.max_daily_trades_override = (
        int(resolution.max_daily_trades) if resolution.max_daily_trades is not None else None
    )
    session.mu_choppy_hard_block_enabled_override = (
        bool(resolution.mu_choppy_hard_block_enabled)
        if resolution.mu_choppy_hard_block_enabled is not None
        else None
    )

    return {
        "message": "Session configured",
        "session": session.to_dict(),
        "overrides": {
            "max_daily_trades": session.max_daily_trades_override,
            "mu_choppy_hard_block_enabled": session.mu_choppy_hard_block_enabled_override,
        },
    }


__all__ = [
    "SessionConfigRequestResolution",
    "apply_session_config_resolution",
    "resolve_session_config_request",
]
