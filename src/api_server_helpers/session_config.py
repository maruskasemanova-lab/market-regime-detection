"""Helpers for parsing session config requests in api_server."""

from __future__ import annotations

from dataclasses import fields
import json
from typing import Any, Dict, Mapping, Optional, Set, Tuple

from ..trading_config import TradingConfig


_TRADING_CONFIG_FIELD_NAMES = tuple(field.name for field in fields(TradingConfig))


def build_session_config_payload(local_values: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in _TRADING_CONFIG_FIELD_NAMES:
        if key in local_values:
            payload[key] = local_values[key]
    return payload


def merge_body_payload(
    *,
    config_payload: Dict[str, Any],
    body_payload: Mapping[str, Any],
    query_param_keys: Set[str],
) -> None:
    # Use TradingConfig field names as whitelist, but allow body-only fields
    # that may not yet exist as explicit query params on the endpoint signature.
    for key in _TRADING_CONFIG_FIELD_NAMES:
        if key in body_payload and key not in query_param_keys:
            config_payload[key] = body_payload[key]


def resolve_json_string_override(
    *,
    override_key: str,
    override_value: str,
    body_payload: Mapping[str, Any],
    query_param_keys: Set[str],
) -> str:
    if (
        override_key not in query_param_keys
        and not str(override_value or "").strip()
        and body_payload.get(override_key) is not None
    ):
        return str(body_payload.get(override_key))
    return override_value


def resolve_optional_override(
    *,
    override_key: str,
    override_value: Any,
    body_payload: Mapping[str, Any],
    query_param_keys: Set[str],
    copy_null: bool = False,
) -> Any:
    if override_value is not None:
        return override_value
    if override_key in query_param_keys:
        return override_value
    if override_key not in body_payload:
        return override_value

    value = body_payload.get(override_key)
    if value is None and not copy_null:
        return override_value
    return value


def parse_momentum_diversification_payload(
    *,
    momentum_diversification_json: str,
    body_payload: Mapping[str, Any],
    query_param_keys: Set[str],
) -> Optional[Dict[str, Any]]:
    if str(momentum_diversification_json or "").strip():
        try:
            parsed_payload = json.loads(momentum_diversification_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid momentum_diversification_json payload: {exc}"
            ) from exc
        if not isinstance(parsed_payload, dict):
            raise ValueError("momentum_diversification_json must decode to an object.")
        return parsed_payload

    if (
        body_payload
        and "momentum_diversification_json" not in query_param_keys
        and "momentum_diversification" in body_payload
    ):
        body_momentum = body_payload.get("momentum_diversification")
        if body_momentum is None:
            return {}
        if isinstance(body_momentum, dict):
            return dict(body_momentum)
        raise ValueError(
            "momentum_diversification must be an object when provided in request body."
        )

    return None


def parse_regime_filter_payload(
    *,
    regime_filter_json: str,
    body_payload: Mapping[str, Any],
    query_param_keys: Set[str],
) -> Optional[Tuple[str, ...]]:
    if str(regime_filter_json or "").strip():
        try:
            parsed_payload = json.loads(regime_filter_json)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed_payload, list):
            return tuple(
                str(regime).strip().upper()
                for regime in parsed_payload
                if str(regime).strip()
            )
        return None

    if (
        body_payload
        and "regime_filter_json" not in query_param_keys
        and "regime_filter" in body_payload
    ):
        body_regime_filter = body_payload.get("regime_filter")
        if isinstance(body_regime_filter, list):
            return tuple(
                str(regime).strip().upper()
                for regime in body_regime_filter
                if str(regime).strip()
            )
        if body_regime_filter is not None:
            raise ValueError("regime_filter must be an array when provided in request body.")

    return None


__all__ = [
    "build_session_config_payload",
    "merge_body_payload",
    "parse_momentum_diversification_payload",
    "parse_regime_filter_payload",
    "resolve_json_string_override",
    "resolve_optional_override",
]
