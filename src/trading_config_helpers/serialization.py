"""Serialization and normalization helpers for TradingConfig."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .serialization_parts.from_dict import (
    build_trading_config_from_dict as _build_trading_config_from_dict_impl,
)
from .serialization_parts.to_session_params import (
    trading_config_to_session_params as _trading_config_to_session_params_impl,
)

if TYPE_CHECKING:
    from ..trading_config import TradingConfig


def build_trading_config_from_dict(cls: type["TradingConfig"], d: Dict[str, Any]) -> "TradingConfig":
    return _build_trading_config_from_dict_impl(cls=cls, d=d)


def trading_config_to_session_params(config: "TradingConfig") -> Dict[str, Any]:
    return _trading_config_to_session_params_impl(config=config)


__all__ = [
    "build_trading_config_from_dict",
    "trading_config_to_session_params",
]
