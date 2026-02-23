from __future__ import annotations

from typing import Any

from .shared import to_float as _to_float


def cfg_float(session: Any, key: str, default: float) -> float:
    return _to_float(getattr(session, key, default), default)


def cfg_int(session: Any, key: str, default: int) -> int:
    try:
        return int(getattr(session, key, default))
    except (TypeError, ValueError):
        return int(default)


def cfg_bool(session: Any, key: str, default: bool) -> bool:
    value = getattr(session, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def side_direction(side: str) -> float:
    return 1.0 if str(side).lower() == "long" else -1.0


__all__ = ["cfg_bool", "cfg_float", "cfg_int", "side_direction"]
