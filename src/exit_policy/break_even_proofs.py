from __future__ import annotations

from typing import Any, Dict, List, Optional

from .break_even_config import cfg_float, cfg_int, side_direction
from .shared import to_float as _to_float


def resolve_intraday_levels_snapshot(session: Any) -> Dict[str, Any]:
    state = getattr(session, "intraday_levels_state", {})
    if not isinstance(state, dict):
        return {}
    snapshot = state.get("snapshot")
    if isinstance(snapshot, dict):
        return snapshot
    return state


def _level_distance_pct(reference_price: float, level_price: float) -> Optional[float]:
    if reference_price <= 0.0 or level_price <= 0.0:
        return None
    return abs((level_price - reference_price) / reference_price) * 100.0


def _intraday_levels_list(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    levels = snapshot.get("levels") if isinstance(snapshot, dict) else None
    if not isinstance(levels, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in levels:
        if not isinstance(item, dict):
            continue
        level_price = _to_float(item.get("price"), 0.0)
        if level_price <= 0.0:
            continue
        result.append(item)
    return result


def _nearest_level_snapshot(
    *,
    side: str,
    close_price: float,
    levels_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    target_kind = "resistance" if str(side).lower() == "long" else "support"
    levels = _intraday_levels_list(levels_snapshot)
    target_levels = [
        row for row in levels if str(row.get("kind", "")).strip().lower() == target_kind
    ]

    nearest: Optional[Dict[str, Any]] = None
    nearest_distance: Optional[float] = None
    for row in target_levels:
        price = _to_float(row.get("price"), 0.0)
        if price <= 0.0:
            continue
        distance_pct = _level_distance_pct(close_price, price)
        if distance_pct is None:
            continue
        if nearest is None or distance_pct < float(nearest_distance):
            nearest = row
            nearest_distance = distance_pct

    if nearest is None and isinstance(levels_snapshot, dict):
        opening_range = (
            levels_snapshot.get("opening_range")
            if isinstance(levels_snapshot.get("opening_range"), dict)
            else {}
        )
        if str(side).lower() == "long":
            or_price = _to_float(opening_range.get("high"), 0.0)
            source = "opening_range_high"
        else:
            or_price = _to_float(opening_range.get("low"), 0.0)
            source = "opening_range_low"
        if or_price > 0.0:
            nearest = {
                "id": None,
                "kind": target_kind,
                "price": or_price,
                "tests": 1,
                "confluence_points": 2,
                "source": source,
            }
            nearest_distance = _level_distance_pct(close_price, or_price)

    return {
        "kind": target_kind,
        "level": nearest,
        "distance_pct": nearest_distance,
    }


def level_proof_snapshot(
    *,
    session: Any,
    side: str,
    close_price: float,
    levels_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    nearest = _nearest_level_snapshot(
        side=side,
        close_price=close_price,
        levels_snapshot=levels_snapshot,
    )
    level = nearest.get("level")
    level_price = _to_float(level.get("price"), 0.0) if isinstance(level, dict) else 0.0
    level_buf_pct = max(0.0, cfg_float(session, "break_even_level_buffer_pct", 0.02))
    max_distance_pct = max(
        level_buf_pct,
        cfg_float(session, "break_even_level_max_distance_pct", 0.60),
    )
    min_confluence = max(0, cfg_int(session, "break_even_level_min_confluence", 2))
    min_tests = max(0, cfg_int(session, "break_even_level_min_tests", 1))
    tests = int(_to_float(level.get("tests"), 0.0)) if isinstance(level, dict) else 0
    confluence = int(_to_float(level.get("confluence_points"), 0.0)) if isinstance(level, dict) else 0
    source = str(level.get("source", "")) if isinstance(level, dict) else ""
    distance_pct = nearest.get("distance_pct")
    close_confirmed = False
    if level_price > 0.0 and close_price > 0.0:
        if str(side).lower() == "long":
            close_confirmed = close_price > (level_price * (1.0 + (level_buf_pct / 100.0)))
        else:
            close_confirmed = close_price < (level_price * (1.0 - (level_buf_pct / 100.0)))
    quality_ok = (
        confluence >= min_confluence
        or tests >= min_tests
        or source.startswith("opening_range")
        or source.startswith("prior_day")
    )
    distance_ok = distance_pct is not None and float(distance_pct) <= max_distance_pct
    passed = bool(close_confirmed and quality_ok and distance_ok)

    no_go_threshold = max(0.0, cfg_float(session, "break_even_5m_no_go_proximity_pct", 0.10))
    near_strong_level = bool(
        distance_pct is not None
        and float(distance_pct) <= no_go_threshold
        and (confluence >= min_confluence or tests >= min_tests)
    )
    no_go_blocked = bool(near_strong_level and not close_confirmed)

    return {
        "enabled": True,
        "passed": passed,
        "no_go_blocked": no_go_blocked,
        "nearest_level_kind": nearest.get("kind"),
        "nearest_level_price": level_price if level_price > 0.0 else None,
        "nearest_level_tests": tests,
        "nearest_level_confluence": confluence,
        "nearest_level_source": source or None,
        "nearest_level_distance_pct": (
            round(float(distance_pct), 6) if distance_pct is not None else None
        ),
        "close_confirmed": bool(close_confirmed),
        "quality_ok": bool(quality_ok),
        "distance_ok": bool(distance_ok),
        "level_buffer_pct": round(level_buf_pct, 6),
        "max_distance_pct": round(max_distance_pct, 6),
    }


def l2_proof_snapshot(
    *,
    session: Any,
    side: str,
    bar: Any,
    spread_bps: Optional[float],
) -> Dict[str, Any]:
    direction = side_direction(side)
    delta = _to_float(getattr(bar, "l2_delta", None), 0.0)
    directional_signed = max(-1.0, min(1.0, delta / 10000.0)) * direction
    directional_imbalance = _to_float(getattr(bar, "l2_imbalance", None), 0.0) * direction
    directional_book_pressure = _to_float(getattr(bar, "l2_book_pressure", None), 0.0) * direction

    signed_thr = max(0.0, cfg_float(session, "break_even_l2_signed_aggression_min", 0.12))
    imbalance_thr = max(0.0, cfg_float(session, "break_even_l2_imbalance_min", 0.15))
    book_thr = max(0.0, cfg_float(session, "break_even_l2_book_pressure_min", 0.10))
    spread_max = max(0.0, cfg_float(session, "break_even_l2_spread_bps_max", 12.0))
    spread_ok = spread_bps is None or float(spread_bps) <= spread_max

    passes_signed = directional_signed >= signed_thr
    passes_imbalance = directional_imbalance >= imbalance_thr
    passes_book = directional_book_pressure >= book_thr
    passed = bool(passes_signed and passes_imbalance and passes_book and spread_ok)
    return {
        "enabled": True,
        "passed": passed,
        "directional_signed_aggression": round(directional_signed, 6),
        "directional_imbalance": round(directional_imbalance, 6),
        "directional_book_pressure": round(directional_book_pressure, 6),
        "signed_threshold": round(signed_thr, 6),
        "imbalance_threshold": round(imbalance_thr, 6),
        "book_pressure_threshold": round(book_thr, 6),
        "spread_bps": round(float(spread_bps), 6) if spread_bps is not None else None,
        "spread_bps_max": round(spread_max, 6),
        "spread_ok": bool(spread_ok),
        "passes_signed": bool(passes_signed),
        "passes_imbalance": bool(passes_imbalance),
        "passes_book_pressure": bool(passes_book),
    }


__all__ = [
    "l2_proof_snapshot",
    "level_proof_snapshot",
    "resolve_intraday_levels_snapshot",
]
