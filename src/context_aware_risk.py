"""Context-aware entry/exit risk adjustments based on intraday level context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_long(side: str) -> bool:
    return str(side or "").strip().lower() in {"long", "buy"}


def _extract_levels(levels_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    levels = levels_payload.get("levels")
    if not isinstance(levels, list):
        return []
    return [row for row in levels if isinstance(row, dict)]


def _extract_profile(levels_payload: Dict[str, Any]) -> Dict[str, Any]:
    profile = levels_payload.get("volume_profile")
    return dict(profile) if isinstance(profile, dict) else {}


def _extract_opening(levels_payload: Dict[str, Any]) -> Dict[str, Any]:
    opening = levels_payload.get("opening_range")
    return dict(opening) if isinstance(opening, dict) else {}


_STRATEGY_ROOM_DEFAULTS: Dict[str, Dict[str, float]] = {
    "momentum":       {"min_room_pct": 0.08, "min_effective_rr": 0.8},
    "pullback":       {"min_room_pct": 0.05, "min_effective_rr": 0.6},
    "mean_reversion": {"min_room_pct": 0.03, "min_effective_rr": 0.4},
    "rotation":       {"min_room_pct": 0.02, "min_effective_rr": 0.3},
    "vwap_magnet":    {"min_room_pct": 0.03, "min_effective_rr": 0.4},
    "volumeprofile":  {"min_room_pct": 0.03, "min_effective_rr": 0.4},
}


@dataclass(frozen=True)
class ContextRiskConfig:
    enabled: bool = False
    sl_buffer_pct: float = 0.03
    min_sl_pct: float = 0.30
    pullback_min_sl_pct: float = 0.50
    min_room_pct: float = 0.15
    min_effective_rr: float = 0.8
    trailing_tighten_zone: float = 0.2
    trailing_tighten_factor: float = 0.5
    level_trail_enabled: bool = True
    max_anchor_search_pct: float = 1.5
    min_level_tests_for_sl: int = 1
    sweep_atr_buffer_multiplier: float = 0.5

    def effective_min_room_pct(self, strategy_key: str = "") -> float:
        key = str(strategy_key or "").strip().lower()
        defaults = _STRATEGY_ROOM_DEFAULTS.get(key)
        if defaults and self.min_room_pct == 0.15:
            return defaults["min_room_pct"]
        return self.min_room_pct

    def effective_min_rr(self, strategy_key: str = "") -> float:
        key = str(strategy_key or "").strip().lower()
        defaults = _STRATEGY_ROOM_DEFAULTS.get(key)
        if defaults and self.min_effective_rr == 0.8:
            return defaults["min_effective_rr"]
        return self.min_effective_rr

    def effective_min_sl_pct(self, strategy_key: str = "") -> float:
        key = str(strategy_key or "").strip().lower()
        if key == "pullback":
            return max(self.min_sl_pct, self.pullback_min_sl_pct)
        return self.min_sl_pct

    @classmethod
    def from_config_obj(cls, cfg: Any) -> "ContextRiskConfig":
        return cls(
            enabled=bool(getattr(cfg, "context_aware_risk_enabled", False)),
            sl_buffer_pct=max(
                0.0,
                _to_float(getattr(cfg, "context_risk_sl_buffer_pct", 0.03), 0.03),
            ),
            min_sl_pct=max(
                0.0,
                _to_float(getattr(cfg, "context_risk_min_sl_pct", 0.30), 0.30),
            ),
            pullback_min_sl_pct=max(
                0.0,
                _to_float(getattr(cfg, "pullback_context_min_sl_pct", 0.50), 0.50),
            ),
            min_room_pct=max(
                0.0,
                _to_float(getattr(cfg, "context_risk_min_room_pct", 0.15), 0.15),
            ),
            min_effective_rr=max(
                0.0,
                _to_float(getattr(cfg, "context_risk_min_effective_rr", 0.8), 0.8),
            ),
            trailing_tighten_zone=min(
                1.0,
                max(
                    0.0,
                    _to_float(
                        getattr(cfg, "context_risk_trailing_tighten_zone", 0.2),
                        0.2,
                    ),
                ),
            ),
            trailing_tighten_factor=min(
                1.0,
                max(
                    0.0,
                    _to_float(
                        getattr(cfg, "context_risk_trailing_tighten_factor", 0.5),
                        0.5,
                    ),
                ),
            ),
            level_trail_enabled=bool(
                getattr(cfg, "context_risk_level_trail_enabled", True)
            ),
            max_anchor_search_pct=max(
                0.1,
                _to_float(
                    getattr(cfg, "context_risk_max_anchor_search_pct", 1.5),
                    1.5,
                ),
            ),
            min_level_tests_for_sl=max(
                0,
                int(_to_float(getattr(cfg, "context_risk_min_level_tests_for_sl", 1), 1)),
            ),
            sweep_atr_buffer_multiplier=max(
                0.0,
                _to_float(getattr(cfg, "sweep_atr_buffer_multiplier", 0.5), 0.5),
            ),
        )


def _level_price(level: Dict[str, Any]) -> Optional[float]:
    value = level.get("price")
    if value is None:
        return None
    parsed = _to_float(value, 0.0)
    return parsed if parsed > 0.0 else None


def _level_tests(level: Dict[str, Any]) -> int:
    return int(_to_float(level.get("tests"), 0.0))


def _level_kind(level: Dict[str, Any]) -> str:
    return str(level.get("kind", "")).strip().lower()


def _level_broken(level: Dict[str, Any]) -> bool:
    return bool(level.get("broken", False))


def _nearest_levels(
    *,
    entry_price: float,
    side: str,
    levels_payload: Dict[str, Any],
    config: ContextRiskConfig,
) -> Dict[str, Optional[Tuple[float, str]]]:
    levels = _extract_levels(levels_payload)
    is_long = _is_long(side)
    max_search_abs = abs(entry_price) * (config.max_anchor_search_pct / 100.0)
    nearest_support: Optional[Tuple[float, str]] = None
    nearest_resistance: Optional[Tuple[float, str]] = None

    for level in levels:
        price = _level_price(level)
        if price is None:
            continue
        if _level_broken(level):
            continue
        distance_abs = abs(entry_price - price)
        if distance_abs > max_search_abs:
            continue
        tests = _level_tests(level)
        kind = _level_kind(level)
        source = str(level.get("source", "")).strip().lower()
        if tests < config.min_level_tests_for_sl and not source.startswith("prior_day"):
            continue

        if kind == "support" and price < entry_price:
            if nearest_support is None or price > nearest_support[0]:
                nearest_support = (price, source or "support")
        elif kind == "resistance" and price > entry_price:
            if nearest_resistance is None or price < nearest_resistance[0]:
                nearest_resistance = (price, source or "resistance")

    profile = _extract_profile(levels_payload)
    val = _to_float(profile.get("value_area_low"), 0.0)
    vah = _to_float(profile.get("value_area_high"), 0.0)
    if val > 0.0 and val < entry_price:
        if nearest_support is None or val > nearest_support[0]:
            nearest_support = (val, "value_area_low")
    if vah > 0.0 and vah > entry_price:
        if nearest_resistance is None or vah < nearest_resistance[0]:
            nearest_resistance = (vah, "value_area_high")

    opening = _extract_opening(levels_payload)
    opening_low = _to_float(opening.get("low"), 0.0)
    opening_high = _to_float(opening.get("high"), 0.0)
    if opening_low > 0.0 and opening_low < entry_price:
        if nearest_support is None or opening_low > nearest_support[0]:
            nearest_support = (opening_low, "opening_range_low")
    if opening_high > 0.0 and opening_high > entry_price:
        if nearest_resistance is None or opening_high < nearest_resistance[0]:
            nearest_resistance = (opening_high, "opening_range_high")

    if not is_long:
        # Preserve same fields for short callers too.
        return {
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
        }
    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
    }


def adjust_entry_risk(
    *,
    entry_price: float,
    side: str,
    original_stop_loss: float,
    original_take_profit: float,
    levels_payload: Dict[str, Any],
    config: ContextRiskConfig,
    atr: Optional[float] = None,
    is_sweep_trade: bool = False,
    strategy_key: str = "",
) -> Dict[str, Any]:
    """
    Return risk adjustment decision for entry execution.

    Output:
    - adjusted_stop_loss / adjusted_take_profit
    - sl_reason / tp_reason
    - skip / skip_reason
    """
    if entry_price <= 0.0:
        return {
            "adjusted_stop_loss": float(original_stop_loss or 0.0),
            "adjusted_take_profit": float(original_take_profit or 0.0),
            "sl_reason": "invalid_entry",
            "tp_reason": "invalid_entry",
            "skip": True,
            "skip_reason": "invalid_entry_price",
        }

    is_long = _is_long(side)
    adjusted_sl = float(original_stop_loss or 0.0)
    adjusted_tp = float(original_take_profit or 0.0)
    sl_reason = "fallback_original"
    tp_reason = "fallback_original"

    nearest = _nearest_levels(
        entry_price=entry_price,
        side=side,
        levels_payload=levels_payload,
        config=config,
    )
    nearest_support = nearest.get("nearest_support")
    nearest_resistance = nearest.get("nearest_resistance")
    buffer_abs = entry_price * (config.sl_buffer_pct / 100.0)
    atr_buffer_abs = 0.0
    if bool(is_sweep_trade):
        atr_abs = max(0.0, _to_float(atr, 0.0))
        atr_buffer_abs = atr_abs * max(0.0, float(config.sweep_atr_buffer_multiplier))
    total_buffer_abs = buffer_abs + atr_buffer_abs

    # Stop-loss anchor
    if is_long and nearest_support is not None:
        anchored = max(0.0, nearest_support[0] - total_buffer_abs)
        if (
            adjusted_sl <= 0.0
            or anchored > adjusted_sl
            or (bool(is_sweep_trade) and atr_buffer_abs > 0.0 and anchored < adjusted_sl)
        ):
            adjusted_sl = anchored
            sl_reason = f"anchored_support:{nearest_support[1]}"
            if atr_buffer_abs > 0.0:
                sl_reason = f"{sl_reason}|sweep_atr_buffer:{atr_buffer_abs:.4f}"
    elif not is_long and nearest_resistance is not None:
        anchored = nearest_resistance[0] + total_buffer_abs
        if (
            adjusted_sl <= 0.0
            or anchored < adjusted_sl
            or (bool(is_sweep_trade) and atr_buffer_abs > 0.0 and anchored > adjusted_sl)
        ):
            adjusted_sl = anchored
            sl_reason = f"anchored_resistance:{nearest_resistance[1]}"
            if atr_buffer_abs > 0.0:
                sl_reason = f"{sl_reason}|sweep_atr_buffer:{atr_buffer_abs:.4f}"

    # Never allow a context-adjusted SL that is closer than configured minimum room.
    effective_min_sl_pct = config.effective_min_sl_pct(strategy_key)
    min_sl_abs = entry_price * (effective_min_sl_pct / 100.0)
    if min_sl_abs > 0.0:
        if is_long:
            min_stop = max(0.0, entry_price - min_sl_abs)
            if adjusted_sl <= 0.0 or adjusted_sl > min_stop:
                adjusted_sl = min_stop
                sl_reason = (
                    f"{sl_reason}|min_sl_floor:{effective_min_sl_pct:.4f}"
                    if sl_reason
                    else f"min_sl_floor:{effective_min_sl_pct:.4f}"
                )
        else:
            min_stop = entry_price + min_sl_abs
            if adjusted_sl <= 0.0 or adjusted_sl < min_stop:
                adjusted_sl = min_stop
                sl_reason = (
                    f"{sl_reason}|min_sl_floor:{effective_min_sl_pct:.4f}"
                    if sl_reason
                    else f"min_sl_floor:{effective_min_sl_pct:.4f}"
                )

    # Take-profit anchor
    profile = _extract_profile(levels_payload)
    poc_price = _to_float(profile.get("effective_poc_price"), 0.0) or _to_float(
        profile.get("poc_price"), 0.0
    )
    target_candidates: List[Tuple[float, str]] = []
    if is_long:
        if nearest_resistance is not None:
            target_candidates.append(
                (nearest_resistance[0], f"nearest_resistance:{nearest_resistance[1]}")
            )
        if poc_price > entry_price:
            target_candidates.append((poc_price, "poc"))
    else:
        if nearest_support is not None:
            target_candidates.append(
                (nearest_support[0], f"nearest_support:{nearest_support[1]}")
            )
        if poc_price > 0.0 and poc_price < entry_price:
            target_candidates.append((poc_price, "poc"))

    if target_candidates:
        if is_long:
            target_price, target_source = min(target_candidates, key=lambda row: row[0])
            if adjusted_tp <= 0.0 or target_price < adjusted_tp:
                adjusted_tp = target_price
                tp_reason = f"anchored_target:{target_source}"
        else:
            target_price, target_source = max(target_candidates, key=lambda row: row[0])
            if adjusted_tp <= 0.0 or target_price > adjusted_tp:
                adjusted_tp = target_price
                tp_reason = f"anchored_target:{target_source}"

    # Room / RR checks
    if is_long:
        risk_abs = max(0.0, entry_price - adjusted_sl) if adjusted_sl > 0.0 else 0.0
        room_abs = max(0.0, adjusted_tp - entry_price) if adjusted_tp > 0.0 else 0.0
    else:
        risk_abs = max(0.0, adjusted_sl - entry_price) if adjusted_sl > 0.0 else 0.0
        room_abs = max(0.0, entry_price - adjusted_tp) if adjusted_tp > 0.0 else 0.0

    risk_pct = (risk_abs / entry_price) * 100.0 if entry_price > 0 else 0.0
    room_pct = (room_abs / entry_price) * 100.0 if entry_price > 0 else 0.0
    eff_min_room = config.effective_min_room_pct(strategy_key)
    eff_min_rr = config.effective_min_rr(strategy_key)
    _common_metrics = {
        "adjusted_stop_loss": adjusted_sl,
        "adjusted_take_profit": adjusted_tp,
        "sl_reason": sl_reason,
        "tp_reason": tp_reason,
        "risk_pct": round(risk_pct, 6),
        "room_pct": round(room_pct, 6),
        "configured_min_room_pct": round(eff_min_room, 6),
        "configured_min_sl_pct": round(effective_min_sl_pct, 6),
        "configured_min_effective_rr": round(eff_min_rr, 6),
        "strategy_key": strategy_key or None,
    }
    if room_pct < eff_min_room:
        return {
            **_common_metrics,
            "skip": True,
            "skip_reason": (
                f"context_room_too_small:{room_pct:.4f}<{eff_min_room:.4f}"
            ),
        }
    if risk_pct <= 0.0:
        return {
            **_common_metrics,
            "skip": True,
            "skip_reason": "context_invalid_risk_distance",
        }

    effective_rr = room_pct / risk_pct
    if effective_rr < eff_min_rr:
        return {
            **_common_metrics,
            "effective_rr": round(effective_rr, 6),
            "skip": True,
            "skip_reason": (
                f"context_effective_rr_low:{effective_rr:.4f}<{eff_min_rr:.4f}"
            ),
        }

    # Optional wall check: opposing level between entry and TP.
    # Only strong walls (tested >= 2 times) block — single-test levels
    # are too weak to justify skipping an otherwise valid entry.
    levels = _extract_levels(levels_payload)
    opposing_kind = "resistance" if is_long else "support"
    min_wall_tests = max(2, config.min_level_tests_for_sl)
    for level in levels:
        if _level_broken(level):
            continue
        if _level_kind(level) != opposing_kind:
            continue
        if _level_tests(level) < min_wall_tests:
            continue
        level_px = _level_price(level)
        if level_px is None:
            continue
        if is_long and entry_price < level_px < adjusted_tp:
            return {
                **_common_metrics,
                "effective_rr": round(effective_rr, 6),
                "skip": True,
                "skip_reason": f"context_opposing_wall:{level_px:.4f}",
                "opposing_wall_price": round(level_px, 6),
                "opposing_wall_tests": _level_tests(level),
            }
        if (not is_long) and adjusted_tp < level_px < entry_price:
            return {
                **_common_metrics,
                "effective_rr": round(effective_rr, 6),
                "skip": True,
                "skip_reason": f"context_opposing_wall:{level_px:.4f}",
                "opposing_wall_price": round(level_px, 6),
                "opposing_wall_tests": _level_tests(level),
            }

    return {
        "adjusted_stop_loss": adjusted_sl,
        "adjusted_take_profit": adjusted_tp,
        "sl_reason": sl_reason,
        "tp_reason": tp_reason,
        "skip": False,
        "skip_reason": "ok",
        "effective_rr": round(effective_rr, 6),
        "risk_pct": round(risk_pct, 6),
        "room_pct": round(room_pct, 6),
        "configured_min_sl_pct": round(effective_min_sl_pct, 6),
    }


def apply_context_trailing(
    *,
    position: Any,
    current_price: float,
    levels_payload: Dict[str, Any],
    poc_migration_bias: str,
    config: ContextRiskConfig,
) -> Dict[str, Any]:
    """
    Apply trailing-tighten and optional level trail.

    Returns metrics about applied adjustments.
    """
    metrics: Dict[str, Any] = {
        "applied": False,
        "tightened": False,
        "level_trailed": False,
    }
    if not config.enabled or current_price <= 0.0 or position is None:
        return metrics

    is_long = _is_long(getattr(position, "side", ""))
    entry_price = _to_float(getattr(position, "entry_price", 0.0), 0.0)
    take_profit = _to_float(getattr(position, "take_profit", 0.0), 0.0)
    trailing_stop_price = _to_float(getattr(position, "trailing_stop_price", 0.0), 0.0)

    if entry_price <= 0.0:
        return metrics

    tighten_factor = 1.0
    if take_profit > 0.0:
        total_distance = (
            (take_profit - entry_price)
            if is_long
            else (entry_price - take_profit)
        )
        covered = (
            (current_price - entry_price)
            if is_long
            else (entry_price - current_price)
        )
        if total_distance > 0.0:
            progress = covered / total_distance
            if progress >= (1.0 - config.trailing_tighten_zone):
                tighten_factor = min(tighten_factor, config.trailing_tighten_factor)
                metrics["tighten_reason"] = "target_approach"

    bias = str(poc_migration_bias or "").strip().lower()
    bias_opposed = (is_long and bias == "downtrend") or ((not is_long) and bias == "uptrend")
    if bias_opposed:
        tighten_factor = min(tighten_factor, max(0.1, config.trailing_tighten_factor * 0.6))
        metrics["tighten_reason"] = "poc_migration_opposed"

    if trailing_stop_price > 0.0 and tighten_factor < 1.0:
        if is_long:
            distance = max(0.0, current_price - trailing_stop_price)
            tightened = trailing_stop_price + (distance * (1.0 - tighten_factor))
            if tightened > trailing_stop_price:
                position.trailing_stop_price = float(tightened)
                metrics["tightened"] = True
        else:
            distance = max(0.0, trailing_stop_price - current_price)
            tightened = trailing_stop_price - (distance * (1.0 - tighten_factor))
            if tightened < trailing_stop_price:
                position.trailing_stop_price = float(tightened)
                metrics["tightened"] = True

    if config.level_trail_enabled:
        levels = _extract_levels(levels_payload)
        buffer_abs = entry_price * (config.sl_buffer_pct / 100.0)
        current_sl = _to_float(getattr(position, "stop_loss", 0.0), 0.0)
        best_anchor: Optional[float] = None
        for level in levels:
            if _level_broken(level):
                continue
            if _level_tests(level) < config.min_level_tests_for_sl:
                continue
            level_px = _level_price(level)
            if level_px is None:
                continue
            kind = _level_kind(level)
            if is_long:
                if kind != "support" or level_px >= current_price:
                    continue
                anchor = level_px - buffer_abs
                if anchor <= current_sl:
                    continue
                if best_anchor is None or anchor > best_anchor:
                    best_anchor = anchor
            else:
                if kind != "resistance" or level_px <= current_price:
                    continue
                anchor = level_px + buffer_abs
                if current_sl > 0.0 and anchor >= current_sl:
                    continue
                if best_anchor is None or anchor < best_anchor:
                    best_anchor = anchor
        if best_anchor is not None:
            position.stop_loss = float(best_anchor)
            metrics["level_trailed"] = True
            metrics["level_trail_stop_loss"] = float(best_anchor)

    metrics["applied"] = bool(metrics["tightened"] or metrics["level_trailed"])
    return metrics
