"""Intraday levels memory management service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .day_trading_models import TradingSession
from .intraday_levels import ensure_intraday_levels_state, build_intraday_levels_snapshot


class IntradayMemoryService:
    """Manages cross-session intraday memory storage and injection."""

    def __init__(self):
        # type: Dict[tuple[str, str], List[Dict[str, Any]]]
        self.day_memory = {}
        # type: Dict[tuple[str, str, str], bool]
        self.export_marker = {}

    @staticmethod
    def _intraday_memory_key(run_id: str, ticker: str) -> tuple[str, str]:
        return (str(run_id), str(ticker).upper())

    @staticmethod
    def _intraday_memory_age_days(current_date: str, source_date: str) -> Optional[int]:
        try:
            current = datetime.strptime(str(current_date), "%Y-%m-%d").date()
            source = datetime.strptime(str(source_date), "%Y-%m-%d").date()
            return int((current - source).days)
        except Exception:
            return None

    @staticmethod
    def _intraday_memory_weight(age_days: int, cfg: Dict[str, Any]) -> float:
        decay_after_days = max(1, int(cfg.get("memory_decay_after_days", 2)))
        decay_weight = min(1.0, max(0.1, float(cfg.get("memory_decay_weight", 0.5))))
        if age_days <= decay_after_days:
            return 1.0
        return decay_weight

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Any:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def persist_intraday_levels_memory(self, session: TradingSession) -> None:
        state = ensure_intraday_levels_state(getattr(session, "intraday_levels_state", {}))
        cfg = state.get("config", {}) if isinstance(state.get("config"), dict) else {}
        if not bool(cfg.get("memory_enabled", True)):
            return

        session_key = (str(session.run_id), str(session.ticker).upper(), str(session.date))
        if self.export_marker.get(session_key):
            return

        min_tests = max(1, int(cfg.get("memory_min_tests", 2)))
        max_levels = max(1, int(cfg.get("memory_max_levels", 12)))
        snapshot = build_intraday_levels_snapshot(state)
        levels = list(state.get("levels", [])) if isinstance(state.get("levels"), list) else []

        export_levels: List[Dict[str, Any]] = []
        for level in levels:
            if bool(level.get("broken", False)):
                continue
            tests = int(level.get("tests", 0))
            if tests < min_tests:
                continue
            price = self._safe_float(level.get("price"), 0.0)
            if not price or price <= 0.0:
                continue
            export_levels.append(
                {
                    "kind": str(level.get("kind", "")),
                    "price": float(price),
                    "tests": tests,
                    "swing_samples": int(level.get("swing_samples", 1)),
                    "source": str(level.get("source", "")),
                }
            )
        export_levels.sort(
            key=lambda lvl: (
                -int(lvl.get("tests", 0)),
                -int(lvl.get("swing_samples", 1)),
            )
        )
        export_levels = export_levels[:max_levels]

        volume_profile = snapshot.get("volume_profile", {}) if isinstance(snapshot.get("volume_profile"), dict) else {}
        opening_range = snapshot.get("opening_range", {}) if isinstance(snapshot.get("opening_range"), dict) else {}
        day_bars = list(session.bars) if isinstance(getattr(session, "bars", None), list) else []
        day_high = None
        day_low = None
        day_close = None
        if day_bars:
            highs = [self._safe_float(getattr(bar, "high", None), None) for bar in day_bars]
            lows = [self._safe_float(getattr(bar, "low", None), None) for bar in day_bars]
            closes = [self._safe_float(getattr(bar, "close", None), None) for bar in day_bars]
            highs = [float(v) for v in highs if v is not None]
            lows = [float(v) for v in lows if v is not None]
            closes = [float(v) for v in closes if v is not None]
            if highs:
                day_high = max(highs)
            if lows:
                day_low = min(lows)
            if closes:
                day_close = closes[-1]
        memory_entry = {
            "date": str(session.date),
            "levels": export_levels,
            "volume_profile": {
                "poc_price": self._safe_float(volume_profile.get("poc_price"), None),
                "value_area_low": self._safe_float(volume_profile.get("value_area_low"), None),
                "value_area_high": self._safe_float(volume_profile.get("value_area_high"), None),
            },
            "opening_range": {
                "high": self._safe_float(opening_range.get("high"), None),
                "low": self._safe_float(opening_range.get("low"), None),
                "mid": self._safe_float(opening_range.get("mid"), None),
                "complete": bool(opening_range.get("complete", False)),
            },
            "day_anchors": {
                "high": self._safe_float(day_high, None),
                "low": self._safe_float(day_low, None),
                "close": self._safe_float(day_close, None),
            },
        }

        key = self._intraday_memory_key(session.run_id, session.ticker)
        rows = list(self.day_memory.get(key, []))
        rows = [row for row in rows if str(row.get("date")) != str(session.date)]
        rows.append(memory_entry)
        rows.sort(key=lambda row: str(row.get("date", "")))

        max_age_days = max(1, int(cfg.get("memory_max_age_days", 5)))
        kept: List[Dict[str, Any]] = []
        for row in rows:
            age = self._intraday_memory_age_days(str(session.date), str(row.get("date", "")))
            if age is None or age < 0 or age > max_age_days:
                continue
            kept.append(row)
        self.day_memory[key] = kept
        self.export_marker[session_key] = True

    def inject_intraday_levels_memory_into_session(self, session: TradingSession) -> None:
        state = ensure_intraday_levels_state(getattr(session, "intraday_levels_state", {}))
        cfg = state.get("config", {}) if isinstance(state.get("config"), dict) else {}
        if not bool(cfg.get("memory_enabled", True)):
            session.intraday_levels_state = state
            return

        key = self._intraday_memory_key(session.run_id, session.ticker)
        rows = list(self.day_memory.get(key, []))
        if not rows:
            session.intraday_levels_state = state
            return

        max_age_days = max(1, int(cfg.get("memory_max_age_days", 5)))
        memory_max_levels = max(1, int(cfg.get("memory_max_levels", 12)))
        candidates: List[Dict[str, Any]] = []
        memory_profiles: List[Dict[str, Any]] = []
        seen = set()
        prior_day_anchor_candidates: List[Dict[str, Any]] = []

        for row in rows:
            source_date = str(row.get("date", ""))
            age_days = self._intraday_memory_age_days(str(session.date), source_date)
            if age_days is None or age_days <= 0 or age_days > max_age_days:
                continue
            weight = self._intraday_memory_weight(age_days, cfg)
            profile = row.get("volume_profile", {}) if isinstance(row.get("volume_profile"), dict) else {}
            day_anchors = row.get("day_anchors", {}) if isinstance(row.get("day_anchors"), dict) else {}
            poc_price = self._safe_float(profile.get("poc_price"), None)
            va_low = self._safe_float(profile.get("value_area_low"), None)
            va_high = self._safe_float(profile.get("value_area_high"), None)
            close_price = self._safe_float(day_anchors.get("close"), None)
            if poc_price is not None or close_price is not None:
                memory_profiles.append(
                    {
                        "date": source_date,
                        "age_days": int(age_days),
                        "weight": float(weight),
                        "poc_price": float(poc_price) if poc_price is not None else None,
                        "value_area_low": va_low,
                        "value_area_high": va_high,
                        "close_price": close_price,
                    }
                )
            for level in row.get("levels", []):
                if not isinstance(level, dict):
                    continue
                kind = str(level.get("kind", "")).strip().lower()
                price = self._safe_float(level.get("price"), None)
                if kind not in {"support", "resistance"} or price is None or price <= 0.0:
                    continue
                dedupe_key = (kind, round(float(price), 4), source_date)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                tests = max(1, int(level.get("tests", 1)))
                candidates.append(
                    {
                        "kind": kind,
                        "price": float(price),
                        "tests": tests,
                        "swing_samples": max(1, int(level.get("swing_samples", 1))),
                        "source": "memory_level",
                        "memory_level": True,
                        "memory_origin_date": source_date,
                        "memory_age_days": int(age_days),
                        "memory_weight": float(weight),
                    }
                )

            if bool(cfg.get("prior_day_anchors_enabled", True)) and age_days == 1:
                day_anchors = row.get("day_anchors", {}) if isinstance(row.get("day_anchors"), dict) else {}
                pd_high = self._safe_float(day_anchors.get("high"), None)
                pd_low = self._safe_float(day_anchors.get("low"), None)
                pd_close = self._safe_float(day_anchors.get("close"), None)

                anchor_rows: List[Dict[str, Any]] = []
                if pd_high is not None and pd_high > 0.0:
                    anchor_rows.append(
                        {
                            "kind": "resistance",
                            "price": float(pd_high),
                            "tests": 2,
                            "swing_samples": 2,
                            "source": "prior_day_high",
                            "memory_level": False,
                            "memory_origin_date": source_date,
                            "memory_age_days": int(age_days),
                            "memory_weight": 1.0,
                        }
                    )
                if pd_low is not None and pd_low > 0.0:
                    anchor_rows.append(
                        {
                            "kind": "support",
                            "price": float(pd_low),
                            "tests": 2,
                            "swing_samples": 2,
                            "source": "prior_day_low",
                            "memory_level": False,
                            "memory_origin_date": source_date,
                            "memory_age_days": int(age_days),
                            "memory_weight": 1.0,
                        }
                    )
                if pd_close is not None and pd_close > 0.0:
                    anchor_rows.extend(
                        [
                            {
                                "kind": "support",
                                "price": float(pd_close),
                                "tests": 2,
                                "swing_samples": 2,
                                "source": "prior_day_close",
                                "memory_level": False,
                                "memory_origin_date": source_date,
                                "memory_age_days": int(age_days),
                                "memory_weight": 1.0,
                            },
                            {
                                "kind": "resistance",
                                "price": float(pd_close),
                                "tests": 2,
                                "swing_samples": 2,
                                "source": "prior_day_close",
                                "memory_level": False,
                                "memory_origin_date": source_date,
                                "memory_age_days": int(age_days),
                                "memory_weight": 1.0,
                            },
                        ]
                    )

                for anchor in anchor_rows:
                    dedupe_anchor_key = (
                        str(anchor.get("kind", "")),
                        round(float(anchor.get("price", 0.0)), 4),
                        str(anchor.get("source", "")),
                        source_date,
                    )
                    if dedupe_anchor_key in seen:
                        continue
                    seen.add(dedupe_anchor_key)
                    prior_day_anchor_candidates.append(anchor)

        if prior_day_anchor_candidates:
            candidates.extend(prior_day_anchor_candidates)

        if not candidates and not memory_profiles:
            session.intraday_levels_state = state
            return

        candidates.sort(
            key=lambda item: (
                int(item.get("memory_age_days", 99)),
                -int(item.get("tests", 0)),
                -float(item.get("memory_weight", 0.0)),
            )
        )
        candidates = candidates[:memory_max_levels]

        next_level_id = int(state.get("next_level_id", 1))
        for level in candidates:
            state["levels"].append(
                {
                    "id": next_level_id,
                    "kind": str(level.get("kind")),
                    "price": round(float(level.get("price", 0.0)), 4),
                    "source": str(level.get("source", "memory_level")),
                    "created_bar_index": -1,
                    "created_timestamp": "",
                    "last_swing_bar_index": -1,
                    "last_swing_timestamp": "",
                    "swing_samples": int(level.get("swing_samples", 1)),
                    "tests": int(level.get("tests", 1)),
                    "last_test_bar_index": -1,
                    "last_event": "memory_seeded",
                    "last_event_bar_index": -1,
                    "broken": False,
                    "broken_bar_index": -1,
                    "memory_level": bool(level.get("memory_level", True)),
                    "memory_origin_date": str(level.get("memory_origin_date", "")),
                    "memory_age_days": int(level.get("memory_age_days", 0)),
                    "memory_weight": float(level.get("memory_weight", 1.0)),
                }
            )
            next_level_id += 1
        state["next_level_id"] = next_level_id
        state["memory_profiles"] = memory_profiles
        state["snapshot"] = build_intraday_levels_snapshot(state)
        session.intraday_levels_state = state
