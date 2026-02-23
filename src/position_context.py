"""
Position Context Monitor - Tracks market conditions at entry and detects changes.

When a position opens, an EntrySnapshot captures the full market context.
On each subsequent bar, the PositionContextMonitor compares current state
against the snapshot and generates structured ContextChangeEvents.

These events are consumed by the ExitPolicyEngine (Phase 3) to implement
regime-aware, flow-aware exit behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Configuration defaults ────────────────────────────────────────────

DEFAULT_CONTEXT_CONFIG = {
    # Flow reversal: signed_aggression flips vs entry direction
    "flow_reversal_threshold": 0.10,          # minimum absolute delta to trigger
    # Momentum stall: flow_score drops significantly from entry
    "momentum_stall_drop_pct": 25.0,          # % drop from entry flow_score
    # Volatility spike: ATR expands beyond entry level
    "volatility_spike_atr_multiplier": 2.0,   # current_atr / entry_atr
    # Grace period: ignore events in first N bars after entry
    "context_grace_bars": 2,
}


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class EntrySnapshot:
    """Market context captured at position entry time."""
    macro_regime: str              # "TRENDING" | "CHOPPY" | "MIXED"
    micro_regime: str              # "TRENDING_UP" | "BREAKOUT" | etc.
    strategy_name: str
    flow_score: float              # 0-100 composite flow quality
    signed_aggression: float       # directional flow intensity at entry
    book_pressure: float           # book_pressure_avg at entry
    directional_consistency: float # flow-price alignment at entry
    delta_acceleration: float      # delta momentum at entry
    entry_bar_index: int
    entry_price: float
    side: str                      # "long" | "short"
    atr_at_entry: float            # for relative volatility comparison
    volatility_at_entry: float     # realized volatility at entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_regime": self.macro_regime,
            "micro_regime": self.micro_regime,
            "strategy_name": self.strategy_name,
            "flow_score": round(self.flow_score, 2),
            "signed_aggression": round(self.signed_aggression, 4),
            "book_pressure": round(self.book_pressure, 4),
            "directional_consistency": round(self.directional_consistency, 4),
            "delta_acceleration": round(self.delta_acceleration, 2),
            "entry_bar_index": self.entry_bar_index,
            "entry_price": round(self.entry_price, 4),
            "side": self.side,
            "atr_at_entry": round(self.atr_at_entry, 4),
            "volatility_at_entry": round(self.volatility_at_entry, 4),
        }


@dataclass
class ContextChangeEvent:
    """A structured event when market context shifts during a position."""
    event_type: str       # "regime_flip" | "flow_reversal" | "momentum_stall" | "volatility_spike"
    severity: str         # "warning" | "critical"
    bar_index: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "bar_index": self.bar_index,
            "details": self.details,
        }


# ── Severity classification helpers ──────────────────────────────────

# Strategy categories for determining severity of regime flips
_MOMENTUM_STRATEGIES = {"momentum", "momentum_flow", "pullback", "gap_liquidity"}
_MEAN_REVERSION_STRATEGIES = {"mean_reversion", "absorption_reversal", "exhaustion_fade"}


def _classify_regime_flip_severity(
    from_regime: str,
    to_regime: str,
    strategy_name: str,
    side: str,
) -> str:
    """Determine severity of a regime flip based on strategy type.

    - Momentum strategies in TRENDING that flip to CHOPPY → critical
    - Mean reversion strategies in CHOPPY that flip to TRENDING → critical
    - Other flips → warning
    """
    strategy_key = (strategy_name or "").lower().strip().replace(" ", "_").replace("-", "_")

    # Momentum in trending → choppy = worst case (trend is gone)
    if strategy_key in _MOMENTUM_STRATEGIES:
        if from_regime == "TRENDING" and to_regime in ("CHOPPY", "MIXED"):
            return "critical"
    # Mean reversion in choppy → trending = worst case (breakout invalidates range)
    if strategy_key in _MEAN_REVERSION_STRATEGIES:
        if from_regime == "CHOPPY" and to_regime in ("TRENDING",):
            return "critical"

    return "warning"


def _classify_flow_reversal_severity(
    entry_signed: float,
    current_signed: float,
    side: str,
    threshold: float,
) -> str:
    """Determine severity of a flow reversal.

    Critical if flow has clearly reversed against the position.
    """
    direction = 1.0 if side == "long" else -1.0
    entry_directional = entry_signed * direction
    current_directional = current_signed * direction
    delta = entry_directional - current_directional

    # Flow was favorable and is now adverse → critical
    if entry_directional > 0 and current_directional < -threshold:
        return "critical"
    # Large swing → critical
    if delta > threshold * 3:
        return "critical"

    return "warning"


# ── Position Context Monitor ─────────────────────────────────────────

class PositionContextMonitor:
    """Monitors market context changes relative to position entry conditions.

    Usage:
        snapshot = EntrySnapshot(...)
        monitor = PositionContextMonitor(snapshot, config)
        # On each bar while position is open:
        events = monitor.update(current_regime, current_micro, flow_metrics, bar_index, atr)
    """

    def __init__(
        self,
        snapshot: EntrySnapshot,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.snapshot = snapshot
        self.config = {**DEFAULT_CONTEXT_CONFIG, **(config or {})}
        self.events: List[ContextChangeEvent] = []
        self._bars_since_entry = 0
        # Track which event types have already fired to avoid spam
        self._fired_regime_flip = False
        self._fired_flow_reversal = False
        self._fired_momentum_stall = False
        self._fired_volatility_spike = False

    def update(
        self,
        current_regime: str,
        current_micro: str,
        current_flow: Dict[str, Any],
        current_bar_index: int,
        current_atr: float = 0.0,
    ) -> List[ContextChangeEvent]:
        """Compare current state to entry snapshot, return new events.

        Args:
            current_regime: Current macro regime string
            current_micro: Current micro regime string
            current_flow: Current flow metrics dict (from _calculate_order_flow_metrics)
            current_bar_index: Current bar index
            current_atr: Current ATR value

        Returns:
            List of new ContextChangeEvents generated this bar (empty if none)
        """
        new_events: List[ContextChangeEvent] = []
        self._bars_since_entry += 1

        grace_bars = max(1, int(self.config.get("context_grace_bars", 2)))
        if self._bars_since_entry <= grace_bars:
            return new_events

        # ── 1. Regime flip detection ──
        if not self._fired_regime_flip and current_regime != self.snapshot.macro_regime:
            severity = _classify_regime_flip_severity(
                from_regime=self.snapshot.macro_regime,
                to_regime=current_regime,
                strategy_name=self.snapshot.strategy_name,
                side=self.snapshot.side,
            )
            event = ContextChangeEvent(
                event_type="regime_flip",
                severity=severity,
                bar_index=current_bar_index,
                details={
                    "from_regime": self.snapshot.macro_regime,
                    "to_regime": current_regime,
                    "from_micro": self.snapshot.micro_regime,
                    "to_micro": current_micro,
                    "bars_since_entry": self._bars_since_entry,
                },
            )
            new_events.append(event)
            self._fired_regime_flip = True

        # ── 2. Flow reversal detection ──
        if not self._fired_flow_reversal:
            threshold = float(self.config.get("flow_reversal_threshold", 0.10))
            current_signed = float(current_flow.get("signed_aggression", 0.0) or 0.0)
            direction = 1.0 if self.snapshot.side == "long" else -1.0
            entry_directional = self.snapshot.signed_aggression * direction
            current_directional = current_signed * direction

            # Flow was favorable (positive directional) and has now flipped
            if entry_directional > 0 and current_directional <= -threshold:
                severity = _classify_flow_reversal_severity(
                    entry_signed=self.snapshot.signed_aggression,
                    current_signed=current_signed,
                    side=self.snapshot.side,
                    threshold=threshold,
                )
                event = ContextChangeEvent(
                    event_type="flow_reversal",
                    severity=severity,
                    bar_index=current_bar_index,
                    details={
                        "entry_signed_aggression": self.snapshot.signed_aggression,
                        "current_signed_aggression": current_signed,
                        "entry_directional": round(entry_directional, 4),
                        "current_directional": round(current_directional, 4),
                        "threshold": threshold,
                        "bars_since_entry": self._bars_since_entry,
                    },
                )
                new_events.append(event)
                self._fired_flow_reversal = True

        # ── 3. Momentum stall detection ──
        if not self._fired_momentum_stall and self.snapshot.flow_score > 0:
            drop_pct = float(self.config.get("momentum_stall_drop_pct", 25.0))
            current_flow_score = float(current_flow.get("flow_score", 0.0) or 0.0)
            score_drop = self.snapshot.flow_score - current_flow_score
            drop_ratio = (score_drop / self.snapshot.flow_score) * 100.0 if self.snapshot.flow_score > 0 else 0.0

            if drop_ratio >= drop_pct:
                event = ContextChangeEvent(
                    event_type="momentum_stall",
                    severity="warning",
                    bar_index=current_bar_index,
                    details={
                        "entry_flow_score": round(self.snapshot.flow_score, 2),
                        "current_flow_score": round(current_flow_score, 2),
                        "drop_pct": round(drop_ratio, 1),
                        "threshold_pct": drop_pct,
                        "bars_since_entry": self._bars_since_entry,
                    },
                )
                new_events.append(event)
                self._fired_momentum_stall = True

        # ── 4. Volatility spike detection ──
        if not self._fired_volatility_spike and self.snapshot.atr_at_entry > 0 and current_atr > 0:
            atr_mult = float(self.config.get("volatility_spike_atr_multiplier", 2.0))
            atr_ratio = current_atr / self.snapshot.atr_at_entry
            if atr_ratio >= atr_mult:
                event = ContextChangeEvent(
                    event_type="volatility_spike",
                    severity="warning" if atr_ratio < atr_mult * 1.5 else "critical",
                    bar_index=current_bar_index,
                    details={
                        "entry_atr": round(self.snapshot.atr_at_entry, 4),
                        "current_atr": round(current_atr, 4),
                        "atr_ratio": round(atr_ratio, 2),
                        "threshold_multiplier": atr_mult,
                        "bars_since_entry": self._bars_since_entry,
                    },
                )
                new_events.append(event)
                self._fired_volatility_spike = True

        self.events.extend(new_events)
        return new_events

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all events for trade record enrichment."""
        return {
            "entry_snapshot": self.snapshot.to_dict(),
            "total_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
            "bars_monitored": self._bars_since_entry,
            "had_regime_flip": self._fired_regime_flip,
            "had_flow_reversal": self._fired_flow_reversal,
            "had_momentum_stall": self._fired_momentum_stall,
            "had_volatility_spike": self._fired_volatility_spike,
        }
