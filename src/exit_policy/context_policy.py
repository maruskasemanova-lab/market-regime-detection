from __future__ import annotations

from typing import Any, Dict, List, Optional

from .break_even import force_move_to_break_even
from .types import ExitDecision


DEFAULT_CONTEXT_EXIT_CONFIG = {
    # Regime flip responses
    "regime_flip_tighten_stop_pct": 0.30,          # tighten SL by 30% of current distance
    "regime_flip_shorten_time_pct": 0.50,           # reduce remaining time exit by 50%
    "regime_flip_exit_when_losing": True,            # exit immediately if losing beyond threshold
    "regime_flip_exit_loss_threshold_pct": 0.3,     # loss % at which regime flip triggers exit

    # Flow reversal responses
    "flow_reversal_move_to_breakeven": True,         # move stop to breakeven when profitable
    "flow_reversal_exit_when_losing": True,           # exit immediately if underwater

    # Momentum stall responses
    "momentum_stall_time_multiplier": 0.7,           # reduce remaining time exit by 30%

    # Volatility spike responses
    "volatility_spike_tighten_pct": 0.20,            # tighten SL by 20%

    # Grace period before responding to context events
    "context_response_grace_bars": 1,                 # bars after event before acting
}


# ── Context-Aware Exit Policy ──────────────────────────────────────

class ContextAwareExitPolicy:
    """Applies exit adjustments in response to PositionContextMonitor events.

    This is where the "professional trader" behavior emerges: when the market
    context shifts during a position, the exit parameters adapt accordingly.

    All parameters have neutral defaults that produce no behavior change
    until explicitly tuned via the adaptive tuner.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONTEXT_EXIT_CONFIG, **(config or {})}
        # Track which responses have already been applied (per position)
        self._applied_regime_flip = False
        self._applied_flow_reversal = False
        self._applied_momentum_stall = False
        self._applied_volatility_spike = False

    def reset(self) -> None:
        """Reset applied flags for a new position."""
        self._applied_regime_flip = False
        self._applied_flow_reversal = False
        self._applied_momentum_stall = False
        self._applied_volatility_spike = False

    def evaluate(
        self,
        context_events: List[Any],
        session: Any,
        pos: Any,
        current_bar_index: int,
    ) -> Optional[ExitDecision]:
        """Process context events and apply exit adjustments.

        Returns an ExitDecision if an immediate exit is warranted,
        otherwise returns None after applying stop/time adjustments in-place.
        """
        if not context_events:
            return None

        for event in context_events:
            event_type = event.event_type if hasattr(event, "event_type") else event.get("event_type", "")
            severity = event.severity if hasattr(event, "severity") else event.get("severity", "warning")

            if event_type == "regime_flip" and not self._applied_regime_flip:
                result = self._handle_regime_flip(session, pos, current_bar_index, severity)
                self._applied_regime_flip = True
                if result:
                    return result

            elif event_type == "flow_reversal" and not self._applied_flow_reversal:
                result = self._handle_flow_reversal(session, pos, current_bar_index, severity)
                self._applied_flow_reversal = True
                if result:
                    return result

            elif event_type == "momentum_stall" and not self._applied_momentum_stall:
                self._handle_momentum_stall(session, pos)
                self._applied_momentum_stall = True

            elif event_type == "volatility_spike" and not self._applied_volatility_spike:
                self._handle_volatility_spike(session, pos)
                self._applied_volatility_spike = True

        return None

    def _handle_regime_flip(
        self, session: Any, pos: Any, current_bar_index: int, severity: str,
    ) -> Optional[ExitDecision]:
        """React to a macro regime change during the position.

        1. Tighten stop by configured % of current SL distance.
        2. Shorten remaining time exit.
        3. If losing beyond threshold, exit immediately.
        """
        # Calculate unrealized P&L
        current_price = session.bars[-1].close if session.bars else pos.entry_price
        if pos.side == "long":
            unrealized_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        else:
            unrealized_pct = (pos.entry_price - current_price) / pos.entry_price * 100

        # Check for immediate exit on losing position
        loss_threshold = float(self.config.get("regime_flip_exit_loss_threshold_pct", 0.3))
        if self.config.get("regime_flip_exit_when_losing", True) and severity == "critical":
            if unrealized_pct < -loss_threshold:
                return ExitDecision(
                    should_exit=True,
                    reason="context_regime_flip_exit",
                    exit_price=current_price,
                    metrics={
                        "trigger": "regime_flip",
                        "severity": severity,
                        "unrealized_pct": round(unrealized_pct, 4),
                        "loss_threshold_pct": loss_threshold,
                    },
                )

        # Tighten stop loss
        tighten_pct = float(self.config.get("regime_flip_tighten_stop_pct", 0.30))
        if tighten_pct > 0 and pos.stop_loss and pos.stop_loss > 0:
            self._tighten_stop(pos, tighten_pct)

        # Shorten time exit
        shorten_pct = float(self.config.get("regime_flip_shorten_time_pct", 0.50))
        if shorten_pct > 0:
            self._shorten_time_exit(session, shorten_pct)

        return None

    def _handle_flow_reversal(
        self, session: Any, pos: Any, current_bar_index: int, severity: str,
    ) -> Optional[ExitDecision]:
        """React to order flow reversing against the position.

        - If profitable: move stop to breakeven + buffer.
        - If underwater: exit immediately.
        """
        current_price = session.bars[-1].close if session.bars else pos.entry_price
        if pos.side == "long":
            unrealized = current_price - pos.entry_price
        else:
            unrealized = pos.entry_price - current_price

        is_profitable = unrealized > 0

        if is_profitable and self.config.get("flow_reversal_move_to_breakeven", True):
            current_bar = session.bars[-1] if getattr(session, "bars", None) else None
            force_move_to_break_even(
                session=session,
                pos=pos,
                bar=current_bar,
                reason="context_flow_reversal",
            )
            return None

        if not is_profitable and self.config.get("flow_reversal_exit_when_losing", True):
            return ExitDecision(
                should_exit=True,
                reason="context_flow_reversal_exit",
                exit_price=current_price,
                metrics={
                    "trigger": "flow_reversal",
                    "severity": severity,
                    "unrealized": round(unrealized, 4),
                    "is_profitable": False,
                },
            )

        return None

    def _handle_momentum_stall(self, session: Any, pos: Any) -> None:
        """React to flow score dropping significantly from entry.

        Reduce remaining time exit window.
        """
        time_mult = float(self.config.get("momentum_stall_time_multiplier", 0.7))
        if time_mult < 1.0:
            self._shorten_time_exit(session, 1.0 - time_mult)

    def _handle_volatility_spike(self, session: Any, pos: Any) -> None:
        """React to ATR expanding beyond entry ATR.

        Tighten stop to reduce exposure to whipsaws.
        """
        tighten_pct = float(self.config.get("volatility_spike_tighten_pct", 0.20))
        if tighten_pct > 0 and pos.stop_loss and pos.stop_loss > 0:
            self._tighten_stop(pos, tighten_pct)

    def _tighten_stop(self, pos: Any, tighten_pct: float) -> None:
        """Move stop closer to current price by tighten_pct of current distance."""
        current_price = pos.highest_price if pos.side == "long" else pos.lowest_price
        if current_price <= 0:
            current_price = pos.entry_price

        if pos.side == "long":
            distance = current_price - pos.stop_loss
            if distance > 0:
                new_stop = pos.stop_loss + (distance * tighten_pct)
                pos.stop_loss = new_stop
                # Also tighten trailing if active
                if pos.trailing_stop_active and pos.trailing_stop_price > 0:
                    trail_distance = current_price - pos.trailing_stop_price
                    if trail_distance > 0:
                        pos.trailing_stop_price = pos.trailing_stop_price + (trail_distance * tighten_pct)
        else:
            distance = pos.stop_loss - current_price
            if distance > 0:
                new_stop = pos.stop_loss - (distance * tighten_pct)
                pos.stop_loss = new_stop
                if pos.trailing_stop_active and pos.trailing_stop_price > 0:
                    trail_distance = pos.trailing_stop_price - current_price
                    if trail_distance > 0:
                        pos.trailing_stop_price = pos.trailing_stop_price - (trail_distance * tighten_pct)

    def _shorten_time_exit(self, session: Any, reduce_pct: float) -> None:
        """Reduce the time_exit_bars on the session by reduce_pct."""
        if hasattr(session, "time_exit_bars") and session.time_exit_bars > 0:
            new_limit = max(3, int(session.time_exit_bars * (1.0 - reduce_pct)))
            session.time_exit_bars = new_limit
        if hasattr(session, "choppy_time_exit_bars") and session.choppy_time_exit_bars > 0:
            new_choppy = max(2, int(session.choppy_time_exit_bars * (1.0 - reduce_pct)))
            session.choppy_time_exit_bars = new_choppy

    def get_applied_summary(self) -> Dict[str, bool]:
        """Return which context responses have been applied."""
        return {
            "regime_flip_response": self._applied_regime_flip,
            "flow_reversal_response": self._applied_flow_reversal,
            "momentum_stall_response": self._applied_momentum_stall,
            "volatility_spike_response": self._applied_volatility_spike,
        }


__all__ = ["ContextAwareExitPolicy", "DEFAULT_CONTEXT_EXIT_CONFIG"]
