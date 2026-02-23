"""
Shared decision result contract used by runtime decision engines.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .strategies.base_strategy import Signal


@dataclass
class DecisionResult:
    """Normalized decision payload consumed by manager, runner, and UI."""

    execute: bool = False
    direction: Optional[str] = None  # 'bullish' or 'bearish'
    signal: Optional[Signal] = None
    confirming_signals: List[Signal] = field(default_factory=list)
    strategy_score: float = 0.0
    combined_raw: float = 0.0
    combined_norm_0_100: float = 0.0
    combined_score: float = 0.0
    threshold: float = 65.0
    trade_gate_threshold: float = 65.0
    threshold_used_reason: str = "base_threshold"
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execute": self.execute,
            "direction": self.direction,
            "signal": self.signal.to_dict() if self.signal else None,
            "confirming_signals": [s.to_dict() for s in self.confirming_signals],
            "strategy_score": round(self.strategy_score, 1),
            "combined_raw": round(self.combined_raw, 2),
            "combined_norm_0_100": round(self.combined_norm_0_100, 1),
            "combined_score": round(self.combined_score, 1),
            "threshold": self.threshold,
            "trade_gate_threshold": self.trade_gate_threshold,
            "threshold_used_reason": self.threshold_used_reason,
            "reasoning": self.reasoning,
        }
