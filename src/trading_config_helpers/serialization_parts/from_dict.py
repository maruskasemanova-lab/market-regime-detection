"""build_trading_config_from_dict implementation."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from ...runtime_exit_formulas import normalize_runtime_exit_formula_fields
from .from_dict_sections import (
    build_core_kwargs,
    build_intraday_level_kwargs,
    build_tail_kwargs,
)

if TYPE_CHECKING:
    from ...trading_config import TradingConfig


def build_trading_config_from_dict(cls: type["TradingConfig"], d: Dict[str, Any]) -> "TradingConfig":
    raw = d if isinstance(d, dict) else {}
    defaults = cls()
    runtime_formula_fields = normalize_runtime_exit_formula_fields(raw)

    stop_mode = str(raw.get("stop_loss_mode", defaults.stop_loss_mode) or defaults.stop_loss_mode).strip().lower()
    if stop_mode not in cls.VALID_STOP_LOSS_MODES:
        stop_mode = defaults.stop_loss_mode

    selection_mode = str(
        raw.get("strategy_selection_mode", defaults.strategy_selection_mode)
        or defaults.strategy_selection_mode
    ).strip().lower()
    if selection_mode not in cls.VALID_STRATEGY_SELECTION_MODES:
        selection_mode = defaults.strategy_selection_mode

    momentum = raw.get("momentum_diversification", defaults.momentum_diversification)
    if not isinstance(momentum, dict):
        momentum = {}

    micro_mode = str(
        raw.get("micro_confirmation_mode", defaults.micro_confirmation_mode)
        or defaults.micro_confirmation_mode
    ).strip().lower()
    if micro_mode not in cls.VALID_MICRO_CONFIRMATION_MODES:
        micro_mode = defaults.micro_confirmation_mode

    kwargs: Dict[str, Any] = {}
    kwargs.update(
        build_core_kwargs(
            cls=cls,
            raw=raw,
            defaults=defaults,
            runtime_formula_fields=runtime_formula_fields,
            stop_mode=stop_mode,
        )
    )
    kwargs.update(
        build_intraday_level_kwargs(
            cls=cls,
            raw=raw,
            defaults=defaults,
        )
    )
    kwargs.update(
        build_tail_kwargs(
            cls=cls,
            raw=raw,
            defaults=defaults,
            selection_mode=selection_mode,
            momentum=momentum,
            micro_mode=micro_mode,
        )
    )

    return cls(**kwargs)
