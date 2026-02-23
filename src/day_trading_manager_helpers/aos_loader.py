"""AOS configuration loading helper for DayTradingManager."""

from __future__ import annotations

from typing import Any, Dict, Optional
import json
import logging
import os

logger = logging.getLogger(__name__)


def load_aos_config(self, config_path: Optional[str] = None) -> None:
    """Load AOS configuration from file if available."""
    # Try default paths.
    if config_path is None:
        possible_paths = [
            "/Users/hotovo/.gemini/antigravity/scratch/backtest-runner/aos_optimization/aos_config.json",
            "aos_optimization/aos_config.json",
            "../backtest-runner/aos_optimization/aos_config.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            loaded_ticker_params: Dict[str, Dict[str, Any]] = {}

            # Load ticker-specific parameters (do not mutate global strategy
            # instances per ticker; apply overrides at signal generation time).
            for ticker, ticker_config in config.get("tickers", {}).items():
                ticker_key = str(ticker).upper()
                primary_strategy = self._canonical_strategy_key(ticker_config.get("strategy", ""))
                backup_strategy = self._canonical_strategy_key(ticker_config.get("backup_strategy", ""))
                loaded_ticker_params[ticker_key] = {
                    "strategy": primary_strategy,
                    "backup_strategy": backup_strategy,
                    "params": ticker_config.get("params", {}),
                    "regime_filter": ticker_config.get("regime_filter", []),
                    "avoid_days": ticker_config.get("avoid_days", []),
                    "trading_hours": ticker_config.get("trading_hours", None),
                    "long_only": ticker_config.get("long_only", False),
                    "time_filter_enabled": ticker_config.get("time_filter_enabled", True),
                    "min_confidence": ticker_config.get("min_confidence", 65.0),
                    "max_daily_trades": ticker_config.get("max_daily_trades", 2),
                    "mu_choppy_hard_block_enabled": self._safe_bool(
                        ticker_config.get("mu_choppy_hard_block_enabled"),
                        True,
                    ),
                    "adverse_flow_consistency_threshold": max(
                        0.02,
                        float(
                            self._safe_float(
                                ticker_config.get("adverse_flow_consistency_threshold"),
                                self.adverse_flow_consistency_threshold,
                            )
                            or self.adverse_flow_consistency_threshold
                        ),
                    ),
                    "adverse_book_pressure_threshold": max(
                        0.02,
                        float(
                            self._safe_float(
                                ticker_config.get("adverse_book_pressure_threshold"),
                                self.adverse_book_pressure_threshold,
                            )
                            or self.adverse_book_pressure_threshold
                        ),
                    ),
                    "strategy_selection_mode": self._normalize_strategy_selection_mode(
                        ticker_config.get("strategy_selection_mode")
                    ),
                    "max_active_strategies": self._normalize_max_active_strategies(
                        ticker_config.get("max_active_strategies"),
                        default=3,
                    ),
                    "l2": (
                        ticker_config.get("l2", {})
                        if isinstance(ticker_config.get("l2"), dict)
                        else {}
                    ),
                    "adaptive": self._normalize_adaptive_config(ticker_config.get("adaptive")),
                }

            self.ticker_params = loaded_ticker_params
            logger.info(
                "Loaded AOS config from %s (%s tickers)",
                config_path,
                len(self.ticker_params),
            )
        except Exception as e:
            logger.warning("Could not load AOS config: %s", e)
