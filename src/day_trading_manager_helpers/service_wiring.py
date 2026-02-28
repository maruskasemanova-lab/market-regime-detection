"""Service/component wiring helpers for DayTradingManager."""

from __future__ import annotations

from typing import Any, Dict

from ..day_trading_config_service import DayTradingConfigService
from ..day_trading_evidence_service import DayTradingEvidenceService
from ..day_trading_gate_impl import GateEvaluationEngine
from ..day_trading_strategy_evaluator import StrategyEvaluatorEngine
from ..day_trading_trade_impl import TradeExecutionEngine
from ..exit_policy_engine import ExitPolicyEngine
from ..golden_setup_detector import GoldenSetupDetector
from ..strategy_factory import build_strategy_registry
from ..trading_orchestrator import TradingOrchestrator
from .preferences import (
    default_micro_regime_preferences,
    default_regime_preferences,
    default_ticker_preferences,
)


def wire_manager_services(
    *,
    manager: Any,
    stop_loss_mode: str,
    fixed_stop_loss_pct: float,
) -> None:
    """Wire runtime engines/services and load AOS ticker config."""

    manager.orchestrator = TradingOrchestrator()
    manager.exit_engine = ExitPolicyEngine()
    manager.strategies = build_strategy_registry()
    manager.default_preference = default_regime_preferences()
    manager.micro_regime_preference = default_micro_regime_preferences()
    manager.ticker_preferences = default_ticker_preferences()
    manager.ticker_params = {}

    manager.config_service = DayTradingConfigService(manager)
    manager.evidence_service = DayTradingEvidenceService(
        canonical_strategy_key=manager._canonical_strategy_key,
        safe_float=manager._safe_float,
    )
    manager.trade_engine = TradeExecutionEngine(
        config_service=manager.config_service,
        evidence_service=manager.evidence_service,
        exit_engine=manager.exit_engine,
        ticker_params=manager.ticker_params,
        get_session_key=manager._get_session_key,
        manager=manager,
    )
    manager.strategy_evaluator = StrategyEvaluatorEngine(
        evidence_service=manager.evidence_service,
    )
    manager.gate_engine = GateEvaluationEngine(
        exit_engine=manager.exit_engine,
        config_service=manager.config_service,
        evidence_service=manager.evidence_service,
        default_momentum_strategies=manager.DEFAULT_MOMENTUM_STRATEGIES,
        manager=manager,
        ticker_params=manager.ticker_params,
    )
    manager.golden_setup_detector = GoldenSetupDetector()
    manager.stop_loss_mode = manager._normalize_stop_loss_mode(stop_loss_mode)
    manager.fixed_stop_loss_pct = max(0.0, float(fixed_stop_loss_pct))

    # Load AOS ticker-specific overrides once dependencies are fully wired.
    manager._load_aos_config()


__all__ = ["wire_manager_services"]

