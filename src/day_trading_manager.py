"""
Session-based Day Trading Manager.
Manages trading sessions for individual days with regime detection and strategy execution.
"""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

from .strategies.base_strategy import BaseStrategy, Signal, Position, Regime
from .strategy_factory import build_strategy_registry
from .trading_orchestrator import TradingOrchestrator
from .exit_policy_engine import ExitPolicyEngine
from .day_trading_config_service import DayTradingConfigService
from .day_trading_evidence_service import DayTradingEvidenceService
from .day_trading_strategy_evaluator import StrategyEvaluatorEngine
from .day_trading_models import (
    BarData,
    DayTrade,
    SessionPhase,
    TradingConfig,
    TradingCosts,
    TradingSession,
)
from .day_trading_runtime_impl import (
    runtime_attach_intrabar_eval_trace,
    runtime_calculate_indicators,
    runtime_detect_liquidity_sweep,
    runtime_evaluate_intraday_levels_entry_quality,
    runtime_generate_signal,
    runtime_process_bar,
    runtime_process_trading_bar,
    runtime_evaluate_intrabar_slice,
)
from .day_trading_regime_impl import (
    regime_build_momentum_route_candidates,
    regime_calc_adx,
    regime_calc_adx_series,
    regime_calc_trend_efficiency,
    regime_calc_volatility,
    regime_calculate_order_flow_metrics,
    regime_classify_micro_regime,
    regime_detect_regime,
    regime_map_adaptive_regime,
    regime_map_micro_to_regime,
    regime_maybe_refresh_regime,
    regime_resolve_momentum_diversification,
    regime_safe_div,
    regime_select_momentum_sleeve,
    regime_select_strategies,
)
from .day_trading_gate_impl import GateEvaluationEngine
from .day_trading_trade_impl import TradeExecutionEngine
from .strategy_formula_engine import evaluate_strategy_formula
from .strategy_formula_engine import StrategyFormulaEvaluationError
from .intraday_levels import (
    build_intraday_levels_snapshot,
    ensure_intraday_levels_state,
    intraday_levels_indicator_payload,
    update_intraday_levels_state,
)
from .golden_setup_detector import GoldenSetupDetector
from .day_trading_intraday_memory import IntradayMemoryService
from .day_trading_manager_helpers.aos_loader import (
    load_aos_config as _load_aos_config_impl,
)
from .day_trading_manager_helpers.intrabar_bar_factory import (
    build_intrabar_slice_bar_data,
)
from .day_trading_manager_helpers.preferences import (
    bootstrap_ticker_preferences,
    default_micro_regime_preferences,
    default_regime_preferences,
    default_ticker_preferences,
)
from .day_trading_manager_helpers.session_config import (
    build_ticker_session_config_payload,
)
from .day_trading_manager_helpers.session_lifecycle import (
    attach_orchestrator_to_new_session,
    recover_existing_session_orchestrator,
)
from .day_trading_manager_helpers.run_defaults import (
    collect_non_none_run_default_overrides,
)


class DayTradingManager:
    """
    Manages multiple trading sessions.
    Each session is identified by run_id + ticker + date.
    """
    VALID_STOP_LOSS_MODES = {"strategy", "fixed", "capped"}
    VALID_STRATEGY_SELECTION_MODES = {"adaptive_top_n", "all_enabled"}
    MACRO_REGIME_KEYS = ("TRENDING", "CHOPPY", "MIXED")
    MICRO_REGIME_KEYS = (
        "TRENDING_UP",
        "TRENDING_DOWN",
        "CHOPPY",
        "ABSORPTION",
        "BREAKOUT",
        "MIXED",
        "TRANSITION",
        "UNKNOWN",
    )
    DEFAULT_FLOW_BIAS_STRATEGIES = (
        "momentum_flow",
        "absorption_reversal",
        "exhaustion_fade",
        "scalp_l2_intrabar",
        "evidence_scalp",
        "options_flow_alpha",
    )
    MOMENTUM_ROUTE_KEYS = ("impulse", "continuation", "defensive")
    DEFAULT_MOMENTUM_STRATEGIES = ("momentum_flow", "momentum", "pullback")
    MAX_MOMENTUM_SLEEVES = 8
    
    def __init__(
        self,
        regime_detection_minutes: int = 30,
        trading_costs: TradingCosts = None,
        max_daily_loss: float = 300.0,
        max_trades_per_day: int = 3,
        trade_cooldown_bars: int = 2,  # Cooldown between trades in bars
        pending_signal_ttl_bars: int = 3,
        consecutive_loss_limit: int = 3,
        consecutive_loss_cooldown_bars: int = 8,
        regime_refresh_bars: int = 30,
        risk_per_trade_pct: float = 1.0,
        max_position_notional_pct: float = 100.0,
        time_exit_bars: int = 40,
        enable_partial_take_profit: bool = True,
        partial_take_profit_rr: float = 1.0,
        partial_take_profit_fraction: float = 0.5,
        trailing_activation_pct: float = 0.15,
        break_even_buffer_pct: float = 0.03,
        break_even_min_hold_bars: int = 3,
        break_even_activation_min_mfe_pct: float = 0.25,
        break_even_activation_min_r: float = 0.60,
        break_even_activation_min_r_trending_5m: float = 0.90,
        break_even_activation_min_r_choppy_5m: float = 0.60,
        break_even_activation_use_levels: bool = True,
        break_even_activation_use_l2: bool = True,
        break_even_level_buffer_pct: float = 0.02,
        break_even_level_max_distance_pct: float = 0.60,
        break_even_level_min_confluence: int = 2,
        break_even_level_min_tests: int = 1,
        break_even_l2_signed_aggression_min: float = 0.12,
        break_even_l2_imbalance_min: float = 0.15,
        break_even_l2_book_pressure_min: float = 0.10,
        break_even_l2_spread_bps_max: float = 12.0,
        break_even_costs_pct: float = 0.03,
        break_even_min_buffer_pct: float = 0.05,
        break_even_atr_buffer_k: float = 0.10,
        break_even_5m_atr_buffer_k: float = 0.10,
        break_even_tick_size: float = 0.01,
        break_even_min_tick_buffer: int = 1,
        break_even_anti_spike_bars: int = 1,
        break_even_anti_spike_hits_required: int = 2,
        break_even_anti_spike_require_close_beyond: bool = True,
        break_even_5m_no_go_proximity_pct: float = 0.10,
        break_even_5m_mfe_atr_factor: float = 0.15,
        break_even_5m_l2_bias_threshold: float = 0.10,
        break_even_5m_l2_bias_tighten_factor: float = 0.85,
        trailing_enabled_in_choppy: bool = False,
        adverse_flow_exit_enabled: bool = True,
        adverse_flow_exit_threshold: float = 0.20,
        adverse_flow_min_hold_bars: int = 6,
        adverse_flow_consistency_threshold: float = 0.45,
        adverse_book_pressure_threshold: float = 0.15,
        stop_loss_mode: str = "strategy",
        fixed_stop_loss_pct: float = 0.0,
        portfolio_drawdown_halt_pct: float = 5.0,
        headwind_activation_score: float = 0.5,
    ):
        self.sessions: Dict[str, TradingSession] = {}  # session_key -> TradingSession
        self.regime_detection_minutes = regime_detection_minutes
        self.trading_costs = trading_costs or TradingCosts()
        self.max_daily_loss = max_daily_loss  # Stop trading if loss exceeds this amount
        self.max_trades_per_day = max_trades_per_day  # Limit trades to reduce fees
        self.trade_cooldown_bars = trade_cooldown_bars  # Cooldown between trades
        self.pending_signal_ttl_bars = max(1, int(pending_signal_ttl_bars))
        self.consecutive_loss_limit = max(1, int(consecutive_loss_limit))
        self.consecutive_loss_cooldown_bars = max(0, int(consecutive_loss_cooldown_bars))
        self.regime_refresh_bars = max(1, int(regime_refresh_bars))
        self.risk_per_trade_pct = max(0.1, float(risk_per_trade_pct))
        self.max_position_notional_pct = max(1.0, float(max_position_notional_pct))
        self.max_fill_participation_rate = 0.20
        self.min_fill_ratio = 0.35
        self.time_exit_bars = max(1, int(time_exit_bars))
        self.enable_partial_take_profit = bool(enable_partial_take_profit)
        self.partial_take_profit_rr = max(0.25, float(partial_take_profit_rr))
        self.partial_take_profit_fraction = min(0.95, max(0.05, float(partial_take_profit_fraction)))
        self.trailing_activation_pct = max(0.0, float(trailing_activation_pct))
        self.break_even_buffer_pct = max(0.0, float(break_even_buffer_pct))
        self.break_even_min_hold_bars = max(1, int(break_even_min_hold_bars))
        self.break_even_activation_min_mfe_pct = max(0.0, float(break_even_activation_min_mfe_pct))
        self.break_even_activation_min_r = max(0.0, float(break_even_activation_min_r))
        self.break_even_activation_min_r_trending_5m = max(
            0.0, float(break_even_activation_min_r_trending_5m)
        )
        self.break_even_activation_min_r_choppy_5m = max(
            0.0, float(break_even_activation_min_r_choppy_5m)
        )
        self.break_even_activation_use_levels = bool(break_even_activation_use_levels)
        self.break_even_activation_use_l2 = bool(break_even_activation_use_l2)
        self.break_even_level_buffer_pct = max(0.0, float(break_even_level_buffer_pct))
        self.break_even_level_max_distance_pct = max(0.01, float(break_even_level_max_distance_pct))
        self.break_even_level_min_confluence = max(0, int(break_even_level_min_confluence))
        self.break_even_level_min_tests = max(0, int(break_even_level_min_tests))
        self.break_even_l2_signed_aggression_min = max(
            0.0, float(break_even_l2_signed_aggression_min)
        )
        self.break_even_l2_imbalance_min = max(0.0, float(break_even_l2_imbalance_min))
        self.break_even_l2_book_pressure_min = max(
            0.0, float(break_even_l2_book_pressure_min)
        )
        self.break_even_l2_spread_bps_max = max(0.0, float(break_even_l2_spread_bps_max))
        self.break_even_costs_pct = max(0.0, float(break_even_costs_pct))
        self.break_even_min_buffer_pct = max(0.0, float(break_even_min_buffer_pct))
        self.break_even_atr_buffer_k = max(0.0, float(break_even_atr_buffer_k))
        self.break_even_5m_atr_buffer_k = max(0.0, float(break_even_5m_atr_buffer_k))
        self.break_even_tick_size = max(0.0, float(break_even_tick_size))
        self.break_even_min_tick_buffer = max(0, int(break_even_min_tick_buffer))
        self.break_even_anti_spike_bars = max(0, int(break_even_anti_spike_bars))
        self.break_even_anti_spike_hits_required = max(1, int(break_even_anti_spike_hits_required))
        self.break_even_anti_spike_require_close_beyond = bool(
            break_even_anti_spike_require_close_beyond
        )
        self.break_even_5m_no_go_proximity_pct = max(
            0.0, float(break_even_5m_no_go_proximity_pct)
        )
        self.break_even_5m_mfe_atr_factor = max(0.0, float(break_even_5m_mfe_atr_factor))
        self.break_even_5m_l2_bias_threshold = max(
            0.0, float(break_even_5m_l2_bias_threshold)
        )
        self.break_even_5m_l2_bias_tighten_factor = max(
            0.1, min(2.0, float(break_even_5m_l2_bias_tighten_factor))
        )
        self.trailing_enabled_in_choppy = bool(trailing_enabled_in_choppy)
        self.adverse_flow_exit_enabled = bool(adverse_flow_exit_enabled)
        self.adverse_flow_exit_threshold = max(0.02, float(adverse_flow_exit_threshold))
        self.adverse_flow_min_hold_bars = max(1, int(adverse_flow_min_hold_bars))
        self.adverse_flow_consistency_threshold = max(0.02, float(adverse_flow_consistency_threshold))
        self.adverse_book_pressure_threshold = max(0.02, float(adverse_book_pressure_threshold))
        self.portfolio_drawdown_halt_pct = max(0.0, float(portfolio_drawdown_halt_pct))
        self.headwind_activation_score = min(0.95, max(0.0, float(headwind_activation_score)))
        self.market_tz = ZoneInfo("America/New_York")
        self.run_defaults: Dict[tuple, Dict[str, Any]] = {}
        self.run_equity_state: Dict[str, Dict[str, Any]] = {}
        self.intraday_memory = IntradayMemoryService()
        
        self.last_trade_bar_index = {}  # Track last trade bar per session

        # Evidence-based orchestrator (new architecture)
        self.orchestrator = TradingOrchestrator()

        # Exit policy engine (extracted from this class)
        self.exit_engine = ExitPolicyEngine()

        # Pre-configured strategies
        self.strategies = build_strategy_registry()

        # Helper services: config normalization + evidence parsing.
        # Bootstrap preferences are used before full defaults/config overrides load.
        self.ticker_preferences = bootstrap_ticker_preferences()

        # Initialize ticker_params dictionary
        self.ticker_params: Dict[str, Dict[str, Any]] = {}
        target_file = "/Users/hotovo/.gemini/antigravity/scratch/backtest-runner/aos_optimization/aos_config.json"
        
        # Load local settings mapping (if provided)
        try:
            self._load_aos_config(target_file)
        except Exception as e:
            logger.warning(f"Could not load AOS config from {target_file}: {e}")

        self.config_service = DayTradingConfigService(self)
        self.evidence_service = DayTradingEvidenceService(
            canonical_strategy_key=self._canonical_strategy_key,
            safe_float=self._safe_float,
        )
        self.trade_engine = TradeExecutionEngine(
            config_service=self.config_service,
            evidence_service=self.evidence_service,
            exit_engine=self.exit_engine,
            ticker_params=self.ticker_params,
            get_session_key=self._get_session_key,
            manager=self,
        )
        self.strategy_evaluator = StrategyEvaluatorEngine(
            evidence_service=self.evidence_service,
        )
        self.gate_engine = GateEvaluationEngine(
            exit_engine=self.exit_engine,
            config_service=self.config_service,
            evidence_service=self.evidence_service,
            default_momentum_strategies=self.DEFAULT_MOMENTUM_STRATEGIES,
            manager=self,
            ticker_params=self.ticker_params,
        )
        self.golden_setup_detector = GoldenSetupDetector()
        self.stop_loss_mode = self._normalize_stop_loss_mode(stop_loss_mode)
        self.fixed_stop_loss_pct = max(0.0, float(fixed_stop_loss_pct))
        
        # Strategy selection by regime
        # Default Preferences - including new volume-focused strategies
        self.default_preference = default_regime_preferences()
        self.micro_regime_preference = default_micro_regime_preferences()
        self.ticker_preferences = default_ticker_preferences()
        
        # AOS ticker-specific parameters (will be loaded from config)
        self.ticker_params = {}
        
        # Try to load AOS config
        self._load_aos_config()
    
    def _load_aos_config(self, config_path: str = None):
        return _load_aos_config_impl(self=self, config_path=config_path)
        
    def _to_market_time(self, ts: datetime) -> datetime:
        """Convert timestamp to US/Eastern for market-hour comparisons."""
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        return ts.astimezone(self.market_tz)

    def _canonical_strategy_key(self, strategy_name: str) -> str:
        return self.config_service.canonical_strategy_key(strategy_name)

    def _canonical_trading_config(self, raw_config: Dict[str, Any]) -> TradingConfig:
        return self.config_service.canonical_trading_config(raw_config)

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        return DayTradingConfigService.safe_float(value, default)

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        return DayTradingConfigService.safe_bool(value, default)

    def _normalize_stop_loss_mode(self, mode: Any) -> str:
        return self.config_service.normalize_stop_loss_mode(mode)

    def _normalize_strategy_selection_mode(self, mode: Any) -> str:
        return self.config_service.normalize_strategy_selection_mode(mode)

    @staticmethod
    def _normalize_max_active_strategies(value: Any, default: int = 3) -> int:
        return DayTradingConfigService.normalize_max_active_strategies(value, default)

    @staticmethod
    def _normalize_non_negative_int(value: Any, default: int = 0, max_value: int = 10_000) -> int:
        return DayTradingConfigService.normalize_non_negative_int(value, default, max_value)

    def _default_macro_preference_map(self) -> Dict[str, List[str]]:
        return self.config_service.default_macro_preference_map()

    def _normalize_strategy_list(
        self,
        raw_value: Any,
        fallback: Optional[List[str]] = None,
    ) -> List[str]:
        return self.config_service.normalize_strategy_list(raw_value, fallback)

    def _normalize_preference_map(
        self,
        raw_value: Any,
        keys: Tuple[str, ...],
        fallback_map: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        return self.config_service.normalize_preference_map(raw_value, keys, fallback_map)

    def _normalize_momentum_route_name(self, value: Any, default: str = "continuation") -> str:
        return self.config_service.normalize_momentum_route_name(value, default)

    @staticmethod
    def _normalize_momentum_sleeve_id(value: Any, fallback: str) -> str:
        return DayTradingConfigService.normalize_momentum_sleeve_id(value, fallback)

    def _default_momentum_route_map(self) -> Dict[str, List[str]]:
        return self.config_service.default_momentum_route_map()

    def _default_micro_route_map(self) -> Dict[str, str]:
        return self.config_service.default_micro_route_map()

    def _normalize_micro_regime_list(self, raw_value: Any) -> List[str]:
        return self.config_service.normalize_micro_regime_list(raw_value)

    def _normalize_momentum_diversification_config(
        self,
        raw_value: Any,
        *,
        include_sleeves: bool = True,
    ) -> Dict[str, Any]:
        return self.config_service.normalize_momentum_diversification_config(
            raw_value,
            include_sleeves=include_sleeves,
        )

    def _normalize_adaptive_config(self, raw_value: Any) -> Dict[str, Any]:
        return self.config_service.normalize_adaptive_config(raw_value)

    def _latest_indicator_value(
        self,
        indicators: Dict[str, Any],
        key: str,
        bars: Optional[List[BarData]] = None,
    ) -> float:
        """Read the latest scalar value from an indicator payload with ATR fallback."""
        raw = indicators.get(key)
        value: Optional[float] = None

        if isinstance(raw, list):
            if raw:
                value = self._safe_float(raw[-1], None)
        elif raw is not None:
            value = self._safe_float(raw, None)

        # ATR can be unavailable early in session if the window is short.
        # Use a short-window true range fallback to avoid misleading 0.00 values.
        if value is None and key == "atr" and bars and len(bars) >= 2:
            tr_values: List[float] = []
            for i in range(1, len(bars)):
                prev_close = bars[i - 1].close
                hl = bars[i].high - bars[i].low
                hc = abs(bars[i].high - prev_close)
                lc = abs(bars[i].low - prev_close)
                tr_values.append(max(hl, hc, lc))
            if tr_values:
                window = min(14, len(tr_values))
                value = sum(tr_values[-window:]) / window

        return float(value) if value is not None else 0.0


    def _apply_trading_config_to_session(
        self,
        session: TradingSession,
        config: TradingConfig,
        *,
        normalize_momentum: bool = True,
    ) -> None:
        self.config_service.apply_trading_config_to_session(
            session=session,
            config=config,
            normalize_momentum=normalize_momentum,
        )

    def _is_day_allowed(self, date_str: str, ticker: str) -> bool:
        return self.config_service.is_day_allowed(date_str, ticker)

    def _resolve_max_daily_trades(
        self,
        ticker: str,
        session: Optional[TradingSession] = None,
    ) -> Optional[int]:
        return self.config_service.resolve_max_daily_trades(ticker, session)

    def _is_mu_choppy_filter_enabled(
        self,
        ticker: str,
        session: Optional[TradingSession] = None,
    ) -> bool:
        return self.config_service.is_mu_choppy_filter_enabled(ticker, session)

    def _get_ticker_strategy_overrides(self, ticker: str, strategy_key: str) -> Dict[str, Any]:
        return self.config_service.get_ticker_strategy_overrides(ticker, strategy_key)



    def _get_session_key(self, run_id: str, ticker: str, date: str) -> str:
        """Generate unique session key."""
        return f"{run_id}:{ticker}:{date}"
    
    def get_or_create_session(
        self, 
        run_id: str, 
        ticker: str, 
        date: str,
        regime_detection_minutes: int = None,
        cold_start_each_day: bool = False,
    ) -> TradingSession:
        """Get existing session or create new one."""
        key = self._get_session_key(run_id, ticker, date)
        resolved_regime_detection_minutes = (
            regime_detection_minutes or self.regime_detection_minutes
        )

        if key in self.sessions:
            session = self.sessions[key]
            # Recover partially created sessions (e.g. previous config failure)
            # so subsequent bar processing does not fail with a missing orchestrator.
            recover_existing_session_orchestrator(
                session,
                self.orchestrator,
                cold_start_each_day=cold_start_each_day,
            )
            return session

        if key not in self.sessions:
            # Refresh AOS config for newly created sessions so dashboard updates
            # are picked up.
            self._load_aos_config()
            # Defensive reset: if a previous run used the same session key,
            # stale cooldown state must not leak into the new replay.
            self.last_trade_bar_index.pop(key, None)

            self.sessions[key] = TradingSession(
                run_id=run_id,
                ticker=ticker,
                date=date,
                regime_detection_minutes=resolved_regime_detection_minutes,
            )
            session = self.sessions[key]
            ticker_cfg = self.ticker_params.get(str(ticker).upper(), {})
            if not isinstance(ticker_cfg, dict):
                ticker_cfg = {}
            config_payload = build_ticker_session_config_payload(
                self,
                ticker_cfg=ticker_cfg,
                regime_detection_minutes=resolved_regime_detection_minutes,
            )
            base_config = self._canonical_trading_config(config_payload)
            self._apply_trading_config_to_session(session, base_config, normalize_momentum=False)
            self._inject_intraday_levels_memory_into_session(session)

            # Attach orchestrator and reset session state for new day
            attach_orchestrator_to_new_session(
                session,
                self.orchestrator,
                cold_start_each_day=cold_start_each_day,
            )

            # Optional config-driven day filter.
            if not self._is_day_allowed(date, ticker):
                session.phase = SessionPhase.CLOSED

        return self.sessions[key]

    def set_run_defaults(
        self,
        run_id: str,
        ticker: str,
        regime_detection_minutes: Optional[int] = None,
        regime_refresh_bars: Optional[int] = None,
        account_size_usd: Optional[float] = None,
        risk_per_trade_pct: Optional[float] = None,
        max_position_notional_pct: Optional[float] = None,
        max_fill_participation_rate: Optional[float] = None,
        min_fill_ratio: Optional[float] = None,
        enable_partial_take_profit: Optional[bool] = None,
        partial_take_profit_rr: Optional[float] = None,
        partial_take_profit_fraction: Optional[float] = None,
        trailing_activation_pct: Optional[float] = None,
        break_even_buffer_pct: Optional[float] = None,
        break_even_min_hold_bars: Optional[int] = None,
        break_even_activation_min_mfe_pct: Optional[float] = None,
        break_even_activation_min_r: Optional[float] = None,
        break_even_activation_min_r_trending_5m: Optional[float] = None,
        break_even_activation_min_r_choppy_5m: Optional[float] = None,
        break_even_activation_use_levels: Optional[bool] = None,
        break_even_activation_use_l2: Optional[bool] = None,
        break_even_level_buffer_pct: Optional[float] = None,
        break_even_level_max_distance_pct: Optional[float] = None,
        break_even_level_min_confluence: Optional[int] = None,
        break_even_level_min_tests: Optional[int] = None,
        break_even_l2_signed_aggression_min: Optional[float] = None,
        break_even_l2_imbalance_min: Optional[float] = None,
        break_even_l2_book_pressure_min: Optional[float] = None,
        break_even_l2_spread_bps_max: Optional[float] = None,
        break_even_costs_pct: Optional[float] = None,
        break_even_min_buffer_pct: Optional[float] = None,
        break_even_atr_buffer_k: Optional[float] = None,
        break_even_5m_atr_buffer_k: Optional[float] = None,
        break_even_tick_size: Optional[float] = None,
        break_even_min_tick_buffer: Optional[int] = None,
        break_even_anti_spike_bars: Optional[int] = None,
        break_even_anti_spike_hits_required: Optional[int] = None,
        break_even_anti_spike_require_close_beyond: Optional[bool] = None,
        break_even_5m_no_go_proximity_pct: Optional[float] = None,
        break_even_5m_mfe_atr_factor: Optional[float] = None,
        break_even_5m_l2_bias_threshold: Optional[float] = None,
        break_even_5m_l2_bias_tighten_factor: Optional[float] = None,
        trailing_enabled_in_choppy: Optional[bool] = None,
        time_exit_bars: Optional[int] = None,
        adverse_flow_exit_enabled: Optional[bool] = None,
        adverse_flow_threshold: Optional[float] = None,
        adverse_flow_min_hold_bars: Optional[int] = None,
        adverse_flow_consistency_threshold: Optional[float] = None,
        adverse_book_pressure_threshold: Optional[float] = None,
        stop_loss_mode: Optional[str] = None,
        fixed_stop_loss_pct: Optional[float] = None,
        l2_confirm_enabled: Optional[bool] = None,
        l2_min_delta: Optional[float] = None,
        l2_min_imbalance: Optional[float] = None,
        l2_min_iceberg_bias: Optional[float] = None,
        l2_lookback_bars: Optional[int] = None,
        l2_min_participation_ratio: Optional[float] = None,
        l2_min_directional_consistency: Optional[float] = None,
        l2_min_signed_aggression: Optional[float] = None,
        cold_start_each_day: Optional[bool] = None,
        strategy_selection_mode: Optional[str] = None,
        max_active_strategies: Optional[int] = None,
        momentum_diversification: Optional[Dict[str, Any]] = None,
        trading_config: Optional[Any] = None,
    ) -> None:
        """Set default parameters for all sessions in a run."""
        key = (run_id, ticker)
        defaults = self.run_defaults.get(key, {})
        if not isinstance(defaults, dict):
            defaults = {}

        updates: Dict[str, Any] = {}
        if isinstance(trading_config, TradingConfig):
            updates.update(trading_config.to_session_params())
        elif isinstance(trading_config, dict):
            updates.update(trading_config)

        updates.update(collect_non_none_run_default_overrides(locals()))

        if momentum_diversification is not None:
            updates["momentum_diversification"] = self._normalize_momentum_diversification_config(
                momentum_diversification
            )
        elif "momentum_diversification" in updates and isinstance(updates["momentum_diversification"], dict):
            updates["momentum_diversification"] = self._normalize_momentum_diversification_config(
                updates["momentum_diversification"]
            )

        if not updates:
            return

        validated = self._canonical_trading_config({**defaults, **updates}).to_session_params()
        for field_name in updates:
            defaults[field_name] = validated[field_name]
        self.run_defaults[key] = defaults
    
    def get_session(self, run_id: str, ticker: str, date: str) -> Optional[TradingSession]:
        """Get existing session."""
        key = self._get_session_key(run_id, ticker, date)
        return self.sessions.get(key)
    
    def process_bar(
        self,
        run_id: str,
        ticker: str,
        timestamp: datetime,
        bar_data: Dict[str, Any],
        warmup_only: bool = False,
    ) -> Dict[str, Any]:
        result = runtime_process_bar(
            self=self,
            run_id=run_id,
            ticker=ticker,
            timestamp=timestamp,
            bar_data=bar_data,
            warmup_only=warmup_only,
        )
        return runtime_attach_intrabar_eval_trace(
            timestamp=timestamp,
            bar_data=bar_data,
            result=result,
        )

    def evaluate_intrabar_slice(
        self,
        run_id: str,
        ticker: str,
        timestamp: datetime,
        bar_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate strategy over a single 5s intrabar slice without advancing session.
        """
        session = self.get_session(run_id, ticker, timestamp.strftime('%Y-%m-%d'))
        if not session:
            return {"error": "Session not found"}

        bar = build_intrabar_slice_bar_data(timestamp=timestamp, bar_data=bar_data)

        return runtime_evaluate_intrabar_slice(
            self=self,
            session=session,
            bar=bar,
            timestamp=timestamp,
        )
    
    def _detect_regime(self, session: TradingSession) -> Regime:
        return regime_detect_regime(
            self=self,
            session=session,
        )

    @staticmethod
    def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        return regime_safe_div(
            numerator=numerator,
            denominator=denominator,
            default=default,
        )

    def _map_micro_to_regime(self, micro_regime: str) -> Regime:
        return regime_map_micro_to_regime(
            self=self,
            micro_regime=micro_regime,
        )

    @staticmethod
    def _map_adaptive_regime(primary: str) -> Regime:
        return regime_map_adaptive_regime(primary=primary)

    def _classify_micro_regime(
        self,
        trend_efficiency: float,
        adx: Optional[float],
        volatility: float,
        price_change_pct: float,
        flow: Dict[str, float],
    ) -> str:
        return regime_classify_micro_regime(
            self=self,
            trend_efficiency=trend_efficiency,
            adx=adx,
            volatility=volatility,
            price_change_pct=price_change_pct,
            flow=flow,
        )

    def _calculate_order_flow_metrics(
        self,
        bars: List[BarData],
        lookback: int = 20,
    ) -> Dict[str, float]:
        return regime_calculate_order_flow_metrics(
            self=self,
            bars=bars,
            lookback=lookback,
        )

    def _resolve_momentum_diversification(
        self,
        session: TradingSession,
        adaptive_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        return regime_resolve_momentum_diversification(
            self=self,
            session=session,
            adaptive_cfg=adaptive_cfg,
        )

    def _select_momentum_sleeve(
        self,
        momentum_cfg: Dict[str, Any],
        *,
        strategy_key: str = "",
        micro_regime: str = "MIXED",
        has_l2_coverage: Optional[bool] = None,
        preferred_sleeve_id: str = "",
    ) -> Tuple[Dict[str, Any], Optional[str], str]:
        return regime_select_momentum_sleeve(
            self=self,
            momentum_cfg=momentum_cfg,
            strategy_key=strategy_key,
            micro_regime=micro_regime,
            has_l2_coverage=has_l2_coverage,
            preferred_sleeve_id=preferred_sleeve_id,
        )

    def _build_momentum_route_candidates(
        self,
        session: TradingSession,
        flow_metrics: Dict[str, Any],
        momentum_cfg: Dict[str, Any],
    ) -> List[str]:
        return regime_build_momentum_route_candidates(
            self=self,
            session=session,
            flow_metrics=flow_metrics,
            momentum_cfg=momentum_cfg,
        )

    def _select_strategies(self, session: TradingSession) -> List[str]:
        return regime_select_strategies(
            self=self,
            session=session,
        )

    def _calc_trend_efficiency(self, bars: List[BarData]) -> float:
        return regime_calc_trend_efficiency(
            self=self,
            bars=bars,
        )

    def _calc_volatility(self, bars: List[BarData]) -> float:
        return regime_calc_volatility(
            self=self,
            bars=bars,
        )

    def _calc_adx(self, bars: List[BarData], period: int = 14) -> Optional[float]:
        return regime_calc_adx(
            self=self,
            bars=bars,
            period=period,
        )

    def _calc_adx_series(self, bars: List[BarData], period: int = 14) -> List[float]:
        return regime_calc_adx_series(
            self=self,
            bars=bars,
            period=period,
        )

    def _maybe_refresh_regime(
        self,
        session: TradingSession,
        current_bar_index: int,
        timestamp: datetime,
    ) -> Optional[Dict[str, Any]]:
        return regime_maybe_refresh_regime(
            self=self,
            session=session,
            current_bar_index=current_bar_index,
            timestamp=timestamp,
        )

    def _strategy_edge_adjustment(self, session: TradingSession, strategy_key: str) -> float:
        """Estimate strategy edge from already realized session trades.

        Applies a warmup ramp: no adjustment for first 2 trades,
        then gradually ramps to full strength over trades 3-5.
        This prevents single-trade statistical noise from suppressing
        valid follow-up signals.
        """
        relevant = [
            t for t in session.trades
            if self._canonical_strategy_key(getattr(t, "strategy", "")) == strategy_key
        ]
        if not relevant:
            return 0.0

        n = len(relevant)
        # Warmup: no edge adjustment until we have enough trades for
        # statistical relevance. Ramp from 0% at trade 2 to 100% at trade 5+.
        edge_warmup_trades = 3
        if n < edge_warmup_trades:
            return 0.0
        ramp = min(1.0, (n - edge_warmup_trades + 1) / 3.0)

        wins = [t for t in relevant if t.pnl_pct > 0]
        losses = [t for t in relevant if t.pnl_pct <= 0]
        # Smoothed win rate to avoid unstable first trades.
        win_rate = (len(wins) + 1.0) / (n + 2.0)
        avg_win = (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (abs(sum(t.pnl_pct for t in losses)) / len(losses)) if losses else 0.0
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        pf = (sum(t.pnl_pct for t in wins) / abs(sum(t.pnl_pct for t in losses))) if losses else (2.0 if wins else 0.0)

        edge = ((win_rate - 0.5) * 36.0) + ((pf - 1.0) * 6.0) + (expectancy * 4.0)
        edge = max(-12.0, min(12.0, edge))
        return edge * ramp
    
    def _process_trading_bar(
        self, 
        session: TradingSession, 
        bar: BarData, 
        timestamp: datetime,
        warmup_only: bool = False,
    ) -> Dict[str, Any]:
        return runtime_process_trading_bar(
            self=self,
            session=session,
            bar=bar,
            timestamp=timestamp,
            warmup_only=warmup_only,
        )

    def _detect_liquidity_sweep(
        self,
        *,
        session: TradingSession,
        current_price: float,
        fv: Any,
        flow_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return runtime_detect_liquidity_sweep(
            self=self,
            session=session,
            current_price=current_price,
            fv=fv,
            flow_metrics=flow_metrics,
        )

    def _evaluate_intraday_levels_entry_quality(
        self,
        *,
        session: TradingSession,
        signal: Signal,
        current_price: float,
        current_bar_index: int,
    ) -> Dict[str, Any]:
        return runtime_evaluate_intraday_levels_entry_quality(
            self=self,
            session=session,
            signal=signal,
            current_price=current_price,
            current_bar_index=current_bar_index,
        )

    def _effective_stop_for_position(self, pos: Position) -> tuple[Optional[float], Optional[str]]:
        return self.exit_engine.effective_stop_for_position(pos)


    def _resolve_exit_for_bar(self, pos: Position, bar: BarData) -> Optional[tuple]:
        return self.exit_engine.resolve_exit_for_bar(pos, bar)

    @staticmethod
    def _safe_intrabar_quote(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _resolve_exit_from_intrabar_quotes(
        self,
        pos: Position,
        stop_level: Optional[float],
        stop_reason: Optional[str],
        intrabar_quotes: List[Dict[str, float]],
    ) -> Optional[tuple]:
        return self.exit_engine._resolve_exit_from_intrabar_quotes(
            pos=pos,
            stop_level=stop_level,
            stop_reason=stop_reason,
            intrabar_quotes=intrabar_quotes,
        )

    def _update_trailing_from_close(self, session: TradingSession, pos: Position, bar: BarData) -> None:
        self.exit_engine.update_trailing_from_close(session, pos, bar)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        return DayTradingEvidenceService.to_float(value, default)

    def _resolve_evidence_weight_context(self, flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return self.evidence_service.resolve_evidence_weight_context(flow_metrics)

    @staticmethod
    def _is_midday_window(bar_time: time) -> bool:
        return DayTradingEvidenceService.is_midday_window(bar_time)

    def _required_confirming_sources(self, session: TradingSession, bar_time: time) -> int:
        return self.evidence_service.required_confirming_sources(session, bar_time)

    def _is_mu_choppy_blocked(self, session: TradingSession, regime: Regime) -> bool:
        return self.evidence_service.is_mu_choppy_blocked(session, regime)

    @staticmethod
    def _signal_direction(signal: Signal) -> Optional[str]:
        return DayTradingEvidenceService.signal_direction(signal)

    def _aligned_evidence_source_keys(self, signal: Signal) -> List[str]:
        return self.evidence_service.aligned_evidence_source_keys(signal)

    def _confirming_source_stats(self, signal: Signal) -> Dict[str, Any]:
        return self.evidence_service.confirming_source_stats(signal)

    @staticmethod
    def _normalize_source_key(value: Any) -> Optional[str]:
        return DayTradingEvidenceService.normalize_source_key(value)

    def _extract_confirming_source_keys_from_metadata(
        self,
        signal_metadata: Dict[str, Any],
        side: str,
        strategy_name: str,
    ) -> List[str]:
        return self.evidence_service.extract_confirming_source_keys_from_metadata(
            signal_metadata=signal_metadata,
            side=side,
            strategy_name=strategy_name,
        )


    def _passes_momentum_flow_delta_confirmation(self, signal: Signal) -> tuple[bool, Dict[str, Any]]:
        return self.evidence_service.passes_momentum_flow_delta_confirmation(signal)

    def _passes_momentum_diversification_gate(
        self,
        session: TradingSession,
        signal: Signal,
        flow_metrics: Dict[str, Any],
    ) -> tuple[bool, Dict[str, Any]]:
        return self.gate_engine.passes_momentum_diversification_gate(
            session=session,
            signal=signal,
            flow_metrics=flow_metrics,
        )

    def _should_momentum_fail_fast_exit(
        self,
        session: TradingSession,
        pos: Position,
        current_bar_index: int,
    ) -> tuple[bool, Dict[str, Any]]:
        return self.gate_engine.should_momentum_fail_fast_exit(
            session=session,
            pos=pos,
            current_bar_index=current_bar_index,
        )

    def _extract_raw_confidence_from_metadata(self, signal_metadata: Dict[str, Any]) -> float:
        return self.evidence_service.extract_raw_confidence_from_metadata(signal_metadata)

    def _cross_asset_headwind_threshold_boost(
        self,
        cross_asset_state: Any,
        decision_direction: Optional[str],
    ) -> tuple[float, Dict[str, Any]]:
        return self.gate_engine.cross_asset_headwind_threshold_boost(
            cross_asset_state=cross_asset_state,
            decision_direction=decision_direction,
            activation_score=self.headwind_activation_score,
        )









    def _generate_signal(
        self, 
        session: TradingSession, 
        bar: BarData, 
        timestamp: datetime
    ) -> Optional[Signal]:
        return runtime_generate_signal(
            self=self,
            session=session,
            bar=bar,
            timestamp=timestamp,
        )


    def _update_intraday_levels(
        self,
        session: TradingSession,
        current_bar_index: int,
    ) -> Dict[str, Any]:
        state = update_intraday_levels_state(
            getattr(session, "intraday_levels_state", {}),
            bars=session.bars,
            current_bar_index=current_bar_index,
        )
        session.intraday_levels_state = state
        snapshot = state.get("snapshot")
        if isinstance(snapshot, dict) and snapshot:
            return snapshot
        built = build_intraday_levels_snapshot(state)
        state["snapshot"] = built
        return built

    def _get_intraday_levels_snapshot(self, session: TradingSession) -> Dict[str, Any]:
        state = ensure_intraday_levels_state(getattr(session, "intraday_levels_state", {}))
        session.intraday_levels_state = state
        snapshot = state.get("snapshot")
        if isinstance(snapshot, dict) and snapshot:
            return snapshot
        built = build_intraday_levels_snapshot(state)
        state["snapshot"] = built
        return built

    def _intraday_levels_indicator_payload(self, session: TradingSession) -> Dict[str, Any]:
        state = ensure_intraday_levels_state(getattr(session, "intraday_levels_state", {}))
        session.intraday_levels_state = state
        return intraday_levels_indicator_payload(state)

    def _persist_intraday_levels_memory(self, session: TradingSession) -> None:
        self.intraday_memory.persist_intraday_levels_memory(session)

    def _inject_intraday_levels_memory_into_session(self, session: TradingSession) -> None:
        self.intraday_memory.inject_intraday_levels_memory_into_session(session)

    @staticmethod
    def _intraday_memory_key(run_id: str, ticker: str) -> tuple[str, str]:
        return IntradayMemoryService._intraday_memory_key(run_id, ticker)

    def _calculate_indicators(
        self,
        bars: List[BarData],
        session: Optional[TradingSession] = None,
    ) -> Dict[str, Any]:
        return runtime_calculate_indicators(self=self, bars=bars, session=session)


    @staticmethod
    def _extract_confirming_sources(signal: Signal) -> Optional[int]:
        return TradeExecutionEngine.extract_confirming_sources(signal)

    @staticmethod
    def _agreement_risk_multiplier(confirming_sources: Optional[int]) -> float:
        return TradeExecutionEngine.agreement_risk_multiplier(confirming_sources)

    @staticmethod
    def _trailing_multiplier(confirming_sources: Optional[int]) -> float:
        return TradeExecutionEngine.trailing_multiplier(confirming_sources)

    def _effective_trailing_stop_pct(self, session: TradingSession, signal: Signal) -> float:
        return self.trade_engine.effective_trailing_stop_pct(session, signal)

    @staticmethod
    def _fixed_stop_price(entry_price: float, side: str, stop_loss_pct: float) -> float:
        return TradeExecutionEngine.fixed_stop_price(entry_price, side, stop_loss_pct)

    def _calculate_position_size(
        self,
        session: TradingSession,
        signal: Signal,
        entry_price: float,
    ) -> float:
        return self.trade_engine.calculate_position_size(session, signal, entry_price)


    def _simulate_entry_fill(
        self,
        desired_size: float,
        bar_volume: float,
        session: TradingSession,
    ) -> Tuple[float, float]:
        return self.trade_engine.simulate_entry_fill(desired_size, bar_volume, session)

    def _bars_held(self, pos: Position, current_bar_index: int) -> int:
        return self.trade_engine.bars_held(pos, current_bar_index)

    def _partial_take_profit_price(self, session: TradingSession, pos: Position) -> float:
        return self.trade_engine.partial_take_profit_price(session, pos)


    def _maybe_take_partial_profit(
        self,
        session: TradingSession,
        pos: Position,
        bar: BarData,
        timestamp: datetime,
    ) -> Optional[DayTrade]:
        return self.trade_engine.maybe_take_partial_profit(session, pos, bar, timestamp)

    def _should_time_exit(self, session: TradingSession, pos: Position, current_bar_index: int) -> bool:
        return self.trade_engine.should_time_exit(session, pos, current_bar_index)

    def _should_adverse_flow_exit(
        self,
        session: TradingSession,
        pos: Position,
        current_bar_index: int,
    ) -> Tuple[bool, Dict[str, float]]:
        return self.trade_engine.should_adverse_flow_exit(session, pos, current_bar_index)

    def _open_position(
        self,
        session: TradingSession,
        signal: Signal,
        entry_price: Optional[float] = None,
        entry_time: Optional[datetime] = None,
        signal_bar_index: Optional[int] = None,
        entry_bar_index: Optional[int] = None,
        entry_bar_volume: Optional[float] = None,
    ) -> Position:
        return self.trade_engine.open_position(session, signal, entry_price, entry_time, signal_bar_index, entry_bar_index, entry_bar_volume)

    def _close_position(
        self,
        session: TradingSession,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        bar_volume: Optional[float] = None,
    ) -> DayTrade:
        return self.trade_engine.close_position(session, exit_price, exit_time, reason, bar_volume)

    def _get_session_summary(self, session: TradingSession) -> Dict[str, Any]:
        """Get summary of trading session."""
        trades = session.trades
        
        if not trades:
            return {
                'ticker': session.ticker,
                'date': session.date,
                'regime': session.detected_regime.value if session.detected_regime else None,
                'micro_regime': session.micro_regime,
                'strategy': session.selected_strategy,
                'selection_warnings': list(session.selection_warnings),
                'total_trades': 0,
                'total_pnl_pct': 0,
                'success': False
            }
        
        winning = [t for t in trades if t.pnl_pct > 0]
        losing = [t for t in trades if t.pnl_pct <= 0]
        
        return {
            'ticker': session.ticker,
            'date': session.date,
            'run_id': session.run_id,
            'regime': session.detected_regime.value if session.detected_regime else None,
            'micro_regime': session.micro_regime,
            'strategy': session.selected_strategy,
            'selection_warnings': list(session.selection_warnings),
            'total_trades': len(trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(trades) * 100 if trades else 0,
            'trades': [t.to_dict() for t in trades],
            'total_pnl_pct': round(session.total_pnl, 2),
            'avg_pnl_pct': round(session.total_pnl / len(trades), 2) if trades else 0,
            'best_trade': round(max(t.pnl_pct for t in trades), 2) if trades else 0,
            'worst_trade': round(min(t.pnl_pct for t in trades), 2) if trades else 0,
            'bars_processed': len(session.bars),
            'pre_market_bars': len(session.pre_market_bars),
            'regime_history': list(session.regime_history),
            'success': session.total_pnl > 0,
        }
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions."""
        return {k: v.to_dict() for k, v in self.sessions.items()}
    
    def end_session(self, run_id: str, ticker: str, date: str) -> Optional[Dict[str, Any]]:
        """Manually end a session and get summary."""
        session = self.get_session(run_id, ticker, date)
        
        if not session:
            return None
        
        # Close any open position at last price
        if session.active_position and session.bars:
            last_price = session.bars[-1].close
            last_time = session.bars[-1].timestamp
            self._close_position(
                session,
                last_price,
                last_time,
                'manual_close',
                bar_volume=session.bars[-1].volume,
            )
        
        session.phase = SessionPhase.CLOSED
        if session.bars:
            session.end_price = session.bars[-1].close
        self._persist_intraday_levels_memory(session)
        
        return self._get_session_summary(session)
    
    def clear_session(self, run_id: str, ticker: str, date: str) -> bool:
        """Clear a session from memory."""
        key = self._get_session_key(run_id, ticker, date)
        if key in self.sessions:
            del self.sessions[key]
            self.last_trade_bar_index.pop(key, None)
            defaults_key = (run_id, ticker)
            memory_key = self._intraday_memory_key(run_id, ticker)
            # Remove run defaults only when no other sessions for this run/ticker remain.
            has_other_sessions = any(
                s.run_id == run_id and s.ticker == ticker for s in self.sessions.values()
            )
            if not has_other_sessions:
                self.run_defaults.pop(defaults_key, None)
                self.intraday_memory.day_memory.pop(memory_key, None)
                self.intraday_memory.export_marker = {
                    marker_key: marker_val
                    for marker_key, marker_val in self.intraday_memory.export_marker.items()
                    if not (
                        marker_key[0] == str(run_id)
                        and marker_key[1] == str(ticker).upper()
                    )
                }
            has_other_run_sessions = any(
                s.run_id == run_id for s in self.sessions.values()
            )
            if not has_other_run_sessions:
                self.run_equity_state.pop(run_id, None)
            return True
        return False

    def clear_sessions_for_run(self, run_id: str, ticker: Optional[str] = None) -> int:
        """Clear all sessions (and sticky per-run state) for a run_id, optionally by ticker."""
        normalized_ticker = str(ticker).upper() if ticker else None

        keys_to_clear = [
            key
            for key, session in self.sessions.items()
            if session.run_id == run_id
            and (normalized_ticker is None or session.ticker == normalized_ticker)
        ]

        for key in keys_to_clear:
            self.sessions.pop(key, None)
            self.last_trade_bar_index.pop(key, None)

        defaults_to_clear = [
            key
            for key in self.run_defaults.keys()
            if key[0] == run_id
            and (normalized_ticker is None or key[1] == normalized_ticker)
        ]
        for key in defaults_to_clear:
            self.run_defaults.pop(key, None)

        memory_to_clear = [
            key
            for key in self.intraday_memory.day_memory.keys()
            if key[0] == str(run_id)
            and (normalized_ticker is None or key[1] == normalized_ticker)
        ]
        for key in memory_to_clear:
            self.intraday_memory.day_memory.pop(key, None)
        self.intraday_memory.export_marker = {
            key: value
            for key, value in self.intraday_memory.export_marker.items()
            if not (
                key[0] == str(run_id)
                and (normalized_ticker is None or key[1] == normalized_ticker)
            )
        }

        has_other_run_sessions = any(
            session.run_id == run_id for session in self.sessions.values()
        )
        if not has_other_run_sessions:
            self.run_equity_state.pop(run_id, None)

        return len(keys_to_clear)

    def reset_backtest_state(self, scope: str = "all", clear_sessions: bool = True) -> Dict[str, Any]:
        """
        Reset manager/orchestrator state for deterministic backtest reruns.

        scope:
          - "session": reset per-session pipeline only
          - "learning": reset learned trade-memory only
          - "all": reset both (default)
        """
        normalized_scope = str(scope or "all").strip().lower()
        if normalized_scope not in {"session", "learning", "all"}:
            raise ValueError(f"Unsupported reset scope: {scope}")

        sessions_before = len(self.sessions)
        defaults_before = len(self.run_defaults)

        if clear_sessions:
            self.sessions.clear()
            self.last_trade_bar_index.clear()
            self.run_defaults.clear()
            self.run_equity_state.clear()
            self.intraday_memory.day_memory.clear()
            self.intraday_memory.export_marker.clear()

        orch = self.orchestrator
        if orch:
            if normalized_scope == "all":
                orch.full_reset()
            elif normalized_scope == "session":
                orch.new_session()
            elif normalized_scope == "learning":
                orch.reset_learning_state()

        return {
            "scope": normalized_scope,
            "sessions_before": sessions_before,
            "sessions_after": len(self.sessions),
            "run_defaults_before": defaults_before,
            "run_defaults_after": len(self.run_defaults),
            "clear_sessions": bool(clear_sessions),
            "orchestrator_present": orch is not None,
        }
