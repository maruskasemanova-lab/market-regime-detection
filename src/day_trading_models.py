"""Core day-trading models used by DayTradingManager."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional

from .exit_policy_engine import ContextAwareExitPolicy
from .position_context import PositionContextMonitor
from .strategies.base_strategy import Position, Regime, Signal
from .trading_config import TradingConfig
from .trading_orchestrator import TradingOrchestrator


class SessionPhase(Enum):
    """Trading session phases."""

    PRE_MARKET = "PRE_MARKET"           # Before 9:30 ET
    REGIME_DETECTION = "REGIME_DETECTION"  # First X minutes
    TRADING = "TRADING"                  # Active trading
    END_OF_DAY = "END_OF_DAY"           # Closing positions
    CLOSED = "CLOSED"                    # Session ended


@dataclass
class BarData:
    """Single bar data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    l2_delta: Optional[float] = None
    l2_buy_volume: Optional[float] = None
    l2_sell_volume: Optional[float] = None
    l2_volume: Optional[float] = None
    l2_imbalance: Optional[float] = None
    l2_bid_depth_total: Optional[float] = None
    l2_ask_depth_total: Optional[float] = None
    l2_book_pressure: Optional[float] = None
    l2_book_pressure_change: Optional[float] = None
    l2_iceberg_buy_count: Optional[float] = None
    l2_iceberg_sell_count: Optional[float] = None
    l2_iceberg_bias: Optional[float] = None
    l2_quality_flags: Optional[List[str]] = None
    l2_quality: Optional[Dict[str, Any]] = None
    # Optional 1-second top-of-book quote snapshots for this minute.
    intrabar_quotes_1s: Optional[List[Dict[str, float]]] = None
    # TCBBO options flow fields (pre-computed, attached by backtest-runner)
    tcbbo_net_premium: Optional[float] = None
    tcbbo_cumulative_net_premium: Optional[float] = None
    tcbbo_call_buy_premium: Optional[float] = None
    tcbbo_put_buy_premium: Optional[float] = None
    tcbbo_call_sell_premium: Optional[float] = None
    tcbbo_put_sell_premium: Optional[float] = None
    tcbbo_sweep_count: Optional[int] = None
    tcbbo_sweep_premium: Optional[float] = None
    tcbbo_trade_count: Optional[int] = None
    tcbbo_has_data: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'l2_delta': self.l2_delta,
            'l2_buy_volume': self.l2_buy_volume,
            'l2_sell_volume': self.l2_sell_volume,
            'l2_volume': self.l2_volume,
            'l2_imbalance': self.l2_imbalance,
            'l2_bid_depth_total': self.l2_bid_depth_total,
            'l2_ask_depth_total': self.l2_ask_depth_total,
            'l2_book_pressure': self.l2_book_pressure,
            'l2_book_pressure_change': self.l2_book_pressure_change,
            'l2_iceberg_buy_count': self.l2_iceberg_buy_count,
            'l2_iceberg_sell_count': self.l2_iceberg_sell_count,
            'l2_iceberg_bias': self.l2_iceberg_bias,
            'l2_quality_flags': self.l2_quality_flags,
            'l2_quality': self.l2_quality,
            'tcbbo_net_premium': self.tcbbo_net_premium,
            'tcbbo_cumulative_net_premium': self.tcbbo_cumulative_net_premium,
            'tcbbo_sweep_count': self.tcbbo_sweep_count,
            'tcbbo_sweep_premium': self.tcbbo_sweep_premium,
            'tcbbo_has_data': self.tcbbo_has_data,
        }


@dataclass
class DayTrade:
    """Completed trade for the day."""

    id: int
    strategy: str
    side: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    size: float
    pnl_pct: float
    pnl_dollars: float
    exit_reason: str
    # Trading costs breakdown
    slippage: float = 0.0
    commission: float = 0.0
    reg_fee: float = 0.0
    sec_fee: float = 0.0
    finra_fee: float = 0.0
    market_impact: float = 0.0
    total_costs: float = 0.0
    gross_pnl_pct: float = 0.0  # PnL before costs
    signal_bar_index: Optional[int] = None
    entry_bar_index: Optional[int] = None
    signal_timestamp: Optional[str] = None
    signal_price: Optional[float] = None
    signal_metadata: Dict[str, Any] = field(default_factory=dict)
    flow_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'strategy': self.strategy,
            'side': self.side,
            'entry_price': round(self.entry_price, 2),
            'entry_time': self.entry_time.isoformat(),
            'exit_price': round(self.exit_price, 2),
            'exit_time': self.exit_time.isoformat(),
            'size': self.size,
            'pnl_pct': round(self.pnl_pct, 2),
            'pnl_dollars': round(self.pnl_dollars, 2),
            'exit_reason': self.exit_reason,
            'costs': {
                'slippage': round(self.slippage, 4),
                'commission': round(self.commission, 4),
                'reg_fee': round(self.reg_fee, 4),
                'sec_fee': round(self.sec_fee, 6),
                'finra_fee': round(self.finra_fee, 6),
                'market_impact': round(self.market_impact, 4),
                'total': round(self.total_costs, 4),
            },
            'gross_pnl_pct': round(self.gross_pnl_pct, 2),
            'signal_bar_index': self.signal_bar_index,
            'entry_bar_index': self.entry_bar_index,
            'signal_timestamp': self.signal_timestamp,
            'signal_price': round(self.signal_price, 4) if self.signal_price is not None else None,
            'signal_metadata': self.signal_metadata,
            'flow_snapshot': self.flow_snapshot,
        }


@dataclass
class TradingCosts:
    """Trading costs (IBKR Tiered, calibrated on MU live scalp fills, Feb 2026).

    Model:
      - Commission: ~$0.0037/share with ~$0.15 minimum per order (per side).
      - Slippage: ~1.18¢ per share round-trip (0.59¢ per side through spread/queue).
      - FINRA TAF: SELL side only, min ~$0.0084, cap $9.79.
      - SEC fee: optional SELL-side notional fee (default 0).
      - reg_fee: optional pass-through per share (default 0).
    """

    # Calibrated to MU IBKR live scalp dataset (2026-01-27..2026-02-10).
    commission_per_share: float = 0.0037
    commission_min: float = 0.15      # per order (per side)
    # 0.59¢ per side so round-trip slippage totals ~1.18¢ per share.
    slippage_per_share: float = 0.0059
    # Slippage and impact realism controls.
    low_volume_threshold_shares: float = 25_000.0
    low_volume_slippage_multiplier: float = 1.75
    impact_coeff_bps: float = 6.0
    impact_participation_cap: float = 0.20
    finra_fee_per_share: float = 0.00018
    finra_fee_cap: float = 9.79
    finra_fee_min: float = 0.0084
    sec_fee_per_dollar: float = 0.0   # SELL-side notional, default 0
    reg_fee_per_share: float = 0.0    # optional pass-through, SELL side only

    def calculate_costs(
        self,
        entry_price: float,
        exit_price: float,
        shares: float,
        side: str,
        avg_bar_volume: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate all trading costs."""
        # Estimate participation to model liquidity-dependent slippage/impact.
        avg_volume = float(avg_bar_volume or 0.0)
        participation_ratio = shares / avg_volume if avg_volume > 0 else 0.0

        volume_multiplier = 1.0
        if avg_volume > 0 and avg_volume < self.low_volume_threshold_shares:
            shortage = (self.low_volume_threshold_shares - avg_volume) / self.low_volume_threshold_shares
            volume_multiplier += shortage * max(0.0, self.low_volume_slippage_multiplier - 1.0)

        participation_multiplier = 1.0 + max(0.0, participation_ratio - 0.02) * 6.0
        dynamic_slippage_per_share = self.slippage_per_share * volume_multiplier * participation_multiplier
        slippage_cost = dynamic_slippage_per_share * shares * 2

        # Commission per side with minimum
        entry_commission = max(self.commission_per_share * shares, self.commission_min)
        exit_commission = max(self.commission_per_share * shares, self.commission_min)
        commission = entry_commission + exit_commission

        # SELL-side only fees (long -> sell on exit, short -> sell on entry)
        sell_price = exit_price if side == "long" else entry_price
        sell_notional = sell_price * shares

        # FINRA TAF: per share, apply cap and minimum (SELL side only)
        raw_finra = shares * self.finra_fee_per_share
        finra_fee = max(self.finra_fee_min, min(raw_finra, self.finra_fee_cap))

        # SEC fee (SELL side only; default 0)
        sec_fee = sell_notional * self.sec_fee_per_dollar

        # Optional pass-through reg fee (SELL side only; default 0)
        reg_fee = shares * self.reg_fee_per_share

        impact_ratio = min(
            1.0,
            participation_ratio / max(1e-6, float(self.impact_participation_cap)),
        )
        avg_notional = ((entry_price + exit_price) / 2.0) * shares
        market_impact = avg_notional * ((self.impact_coeff_bps * impact_ratio) / 10_000.0)

        total = slippage_cost + commission + reg_fee + finra_fee + sec_fee + market_impact

        return {
            'slippage': slippage_cost,
            'commission': commission,
            'reg_fee': reg_fee,
            'sec_fee': sec_fee,
            'finra_fee': finra_fee,
            'market_impact': market_impact,
            'dynamic_slippage_per_share': dynamic_slippage_per_share,
            'participation_ratio': participation_ratio,
            'total': total,
        }


@dataclass
class TradingSession:
    """Single day trading session for a ticker."""

    run_id: str
    ticker: str
    date: str
    regime_detection_minutes: int = 15
    regime_refresh_bars: int = 30

    # Session state
    phase: SessionPhase = SessionPhase.PRE_MARKET
    detected_regime: Optional[Regime] = None
    micro_regime: str = "MIXED"
    active_strategies: List[str] = field(default_factory=list)
    selected_strategy: Optional[str] = None  # Helper for backwards compat logic
    selection_warnings: List[str] = field(default_factory=list)
    strategy_selection_mode: str = "adaptive_top_n"  # adaptive_top_n | all_enabled
    max_active_strategies: int = 3
    momentum_diversification: Dict[str, Any] = field(default_factory=dict)
    momentum_diversification_override: bool = False
    premarket_trading_enabled: bool = False

    # Data storage
    bars: List[BarData] = field(default_factory=list)
    pre_market_bars: List[BarData] = field(default_factory=list)
    intraday_levels_state: Dict[str, Any] = field(default_factory=dict)

    # Trading state
    active_position: Optional[Position] = None
    trades: List[DayTrade] = field(default_factory=list)
    signals: List[Signal] = field(default_factory=list)
    trade_counter: int = 0
    pending_signal: Optional[Signal] = None
    pending_signal_bar_index: int = -1
    potential_sweep_active: bool = False
    potential_sweep_context: Dict[str, Any] = field(default_factory=dict)
    golden_setup_result: Optional[Dict[str, Any]] = None
    golden_setup_entries_today: int = 0
    golden_setup_last_entry_bar_index: int = -99
    consecutive_losses: int = 0
    loss_cooldown_until_bar_index: int = -1
    last_exit_bar_index: int = -1
    last_regime_refresh_bar_index: int = -1
    last_strategy_switch_bar_index: int = -1
    regime_history: List[Dict[str, Any]] = field(default_factory=list)
    orchestrator: Optional[TradingOrchestrator] = None
    _position_context: Optional[PositionContextMonitor] = field(default=None, repr=False)
    _context_exit_policy: Optional[ContextAwareExitPolicy] = field(default=None, repr=False)

    # Market hours (ET)
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    pre_market_start: time = field(default_factory=lambda: time(4, 0))

    # Trailing stop config
    trailing_stop_pct: float = 0.8
    trailing_activation_pct: float = 0.15  # Reduced from 0.30 for faster break-even
    break_even_buffer_pct: float = 0.03    # Tightened from 0.05
    break_even_min_hold_bars: int = 3
    break_even_activation_min_mfe_pct: float = 0.25
    break_even_activation_min_r: float = 0.60
    break_even_activation_min_r_trending_5m: float = 0.90
    break_even_activation_min_r_choppy_5m: float = 0.60
    break_even_activation_use_levels: bool = True
    break_even_activation_use_l2: bool = True
    break_even_level_buffer_pct: float = 0.02
    break_even_level_max_distance_pct: float = 0.60
    break_even_level_min_confluence: int = 2
    break_even_level_min_tests: int = 1
    break_even_l2_signed_aggression_min: float = 0.12
    break_even_l2_imbalance_min: float = 0.15
    break_even_l2_book_pressure_min: float = 0.10
    break_even_l2_spread_bps_max: float = 12.0
    break_even_costs_pct: float = 0.03
    break_even_min_buffer_pct: float = 0.05
    break_even_atr_buffer_k: float = 0.10
    break_even_5m_atr_buffer_k: float = 0.10
    break_even_tick_size: float = 0.01
    break_even_min_tick_buffer: int = 1
    break_even_anti_spike_bars: int = 1
    break_even_anti_spike_hits_required: int = 2
    break_even_anti_spike_require_close_beyond: bool = True
    break_even_5m_no_go_proximity_pct: float = 0.10
    break_even_5m_mfe_atr_factor: float = 0.15
    break_even_5m_l2_bias_threshold: float = 0.10
    break_even_5m_l2_bias_tighten_factor: float = 0.85
    break_even_movement_formula_enabled: bool = False
    break_even_movement_formula: str = ""
    break_even_proof_formula_enabled: bool = False
    break_even_proof_formula: str = ""
    break_even_activation_formula_enabled: bool = False
    break_even_activation_formula: str = ""
    break_even_trailing_handoff_formula_enabled: bool = False
    break_even_trailing_handoff_formula: str = ""
    trailing_enabled_in_choppy: bool = False  # Disable trailing in CHOPPY regime
    choppy_time_exit_bars: int = 12         # Shorter time-stop in CHOPPY regime

    # Session results
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    total_pnl: float = 0.0
    regime_start_ts: Optional[datetime] = None
    account_size_usd: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    max_position_notional_pct: float = 100.0
    min_position_size: float = 1.0
    max_fill_participation_rate: float = 0.20
    min_fill_ratio: float = 0.35
    enable_partial_take_profit: bool = True
    partial_take_profit_rr: float = 1.0
    partial_take_profit_fraction: float = 0.5
    partial_flow_deterioration_min_r: float = 0.5
    partial_flow_deterioration_skip_be: bool = True
    partial_protect_min_mfe_r: float = 0.0
    time_exit_bars: int = 40
    time_exit_formula_enabled: bool = False
    time_exit_formula: str = ""
    adverse_flow_exit_enabled: bool = True
    adverse_flow_threshold: float = 0.20
    adverse_flow_min_hold_bars: int = 6
    adverse_flow_consistency_threshold: float = 0.45
    adverse_book_pressure_threshold: float = 0.15
    adverse_flow_exit_formula_enabled: bool = False
    adverse_flow_exit_formula: str = ""
    stop_loss_mode: str = "strategy"  # strategy | fixed | capped
    fixed_stop_loss_pct: float = 0.0
    l2_confirm_enabled: bool = False
    l2_min_delta: float = 0.0
    l2_min_imbalance: float = 0.0
    l2_min_iceberg_bias: float = 0.0
    l2_lookback_bars: int = 3
    l2_min_participation_ratio: float = 0.0
    l2_min_directional_consistency: float = 0.0
    l2_min_signed_aggression: float = 0.0
    l2_gate_mode: str = "weighted"  # "weighted" | "all_pass"
    l2_gate_threshold: float = 0.50
    tcbbo_gate_enabled: bool = False
    tcbbo_min_net_premium: float = 0.0
    tcbbo_sweep_boost: float = 5.0
    tcbbo_lookback_bars: int = 5
    config: TradingConfig = field(default_factory=TradingConfig)
    max_daily_trades_override: Optional[int] = None
    mu_choppy_hard_block_enabled_override: Optional[bool] = None

    def apply_trading_config(self, config: TradingConfig) -> None:
        """Apply canonical runtime config to mutable session fields."""
        self.config = config
        self.regime_detection_minutes = config.regime_detection_minutes
        self.regime_refresh_bars = config.regime_refresh_bars
        self.account_size_usd = config.account_size_usd
        self.risk_per_trade_pct = config.risk_per_trade_pct
        self.max_position_notional_pct = config.max_position_notional_pct
        self.max_fill_participation_rate = config.max_fill_participation_rate
        self.min_fill_ratio = config.min_fill_ratio
        self.enable_partial_take_profit = config.enable_partial_take_profit
        self.partial_take_profit_rr = config.partial_take_profit_rr
        self.partial_take_profit_fraction = config.partial_take_profit_fraction
        self.partial_flow_deterioration_min_r = config.partial_flow_deterioration_min_r
        self.partial_flow_deterioration_skip_be = config.partial_flow_deterioration_skip_be
        self.partial_protect_min_mfe_r = config.partial_protect_min_mfe_r
        self.trailing_activation_pct = config.trailing_activation_pct
        self.break_even_buffer_pct = config.break_even_buffer_pct
        self.break_even_min_hold_bars = config.break_even_min_hold_bars
        self.break_even_activation_min_mfe_pct = config.break_even_activation_min_mfe_pct
        self.break_even_activation_min_r = config.break_even_activation_min_r
        self.break_even_activation_min_r_trending_5m = config.break_even_activation_min_r_trending_5m
        self.break_even_activation_min_r_choppy_5m = config.break_even_activation_min_r_choppy_5m
        self.break_even_activation_use_levels = config.break_even_activation_use_levels
        self.break_even_activation_use_l2 = config.break_even_activation_use_l2
        self.break_even_level_buffer_pct = config.break_even_level_buffer_pct
        self.break_even_level_max_distance_pct = config.break_even_level_max_distance_pct
        self.break_even_level_min_confluence = config.break_even_level_min_confluence
        self.break_even_level_min_tests = config.break_even_level_min_tests
        self.break_even_l2_signed_aggression_min = config.break_even_l2_signed_aggression_min
        self.break_even_l2_imbalance_min = config.break_even_l2_imbalance_min
        self.break_even_l2_book_pressure_min = config.break_even_l2_book_pressure_min
        self.break_even_l2_spread_bps_max = config.break_even_l2_spread_bps_max
        self.break_even_costs_pct = config.break_even_costs_pct
        self.break_even_min_buffer_pct = config.break_even_min_buffer_pct
        self.break_even_atr_buffer_k = config.break_even_atr_buffer_k
        self.break_even_5m_atr_buffer_k = config.break_even_5m_atr_buffer_k
        self.break_even_tick_size = config.break_even_tick_size
        self.break_even_min_tick_buffer = config.break_even_min_tick_buffer
        self.break_even_anti_spike_bars = config.break_even_anti_spike_bars
        self.break_even_anti_spike_hits_required = config.break_even_anti_spike_hits_required
        self.break_even_anti_spike_require_close_beyond = (
            config.break_even_anti_spike_require_close_beyond
        )
        self.break_even_5m_no_go_proximity_pct = config.break_even_5m_no_go_proximity_pct
        self.break_even_5m_mfe_atr_factor = config.break_even_5m_mfe_atr_factor
        self.break_even_5m_l2_bias_threshold = config.break_even_5m_l2_bias_threshold
        self.break_even_5m_l2_bias_tighten_factor = config.break_even_5m_l2_bias_tighten_factor
        self.break_even_movement_formula_enabled = config.break_even_movement_formula_enabled
        self.break_even_movement_formula = config.break_even_movement_formula
        self.break_even_proof_formula_enabled = config.break_even_proof_formula_enabled
        self.break_even_proof_formula = config.break_even_proof_formula
        self.break_even_activation_formula_enabled = config.break_even_activation_formula_enabled
        self.break_even_activation_formula = config.break_even_activation_formula
        self.break_even_trailing_handoff_formula_enabled = (
            config.break_even_trailing_handoff_formula_enabled
        )
        self.break_even_trailing_handoff_formula = config.break_even_trailing_handoff_formula
        self.trailing_enabled_in_choppy = config.trailing_enabled_in_choppy
        self.time_exit_bars = config.time_exit_bars
        self.time_exit_formula_enabled = config.time_exit_formula_enabled
        self.time_exit_formula = config.time_exit_formula
        self.adverse_flow_exit_enabled = config.adverse_flow_exit_enabled
        self.adverse_flow_threshold = config.adverse_flow_threshold
        self.adverse_flow_min_hold_bars = config.adverse_flow_min_hold_bars
        self.adverse_flow_consistency_threshold = config.adverse_flow_consistency_threshold
        self.adverse_book_pressure_threshold = config.adverse_book_pressure_threshold
        self.adverse_flow_exit_formula_enabled = config.adverse_flow_exit_formula_enabled
        self.adverse_flow_exit_formula = config.adverse_flow_exit_formula
        self.stop_loss_mode = config.stop_loss_mode
        self.fixed_stop_loss_pct = config.fixed_stop_loss_pct
        self.l2_confirm_enabled = config.l2_confirm_enabled
        self.l2_min_delta = config.l2_min_delta
        self.l2_min_imbalance = config.l2_min_imbalance
        self.l2_min_iceberg_bias = config.l2_min_iceberg_bias
        self.l2_lookback_bars = config.l2_lookback_bars
        self.l2_min_participation_ratio = config.l2_min_participation_ratio
        self.l2_min_directional_consistency = config.l2_min_directional_consistency
        self.l2_min_signed_aggression = config.l2_min_signed_aggression
        self.l2_gate_mode = config.l2_gate_mode
        self.l2_gate_threshold = config.l2_gate_threshold
        self.tcbbo_gate_enabled = config.tcbbo_gate_enabled
        self.tcbbo_min_net_premium = config.tcbbo_min_net_premium
        self.tcbbo_sweep_boost = config.tcbbo_sweep_boost
        self.tcbbo_lookback_bars = config.tcbbo_lookback_bars
        self.strategy_selection_mode = config.strategy_selection_mode
        self.max_active_strategies = config.max_active_strategies
        self.potential_sweep_active = False
        self.potential_sweep_context = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'ticker': self.ticker,
            'date': self.date,
            'phase': self.phase.value,
            'regime': self.detected_regime.value if self.detected_regime else None,
            'micro_regime': self.micro_regime,
            'selected_strategy': self.selected_strategy,
            'active_strategies': list(self.active_strategies),
            'selection_warnings': list(self.selection_warnings),
            'strategy_selection_mode': self.strategy_selection_mode,
            'max_active_strategies': self.max_active_strategies,
            'momentum_diversification': dict(self.momentum_diversification),
            'momentum_diversification_override': bool(self.momentum_diversification_override),
            'last_strategy_switch_bar_index': self.last_strategy_switch_bar_index,
            'bars_count': len(self.bars),
            'pre_market_bars_count': len(self.pre_market_bars),
            'intraday_levels': (
                dict(self.intraday_levels_state.get('snapshot', {}))
                if isinstance(self.intraday_levels_state, dict)
                else {}
            ),
            'active_position': self.active_position.to_dict() if self.active_position else None,
            'has_pending_signal': self.pending_signal is not None,
            'potential_sweep_active': self.potential_sweep_active,
            'potential_sweep_context': dict(self.potential_sweep_context),
            'golden_setup': (
                dict(self.golden_setup_result)
                if isinstance(self.golden_setup_result, dict)
                else None
            ),
            'golden_setup_entries_today': int(self.golden_setup_entries_today),
            'golden_setup_last_entry_bar_index': int(self.golden_setup_last_entry_bar_index),
            'consecutive_losses': self.consecutive_losses,
            'loss_cooldown_until_bar_index': self.loss_cooldown_until_bar_index,
            'trades_count': len(self.trades),
            'signals_count': len(self.signals),
            'total_pnl': round(self.total_pnl, 2),
            'start_price': self.start_price,
            'end_price': self.end_price,
            'l2_confirm_enabled': self.l2_confirm_enabled,
            'l2_min_delta': self.l2_min_delta,
            'l2_min_imbalance': self.l2_min_imbalance,
            'l2_min_iceberg_bias': self.l2_min_iceberg_bias,
            'l2_lookback_bars': self.l2_lookback_bars,
            'l2_min_participation_ratio': self.l2_min_participation_ratio,
            'l2_min_directional_consistency': self.l2_min_directional_consistency,
            'l2_min_signed_aggression': self.l2_min_signed_aggression,
            'tcbbo_gate_enabled': self.tcbbo_gate_enabled,
            'tcbbo_min_net_premium': self.tcbbo_min_net_premium,
            'tcbbo_sweep_boost': self.tcbbo_sweep_boost,
            'tcbbo_lookback_bars': self.tcbbo_lookback_bars,
            'max_daily_trades_override': self.max_daily_trades_override,
            'mu_choppy_hard_block_enabled_override': self.mu_choppy_hard_block_enabled_override,
            'orchestrator_active': self.orchestrator is not None,
            'regime_refresh_bars': self.regime_refresh_bars,
            'regime_history': list(self.regime_history),
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'max_position_notional_pct': self.max_position_notional_pct,
            'min_position_size': self.min_position_size,
            'max_fill_participation_rate': self.max_fill_participation_rate,
            'min_fill_ratio': self.min_fill_ratio,
            'enable_partial_take_profit': self.enable_partial_take_profit,
            'partial_take_profit_rr': self.partial_take_profit_rr,
            'partial_take_profit_fraction': self.partial_take_profit_fraction,
            'trailing_activation_pct': self.trailing_activation_pct,
            'break_even_buffer_pct': self.break_even_buffer_pct,
            'break_even_min_hold_bars': self.break_even_min_hold_bars,
            'break_even_activation_min_mfe_pct': self.break_even_activation_min_mfe_pct,
            'break_even_activation_min_r': self.break_even_activation_min_r,
            'break_even_activation_min_r_trending_5m': self.break_even_activation_min_r_trending_5m,
            'break_even_activation_min_r_choppy_5m': self.break_even_activation_min_r_choppy_5m,
            'break_even_activation_use_levels': self.break_even_activation_use_levels,
            'break_even_activation_use_l2': self.break_even_activation_use_l2,
            'break_even_level_buffer_pct': self.break_even_level_buffer_pct,
            'break_even_level_max_distance_pct': self.break_even_level_max_distance_pct,
            'break_even_level_min_confluence': self.break_even_level_min_confluence,
            'break_even_level_min_tests': self.break_even_level_min_tests,
            'break_even_l2_signed_aggression_min': self.break_even_l2_signed_aggression_min,
            'break_even_l2_imbalance_min': self.break_even_l2_imbalance_min,
            'break_even_l2_book_pressure_min': self.break_even_l2_book_pressure_min,
            'break_even_l2_spread_bps_max': self.break_even_l2_spread_bps_max,
            'break_even_costs_pct': self.break_even_costs_pct,
            'break_even_min_buffer_pct': self.break_even_min_buffer_pct,
            'break_even_atr_buffer_k': self.break_even_atr_buffer_k,
            'break_even_5m_atr_buffer_k': self.break_even_5m_atr_buffer_k,
            'break_even_tick_size': self.break_even_tick_size,
            'break_even_min_tick_buffer': self.break_even_min_tick_buffer,
            'break_even_anti_spike_bars': self.break_even_anti_spike_bars,
            'break_even_anti_spike_hits_required': self.break_even_anti_spike_hits_required,
            'break_even_anti_spike_require_close_beyond': self.break_even_anti_spike_require_close_beyond,
            'break_even_5m_no_go_proximity_pct': self.break_even_5m_no_go_proximity_pct,
            'break_even_5m_mfe_atr_factor': self.break_even_5m_mfe_atr_factor,
            'break_even_5m_l2_bias_threshold': self.break_even_5m_l2_bias_threshold,
            'break_even_5m_l2_bias_tighten_factor': self.break_even_5m_l2_bias_tighten_factor,
            'break_even_movement_formula_enabled': self.break_even_movement_formula_enabled,
            'break_even_movement_formula': self.break_even_movement_formula,
            'break_even_proof_formula_enabled': self.break_even_proof_formula_enabled,
            'break_even_proof_formula': self.break_even_proof_formula,
            'break_even_activation_formula_enabled': self.break_even_activation_formula_enabled,
            'break_even_activation_formula': self.break_even_activation_formula,
            'break_even_trailing_handoff_formula_enabled': (
                self.break_even_trailing_handoff_formula_enabled
            ),
            'break_even_trailing_handoff_formula': self.break_even_trailing_handoff_formula,
            'trailing_enabled_in_choppy': self.trailing_enabled_in_choppy,
            'time_exit_bars': self.time_exit_bars,
            'time_exit_formula_enabled': self.time_exit_formula_enabled,
            'time_exit_formula': self.time_exit_formula,
            'adverse_flow_exit_enabled': self.adverse_flow_exit_enabled,
            'adverse_flow_threshold': self.adverse_flow_threshold,
            'adverse_flow_min_hold_bars': self.adverse_flow_min_hold_bars,
            'adverse_flow_consistency_threshold': self.adverse_flow_consistency_threshold,
            'adverse_book_pressure_threshold': self.adverse_book_pressure_threshold,
            'adverse_flow_exit_formula_enabled': self.adverse_flow_exit_formula_enabled,
            'adverse_flow_exit_formula': self.adverse_flow_exit_formula,
            'stop_loss_mode': self.stop_loss_mode,
            'fixed_stop_loss_pct': self.fixed_stop_loss_pct,
            'trading_config': self.config.to_session_params(),
        }


__all__ = [
    "BarData",
    "DayTrade",
    "SessionPhase",
    "TradingConfig",
    "TradingCosts",
    "TradingSession",
]
