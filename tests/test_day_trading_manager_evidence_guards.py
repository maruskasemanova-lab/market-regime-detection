from typing import Any, Dict, List, Optional, Tuple
"""Tests for evidence guardrails in DayTradingManager."""
from datetime import datetime, time, timedelta, timezone

from src.cross_asset import CrossAssetState
from src.day_trading_manager import BarData, DayTradingManager, SessionPhase
from src.multi_layer_decision import DecisionResult
from src.strategies.base_strategy import Position, Regime, Signal, SignalType
from src.trading_config import TradingConfig


def _signal(
    strategy_name: str = "MomentumFlow",
    signal_type: SignalType = SignalType.BUY,
    metadata: Optional[dict] = None,
) -> Signal:
    return Signal(
        strategy_name=strategy_name,
        signal_type=signal_type,
        price=100.0,
        timestamp=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        confidence=80.0,
        stop_loss=99.0,
        take_profit=103.0,
        trailing_stop=False,
        reasoning="unit-test",
        metadata=metadata or {},
    )


def _l2_window(
    *,
    start: datetime,
    direction: float = 1.0,
    close_start: float = 100.0,
) -> List[BarData]:
    bars: List[BarData] = []
    for idx in range(3):
        ts = start + timedelta(minutes=idx)
        close = close_start + (direction * 0.1 * idx)
        bars.append(
            BarData(
                timestamp=ts,
                open=close - 0.05,
                high=close + 0.10,
                low=close - 0.10,
                close=close,
                volume=100_000.0,
                vwap=close,
                l2_delta=direction * (1200.0 + (idx * 100.0)),
                l2_volume=60_000.0,
                l2_imbalance=direction * 0.12,
                l2_iceberg_bias=direction * 0.08,
                l2_book_pressure=direction * 0.05,
            )
        )
    return bars


def test_required_confirming_sources_is_dynamic() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03", 100.0)

    session.detected_regime = Regime.TRENDING
    session.micro_regime = "TRENDING_UP"
    assert manager._required_confirming_sources(session, time(9, 45)) == 2
    assert manager._required_confirming_sources(session, time(11, 0)) == 3

    session.detected_regime = Regime.CHOPPY
    session.micro_regime = "CHOPPY"
    assert manager._required_confirming_sources(session, time(11, 0)) == 4


def test_mu_choppy_filter_blocks_only_mu() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    mu_session = manager.get_or_create_session("run", "MU", "2026-02-03", 100.0)
    nvda_session = manager.get_or_create_session("run", "NVDA", "2026-02-03", 100.0)

    mu_session.micro_regime = "CHOPPY"
    assert manager._is_mu_choppy_blocked(mu_session, Regime.TRENDING) is True

    mu_session.micro_regime = "TRENDING_UP"
    assert manager._is_mu_choppy_blocked(mu_session, Regime.CHOPPY) is True

    nvda_session.micro_regime = "CHOPPY"
    assert manager._is_mu_choppy_blocked(nvda_session, Regime.CHOPPY) is False


def test_mu_choppy_filter_allows_scalp_intrabar_override() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    mu_session = manager.get_or_create_session("run", "MU", "2026-02-03", 100.0)

    mu_session.micro_regime = "CHOPPY"
    mu_session.active_strategies = ["momentum_flow", "scalp_l2_intrabar"]
    assert manager._is_mu_choppy_blocked(mu_session, Regime.CHOPPY) is False


def test_mu_choppy_filter_respects_ticker_override_flag() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    manager.ticker_params["MU"] = {"mu_choppy_hard_block_enabled": False}
    assert manager._is_mu_choppy_filter_enabled("MU") is False
    assert manager._is_mu_choppy_filter_enabled("NVDA") is True


def test_resolve_max_daily_trades_supports_unlimited() -> None:
    manager = DayTradingManager(regime_detection_minutes=0, max_trades_per_day=3)

    manager.ticker_params["MU"] = {"max_daily_trades": 8}
    assert manager._resolve_max_daily_trades("MU") == 8

    manager.ticker_params["MU"] = {"max_daily_trades": 0}
    assert manager._resolve_max_daily_trades("MU") is None

    manager.ticker_params["MU"] = {"max_daily_trades": -5}
    assert manager._resolve_max_daily_trades("MU") is None

    manager.ticker_params["MU"] = {}
    assert manager._resolve_max_daily_trades("MU") == 3


def test_resolve_max_daily_trades_prefers_session_override() -> None:
    manager = DayTradingManager(regime_detection_minutes=0, max_trades_per_day=3)
    session = manager.get_or_create_session("run", "MU", "2026-02-03", 100.0)

    manager.ticker_params["MU"] = {"max_daily_trades": 7}
    session.max_daily_trades_override = 0
    assert manager._resolve_max_daily_trades("MU", session=session) is None

    session.max_daily_trades_override = 4
    assert manager._resolve_max_daily_trades("MU", session=session) == 4


def test_mu_choppy_filter_prefers_session_override() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03", 100.0)

    manager.ticker_params["MU"] = {"mu_choppy_hard_block_enabled": True}
    session.mu_choppy_hard_block_enabled_override = False
    assert manager._is_mu_choppy_filter_enabled("MU", session=session) is False

    session.mu_choppy_hard_block_enabled_override = True
    assert manager._is_mu_choppy_filter_enabled("MU", session=session) is True


def test_cross_asset_headwind_boost_applies_for_opposing_index_move() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    cross_asset = CrossAssetState(
        index_available=True,
        index_momentum_5=-1.2,
        sector_relative=-0.35,
        headwind_score=0.75,
    )

    boost, metrics = manager._cross_asset_headwind_threshold_boost(
        cross_asset_state=cross_asset,
        decision_direction="bullish",
    )

    assert 5.0 <= boost <= 10.0
    assert metrics["applied"] is True
    assert metrics["index_opposes_direction"] is True


def test_cross_asset_headwind_boost_not_applied_on_tailwind() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    cross_asset = CrossAssetState(
        index_available=True,
        index_momentum_5=1.0,
        sector_relative=0.2,
        headwind_score=0.8,
    )

    boost, metrics = manager._cross_asset_headwind_threshold_boost(
        cross_asset_state=cross_asset,
        decision_direction="bullish",
    )

    assert boost == 0.0
    assert metrics["applied"] is False
    assert metrics["reason"] == "index_supports_direction"


def test_headwind_gate_rejects_borderline_bullish_signal() -> None:
    manager = DayTradingManager(
        regime_detection_minutes=0,
        trade_cooldown_bars=0,
        max_trades_per_day=10,
    )
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.phase = SessionPhase.TRADING
    session.selected_strategy = "momentum_flow"
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "TRENDING_UP"
    session.active_strategies = ["momentum_flow"]

    ts = datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc)
    bar = BarData(
        timestamp=ts,
        open=100.0,
        high=100.2,
        low=99.8,
        close=100.0,
        volume=100_000.0,
        vwap=100.0,
    )
    session.bars.append(bar)

    signal = _signal(strategy_name="MomentumFlow")
    decision = DecisionResult(
        execute=True,
        direction="bullish",
        signal=signal,
        combined_score=60.0,
        combined_raw=60.0,
        combined_norm_0_100=60.0,
        strategy_score=80.0,
        threshold=55.0,
        trade_gate_threshold=55.0,
    )

    class _StubEvidenceEngine:
        def __init__(self, fixed_decision: DecisionResult):
            self._decision = fixed_decision

        def evaluate(self, **_kwargs) -> DecisionResult:
            return self._decision

    class _StubConfig:
        use_evidence_engine = True

    class _StubOrchestrator:
        def __init__(self):
            self.config = _StubConfig()
            self.evidence_engine = _StubEvidenceEngine(decision)
            self.current_feature_vector = None
            self.current_regime_state = None
            self.current_cross_asset_state = CrossAssetState(
                index_available=True,
                index_momentum_5=-1.3,
                sector_relative=-0.25,
                headwind_score=0.78,
            )

    session.orchestrator = _StubOrchestrator()

    result = manager._process_trading_bar(session, bar, ts)

    assert result.get("signal_rejected", {}).get("gate") == "cross_asset_headwind"
    assert result.get("layer_scores", {}).get("headwind_threshold_boost", 0.0) >= 5.0
    assert session.pending_signal is None


def test_intrabar_eval_trace_re_evaluates_checkpoint_layer_scores() -> None:
    manager = DayTradingManager(
        regime_detection_minutes=0,
        trade_cooldown_bars=0,
        max_trades_per_day=10,
    )
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.phase = SessionPhase.TRADING
    session.selected_strategy = "momentum_flow"
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "TRENDING_UP"
    session.active_strategies = ["momentum_flow"]
    session.config = TradingConfig.from_dict({"intrabar_eval_step_seconds": 5})

    ts = datetime(2026, 2, 3, 15, 1, tzinfo=timezone.utc)
    intrabar_quotes = []
    for sec in range(1, 13):
        mid = 100.0 + (0.2 * sec)
        intrabar_quotes.append({"s": sec, "bid": mid - 0.01, "ask": mid + 0.01})

    bar = BarData(
        timestamp=ts,
        open=100.0,
        high=103.0,
        low=99.8,
        close=102.4,
        volume=120_000.0,
        vwap=101.0,
        intrabar_quotes_1s=intrabar_quotes,
    )
    session.bars.append(bar)

    class _StubEvidenceEngine:
        def __init__(self):
            self.observed_closes: List[float] = []

        def evaluate(self, **kwargs) -> DecisionResult:
            close_now = float(kwargs["ohlcv"]["close"][-1])
            self.observed_closes.append(close_now)
            return DecisionResult(
                execute=False,
                direction="bullish",
                signal=None,
                combined_score=close_now,
                combined_raw=close_now,
                combined_norm_0_100=close_now,
                strategy_score=close_now,
                threshold=999.0,
                trade_gate_threshold=999.0,
                reasoning="trace-test",
            )

    class _StubConfig:
        use_evidence_engine = True

    class _StubOrchestrator:
        def __init__(self):
            self.config = _StubConfig()
            self.evidence_engine = _StubEvidenceEngine()
            self.current_feature_vector = None
            self.current_regime_state = None
            self.current_cross_asset_state = CrossAssetState()

    session.orchestrator = _StubOrchestrator()

    result = manager._process_trading_bar(session, bar, ts)

    trace = result.get("intrabar_eval_trace", {})
    checkpoints = trace.get("checkpoints", [])

    assert trace.get("checkpoint_count") == 3
    assert [cp.get("offset_sec") for cp in checkpoints] == [5, 10, 12]
    assert all(isinstance(cp.get("layer_scores"), dict) for cp in checkpoints)

    combined_scores = [cp["layer_scores"]["combined_score"] for cp in checkpoints]
    assert combined_scores[0] < combined_scores[1] < combined_scores[2]
    assert result.get("layer_scores", {}).get("combined_score") == combined_scores[-1]
    assert len(session.orchestrator.evidence_engine.observed_closes) == 3


def test_l2_confirmation_hard_book_pressure_block_long() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.l2_confirm_enabled = True
    session.l2_gate_mode = "all_pass"
    session.bars = _l2_window(start=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc), direction=1.0)
    signal = _signal(strategy_name="MomentumFlow", signal_type=SignalType.BUY)

    passed, metrics = manager.gate_engine.passes_l2_confirmation(
        session,
        signal,
        flow_metrics={
            "l2_book_pressure_z": -2.8,
            "delta_acceleration": 120.0,
        },
    )

    assert passed is False
    assert metrics["hard_block"] is True
    assert metrics["reason"] == "book_pressure_block_long"
    assert metrics["passes_book_pressure_block"] is False


def test_l2_confirmation_hard_book_pressure_block_short() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.l2_confirm_enabled = True
    session.l2_gate_mode = "all_pass"
    session.bars = _l2_window(start=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc), direction=-1.0)
    signal = _signal(strategy_name="MomentumFlow", signal_type=SignalType.SELL)

    passed, metrics = manager.gate_engine.passes_l2_confirmation(
        session,
        signal,
        flow_metrics={
            "l2_book_pressure_z": 2.8,
            "delta_acceleration": -120.0,
        },
    )

    assert passed is False
    assert metrics["hard_block"] is True
    assert metrics["reason"] == "book_pressure_block_short"
    assert metrics["passes_book_pressure_block"] is False


def test_l2_confirmation_pullback_long_ignores_delta_acceleration_filter() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.l2_confirm_enabled = True
    session.l2_gate_mode = "all_pass"
    session.bars = _l2_window(start=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc), direction=1.0)
    signal = _signal(strategy_name="Pullback", signal_type=SignalType.BUY)

    passed, metrics = manager.gate_engine.passes_l2_confirmation(
        session,
        signal,
        flow_metrics={
            "l2_book_pressure_z": 0.0,
            "delta_acceleration": -25.0,
        },
    )

    assert passed is True
    assert metrics["hard_block"] is False
    assert metrics["l2_effective_mode"] == "hard_block_only"


def test_l2_confirmation_allows_pullback_long_when_delta_acceleration_non_negative() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.l2_confirm_enabled = True
    session.l2_gate_mode = "all_pass"
    session.bars = _l2_window(start=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc), direction=1.0)
    signal = _signal(strategy_name="Pullback", signal_type=SignalType.BUY)

    passed, metrics = manager.gate_engine.passes_l2_confirmation(
        session,
        signal,
        flow_metrics={
            "l2_book_pressure_z": 0.2,
            "delta_acceleration": 0.0,
        },
    )

    assert passed is True
    assert metrics["hard_block"] is False
    assert metrics["passes_book_pressure_block"] is True


def test_l2_hard_block_is_not_overridden_by_weak_l2_fast_break_even() -> None:
    manager = DayTradingManager(
        regime_detection_minutes=0,
        trade_cooldown_bars=0,
        max_trades_per_day=10,
    )
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.phase = SessionPhase.TRADING
    session.selected_strategy = "pullback"
    session.detected_regime = Regime.TRENDING
    session.micro_regime = "TRENDING_UP"
    session.active_strategies = ["pullback"]
    session.l2_confirm_enabled = True
    session.l2_gate_mode = "all_pass"
    session.config = TradingConfig(
        l2_confirm_enabled=True,
        weak_l2_fast_break_even_enabled=True,
        weak_l2_aggression_threshold=1.0,
        weak_l2_break_even_min_hold_bars=2,
    )

    ts = datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc)
    bar = BarData(
        timestamp=ts,
        open=100.0,
        high=100.2,
        low=99.8,
        close=100.1,
        volume=100_000.0,
        vwap=100.0,
    )
    session.bars.append(bar)

    signal = _signal(strategy_name="Pullback", signal_type=SignalType.BUY)
    decision = DecisionResult(
        execute=True,
        direction="bullish",
        signal=signal,
        combined_score=70.0,
        combined_raw=70.0,
        combined_norm_0_100=70.0,
        strategy_score=80.0,
        threshold=55.0,
        trade_gate_threshold=55.0,
    )

    class _StubEvidenceEngine:
        def __init__(self, fixed_decision: DecisionResult):
            self._decision = fixed_decision

        def evaluate(self, **_kwargs) -> DecisionResult:
            return self._decision

    class _StubConfig:
        use_evidence_engine = True

    class _StubOrchestrator:
        def __init__(self):
            self.config = _StubConfig()
            self.evidence_engine = _StubEvidenceEngine(decision)
            self.current_feature_vector = None
            self.current_regime_state = None
            self.current_cross_asset_state = None

    session.orchestrator = _StubOrchestrator()

    def _hard_block_gate(
        _session: Any,
        _signal: Signal,
        flow_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        return (
            False,
            {
                "passed": False,
                "hard_block": True,
                "reason": "book_pressure_block_long",
                "signed_aggression_avg": 0.0,
                "flow_metrics": dict(flow_metrics or {}),
            },
        )

    manager.gate_engine.passes_l2_confirmation = _hard_block_gate  # type: ignore[assignment]

    result = manager._process_trading_bar(session, bar, ts)

    assert result.get("action") == "l2_filtered"
    assert result.get("reason") == "book_pressure_block_long"
    assert result.get("weak_l2_entry") is None


def test_tcbbo_directional_override_applies_for_massive_bullish_flow() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.tcbbo_gate_enabled = True
    session.tcbbo_min_net_premium = 250_000.0
    session.tcbbo_lookback_bars = 3
    session.bars = [
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.1,
            volume=100_000.0,
            vwap=100.0,
            tcbbo_net_premium=420_000.0,
            tcbbo_cumulative_net_premium=700_000.0,
            tcbbo_has_data=True,
        ),
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc),
            open=100.1,
            high=100.3,
            low=99.9,
            close=100.2,
            volume=95_000.0,
            vwap=100.1,
            tcbbo_net_premium=380_000.0,
            tcbbo_cumulative_net_premium=1_080_000.0,
            tcbbo_has_data=True,
        ),
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 32, tzinfo=timezone.utc),
            open=100.2,
            high=100.4,
            low=100.0,
            close=100.3,
            volume=90_000.0,
            vwap=100.2,
            tcbbo_net_premium=310_000.0,
            tcbbo_cumulative_net_premium=1_390_000.0,
            tcbbo_has_data=True,
        ),
    ]

    metrics = manager.gate_engine.tcbbo_directional_override(session)

    assert metrics["applied"] is True
    assert metrics["direction"] == "bullish"
    assert metrics["override_threshold"] >= 1_000_000.0


def test_tcbbo_override_bypasses_mu_choppy_hard_block() -> None:
    manager = DayTradingManager(
        regime_detection_minutes=0,
        trade_cooldown_bars=0,
        max_trades_per_day=10,
    )
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.phase = SessionPhase.TRADING
    session.selected_strategy = "momentum_flow"
    session.detected_regime = Regime.CHOPPY
    session.micro_regime = "CHOPPY"
    session.active_strategies = ["momentum_flow"]
    session.tcbbo_gate_enabled = True
    session.tcbbo_min_net_premium = 250_000.0
    session.tcbbo_lookback_bars = 3

    bar_1 = BarData(
        timestamp=datetime(2026, 2, 3, 14, 58, tzinfo=timezone.utc),
        open=100.0,
        high=100.2,
        low=99.8,
        close=100.1,
        volume=100_000.0,
        vwap=100.0,
        tcbbo_net_premium=420_000.0,
        tcbbo_cumulative_net_premium=700_000.0,
        tcbbo_has_data=True,
    )
    bar_2 = BarData(
        timestamp=datetime(2026, 2, 3, 14, 59, tzinfo=timezone.utc),
        open=100.1,
        high=100.3,
        low=99.9,
        close=100.2,
        volume=95_000.0,
        vwap=100.1,
        tcbbo_net_premium=380_000.0,
        tcbbo_cumulative_net_premium=1_080_000.0,
        tcbbo_has_data=True,
    )
    bar = BarData(
        timestamp=datetime(2026, 2, 3, 15, 0, tzinfo=timezone.utc),
        open=100.2,
        high=100.4,
        low=100.0,
        close=100.3,
        volume=90_000.0,
        vwap=100.2,
        tcbbo_net_premium=310_000.0,
        tcbbo_cumulative_net_premium=1_390_000.0,
        tcbbo_has_data=True,
    )
    session.bars = [bar_1, bar_2, bar]

    decision = DecisionResult(
        execute=False,
        direction="bullish",
        signal=None,
        combined_score=20.0,
        combined_raw=20.0,
        combined_norm_0_100=20.0,
        strategy_score=20.0,
        threshold=55.0,
        trade_gate_threshold=55.0,
    )

    class _StubEvidenceEngine:
        def __init__(self, fixed_decision: DecisionResult):
            self._decision = fixed_decision

        def evaluate(self, **_kwargs) -> DecisionResult:
            return self._decision

    class _StubConfig:
        use_evidence_engine = True

    class _StubOrchestrator:
        def __init__(self):
            self.config = _StubConfig()
            self.evidence_engine = _StubEvidenceEngine(decision)
            self.current_feature_vector = None
            self.current_regime_state = None
            self.current_cross_asset_state = None

    session.orchestrator = _StubOrchestrator()

    result = manager._process_trading_bar(session, bar, bar.timestamp)

    assert result.get("action") != "regime_filter"
    assert result.get("tcbbo_regime_override", {}).get("applied") is True


def test_momentum_flow_requires_delta_divergence_confirmation() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)

    no_delta_signal = _signal(
        metadata={
            "evidence_sources": [
                {"type": "feature", "name": "rsi_oversold", "direction": "bullish"},
                {"type": "strategy_signal", "name": "momentum_flow", "direction": "bullish"},
            ],
            "layer_scores": {"source_weights": {"strategy_signal:momentum_flow": 0.6}},
        }
    )
    passed, metrics = manager._passes_momentum_flow_delta_confirmation(no_delta_signal)
    assert passed is True
    assert metrics["enabled"] is False
    assert metrics["reason"] == "momentum_flow_delta_divergence_filter_disabled"

    with_delta_signal = _signal(
        metadata={
            "evidence_sources": [
                {"type": "l2_flow", "name": "delta_divergence", "direction": "bullish"},
                {"type": "strategy_signal", "name": "momentum_flow", "direction": "bullish"},
            ],
            "layer_scores": {
                "source_weights": {
                    "l2_flow:delta_divergence": 0.35,
                    "strategy_signal:momentum_flow": 0.65,
                }
            },
        }
    )
    passed, _ = manager._passes_momentum_flow_delta_confirmation(with_delta_signal)
    assert passed is True


def test_confirming_source_stats_prefers_ensemble_count() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    signal = _signal(
        metadata={
            "evidence_sources": [
                {"type": "feature", "name": "rsi_oversold", "direction": "bullish"},
                {"type": "l2_flow", "name": "aggression", "direction": "bullish"},
                {"type": "strategy_signal", "name": "momentum_flow", "direction": "bullish"},
            ],
            "layer_scores": {"confirming_sources": 1},
        }
    )

    stats = manager._confirming_source_stats(signal)
    assert stats["confirming_sources"] == 1
    assert stats["aligned_evidence_sources"] == 3
    assert stats["count_source"] == "ensemble_layer_scores"


def test_extract_confirming_source_keys_excludes_primary_strategy() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    signal_metadata = {
        "layer_scores": {
            "aligned_source_keys": [
                "strategy:momentum_flow",
                "l2_flow:delta_divergence",
                "feature:rsi_oversold",
                "feature:rsi_oversold",
            ]
        }
    }

    keys = manager._extract_confirming_source_keys_from_metadata(
        signal_metadata=signal_metadata,
        side="long",
        strategy_name="MomentumFlow",
    )
    assert keys == ["l2_flow:delta_divergence", "feature:rsi_oversold"]


def test_extract_confirming_source_keys_falls_back_to_evidence_sources() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    signal_metadata = {
        "evidence_sources": [
            {"type": "strategy", "name": "momentum_flow", "direction": "bullish"},
            {"type": "feature", "name": "rsi_oversold", "direction": "bullish"},
            {"type": "l2_flow", "name": "aggression", "direction": "bearish"},
        ]
    }

    keys = manager._extract_confirming_source_keys_from_metadata(
        signal_metadata=signal_metadata,
        side="long",
        strategy_name="momentum_flow",
    )
    assert keys == ["feature:rsi_oversold"]


def test_extract_raw_confidence_prefers_adjusted_then_strategy_score() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    adjusted = manager._extract_raw_confidence_from_metadata(
        {
            "confidence_adjustment": {"adjusted_confidence": 77.5},
            "layer_scores": {"strategy_score": 88.0},
        }
    )
    assert adjusted == 77.5

    from_strategy = manager._extract_raw_confidence_from_metadata(
        {"layer_scores": {"strategy_score": 88.0}}
    )
    assert from_strategy == 88.0


def test_close_position_forwards_confirming_sources_and_confidence() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.detected_regime = Regime.TRENDING

    class _StubOrchestrator:
        def __init__(self):
            self.calls = []

        def record_trade_outcome(self, **kwargs):
            self.calls.append(kwargs)

    stub = _StubOrchestrator()
    session.orchestrator = stub

    position = Position(
        strategy_name="momentum_flow",
        entry_price=100.0,
        entry_time=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        side="long",
        size=10.0,
        stop_loss=99.0,
    )
    position.signal_metadata = {
        "confidence_adjustment": {"adjusted_confidence": 74.0},
        "layer_scores": {
            "strategy_score": 80.0,
            "aligned_source_keys": [
                "strategy:momentum_flow",
                "feature:rsi_oversold",
                "l2_flow:delta_divergence",
            ],
        },
    }
    session.active_position = position

    manager._close_position(
        session=session,
        exit_price=101.0,
        exit_time=datetime(2026, 2, 3, 14, 35, tzinfo=timezone.utc),
        reason="take_profit",
        bar_volume=100_000.0,
    )

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["raw_confidence"] == 74.0
    assert call["confirming_sources"] == [
        "feature:rsi_oversold",
        "l2_flow:delta_divergence",
    ]


def test_momentum_diversification_gate_blocks_weak_l2_context() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.micro_regime = "TRENDING_UP"
    manager.ticker_params["MU"] = {
        "adaptive": {
            "momentum_diversification": {
                "enabled": True,
                "apply_to_strategies": ["momentum_flow"],
                "min_flow_score": 60.0,
                "min_directional_consistency": 0.40,
                "min_signed_aggression": 0.06,
                "min_imbalance": 0.04,
                "min_delta_acceleration": 100.0,
                "min_delta_price_divergence": -0.20,
            }
        }
    }
    signal = _signal(strategy_name="MomentumFlow", signal_type=SignalType.BUY)

    weak_flow = {
        "has_l2_coverage": True,
        "flow_score": 55.0,
        "directional_consistency": 0.50,
        "signed_aggression": 0.03,
        "imbalance_avg": 0.02,
        "delta_acceleration": 80.0,
        "delta_price_divergence": -0.30,
    }
    passed, metrics = manager._passes_momentum_diversification_gate(session, signal, weak_flow)
    assert passed is False
    assert metrics["reason"] == "momentum_diversification_gate_failed"

    strong_flow = {
        "has_l2_coverage": True,
        "flow_score": 72.0,
        "directional_consistency": 0.62,
        "signed_aggression": 0.11,
        "imbalance_avg": 0.08,
        "delta_acceleration": 240.0,
        "delta_price_divergence": 0.05,
    }
    passed, metrics = manager._passes_momentum_diversification_gate(session, signal, strong_flow)
    assert passed is True
    assert metrics["passed"] is True


def test_momentum_diversification_gate_weighted_scoring() -> None:
    """Weighted scoring gives partial credit: medium-quality flow passes, terrible flow fails."""
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.micro_regime = "TRENDING_UP"
    manager.ticker_params["MU"] = {
        "adaptive": {
            "momentum_diversification": {
                "enabled": True,
                "apply_to_strategies": ["momentum_flow"],
                "gate_mode": "weighted",
                "gate_threshold": 0.55,
                "gate_flow_floor": 40.0,
                "min_flow_score": 40.0,
                "min_directional_consistency": 0.40,
                "min_signed_aggression": 0.06,
                "min_imbalance": 0.04,
                "min_cvd": 1500.0,
                "min_directional_price_change_pct": 0.10,
                "min_price_trend_efficiency": 0.25,
                "min_last_bar_body_ratio": 0.35,
                "min_last_bar_close_location": 0.60,
                "min_delta_acceleration": 100.0,
                "min_delta_price_divergence": -0.20,
            }
        }
    }
    signal = _signal(strategy_name="MomentumFlow", signal_type=SignalType.BUY)

    # Medium flow: some metrics below threshold but overall composite is adequate
    medium_flow = {
        "has_l2_coverage": True,
        "flow_score": 70.0,
        "directional_consistency": 0.55,
        "signed_aggression": 0.09,
        "imbalance_avg": 0.07,
        "cumulative_delta": 1100.0,
        "delta_acceleration": 220.0,
        "delta_price_divergence": 0.08,
        "price_change_pct": 0.04,
        "price_trend_efficiency": 0.12,
        "latest_bar_body_ratio": 0.20,
        "latest_bar_close_location": 0.53,
    }
    passed, metrics = manager._passes_momentum_diversification_gate(session, signal, medium_flow)
    # Weighted scoring allows partial credit: medium flow passes despite some weak dimensions
    assert passed is True
    assert metrics["gate_mode"] == "weighted"
    assert metrics["gate_score"] is not None
    assert metrics["gate_score"] > 0.55
    assert metrics["passes_flow_floor"] is True

    # Terrible flow: below flow floor
    terrible_flow = {
        "has_l2_coverage": True,
        "flow_score": 30.0,
        "directional_consistency": 0.20,
        "signed_aggression": 0.01,
        "imbalance_avg": 0.01,
        "cumulative_delta": 200.0,
        "delta_acceleration": 10.0,
        "delta_price_divergence": -0.40,
        "price_change_pct": 0.01,
        "price_trend_efficiency": 0.05,
        "latest_bar_body_ratio": 0.10,
        "latest_bar_close_location": 0.30,
    }
    passed, metrics = manager._passes_momentum_diversification_gate(session, signal, terrible_flow)
    assert passed is False
    assert metrics["passes_flow_floor"] is False

    # Strong flow: well above threshold
    strong_flow = {
        "has_l2_coverage": True,
        "flow_score": 74.0,
        "directional_consistency": 0.62,
        "signed_aggression": 0.11,
        "imbalance_avg": 0.09,
        "cumulative_delta": 2300.0,
        "delta_acceleration": 340.0,
        "delta_price_divergence": 0.11,
        "price_change_pct": 0.24,
        "price_trend_efficiency": 0.42,
        "latest_bar_body_ratio": 0.56,
        "latest_bar_close_location": 0.82,
    }
    passed, metrics = manager._passes_momentum_diversification_gate(session, signal, strong_flow)
    assert passed is True
    assert metrics["gate_score"] > metrics["gate_threshold"]


def test_momentum_diversification_gate_all_pass_mode() -> None:
    """Legacy all_pass mode: individual metric failures cause rejection."""
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.micro_regime = "TRENDING_UP"
    manager.ticker_params["MU"] = {
        "adaptive": {
            "momentum_diversification": {
                "enabled": True,
                "apply_to_strategies": ["momentum_flow"],
                "gate_mode": "all_pass",
                "min_flow_score": 40.0,
                "min_directional_consistency": 0.0,
                "min_signed_aggression": 0.0,
                "min_imbalance": 0.0,
                "min_cvd": 1500.0,
                "min_directional_price_change_pct": 0.10,
                "min_price_trend_efficiency": 0.25,
                "min_last_bar_body_ratio": 0.35,
                "min_last_bar_close_location": 0.60,
                "min_delta_acceleration": -1000.0,
                "min_delta_price_divergence": -10.0,
            }
        }
    }
    signal = _signal(strategy_name="MomentumFlow", signal_type=SignalType.BUY)

    weak_flow = {
        "has_l2_coverage": True,
        "flow_score": 70.0,
        "directional_consistency": 0.55,
        "signed_aggression": 0.09,
        "imbalance_avg": 0.07,
        "cumulative_delta": 1100.0,
        "delta_acceleration": 220.0,
        "delta_price_divergence": 0.08,
        "price_change_pct": 0.04,
        "price_trend_efficiency": 0.12,
        "latest_bar_body_ratio": 0.20,
        "latest_bar_close_location": 0.53,
    }
    passed, metrics = manager._passes_momentum_diversification_gate(session, signal, weak_flow)
    assert passed is False
    assert metrics["passes_cvd"] is False
    assert metrics["passes_directional_price_change_pct"] is False
    assert metrics["passes_price_trend_efficiency"] is False
    assert metrics["passes_last_bar_body_ratio"] is False
    assert metrics["passes_last_bar_close_location"] is False


def test_momentum_diversification_multi_sleeve_selects_matching_sleeve() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.micro_regime = "TRENDING_UP"
    manager.ticker_params["MU"] = {
        "adaptive": {
            "momentum_diversification": {
                "enabled": True,
                "sleeves": [
                    {
                        "sleeve_id": "strict",
                        "enabled": True,
                        "apply_to_strategies": ["momentum_flow"],
                        "min_flow_score": 78.0,
                        "min_directional_consistency": 0.75,
                        "min_signed_aggression": 0.15,
                        "min_imbalance": 0.10,
                    },
                    {
                        "sleeve_id": "balanced",
                        "enabled": True,
                        "apply_to_strategies": ["momentum_flow"],
                        "min_flow_score": 58.0,
                        "min_directional_consistency": 0.45,
                        "min_signed_aggression": 0.05,
                        "min_imbalance": 0.03,
                    },
                ],
            }
        }
    }
    signal = _signal(strategy_name="MomentumFlow", signal_type=SignalType.BUY)
    flow = {
        "has_l2_coverage": True,
        "flow_score": 64.0,
        "directional_consistency": 0.52,
        "signed_aggression": 0.08,
        "imbalance_avg": 0.05,
        "delta_acceleration": 120.0,
        "delta_price_divergence": -0.05,
    }

    passed, metrics = manager._passes_momentum_diversification_gate(session, signal, flow)
    assert passed is True
    assert metrics["selected_sleeve_id"] == "balanced"
    assert metrics["sleeve_mode"] == "multi"


def test_momentum_fail_fast_exit_triggers_for_early_l2_flip() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.micro_regime = "TRENDING_UP"
    session.momentum_diversification_override = True
    session.momentum_diversification = {
        "enabled": True,
        "apply_to_strategies": ["momentum_flow"],
        "fail_fast_exit_enabled": True,
        "fail_fast_max_bars": 3,
        "fail_fast_signed_aggression_max": -0.04,
        "fail_fast_book_pressure_max": -0.08,
        "fail_fast_directional_consistency_max": 0.45,
    }

    session.bars = [
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.1,
            volume=20000.0,
            l2_delta=-1500.0,
            l2_volume=12000.0,
            l2_imbalance=-0.20,
            l2_book_pressure=-0.16,
        ),
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc),
            open=100.1,
            high=100.2,
            low=99.7,
            close=99.9,
            volume=19000.0,
            l2_delta=-1800.0,
            l2_volume=12500.0,
            l2_imbalance=-0.24,
            l2_book_pressure=-0.18,
        ),
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 32, tzinfo=timezone.utc),
            open=99.9,
            high=100.0,
            low=99.5,
            close=99.6,
            volume=21000.0,
            l2_delta=-2100.0,
            l2_volume=13000.0,
            l2_imbalance=-0.28,
            l2_book_pressure=-0.22,
        ),
    ]
    pos = Position(
        strategy_name="momentum_flow",
        entry_price=100.0,
        entry_time=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        side="long",
        size=10.0,
        stop_loss=99.0,
    )
    pos.entry_bar_index = 0

    should_exit, metrics = manager._should_momentum_fail_fast_exit(
        session=session,
        pos=pos,
        current_bar_index=2,
    )
    assert should_exit is True
    assert metrics["reason"] == "momentum_fail_fast_l2_flip"


def test_momentum_fail_fast_uses_entry_selected_sleeve() -> None:
    manager = DayTradingManager(regime_detection_minutes=0)
    session = manager.get_or_create_session("run", "MU", "2026-02-03")
    session.micro_regime = "TRENDING_UP"
    session.momentum_diversification_override = True
    session.momentum_diversification = {
        "enabled": True,
        "sleeves": [
            {
                "sleeve_id": "impulse",
                "enabled": True,
                "apply_to_strategies": ["momentum_flow"],
                "fail_fast_exit_enabled": False,
                "fail_fast_max_bars": 3,
            },
            {
                "sleeve_id": "defensive",
                "enabled": True,
                "apply_to_strategies": ["momentum_flow"],
                "fail_fast_exit_enabled": True,
                "fail_fast_max_bars": 3,
                "fail_fast_signed_aggression_max": -0.04,
                "fail_fast_book_pressure_max": -0.08,
                "fail_fast_directional_consistency_max": 0.45,
            },
        ],
    }

    session.bars = [
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.1,
            volume=20000.0,
            l2_delta=-1500.0,
            l2_volume=12000.0,
            l2_imbalance=-0.20,
            l2_book_pressure=-0.16,
        ),
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc),
            open=100.1,
            high=100.2,
            low=99.7,
            close=99.9,
            volume=19000.0,
            l2_delta=-1800.0,
            l2_volume=12500.0,
            l2_imbalance=-0.24,
            l2_book_pressure=-0.18,
        ),
        BarData(
            timestamp=datetime(2026, 2, 3, 14, 32, tzinfo=timezone.utc),
            open=99.9,
            high=100.0,
            low=99.5,
            close=99.6,
            volume=21000.0,
            l2_delta=-2100.0,
            l2_volume=13000.0,
            l2_imbalance=-0.28,
            l2_book_pressure=-0.22,
        ),
    ]
    pos = Position(
        strategy_name="momentum_flow",
        entry_price=100.0,
        entry_time=datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc),
        side="long",
        size=10.0,
        stop_loss=99.0,
    )
    pos.entry_bar_index = 0
    pos.signal_metadata = {
        "momentum_diversification": {
            "selected_sleeve_id": "defensive",
        }
    }

    should_exit, metrics = manager._should_momentum_fail_fast_exit(
        session=session,
        pos=pos,
        current_bar_index=2,
    )
    assert should_exit is True
    assert metrics["selected_sleeve_id"] == "defensive"
