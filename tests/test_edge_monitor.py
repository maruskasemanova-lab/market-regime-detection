"""Tests for EdgeMonitor: strategy edge tracking and degradation detection."""
import pytest
from src.edge_monitor import (
    EdgeMonitor, EdgeStatus, RecommendedAction, EdgeHealth,
    StrategyEdgeTracker, MIN_TRADES_FOR_ASSESSMENT,
)


class TestStrategyEdgeTracker:
    def test_empty(self):
        t = StrategyEdgeTracker()
        assert t.n_trades == 0
        assert t.rolling_win_rate == 0.5
        assert t.rolling_sharpe == 0.0

    def test_all_wins(self):
        t = StrategyEdgeTracker()
        for _ in range(10):
            t.add_trade(1.5, True)
        assert t.rolling_win_rate == 1.0
        assert t.rolling_sharpe > 0

    def test_all_losses(self):
        t = StrategyEdgeTracker()
        for _ in range(10):
            t.add_trade(-1.0, False)
        assert t.rolling_win_rate == 0.0
        assert t.rolling_sharpe < 0

    def test_profit_factor(self):
        t = StrategyEdgeTracker()
        for _ in range(5):
            t.add_trade(2.0, True)
        for _ in range(5):
            t.add_trade(-1.0, False)
        assert t.profit_factor == pytest.approx(2.0)

    def test_max_drawdown(self):
        t = StrategyEdgeTracker()
        t.add_trade(3.0, True)
        t.add_trade(-1.0, False)
        t.add_trade(-1.0, False)
        t.add_trade(-1.0, False)
        assert t.max_drawdown_r == 3.0


class TestEdgeMonitor:
    def test_insufficient_data(self):
        monitor = EdgeMonitor()
        health = monitor.get_health('momentum', 'TRENDING')
        assert health.status == EdgeStatus.INSUFFICIENT
        assert health.action == RecommendedAction.TRADE

    def test_strong_edge(self):
        monitor = EdgeMonitor()
        for _ in range(25):
            monitor.update_trade('momentum', 'TRENDING', pnl_r=2.0, was_profitable=True)
        health = monitor.get_health('momentum', 'TRENDING')
        assert health.status == EdgeStatus.STRONG
        assert health.action == RecommendedAction.TRADE
        assert health.threshold_adjustment == 0.0

    def test_dead_edge(self):
        monitor = EdgeMonitor()
        for _ in range(25):
            monitor.update_trade('bad_strat', 'CHOPPY', pnl_r=-1.5, was_profitable=False)
        health = monitor.get_health('bad_strat', 'CHOPPY')
        assert health.status == EdgeStatus.DEAD
        assert health.action == RecommendedAction.PAUSE
        assert health.threshold_adjustment > 0

    def test_degraded_edge(self):
        monitor = EdgeMonitor()
        # Mix of wins and losses with negative expectancy
        for i in range(25):
            if i % 4 == 0:
                monitor.update_trade('strat', 'MIXED', pnl_r=0.5, was_profitable=True)
            else:
                monitor.update_trade('strat', 'MIXED', pnl_r=-0.8, was_profitable=False)
        health = monitor.get_health('strat', 'MIXED')
        assert health.status in (EdgeStatus.DEGRADED, EdgeStatus.DEAD)
        assert health.threshold_adjustment > 0

    def test_should_refit_false_initially(self):
        monitor = EdgeMonitor()
        assert monitor.should_refit() is False

    def test_should_refit_true_when_degraded(self):
        monitor = EdgeMonitor()
        # Multiple strategies all losing
        for strat in ['a', 'b', 'c']:
            for _ in range(25):
                monitor.update_trade(strat, 'TRENDING', pnl_r=-1.0, was_profitable=False)
        assert monitor.should_refit() is True

    def test_get_all_health(self):
        monitor = EdgeMonitor()
        for _ in range(5):
            monitor.update_trade('a', 'TRENDING', pnl_r=1.0, was_profitable=True)
            monitor.update_trade('b', 'CHOPPY', pnl_r=-0.5, was_profitable=False)
        all_health = monitor.get_all_health()
        assert len(all_health) == 2

    def test_global_health(self):
        monitor = EdgeMonitor()
        for _ in range(25):
            monitor.update_trade('a', 'TRENDING', pnl_r=1.5, was_profitable=True)
        global_h = monitor.get_global_health()
        assert global_h.total_trades == 25
        assert global_h.rolling_win_rate == 1.0

    def test_degradation_events_logged(self):
        monitor = EdgeMonitor()
        for _ in range(25):
            monitor.update_trade('bad', 'CHOPPY', pnl_r=-2.0, was_profitable=False)
        events = monitor.get_degradation_events()
        assert len(events) > 0
        assert events[-1]['strategy'] == 'bad'

    def test_reset(self):
        monitor = EdgeMonitor()
        monitor.update_trade('a', 'TRENDING', pnl_r=1.0, was_profitable=True)
        monitor.reset()
        assert monitor.get_all_health() == []
