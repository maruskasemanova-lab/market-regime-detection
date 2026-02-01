import { useState, useEffect } from 'react';
import useBackendApi from '../hooks/useBackendApi';
import RegimeIndicator from './RegimeIndicator';
import StrategyCard from './StrategyCard';
import SignalsTable from './SignalsTable';
import TradesHistory from './TradesHistory';
import PositionsList from './PositionsList';
import PerformanceStats from './PerformanceStats';

function Dashboard() {
  const {
    state,
    loading,
    error,
    fetchState,
    fetchRegime,
    fetchSignals,
    fetchTrades,
    fetchPositions,
    fetchPerformance,
    fetchCurrent,
    stepBacktest,
    runBacktest,
    resetEngine,
    toggleStrategy,
  } = useBackendApi();

  const [regime, setRegime] = useState(null);
  const [signals, setSignals] = useState([]);
  const [trades, setTrades] = useState([]);
  const [positions, setPositions] = useState({});
  const [performance, setPerformance] = useState(null);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [stepResult, setStepResult] = useState(null);

  // Fetch all data
  const refreshData = async () => {
    const [regimeData, signalsData, tradesData, positionsData, perfData, currentData] = 
      await Promise.all([
        fetchRegime(),
        fetchSignals(),
        fetchTrades(),
        fetchPositions(),
        fetchPerformance(),
        fetchCurrent(),
      ]);
    
    if (regimeData) setRegime(regimeData);
    if (signalsData) setSignals(signalsData.signals || []);
    if (tradesData) setTrades(tradesData.trades || []);
    if (positionsData) setPositions(positionsData.open_positions || {});
    if (perfData) setPerformance(perfData);
    if (currentData) setCurrentPrice(currentData);
  };

  useEffect(() => {
    refreshData();
    // Refresh every 5 seconds
    const interval = setInterval(refreshData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleStep = async () => {
    const result = await stepBacktest();
    if (result) {
      setStepResult(result);
      await refreshData();
    }
  };

  const handleRun = async (bars) => {
    setIsRunning(true);
    for (let i = 0; i < bars; i++) {
      const result = await stepBacktest();
      if (result) {
        setStepResult(result);
        await refreshData();
      }
      await new Promise(r => setTimeout(r, 200)); // Small delay for visualization
    }
    setIsRunning(false);
  };

  const handleReset = async () => {
    await resetEngine();
    setStepResult(null);
    await refreshData();
  };

  const handleToggleStrategy = async (name, enabled) => {
    await toggleStrategy(name, enabled);
    await fetchState();
  };

  if (loading && !state) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        Loading...
      </div>
    );
  }

  if (error) {
    return (
      <div className="card" style={{ textAlign: 'center', color: 'var(--accent-red)' }}>
        <h3>⚠️ Connection Error</h3>
        <p>{error}</p>
        <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>
          Make sure the API server is running on port 8001
        </p>
        <button className="btn btn-primary" onClick={refreshData} style={{ marginTop: '1rem' }}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="dashboard-grid">
      {/* Sidebar */}
      <div className="sidebar">
        {/* Regime Indicator */}
        <RegimeIndicator regime={regime} />
        
        {/* Controls */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Controls</span>
          </div>
          <div className="controls" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
            <button 
              className="btn btn-primary" 
              onClick={handleStep}
              disabled={isRunning}
            >
              ▶ Step
            </button>
            <button 
              className="btn btn-success" 
              onClick={() => handleRun(10)}
              disabled={isRunning}
            >
              {isRunning ? '⏳ Running...' : '⏩ Run 10 Bars'}
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={() => handleRun(50)}
              disabled={isRunning}
            >
              ⏭ Run 50 Bars
            </button>
            <button 
              className="btn btn-danger" 
              onClick={handleReset}
              disabled={isRunning}
            >
              🔄 Reset
            </button>
          </div>
        </div>

        {/* Current Price */}
        {currentPrice && (
          <div className="card">
            <div className="card-header">
              <span className="card-title">Current Bar</span>
            </div>
            <div className="current-price">${currentPrice.current_price?.toFixed(2)}</div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
              Bar #{currentPrice.bar || 0}
            </div>
          </div>
        )}

        {/* Open Positions */}
        <PositionsList positions={positions} />
      </div>

      {/* Main Panel */}
      <div className="main-panel">
        {/* Performance Stats */}
        <PerformanceStats performance={performance} />

        {/* Strategies */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Active Strategies</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              {regime?.active_strategies?.length || 0} active
            </span>
          </div>
          <div className="strategies-grid">
            {state?.strategies && Object.entries(state.strategies).map(([key, strategy]) => (
              <StrategyCard
                key={key}
                strategy={strategy}
                isActive={regime?.active_strategies?.includes(strategy.name)}
                currentRegime={regime?.regime}
                onToggle={(enabled) => handleToggleStrategy(key, enabled)}
              />
            ))}
          </div>
        </div>

        {/* Latest Step Result */}
        {stepResult && (
          <div className="card">
            <div className="card-header">
              <span className="card-title">Latest Step Result</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                Bar #{stepResult.bar}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
              <div className="stat-item">
                <div className="stat-value">${stepResult.price?.toFixed(2)}</div>
                <div className="stat-label">Price</div>
              </div>
              <div className="stat-item">
                <div className={`stat-value regime-badge ${stepResult.regime?.toLowerCase()}`} style={{ fontSize: '1rem', padding: '4px 12px' }}>
                  {stepResult.regime}
                </div>
                <div className="stat-label">Regime</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{stepResult.signals?.length || 0}</div>
                <div className="stat-label">Signals</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{stepResult.closed_trades?.length || 0}</div>
                <div className="stat-label">Trades Closed</div>
              </div>
            </div>
            {stepResult.signals?.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <strong>New Signals:</strong>
                {stepResult.signals.map((sig, i) => (
                  <div key={i} style={{ 
                    marginTop: '0.5rem', 
                    padding: '0.5rem', 
                    background: 'var(--bg-secondary)', 
                    borderRadius: '8px',
                    fontSize: '0.85rem'
                  }}>
                    <span className={`signal-type ${sig.signal?.toLowerCase()}`}>{sig.signal}</span>
                    <span style={{ marginLeft: '0.5rem' }}>{sig.strategy}</span>
                    <span style={{ marginLeft: '0.5rem', color: 'var(--text-muted)' }}>
                      @ ${sig.price?.toFixed(2)} | Confidence: {sig.confidence?.toFixed(0)}%
                    </span>
                    <div style={{ marginTop: '0.25rem', color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                      {sig.reasoning}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Signals Table */}
        <SignalsTable signals={signals} />

        {/* Trades History */}
        <TradesHistory trades={trades} />
      </div>
    </div>
  );
}

export default Dashboard;
