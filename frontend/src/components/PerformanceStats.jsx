function PerformanceStats({ performance }) {
  if (!performance) {
    return (
      <div className="card">
        <div className="card-header">
          <span className="card-title">Performance</span>
        </div>
        <div className="stats-grid">
          <div className="stat-item">
            <div className="stat-value">-</div>
            <div className="stat-label">Total Trades</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">-</div>
            <div className="stat-label">Win Rate</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">-</div>
            <div className="stat-label">Total PnL</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Performance Summary</span>
      </div>
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-value">{performance.total_trades}</div>
          <div className="stat-label">Total Trades</div>
        </div>
        <div className="stat-item">
          <div className="stat-value positive">{performance.winning_trades}</div>
          <div className="stat-label">Winning</div>
        </div>
        <div className="stat-item">
          <div className="stat-value negative">{performance.losing_trades}</div>
          <div className="stat-label">Losing</div>
        </div>
        <div className="stat-item">
          <div className={`stat-value ${performance.win_rate >= 50 ? 'positive' : 'negative'}`}>
            {performance.win_rate?.toFixed(1)}%
          </div>
          <div className="stat-label">Win Rate</div>
        </div>
        <div className="stat-item">
          <div className={`stat-value ${performance.total_pnl_pct >= 0 ? 'positive' : 'negative'}`}>
            {performance.total_pnl_pct >= 0 ? '+' : ''}{performance.total_pnl_pct?.toFixed(2)}%
          </div>
          <div className="stat-label">Total PnL</div>
        </div>
        <div className="stat-item">
          <div className={`stat-value ${performance.avg_pnl_pct >= 0 ? 'positive' : 'negative'}`}>
            {performance.avg_pnl_pct >= 0 ? '+' : ''}{performance.avg_pnl_pct?.toFixed(2)}%
          </div>
          <div className="stat-label">Avg PnL</div>
        </div>
        <div className="stat-item">
          <div className="stat-value positive">+{performance.best_trade?.toFixed(2)}%</div>
          <div className="stat-label">Best Trade</div>
        </div>
        <div className="stat-item">
          <div className="stat-value negative">{performance.worst_trade?.toFixed(2)}%</div>
          <div className="stat-label">Worst Trade</div>
        </div>
      </div>

      {performance.by_strategy && Object.keys(performance.by_strategy).length > 0 && (
        <div style={{ marginTop: '1.5rem' }}>
          <div style={{ 
            fontSize: '0.75rem', 
            color: 'var(--text-muted)', 
            marginBottom: '0.75rem',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}>
            By Strategy
          </div>
          <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
            {Object.entries(performance.by_strategy).map(([stratName, stats]) => (
              <div key={stratName} className="stat-item" style={{ textAlign: 'left', padding: '0.75rem' }}>
                <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>{stratName}</div>
                <div style={{ fontSize: '0.8rem', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Trades:</span>
                  <span>{stats.trades}</span>
                </div>
                <div style={{ fontSize: '0.8rem', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Win Rate:</span>
                  <span style={{ 
                    color: stats.win_rate >= 50 ? 'var(--accent-green)' : 'var(--accent-red)' 
                  }}>
                    {stats.win_rate?.toFixed(1)}%
                  </span>
                </div>
                <div style={{ fontSize: '0.8rem', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>PnL:</span>
                  <span style={{ 
                    color: stats.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' 
                  }}>
                    {stats.pnl >= 0 ? '+' : ''}{stats.pnl?.toFixed(2)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default PerformanceStats;
