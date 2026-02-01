function TradesHistory({ trades }) {
  if (!trades || trades.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <span className="card-title">Trade History</span>
        </div>
        <div className="empty-state">
          No trades yet. Signals will execute trades when conditions are met.
        </div>
      </div>
    );
  }

  // Show most recent first
  const sortedTrades = [...trades].reverse().slice(0, 20);

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Trade History</span>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
          {trades.length} total trades
        </span>
      </div>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Strategy</th>
              <th>Side</th>
              <th>Entry</th>
              <th>Exit</th>
              <th>PnL %</th>
              <th>Exit Reason</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {sortedTrades.map((trade) => (
              <tr key={trade.id}>
                <td style={{ color: 'var(--text-muted)' }}>#{trade.id}</td>
                <td><strong>{trade.strategy}</strong></td>
                <td>
                  <span className={`position-side ${trade.side}`}>
                    {trade.side?.toUpperCase()}
                  </span>
                </td>
                <td style={{ fontFamily: 'monospace' }}>
                  ${trade.entry_price?.toFixed(2)}
                </td>
                <td style={{ fontFamily: 'monospace' }}>
                  ${trade.exit_price?.toFixed(2)}
                </td>
                <td style={{ 
                  fontFamily: 'monospace', 
                  fontWeight: 600,
                  color: trade.pnl_pct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'
                }}>
                  {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct?.toFixed(2)}%
                </td>
                <td>
                  <span style={{
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    background: getExitReasonColor(trade.exit_reason).bg,
                    color: getExitReasonColor(trade.exit_reason).text
                  }}>
                    {formatExitReason(trade.exit_reason)}
                  </span>
                </td>
                <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {formatTime(trade.exit_time)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatTime(timestamp) {
  if (!timestamp) return '-';
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit'
    });
  } catch {
    return timestamp;
  }
}

function formatExitReason(reason) {
  const reasons = {
    'stop_loss': '🛑 Stop Loss',
    'take_profit': '🎯 Take Profit',
    'trailing_stop': '📈 Trailing SL',
    'signal': '📊 Signal'
  };
  return reasons[reason] || reason;
}

function getExitReasonColor(reason) {
  const colors = {
    'stop_loss': { bg: 'rgba(239, 68, 68, 0.2)', text: 'var(--accent-red)' },
    'take_profit': { bg: 'rgba(16, 185, 129, 0.2)', text: 'var(--accent-green)' },
    'trailing_stop': { bg: 'rgba(59, 130, 246, 0.2)', text: 'var(--accent-blue)' },
    'signal': { bg: 'rgba(139, 92, 246, 0.2)', text: 'var(--accent-purple)' }
  };
  return colors[reason] || { bg: 'var(--bg-secondary)', text: 'var(--text-secondary)' };
}

export default TradesHistory;
