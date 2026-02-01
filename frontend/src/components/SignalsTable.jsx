function SignalsTable({ signals }) {
  if (!signals || signals.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <span className="card-title">Recent Signals</span>
        </div>
        <div className="empty-state">
          No signals yet. Run the backtest to generate signals.
        </div>
      </div>
    );
  }

  // Show most recent first
  const sortedSignals = [...signals].reverse().slice(0, 20);

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Recent Signals</span>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
          {signals.length} total
        </span>
      </div>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Strategy</th>
              <th>Signal</th>
              <th>Price</th>
              <th>SL</th>
              <th>TP</th>
              <th>Confidence</th>
              <th>Reasoning</th>
            </tr>
          </thead>
          <tbody>
            {sortedSignals.map((signal, index) => (
              <tr key={index}>
                <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {formatTime(signal.timestamp)}
                </td>
                <td>
                  <strong>{signal.strategy}</strong>
                </td>
                <td>
                  <span className={`signal-type ${signal.signal?.toLowerCase()}`}>
                    {signal.signal}
                  </span>
                </td>
                <td style={{ fontFamily: 'monospace' }}>
                  ${signal.price?.toFixed(2)}
                </td>
                <td style={{ fontFamily: 'monospace', color: 'var(--accent-red)' }}>
                  ${signal.stop_loss?.toFixed(2) || '-'}
                </td>
                <td style={{ fontFamily: 'monospace', color: 'var(--accent-green)' }}>
                  ${signal.take_profit?.toFixed(2) || '-'}
                </td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div className="confidence-bar">
                      <div 
                        className={`confidence-fill ${
                          signal.confidence >= 80 ? 'high' : 
                          signal.confidence >= 60 ? 'medium' : 'low'
                        }`}
                        style={{ width: `${signal.confidence}%` }}
                      ></div>
                    </div>
                    <span style={{ fontSize: '0.8rem' }}>
                      {signal.confidence?.toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td style={{ 
                  fontSize: '0.75rem', 
                  color: 'var(--text-secondary)',
                  maxWidth: '300px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}>
                  {signal.reasoning || '-'}
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
      minute: '2-digit',
      second: '2-digit'
    });
  } catch {
    return timestamp;
  }
}

export default SignalsTable;
