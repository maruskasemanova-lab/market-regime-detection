function PositionsList({ positions }) {
  const positionsList = Object.entries(positions || {});

  if (positionsList.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <span className="card-title">Open Positions</span>
        </div>
        <div className="empty-state" style={{ padding: '1rem' }}>
          No open positions
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Open Positions</span>
        <span style={{ 
          background: 'var(--accent-blue)', 
          padding: '2px 8px', 
          borderRadius: '10px', 
          fontSize: '0.75rem' 
        }}>
          {positionsList.length}
        </span>
      </div>
      
      {positionsList.map(([strategyName, position]) => (
        <div key={strategyName} className={`position-card ${position.side}`}>
          <div className="position-header">
            <span className="position-strategy">{position.strategy}</span>
            <span className={`position-side ${position.side}`}>
              {position.side?.toUpperCase()}
            </span>
          </div>
          
          <div className="position-details">
            <div className="position-detail">
              <span className="position-detail-label">Entry</span>
              <span style={{ fontFamily: 'monospace' }}>
                ${position.entry_price?.toFixed(2)}
              </span>
            </div>
            <div className="position-detail">
              <span className="position-detail-label">Stop Loss</span>
              <span style={{ fontFamily: 'monospace', color: 'var(--accent-red)' }}>
                ${position.stop_loss?.toFixed(2)}
              </span>
            </div>
            <div className="position-detail">
              <span className="position-detail-label">Take Profit</span>
              <span style={{ fontFamily: 'monospace', color: 'var(--accent-green)' }}>
                ${position.take_profit?.toFixed(2)}
              </span>
            </div>
          </div>
          
          {position.trailing_stop_active && (
            <div style={{ 
              marginTop: '0.5rem', 
              padding: '0.25rem 0.5rem', 
              background: 'rgba(59, 130, 246, 0.1)', 
              borderRadius: '4px',
              fontSize: '0.75rem',
              color: 'var(--accent-blue)'
            }}>
              🎯 Trailing Stop @ ${position.trailing_stop_price?.toFixed(2) || 'Active'}
            </div>
          )}
          
          <div style={{ 
            marginTop: '0.5rem', 
            display: 'flex', 
            justifyContent: 'space-between',
            fontSize: '0.8rem'
          }}>
            <span style={{ color: 'var(--text-muted)' }}>
              PnL:
            </span>
            <span style={{ 
              fontWeight: 600, 
              color: position.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' 
            }}>
              {position.pnl >= 0 ? '+' : ''}{position.pnl?.toFixed(2)}%
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default PositionsList;
