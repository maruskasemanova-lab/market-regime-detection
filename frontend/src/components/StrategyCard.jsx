function StrategyCard({ strategy, isActive, currentRegime, onToggle }) {
  if (!strategy) return null;

  const lastSignal = strategy.last_signal;
  
  return (
    <div 
      className={`strategy-card ${isActive ? 'active' : ''} ${!strategy.enabled ? 'disabled' : ''}`}
      onClick={() => onToggle(!strategy.enabled)}
    >
      <div className="strategy-header">
        <span className="strategy-name">{strategy.name}</span>
        <span className={`strategy-status ${isActive && strategy.enabled ? 'active' : ''}`}></span>
      </div>
      
      <div className="strategy-regimes">
        {strategy.allowed_regimes?.map(regime => (
          <span 
            key={regime} 
            className={`regime-tag ${regime.toLowerCase()}`}
            style={{ 
              opacity: currentRegime === regime ? 1 : 0.5,
              fontWeight: currentRegime === regime ? 700 : 400
            }}
          >
            {regime}
          </span>
        ))}
      </div>
      
      {lastSignal && (
        <div className="strategy-signal">
          <span className={`signal-type ${lastSignal.signal?.toLowerCase()}`}>
            {lastSignal.signal}
          </span>
          <span style={{ marginLeft: '0.5rem', color: 'var(--text-secondary)' }}>
            @ ${lastSignal.price?.toFixed(2)}
          </span>
          <div style={{ 
            marginTop: '0.25rem', 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.5rem' 
          }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Confidence:
            </span>
            <div className="confidence-bar">
              <div 
                className={`confidence-fill ${
                  lastSignal.confidence >= 80 ? 'high' : 
                  lastSignal.confidence >= 60 ? 'medium' : 'low'
                }`}
                style={{ width: `${lastSignal.confidence}%` }}
              ></div>
            </div>
            <span style={{ fontSize: '0.7rem' }}>
              {lastSignal.confidence?.toFixed(0)}%
            </span>
          </div>
        </div>
      )}
      
      {!lastSignal && (
        <div className="strategy-signal" style={{ color: 'var(--text-muted)' }}>
          No recent signals
        </div>
      )}
      
      <div style={{ 
        marginTop: '0.5rem', 
        fontSize: '0.7rem', 
        color: 'var(--text-muted)', 
        display: 'flex', 
        justifyContent: 'space-between' 
      }}>
        <span>Positions: {strategy.open_positions || 0}</span>
        <span>Signals: {strategy.total_signals || 0}</span>
      </div>
    </div>
  );
}

export default StrategyCard;
