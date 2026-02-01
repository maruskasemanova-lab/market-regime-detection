function RegimeIndicator({ regime }) {
  if (!regime) {
    return (
      <div className="card regime-indicator">
        <div className="card-title">Current Regime</div>
        <div className="regime-badge mixed">Loading...</div>
      </div>
    );
  }

  const regimeClass = regime.regime?.toLowerCase() || 'mixed';
  
  const regimeEmojis = {
    trending: '📈',
    choppy: '📊',
    mixed: '🔀'
  };

  return (
    <div className="card regime-indicator">
      <div className="card-title">Current Regime</div>
      <div className={`regime-badge ${regimeClass}`}>
        {regimeEmojis[regimeClass]} {regime.regime}
      </div>
      <p className="regime-description">
        {regime.description}
      </p>
      {regime.active_strategies && regime.active_strategies.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Active Strategies:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', justifyContent: 'center' }}>
            {regime.active_strategies.map(strat => (
              <span 
                key={strat} 
                className={`regime-tag ${regimeClass}`}
                style={{ fontSize: '0.7rem' }}
              >
                {strat}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default RegimeIndicator;
