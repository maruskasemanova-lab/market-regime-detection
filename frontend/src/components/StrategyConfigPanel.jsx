import { useState, useEffect } from 'react';

function StrategyConfigPanel({ strategies, tradingConfig, onUpdateStrategy, onUpdateTradingConfig }) {
  const [localStrategies, setLocalStrategies] = useState({});
  const [localConfig, setLocalConfig] = useState({
    regime_detection_minutes: 60,
    max_daily_loss: 300,
    max_trades_per_day: 3,
    trade_cooldown_bars: 15
  });
  const [expandedStrategy, setExpandedStrategy] = useState(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    if (strategies) {
      setLocalStrategies(strategies);
    }
  }, [strategies]);

  useEffect(() => {
    if (tradingConfig) {
      setLocalConfig(tradingConfig);
    }
  }, [tradingConfig]);

  const handleConfigChange = (field, value) => {
    setLocalConfig(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
  };

  const handleStrategyParamChange = (strategyKey, param, value) => {
    setLocalStrategies(prev => ({
      ...prev,
      [strategyKey]: {
        ...prev[strategyKey],
        [param]: parseFloat(value) || value
      }
    }));
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      const result = await onUpdateTradingConfig(localConfig);
      if (result) {
        setMessage({ type: 'success', text: 'Trading config saved!' });
      }
    } catch (e) {
      setMessage({ type: 'error', text: 'Failed to save config' });
    }
    setSaving(false);
    setTimeout(() => setMessage(null), 3000);
  };

  const saveStrategyParams = async (strategyKey) => {
    setSaving(true);
    try {
      const strategy = localStrategies[strategyKey];
      // Extract editable numeric params
      const params = {};
      Object.entries(strategy).forEach(([key, val]) => {
        if (typeof val === 'number' && !['open_positions', 'total_signals'].includes(key)) {
          params[key] = val;
        }
      });
      const result = await onUpdateStrategy(strategyKey, params);
      if (result) {
        setMessage({ type: 'success', text: `${strategy.name} params saved!` });
      }
    } catch (e) {
      setMessage({ type: 'error', text: 'Failed to save params' });
    }
    setSaving(false);
    setTimeout(() => setMessage(null), 3000);
  };

  const editableParams = [
    'entry_deviation_pct', 'min_confidence', 'volume_lookback', 'volume_exhaustion_ratio',
    'volume_stop_pct', 'trailing_stop_pct', 'consolidation_bars', 'volume_threshold',
    'consolidation_range_pct', 'breakout_pct', 'rr_ratio', 'pullback_threshold_pct',
    'ma_fast_period', 'ma_slow_period', 'volume_surge_ratio', 'rotation_threshold',
    'lookback_bars', 'magnet_threshold', 'atr_multiplier'
  ];

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">⚙️ Configuration</span>
      </div>
      
      {message && (
        <div style={{
          padding: '0.75rem',
          marginBottom: '1rem',
          borderRadius: '8px',
          background: message.type === 'success' ? 'rgba(46, 204, 113, 0.15)' : 'rgba(231, 76, 60, 0.15)',
          color: message.type === 'success' ? 'var(--accent-green)' : 'var(--accent-red)',
          fontSize: '0.85rem'
        }}>
          {message.text}
        </div>
      )}

      {/* Trading Configuration */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          Trading Settings
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
          <div className="config-field">
            <label>Regime Detection (min)</label>
            <input 
              type="number"
              value={localConfig.regime_detection_minutes}
              onChange={(e) => handleConfigChange('regime_detection_minutes', e.target.value)}
            />
          </div>
          <div className="config-field">
            <label>Max Daily Loss ($)</label>
            <input 
              type="number"
              value={localConfig.max_daily_loss}
              onChange={(e) => handleConfigChange('max_daily_loss', e.target.value)}
            />
          </div>
          <div className="config-field">
            <label>Max Trades/Day</label>
            <input 
              type="number"
              value={localConfig.max_trades_per_day}
              onChange={(e) => handleConfigChange('max_trades_per_day', e.target.value)}
            />
          </div>
          <div className="config-field">
            <label>Cooldown (bars)</label>
            <input 
              type="number"
              value={localConfig.trade_cooldown_bars}
              onChange={(e) => handleConfigChange('trade_cooldown_bars', e.target.value)}
            />
          </div>
        </div>
        <button 
          className="btn btn-primary" 
          onClick={saveConfig}
          disabled={saving}
          style={{ marginTop: '0.75rem', width: '100%' }}
        >
          {saving ? '⏳ Saving...' : '💾 Save Trading Config'}
        </button>
      </div>

      {/* Strategy Parameters */}
      <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
        Strategy Parameters
      </h4>
      {Object.entries(localStrategies).map(([key, strategy]) => (
        <div 
          key={key} 
          style={{
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            marginBottom: '0.5rem',
            overflow: 'hidden'
          }}
        >
          <div 
            onClick={() => setExpandedStrategy(expandedStrategy === key ? null : key)}
            style={{
              padding: '0.75rem 1rem',
              background: 'var(--bg-secondary)',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span style={{ fontWeight: 500 }}>{strategy.name}</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              {expandedStrategy === key ? '▲' : '▼'}
            </span>
          </div>
          
          {expandedStrategy === key && (
            <div style={{ padding: '1rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                {Object.entries(strategy).map(([param, value]) => {
                  if (!editableParams.includes(param)) return null;
                  return (
                    <div key={param} className="config-field">
                      <label style={{ fontSize: '0.75rem' }}>
                        {param.replace(/_/g, ' ')}
                      </label>
                      <input 
                        type="number"
                        step="0.1"
                        value={value}
                        onChange={(e) => handleStrategyParamChange(key, param, e.target.value)}
                      />
                    </div>
                  );
                })}
              </div>
              <button 
                className="btn btn-success" 
                onClick={() => saveStrategyParams(key)}
                disabled={saving}
                style={{ marginTop: '0.75rem', width: '100%' }}
              >
                💾 Save {strategy.name} Params
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default StrategyConfigPanel;
