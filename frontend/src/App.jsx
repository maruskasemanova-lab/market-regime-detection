import './index.css'
import Dashboard from './components/Dashboard'

function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🎯 Market Regime Trading System</h1>
          <div className="controls">
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Strategy API: <span style={{ color: 'var(--accent-green)' }}>localhost:8001</span>
            </span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginLeft: '1rem' }}>
              Backtest API: <span style={{ color: 'var(--accent-blue)' }}>localhost:8000</span>
            </span>
          </div>
        </div>
      </header>
      
      <main className="main-content">
        <Dashboard />
      </main>
      
      <footer style={{ 
        textAlign: 'center', 
        padding: '1rem', 
        color: 'var(--text-muted)',
        fontSize: '0.8rem',
        borderTop: '1px solid var(--border-color)'
      }}>
        Market Regime Detection • Strategies: Mean Reversion | Pullback | Momentum | Rotation | VWAP Magnet
      </footer>
    </div>
  )
}

export default App
