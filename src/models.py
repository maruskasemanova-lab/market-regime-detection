"""
Market Regime Detection Models.
"""
import pandas as pd
import numpy as np
import pickle
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional

class BaseDetector:
    def __init__(self, name: str):
        self.name = name
        self.regime_map = {0: 'CHOPPY', 1: 'MIXED', 2: 'TRENDING'}
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class HMMDetector(BaseDetector):
    def __init__(self, n_components: int = 3, n_iter: int = 100, random_state: int = 42):
        super().__init__("HMM")
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type="full", 
            n_iter=n_iter, 
            random_state=random_state
        )
        self.scaler = StandardScaler()
        
    def fit_predict(self, df: pd.DataFrame, features: List[str] = ['returns', 'volatility_5d']) -> pd.DataFrame:
        results = []
        
        for symbol in df['symbol'].unique():
            sym_data = df[df['symbol'] == symbol].copy().sort_values('date')
            X = sym_data[features].dropna()
            
            if len(X) < self.n_components * 2:
                continue
                
            X_scaled = self.scaler.fit_transform(X)
            
            try:
                self.model.fit(X_scaled)
                hidden_states = self.model.predict(X_scaled)
                
                # Map states to regimes based on volatility (feature index 1 usually)
                # Assuming volatility is the second feature, or use returns magnitude
                # Here we use volatility if available, else abs(returns)
                
                vol_idx = 1 if 'volatility_5d' in features else 0
                state_means = []
                for s in range(self.n_components):
                    mask = hidden_states == s
                    if mask.sum() > 0:
                        # Use mean of the volatility feature for that state
                        mean_val = X_scaled[mask, vol_idx].mean() if 'volatility' in features[vol_idx] else np.abs(X_scaled[mask, 0]).mean()
                        state_means.append((s, mean_val))
                
                # Sort: Lowest metric -> Choppy, Highest -> Trending
                state_means.sort(key=lambda x: x[1])
                
                current_map = {}
                if len(state_means) == 3:
                    current_map[state_means[0][0]] = 'CHOPPY'
                    current_map[state_means[1][0]] = 'MIXED'
                    current_map[state_means[2][0]] = 'TRENDING'
                elif len(state_means) == 2:
                     current_map[state_means[0][0]] = 'CHOPPY'
                     current_map[state_means[1][0]] = 'TRENDING'
                else:
                    current_map[state_means[0][0]] = 'MIXED'

                # Align results
                aligned_results = pd.DataFrame({
                    'symbol': symbol,
                    'date': sym_data.loc[X.index, 'date'],
                    f'{self.name}_regime': [current_map.get(s, 'MIXED') for s in hidden_states],
                    f'{self.name}_state': hidden_states
                })
                results.append(aligned_results)
                
            except Exception as e:
                print(f"HMM fit failed for {symbol}: {e}")
                continue
                
        if not results:
            return pd.DataFrame()
        return pd.concat(results)

class GMMDetector(BaseDetector):
    def __init__(self, n_components: int = 3, random_state: int = 42):
        super().__init__("GMM")
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        
    def fit_predict(self, df: pd.DataFrame, features: List[str] = ['returns', 'volatility_5d', 'daily_range_pct']) -> pd.DataFrame:
        results = []
        for symbol in df['symbol'].unique():
            sym_data = df[df['symbol'] == symbol].copy().sort_values('date')
            X = sym_data[features].dropna()
            
            if len(X) < 5: continue
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
            clusters = self.model.predict(X_scaled)
            
            # Map based on volatility (feature index 1)
            state_metrics = []
            for c in range(self.n_components):
                mask = clusters == c
                if mask.sum() > 0:
                    state_metrics.append((c, X_scaled[mask, 1].mean()))
            
            state_metrics.sort(key=lambda x: x[1])
            
            current_map = {}
            if len(state_metrics) == 3:
                current_map[state_metrics[0][0]] = 'CHOPPY'
                current_map[state_metrics[1][0]] = 'MIXED'
                current_map[state_metrics[2][0]] = 'TRENDING'
            
            aligned_results = pd.DataFrame({
                'symbol': symbol,
                'date': sym_data.loc[X.index, 'date'],
                f'{self.name}_regime': [current_map.get(c, 'MIXED') for c in clusters]
            })
            results.append(aligned_results)
            
        if not results: return pd.DataFrame()
        return pd.concat(results)

class KMeansDetector(BaseDetector):
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        super().__init__("KMeans")
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        
    def fit_predict(self, df: pd.DataFrame, features: List[str] = ['returns', 'volatility_5d', 'daily_range_pct']) -> pd.DataFrame:
        results = []
        for symbol in df['symbol'].unique():
            sym_data = df[df['symbol'] == symbol].copy().sort_values('date')
            X = sym_data[features].dropna()
            
            if len(X) < 5: continue
            
            X_scaled = self.scaler.fit_transform(X)
            clusters = self.model.fit_predict(X_scaled)
            
            state_metrics = []
            for c in range(self.n_clusters):
                mask = clusters == c
                if mask.sum() > 0:
                    state_metrics.append((c, X_scaled[mask, 1].mean()))
            
            state_metrics.sort(key=lambda x: x[1])
            
            current_map = {}
            if len(state_metrics) == 3:
                current_map[state_metrics[0][0]] = 'CHOPPY'
                current_map[state_metrics[1][0]] = 'MIXED'
                current_map[state_metrics[2][0]] = 'TRENDING'
            
            aligned_results = pd.DataFrame({
                'symbol': symbol,
                'date': sym_data.loc[X.index, 'date'],
                f'{self.name}_regime': [current_map.get(c, 'MIXED') for c in clusters]
            })
            results.append(aligned_results)
            
        if not results: return pd.DataFrame()
        return pd.concat(results)

class EnsembleDetector:
    def __init__(self):
        self.detectors = [
            HMMDetector(),
            GMMDetector(),
            KMeansDetector()
        ]
        
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Run all detectors
        results = df[['symbol', 'date']].copy()
        
        for detector in self.detectors:
            # Select features appropriate for each model (defaults used in classes)
            det_res = detector.fit_predict(df)
            if not det_res.empty:
                results = results.merge(det_res, on=['symbol', 'date'], how='left')
        
        # Simple Voting
        regime_cols = [c for c in results.columns if '_regime' in c]
        
        def vote(row):
            votes = [row[c] for c in regime_cols if pd.notna(row[c])]
            if not votes: return 'MIXED'
            
            # Weighted voting could be added here
            counts = {
                'CHOPPY': votes.count('CHOPPY'),
                'MIXED': votes.count('MIXED'),
                'TRENDING': votes.count('TRENDING')
            }
            return max(counts, key=counts.get)
            
        results['Ensemble_regime'] = results.apply(vote, axis=1)
        return results
