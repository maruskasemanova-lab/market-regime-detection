"""
Feature engineering for market regime detection.
"""
import pandas as pd
import numpy as np
from datetime import time
from typing import List, Optional

class FeatureEngineer:
    def __init__(self):
        pass
        
    def calculate_daily_metrics(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate core daily metrics like returns, range, volatility."""
        df = daily_df.copy()
        
        if 'date' not in df.columns and 'ts_event' in df.columns:
            df['date'] = df['ts_event'].dt.date
            
        df = df.sort_values(['symbol', 'date'])
        
        # Basic Price Action
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df.groupby('symbol')['close'].shift(1))
        
        # Range Metrics
        df['daily_range'] = df['high'] - df['low']
        df['daily_range_pct'] = (df['daily_range'] / df['open']) * 100
        df['net_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100
        
        # Trend Efficiency: |Close-Open| / (High-Low)
        # 1.0 = perfect trend (close at high/low), 0.0 = perfect chop (close at open)
        df['trend_efficiency'] = (np.abs(df['close'] - df['open']) / df['daily_range']).fillna(0).clip(0, 1)
        
        # Volatility (Rolling Standard Deviation of Returns)
        df['volatility_5d'] = df.groupby('symbol')['returns'].transform(lambda x: x.rolling(5, min_periods=2).std())
        df['volatility_10d'] = df.groupby('symbol')['returns'].transform(lambda x: x.rolling(10, min_periods=3).std())
        df['vol_ratio'] = df['volatility_5d'] / (df['volatility_10d'] + 1e-6)
        
        # Gaps
        df['prev_close'] = df.groupby('symbol')['close'].shift(1)
        df['gap_pct'] = ((df['open'] - df['prev_close']) / df['prev_close'] * 100).fillna(0)
        
        # Momentum
        df['prev_trend_eff'] = df.groupby('symbol')['trend_efficiency'].shift(1)
        
        return df

    def calculate_intraday_metrics(self, intraday_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics derived from intraday data (e.g., Opening Range)."""
        df = intraday_df.copy()
        if 'time' not in df.columns:
            df['time'] = df['ts_event'].dt.time
            df['date'] = df['ts_event'].dt.date
            
        # Opening Range definitions (UTC times for US Market)
        # Market Open: 14:30 UTC
        or_start = time(14, 30)
        or_15_end = time(14, 45)
        or_30_end = time(15, 0)
        or_60_end = time(15, 30)
        
        # 15-min Opening Range
        or_15_mask = (df['time'] >= or_start) & (df['time'] < or_15_end)
        or_15 = df[or_15_mask].groupby(['symbol', 'date']).agg({
            'high': 'max', 'low': 'min', 'open': 'first', 'close': 'last', 'volume': 'sum'
        }).reset_index()
        or_15.columns = ['symbol', 'date', 'or15_high', 'or15_low', 'or15_open', 'or15_close', 'or15_volume']
        or_15['or15_range_pct'] = (or_15['or15_high'] - or_15['or15_low']) / or_15['or15_open'] * 100
        
        # First Hour Volume
        fh_mask = (df['time'] >= or_start) & (df['time'] < or_60_end)
        fh = df[fh_mask].groupby(['symbol', 'date']).agg({'volume': 'sum'}).reset_index()
        fh.columns = ['symbol', 'date', 'first_hour_volume']
        
        # Merge metrics
        metrics = or_15.merge(fh, on=['symbol', 'date'], how='left')
        
        return metrics

    def prepare_dataset(self, daily_df: pd.DataFrame, intraday_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Combine daily and intraday features into a final dataset."""
        daily_feats = self.calculate_daily_metrics(daily_df)
        
        if intraday_df is not None:
            intraday_feats = self.calculate_intraday_metrics(intraday_df)
            combined = daily_feats.merge(intraday_feats, on=['symbol', 'date'], how='left')
            
            # Relative Volume (RVOL)
            # Need strict alignment, simpler logic used here for brevity
            combined['rvol_first_hour'] = combined['first_hour_volume'] / combined.groupby('symbol')['first_hour_volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            
            return combined
        
        return daily_feats
