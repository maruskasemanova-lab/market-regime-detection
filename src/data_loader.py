"""
Data loading and preprocessing utilities.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_databento_parquet(self, daily_file: str, intraday_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load daily and intraday parquet files from Databento.
        
        Args:
            daily_file: Filename of the daily data
            intraday_file: Filename of the intraday data
            
        Returns:
            Tuple containing (daily_df, intraday_df)
        """
        daily_path = self.data_dir / daily_file
        intraday_path = self.data_dir / intraday_file
        
        if not daily_path.exists() or not intraday_path.exists():
            raise FileNotFoundError(f"Data files not found in {self.data_dir}")
            
        daily = pd.read_parquet(daily_path)
        intraday = pd.read_parquet(intraday_path)
        
        # Reset index if ts_event is index
        if 'ts_event' not in daily.columns and 'ts_event' in daily.index.names:
            daily = daily.reset_index()
        if 'ts_event' not in intraday.columns and 'ts_event' in intraday.index.names:
            intraday = intraday.reset_index()
            
        return daily, intraday

    def clean_daily_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning for daily data."""
        df = df.copy()
        if 'ts_event' in df.columns:
            df['date'] = df['ts_event'].dt.date
        df = df.sort_values(['symbol', 'date'])
        return df
