"""
Main script to run Market Regime Detection comparison.
"""
import pandas as pd
from pathlib import Path
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models import HMMDetector, GMMDetector, KMeansDetector, EnsembleDetector

# Config
DATA_DIR = "/Users/hotovo/.gemini/antigravity/scratch/ibkr-l2-script/databento_data"
DAILY_FILE = "january_2026_daily.parquet"
INTRADAY_FILE = "january_2026_1min.parquet"

def main():
    print("=" * 60)
    print("MARKET REGIME DETECTION PROJECT")
    print("=" * 60)
    
    # 1. Load Data
    print("\n1. Loading Data...")
    loader = DataLoader(DATA_DIR)
    daily, intraday = loader.load_databento_parquet(DAILY_FILE, INTRADAY_FILE)
    print(f"Loaded {len(daily)} daily records and {len(intraday)} intraday records.")
    
    # 2. Feature Engineering
    print("\n2. Processing Features...")
    engineer = FeatureEngineer()
    df = engineer.prepare_dataset(daily, intraday)
    print(f"Feature dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # 3. Models Execution
    print("\n3. Running Detectors...")
    
    # HMM
    print("Running HMM...")
    hmm = HMMDetector()
    hmm_res = hmm.fit_predict(df)
    
    # GMM
    print("Running GMM...")
    gmm = GMMDetector()
    gmm_res = gmm.fit_predict(df)
    
    # Ensemble
    print("Running Ensemble...")
    ens = EnsembleDetector()
    ens_res = ens.fit_predict(df)
    
    # 4. Compare and Save
    print("\n4. Consolidating Results...")
    final = df[['symbol', 'date', 'trend_efficiency', 'daily_range_pct']].copy()
    
    # Define ground truth for accuracy check
    def actual_regime(eff):
        if eff >= 0.55: return 'TRENDING'
        elif eff <= 0.30: return 'CHOPPY'
        else: return 'MIXED'
        
    final['Actual'] = final['trend_efficiency'].apply(actual_regime)
    
    # Merge predictions
    if not hmm_res.empty:
        final = final.merge(hmm_res[['symbol', 'date', 'HMM_regime']], on=['symbol', 'date'], how='left')
    if not gmm_res.empty:
        final = final.merge(gmm_res[['symbol', 'date', 'GMM_regime']], on=['symbol', 'date'], how='left')
    if not ens_res.empty:
        final = final.merge(ens_res[['symbol', 'date', 'Ensemble_regime']], on=['symbol', 'date'], how='left')
    
    # Calculate basic accuracy
    total = len(final)
    for model_name in ['HMM', 'GMM', 'Ensemble']:
        col = f'{model_name}_regime'
        if col in final.columns:
            correct = len(final[final[col] == final['Actual']])
            acc = correct / total * 100
            print(f"{model_name} Accuracy: {acc:.1f}%")
            
    # Save
    output_path = Path("results.csv")
    final.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path.absolute()}")

if __name__ == "__main__":
    main()
