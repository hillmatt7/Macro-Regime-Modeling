# generate_demo_data.py
"""
Generate demo CSV data for testing the macro regime modelling pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_demo_macro_data(start_date="2014-01-01", end_date="2024-01-01", n_features=20):
    """Generate synthetic macro data with regime-like patterns"""
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Define regime parameters
    n_regimes = 3
    regime_means = [
        np.array([0, 0, 0]),      # Regime 0: Normal
        np.array([2, -1, 0.5]),   # Regime 1: Bull market
        np.array([-2, 2, -0.5])   # Regime 2: Bear market
    ]
    
    regime_stds = [1.0, 1.5, 2.0]  # Different volatilities
    
    # Generate regime sequence with persistence
    regime_sequence = []
    current_regime = 0
    regime_duration = 0
    
    for i in range(n_days):
        # Randomly switch regimes with some persistence
        if regime_duration > np.random.poisson(100):  # Average 100 days per regime
            current_regime = np.random.choice([0, 1, 2])
            regime_duration = 0
        
        regime_sequence.append(current_regime)
        regime_duration += 1
    
    regime_sequence = np.array(regime_sequence)
    
    # Generate base features
    data = {'date': dates}
    
    # Core macro features with regime influence
    for i in range(min(n_features, 3)):
        feature_data = np.zeros(n_days)
        for day in range(n_days):
            regime = regime_sequence[day]
            feature_data[day] = np.random.normal(
                regime_means[regime][i % 3],
                regime_stds[regime]
            )
        
        # Add trend and autocorrelation
        trend = np.linspace(0, 1, n_days) * 0.1
        feature_data = feature_data + trend
        
        # Smooth with moving average
        window = 5
        feature_data = pd.Series(feature_data).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        data[f'macro_feature_{i}'] = feature_data
    
    # Add market-like features
    # VIX-like volatility indicator
    vix_base = 15 + 10 * np.where(regime_sequence == 2, 1, 0) + np.random.normal(0, 3, n_days)
    data['volatility_index'] = np.maximum(vix_base, 10)
    
    # Yield-like features
    data['short_yield'] = 2 + 0.5 * regime_sequence + np.random.normal(0, 0.2, n_days)
    data['long_yield'] = 3.5 + 0.3 * regime_sequence + np.random.normal(0, 0.15, n_days)
    data['term_spread'] = data['long_yield'] - data['short_yield']
    
    # Economic indicators
    data['growth_indicator'] = np.where(regime_sequence == 1, 3, np.where(regime_sequence == 2, -1, 1)) + np.random.normal(0, 0.5, n_days)
    data['inflation_proxy'] = 2 + 0.5 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 0.3, n_days)
    
    # Market returns
    returns = np.where(regime_sequence == 1, 0.0005, np.where(regime_sequence == 2, -0.0003, 0.0001))
    returns += np.random.normal(0, 0.01, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    data['market_price'] = prices
    data['market_returns'] = returns
    
    # Technical indicators
    data['rsi'] = 50 + 20 * np.where(regime_sequence == 1, 1, np.where(regime_sequence == 2, -1, 0)) + np.random.normal(0, 10, n_days)
    data['rsi'] = np.clip(data['rsi'], 0, 100)
    
    # Add remaining random features if needed
    for i in range(11, n_features):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_days)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic missing values
    for col in df.columns:
        if col != 'date':
            # Set some values to NaN
            mask = np.random.random(n_days) < 0.02  # 2% missing
            df.loc[mask, col] = np.nan
    
    return df, regime_sequence

def main():
    """Generate and save demo data"""
    print("Generating demo macro data...")
    
    # Generate data
    df, true_regimes = generate_demo_macro_data(
        start_date="2014-01-01",
        end_date="2024-01-01",
        n_features=20
    )
    
    # Save data
    df.to_csv("demo_macro_data.csv", index=False)
    print(f"✓ Saved demo_macro_data.csv with {len(df)} rows and {len(df.columns)} columns")
    
    regime_df = pd.DataFrame({
        'date': df['date'],
        'true_regime': true_regimes
    })
    regime_df.to_csv("demo_true_regimes.csv", index=False)
    print(f"✓ Saved demo_true_regimes.csv for comparison")
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Features: {[col for col in df.columns if col != 'date']}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\nTrue regime distribution:")
    unique, counts = np.unique(true_regimes, return_counts=True)
    for regime, count in zip(unique, counts):
        print(f"  Regime {regime}: {count} days ({count/len(true_regimes)*100:.1f}%)")

if __name__ == "__main__":
    main()
