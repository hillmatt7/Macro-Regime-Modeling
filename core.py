# core.py
"""
Core components: Data ingestion, preprocessing, GMM fitting, and selection
Consolidates Steps 1-4 of the pipeline with COMPLETE GMM implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import json
import yfinance as yf
import pandas_datareader as pdr
from dotenv import load_dotenv
import warnings
from optimization_utils import (
    parallel_bootstrap_stability, 
    optimized_gmm_fit, 
    compute_icl_bic_aic,
    MemoryMonitor,
    fast_elbow_detection
)

warnings.filterwarnings('ignore')


class DataIngestionPreprocessing:
    """Combines data ingestion (Step 1) and preprocessing (Step 2)"""
    
    def __init__(self):
        load_dotenv()
        
        # Create directories
        self.raw_snapshot_dir = "raw_snapshots"
        self.processed_snapshot_dir = "processed_snapshots"
        os.makedirs(self.raw_snapshot_dir, exist_ok=True)
        os.makedirs(self.processed_snapshot_dir, exist_ok=True)
        
        self.transform_params = {}
        
        # FRED API configuration
        self.fred_api_key = os.getenv('FRED_API_KEY', 'demo_key')
        
        # Define FRED series to always fetch
        self.fred_series = {
            'gdp': 'GDP',
            'unemployment': 'UNRATE', 
            'cpi': 'CPIAUCSL',
            'pce': 'PCE',
            'fed_funds': 'DFF',
            'treasury_30y': 'DGS30',         # Daily
            'treasury_10y': 'DGS10',         # Daily
            'treasury_5y': 'DGS5',           # Daily
            'treasury_2y': 'DGS2',           # Daily
            'industrial_prod': 'INDPRO',
            'retail_sales': 'RSXFS',
            'consumer_sentiment': 'UMCSENT',
            'm2_money': 'M2SL',
            'housing_starts': 'HOUST'
        }
        
        # Define market tickers to always fetch
        self.market_tickers = {
            'spy': 'SPY',
            'es': 'ES=F',  # E-mini S&P futures
            'nq': 'NQ=F',  # E-mini Nasdaq futures
            'vix': '^VIX',
            'dxy': 'DX-Y.NYB',  # Dollar index
            'tlt': 'TLT',  # 20+ year treasuries
            'gld': 'GLD',  # Gold
            'uso': 'USO'   # Oil
        }
    
    def ingest_and_preprocess(self, window_start: str, window_end: str,
                            user_csv_path: Optional[str] = None,
                            use_demo_data: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Ingest data from multiple sources and preprocess
        Always fetches FRED + market data, optionally adds user CSV
        """
        # Step 1: Collect data from all sources with validation
        all_data = []

        # Always fetch FRED data
        fred_data = self._fetch_fred_data(window_start, window_end)
        fred_data = self._validate_and_clean_data(fred_data, "FRED")
        if not fred_data.empty:
            all_data.append(fred_data)

        # Always fetch market OHLCV data
        market_data = self._fetch_market_data(window_start, window_end)
        market_data = self._validate_and_clean_data(market_data, "Market")
        if not market_data.empty:
            all_data.append(market_data)

        # Add user CSV if provided
        if user_csv_path:
            try:
                user_data = self._load_user_csv(user_csv_path, window_start, window_end)
                user_data = self._validate_and_clean_data(user_data, "User CSV")
                if not user_data.empty:
                    all_data.append(user_data)
            except Exception as e:
                print(f"Warning: Could not load user CSV: {e}")

        # Use demo data if requested or if insufficient real data
        if use_demo_data or len(all_data) == 0:
            demo_data = self._generate_demo_data(window_start, window_end)
            demo_data = self._validate_and_clean_data(demo_data, "Demo")
            all_data = [demo_data]

        # Merge all data sources with improved logic
        if len(all_data) == 0:
            raise ValueError("No data available from any source")

        # Start with the dataset that has the most recent/complete date range
        # Usually market data is more complete
        if len(all_data) > 1:
            # Sort by date range completeness (prefer datasets with more recent data)
            all_data = sorted(all_data, key=lambda x: (x['date'].max(), len(x)), reverse=True)

        raw_df = all_data[0].copy()
        print(f"Base dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]-1} features")

        for i, df in enumerate(all_data[1:], 1):
            before_cols = len(raw_df.columns)
            before_rows = len(raw_df)

            # Use inner join to only keep dates where we have data from both sources
            # This prevents the coverage issues
            raw_df = pd.merge(raw_df, df, on='date', how='inner', suffixes=('', f'_dup{i}'))

            # Remove duplicate columns
            dup_cols = [col for col in raw_df.columns if col.endswith(f'_dup{i}')]
            if dup_cols:
                raw_df = raw_df.drop(dup_cols, axis=1)

            after_cols = len(raw_df.columns)
            after_rows = len(raw_df)
            print(f"After merging source {i+1}: {after_rows} rows ({before_rows - after_rows} dropped), "
                  f"{after_cols - before_cols} new columns")

        # Sort by date and filter to exact window
        raw_df = raw_df.sort_values('date')
        start_dt = pd.to_datetime(window_start)
        end_dt = pd.to_datetime(window_end)
        raw_df = raw_df[(raw_df['date'] >= start_dt) & (raw_df['date'] <= end_dt)]

        print(f"Final dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]-1} features")

        # Check if we have enough data
        if len(raw_df) < 252:  # Less than 1 year of trading days
            print(f"Warning: Only {len(raw_df)} days of data available (less than 1 year)")
        
        # Validate coverage
        self._validate_coverage(raw_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(raw_df)
        
        # Add technical indicators
        processed_df = self._add_technical_indicators(processed_df)
        
        # Persist raw snapshot
        self._persist_raw_snapshot(processed_df, window_start, window_end)
        
        # Step 2: Preprocessing
        normalised_tensor = self._preprocess_data(processed_df, window_end)
        
        return processed_df, normalised_tensor
    
    def _fetch_fred_data(self, window_start: str, window_end: str) -> pd.DataFrame:
        """Fetch macroeconomic data from FRED with proper forward fill strategy"""
        if self.fred_api_key == 'demo_key':
            print("Using simulated FRED data (set FRED_API_KEY in .env for real data)")
            return self._simulate_fred_data(window_start, window_end)
        
        try:
            all_series = {}
            failed_series = []
            
            # Create target date range (business days only)
            start_dt = pd.to_datetime(window_start)
            end_dt = pd.to_datetime(window_end)
            target_dates = pd.bdate_range(start=start_dt, end=end_dt, freq='B')
            
            for name, series_id in self.fred_series.items():
                try:
                    # Fetch the raw data (with some buffer before start date for forward fill)
                    buffer_start = start_dt - pd.DateOffset(years=1)  # Get 1 year of history for context
                    
                    data = pdr.get_data_fred(
                        series_id, 
                        start=buffer_start, 
                        end=end_dt
                    )
                    
                    if data is not None and not data.empty:
                        # Handle different DataFrame structures
                        if isinstance(data, pd.DataFrame):
                            if len(data.columns) == 1:
                                series_data = data.iloc[:, 0]
                            else:
                                if series_id in data.columns:
                                    series_data = data[series_id]
                                else:
                                    series_data = data.iloc[:, 0]
                        elif isinstance(data, pd.Series):
                            series_data = data
                        
                        # Remove NaN values
                        series_data = series_data.dropna()
                        
                        if len(series_data) > 0:
                            # Convert to DataFrame for resampling
                            temp_df = pd.DataFrame({
                                'date': series_data.index,
                                'value': series_data.values
                            })
                            temp_df['date'] = pd.to_datetime(temp_df['date'])
                            temp_df = temp_df.set_index('date').sort_index()
                            
                            # CRITICAL: Forward fill to target business days
                            # This ensures each business day has the most recent known value
                            resampled = temp_df.reindex(target_dates, method='ffill')
                            
                            # Store the forward-filled series
                            all_series[f'fred_{name}'] = resampled['value']
                            
                            # Log the resampling info
                            original_count = len(series_data)
                            final_count = len(resampled.dropna())
                            print(f"  FRED {name}: {original_count} releases → {final_count} business days")
                        else:
                            failed_series.append(name)
                    else:
                        failed_series.append(name)
                        
                except Exception as e:
                    print(f"Could not fetch {name} ({series_id}): {str(e)}")
                    failed_series.append(name)
            
            if failed_series:
                print(f"Failed to fetch FRED series: {failed_series}")
            
            if all_series:
                # Create final DataFrame
                fred_df = pd.DataFrame(all_series, index=target_dates)
                fred_df = fred_df.reset_index()
                fred_df = fred_df.rename(columns={'index': 'date'})
                
                # Remove any rows where ALL FRED series are NaN
                # (this happens at the very beginning if no series have data yet)
                fred_cols = [col for col in fred_df.columns if col.startswith('fred_')]
                fred_df = fred_df.dropna(subset=fred_cols, how='all')
                
                print(f"FRED data forward-filled to {len(fred_df)} business days")
                return fred_df
                
        except Exception as e:
            print(f"FRED fetch error: {str(e)}")
        
        return None
    
    def _fetch_market_data(self, window_start: str, window_end: str) -> pd.DataFrame:
        """Fetch OHLCV data with proper business day alignment"""
        all_data = {}

        # Create target business day range
        start_dt = pd.to_datetime(window_start)
        end_dt = pd.to_datetime(window_end)
        target_dates = pd.bdate_range(start=start_dt, end=end_dt, freq='B')

        for name, ticker in self.market_tickers.items():
            try:
                # Download with proper error handling
                data = yf.download(
                    ticker,
                    start=window_start,
                    end=end_dt,
                    progress=False,
                    auto_adjust=True,
                    prepost=False,
                    threads=True
                )

                if data.empty:
                    print(f"No data returned for {ticker}")
                    continue

                # Handle MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == ticker else f"{col[0]}_{col[1]}"
                                    for col in data.columns]

                # Ensure we have the expected columns
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                available_cols = [col for col in expected_cols if col in data.columns]

                if not available_cols:
                    print(f"No expected columns found for {ticker}")
                    continue

                # Reindex to target business days with forward fill
                data = data.reindex(target_dates, method='ffill')

                # Add OHLCV data with proper naming
                for col in available_cols:
                    if col in data.columns:
                        all_data[f'{name}_{col.lower()}'] = data[col]

                # Add calculated features
                if 'Close' in data.columns:
                    close_prices = data['Close']

                    # Returns
                    filled_close = close_prices.fillna(method='ffill')
                    all_data[f'{name}_returns'] = filled_close.pct_change()

                    # Volatility, 20-day rolling
                    returns = filled_close.pct_change()
                    all_data[f'{name}_volatility'] = returns.rolling(20).std() * np.sqrt(252)

                    # RSI
                    all_data[f'{name}_rsi'] = self._calculate_rsi(filled_close)

            except Exception as e:
                print(f"Could not fetch {ticker}: {str(e)}")
                continue

        if all_data:
            market_df = pd.DataFrame(all_data, index=target_dates)
            market_df = market_df.reset_index()
            market_df = market_df.rename(columns={'index': 'date'})

            # Forward fill NaNs
            market_cols = [col for col in market_df.columns if col != 'date']
            market_df[market_cols] = market_df[market_cols].fillna(method='ffill')

            print(f"✓ Market data aligned to {len(market_df)} business days")
            return market_df

        return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Validate and clean data from any source"""
        if df is None or df.empty:
            print(f"No data from {source_name}")
            return pd.DataFrame()
        
        # Ensure date column exists
        if 'date' not in df.columns:
            if df.index.name in ['Date', 'DATE', 'date'] or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: 'date'})
            else:
                print(f"Warning: No date column found in {source_name}")
                return pd.DataFrame()
        
        # Ensure date is datetime
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            print(f"Could not convert date column in {source_name}: {e}")
            return pd.DataFrame()
        
        # Remove any non-numeric columns except date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'date' not in numeric_cols:
            numeric_cols = ['date'] + numeric_cols
        
        df = df[numeric_cols]
        
        # Remove columns that are all NaN
        df = df.dropna(axis=1, how='all')
        
        # Sort by date
        df = df.sort_values('date')
        
        print(f"Validated {source_name}: {df.shape[0]} rows, {df.shape[1]-1} features")
        
        return df
    
    def _load_user_csv(self, csv_path: str, window_start: str, window_end: str) -> pd.DataFrame:
        """Load user-provided CSV data"""
        df = pd.read_csv(csv_path)
        
        # Find date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df['date'] = pd.to_datetime(df[date_col])
            if date_col != 'date':
                df = df.drop(date_col, axis=1)
        else:
            # Assume first column is date
            df.columns = ['date'] + list(df.columns[1:])
            df['date'] = pd.to_datetime(df['date'])
        
        # Filter to window
        start_dt = pd.to_datetime(window_start)
        end_dt = pd.to_datetime(window_end)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        # Prefix user columns to avoid conflicts
        user_cols = [col for col in df.columns if col != 'date']
        rename_dict = {col: f'user_{col}' for col in user_cols}
        df = df.rename(columns=rename_dict)
        
        return df
    
    def _simulate_fred_data(self, window_start: str, window_end: str) -> pd.DataFrame:
        """Simulate FRED-like macro data for demo purposes"""
        dates = pd.date_range(start=window_start, end=window_end, freq='D')
        n_days = len(dates)
        
        # Simulate macro indicators with realistic patterns
        data = {
            'date': dates,
            'fred_gdp': 2 + 0.03 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 0.1, n_days),
            'fred_unemployment': 5 + 2 * np.sin(np.arange(n_days) * 2 * np.pi / 1000) + np.random.normal(0, 0.2, n_days),
            'fred_cpi': 2 + 0.5 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.cumsum(np.random.normal(0, 0.01, n_days)),
            'fred_fed_funds': 2 + np.sin(np.arange(n_days) * 2 * np.pi / 500) + np.random.normal(0, 0.1, n_days),
            'fred_industrial_prod': 100 + 5 * np.sin(np.arange(n_days) * 2 * np.pi / 250) + np.random.normal(0, 1, n_days)
        }
        
        return pd.DataFrame(data)
    
    def _generate_demo_data(self, window_start: str, window_end: str) -> pd.DataFrame:
        """Generate complete demo dataset with regime-like patterns"""
        dates = pd.date_range(start=window_start, end=window_end, freq='D')
        n_days = len(dates)
        
        # Generate regime sequence
        regime_sequence = []
        current_regime = 0
        regime_duration = 0
        
        for i in range(n_days):
            if regime_duration > np.random.poisson(100):  # Average 100 days per regime
                current_regime = np.random.choice([0, 1, 2])
                regime_duration = 0
            regime_sequence.append(current_regime)
            regime_duration += 1
        
        regime_sequence = np.array(regime_sequence)
        
        # Base data
        data = {'date': dates}
        
        # Regime-dependent features
        regime_means = {
            0: {'return': 0.0001, 'vol': 0.15},  # Normal
            1: {'return': 0.0005, 'vol': 0.10},  # Bull
            2: {'return': -0.0003, 'vol': 0.25}  # Bear
        }
        
        # Market data
        returns = np.array([regime_means[r]['return'] + np.random.normal(0, regime_means[r]['vol']/np.sqrt(252)) 
                           for r in regime_sequence])
        prices = 100 * np.exp(np.cumsum(returns))
        
        data['spy_close'] = prices
        data['spy_returns'] = returns
        data['spy_volatility'] = pd.Series(returns).rolling(20).std() * np.sqrt(252)
        
        # VIX-like
        data['vix_close'] = 15 + 10 * (regime_sequence == 2) + np.random.normal(0, 3, n_days)
        
        # Macro features
        data['fred_unemployment'] = 5 + 2 * (regime_sequence == 2) - (regime_sequence == 1) + np.random.normal(0, 0.5, n_days)
        data['fred_gdp'] = 2 + 2 * (regime_sequence == 1) - 3 * (regime_sequence == 2) + np.random.normal(0, 0.5, n_days)
        data['fred_cpi'] = 2 + 0.5 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 0.3, n_days)
        
        # Technical indicators
        data['spy_rsi'] = 50 + 20 * (regime_sequence == 1) - 20 * (regime_sequence == 2) + np.random.normal(0, 10, n_days)
        
        # Ensure no negative values where inappropriate
        data['vix_close'] = np.maximum(data['vix_close'], 5)
        data['spy_rsi'] = np.clip(data['spy_rsi'], 0, 100)
        
        df = pd.DataFrame(data)
        
        # Fill NaNs from rolling calculations
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Final check - fill any remaining NaNs with column mean
        for col in df.columns:
            if col != 'date' and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data"""
        df = df.copy()
        
        # For each ticker with close price, add indicators
        close_cols = [col for col in df.columns if col.endswith('_close') and not col.startswith('user_')]
        
        for close_col in close_cols:
            ticker = close_col.replace('_close', '')
            
            if f'{ticker}_close' in df.columns:
                # RSI
                df[f'{ticker}_rsi'] = self._calculate_rsi(df[f'{ticker}_close'])
                
                # Moving averages
                df[f'{ticker}_ma20'] = df[f'{ticker}_close'].rolling(20).mean()
                df[f'{ticker}_ma50'] = df[f'{ticker}_close'].rolling(50).mean()
                
                # Bollinger bands
                ma20 = df[f'{ticker}_ma20']
                std20 = df[f'{ticker}_close'].rolling(20).std()
                df[f'{ticker}_bb_upper'] = ma20 + 2 * std20
                df[f'{ticker}_bb_lower'] = ma20 - 2 * std20
        
        # Term structure if we have yields
        if 'tlt_close' in df.columns and 'spy_close' in df.columns:
            # Simplified term spread proxy
            df['term_spread'] = df['tlt_returns'] - df['spy_returns'].rolling(20).mean()
        
        # Fill NaN values created by rolling calculations
        # Forward fill first, then backward fill for any remaining at the start
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaNs, fill with column mean
        for col in df.columns:
            if col != 'date' and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _validate_coverage(self, df: pd.DataFrame) -> None:
        """Validate coverage with different thresholds for different data types"""
        coverage_threshold = 0.97
        macro_threshold = 0.70  # Relaxed threshold for macro data

        for col in df.columns:
            if col == 'date':
                continue

            coverage = df[col].notna().sum() / len(df)

            # Use relaxed threshold for FRED macro data
            if col.startswith('fred_'):
                threshold = macro_threshold
                data_type = "macro"
            else:
                threshold = coverage_threshold
                data_type = "market"

            if coverage < threshold:
                print(f"Warning: {data_type} series '{col}' has {coverage:.1%} coverage, "
                      f"below {threshold:.0%} threshold")

                # Only raise error for market data with very low coverage
                if not col.startswith('fred_') and coverage < 0.50:
                    raise ValueError(
                        f"Market series '{col}' has {coverage:.1%} coverage, "
                        f"below minimum 50% threshold"
                    )
            else:
                print(f"✓ {col}: {coverage:.1%} coverage")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill ≤ 3 consecutive NaNs and drop rows with wider gaps"""
        df = df.copy()
        
        # First, forward fill up to 3 consecutive NaNs
        for col in df.columns:
            if col == 'date':
                continue
            df[col] = df[col].fillna(method='ffill', limit=3)
        
        # Check how many NaNs remain
        nan_counts = df.isna().sum()
        if nan_counts.any():
            print(f"NaN counts before dropping rows:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"  {col}: {count} NaNs")
        
        # Drop rows with remaining NaNs
        before_rows = len(df)
        df = df.dropna()
        after_rows = len(df)
        
        if after_rows < before_rows:
            print(f"Dropped {before_rows - after_rows} rows with missing values")
            
        # Ensure we still have enough data
        if len(df) < 100:
            raise ValueError(f"Only {len(df)} rows remaining after cleaning. Need at least 100 rows.")
        
        return df
    
    def _persist_raw_snapshot(self, df: pd.DataFrame, 
                            window_start: str, window_end: str) -> str:
        """Persist the data snapshot"""
        content_hash = hashlib.sha256(
            df.to_csv(index=False).encode()
        ).hexdigest()[:16]
        
        filename = f"raw_snapshot_{window_start}_{window_end}_{content_hash}.pkl"
        filepath = os.path.join(self.raw_snapshot_dir, filename)
        
        snapshot = {
            'data': df,
            'window_start': window_start,
            'window_end': window_end,
            'created_at': datetime.now().isoformat(),
            'content_hash': content_hash,
            'feature_list': [col for col in df.columns if col != 'date']
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(snapshot, f)
        
        return filepath
    
    def _preprocess_data(self, df: pd.DataFrame, window_end: str) -> np.ndarray:
        """Winsorise and z-score normalise data"""
        # Check cache first
        cache_key = self._generate_cache_key(df, window_end)
        cached_result = self._check_cache(cache_key)
        
        if cached_result is not None:
            self.transform_params[window_end] = cached_result['transform_params']
            return cached_result['tensor']
        
        # Separate date column
        if 'date' in df.columns:
            dates = df['date']
            numeric_df = df.drop('date', axis=1)
        else:
            numeric_df = df
            dates = None
        
        # Final check for NaNs before processing
        if numeric_df.isna().any().any():
            print("Warning: NaN values detected before preprocessing. Filling with column means.")
            for col in numeric_df.columns:
                if numeric_df[col].isna().any():
                    numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
        
        # Winsorise at 0.1% and 99.9% quantiles
        winsorised_df = numeric_df.copy()
        for col in numeric_df.columns:
            q_low = numeric_df[col].quantile(0.001)
            q_high = numeric_df[col].quantile(0.999)
            winsorised_df[col] = numeric_df[col].clip(lower=q_low, upper=q_high)
        
        # Z-score normalisation
        transform_params = {}
        normalised_data = []
        
        for col in winsorised_df.columns:
            mean = winsorised_df[col].mean()
            std = winsorised_df[col].std()
            
            transform_params[col] = {'mean': mean, 'std': std}
            
            if std > 0:
                normalised_col = (winsorised_df[col] - mean) / std
            else:
                normalised_col = winsorised_df[col] - mean
            
            normalised_data.append(normalised_col.values)
        
        normalised_tensor = np.column_stack(normalised_data)
        
        # Final check for NaN/Inf in tensor
        if np.isnan(normalised_tensor).any() or np.isinf(normalised_tensor).any():
            print("Warning: NaN or Inf detected in normalized tensor. Replacing with 0.")
            normalised_tensor = np.nan_to_num(normalised_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store transform parameters and feature names
        self.transform_params[window_end] = transform_params
        self.feature_names = list(winsorised_df.columns)
        
        # Cache the result
        self._write_processed_snapshot(
            normalised_tensor, window_end, cache_key, 
            transform_params, dates
        )
        
        return normalised_tensor
    
    def _generate_cache_key(self, df: pd.DataFrame, window_end: str) -> str:
        """Generate unique cache key"""
        if 'date' in df.columns:
            content = df.drop('date', axis=1).to_csv(index=False)
        else:
            content = df.to_csv(index=False)
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"{window_end}_{content_hash}"
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if processed snapshot exists"""
        filename = f"processed_snapshot_{cache_key}.pkl"
        filepath = os.path.join(self.processed_snapshot_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def _write_processed_snapshot(self, tensor: np.ndarray, window_end: str,
                                cache_key: str, transform_params: Dict,
                                dates: pd.Series = None) -> None:
        """Write processed snapshot"""
        checksum = hashlib.sha256(tensor.tobytes()).hexdigest()
        
        snapshot = {
            'tensor': tensor,
            'window_end': window_end,
            'cache_key': cache_key,
            'checksum': checksum,
            'transform_params': transform_params,
            'dates': dates,
            'feature_names': list(transform_params.keys()),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        filename = f"processed_snapshot_{cache_key}.pkl"
        filepath = os.path.join(self.processed_snapshot_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(snapshot, f)


@dataclass
class GMMCandidate:
    """Container for GMM candidate results with complete implementation"""
    k: int
    log_likelihood: float
    bic: float
    icl: float
    aic: float
    labels: np.ndarray
    posterior_probs: np.ndarray
    model: GaussianMixture
    converged: bool
    n_iter: int


class GMMFittingSelection:
    """Complete GMM implementation with proper BIC/ICL selection"""
    
    def __init__(self, k_min: int = 1, k_max: int = 10, random_seed: int = 42):
        self.k_min = k_min
        self.k_max = k_max
        self.random_seed = random_seed
        self.bootstrap_samples = 500
        self.ari_threshold = 0.85
        
        # GMM parameters
        self.n_init = 100  # Number of initializations as specified
        self.max_iter = 1000
        self.covariance_type = 'full'
        self.reg_covar = 1e-6
        
        # Create directories
        self.candidates_dir = "gmm_candidates"
        self.results_dir = "selection_results"
        self.labels_dir = "regime_labels"
        os.makedirs(self.candidates_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
    
    def fit_and_select(
        self,
        tensor: np.ndarray,
        window_end: str,
        force_k: Optional[int] = None
    ) -> Tuple[int, Dict, Dict[int, GMMCandidate], np.ndarray]:
        """
        Fit GMM(s) and return (k_star, metadata, all_candidates, posteriors)
        """

        
        # 1.  Forced-k path  → fit ONE model, skip auto-selection
        if force_k is not None:
            print(f"Forced k={force_k}: fitting only that model.")
            candidate = self._fit_single_gmm_complete(tensor, force_k)
            candidates = {force_k: candidate}

            k_star, selection_metadata = self._force_selection(
                candidates, tensor, window_end, force_k
            )

            # nothing else to pick, so continue with post-processing
        else:
            
            # 2.  Normal path  → fit full grid, then choose k*
            print(f"Fitting GMMs for k={self.k_min} to k={self.k_max}...")
            candidates = self._fit_all_gmm_candidates(tensor, window_end)

            print("Selecting optimal k using ICL criterion...")
            k_star, selection_metadata = self._select_optimal_k_icl(
                candidates, tensor, window_end
            )

        
        posterior_probs = candidates[k_star].posterior_probs
        self._persist_regime_outputs(candidates[k_star], window_end, selection_metadata)

        return k_star, selection_metadata, candidates, posterior_probs

    
    def _fit_all_gmm_candidates(self, tensor: np.ndarray, window_end: str) -> Dict[int, GMMCandidate]:
        """Fit GMMs for all k values"""
        from optimization_utils import parallel_gmm_grid_search_safe

        print(f"Fitting of GMMs for k={self.k_min} to k={self.k_max}...")

        candidates = parallel_gmm_grid_search_safe(
            tensor=tensor,
            k_min=self.k_min,
            k_max=self.k_max,
            n_init=self.n_init,
            max_iter=self.max_iter,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_seed=self.random_seed,
            window_end=window_end
        )

        # Log results
        for k, candidate in candidates.items():
            print(f"  k={k}: Log-likelihood={candidate.log_likelihood:.2f}, "
                  f"BIC={candidate.bic:.2f}, ICL={candidate.icl:.2f}, "
                  f"Converged={candidate.converged}")

            self._serialise_candidate(candidate, window_end)

        return candidates
    
    def _fit_single_gmm_complete(self, tensor: np.ndarray, k: int) -> GMMCandidate:
        """Fit a single GMM with complete implementation"""
        # Initialize and fit GMM
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=self.covariance_type,
            n_init=self.n_init,  # 100 restarts as specified
            random_state=self.random_seed,
            max_iter=self.max_iter,
            init_params='kmeans',
            reg_covar=self.reg_covar,
            warm_start=False
        )
        
        # Fit the model
        gmm.fit(tensor)
        
        # Get predictions and probabilities
        labels = gmm.predict(tensor)
        posterior_probs = gmm.predict_proba(tensor)
        
        # Calculate metrics
        n_samples = len(tensor)
        n_features = tensor.shape[1]
        
        # Log-likelihood
        log_likelihood = gmm.score_samples(tensor).sum()
        
        # BIC = -2 * log_likelihood + k * log(n)
        n_parameters = self._count_gmm_parameters(k, n_features)
        bic = -2 * log_likelihood + n_parameters * np.log(n_samples)
        
        # AIC = -2 * log_likelihood + 2 * k
        aic = -2 * log_likelihood + 2 * n_parameters
        
        # ICL = BIC - 2 * entropy(classification)
        # Entropy measures uncertainty in cluster assignments
        entropy = 0.0
        for i in range(n_samples):
            for j in range(k):
                if posterior_probs[i, j] > 0:
                    entropy -= posterior_probs[i, j] * np.log(posterior_probs[i, j])
        
        icl = bic - 2 * entropy
        
        return GMMCandidate(
            k=k,
            log_likelihood=log_likelihood,
            bic=bic,
            icl=icl,
            aic=aic,
            labels=labels,
            posterior_probs=posterior_probs,
            model=gmm,
            converged=gmm.converged_,
            n_iter=gmm.n_iter_
        )
    
    def _count_gmm_parameters(self, k: int, n_features: int) -> int:
        """Count number of parameters in GMM with full covariance"""
        # Means: k * n_features
        # Covariances (full): k * n_features * (n_features + 1) / 2
        # Weights: k - 1 (sum to 1 constraint)
        n_means = k * n_features
        n_cov = k * n_features * (n_features + 1) // 2
        n_weights = k - 1
        
        return n_means + n_cov + n_weights
    
    def _select_optimal_k_icl(self, candidates: Dict, tensor: np.ndarray, 
                            window_end: str) -> Tuple[int, Dict]:
        """Select optimal k using ICL criterion with elbow detection"""
        # Extract ICL values
        k_values = sorted(candidates.keys())
        icl_values = [candidates[k].icl for k in k_values]
        bic_values = [candidates[k].bic for k in k_values]
        
        # Primary criterion: minimum ICL
        k_star_icl = k_values[np.argmin(icl_values)]
        
        # Secondary check: elbow in log-likelihood
        k_elbow = self._detect_elbow(candidates)
        
        # Choose k_star from neighborhood of elbow, minimizing ICL
        k_neighborhood = [k for k in [k_elbow - 1, k_elbow, k_elbow + 1] 
                         if k in candidates and k >= self.k_min]
        
        k_star = min(k_neighborhood, key=lambda k: candidates[k].icl)
        
        print(f"Selected k_star={k_star} (ICL={candidates[k_star].icl:.2f})")
        
        # Bootstrap stability validation
        print(f"Running bootstrap stability check with {self.bootstrap_samples} samples...")
        bootstrap_results = self._bootstrap_stability_check(
            tensor, k_star, candidates[k_star].labels
        )
        
        mean_ari = bootstrap_results['mean_ari']
        std_ari = bootstrap_results['std_ari']
        
        print(f"Bootstrap ARI: {mean_ari:.3f} ± {std_ari:.3f}")
        
        if mean_ari < self.ari_threshold:
            raise RuntimeError(
                f"NO_STABLE_REGIME: Bootstrap mean ARI {mean_ari:.3f} < "
                f"required threshold {self.ari_threshold}"
            )
        
        # Compile selection metadata
        selection_metadata = {
            'k_star': k_star,
            'k_elbow': k_elbow,
            'k_icl_min': k_star_icl,
            'bic': candidates[k_star].bic,
            'icl': candidates[k_star].icl,
            'aic': candidates[k_star].aic,
            'log_likelihood': candidates[k_star].log_likelihood,
            'bootstrap_mean_ari': mean_ari,
            'bootstrap_std_ari': std_ari,
            'bootstrap_samples': self.bootstrap_samples,
            'ari_threshold': self.ari_threshold,
            'window_end': window_end,
            'selection_criteria': 'ICL with elbow detection',
            'converged': candidates[k_star].converged,
            'n_iter': candidates[k_star].n_iter
        }
        
        self._save_selection_results(selection_metadata, window_end)
        
        return k_star, selection_metadata
    
    def _detect_elbow(self, candidates: Dict) -> int:
        """Detect elbow in log-likelihood curve"""
        k_values = sorted(candidates.keys())
        
        if len(k_values) < 2:
            return k_values[0]
        
        # Calculate log-likelihood differences
        delta_logl = []
        for i in range(1, len(k_values)):
            k_curr = k_values[i]
            k_prev = k_values[i-1]
            delta = candidates[k_curr].log_likelihood - candidates[k_prev].log_likelihood
            delta_logl.append((k_curr, delta))
        
        if not delta_logl:
            return self.k_min
        
        # Find maximum jump
        max_jump = max(delta for _, delta in delta_logl)
        threshold = 0.1 * max_jump  # 10% of max as specified
        
        # Find first k where jump is less than threshold
        k_elbow = self.k_min
        for k, delta in delta_logl:
            if delta < threshold:
                k_elbow = k
                break
        
        return k_elbow
    
    
    def _bootstrap_stability_check(self, tensor: np.ndarray, k_star: int,
                                 reference_labels: np.ndarray) -> Dict:
        """Bootstrap validation with parallel processing"""
        results = parallel_bootstrap_stability(
            tensor=tensor,
            k_star=k_star,
            reference_labels=reference_labels,
            n_samples=self.bootstrap_samples,
            random_seed=self.random_seed,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar
        )
        
        return results
    
    def _force_selection(self, candidates: Dict, tensor: np.ndarray,
                       window_end: str, force_k: int) -> Tuple[int, Dict]:
        """Force selection of specific k with validation"""
        if force_k not in candidates:
            raise ValueError(f"k={force_k} not in candidates ({self.k_min}-{self.k_max})")
        
        print(f"Running bootstrap validation for forced k={force_k}...")
        
        bootstrap_results = self._bootstrap_stability_check(
            tensor, force_k, candidates[force_k].labels
        )
        
        metadata = {
            'k_star': force_k,
            'k_elbow': force_k,
            'k_icl_min': force_k,
            'bic': candidates[force_k].bic,
            'icl': candidates[force_k].icl,
            'aic': candidates[force_k].aic,
            'log_likelihood': candidates[force_k].log_likelihood,
            'bootstrap_mean_ari': bootstrap_results['mean_ari'],
            'bootstrap_std_ari': bootstrap_results['std_ari'],
            'bootstrap_samples': self.bootstrap_samples,
            'ari_threshold': self.ari_threshold,
            'window_end': window_end,
            'forced_selection': True,
            'selection_criteria': 'Forced k selection',
            'converged': candidates[force_k].converged,
            'n_iter': candidates[force_k].n_iter
        }
        
        self._save_selection_results(metadata, window_end)
        
        return force_k, metadata
    
    def _persist_regime_outputs(self, candidate: GMMCandidate, 
                              window_end: str, metadata: Dict) -> None:
        """Persist regime labels and posterior probabilities"""

        # Save hard labels
        labels_file = os.path.join(
            self.labels_dir,
            f"regime_labels_{window_end}.npy"
        )
        np.save(labels_file, candidate.labels)
        
        # Save posterior probabilities
        probs_file = os.path.join(
            self.labels_dir,
            f"regime_posteriors_{window_end}.npy"
        )
        np.save(probs_file, candidate.posterior_probs)
        
        # Save complete GMM model
        model_file = os.path.join(
            self.candidates_dir,
            f"gmm_model_k{candidate.k}_{window_end}.pkl"
        )
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': candidate.model,
                'metadata': metadata,
                'feature_params': {
                    'n_features': candidate.model.means_.shape[1],
                    'n_components': candidate.k
                }
            }, f)
        
        # Save regime statistics
        self._save_regime_statistics(candidate, window_end)
    
    def _save_regime_statistics(self, candidate: GMMCandidate, window_end: str) -> None:
        """Save detailed regime statistics"""
        stats = {
            'k': candidate.k,
            'window_end': window_end,
            'regime_counts': np.bincount(candidate.labels).tolist(),
            'regime_proportions': (np.bincount(candidate.labels) / len(candidate.labels)).tolist(),
            'regime_means': candidate.model.means_.tolist(),
            'regime_covariances_diagonal': [np.diag(cov).tolist() 
                                           for cov in candidate.model.covariances_],
            'transition_matrix': self._calculate_transition_matrix(candidate.labels),
            'average_posterior_confidence': np.mean(np.max(candidate.posterior_probs, axis=1))
        }
        
        stats_file = os.path.join(
            self.results_dir,
            f"regime_statistics_{window_end}.json"
        )
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _calculate_transition_matrix(self, labels: np.ndarray) -> List[List[float]]:
        """Calculate empirical regime transition matrix"""
        n_regimes = len(np.unique(labels))
        transitions = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(labels) - 1):
            transitions[labels[i], labels[i+1]] += 1
        
        # Normalize rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transitions, row_sums, 
                                    where=row_sums > 0, 
                                    out=np.zeros_like(transitions))
        
        return transition_matrix.tolist()
    
    def _serialise_candidate(self, candidate: GMMCandidate, window_end: str) -> None:
        """Serialise complete candidate for diagnostics"""
        filename = f"gmm_candidate_k{candidate.k}_{window_end}.pkl"
        filepath = os.path.join(self.candidates_dir, filename)
        
        save_candidate = GMMCandidate(
            k=candidate.k,
            log_likelihood=candidate.log_likelihood,
            bic=candidate.bic,
            icl=candidate.icl,
            aic=candidate.aic,
            labels=candidate.labels,
            posterior_probs=candidate.posterior_probs,
            model=None,
            converged=candidate.converged,
            n_iter=candidate.n_iter
        )
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_candidate, f)
    
    def _save_selection_results(self, metadata: Dict, window_end: str) -> None:
        """Save selection results as JSON"""
        filename = f"selection_results_{window_end}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_gmm_model(self, k: int, window_end: str) -> GaussianMixture:
        """Load a previously fitted GMM model"""
        model_file = os.path.join(
            self.candidates_dir,
            f"gmm_model_k{k}_{window_end}.pkl"
        )
        
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
            return data['model']
    
    def load_regime_outputs(self, window_end: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load regime labels and posterior probabilities"""
        labels_file = os.path.join(
            self.labels_dir,
            f"regime_labels_{window_end}.npy"
        )
        labels = np.load(labels_file)
        
        probs_file = os.path.join(
            self.labels_dir,
            f"regime_posteriors_{window_end}.npy"
        )
        posteriors = np.load(probs_file)
        
        return labels, posteriors