# data_connectors.py
"""
Data connectors for macro/OHLCV data sources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta
import requests


class MacroFeatureStore(ABC):
    """Abstract base class for macro feature stores"""
    
    @abstractmethod
    def execute(self, query: str) -> pd.DataFrame:
        """Execute query and return dataframe with macro features"""
        pass


class YahooFinanceConnector(MacroFeatureStore):
    """Connect to Yahoo Finance for market data"""
    
    def __init__(self, tickers: List[str], macro_indicators: Dict[str, str]):
        """
        Args:
            tickers: List of tickers for OHLCV data (e.g., ['SPY', 'GLD', 'TLT'])
            macro_indicators: Dict mapping feature names to Yahoo tickers
                e.g., {'vix': '^VIX', 'dollar_index': 'DX-Y.NYB', '10y_yield': '^TNX'}
        """
        self.tickers = tickers
        self.macro_indicators = macro_indicators
    
    def execute(self, query: str) -> pd.DataFrame:
        """
        Parse query for date range and fetch all configured data
        Query format: "SELECT * FROM macro_features WHERE date >= 'start' AND date <= 'end'"
        """
        # Parse dates from query
        import re
        date_pattern = r"date >= '(\d{4}-\d{2}-\d{2})' AND date <= '(\d{4}-\d{2}-\d{2})'"
        match = re.search(date_pattern, query)
        
        if not match:
            raise ValueError(f"Could not parse dates from query: {query}")
        
        start_date = match.group(1)
        end_date = match.group(2)
        
        # Fetch all data
        all_features = {}
        
        # 1. Fetch OHLCV data for tickers
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                all_features[f'{ticker}_open'] = data['Open']
                all_features[f'{ticker}_high'] = data['High']
                all_features[f'{ticker}_low'] = data['Low']
                all_features[f'{ticker}_close'] = data['Close']
                all_features[f'{ticker}_volume'] = data['Volume']
                
                # Add technical indicators
                all_features[f'{ticker}_returns'] = data['Close'].pct_change()
                all_features[f'{ticker}_volatility'] = data['Close'].pct_change().rolling(20).std()
                all_features[f'{ticker}_rsi'] = self._calculate_rsi(data['Close'])
                
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        # 2. Fetch macro indicators
        for feature_name, ticker in self.macro_indicators.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                all_features[feature_name] = data['Close']
            except Exception as e:
                print(f"Error fetching {feature_name} ({ticker}): {e}")
        
        # 3. Create combined dataframe
        df = pd.DataFrame(all_features)
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date'})
        
        # 4. Add additional calculated features
        df = self._add_calculated_features(df)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_calculated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated macro features"""
        # Term spread (10Y - 2Y)
        if '10y_yield' in df.columns and '2y_yield' in df.columns:
            df['term_spread'] = df['10y_yield'] - df['2y_yield']
        
        # Market regime indicators
        if 'SPY_close' in df.columns:
            df['spy_ma50'] = df['SPY_close'].rolling(50).mean()
            df['spy_ma200'] = df['SPY_close'].rolling(200).mean()
            df['spy_trend'] = (df['SPY_close'] > df['spy_ma200']).astype(int)
        
        return df


class FREDConnector(MacroFeatureStore):
    """Connect to Federal Reserve Economic Data (FRED)"""
    
    def __init__(self, api_key: str, series_ids: Dict[str, str]):
        """
        Args:
            api_key: FRED API key
            series_ids: Dict mapping feature names to FRED series IDs
                e.g., {'gdp_growth': 'GDP', 'unemployment': 'UNRATE', 'cpi': 'CPIAUCSL'}
        """
        self.api_key = api_key
        self.series_ids = series_ids
    
    def execute(self, query: str) -> pd.DataFrame:
        """Fetch data from FRED API"""
        # Parse dates
        import re
        date_pattern = r"date >= '(\d{4}-\d{2}-\d{2})' AND date <= '(\d{4}-\d{2}-\d{2})'"
        match = re.search(date_pattern, query)
        
        if not match:
            raise ValueError(f"Could not parse dates from query: {query}")
        
        start_date = match.group(1)
        end_date = match.group(2)
        
        all_features = {}
        
        for feature_name, series_id in self.series_ids.items():
            try:
                # Use pandas_datareader for FRED
                data = pdr.get_data_fred(series_id, start=start_date, end=end_date)
                all_features[feature_name] = data
            except Exception as e:
                print(f"Error fetching {feature_name} ({series_id}): {e}")
        
        # Combine all series
        df = pd.DataFrame(all_features)
        
        # FRED data often has different frequencies - resample to daily
        df = df.resample('D').interpolate(method='linear', limit=3)
        
        df = df.reset_index()
        df = df.rename(columns={'DATE': 'date'})
        
        return df


class CSVFileConnector(MacroFeatureStore):
    """Load macro features from CSV files"""
    
    def __init__(self, file_paths: Dict[str, str], date_column: str = 'date'):
        """
        Args:
            file_paths: Dict mapping feature names to CSV file paths
            date_column: Name of date column in CSV files
        """
        self.file_paths = file_paths
        self.date_column = date_column
    
    def execute(self, query: str) -> pd.DataFrame:
        """Load and filter CSV data"""
        # Parse dates
        import re
        date_pattern = r"date >= '(\d{4}-\d{2}-\d{2})' AND date <= '(\d{4}-\d{2}-\d{2})'"
        match = re.search(date_pattern, query)
        
        if not match:
            raise ValueError(f"Could not parse dates from query: {query}")
        
        start_date = pd.to_datetime(match.group(1))
        end_date = pd.to_datetime(match.group(2))
        
        all_data = []
        
        for feature_name, file_path in self.file_paths.items():
            try:
                # Read CSV
                df = pd.read_csv(file_path)
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                
                # Filter date range
                mask = (df[self.date_column] >= start_date) & (df[self.date_column] <= end_date)
                df = df[mask]
                
                # Set date as index for merging
                df = df.set_index(self.date_column)
                
                # Rename value columns to feature name if needed
                if len(df.columns) == 1:
                    df = df.rename(columns={df.columns[0]: feature_name})
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading {feature_name} from {file_path}: {e}")
        
        # Merge all dataframes
        if all_data:
            result = pd.concat(all_data, axis=1, join='outer')
            result = result.reset_index()
            result = result.rename(columns={self.date_column: 'date'})
            return result
        else:
            raise ValueError("No data loaded from CSV files")


class CombinedFeatureStore(MacroFeatureStore):
    """Combine multiple data sources into one feature store"""
    
    def __init__(self, connectors: List[MacroFeatureStore]):
        """
        Args:
            connectors: List of data connectors to combine
        """
        self.connectors = connectors
    
    def execute(self, query: str) -> pd.DataFrame:
        """Fetch from all connectors and merge"""
        all_data = []
        
        for connector in self.connectors:
            try:
                data = connector.execute(query)
                if 'date' in data.columns:
                    data = data.set_index('date')
                all_data.append(data)
            except Exception as e:
                print(f"Error in connector {type(connector).__name__}: {e}")
        
        if all_data:
            # Merge all dataframes on date
            result = pd.concat(all_data, axis=1, join='outer')
            
            # Reset index
            result = result.reset_index()
            if 'index' in result.columns:
                result = result.rename(columns={'index': 'date'})
            
            # Sort by date
            result = result.sort_values('date')
            
            return result
        else:
            raise ValueError("No data fetched from any connector")