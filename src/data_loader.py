import platform
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def is_windows():
    """Check if running on Windows (not WSL)"""
    return platform.system() == 'Windows'

class DataLoader:
    def __init__(self, symbol='GC=F', interval='1h', period='1y', data_path='data/gold_data.csv'):
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.data_path = data_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self, force_download=False, source='auto'):
        """
        Fetch data.
        source: 'auto' (platform detection), 'mt5', or 'yfinance'
        """
        if not force_download and os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            return df

        # Auto-detect platform
        if source == 'auto':
            if is_windows():
                print("[Platform] Windows detected -> using MetaTrader5")
                source = 'mt5'
            else:
                print("[Platform] Linux/WSL detected -> using Yahoo Finance")
                source = 'yfinance'

        if source == 'mt5':
            return self._fetch_from_mt5()
        else:
            return self._fetch_from_yfinance()

    def _fetch_from_mt5(self):
        try:
            import MetaTrader5 as mt5
        except ImportError:
            print("[ERROR] MetaTrader5 package not installed.")
            return pd.DataFrame()

        print("Initializing MetaTrader5...")
        if not mt5.initialize():
            print(f"[ERROR] MT5 Initialize failed, error code = {mt5.last_error()}")
            print("Falling back to yfinance or exiting.")
            return self._fetch_from_yfinance()

        print(f"Downloading {self.symbol} data from MT5 ({self.interval})...")
        
        # Map interval string to MT5 constant
        timeframe_map = {
            '1h': mt5.TIMEFRAME_H1,
            '30m': mt5.TIMEFRAME_M30,
            '15m': mt5.TIMEFRAME_M15,
            'd1': mt5.TIMEFRAME_D1
        }
        tf = timeframe_map.get(self.interval, mt5.TIMEFRAME_H1)
        
        # Fetch 5000 candles (approx 1-2 years for H1)
        count = 5000 
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            print("[ERROR] No data received from MT5.")
            return self._fetch_from_yfinance()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # Rename columns to match conventions
        df.rename(columns={'tick_volume': 'Volume', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path)
        print(f"Data saved to {self.data_path}")
        return df

    def _fetch_from_yfinance(self):
        print(f"Downloading data for {self.symbol} ({self.interval}) from Yahoo Finance...")
        # Map symbol if needed (MT5 "XAUUSD" -> YF "GC=F")
        yf_symbol = "GC=F" if "XAU" in self.symbol else self.symbol
        
        period = self.period
        interval = self.interval
        
        # Simple parsing to check limits
        # Yahoo Finance intraday data limits:
        # - 1m: 7 days max
        # - 5m, 15m, 30m: 60 days max  
        # - 1h: 730 days max (approx 2y)
        if interval in ['1h', '60m']:
            # If period implies > 730 days (e.g. '5y', '2y', 'max'), cap it to '2y'
            # yfinance 1h limit is roughly 730 days
            if any(x in period for x in ['5y', '10y', 'max']):
                print(f"[INFO] YFinance limit: {interval} data max 730 days. Using '2y' instead of {period}")
                period = '2y'
        elif interval in ['5m', '15m', '30m']:
            # Max 60 days for these intervals
            if any(x in period for x in ['1y', '2y', '5y', '10y', 'max', '6mo', '3mo']):
                print(f"[INFO] YFinance limit: {interval} data max 60 days. Using '60d' instead of {period}")
                period = '60d'
        elif interval == '1m':
            # Max 7 days for 1m
            if any(x in period for x in ['1y', '2y', '5y', '10y', 'max', '6mo', '3mo', '1mo', '60d', '30d']):
                print(f"[INFO] YFinance limit: {interval} data max 7 days. Using '7d' instead of {period}")
                period = '7d'
        
        try:
            # Use Ticker.history which avoids the complex batch downloader (and ImpersonateError)
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(interval=interval, period=period)
        except Exception as e:
             print(f"[ERROR] Ticker.history failed: {e}")
             df = pd.DataFrame()

        if df.empty:
            raise ValueError("No data fetched! Check symbol or internet connection.")
        
        # Handle MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Try to get level 0 if it matches standard columns, or just Flatten
                df.columns = df.columns.get_level_values(0)
            except:
                pass
        
        # Ensure correct column names (Case sensitive)
        # yfinance usually returns: Open, High, Low, Close, Volume
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check standard casing
        if not all(col in df.columns for col in required):
            # Try lowercase mapping
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            
        # Select and Drop NaNs
        try:
            df = df[required].dropna()
        except KeyError as e:
            # If still missing, verify columns
            print(f"Columns found: {df.columns}")
            raise e

        # Handle Empty again after dropna
        if df.empty:
             raise ValueError("Data fetched but empty after cleaning.")

        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path)
        print(f"Data saved to {self.data_path} ({len(df)} rows)")
        return df

    def prepare_data_for_lstm(self, df, look_back=60, target_col='Close'):
        """
        Prepare data for LSTM: (Samples, TimeSteps, Features)
        """
        data = df[[target_col]].values
        data_scaled = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(data_scaled) - look_back):
            X.append(data_scaled[i:i + look_back])
            # y is the NEXT step
            y.append(data_scaled[i + look_back])
            
        X, y = np.array(X), np.array(y)
        return X, y, data_scaled

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
