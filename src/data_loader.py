import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class DataLoader:
    def __init__(self, symbol='GC=F', interval='1h', period='1y', data_path='data/gold_data.csv'):
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.data_path = data_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self, force_download=False, source='mt5'):
        """
        Fetch data.
        source: 'mt5' or 'yfinance'
        """
        if not force_download and os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            return df

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
        
        df = yf.download(yf_symbol, interval=self.interval, period=self.period, progress=False)
        if df.empty:
            raise ValueError("No data fetched! Check symbol or internet connection.")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path)
        print(f"Data saved to {self.data_path}")
        return df

    def prepare_data_for_lstm(self, df, look_back=60, target_col='Close'):
        """
        Prepare data for LSTM: (Samples, TimeSteps, Features)
        We will use just the target_col for univariate prediction as per the paper basics,
        but can extend to multivariate.
        Paper uses: Xt-1, ... Xt-p (Univariate lags mostly)
        """
        data = df[[target_col]].values
        data_scaled = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(data_scaled) - look_back):
            X.append(data_scaled[i:i + look_back])
            y.append(data_scaled[i + look_back])
            
        X, y = np.array(X), np.array(y)
        # Reshape X to (samples, time steps, features) which is already correct if X is created as listed
        # X shape: [samples, look_back, 1]
        
        return X, y, data_scaled

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__ == "__main__":
    # Test
    loader = DataLoader(period='1mo') # small period for test
    df = loader.fetch_data(force_download=True)
    print(df.head())
    X, y, scaled = loader.prepare_data_for_lstm(df)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
