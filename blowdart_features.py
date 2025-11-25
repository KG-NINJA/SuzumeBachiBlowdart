"""
blowdart_features.py - Feature engineering for SuzumeBachiBlowdart
Calculates: RSI, MACD, Bollinger Bands, Moving Averages, Advanced Technical Indicators
Integrated with advanced_features.py for enhanced prediction
"""

import pandas as pd
import numpy as np
from advanced_features import build_advanced_features


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def build_feature_set(price_data, ticker):
    """
    Build comprehensive feature set from price data
    
    Args:
        price_data: DataFrame with OHLCV data
        ticker: Stock symbol (for logging)
    
    Returns:
        DataFrame: Features ready for ML, or None if failed
    """
    try:
        if price_data is None or price_data.empty:
            return None
        
        df = price_data.copy()
        
        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure required columns
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            print(f"  [FEATURES] Missing columns: {set(required) - set(df.columns)}")
            return None
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 30:
            print(f"  [FEATURES] Insufficient data: {len(df)} rows")
            return None
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # ===== Technical Indicators (Basic) =====
        
        # Moving Averages
        df['ma5'] = close.rolling(5).mean()
        df['ma10'] = close.rolling(10).mean()
        df['ma20'] = close.rolling(20).mean()
        df['ma50'] = close.rolling(50).mean()
        
        # Exponential Moving Average
        df['ema12'] = close.ewm(span=12).mean()
        df['ema26'] = close.ewm(span=26).mean()
        
        # RSI
        df['rsi14'] = calculate_rsi(close, 14)
        df['rsi7'] = calculate_rsi(close, 7)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(close)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(close)
        
        # Rate of Change
        df['roc10'] = ((close - close.shift(10)) / close.shift(10)) * 100
        df['roc20'] = ((close - close.shift(20)) / close.shift(20)) * 100
        
        # Momentum
        df['momentum'] = close - close.shift(10)
        
        # ATR (Average True Range)
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Volume indicators
        df['volume_ma20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_ma20'] + 0.0001)
        
        # Price patterns
        df['high_low_ratio'] = (high - low) / close
        df['close_open_ratio'] = (close - df['open']) / df['open']
        
        # Returns
        df['daily_return'] = close.pct_change() * 100
        
        # Volatility
        df['volatility'] = df['daily_return'].rolling(20).std()
        
        # ===== Advanced Features (NEW) =====
        print(f"  [FEATURES] Adding advanced indicators for {ticker}...")
        
        df = build_advanced_features(df)
        
        print(f"  [FEATURES] Total features: {df.shape[1]}")
        
        # ===== Data Cleaning =====
        
        # Handle NaN and Inf
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Remove rows with NaN (from technical indicators)
        df = df.dropna()
        
        if len(df) < 30:
            print(f"  [FEATURES] Insufficient data after feature engineering: {len(df)} rows")
            return None
        
        return df
    
    except Exception as e:
        print(f"  [FEATURES ERROR] {ticker}: {str(e)[:60]}")
        return None
