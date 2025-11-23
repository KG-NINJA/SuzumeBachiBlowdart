"""
blowdart_features.py - Feature engineering for SuzumeBachiBlowdart
Calculates: RSI, MACD, Bollinger Bands, Moving Averages, etc.
"""

import pandas as pd
import numpy as np


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
        
        # ===== Technical Indicators =====
        
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
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(close, 20)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Returns & Volatility
        df['return_1d'] = close.pct_change()
        df['return_5d'] = close.pct_change(5)
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # Volume indicators
        df['volume_ma20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_ma20']
        
        # Price range
        df['high_low_range'] = (high - low) / close
        df['open_close_range'] = (close - df['open']) / df['open']
        
        # Trend indicators
        df['ma5_ma20_ratio'] = df['ma5'] / df['ma20']
        df['close_to_ma20'] = close / df['ma20']
        df['price_to_bb_upper'] = close / df['bb_upper']
        
        # Drop rows with NaN (indicators need warmup period)
        df = df.dropna()
        
        if len(df) < 20:
            print(f"  [FEATURES] Insufficient valid features: {len(df)} rows after NaN drop")
            return None
        
        # Rename columns to be consistent with ML engine
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return df
    
    except Exception as e:
        print(f"  [FEATURES ERROR] {ticker}: {str(e)[:60]}")
        return None
