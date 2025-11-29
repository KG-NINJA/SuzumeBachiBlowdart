"""
blowdart_features.py - Feature engineering for SuzumeBachiBlowdart
Calculates: RSI, MACD, Bollinger Bands, Moving Averages, Advanced Technical Indicators
Integrated with advanced_features.py for enhanced prediction
"""

import numpy as np
import pandas as pd

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


def build_feature_set(price_data, ticker, use_feature_reduction=True):
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

        # Step 1: Normalize column names - convert ALL to proper Title Case (Close, Open, High, Low, Volume)
        print(f"  [FEATURES] Original columns: {df.columns.tolist()}")

        column_mapping = {}
        lower_cols = {col.lower(): col for col in df.columns}

        standard_names = ['date', 'open', 'high', 'low', 'close', 'volume']
        for std_name in standard_names:
            if std_name in lower_cols:
                actual_col = lower_cols[std_name]
                column_mapping[actual_col] = std_name.capitalize()

        df = df.rename(columns=column_mapping)
        print(f"  [FEATURES] Renamed columns: {df.columns.tolist()}")

        # Step 2: Validate required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"  [FEATURES] Missing required columns: {missing}")
            print(f"  [FEATURES] Available columns: {df.columns.tolist()}")
            return None

        # Convert to numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        if len(df) < 30:
            print(f"  [FEATURES] Insufficient data: {len(df)} rows")
            return None

        # Sort by date if available
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # ===== Technical Indicators (Basic) =====

        # Moving Averages
        df['MA5'] = close.rolling(5).mean()
        df['MA10'] = close.rolling(10).mean()
        df['MA20'] = close.rolling(20).mean()
        df['MA50'] = close.rolling(50).mean()

        # Exponential Moving Average
        df['EMA12'] = close.ewm(span=12).mean()
        df['EMA26'] = close.ewm(span=26).mean()

        # RSI
        df['RSI14'] = calculate_rsi(close, 14)
        df['RSI7'] = calculate_rsi(close, 7)

        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(close)

        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(close)

        # Rate of Change
        df['ROC10'] = ((close - close.shift(10)) / close.shift(10)) * 100
        df['ROC20'] = ((close - close.shift(20)) / close.shift(20)) * 100

        # Momentum
        df['Momentum'] = close - close.shift(10)

        # ATR (Average True Range)
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # Volume indicators
        df['Volume_MA20'] = volume.rolling(20).mean()
        df['Volume_Ratio'] = volume / (df['Volume_MA20'] + 0.0001)

        # Price patterns
        df['HighLowRatio'] = (high - low) / close
        df['CloseOpenRatio'] = (close - df['Open']) / df['Open']

        # Returns
        df['DailyReturn'] = close.pct_change() * 100

        # Volatility
        df['Volatility'] = df['DailyReturn'].rolling(20).std()

        print(f"  [FEATURES] Basic features added: {df.shape[1]} columns")

        # ===== Advanced Features (NEW) =====
        print(f"  [FEATURES] Adding advanced indicators for {ticker}...")

        try:
            df = build_advanced_features(df)
            print("  [FEATURES] Advanced features added successfully")
        except KeyError as e:
            print(f"  [FEATURES] KeyError: {e} | Available: {df.columns.tolist()}")
            print("  [FEATURES] Continuing with basic features only")
        except Exception as e:
            print(f"  [FEATURES] Error: {str(e)[:100]}")
            print("  [FEATURES] Continuing with basic features only")

        print(f"  [FEATURES] Total features: {df.shape[1]}")

        # ===== Data Cleaning =====

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        df = df.dropna()

        if len(df) < 30:
            print(f"  [FEATURES] Insufficient data after feature engineering: {len(df)} rows")
            return None

        print(f"  [FEATURES] Final dataset: {len(df)} rows × {df.shape[1]} columns")
        return df

    except KeyError as e:
        print(f"  [FEATURES ERROR] {ticker}: KeyError - {str(e)}")
        print(f"  [FEATURES ERROR] Available columns: {df.columns.tolist() if 'df' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"  [FEATURES ERROR] {ticker}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None
