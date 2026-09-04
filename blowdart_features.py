"""
blowdart_features.py - Feature engineering for SuzumeBachiBlowdart
Calculates: RSI, MACD, Bollinger Bands, Moving Averages, Advanced Technical Indicators
Integrated with advanced_features.py for enhanced prediction
"""

import numpy as np
import pandas as pd

from advanced_features import build_advanced_features
from pipeline_safety import normalize_ohlcv_columns


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


def build_feature_set(
    price_data,
    ticker,
    use_feature_reduction=True,
    raise_on_error=False,
):
    """
    Build comprehensive feature set from price data.

    Args:
        price_data: DataFrame with OHLCV data.
        ticker: Stock symbol (for logging).
        use_feature_reduction: Retained for backward compatibility.
        raise_on_error: Re-raise feature failures for pipeline diagnostics.

    Returns:
        DataFrame: Features ready for ML, or None if failed and
        ``raise_on_error`` is False.
    """
    try:
        if price_data is None or price_data.empty:
            message = f"{ticker}: price data is empty"
            if raise_on_error:
                raise ValueError(message)
            print(f"  [FEATURES] {message}")
            return None

        print(f"  [FEATURES] Original columns: {price_data.columns.tolist()}")

        # yfinance can return flat columns or a MultiIndex, depending on its
        # version and arguments. Normalize both forms before any indicator code.
        df = normalize_ohlcv_columns(price_data, ticker=ticker)
        print(f"  [FEATURES] Normalized columns: {df.columns.tolist()}")

        required = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Convert to numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=required)

        if len(df) < 30:
            message = f"{ticker}: insufficient price data ({len(df)} rows; need at least 30)"
            if raise_on_error:
                raise ValueError(message)
            print(f"  [FEATURES] {message}")
            return None

        # Sort by date if available. yfinance normally keeps the date in the
        # index, which is already chronological; preserve that index when no
        # explicit Date column exists.
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        else:
            df = df.sort_index().reset_index(drop=True)

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
        # LEAK FIX: Shifted by 1 to avoid data leakage (using yesterday's data for today's prediction)
        df['HighLowRatio_lag1'] = ((high - low) / close).shift(1)
        df['CloseOpenRatio_lag1'] = ((close - df['Open']) / df['Open']).shift(1)

        # Returns
        # LEAK FIX: Shifted by 1
        df['DailyReturn_lag1'] = (close.pct_change() * 100).shift(1)

        # Volatility
        df['Volatility'] = df['DailyReturn_lag1'].rolling(20).std()

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
            message = (
                f"{ticker}: insufficient data after feature engineering "
                f"({len(df)} rows; need at least 30)"
            )
            if raise_on_error:
                raise ValueError(message)
            print(f"  [FEATURES] {message}")
            return None

        print(f"  [FEATURES] Final dataset: {len(df)} rows × {df.shape[1]} columns")
        return df

    except KeyError as e:
        print(f"  [FEATURES ERROR] {ticker}: KeyError - {str(e)}")
        print(f"  [FEATURES ERROR] Available columns: {df.columns.tolist() if 'df' in locals() else 'N/A'}")
        if raise_on_error:
            raise
        return None
    except Exception as e:
        print(f"  [FEATURES ERROR] {ticker}: {str(e)[:200]}")
        if raise_on_error:
            raise
        import traceback
        traceback.print_exc()
        return None
