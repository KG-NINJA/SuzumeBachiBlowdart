import numpy as np
import pandas as pd
from typing import Tuple


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2) -> pd.DataFrame:
    """Add Bollinger Bands features"""
    
    df = df.copy()
    
    # Calculate moving average and standard deviation
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    
    # Bollinger Bands
    df['BB_Upper'] = sma + (std * num_std)
    df['BB_Lower'] = sma - (std * num_std)
    df['BB_Middle'] = sma
    
    # Bollinger Band Position (0-1, where 1 = at upper band)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Position'] = df['BB_Position'].fillna(0.5).clip(0, 1)
    
    # Bollinger Band Width (volatility indicator)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Width'] = df['BB_Width'].fillna(0)
    
    # Band Squeeze (when width is narrow)
    df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(50).quantile(0.25)).astype(int)
    
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average Directional Index (ADX) - trend strength indicator"""
    
    df = df.copy()
    
    # Calculate True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Directional Movements
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(period).mean()
    
    df['ADX'] = adx.fillna(0)
    df['Plus_DI'] = plus_di.fillna(0)
    df['Minus_DI'] = minus_di.fillna(0)
    
    # Trend Direction
    df['Trend_Strength'] = (df['Plus_DI'] > df['Minus_DI']).astype(int)
    
    return df


def add_rsi_divergence(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI and RSI Divergence detection"""
    
    df = df.copy()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 0.0001)
    rsi = 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi.fillna(50)
    
    # RSI Divergence (price makes new high but RSI doesn't)
    price_high = df['Close'] == df['Close'].rolling(period).max()
    rsi_high = df['RSI'] == df['RSI'].rolling(period).max()
    
    df['RSI_Divergence_Bearish'] = (price_high & ~rsi_high).astype(int)
    df['RSI_Divergence_Bullish'] = (~price_high & rsi_high).astype(int)
    
    # RSI Zones
    df['RSI_Zone'] = pd.cut(df['RSI'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutral', 'Overbought'])
    df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD (Moving Average Convergence Divergence)"""
    
    df = df.copy()
    
    # Calculate MACD
    ema_fast = df['Close'].ewm(span=fast).mean()
    ema_slow = df['Close'].ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    df['MACD'] = macd_line.fillna(0)
    df['MACD_Signal'] = signal_line.fillna(0)
    df['MACD_Histogram'] = histogram.fillna(0)
    
    # MACD Crossover
    df['MACD_Bullish_Cross'] = ((macd_line > signal_line) & 
                                 (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
    df['MACD_Bearish_Cross'] = ((macd_line < signal_line) & 
                                 (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
    
    return df


def add_volume_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volume-based features"""
    
    df = df.copy()
    
    # Volume Moving Average
    df['Volume_SMA'] = df['Volume'].rolling(period).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 0.0001)
    
    # On-Balance Volume (OBV)
    obv = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                   np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    df['OBV'] = pd.Series(obv).cumsum()
    df['OBV_SMA'] = df['OBV'].rolling(period).mean()
    
    # Volume Rate of Change
    df['VROC'] = ((df['Volume'] - df['Volume'].shift(period)) / 
                   df['Volume'].shift(period) * 100).fillna(0)
    
    # Price Volume Trend
    df['PVT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
    
    return df


def add_volatility_features(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add volatility clustering indicators"""
    
    df = df.copy()
    
    # Historical Volatility
    returns = df['Close'].pct_change()
    df['HV'] = returns.rolling(period).std() * np.sqrt(252)  # Annualized
    
    # Volatility of Volatility
    df['Vol_of_Vol'] = df['HV'].rolling(20).std()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period).mean()
    df['ATR_Percent'] = (df['ATR'] / df['Close'] * 100).fillna(0)
    
    # Bollinger Band Width (already added, but included here for reference)
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Width_Vol'] = ((sma + 2*std) - (sma - 2*std)) / sma
    
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators"""
    
    df = df.copy()
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
                                df['Close'].shift(period) * 100).fillna(0)
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
    
    # Stochastic Oscillator
    period = 14
    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    
    k_percent = 100 * (df['Close'] - low_min) / (high_max - low_min + 0.0001)
    df['Stoch_K'] = k_percent.rolling(3).mean().fillna(50)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean().fillna(50)
    
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend-following features"""
    
    df = df.copy()
    
    # Moving Average Crossovers
    for fast, slow in [(5, 20), (10, 50), (20, 50)]:
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        
        df[f'EMA_{fast}_{slow}'] = (ema_fast > ema_slow).astype(int)
        df[f'EMA_Distance_{fast}_{slow}'] = ((ema_fast - ema_slow) / ema_slow * 100).fillna(0)
    
    # Support/Resistance Levels
    df['Support'] = df['Low'].rolling(20).min()
    df['Resistance'] = df['High'].rolling(20).max()
    
    df['Distance_to_Support'] = (df['Close'] - df['Support']) / df['Close'] * 100
    df['Distance_to_Resistance'] = (df['Resistance'] - df['Close']) / df['Close'] * 100
    
    return df


def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all advanced features"""
    
    print("  [Features] Adding advanced indicators...")
    
    # Apply all feature engineering functions
    df = add_bollinger_bands(df)
    print("    ✓ Bollinger Bands")
    
    df = add_adx(df)
    print("    ✓ ADX")
    
    df = add_rsi_divergence(df)
    print("    ✓ RSI & Divergence")
    
    df = add_macd(df)
    print("    ✓ MACD")
    
    df = add_volume_features(df)
    print("    ✓ Volume Features")
    
    df = add_volatility_features(df)
    print("    ✓ Volatility Features")
    
    df = add_momentum_features(df)
    print("    ✓ Momentum Features")
    
    df = add_trend_features(df)
    print("    ✓ Trend Features")
    
    # Handle any NaN or Inf values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"  [Features] Total features: {df.shape[1]}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = "AAPL"
    data = yf.download(ticker, period="6mo", progress=False)
    
    # Build advanced features
    data_with_features = build_advanced_features(data)
    
    print(f"\nOriginal columns: {len(data.columns)}")
    print(f"New columns: {len(data_with_features.columns)}")
    print(f"\nFeature columns added:")
    new_cols = set(data_with_features.columns) - set(data.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
