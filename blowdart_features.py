"""
Feature engineering utilities for the Blowdart ML engine.
Relies on a resilient, API-based price fetcher and integrates CV runner signals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils_data_fetch import safe_price_download

DATA_DIR = Path("data")
LOG_DIR = Path("logs")


def ensure_directories() -> None:
    """Create required data directories."""
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "cache").mkdir(parents=True, exist_ok=True)


def _latest_file(pattern: str, directory: Path) -> Optional[Path]:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_cv_runner_features(log_dir: Path = LOG_DIR) -> Dict[str, float]:
    """Load confidence and signal_strength from the latest CV runner output.

    Returns defaults when files are unavailable to keep the pipeline robust.
    """
    defaults = {"cv_confidence": 0.5, "signal_strength": 0.5}
    if not log_dir.exists():
        return defaults

    latest = _latest_file("cv_run_*.json", log_dir)
    if not latest:
        return defaults

    try:
        with latest.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return defaults

    confidence = payload.get("cv_average") or payload.get("confidence")
    signal_strength = payload.get("signal_strength") or payload.get("signal_strength_pct")

    def _normalize(value: Optional[float]) -> float:
        if value is None:
            return 0.5
        try:
            val = float(value)
        except Exception:
            return 0.5
        # most cv values are 0-100; scale to 0-1
        return max(0.0, min(1.0, val / 100.0)) if val > 1 else max(0.0, min(1.0, val))

    return {
        "cv_confidence": _normalize(confidence),
        "signal_strength": _normalize(signal_strength),
    }


def download_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Download price history and standardize columns."""
    ensure_directories()
    price_df = safe_price_download(ticker, range=period)
    # ensure standard columns and ordering
    price_df = price_df.reset_index(drop=True)
    price_df.sort_values("DATE", inplace=True)
    price_df.reset_index(drop=True, inplace=True)
    return price_df


def _rolling_feature(df: pd.DataFrame, column: str, windows: List[int]) -> None:
    for window in windows:
        df[f"{column}_MA_{window}"] = df[column].rolling(window).mean()
        df[f"{column}_STD_{window}"] = df[column].rolling(window).std()
        df[f"{column}_EMA_{window}"] = df[column].ewm(span=window, adjust=False).mean()


def _momentum_features(df: pd.DataFrame) -> None:
    df["RETURN_1D"] = df["CLOSE"].pct_change()
    df["RETURN_3D"] = df["CLOSE"].pct_change(3)
    df["RETURN_7D"] = df["CLOSE"].pct_change(7)
    df["RETURN_14D"] = df["CLOSE"].pct_change(14)
    df["RETURN_21D"] = df["CLOSE"].pct_change(21)
    df["RETURN_30D"] = df["CLOSE"].pct_change(30)
    df["LOG_RETURN"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
    df["MOMENTUM_ACCEL"] = df["RETURN_1D"].diff()


def _volatility_features(df: pd.DataFrame) -> None:
    df["VOLATILITY_5"] = df["RETURN_1D"].rolling(5).std()
    df["VOLATILITY_10"] = df["RETURN_1D"].rolling(10).std()
    df["VOLATILITY_20"] = df["RETURN_1D"].rolling(20).std()
    df["HIGH_LOW_SPREAD"] = (df["HIGH"] - df["LOW"]) / df["CLOSE"]
    df["INTRADAY_RANGE"] = (df["CLOSE"] - df["OPEN"]) / df["OPEN"]
    df["TRUE_RANGE"] = (df[["HIGH", "LOW", "CLOSE"]].max(axis=1) - df[["HIGH", "LOW", "CLOSE"]].min(axis=1)) / df["CLOSE"]


def _momentum_indicators(df: pd.DataFrame) -> None:
    delta = df["CLOSE"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

    ema_fast = df["CLOSE"].ewm(span=12, adjust=False).mean()
    ema_slow = df["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_SIGNAL"] = signal
    df["MACD_HIST"] = macd - signal

    rolling_mean = df["CLOSE"].rolling(window=20).mean()
    rolling_std = df["CLOSE"].rolling(window=20).std()
    df["BB_UPPER"] = rolling_mean + (rolling_std * 2)
    df["BB_LOWER"] = rolling_mean - (rolling_std * 2)
    df["BB_WIDTH"] = df["BB_UPPER"] - df["BB_LOWER"]


def _trend_features(df: pd.DataFrame) -> None:
    for window in [5, 10, 20, 50]:
        rolling = df["CLOSE"].rolling(window)
        df[f"TREND_SLOPE_{window}"] = rolling.apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan,
            raw=False,
        )
        high_roll = df["HIGH"].rolling(window).max()
        low_roll = df["LOW"].rolling(window).min()
        df[f"RANGE_POS_{window}"] = (df["CLOSE"] - low_roll) / ((high_roll - low_roll) + 1e-9)
    df["MA_GAP_5_20"] = df["CLOSE_MA_5"] - df["CLOSE_MA_20"]
    df["MA_GAP_20_50"] = df["CLOSE_MA_20"] - df["CLOSE_MA_50"]


def _volume_pressure(df: pd.DataFrame) -> None:
    df["VOLUME_PRESSURE"] = (df["CLOSE"] - df["OPEN"]) * df["VOLUME"]
    df["VOLUME_RATIO"] = df["VOLUME"] / df["VOLUME_MA_20"]
    df["VOLUME_CHANGE"] = df["VOLUME"].pct_change()


def _lagged_targets(df: pd.DataFrame) -> None:
    for lag in [1, 2, 3, 5]:
        df[f"LAG_CLOSE_{lag}"] = df["CLOSE"].shift(lag)
        df[f"LAG_RETURN_{lag}"] = df["RETURN_1D"].shift(lag)


def _seasonality_features(df: pd.DataFrame) -> None:
    df["DAY_OF_WEEK"] = df["DATE"].dt.weekday
    df["MONTH"] = df["DATE"].dt.month
    df["QUARTER"] = df["DATE"].dt.quarter


def add_cv_runner_columns(df: pd.DataFrame, cv_features: Dict[str, float]) -> None:
    df["CV_CONFIDENCE"] = cv_features.get("cv_confidence", 0.5)
    df["CV_SIGNAL_STRENGTH"] = cv_features.get("signal_strength", 0.5)


def engineer_features(df: pd.DataFrame, cv_features: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Compute a rich set of features plus the binary target."""
    if df.empty:
        return df

    cv_features = cv_features or load_cv_runner_features()

    df = df.copy()
    _rolling_feature(df, "CLOSE", [5, 10, 20, 50])
    _rolling_feature(df, "VOLUME", [5, 20, 50])
    _momentum_features(df)
    _volatility_features(df)
    _momentum_indicators(df)
    _trend_features(df)
    _volume_pressure(df)
    _lagged_targets(df)
    _seasonality_features(df)
    add_cv_runner_columns(df, cv_features)

    df["PRICE_TO_MA5"] = df["CLOSE"] / df["CLOSE_MA_5"]
    df["PRICE_TO_MA20"] = df["CLOSE"] / df["CLOSE_MA_20"]
    df["PRICE_TO_MA50"] = df["CLOSE"] / df["CLOSE_MA_50"]
    df["VOLUME_TO_STD20"] = df["VOLUME"] / (df["VOLUME_STD_20"] + 1e-6)
    df["TARGET"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)

    # Robust filling to keep smaller tickers usable while avoiding leakage
    df.sort_values("DATE", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def build_feature_set(ticker: str, period: str = "3y") -> Tuple[pd.DataFrame, List[str]]:
    """Download data, engineer features, and return dataframe with feature columns."""
    price_df = download_price_history(ticker, period=period)
    enriched = engineer_features(price_df)
    if enriched.empty:
        return enriched, []

    feature_cols = [
        col
        for col in enriched.columns
        if col
        not in {
            "DATE",
            "TARGET",
            "CLOSE",
            "OPEN",
            "HIGH",
            "LOW",
        }
    ]
    return enriched, feature_cols


__all__ = [
    "build_feature_set",
    "engineer_features",
    "download_price_history",
    "load_cv_runner_features",
    "ensure_directories",
]
