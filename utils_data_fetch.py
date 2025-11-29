"""
Robust, rate-limit-aware price downloader supporting Polygon.io, AlphaVantage, and Tiingo.
Provides daily caching to minimize API calls and automatic provider fallback.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import pandas as pd
import requests
import yfinance as yf

REQUEST_UA = os.getenv("REQUEST_UA", "Mozilla/5.0 (Codex GitHub Runner)")
DATA_CACHE_DIR = Path("data") / "cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]


def _headers() -> Dict[str, str]:
    return {"User-Agent": REQUEST_UA, "Accept": "application/json"}


def _range_to_start_date(range_value: str = "2y") -> datetime:
    range_value = range_value.lower().strip()
    if range_value.endswith("y"):
        try:
            years = int(range_value[:-1])
        except ValueError:
            years = 2
        return datetime.utcnow() - timedelta(days=365 * years)
    if range_value.endswith("d"):
        try:
            days = int(range_value[:-1])
        except ValueError:
            days = 365 * 2
        return datetime.utcnow() - timedelta(days=days)
    return datetime.utcnow() - timedelta(days=365 * 2)


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Price dataframe is empty after download")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Price dataframe missing required columns: {missing}")
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.sort_values("DATE", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _cache_path(ticker: str) -> Path:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return DATA_CACHE_DIR / f"{ticker.upper()}_{today}.json"


def _load_cache(ticker: str) -> Optional[pd.DataFrame]:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        df = pd.DataFrame(payload)
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"])
        return _validate_df(df)
    except Exception as exc:  # pragma: no cover - cache failures fallback to fetch
        print(f"[cache] Failed to load cache for {ticker}: {exc}")
        return None


def _load_legacy_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Load legacy CSV caches shipped in ``data/cache`` as an offline fallback.

    The repository includes historical CSVs with ticker-suffixed columns (e.g.,
    ``CLOSE_AAPL``). This helper normalizes them into the unified schema to
    allow pipelines to execute without live network/API access.
    """

    legacy_path = DATA_CACHE_DIR / f"{ticker.upper()}.csv"
    if not legacy_path.exists():
        return None

    try:
        df = pd.read_csv(legacy_path)
        if df.empty:
            return None

        # Normalize column names to uppercase and strip ticker suffixes.
        df = df.rename(columns={col: str(col).upper() for col in df.columns})
        suffix = f"_{ticker.upper()}"
        normalized = {}
        for col in df.columns:
            base = col
            if col.endswith(suffix):
                base = col[: -len(suffix)]
            if base == "DATE_":
                base = "DATE"
            normalized[col] = base
        df.rename(columns=normalized, inplace=True)

        # Prefer adjusted close when close is missing.
        if "ADJ CLOSE" in df.columns and "CLOSE" not in df.columns:
            df["CLOSE"] = df["ADJ CLOSE"]

        required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"legacy cache missing columns: {missing}")

        df = df[required]
        return _validate_df(df)
    except Exception as exc:  # pragma: no cover - legacy fallback best-effort
        print(f"[cache] Failed to load legacy cache for {ticker}: {exc}")
        return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    try:
        path = _cache_path(ticker)
        payload = df.copy()
        payload["DATE"] = payload["DATE"].dt.strftime("%Y-%m-%d")
        path.write_text(payload.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - caching should not break pipeline
        print(f"[cache] Failed to save cache for {ticker}: {exc}")


def polygon_fetch(ticker: str, range: str = "2y") -> pd.DataFrame:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY is not set")
    start_date = _range_to_start_date(range).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "limit": 50000, "sort": "asc", "apiKey": api_key}
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("Polygon rate limit reached (429)")
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        raise ValueError("Polygon returned no data")
    records = []
    for row in results:
        records.append(
            {
                "DATE": datetime.utcfromtimestamp(row.get("t", 0) / 1000.0),
                "OPEN": row.get("o"),
                "HIGH": row.get("h"),
                "LOW": row.get("l"),
                "CLOSE": row.get("c"),
                "VOLUME": row.get("v"),
            }
        )
    df = pd.DataFrame(records)
    return _validate_df(df)


def alpha_fetch(ticker: str, range: str = "2y") -> pd.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY is not set")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": api_key,
        "outputsize": "full",
    }
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("AlphaVantage rate limit reached (429)")
    resp.raise_for_status()
    payload = resp.json()
    if "Error Message" in payload:
        raise ValueError(payload.get("Error Message"))
    time_series = payload.get("Time Series (Daily)", {})
    if not time_series:
        raise ValueError("AlphaVantage returned no data")
    df = pd.DataFrame.from_dict(time_series, orient="index")
    df.reset_index(inplace=True)
    df.rename(
        columns={
            "index": "DATE",
            "1. open": "OPEN",
            "2. high": "HIGH",
            "3. low": "LOW",
            "4. close": "CLOSE",
            "6. volume": "VOLUME",
        },
        inplace=True,
    )
    df = df[REQUIRED_COLUMNS]
    df["OPEN"] = pd.to_numeric(df["OPEN"], errors="coerce")
    df["HIGH"] = pd.to_numeric(df["HIGH"], errors="coerce")
    df["LOW"] = pd.to_numeric(df["LOW"], errors="coerce")
    df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
    df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")
    start_cutoff = _range_to_start_date(range)
    df = df[df["DATE"] >= start_cutoff.strftime("%Y-%m-%d")]
    return _validate_df(df)


def tiingo_fetch(ticker: str, range: str = "2y") -> pd.DataFrame:
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise ValueError("TIINGO_API_KEY is not set")
    start_date = _range_to_start_date(range).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "resampleFreq": "daily",
        "format": "json",
        "token": api_key,
    }
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    if resp.status_code == 429:
        raise RuntimeError("Tiingo rate limit reached (429)")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or not data:
        raise ValueError("Tiingo returned no data")
    records = []
    for row in data:
        records.append(
            {
                "DATE": row.get("date"),
                "OPEN": row.get("open"),
                "HIGH": row.get("high"),
                "LOW": row.get("low"),
                "CLOSE": row.get("close"),
                "VOLUME": row.get("volume"),
            }
        )
    df = pd.DataFrame(records)
    return _validate_df(df)


def yfinance_fetch(ticker: str, range: str = "2y") -> pd.DataFrame:
    start = _range_to_start_date(range)
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError("yfinance returned no data")
    df = df.reset_index()
    df.rename(
        columns={
            "Date": "DATE",
            "Open": "OPEN",
            "High": "HIGH",
            "Low": "LOW",
            "Close": "CLOSE",
            "Volume": "VOLUME",
        },
        inplace=True,
    )
    df = df[REQUIRED_COLUMNS]
    return _validate_df(df)


def _attempt_fetchers(
    fetchers: Iterable[Callable[[str, str], pd.DataFrame]], ticker: str, range: str
) -> Optional[pd.DataFrame]:
    for fetcher in fetchers:
        try:
            print(f"[fetch:{fetcher.__name__}] starting for {ticker}")
            df = fetcher(ticker, range)
            print(f"[fetch:{fetcher.__name__}] success for {ticker} -> {len(df)} rows")
            return df
        except Exception as exc:
            print(f"[fetch:{fetcher.__name__}] {ticker} failed: {exc}")
    return None


def safe_price_download(
    ticker: str,
    range_: str | None = "2y",
    *,
    days: Optional[int] = None,
    max_attempts: int = 6,
) -> pd.DataFrame:
    """Download price data with caching and flexible lookback arguments.

    Supports the historical ``range`` string used elsewhere in the repo and the
    ``days`` keyword expected by some ad-hoc runners (e.g., the provided
    training harness). ``days`` takes precedence when supplied.
    """

    ticker = ticker.upper()

    lookback_range = range_
    if days is not None:
        try:
            lookback_range = f"{int(days)}d"
        except Exception:
            lookback_range = "2y"
    if not lookback_range:
        lookback_range = "2y"

    cached = _load_cache(ticker)
    if cached is not None:
        print(f"[cache] hit for {ticker}")
        return cached

    legacy_cached = _load_legacy_cache(ticker)
    if legacy_cached is not None:
        print(f"[cache] legacy hit for {ticker}")
        return legacy_cached

    # Prefer Tiingo first because it is the most reliable with a free tier,
    # then AlphaVantage, then Polygon, and finally yfinance as a network-only fallback.
    fetchers = [tiingo_fetch, alpha_fetch, polygon_fetch, yfinance_fetch]
    for attempt in range(max_attempts):
        print(
            f"[download] Attempt {attempt + 1}/{max_attempts} for {ticker} (range={lookback_range})"
        )
        df = _attempt_fetchers(fetchers, ticker, lookback_range)
        if df is not None:
            df = _validate_df(df)
            _save_cache(ticker, df)
            return df
        sleep_time = min(60, 2 ** attempt)
        print(f"[download] Retry in {sleep_time}s for {ticker}")
        time.sleep(sleep_time)

    raise RuntimeError(f"Failed to download price data for {ticker} after {max_attempts} attempts")


__all__ = [
    "safe_price_download",
    "polygon_fetch",
    "alpha_fetch",
    "tiingo_fetch",
    "yfinance_fetch",
]
