"""
utils_data_fetch.py - Unified robust data fetcher for SuzumeBachiBlowdart
Supports: Polygon.io, AlphaVantage, Tiingo, yfinance (fallback)
"""

import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

# ===== Configuration =====
CACHE_DIR = "data/cache"
LOGS_DIR = "logs"
FETCH_TIMEOUT = 30

# API Keys from GitHub Secrets (environment variables)
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '').strip()
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '').strip()
TIINGO_API_KEY = os.environ.get('TIINGO_API_KEY', '').strip()

# Ensure directories exist
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# ===== Logging =====
def log_fetch_event(ticker, source, status, message=""):
    """Log fetch attempts for debugging"""
    log_file = f"{LOGS_DIR}/fetch_log_{datetime.now().strftime('%Y%m%d')}.txt"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {ticker:6s} | {source:15s} | {status:8s} | {message}\n"
    
    try:
        with open(log_file, 'a') as f:
            f.write(log_entry)
        print(log_entry.strip())
    except Exception as e:
        print(f"Log write failed: {e}")


def get_cached_data(ticker):
    """Load data from local cache"""
    cache_file = f"{CACHE_DIR}/{ticker}.csv"
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        df = pd.read_csv(cache_file, parse_dates=['Date'])
        log_fetch_event(ticker, "LOCAL_CACHE", "SUCCESS", f"{len(df)} rows")
        return df
    except Exception as e:
        log_fetch_event(ticker, "LOCAL_CACHE", "FAIL", str(e)[:50])
        return None


def save_to_cache(ticker, df):
    """Save fetched data to local cache"""
    if df is None or df.empty:
        return False
    
    try:
        cache_file = f"{CACHE_DIR}/{ticker}.csv"
        df.to_csv(cache_file, index=False)
        return True
    except Exception as e:
        log_fetch_event(ticker, "CACHE_WRITE", "FAIL", str(e)[:50])
        return False


def fetch_polygon_io(ticker, days=180):
    """Fetch from Polygon.io API"""
    if not POLYGON_API_KEY or len(POLYGON_API_KEY) < 10:
        log_fetch_event(ticker, "POLYGON", "SKIP", "No API key")
        return None
    
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"apikey": POLYGON_API_KEY, "limit": 50000, "sort": "asc"}
        
        print(f"  [POLYGON] Fetching {ticker}...")
        response = requests.get(url, params=params, timeout=FETCH_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'OK' or 'results' not in data:
            log_fetch_event(ticker, "POLYGON", "FAIL", f"Status: {data.get('status')}")
            return None
        
        results = data['results']
        if not results:
            log_fetch_event(ticker, "POLYGON", "EMPTY", "No data")
            return None
        
        records = []
        for bar in results:
            try:
                records.append({
                    'Date': pd.to_datetime(bar['t'], unit='ms'),
                    'Open': float(bar.get('o', 0)),
                    'High': float(bar.get('h', 0)),
                    'Low': float(bar.get('l', 0)),
                    'Close': float(bar.get('c', 0)),
                    'Volume': float(bar.get('v', 0))
                })
            except:
                continue
        
        if not records:
            log_fetch_event(ticker, "POLYGON", "PARSE_FAIL", "No valid records")
            return None
        
        df = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)
        log_fetch_event(ticker, "POLYGON", "SUCCESS", f"{len(df)} rows")
        return df
    
    except requests.exceptions.Timeout:
        log_fetch_event(ticker, "POLYGON", "TIMEOUT", f"{FETCH_TIMEOUT}s")
    except requests.exceptions.RequestException as e:
        log_fetch_event(ticker, "POLYGON", "NETWORK", str(e)[:40])
    except Exception as e:
        log_fetch_event(ticker, "POLYGON", "ERROR", str(e)[:40])
    
    return None


def fetch_alpha_vantage(ticker, days=180):
    """Fetch from Alpha Vantage API"""
    if not ALPHA_VANTAGE_KEY or len(ALPHA_VANTAGE_KEY) < 10:
        log_fetch_event(ticker, "ALPHA_VANTAGE", "SKIP", "No API key")
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_KEY,
            "outputsize": "full"
        }
        
        print(f"  [ALPHA_VANTAGE] Fetching {ticker}...")
        response = requests.get(url, params=params, timeout=FETCH_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if "Note" in data or "Error Message" in data:
            log_fetch_event(ticker, "ALPHA_VANTAGE", "API_ERR", data.get("Note", data.get("Error Message"))[:40])
            return None
        
        if "Time Series (Daily)" not in data:
            log_fetch_event(ticker, "ALPHA_VANTAGE", "FAIL", "No Time Series")
            return None
        
        time_series = data["Time Series (Daily)"]
        records = []
        
        for date_str, values in time_series.items():
            try:
                records.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values.get('1. open', 0)),
                    'High': float(values.get('2. high', 0)),
                    'Low': float(values.get('3. low', 0)),
                    'Close': float(values.get('4. close', 0)),
                    'Volume': float(values.get('5. volume', 0))
                })
            except:
                continue
        
        if not records:
            log_fetch_event(ticker, "ALPHA_VANTAGE", "PARSE_FAIL", "No valid records")
            return None
        
        df = pd.DataFrame(records).sort_values('Date').tail(days).reset_index(drop=True)
        log_fetch_event(ticker, "ALPHA_VANTAGE", "SUCCESS", f"{len(df)} rows")
        return df
    
    except Exception as e:
        log_fetch_event(ticker, "ALPHA_VANTAGE", "ERROR", str(e)[:40])
    
    return None


def fetch_tiingo(ticker, days=180):
    """Fetch from Tiingo API"""
    if not TIINGO_API_KEY or len(TIINGO_API_KEY) < 10:
        log_fetch_event(ticker, "TIINGO", "SKIP", "No API key")
        return None
    
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "token": TIINGO_API_KEY
        }
        
        print(f"  [TIINGO] Fetching {ticker}...")
        response = requests.get(url, params=params, timeout=FETCH_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            log_fetch_event(ticker, "TIINGO", "EMPTY", "No data")
            return None
        
        records = []
        for item in data:
            try:
                records.append({
                    'Date': pd.to_datetime(item['date']),
                    'Open': float(item.get('open', 0)),
                    'High': float(item.get('high', 0)),
                    'Low': float(item.get('low', 0)),
                    'Close': float(item.get('close', 0)),
                    'Volume': float(item.get('volume', 0))
                })
            except:
                continue
        
        if not records:
            log_fetch_event(ticker, "TIINGO", "PARSE_FAIL", "No valid records")
            return None
        
        df = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)
        log_fetch_event(ticker, "TIINGO", "SUCCESS", f"{len(df)} rows")
        return df
    
    except Exception as e:
        log_fetch_event(ticker, "TIINGO", "ERROR", str(e)[:40])
    
    return None


def fetch_yfinance(ticker, days=180):
    """Fetch from yfinance (free, no auth needed)"""
    try:
        print(f"  [YFINANCE] Fetching {ticker}...")
        
        df = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=False)
        
        if df is None or df.empty:
            log_fetch_event(ticker, "YFINANCE", "EMPTY", "No data")
            return None
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        # Standardize column names
        if 'date' not in df.columns and 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
        
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            log_fetch_event(ticker, "YFINANCE", "COLUMNS", f"Missing: {set(required_cols) - set(df.columns)}")
            return None
        
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        log_fetch_event(ticker, "YFINANCE", "SUCCESS", f"{len(df)} rows")
        return df
    
    except Exception as e:
        log_fetch_event(ticker, "YFINANCE", "ERROR", str(e)[:40])
    
    return None


def safe_price_download(ticker, days=180):
    """
    Main fetcher: Try multiple sources with fallback
    Priority: Cache > Polygon > AlphaVantage > Tiingo > yfinance
    """
    print(f"\n{'='*60}")
    print(f"Fetching {ticker}...")
    print(f"{'='*60}")
    
    # Step 1: Try cache first
    cached_df = get_cached_data(ticker)
    if cached_df is not None and not cached_df.empty and len(cached_df) >= 20:
        return cached_df
    
    # Step 2: Try APIs in order
    fetchers = [
        ("Polygon.io", fetch_polygon_io),
        ("Alpha Vantage", fetch_alpha_vantage),
        ("Tiingo", fetch_tiingo),
        ("yfinance", fetch_yfinance)
    ]
    
    for name, fetcher in fetchers:
        df = fetcher(ticker, days)
        if df is not None and not df.empty and len(df) >= 20:
            save_to_cache(ticker, df)
            return df
        time.sleep(1)  # Rate limiting
    
    # Step 3: Return cache even if old (better than nothing)
    log_fetch_event(ticker, "FALLBACK", "USING_CACHE", "All APIs failed")
    return cached_df  # May be None
