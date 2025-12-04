#!/usr/bin/env python3
"""デバッグスクリプト v2"""
import logging
import sys
import traceback

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, 
                    format='%(name)s: %(message)s')

from utils_data_fetch import safe_price_download
from blowdart_features import build_feature_set
from blowdart_ml_engine import train_model

ticker = 'NVDA'
print(f'=== {ticker} デバッグ v2 ===')

try:
    # データ取得
    price_data = safe_price_download(ticker, days=180)
    print(f'1. Price data: {len(price_data)} rows')
    print(f'   Columns: {price_data.columns.tolist()}')

    # 特徴量生成
    features_df = build_feature_set(price_data, ticker)
    print(f'2. Features: {len(features_df)} rows')
    print(f'   Columns (first 10): {features_df.columns.tolist()[:10]}')
    print(f'   Has Close: {"Close" in features_df.columns}')
    
    # NaNチェック
    nan_count = features_df.isna().sum().sum()
    print(f'3. NaN count: {nan_count}')
    
    # 訓練
    print(f'\n4. Training...')
    acc = train_model(features_df.copy(), ticker)
    print(f'   Result: {acc}')
    
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()
