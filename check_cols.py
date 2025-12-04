#!/usr/bin/env python3
"""カラム名確認"""
from utils_data_fetch import safe_price_download
from blowdart_features import build_feature_set

ticker = 'NVDA'
price_data = safe_price_download(ticker, days=180)
features_df = build_feature_set(price_data, ticker)

print("Available columns:")
for col in sorted(features_df.columns):
    print(f"  {col}")
