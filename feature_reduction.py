# feature_reduction.py

"""
特徴量削減ツール
74特徴 → 20特徴に削減し、モデルの過学習を防ぐ
"""

import pandas as pd
import numpy as np

# Phase 2 実行結果から特定された重要な20特徴
TOP_20_FEATURES = [
    'DailyReturn',
    'Momentum_10',
    'OBV',
    'Volume',
    'HighLowRatio',
    'MACD',
    'ATR_Percent',
    'Minus_DI',
    'EMA_Distance_10_50',
    'Momentum',
    'VROC',
    'RSI7',
    'EMA26',
    'Momentum_5',
    'CloseOpenRatio',
    'ATR',
    'Volume_Ratio',
    'Distance_to_Support',
    'PVT',
    'Low'
]

def select_top_features(features_df):
    """
    データフレームから重要な20特徴のみを選択
    
    Args:
        features_df: 全特徴を含むDataFrame
    
    Returns:
        削減されたDataFrame（20特徴 + Close）
    """
    
    # 利用可能な特徴を確認
    available_features = [f for f in TOP_20_FEATURES if f in features_df.columns]
    
    print(f"  [REDUCTION] Selecting {len(available_features)} features from {len(features_df.columns)}")
    
    if 'Close' in features_df.columns:
        selected_cols = available_features + ['Close']
    else:
        selected_cols = available_features
    
    # 特徴を選択
    reduced_df = features_df[selected_cols].copy()
    
    print(f"  [REDUCTION] ✓ Reduced from {len(features_df.columns)} → {len(reduced_df.columns)} features")
    
    return reduced_df


if __name__ == "__main__":
    print("Feature Reduction Module Loaded")
    print(f"Target: {len(TOP_20_FEATURES)} features")
