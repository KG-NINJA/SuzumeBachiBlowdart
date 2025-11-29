"""
feature_analysis.py - 特徴量の相関分析とスクリーニング
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from blowdart_features import build_feature_set
from utils_data_fetch import safe_price_download

# Configuration
ANALYSIS_DIR = Path("feature_analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_feature_correlation(ticker, features_df):
    """
    特徴量間の相関を分析
    """
    # 相関行列計算
    corr_matrix = features_df.corr().abs()
    
    # 相関 > 0.9 のペアを検出
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = []
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            if upper_triangle.loc[row, col] > 0.9:
                high_corr_pairs.append({
                    'feature1': col,
                    'feature2': row,
                    'correlation': upper_triangle.loc[row, col]
                })
    
    return corr_matrix, high_corr_pairs


def get_feature_importance(ticker, features_df, target):
    """
    特徴の重要度を計算
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # NaN処理
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # ランダムフォレストで重要度を計算
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features_df, target)
    
    # 重要度をDataFrameに変換
    importance_df = pd.DataFrame({
        'feature': features_df.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


def select_top_features(importance_df, n_features=20):
    """
    重要度上位 N 個の特徴を選択
    """
    return importance_df.head(n_features)['feature'].tolist()


def generate_feature_report(ticker, importance_df, high_corr_pairs):
    """
    特徴量分析レポート生成
    """
    report = f"""
=== Feature Analysis Report for {ticker} ===

Top 20 Important Features:
{importance_df.head(20).to_string()}

High Correlation Pairs (>0.9):
"""
    
    for pair in high_corr_pairs:
        report += f"\n  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}"
    
    return report


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("Feature Analysis for All Tickers")
    print("=" * 70)
    
    TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]
    
    all_results = {}
    
    for ticker in TICKERS:
        print(f"\n>>> {ticker}")
        
        try:
            # データ取得
            price_data = safe_price_download(ticker, days=180)
            
            if price_data is None or price_data.empty:
                print(f"  ✗ No data for {ticker}")
                continue
            
            # 特徴生成
            features_df = build_feature_set(price_data, ticker)
            
            if features_df is None or features_df.empty:
                print(f"  ✗ Feature engineering failed for {ticker}")
                continue
            
            # ターゲット作成
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Close' not in numeric_cols:
                print(f"  ✗ Close column not found for {ticker}")
                continue
            
            df_with_target = features_df.copy()
            df_with_target['Target'] = (df_with_target['Close'].shift(-1) > df_with_target['Close']).astype(int)
            df_with_target = df_with_target.dropna()
            
            if len(df_with_target) < 30:
                print(f"  ✗ Insufficient data for {ticker}")
                continue
            
            # 相関分析
            corr_matrix, high_corr_pairs = analyze_feature_correlation(
                ticker, 
                df_with_target.select_dtypes(include=[np.number])
            )
            
            # 特徴重要度計算
            feature_cols = [c for c in numeric_cols if c != 'Close']
            X = df_with_target[feature_cols]
            y = df_with_target['Target']
            
            importance_df = get_feature_importance(ticker, X, y)
            
            # Top 20 特徴選択
            top_features = select_top_features(importance_df, n_features=20)
            
            # レポート生成
            report = generate_feature_report(ticker, importance_df, high_corr_pairs)
            
            # 結果保存
            all_results[ticker] = {
                'importance': importance_df,
                'top_features': top_features,
                'high_corr_pairs': high_corr_pairs,
                'corr_matrix': corr_matrix
            }
            
            print(f"  ✓ Analysis complete")
            print(f"    - Total features: {len(feature_cols)}")
            print(f"    - Top 20 features selected")
            print(f"    - High correlation pairs: {len(high_corr_pairs)}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:60]}")
    
    print("\n" + "=" * 70)
    print("Feature Analysis Complete")
    print("=" * 70)
    print(f"\nAnalyzed tickers: {len(all_results)}/{len(TICKERS)}")
    
    # Summary statistics
    print("\nTop 10 Most Important Features Across All Tickers:")
    all_importance = []
    for ticker, results in all_results.items():
        for _, row in results['importance'].head(10).iterrows():
            all_importance.append({
                'ticker': ticker,
                'feature': row['feature'],
                'importance': row['importance']
            })
    
    summary_df = pd.DataFrame(all_importance)
    print(summary_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(20))
