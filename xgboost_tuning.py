"""
xgboost_tuning.py - XGBoost ハイパーパラメータ最適化
全ティッカーに対して GridSearchCV でパラメータをチューニング
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from datetime import datetime

# Configuration
TUNING_RESULTS_DIR = Path("tuning_results")
TUNING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]


def tune_xgboost_for_ticker(ticker, X_train, y_train, X_test, y_test):
    """
    XGBoost パラメータをチューニング
    """
    
    print(f"\n>>> Tuning XGBoost for {ticker}")
    print(f"    Training set: {X_train.shape}")
    
    # パラメータグリッド（小さめに設定）
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
    }
    
    # 基本パラメータ
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 100,
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    try:
        # GridSearchCV でチューニング
        xgb_model = xgb.XGBClassifier(**base_params)
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=2,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            error_score=0
        )
        
        print(f"    Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        # 最適パラメータで評価
        best_model = grid_search.best_estimator_
        train_acc = best_model.score(X_train, y_train)
        test_acc = best_model.score(X_test, y_test)
        
        # 結果を記録
        results = {
            'ticker': ticker,
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'improvement': float(test_acc - 0.60)
        }
        
        print(f"    ✓ Best params: {results['best_params']}")
        print(f"    ✓ CV Score: {results['best_cv_score']:.4f}")
        print(f"    ✓ Test Accuracy: {test_acc:.4f}")
        print(f"    ✓ Improvement: {results['improvement']:+.4f}")
        
        return best_model, results
    
    except Exception as e:
        print(f"    ✗ Error: {str(e)[:60]}")
        return None, None


def run_tuning_for_all_tickers():
    """
    全ティッカーでチューニング実行
    """
    from blowdart_features import build_feature_set
    from utils_data_fetch import safe_price_download
    
    print("=" * 70)
    print("XGBoost Hyperparameter Tuning")
    print("=" * 70)
    
    all_results = []
    
    for ticker in TICKERS:
        try:
            # データ取得
            price_data = safe_price_download(ticker, days=180)
            
            if price_data is None or price_data.empty:
                print(f"  ✗ {ticker}: No data")
                continue
            
            # 特徴生成
            features_df = build_feature_set(price_data, ticker)
            
            if features_df is None or features_df.empty:
                print(f"  ✗ {ticker}: Feature engineering failed")
                continue
            
            # ターゲット作成
            df = features_df.copy()
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.dropna()
            
            if len(df) < 30:
                print(f"  ✗ {ticker}: Insufficient data")
                continue
            
            # 特徴とターゲットを分離
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != 'Close']
            
            X = df[feature_cols].copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            y = df['Target'].copy()
            
            # スケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 訓練テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # チューニング実行
            best_model, results = tune_xgboost_for_ticker(
                ticker, X_train, y_train, X_test, y_test
            )
            
            if results:
                all_results.append(results)
        
        except Exception as e:
            print(f"  ✗ {ticker}: {str(e)[:60]}")
    
    # 結果をまとめてレポート
    print("\n" + "=" * 70)
    print("TUNING RESULTS SUMMARY")
    print("=" * 70)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string())
        
        # 平均改善度
        avg_improvement = results_df['improvement'].mean()
        print(f"\nAverage Improvement: {avg_improvement:+.4f}")
        print(f"Expected Accuracy: {0.60 + avg_improvement:.4f} (target: 70%)")
        
        # 結果を保存
        results_df.to_csv(TUNING_RESULTS_DIR / "tuning_results.csv", index=False)
        
        # JSON形式でも保存
        with open(TUNING_RESULTS_DIR / "tuning_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Results saved to {TUNING_RESULTS_DIR}/")
        return results_df
    else:
        print("\n⚠️ No tuning results")
        return None


if __name__ == "__main__":
    results_df = run_tuning_for_all_tickers()
    
    print("\n" + "=" * 70)
    print("XGBoost Tuning Complete")
    print("=" * 70)
