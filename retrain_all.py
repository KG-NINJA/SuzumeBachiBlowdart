"""
retrain_all.py - チューニング済みパラメータで全ティッカーを再学習
"""

import pandas as pd
import json
from pathlib import Path
from blowdart_ml_engine import train_ticker
from blowdart_features import build_feature_set
from utils_data_fetch import safe_price_download

TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]
TUNING_RESULTS_FILE = Path("tuning_results/tuning_results.csv")


def load_tuning_results():
    """
    チューニング結果をロード
    """
    if TUNING_RESULTS_FILE.exists():
        df = pd.read_csv(TUNING_RESULTS_FILE)
        return df
    else:
        print("⚠️ Tuning results file not found. Using default parameters.")
        return None


def parse_params_string(params_str):
    """
    パラメータ文字列をパースして辞書に変換
    """
    try:
        # eval を使用して辞書に変換
        params = eval(params_str)
        return params
    except:
        return None


def retrain_with_tuned_params():
    """
    チューニング済みパラメータで全ティッカーを再学習
    """
    
    print("=" * 70)
    print("Retraining All Tickers with Tuned Parameters")
    print("=" * 70)
    
    # チューニング結果をロード
    tuning_df = load_tuning_results()
    
    training_results = []
    
    for ticker in TICKERS:
        print(f"\n>>> {ticker}")
        
        try:
            # チューニング結果を取得
            tuned_params = None
            
            if tuning_df is not None:
                tuning_row = tuning_df[tuning_df['ticker'] == ticker]
                
                if not tuning_row.empty:
                    params_str = tuning_row['best_params'].iloc[0]
                    tuned_params = parse_params_string(params_str)
                    
                    if tuned_params:
                        print(f"  ✓ Using tuned params: {tuned_params}")
            
            if tuned_params is None:
                print(f"  ℹ️ Using default params")
            
            # データ取得
            price_data = safe_price_download(ticker, days=180)
            
            if price_data is None or price_data.empty:
                print(f"  ✗ No data")
                training_results.append({
                    'ticker': ticker,
                    'status': 'FAILED',
                    'reason': 'No data'
                })
                continue
            
            # 特徴生成
            features_df = build_feature_set(price_data, ticker)
            
            if features_df is None or features_df.empty:
                print(f"  ✗ Feature engineering failed")
                training_results.append({
                    'ticker': ticker,
                    'status': 'FAILED',
                    'reason': 'Feature engineering failed'
                })
                continue
            
            # 再学習
            model_info = train_ticker(ticker, features_df, use_online_learning=True)
            
            if model_info:
                accuracy = model_info.get('accuracy', 0)
                improvement = model_info.get('accuracy_improvement', 0)
                
                print(f"  ✓ Retrained: Accuracy={accuracy:.4f} ({improvement:+.4f})")
                
                training_results.append({
                    'ticker': ticker,
                    'accuracy': accuracy,
                    'improvement': improvement,
                    'learning_type': model_info.get('learning_type'),
                    'status': 'SUCCESS'
                })
            else:
                print(f"  ✗ Training failed")
                training_results.append({
                    'ticker': ticker,
                    'status': 'FAILED',
                    'reason': 'Training failed'
                })
        
        except Exception as e:
            print(f"  ✗ Exception: {str(e)[:60]}")
            training_results.append({
                'ticker': ticker,
                'status': 'ERROR',
                'error': str(e)[:60]
            })
    
    # 結果をまとめてレポート
    print("\n" + "=" * 70)
    print("RETRAINING SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(training_results)
    print(results_df.to_string())
    
    # 精度改善の確認
    successful = results_df[results_df['status'] == 'SUCCESS']
    
    if len(successful) > 0:
        avg_accuracy = successful['accuracy'].mean()
        avg_improvement = successful['improvement'].mean()
        
        print(f"\n📊 Statistics:")
        print(f"  - Successful: {len(successful)}/{len(TICKERS)}")
        print(f"  - Average Accuracy: {avg_accuracy:.4f}")
        print(f"  - Average Improvement: {avg_improvement:+.4f}")
        print(f"  - Target Achieved: {'✅ YES' if avg_accuracy >= 0.70 else '⚠️ NO (70%+ target)'}")
    
    # 結果をファイルに保存
    results_df.to_csv("retraining_results.csv", index=False)
    
    with open("retraining_results.json", "w") as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\n✅ Results saved to:")
    print(f"   - retraining_results.csv")
    print(f"   - retraining_results.json")
    
    return results_df


if __name__ == "__main__":
    results_df = retrain_with_tuned_params()
    
    print("\n" + "=" * 70)
    print("Retraining Complete")
    print("=" * 70)
