# retrain_all.py  ← このファイル名で保存（宗叡最終対応版）

import pandas as pd
import json
from pathlib import Path
from blowdart_ml_engine import train_ticker   # ← 宗叡版のtrain_ticker（引数なし）
from blowdart_features import build_feature_set
from utils_data_fetch import safe_price_download

TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]

def retrain_with_soei():
    print("=" * 70)
    print("宗叡最終版による全ティッカー再訓練開始")
    print("=" * 70)
    
    results = []
    
    for ticker in TICKERS:
        print(f"\n>>> {ticker} 訓練中...")
        
        try:
            # データ取得
            price_data = safe_price_download(ticker, days=180)
            if price_data is None or price_data.empty:
                print("  データなし")
                results.append({"ticker": ticker, "status": "FAILED", "reason": "No data"})
                continue
                
            # 特徴量生成
            features_df = build_feature_set(price_data, ticker)
            if features_df is None or features_df.empty:
                print("  特徴量生成失敗")
                results.append({"ticker": ticker, "status": "FAILED", "reason": "Feature failed"})
                continue
            
            # 宗叡最終版呼び出し（引数ゼロ！）
            model_info = train_ticker(ticker, features_df)   # ← これだけ！
            
            if model_info:
                regime = model_info.get('regime', 'UNKNOWN')
                acc = model_info.get('accuracies', {}).get('hybrid', 0)
                print(f"  成功 | Regime: {regime} | Hybrid Acc: {acc:.4f}")
                results.append({
                    "ticker": ticker,
                    "status": "SUCCESS",
                    "regime": regime,
                    "accuracy": round(acc, 4)
                })
            else:
                print("  訓練失敗")
                results.append({"ticker": ticker, "status": "FAILED", "reason": "Training failed"})
                
        except Exception as e:
            print(f"  例外: {str(e)[:50]}")
            results.append({"ticker": ticker, "status": "ERROR", "error": str(e)[:50]})
    
    # 結果表示
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("宗叡最終版 訓練結果")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # 保存
    df.to_csv("retraining_results.csv", index=False)
    with open("retraining_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果保存完了 → retraining_results.csv / json")
    print("次は本物の性能が見える。")
    return df

if __name__ == "__main__":
    retrain_with_soei()
