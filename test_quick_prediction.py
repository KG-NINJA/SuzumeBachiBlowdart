#!/usr/bin/env python3
"""
test_quick_prediction.py - クイック予測テスト
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# モジュールインポート
from prediction_engine.models.multitimescale import MultiTimeScaleForecast
from prediction_engine.models.confidence_scoring import ConfidenceScoring

def create_sample_data(days=150):
    """サンプルデータ生成"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days)
    close = 100 + np.cumsum(np.random.randn(days) * 0.5)
    
    return pd.DataFrame({
        'Open': close - np.random.randn(days) * 0.3,
        'High': close + np.random.rand(days) * 2,
        'Low': close - np.random.rand(days) * 2,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)

def main():
    print("=" * 60)
    print("📊 予測エンジン クイックテスト")
    print("=" * 60)
    
    # サンプルデータ作成
    print("\n[1/4] サンプルデータ生成中...")
    data = create_sample_data(150)
    print(f"  ✓ {len(data)} 日分のデータを生成")
    
    # モデル作成
    print("\n[2/4] モデル訓練中 (LSTM無効で高速化)...")
    mts = MultiTimeScaleForecast(use_lstm=False)
    train_result = mts.train(data, 'TEST', epochs=10, verbose=0)
    
    print(f"  ✓ 訓練完了")
    print(f"    - 訓練サンプル: {train_result['training_samples']}")
    print(f"    - テストサンプル: {train_result['test_samples']}")
    
    for model_name, acc in train_result['accuracies'].items():
        print(f"    - {model_name}: {acc:.2%}")
    
    # 予測実行
    print("\n[3/4] 予測実行中...")
    prediction = mts.predict('TEST', data)
    
    current_price = prediction['current_price']
    print(f"  ✓ 予測完了")
    print(f"    現在価格: ${current_price:.2f}")
    
    for term, pred in prediction['predictions'].items():
        direction = pred.get('direction', 'N/A')
        emoji = "📈" if direction == "UP" else "📉"
        confidence = pred.get('confidence', 0)
        
        print(f"\n    【{term}】")
        print(f"      方向: {emoji} {direction}")
        print(f"      信頼度: {confidence:.2%}")
        
        if 'forecast_3d' in pred:
            change = (pred['forecast_3d'] / current_price - 1) * 100
            print(f"      3日後: ${pred['forecast_3d']:.2f} ({change:+.2f}%)")
        elif 'forecast_4w' in pred:
            change = (pred['forecast_4w'] / current_price - 1) * 100
            print(f"      4週後: ${pred['forecast_4w']:.2f} ({change:+.2f}%)")
        elif 'forecast_3m' in pred:
            change = (pred['forecast_3m'] / current_price - 1) * 100
            print(f"      3ヶ月後: ${pred['forecast_3m']:.2f} ({change:+.2f}%)")
    
    # 信頼度評価
    print("\n[4/4] 信頼度評価中...")
    scorer = ConfidenceScoring()
    confidence = scorer.score(prediction, data, train_result['accuracies'])
    
    print(f"  ✓ 評価完了")
    print(f"    総合スコア: {confidence['confidence']:.2%}")
    print(f"    レベル: {confidence['confidence_level']}")
    print(f"    推奨: {confidence['recommendation']}")
    
    print("\n" + "=" * 60)
    print("✅ 全テスト成功!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
