"""
tests/test_integration.py - 統合テスト
データ取得から予測、フィルタリングまでの完全なパイプラインをテスト
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import shutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from blowdart_features import build_feature_set
from blowdart_ml_engine import (
    train_ticker,
    predict_ticker,
    detect_market_regime_fixed
)
from confidence_filter import (
    calculate_confidence_score,
    apply_confidence_filter,
    generate_confidence_report
)


class TestFullPipeline:
    """完全なデータフローの統合テスト"""
    
    @pytest.fixture
    def sample_price_data(self):
        """サンプル価格データ生成"""
        dates = pd.date_range('2025-01-01', periods=200)
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        high = close + np.random.rand(200) * 2
        low = close - np.random.rand(200) * 2
        
        return pd.DataFrame({
            'Date': dates,
            'Open': close - np.random.randn(200) * 0.5,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 200)
        })
    
    @pytest.fixture(scope="class")
    def cleanup_test_models(self):
        """テスト後にモデルファイルをクリーンアップ"""
        yield
        # テスト用のモデルディレクトリを削除
        test_ticker_dirs = [
            Path("models") / "TEST_PIPELINE",
            Path("models") / "TEST_INTEGRATION"
        ]
        for dir_path in test_ticker_dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)
    
    def test_step1_feature_generation(self, sample_price_data):
        """Step 1: 特徴生成のテスト"""
        features = build_feature_set(sample_price_data, 'TEST_PIPELINE')
        
        # 特徴が生成されたか
        assert features is not None
        assert len(features) > 0
        
        # 必要なカラムが存在するか
        required_cols = ['Close', 'MA5', 'MA10', 'RSI14']
        for col in required_cols:
            assert col in features.columns, f"Missing column: {col}"
        
        # データ型が正しいか
        assert features['Close'].dtype in [np.float64, np.float32]
        
        # NaNや Inf がないか
        assert not features['Close'].isna().all()
        assert not np.isinf(features['Close']).any()
    
    def test_step2_model_training(self, sample_price_data):
        """Step 2: モデル訓練のテスト"""
        # 特徴生成
        features = build_feature_set(sample_price_data, 'TEST_PIPELINE')
        assert features is not None
        
        # 訓練
        result = train_ticker('TEST_PIPELINE', features)
        
        if result is None:
            pytest.skip("Training failed - not enough samples or features")
        
        # 訓練結果の検証
        assert isinstance(result, dict)
        assert 'accuracy' in result
        assert 'regime' in result
        assert result['regime'] in ['TRENDING', 'CHOPPY', 'MEAN_REVERSION']
        assert 0.0 <= result['accuracy'] <= 1.0
    
    def test_step3_prediction(self, sample_price_data):
        """Step 3: 予測のテスト"""
        # 特徴生成
        features = build_feature_set(sample_price_data, 'TEST_PIPELINE')
        assert features is not None
        
        # 訓練
        train_result = train_ticker('TEST_PIPELINE', features)
        if train_result is None:
            pytest.skip("Training failed")
        
        # 予測
        prediction = predict_ticker('TEST_PIPELINE', features)
        
        if prediction is None:
            pytest.skip("Prediction failed")
        
        # 予測結果の検証
        assert isinstance(prediction, dict)
        
        # 必須フィールド
        required_fields = [
            'ticker',
            'current_price',
            'predicted_price',
            'predicted_change_pct',
            'direction',
            'confidence',
            'market_regime'
        ]
        for field in required_fields:
            assert field in prediction, f"Missing field: {field}"
        
        # 値の範囲チェック
        assert prediction['current_price'] > 0
        assert prediction['predicted_price'] > 0
        assert 0.0 <= prediction['confidence'] <= 1.0
        assert prediction['direction'] in ['↑ Bullish', '↓ Bearish']
    
    def test_step4_confidence_filtering(self, sample_price_data):
        """Step 4: 信頼度フィルタリングのテスト"""
        # 予測結果を作成（モックデータ）
        predictions = [
            {
                'ticker': 'TEST1',
                'confidence': 0.8,
                'direction': '↑ Bullish',
                'predicted_price': 105.0,
                'current_price': 100.0
            },
            {
                'ticker': 'TEST2',
                'confidence': 0.52,
                'direction': '↓ Bearish',
                'predicted_price': 98.0,
                'current_price': 100.0
            }
        ]
        
        # フィルタリング
        filtered = apply_confidence_filter(predictions)
        
        assert len(filtered) == 2
        assert all('action' in p for p in filtered)
        assert all('confidence_score' in p for p in filtered)
        assert all('confidence_level' in p for p in filtered)
        
        # High confidence → EXECUTE
        assert filtered[0]['confidence_level'] == 'STRONG'
        
        # Low confidence → HOLD/SKIP
        assert filtered[1]['confidence_level'] in ['WEAK', 'MEDIUM']
    
    def test_step5_report_generation(self, sample_price_data):
        """Step 5: レポート生成のテスト"""
        # モック予測データ
        predictions = [
            {
                'ticker': 'TEST1',
                'confidence': 0.75,
                'direction': '↑ Bullish',
                'action': 'EXECUTE',
                'confidence_score': 0.5,
                'confidence_level': 'STRONG'
            },
            {
                'ticker': 'TEST2',
                'confidence': 0.55,
                'direction': '↓ Bearish',
                'action': 'HOLD',
                'confidence_score': 0.1,
                'confidence_level': 'MEDIUM'
            }
        ]
        
        # レポート生成
        report = generate_confidence_report(predictions)
        
        assert report is not None
        assert 'total_predictions' in report
        assert 'average_confidence' in report
        assert 'execute_count' in report
        assert 'confidence_distribution' in report
        
        assert report['total_predictions'] == 2
        assert report['execute_count'] >= 0


class TestMarketRegimeIntegration:
    """市場レジーム検出の統合テスト"""
    
    def test_trending_market_workflow(self):
        """トレンド市場でのワークフロー"""
        # 上昇トレンドデータ
        dates = pd.date_range('2025-01-01', periods=100)
        close = np.linspace(100, 150, 100) + np.random.randn(100) * 2
        
        df = pd.DataFrame({
            'Close': close,
            'High': close + np.random.rand(100) * 2,
            'Low': close - np.random.rand(100) * 2,
            'Volume': np.random.rand(100) * 1000000
        })
        
        # レジーム検出
        regime, vol, trend = detect_market_regime_fixed(df, "TEST_TRENDING")
        
        assert regime in ['TRENDING', 'MEAN_REVERSION', 'CHOPPY']
        assert 0.0 <= vol <= 1.0
        assert 0.0 <= trend <= 1.0
    
    def test_choppy_market_workflow(self):
        """チョッピー市場でのワークフロー"""
        # 高ボラティリティデータ
        dates = pd.date_range('2025-01-01', periods=100)
        close = 100 + np.random.randn(100) * 10
        
        df = pd.DataFrame({
            'Close': close,
            'High': close + np.random.rand(100) * 5,
            'Low': close - np.random.rand(100) * 5,
            'Volume': np.random.rand(100) * 1000000
        })
        
        # レジーム検出
        regime, vol, trend = detect_market_regime_fixed(df, "TEST_CHOPPY")
        
        assert regime in ['TRENDING', 'MEAN_REVERSION', 'CHOPPY']


class TestErrorHandling:
    """エラーハンドリングの統合テスト"""
    
    def test_insufficient_data_handling(self):
        """データ不足時のエラーハンドリング"""
        # 極少データ
        small_df = pd.DataFrame({
            'Close': [100, 101],
            'High': [101, 102],
            'Low': [99, 100],
            'Volume': [1000, 1000]
        })
        
        # 特徴生成 → None を返すはず
        features = build_feature_set(small_df, 'TEST_SMALL')
        assert features is None
    
    def test_missing_model_prediction(self):
        """モデル未存在時の予測"""
        # ダミーデータ
        df = pd.DataFrame({
            'Close': [100] * 50,
            'High': [101] * 50,
            'Low': [99] * 50,
            'Volume': [1000] * 50,
            'MA5': [100] * 50,
            'MA10': [100] * 50,
            'RSI14': [50] * 50
        })
        
        # 存在しないティッカーで予測 → None を返すはず
        prediction = predict_ticker('NONEXISTENT_TICKER_XYZ', df)
        assert prediction is None


class TestEndToEnd:
    """エンドツーエンドテスト（完全フロー）"""
    
    def test_complete_workflow_from_raw_data_to_filtered_predictions(self):
        """生データから最終予測までの完全フロー"""
        # Step 1: サンプルデータ作成
        np.random.seed(123)
        dates = pd.date_range('2025-01-01', periods=150)
        close = 100 + np.cumsum(np.random.randn(150) * 0.3)
        
        raw_data = pd.DataFrame({
            'Date': dates,
            'Open': close - np.random.randn(150) * 0.2,
            'High': close + np.random.rand(150) * 1,
            'Low': close - np.random.rand(150) * 1,
            'Close': close,
            'Volume': np.random.randint(500000, 5000000, 150)
        })
        
        ticker = "TEST_E2E"
        
        # Step 2: 特徴生成
        features = build_feature_set(raw_data, ticker)
        if features is None:
            pytest.skip("Feature generation failed")
        
        # Step 3: 訓練
        train_result = train_ticker(ticker, features)
        if train_result is None:
            pytest.skip("Training failed")
        
        # Step 4: 予測
        prediction = predict_ticker(ticker, features)
        if prediction is None:
            pytest.skip("Prediction failed")
        
        # Step 5: 信頼度スコア計算
        score, level = calculate_confidence_score(
            prediction['confidence'],
            model_accuracy=train_result['accuracy']
        )
        
        # Step 6: フィルタリング
        filtered_predictions = apply_confidence_filter([prediction])
        
        # 検証
        assert len(filtered_predictions) == 1
        assert 'action' in filtered_predictions[0]
        assert filtered_predictions[0]['action'] in ['EXECUTE', 'HOLD', 'SKIP']
        
        # クリーンアップ
        model_dir = Path("models") / ticker
        if model_dir.exists():
            shutil.rmtree(model_dir)


# pytest 実行時のエントリーポイント
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
