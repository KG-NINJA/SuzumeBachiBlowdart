"""
test_multitimescale.py - MultiTimeScaleForecast のユニットテスト

"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prediction_engine.models.multitimescale import MultiTimeScaleForecast


class TestMultiTimeScaleForecast:
    """マルチタイムスケール予測のテスト"""
    
    @pytest.fixture
    def sample_data(self):
        """サンプルデータ生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=150)
        close = 100 + np.cumsum(np.random.randn(150) * 0.5)
        
        return pd.DataFrame({
            'Date': dates,
            'Open': close - np.random.randn(150) * 0.3,
            'High': close + np.random.rand(150) * 2,
            'Low': close - np.random.rand(150) * 2,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 150)
        }).set_index('Date')
    
    @pytest.fixture
    def model(self):
        """モデルインスタンス"""
        return MultiTimeScaleForecast(use_lstm=False)  # テスト高速化のためLSTMオフ
    
    def test_initialization(self, model):
        """初期化テスト"""
        assert model is not None
        assert model.is_trained == False
    
    def test_prepare_features(self, model, sample_data):
        """特徴量準備テスト"""
        features = model._prepare_features(sample_data)
        
        assert features is not None
        assert len(features) > 0
        assert 'return_1d' in features.columns
        assert 'volatility_10d' in features.columns
    
    def test_train(self, model, sample_data):
        """訓練テスト"""
        result = model.train(sample_data, 'TEST', epochs=5)
        
        assert result is not None
        assert 'accuracies' in result
        assert model.is_trained == True
    
    def test_predict_without_training(self, model, sample_data):
        """未訓練時の予測テスト"""
        result = model.predict('TEST', sample_data)
        
        # フォールバック予測が返される
        assert result is not None
        assert 'predictions' in result
        assert result['model_info'].get('fallback') == True
    
    def test_predict_after_training(self, model, sample_data):
        """訓練後の予測テスト"""
        # 訓練
        model.train(sample_data, 'TEST', epochs=5)
        
        # 予測
        result = model.predict('TEST', sample_data)
        
        assert result is not None
        assert 'ticker' in result
        assert 'predictions' in result
        
        # 各時間軸の予測を確認
        assert 'short_term' in result['predictions']
        assert 'medium_term' in result['predictions']
        assert 'long_term' in result['predictions']
        
        # 短期予測の詳細
        short = result['predictions']['short_term']
        assert 'forecast_3d' in short
        assert 'direction' in short
        assert 'confidence' in short
        assert short['direction'] in ['UP', 'DOWN']
        assert 0 <= short['confidence'] <= 1
    
    def test_prediction_prices_reasonable(self, model, sample_data):
        """予測価格が妥当な範囲かテスト"""
        model.train(sample_data, 'TEST', epochs=5)
        result = model.predict('TEST', sample_data)
        
        current = result['current_price']
        
        # 短期予測は±10%以内
        short_3d = result['predictions']['short_term']['forecast_3d']
        assert 0.9 * current <= short_3d <= 1.1 * current
        
        # 長期予測は±20%以内
        long_3m = result['predictions']['long_term']['forecast_3m']
        assert 0.8 * current <= long_3m <= 1.2 * current


class TestConfidenceScoring:
    """信頼度スコアリングのテスト"""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)
        return pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    @pytest.fixture
    def sample_prediction(self):
        return {
            'ticker': 'TEST',
            'predictions': {
                'short_term': {'probability': 0.72, 'confidence': 0.44},
                'medium_term': {'probability': 0.58, 'confidence': 0.16},
                'long_term': {'probability': 0.52, 'confidence': 0.04}
            }
        }
    
    def test_score_returns_valid_result(self, sample_data, sample_prediction):
        from prediction_engine.models.confidence_scoring import ConfidenceScoring
        
        scorer = ConfidenceScoring()
        result = scorer.score(sample_prediction, sample_data)
        
        assert 'confidence' in result
        assert 'confidence_level' in result
        assert 'breakdown' in result
        assert 0 <= result['confidence'] <= 1
        assert result['confidence_level'] in ['HIGH', 'MODERATE', 'LOW']


# pytest 実行時のエントリーポイント
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
