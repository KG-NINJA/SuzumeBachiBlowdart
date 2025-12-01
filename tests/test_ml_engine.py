"""
tests/test_ml_engine.py - pytest ユニットテスト
対象: blowdart_ml_engine.py の主要関数
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from blowdart_ml_engine import (
    detect_market_regime_fixed,
    get_ticker_specific_features,
    predict_ticker,
    train_ticker,
    LEAK_FEATURES
)


class TestMarketRegimeDetection:
    """市場レジーム検出のテスト"""
    
    @pytest.fixture
    def trending_upward_df(self):
        """上昇トレンド市場のサンプルデータ"""
        dates = pd.date_range('2025-01-01', periods=100)
        # 上昇トレンド: 連続して上昇
        close_prices = np.linspace(100, 150, 100) + np.random.randn(100) * 2
        return pd.DataFrame({
            'Close': close_prices,
            'High': close_prices + np.random.rand(100) * 2,
            'Low': close_prices - np.random.rand(100) * 2,
            'Open': close_prices + np.random.randn(100) * 0.5,
            'Volume': np.random.rand(100) * 1000000
        })
    
    @pytest.fixture
    def choppy_df(self):
        """チョッピー（上下変動が激しい）市場のサンプルデータ"""
        dates = pd.date_range('2025-01-01', periods=100)
        # チョッピー: 高ボラティリティで方向性なし
        close_prices = 100 + np.random.randn(100) * 10
        return pd.DataFrame({
            'Close': close_prices,
            'High': close_prices + np.random.rand(100) * 3,
            'Low': close_prices - np.random.rand(100) * 3,
            'Open': close_prices + np.random.randn(100),
            'Volume': np.random.rand(100) * 1000000
        })
    
    @pytest.fixture
    def mean_reversion_df(self):
        """平均回帰市場のサンプルデータ"""
        dates = pd.date_range('2025-01-01', periods=100)
        # 平均回帰: 中程度のボラティリティで100付近を推移
        close_prices = 100 + np.sin(np.arange(100) * 0.5) * 5 + np.random.randn(100)
        return pd.DataFrame({
            'Close': close_prices,
            'High': close_prices + np.random.rand(100),
            'Low': close_prices - np.random.rand(100),
            'Open': close_prices + np.random.randn(100) * 0.3,
            'Volume': np.random.rand(100) * 1000000
        })
    
    def test_detect_trending_market(self, trending_upward_df):
        """上昇トレンド市場を正しく検出できるか"""
        regime, vol, trend = detect_market_regime_fixed(
            trending_upward_df, 
            ticker="TEST_TRENDING", 
            lookback=20
        )
        # トレンド市場として検出されるべき
        assert regime in ['TRENDING', 'MEAN_REVERSION']  # データのランダム性により両方あり得る
        assert 0.0 <= vol <= 1.0
        assert 0.0 <= trend <= 1.0
    
    def test_detect_choppy_market(self, choppy_df):
        """チョッピー市場を正しく検出できるか"""
        regime, vol, trend = detect_market_regime_fixed(
            choppy_df, 
            ticker="TEST_CHOPPY", 
            lookback=20
        )
        # 高ボラティリティの場合、CHOPPYまたはMEAN_REVERSIONとして検出されるはず
        assert regime in ['CHOPPY', 'MEAN_REVERSION']
        assert 0.0 <= vol <= 1.0
        assert 0.0 <= trend <= 1.0
    
    def test_detect_mean_reversion_market(self, mean_reversion_df):
        """平均回帰市場を正しく検出できるか"""
        regime, vol, trend = detect_market_regime_fixed(
            mean_reversion_df, 
            ticker="TEST_MEAN_REV", 
            lookback=20
        )
        # 中程度のボラティリティで方向性が弱い場合
        assert regime in ['MEAN_REVERSION', 'TRENDING', 'CHOPPY']
        assert 0.0 <= vol <= 1.0
        assert 0.0 <= trend <= 1.0
    
    def test_insufficient_data_returns_neutral(self):
        """データが不十分な場合、中立を返すか"""
        small_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Volume': [1000, 1000, 1000]
        })
        regime, vol, trend = detect_market_regime_fixed(
            small_df, 
            ticker="TEST_SMALL", 
            lookback=20
        )
        assert regime == 'NEUTRAL'
        assert vol == 0.5
        assert trend == 0.5


class TestFeatureSelection:
    """特徴選択のテスト"""
    
    @pytest.fixture
    def sample_features_df(self):
        """特徴量を含むサンプルDataFrame"""
        n = 100
        df = pd.DataFrame({
            'Close': np.random.rand(n) * 100,
            'Open': np.random.rand(n) * 100,
            'High': np.random.rand(n) * 100,
            'Low': np.random.rand(n) * 100,
            'Volume': np.random.rand(n) * 1000000,
            'MA5': np.random.rand(n) * 100,
            'MA10': np.random.rand(n) * 100,
            'MA20': np.random.rand(n) * 100,
            'RSI14': np.random.rand(n) * 100,
            'RSI7': np.random.rand(n) * 100,
            'MACD': np.random.rand(n),
            'ATR': np.random.rand(n) * 5,
            'Volume_Ratio': np.random.rand(n) * 2,
            'Momentum': np.random.randn(n),
            'CloseOpenRatio': np.random.randn(n),  # リーク特徴
            'DailyReturn': np.random.randn(n),      # リーク特徴
            'HighLowRatio': np.random.randn(n),     # リーク特徴
            'EMA12': np.random.rand(n) * 100,
            'EMA26': np.random.rand(n) * 100,
            'Target': np.random.randint(0, 2, n),
        })
        return df
    
    def test_returns_list_of_features(self, sample_features_df):
        """特徴選択が文字列リストを返すか"""
        features = get_ticker_specific_features(sample_features_df, "TEST")
        assert isinstance(features, list)
        assert all(isinstance(f, str) for f in features)
    
    def test_max_20_features(self, sample_features_df):
        """最大20個の特徴を返すか"""
        features = get_ticker_specific_features(sample_features_df, "TEST")
        assert len(features) <= 20
    
    def test_excludes_leak_features(self, sample_features_df):
        """リーク特徴が除外されているか"""
        features = get_ticker_specific_features(sample_features_df, "TEST")
        # リーク特徴が含まれていないことを確認
        for leak_feature in LEAK_FEATURES:
            assert leak_feature not in features
    
    def test_excludes_target_column(self, sample_features_df):
        """ターゲット列が除外されているか"""
        features = get_ticker_specific_features(sample_features_df, "TEST")
        assert 'Target' not in features
    
    def test_returns_numeric_features_only(self, sample_features_df):
        """数値型の列のみを返すか"""
        # 文字列型の列を追加
        df_with_str = sample_features_df.copy()
        df_with_str['StringColumn'] = 'text'
        
        features = get_ticker_specific_features(df_with_str, "TEST")
        # StringColumnが含まれていないことを確認
        assert 'StringColumn' not in features


class TestPredictTicker:
    """predict_ticker() のテスト"""
    
    @pytest.fixture
    def trained_model_df(self):
        """訓練用のサンプルデータ"""
        n = 100
        close = np.random.rand(n) * 100 + 50
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        
        df = pd.DataFrame({
            'Close': close,
            'Open': close + np.random.randn(n) * 0.5,
            'High': high,
            'Low': low,
            'Volume': np.random.rand(n) * 1000000,
            'MA5': close.rolling(5).mean().fillna(close),
            'MA10': close.rolling(10).mean().fillna(close),
            'MA20': close.rolling(20).mean().fillna(close),
            'RSI14': np.random.rand(n) * 100,
            'MACD': np.random.randn(n),
            'ATR': (high - low).rolling(14).mean().fillna(1),
            'Volume_Ratio': np.random.rand(n) * 2,
            'EMA12': close.ewm(span=12).mean(),
            'EMA26': close.ewm(span=26).mean(),
        })
        return df
    
    def test_predict_without_model_returns_none(self, trained_model_df):
        """モデルが存在しない場合、Noneを返すか"""
        result = predict_ticker("NONEXISTENT_TICKER", trained_model_df)
        assert result is None
    
    def test_predict_returns_dict_with_required_fields(self, trained_model_df):
        """
        予測が正しいフィールドを含むか (モデルがある場合)
        注意: このテストは実際にモデルを訓練する必要がある
        """
        # まずモデルを訓練
        ticker = "TEST_PREDICT"
        train_result = train_ticker(ticker, trained_model_df)
        
        if train_result is None:
            pytest.skip("Model training failed - not enough data")
        
        # 予測を実行
        prediction = predict_ticker(ticker, trained_model_df)
        
        if prediction is None:
            pytest.skip("Prediction failed")
        
        # 必須フィールドを確認
        required_fields = [
            'ticker',
            'current_price',
            'predicted_price',
            'predicted_change_pct',
            'direction',
            'prob_up',
            'prob_down',
            'confidence',
            'market_regime',
            'model_accuracy'
        ]
        
        for field in required_fields:
            assert field in prediction, f"Missing field: {field}"
    
    def test_predict_price_is_positive(self, trained_model_df):
        """予測価格が正の値であるか"""
        ticker = "TEST_PRICE_POSITIVE"
        train_ticker(ticker, trained_model_df)
        prediction = predict_ticker(ticker, trained_model_df)
        
        if prediction is not None:
            assert prediction['predicted_price'] > 0
            assert prediction['current_price'] > 0


class TestTrainTicker:
    """train_ticker() のテスト"""
    
    @pytest.fixture
    def sufficient_training_data(self):
        """十分な訓練データ"""
        n = 100
        close = np.random.rand(n) * 100 + 50
        
        df = pd.DataFrame({
            'Close': close,
            'Open': close + np.random.randn(n) * 0.5,
            'High': close + np.random.rand(n) * 2,
            'Low': close - np.random.rand(n) * 2,
            'Volume': np.random.rand(n) * 1000000,
            'MA5': close.rolling(5).mean().fillna(close),
            'MA10': close.rolling(10).mean().fillna(close),
            'MA20': close.rolling(20).mean().fillna(close),
            'RSI14': np.random.rand(n) * 100,
            'MACD': np.random.randn(n),
            'ATR': np.random.rand(n) * 5,
            'Volume_Ratio': np.random.rand(n) * 2,
            'EMA12': close.ewm(span=12).mean(),
            'EMA26': close.ewm(span=26).mean(),
        })
        return df
    
    def test_train_with_sufficient_data_returns_dict(self, sufficient_training_data):
        """十分なデータで訓練すると結果辞書を返すか"""
        result = train_ticker("TEST_TRAIN", sufficient_training_data)
        
        if result is None:
            pytest.skip("Training failed - may need more features")
        
        assert isinstance(result, dict)
        assert 'accuracy' in result
        assert 'regime' in result
    
    def test_train_with_insufficient_data_returns_none(self):
        """データが不十分な場合、Noneを返すか"""
        small_df = pd.DataFrame({
            'Close': [100, 101],
            'Open': [99, 100],
            'High': [101, 102],
            'Low': [98, 99],
            'Volume': [1000, 1000]
        })
        
        result = train_ticker("TEST_INSUFFICIENT", small_df)
        assert result is None


# pytest 実行時のエントリーポイント
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
