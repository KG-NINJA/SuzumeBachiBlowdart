"""
multitimescale.py - マルチタイムスケール予測モジュール

複数時間軸（短期1-3日、中期1-4週、長期1-3ヶ月）での同時予測を提供
XGBoost + LSTM ハイブリッドモデルで高精度予測を実現

Author: SuzumeBachiBlowdart Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
import warnings

# XGBoostとTensorFlow/Keras
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoostがインストールされていません。pip install xgboost")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    warnings.warn("TensorFlow/Kerasがインストールされていません。pip install tensorflow")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger("prediction_engine.multitimescale")


class MultiTimeScaleForecast:
    """
    複数時間軸での同時予測を生成するハイブリッドモデル
    
    予測期間:
    - 短期 (1-3日): 日中変動、終値、ボラティリティ
    - 中期 (1-4週): トレンド方向、サポート/レジスタンス
    - 長期 (1-3ヶ月): 大局トレンド、外部要因影響度
    """
    
    def __init__(self, use_lstm: bool = True):
        """
        初期化
        
        Args:
            use_lstm: LSTMを使用するか（Falseの場合XGBoostのみ）
        """
        self.use_lstm = use_lstm and HAS_KERAS
        
        # 各時間軸用モデル
        self.short_term_xgb = None  # 短期用XGBoost
        self.short_term_lstm = None  # 短期用LSTM
        self.medium_term_xgb = None  # 中期用XGBoost
        self.long_term_xgb = None  # 長期用XGBoost
        
        # スケーラー（LSTM用）
        self.scalers = {}
        
        # 学習済みフラグ
        self.is_trained = False
        
        logger.info(f"[INIT] MultiTimeScaleForecast 初期化完了 (LSTM: {self.use_lstm})")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        予測用特徴量を準備
        
        Args:
            df: OHLCV + テクニカル指標を含むDataFrame
        
        Returns:
            特徴量DataFrame
        """
        features = df.copy()
        
        # 基本テクニカル指標が存在することを確認
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in features.columns:
                raise ValueError(f"必須カラム '{col}' が不足しています")
        
        # 追加特徴量を計算
        close = features['Close']
        
        # リターン（各期間）
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}d'] = close.pct_change(period)
        
        # ボラティリティ（各期間）
        for period in [5, 10, 20]:
            features[f'volatility_{period}d'] = close.pct_change().rolling(period).std()
        
        # 移動平均との乖離率
        for period in [5, 10, 20, 50]:
            ma = close.rolling(period).mean()
            features[f'ma_{period}_deviation'] = (close - ma) / ma
        
        # 価格モメンタム
        features['momentum_3'] = close / close.shift(3) - 1
        features['momentum_10'] = close / close.shift(10) - 1
        
        # 出来高変化
        if 'Volume' in features.columns:
            features['volume_change'] = features['Volume'].pct_change()
            features['volume_ma_ratio'] = features['Volume'] / features['Volume'].rolling(20).mean()
        
        # NaN処理
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _create_targets(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        各時間軸のターゲット変数を作成
        
        Returns:
            (short_target, medium_target, long_target)
        """
        close = df['Close']
        
        # 短期ターゲット: 3日後のリターン方向 (1=上昇, 0=下落)
        short_target = (close.shift(-3) > close).astype(int)
        
        # 中期ターゲット: 2週間後のリターン方向
        medium_target = (close.shift(-10) > close).astype(int)
        
        # 長期ターゲット: 1ヶ月後のリターン方向
        long_target = (close.shift(-20) > close).astype(int)
        
        return short_target, medium_target, long_target
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        LSTM モデルを構築
        
        Args:
            input_shape: (timesteps, features)
        
        Returns:
            Kerasモデル
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_lstm_data(
        self, 
        features: np.ndarray, 
        target: np.ndarray, 
        lookback: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM用のシーケンスデータを準備
        
        Args:
            features: 特徴量配列
            target: ターゲット配列
            lookback: 過去何日分を使用するか
        
        Returns:
            (X_lstm, y_lstm)
        """
        X, y = [], []
        
        for i in range(lookback, len(features)):
            X.append(features[i - lookback:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def train(
        self, 
        df: pd.DataFrame, 
        ticker: str,
        epochs: int = 50,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        全時間軸のモデルを訓練
        
        Args:
            df: OHLCV + 特徴量DataFrame
            ticker: 銘柄コード
            epochs: LSTMのエポック数
            verbose: ログレベル
        
        Returns:
            訓練結果の辞書
        """
        logger.info(f"[TRAIN] {ticker} のマルチスケールモデル訓練開始")
        
        # 特徴量準備
        features_df = self._prepare_features(df)
        short_target, medium_target, long_target = self._create_targets(features_df)
        
        # ターゲットがあるデータのみ抽出
        valid_idx = ~(short_target.isna() | medium_target.isna() | long_target.isna())
        features_df = features_df[valid_idx].reset_index(drop=True)
        short_target = short_target[valid_idx].reset_index(drop=True)
        medium_target = medium_target[valid_idx].reset_index(drop=True)
        long_target = long_target[valid_idx].reset_index(drop=True)
        
        # 特徴量カラム選択（数値のみ）
        feature_cols = [c for c in features_df.columns 
                       if features_df[c].dtype in ['float64', 'float32', 'int64', 'int32']
                       and c not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Date']]
        
        X = features_df[feature_cols].values
        
        # データ分割
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        results = {
            'ticker': ticker,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracies': {}
        }
        
        # ========== 短期モデル訓練 (XGBoost + LSTM) ==========
        logger.info(f"[TRAIN] 短期モデル訓練中...")
        
        y_short_train = short_target[:split_idx].values
        y_short_test = short_target[split_idx:].values
        
        # XGBoost
        self.short_term_xgb = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        self.short_term_xgb.fit(X_train, y_short_train)
        
        xgb_acc = self.short_term_xgb.score(X_test, y_short_test)
        results['accuracies']['short_term_xgb'] = float(xgb_acc)
        
        # LSTM（オプション）
        if self.use_lstm and len(X_train) > 30:
            # スケーリング
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['short_term'] = scaler
            
            # シーケンスデータ作成
            lookback = 20
            X_lstm, y_lstm = self._prepare_lstm_data(
                X_scaled, short_target.values, lookback
            )
            
            if len(X_lstm) > 50:
                split_lstm = int(len(X_lstm) * 0.8)
                X_lstm_train = X_lstm[:split_lstm]
                X_lstm_test = X_lstm[split_lstm:]
                y_lstm_train = y_lstm[:split_lstm]
                y_lstm_test = y_lstm[split_lstm:]
                
                self.short_term_lstm = self._build_lstm_model(
                    (X_lstm_train.shape[1], X_lstm_train.shape[2])
                )
                
                self.short_term_lstm.fit(
                    X_lstm_train, y_lstm_train,
                    epochs=epochs,
                    batch_size=16,
                    validation_split=0.2,
                    verbose=verbose
                )
                
                lstm_loss, lstm_acc = self.short_term_lstm.evaluate(
                    X_lstm_test, y_lstm_test, verbose=0
                )
                results['accuracies']['short_term_lstm'] = float(lstm_acc)
        
        # ========== 中期モデル訓練 (XGBoost) ==========
        logger.info(f"[TRAIN] 中期モデル訓練中...")
        
        y_medium_train = medium_target[:split_idx].values
        y_medium_test = medium_target[split_idx:].values
        
        self.medium_term_xgb = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        self.medium_term_xgb.fit(X_train, y_medium_train)
        
        medium_acc = self.medium_term_xgb.score(X_test, y_medium_test)
        results['accuracies']['medium_term'] = float(medium_acc)
        
        # ========== 長期モデル訓練 (XGBoost) ==========
        logger.info(f"[TRAIN] 長期モデル訓練中...")
        
        y_long_train = long_target[:split_idx].values
        y_long_test = long_target[split_idx:].values
        
        self.long_term_xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.6,
            colsample_bytree=0.6,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        self.long_term_xgb.fit(X_train, y_long_train)
        
        long_acc = self.long_term_xgb.score(X_test, y_long_test)
        results['accuracies']['long_term'] = float(long_acc)
        
        # 特徴量カラム保存
        self.feature_cols = feature_cols
        self.is_trained = True
        
        logger.info(f"[TRAIN] 訓練完了: {results['accuracies']}")
        
        return results
    
    def predict(
        self, 
        ticker: str, 
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        マルチタイムスケール予測を生成
        
        Args:
            ticker: 銘柄コード
            df: 最新のOHLCV + 特徴量DataFrame
            current_price: 現在価格（Noneの場合はdfから取得）
        
        Returns:
            予測結果の辞書
        """
        if not self.is_trained:
            logger.warning(f"[PREDICT] モデルが未訓練です")
            return self._generate_fallback_prediction(ticker, df, current_price)
        
        # 特徴量準備
        features_df = self._prepare_features(df)
        
        # 現在価格取得
        if current_price is None:
            current_price = float(df['Close'].iloc[-1])
        
        # 特徴量抽出
        X = features_df[self.feature_cols].iloc[-1:].values
        
        # ========== 短期予測 ==========
        short_proba_xgb = self.short_term_xgb.predict_proba(X)[0]
        
        # LSTMがあれば統合
        if self.short_term_lstm is not None and 'short_term' in self.scalers:
            lookback = 20
            if len(features_df) >= lookback:
                X_scaled = self.scalers['short_term'].transform(
                    features_df[self.feature_cols].values
                )
                X_lstm = X_scaled[-lookback:].reshape(1, lookback, -1)
                lstm_proba = float(self.short_term_lstm.predict(X_lstm, verbose=0)[0, 0])
                
                # アンサンブル（XGBoost 60% + LSTM 40%）
                short_up_prob = 0.6 * short_proba_xgb[1] + 0.4 * lstm_proba
            else:
                short_up_prob = short_proba_xgb[1]
        else:
            short_up_prob = short_proba_xgb[1]
        
        # 中期予測
        medium_proba = self.medium_term_xgb.predict_proba(X)[0]
        medium_up_prob = medium_proba[1]
        
        # 長期予測
        long_proba = self.long_term_xgb.predict_proba(X)[0]
        long_up_prob = long_proba[1]
        
        # ボラティリティ計算
        volatility = float(features_df['volatility_10d'].iloc[-1]) if 'volatility_10d' in features_df else 0.02
        
        # 予測価格計算
        short_change = (short_up_prob - 0.5) * 0.04  # 最大±2%
        medium_change = (medium_up_prob - 0.5) * 0.08  # 最大±4%
        long_change = (long_up_prob - 0.5) * 0.15  # 最大±7.5%
        
        # サポート/レジスタンス推定
        recent_high = float(df['High'].tail(20).max())
        recent_low = float(df['Low'].tail(20).min())
        
        result = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "predictions": {
                "short_term": {
                    "horizon": "1-3 days",
                    "forecast_1d": round(current_price * (1 + short_change * 0.3), 2),
                    "forecast_3d": round(current_price * (1 + short_change), 2),
                    "direction": "UP" if short_up_prob > 0.5 else "DOWN",
                    "probability": round(float(short_up_prob), 4),
                    "volatility": round(volatility * 100, 2),  # パーセント表示
                    "confidence": round(abs(short_up_prob - 0.5) * 2, 4)
                },
                "medium_term": {
                    "horizon": "1-4 weeks",
                    "forecast_1w": round(current_price * (1 + medium_change * 0.25), 2),
                    "forecast_4w": round(current_price * (1 + medium_change), 2),
                    "direction": "UP" if medium_up_prob > 0.5 else "DOWN",
                    "probability": round(float(medium_up_prob), 4),
                    "resistance": round(recent_high, 2),
                    "support": round(recent_low, 2),
                    "confidence": round(abs(medium_up_prob - 0.5) * 2, 4)
                },
                "long_term": {
                    "horizon": "1-3 months",
                    "forecast_1m": round(current_price * (1 + long_change * 0.33), 2),
                    "forecast_3m": round(current_price * (1 + long_change), 2),
                    "direction": "UP" if long_up_prob > 0.5 else "DOWN",
                    "probability": round(float(long_up_prob), 4),
                    "trend_strength": round(abs(long_up_prob - 0.5) * 2, 4),
                    "confidence": round(abs(long_up_prob - 0.5) * 2 * 0.85, 4)  # 長期は信頼度を下げる
                }
            },
            "model_info": {
                "uses_lstm": self.short_term_lstm is not None,
                "ensemble_weights": {"xgboost": 0.6, "lstm": 0.4} if self.short_term_lstm else {"xgboost": 1.0}
            }
        }
        
        logger.info(f"[PREDICT] {ticker} 予測完了")
        
        return result
    
    def _generate_fallback_prediction(
        self, 
        ticker: str, 
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        モデル未訓練時のフォールバック予測
        
        シンプルな統計的手法で予測を生成
        """
        if current_price is None:
            current_price = float(df['Close'].iloc[-1])
        
        # 直近のトレンドから予測
        recent_returns = df['Close'].pct_change().tail(10)
        mean_return = float(recent_returns.mean())
        volatility = float(recent_returns.std())
        
        # トレンド方向
        trend_up = mean_return > 0
        trend_prob = 0.5 + min(abs(mean_return) / (volatility + 0.001), 0.2)
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "predictions": {
                "short_term": {
                    "horizon": "1-3 days",
                    "forecast_1d": round(current_price * (1 + mean_return), 2),
                    "forecast_3d": round(current_price * (1 + mean_return * 3), 2),
                    "direction": "UP" if trend_up else "DOWN",
                    "probability": round(trend_prob if trend_up else 1 - trend_prob, 4),
                    "volatility": round(volatility * 100, 2),
                    "confidence": 0.3  # 低信頼度
                },
                "medium_term": {
                    "horizon": "1-4 weeks",
                    "forecast_1w": round(current_price * (1 + mean_return * 5), 2),
                    "forecast_4w": round(current_price * (1 + mean_return * 20), 2),
                    "direction": "UP" if trend_up else "DOWN",
                    "probability": 0.5,
                    "confidence": 0.2
                },
                "long_term": {
                    "horizon": "1-3 months",
                    "forecast_1m": round(current_price * (1 + mean_return * 20), 2),
                    "forecast_3m": round(current_price * (1 + mean_return * 60), 2),
                    "direction": "NEUTRAL",
                    "probability": 0.5,
                    "confidence": 0.1
                }
            },
            "model_info": {
                "uses_lstm": False,
                "fallback": True,
                "reason": "モデル未訓練のため統計的手法で予測"
            }
        }


# ========== テスト用コード ==========
if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== MultiTimeScaleForecast テスト ===\n")
    
    # サンプルデータ取得
    ticker = "NVDA"
    print(f"[TEST] {ticker} のデータを取得中...")
    data = yf.download(ticker, period="6mo", progress=False)
    
    if len(data) < 50:
        print("[ERROR] データが不足しています")
        exit(1)
    
    print(f"[TEST] データ取得完了: {len(data)} 行\n")
    
    # モデル作成と訓練
    mts = MultiTimeScaleForecast(use_lstm=HAS_KERAS)
    
    print("[TEST] モデル訓練中...")
    train_result = mts.train(data, ticker, epochs=30, verbose=0)
    
    print(f"\n[RESULT] 訓練結果:")
    print(f"  - サンプル数: {train_result['training_samples']}")
    for model_name, acc in train_result['accuracies'].items():
        print(f"  - {model_name}: {acc:.4f}")
    
    # 予測実行
    print("\n[TEST] 予測実行中...")
    prediction = mts.predict(ticker, data)
    
    print(f"\n[RESULT] 予測結果:")
    print(f"  現在価格: ${prediction['current_price']:.2f}")
    
    for term, pred in prediction['predictions'].items():
        print(f"\n  【{term}】")
        print(f"    方向: {pred['direction']} (確率: {pred.get('probability', 'N/A')})")
        print(f"    信頼度: {pred['confidence']:.2%}")
        if 'forecast_3d' in pred:
            print(f"    3日後予測: ${pred['forecast_3d']:.2f}")
        if 'forecast_4w' in pred:
            print(f"    4週後予測: ${pred['forecast_4w']:.2f}")
        if 'forecast_3m' in pred:
            print(f"    3ヶ月後予測: ${pred['forecast_3m']:.2f}")
