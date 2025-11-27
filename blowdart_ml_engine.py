"""
blowdart_ml_engine.py - Market Regime Detection & Hybrid Model Selection
KG-NINJA × Claude 最終形態：メタモデルで市場レジームを自動検知
シンプル（安定）モデル × 過激（攻撃）モデルを動的に切り替え
既存パイプラインとの完全互換性を保持
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIG =====
MODELS_ROOT = Path("models")
TRAINING_LOG_DIR = Path("analytics")
EXPERIMENT_LOG = Path("experiments")
REGIME_LOG = Path("regime_detection")

for p in [MODELS_ROOT, TRAINING_LOG_DIR, EXPERIMENT_LOG, REGIME_LOG]:
    p.mkdir(parents=True, exist_ok=True)

# ===== HYBRID CONFIGURATION =====
USE_HYBRID_MODE = True
REGIME_DETECTION = True


def detect_market_regime(features_df, ticker, lookback=20):
    """
    市場レジーム検知：ボラティリティ、トレンド、モメンタムから現在の市場状態を判定
    
    Returns:
        regime: 'TRENDING', 'MEAN_REVERSION', 'CHOPPY'
        volatility: ボラティリティスコア (0-1)
        trend_strength: トレンド強度 (0-1)
    """
    
    if len(features_df) < lookback:
        return 'NEUTRAL', 0.5, 0.5
    
    recent = features_df.iloc[-lookback:].copy()
    
    # ===== 指標1: ボラティリティ =====
    returns = recent['Close'].pct_change().dropna()
    volatility = returns.std()
    vol_percentile = (volatility - returns.std().quantile(0.25)) / (returns.std().quantile(0.75) - returns.std().quantile(0.25) + 1e-6)
    vol_percentile = np.clip(vol_percentile, 0, 1)
    
    # ===== 指標2: トレンド強度 =====
    high_low = recent['High'] - recent['Low']
    high_close = np.abs(recent['High'] - recent['Close'].shift())
    low_close = np.abs(recent['Low'] - recent['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.mean()
    
    close_diff = recent['Close'].diff()
    up_count = (close_diff > 0).sum()
    down_count = (close_diff < 0).sum()
    
    if up_count + down_count > 0:
        trend_direction = (up_count - down_count) / (up_count + down_count)
        trend_strength = abs(trend_direction)
    else:
        trend_strength = 0.5
    
    # ===== レジーム判定 =====
    if trend_strength > 0.6 and vol_percentile < 0.6:
        regime = 'TRENDING'
    elif vol_percentile > 0.7:
        regime = 'CHOPPY'
    else:
        regime = 'MEAN_REVERSION'
    
    # ログ記録
    regime_record = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "volatility": float(vol_percentile),
        "trend_strength": float(trend_strength),
        "atr": float(atr)
    }
    
    regime_file = REGIME_LOG / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(regime_file, 'w') as f:
        json.dump(regime_record, f, indent=2)
    
    print(f"  [REGIME] {ticker}: {regime} | Vol={vol_percentile:.2f} | Trend={trend_strength:.2f}")
    
    return regime, vol_percentile, trend_strength


def get_regime_weights(regime, volatility):
    """
    市場レジームに基づいてモデルの重みを決定
    """
    if regime == 'TRENDING' and volatility < 0.5:
        simple_weight = 0.7
        aggressive_weight = 0.3
        description = "STABLE_TREND"
    elif regime == 'CHOPPY' and volatility > 0.7:
        simple_weight = 0.2
        aggressive_weight = 0.8
        description = "VOLATILE_CHOPPY"
    elif regime == 'MEAN_REVERSION':
        simple_weight = 0.5
        aggressive_weight = 0.5
        description = "BALANCED_MEANREV"
    else:
        simple_weight = 0.5
        aggressive_weight = 0.5
        description = "NEUTRAL"
    
    return simple_weight, aggressive_weight, description


def get_ticker_dir(ticker):
    ticker_dir = MODELS_ROOT / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir


def _train_simple(ticker, features_df):
    """内部：シンプルモデル訓練"""
    try:
        if features_df is None or features_df.empty or len(features_df) < 30:
            return None
        
        df = features_df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'Close' not in df.columns:
            return None
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        if len(df) < 30:
            return None
        
        feature_cols = [c for c in numeric_cols if c != 'Close']
        if 'Close' in feature_cols:
            feature_cols.remove('Close')
        
        X = df[feature_cols]
        y = df['Target']
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
        
        params = {
            'objective': 'binary:logistic',
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss'
        }
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        predictions = model.predict(dtest)
        accuracy = np.mean((predictions > 0.5).astype(int) == y_test)
        
        return {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "accuracy": accuracy,
            "model_type": "SIMPLE"
        }
    except Exception as e:
        print(f"  [SIMPLE ERROR] {ticker}: {e}")
        return None


def _train_aggressive(ticker, features_df):
    """内部：アグレッシブモデル訓練"""
    try:
        if features_df is None or features_df.empty or len(features_df) < 40:
            return None
        
        df = features_df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'Close' not in df.columns:
            return None
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        if len(df) < 40:
            return None
        
        feature_cols = [c for c in numeric_cols if c != 'Close']
        if 'Close' in feature_cols:
            feature_cols.remove('Close')
        
        X = df[feature_cols]
        y = df['Target']
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        models_ensemble = []
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_params = {
                'objective': 'binary:logistic',
                'max_depth': 4 + fold % 2,
                'learning_rate': 0.08 * (0.9 + fold * 0.02),
                'subsample': 0.7 + fold * 0.04,
                'colsample_bytree': 0.75 + fold * 0.03,
                'gamma': 0.5 + fold * 0.1,
                'eval_metric': 'logloss'
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
            
            model = xgb.train(
                params=fold_params,
                dtrain=dtrain,
                num_boost_round=200,
                evals=[(dval, 'eval')],
                early_stopping_rounds=15,
                verbose_eval=False
            )
            
            models_ensemble.append(model)
            val_pred = model.predict(dval)
            val_acc = np.mean((val_pred > 0.5).astype(int) == y_val)
            cv_scores.append(val_acc)
        
        ensemble_accuracy = np.mean(cv_scores)
        
        return {
            "models": models_ensemble,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "accuracy": ensemble_accuracy,
            "model_type": "AGGRESSIVE_ENSEMBLE"
        }
    except Exception as e:
        print(f"  [AGGRESSIVE ERROR] {ticker}: {e}")
        return None


def train_ticker(ticker, features_df, use_online_learning=True, use_feature_reduction=True):
    """
    === メイン訓練関数（互換性維持）===
    市場レジーム検知により、シンプル＆アグレッシブモデルを最適に統合
    
    Args:
        ticker: ティッカーシンボル
        features_df: 特徴量データフレーム
        use_online_learning: オンライン学習フラグ（互換性用、現在はハイブリッドに統合）
        use_feature_reduction: 特徴削減フラグ（互換性用）
    
    Returns:
        メトリクス辞書
    """
    try:
        print(f"\n  [HYBRID] Starting hybrid training for {ticker}")
        
        if features_df is None or features_df.empty:
            return None
        
        # ===== Step 1: 市場レジーム検知 =====
        regime, volatility, trend_strength = detect_market_regime(features_df, ticker)
        simple_weight, aggressive_weight, regime_desc = get_regime_weights(regime, volatility)
        
        print(f"  [HYBRID] Model weights: Simple={simple_weight:.1%} | Aggressive={aggressive_weight:.1%}")
        
        # ===== Step 2: 両モデルを訓練 =====
        result_simple = _train_simple(ticker, features_df)
        result_aggressive = _train_aggressive(ticker, features_df)
        
        if result_simple is None or result_aggressive is None:
            print(f"  [HYBRID] One or both models failed to train")
            return None
        
        simple_acc = result_simple['accuracy']
        aggressive_acc = result_aggressive['accuracy']
        
        print(f"  [SIMPLE] Accuracy: {simple_acc:.4f}")
        print(f"  [AGGRESSIVE] Accuracy: {aggressive_acc:.4f}")
        
        # ===== Step 3: ハイブリッド精度計算 =====
        hybrid_accuracy = simple_acc * simple_weight + aggressive_acc * aggressive_weight
        
        # ===== Step 4: モデル保存 =====
        ticker_dir = get_ticker_dir(ticker)
        
        # シンプルモデル
        result_simple['model'].save_model(str(ticker_dir / "model_simple.json"))
        with open(ticker_dir / "scaler_simple.pkl", 'wb') as f:
            pickle.dump(result_simple['scaler'], f)
        
        # アグレッシブモデル
        for i, model in enumerate(result_aggressive['models']):
            model.save_model(str(ticker_dir / f"model_aggressive_{i}.json"))
        with open(ticker_dir / "scaler_aggressive.pkl", 'wb') as f:
            pickle.dump(result_aggressive['scaler'], f)
        
        # メタデータ
        metadata = {
            "ticker": ticker,
            "hybrid_mode": True,
            "regime_detection": {
                "regime": regime,
                "volatility": float(volatility),
                "trend_strength": float(trend_strength)
            },
            "model_weights": {
                "simple": float(simple_weight),
                "aggressive": float(aggressive_weight),
                "regime_description": regime_desc
            },
            "accuracies": {
                "simple": float(simple_acc),
                "aggressive": float(aggressive_acc),
                "hybrid": float(hybrid_accuracy)
            },
            "feature_names": result_simple['feature_cols'],
            "saved_at": datetime.now().isoformat()
        }
        
        with open(ticker_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  [HYBRID] Blended Accuracy: {hybrid_accuracy:.4f} ({regime_desc})")
        print(f"  [RESULT] {ticker} | Acc: {hybrid_accuracy:.4f} | {regime_desc}")
        
        return {
            "accuracy": hybrid_accuracy,
            "regime": regime,
            "simple_weight": simple_weight,
            "aggressive_weight": aggressive_weight,
            "simple_accuracy": simple_acc,
            "aggressive_accuracy": aggressive_acc,
            "learning_type": "HYBRID_MODE"
        }
    
    except Exception as e:
        print(f"  [TRAIN ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_ticker(ticker, features_df):
    """
    === メイン予測関数（互換性維持）===
    ハイブリッド予測：市場レジームに応じて最適な予測を返す
    """
    try:
        if features_df is None or features_df.empty:
            return None
        
        ticker_dir = get_ticker_dir(ticker)
        metadata_path = ticker_dir / "metadata.json"
        
        if not metadata_path.exists():
            print(f"  [PREDICT] Model not found for {ticker}")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != 'Close']
        
        # 最新データ準備
        latest_row = features_df.iloc[-1]
        current_close = latest_row.get('Close')
        X_latest = features_df[feature_cols].iloc[-1:].copy()
        X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # ===== シンプルモデル予測 =====
        scaler_simple = pickle.load(open(ticker_dir / "scaler_simple.pkl", 'rb'))
        booster_simple = xgb.Booster()
        booster_simple.load_model(str(ticker_dir / "model_simple.json"))
        
        X_simple = scaler_simple.transform(X_latest)
        dmatrix_simple = xgb.DMatrix(X_simple, feature_names=feature_cols)
        pred_simple = booster_simple.predict(dmatrix_simple)[0]
        
        # ===== アグレッシブモデル予測（アンサンブル） =====
        scaler_aggressive = pickle.load(open(ticker_dir / "scaler_aggressive.pkl", 'rb'))
        X_aggressive = scaler_aggressive.transform(X_latest)
        
        predictions_aggressive = []
        for i in range(5):
            booster_agg = xgb.Booster()
            booster_agg.load_model(str(ticker_dir / f"model_aggressive_{i}.json"))
            dmatrix_agg = xgb.DMatrix(X_aggressive, feature_names=feature_cols)
            pred_agg = booster_agg.predict(dmatrix_agg)[0]
            predictions_aggressive.append(pred_agg)
        
        pred_aggressive = np.mean(predictions_aggressive)
        
        # ===== ハイブリッド予測 =====
        model_weights = metadata['model_weights']
        simple_weight = model_weights['simple']
        aggressive_weight = model_weights['aggressive']
        
        pred_hybrid = pred_simple * simple_weight + pred_aggressive * aggressive_weight
        
        # 結果
        direction = "↑ Bullish" if pred_hybrid > 0.5 else "↓ Bearish"
        confidence = abs(pred_hybrid - 0.5) * 2
        
        regime = metadata['regime_detection']['regime']
        if regime == 'TRENDING':
            signal_boost = 1.2
        elif regime == 'CHOPPY':
            signal_boost = 0.8
        else:
            signal_boost = 1.0
        
        adjusted_confidence = np.clip(confidence * signal_boost, 0, 1)
        price_change = adjusted_confidence * 0.04 * (1 if pred_hybrid > 0.5 else -1)
        predicted_price = current_close * (1 + price_change)
        
        return {
            "ticker": ticker,
            "current_price": float(current_close),
            "predicted_price": float(predicted_price),
            "direction": direction,
            "confidence": float(adjusted_confidence),
            "market_regime": regime,
            "simple_pred": float(pred_simple),
            "aggressive_pred": float(pred_aggressive),
            "hybrid_pred": float(pred_hybrid),
            "simple_weight": float(simple_weight),
            "aggressive_weight": float(aggressive_weight),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        print(f"  [PREDICT ERROR] {ticker}: {e}")
        return None
