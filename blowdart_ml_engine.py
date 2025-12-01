"""
blowdart_ml_engine.py - 修正版（日本語コメント付き）
予測値が正しく出力されるように完全修正
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pickle
from datetime import datetime
import warnings
from typing import Optional, Dict, List, Tuple, Any
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
from confidence_filter import calculate_confidence_score
from config.hyperparameters import (
    XGBoostConfig,
    MarketRegimeConfig,
    FeatureSelectionConfig,
    TrainingConfig,
    PredictionConfig,
    PathConfig
)

# パス設定（config から取得）
MODELS_ROOT = Path(PathConfig.MODELS_ROOT)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
REGIME_LOG = Path(PathConfig.REGIME_LOG)
REGIME_LOG.mkdir(parents=True, exist_ok=True)

# 特徴選択の設定（config から取得）
LEAK_FEATURES = FeatureSelectionConfig.LEAK_FEATURES
KEEP_FEATURES = FeatureSelectionConfig.KEEP_FEATURES


def detect_market_regime_fixed(df: pd.DataFrame, ticker: str, lookback: int = MarketRegimeConfig.LOOKBACK_PERIOD) -> Tuple[str, float, float]:
    """市場レジーム判定（改善版）"""
    
    if len(df) < lookback:
        return 'NEUTRAL', 0.5, 0.5
    
    recent = df.iloc[-lookback:].copy()
    returns = recent['Close'].pct_change().dropna()
    
    if len(returns) == 0:
        return 'NEUTRAL', 0.5, 0.5
    
    volatility_raw = returns.std()
    
    # 過去全期間での正規化
    all_returns = df['Close'].pct_change().dropna()
    if len(all_returns) > 50:
        try:
            q25 = float(all_returns.quantile(0.25))
            q75 = float(all_returns.quantile(0.75))
            vol_percentile = np.clip((volatility_raw - q25) / (q75 - q25 + 1e-8), 0, 1)
        except Exception:
            vol_percentile = 0.5
    else:
        vol_percentile = 0.5
    
    # トレンド強度計算
    close_diff = recent['Close'].diff().fillna(0)
    up_days = (close_diff > 0).sum()
    down_days = (close_diff < 0).sum()
    total = up_days + down_days + 1e-6
    trend_strength = abs(up_days - down_days) / total
    
    # レジーム判定（config の閾値を使用）
    if trend_strength > MarketRegimeConfig.TREND_STRENGTH_THRESHOLD and vol_percentile < MarketRegimeConfig.VOLATILITY_THRESHOLD_LOW:
        regime = 'TRENDING'
    elif vol_percentile > MarketRegimeConfig.VOLATILITY_THRESHOLD_HIGH:
        regime = 'CHOPPY'
    else:
        regime = 'MEAN_REVERSION'
    
    logger.info(f"  [REGIME] {ticker}: {regime:15s} | Vol={vol_percentile:.2f} | Trend={trend_strength:.2f}")
    
    return regime, float(vol_percentile), float(trend_strength)


def get_ticker_dir(ticker: str) -> Path:
    """ティッカー専用のモデルディレクトリを取得"""
    ticker_dir = MODELS_ROOT / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir


def load_existing_model(ticker: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    既存モデルをロード
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        (model_booster, scaler) または (None, None) if failed
    """
    try:
        ticker_dir = get_ticker_dir(ticker)
        model_path = ticker_dir / "model_simple.json"
        scaler_path = ticker_dir / "scaler_simple.pkl"
        
        # ファイル存在チェック
        if not model_path.exists():
            logger.debug(f"{ticker}: Model file not found at {model_path}")
            return None, None
        
        if not scaler_path.exists():
            logger.debug(f"{ticker}: Scaler file not found at {scaler_path}")
            return None, None
        
        # モデルロード
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            logger.debug(f"{ticker}: Model loaded successfully from {model_path}")
        except (FileNotFoundError, OSError) as e:
            logger.error(f"{ticker}: Model file access error: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"{ticker}: Failed to load model: {type(e).__name__}: {str(e)}")
            return None, None
        
        # スケーラーロード
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.debug(f"{ticker}: Scaler loaded successfully from {scaler_path}")
        except pickle.UnpicklingError as e:
            logger.error(f"{ticker}: Scaler file corrupted (UnpicklingError): {str(e)}")
            return None, None
        except (FileNotFoundError, OSError) as e:
            logger.error(f"{ticker}: Scaler file access error: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"{ticker}: Unexpected error loading scaler: {type(e).__name__}: {str(e)}")
            return None, None
        
        return model.get_booster(), scaler
        
    except Exception as e:
        logger.error(f"{ticker}: Failed to load existing model: {type(e).__name__}: {str(e)}", exc_info=True)
        return None, None


def load_model_info(ticker: str) -> Optional[Dict[str, Any]]:
    """
    モデル情報（メタデータ）を読み込み
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        メタデータ辞書 または None if failed
    """
    try:
        ticker_dir = get_ticker_dir(ticker)
        metadata_path = ticker_dir / "metadata.json"
        
        if not metadata_path.exists():
            logger.debug(f"{ticker}: Metadata file not found at {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.debug(f"{ticker}: Metadata loaded successfully")
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"{ticker}: Metadata file corrupted (JSONDecodeError): {str(e)}")
            return None
        except (FileNotFoundError, OSError) as e:
            logger.error(f"{ticker}: Metadata file access error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"{ticker}: Unexpected error loading metadata: {type(e).__name__}: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"{ticker}: Failed to load model info: {type(e).__name__}: {str(e)}", exc_info=True)
        return None


def get_ticker_specific_features(df: pd.DataFrame, ticker: str) -> List[str]:
    """ティッカー固有の特徴選択"""
    
    # 利用可能な全特徴を取得（ターゲットとリーク特徴を除外）
    all_cols = set(df.columns) - {'Target', 'Close_shift'} - LEAK_FEATURES
    
    # 数値列のみ抽出
    numeric_cols = []
    for col in all_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            numeric_cols.append(col)
    
    # KEEP_FEATURESとの交差を優先
    keep_intersection = [f for f in numeric_cols if f in KEEP_FEATURES]
    other = [f for f in numeric_cols if f not in KEEP_FEATURES]
    
    # 最大特徴数（config から取得）
    final_features = (keep_intersection + other)[:FeatureSelectionConfig.MAX_FEATURES]
    
    if len(final_features) < FeatureSelectionConfig.MIN_FEATURES:
        logger.warning(f"{ticker}: Only {len(final_features)} features available")
    
    logger.info(f"[FEATURES] {ticker}: {len(final_features)} selected")
    
    return final_features


def train_ticker(ticker: str, features_df: pd.DataFrame, use_online_learning: bool = True, use_feature_reduction: bool = True) -> Optional[Dict[str, Any]]:
    """ティッカーのモデル訓練（改善版）"""
    
    try:
        logger.info(f"[TRAIN] {ticker} - 改善版モード起動")
        
        if features_df is None or features_df.empty or len(features_df) < TrainingConfig.MIN_TRAINING_SAMPLES:
            logger.error(f"Insufficient data for {ticker}")
            return None
        
        df = features_df.copy()
        
        # ターゲット生成（翌日の価格上昇を予測）
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        if len(df) < TrainingConfig.MIN_TRAINING_SAMPLES:
            return None
        
        # 特徴選択
        feature_cols = get_ticker_specific_features(df, ticker)
        
        if not feature_cols:
            logger.error(f"No valid features for {ticker}")
            return None
        
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df['Target']
        
        # レジーム検知
        regime, vol, trend = detect_market_regime_fixed(df, ticker)
        
        # 時系列分割（80% train, 20% test）
        split_idx = int(len(X) * TrainingConfig.TRAIN_TEST_SPLIT_RATIO)
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")
        
        # オンライン学習ロジック
        existing_model = None
        existing_scaler = None
        previous_accuracy = 0
        
        if use_online_learning:
            existing_model, existing_scaler = load_existing_model(ticker)
            model_info = load_model_info(ticker)
            
            if model_info:
                previous_accuracy = model_info.get('accuracies', {}).get('hybrid', 0)
        
        # レジームに応じた重み付け（config から取得）
        if regime == 'TRENDING':
            w_simple, w_agg = MarketRegimeConfig.TRENDING_WEIGHTS
            desc = "TREND-FOCUSED"
        elif regime == 'CHOPPY':
            w_simple, w_agg = MarketRegimeConfig.CHOPPY_WEIGHTS
            desc = "CONSERVATIVE"
        else:
            w_simple, w_agg = MarketRegimeConfig.MEAN_REVERSION_WEIGHTS
            desc = "BALANCED"
        
        logger.info(f"[WEIGHTS] Simple={w_simple:.0%} | Aggressive={w_agg:.0%} ({desc})")
        
        # シンプルモデル（config から取得）
        scaler_s = StandardScaler()
        X_train_s = scaler_s.fit_transform(X_train)
        X_test_s = scaler_s.transform(X_test)
        
        model_s = xgb.XGBClassifier(**XGBoostConfig.get_simple_params())
        model_s.fit(X_train_s, y_train)
        acc_s = model_s.score(X_test_s, y_test)
        
        # アグレッシブアンサンブル
        scaler_a = RobustScaler()
        X_train_a = scaler_a.fit_transform(X_train)
        X_test_a = scaler_a.transform(X_test)
        
        models_a = []
        accs_a = []
        
        for fold in range(XGBoostConfig.AGGRESSIVE_N_FOLDS):
            # アグレッシブモデル（config から取得）
            m = xgb.XGBClassifier(**XGBoostConfig.get_aggressive_params(fold))
            m.fit(X_train_a, y_train)
            models_a.append(m)
            accs_a.append(m.score(X_test_a, y_test))
        
        acc_a = np.mean(accs_a)
        
        # ハイブリッド精度
        hybrid_acc = acc_s * w_simple + acc_a * w_agg
        
        # 精度向上率
        accuracy_improvement = hybrid_acc - previous_accuracy
        
        # モデル保存
        ticker_dir = get_ticker_dir(ticker)
        
        model_s.save_model(str(ticker_dir / "model_simple.json"))
        with open(ticker_dir / "scaler_simple.pkl", 'wb') as f:
            pickle.dump(scaler_s, f)
        
        with open(ticker_dir / "scaler_agg.pkl", 'wb') as f:
            pickle.dump(scaler_a, f)
        
        for fold, m in enumerate(models_a):
            m.save_model(str(ticker_dir / f"model_agg_{fold}.json"))
        
        # メタデータ保存
        metadata = {
            "ticker": ticker,
            "regime": regime,
            "regime_vol": float(vol),
            "trend_strength": float(trend),
            "weights": {"simple": w_simple, "aggressive": w_agg},
            "accuracies": {
                "simple": float(acc_s),
                "aggressive": float(acc_a),
                "hybrid": float(hybrid_acc)
            },
            "feature_names": feature_cols,
            "feature_count": len(feature_cols),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "version": "2025-11-30-fixed",
            "saved_at": datetime.now().isoformat()
        }
        
        with open(ticker_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[RESULT] {ticker} | Hybrid Acc: {hybrid_acc:.4f} | Improvement: {accuracy_improvement:+.4f}")
        
        return {
            "accuracy": float(hybrid_acc),
            "previous_accuracy": previous_accuracy,
            "accuracy_improvement": accuracy_improvement,
            "train_samples": len(X_train),
            "regime": regime,
            "simple_weight": w_simple,
            "aggressive_weight": w_agg,
            "learning_type": "HYBRID_FIXED"
        }
    
    except Exception as e:
        logger.error(f"{ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_ticker(ticker: str, features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """ティッカーの予測（改善版 - 予測価格を正しく計算）"""
    
    try:
        if features_df is None or features_df.empty:
            return None
        
        ticker_dir = get_ticker_dir(ticker)
        metadata_path = ticker_dir / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"[PREDICT] Model not found for {ticker}")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_cols = metadata['feature_names']
        regime = metadata.get('regime', 'UNKNOWN')
        hybrid_accuracy = metadata['accuracies']['hybrid']
        
        # 最新データ準備
        latest_row = features_df.iloc[-1]
        current_close = float(latest_row.get('Close', 0))
        
        if current_close <= 0:
            logger.error(f"Invalid current price for {ticker}")
            return None
        
        X_latest = features_df[feature_cols].iloc[-1:].copy()
        X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # シンプルモデル予測
        scaler_s = pickle.load(open(ticker_dir / "scaler_simple.pkl", "rb"))
        model_s = xgb.XGBClassifier()
        model_s.load_model(str(ticker_dir / "model_simple.json"))
        
        X_s = scaler_s.transform(X_latest)
        pred_s = float(model_s.predict_proba(X_s)[0][1])
        
        # アグレッシブアンサンブル予測
        scaler_a = pickle.load(open(ticker_dir / "scaler_agg.pkl", "rb"))
        X_a = scaler_a.transform(X_latest)
        
        pred_a_list = []
        for fold in range(5):
            m = xgb.XGBClassifier()
            m.load_model(str(ticker_dir / f"model_agg_{fold}.json"))
            pred = float(m.predict_proba(X_a)[0][1])
            pred_a_list.append(pred)
        
        pred_a = np.mean(pred_a_list)
        
        # ハイブリッド予測
        w_s = metadata['weights']['simple']
        w_a = metadata['weights']['aggressive']
        
        pred_hybrid = pred_s * w_s + pred_a * w_a
        
        # 方向判定
        pred_class = 1 if pred_hybrid > 0.5 else 0
        
        if pred_class == 1:
            direction = "↑ Bullish"
            prob_up = pred_hybrid
            prob_down = 1 - pred_hybrid
        else:
            direction = "↓ Bearish"
            prob_up = pred_hybrid
            prob_down = 1 - pred_hybrid
        
        # 予測価格の計算（config から取得）
        # ハイブリッド信頼度に基づいて価格変動を予測
        confidence_delta = abs(pred_hybrid - 0.5)  # 0-0.5の範囲
        
        # 変動率: 信頼度が高いほど大きな変動を予測
        price_change_pct = confidence_delta * PredictionConfig.MAX_PRICE_CHANGE_PCT
        
        if pred_class == 1:
            predicted_price = current_close * (1 + price_change_pct)
        else:
            predicted_price = current_close * (1 - price_change_pct)
        
        # 信頼度の計算（統一関数を使用）
        final_confidence, _ = calculate_confidence_score(
            pred_proba=pred_hybrid,
            model_accuracy=hybrid_accuracy
        )
        
        return {
            "ticker": ticker,
            "current_price": float(current_close),
            "predicted_price": float(predicted_price),
            "predicted_change_pct": float((predicted_price - current_close) / current_close * 100),
            "direction": direction,
            "prob_up": float(prob_up),
            "prob_down": float(prob_down),
            "confidence": float(final_confidence),
            "market_regime": regime,
            "model_accuracy": float(hybrid_accuracy),
            "simple_pred": float(pred_s),
            "aggressive_pred": float(pred_a),
            "hybrid_pred": float(pred_hybrid),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"[PREDICT ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None
