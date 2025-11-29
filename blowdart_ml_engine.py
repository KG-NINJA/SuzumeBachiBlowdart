"""
blowdart_ml_engine.py - Fixed Version
完全な相関性問題を排除した修正版
- リーク特徴の除外を調整
- ティッカー固有の特徴選択
- 精度ベースの信頼度計算
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
warnings.filterwarnings('ignore')

MODELS_ROOT = Path("models")
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
REGIME_LOG = Path("regime_detection")
REGIME_LOG.mkdir(parents=True, exist_ok=True)

# ===== IMPROVED LEAK FEATURES (過度な除外を避ける) =====
# 本当に当日の情報のみ
LEAK_FEATURES = {
    'CloseOpenRatio',           # 当日の Open/Close
    'DailyReturn',              # 当日のリターン
    'HighLowRatio',             # 当日の High/Low
}

# ===== これらは除外しない（前営業日情報として有効） =====
KEEP_FEATURES = {
    'ATR', 'OBV', 'MACD', 'RSI7', 'RSI14',
    'Momentum', 'Momentum_5', 'Momentum_10',
    'Volume', 'Volume_Ratio',
    'EMA12', 'EMA26', 'MA5', 'MA10', 'MA20', 'MA50',
    'Plus_DI', 'Minus_DI', 'ADX',
    'VROC', 'PVT',
    'Low', 'High', 'Open'
}


def detect_market_regime_fixed(df, ticker, lookback=20):
    """改善版：ティッカーごとに異なるレジーム判定"""
    
    if len(df) < lookback:
        return 'NEUTRAL', 0.5, 0.5
    
    recent = df.iloc[-lookback:].copy()
    returns = recent['Close'].pct_change().dropna()
    
    if len(returns) == 0:
        return 'NEUTRAL', 0.5, 0.5
    
    volatility_raw = returns.std()
    
    # 過去全期間での正規化（より安定的）
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
    
    # トレンド強度
    close_diff = recent['Close'].diff().fillna(0)
    up_days = (close_diff > 0).sum()
    down_days = (close_diff < 0).sum()
    total = up_days + down_days + 1e-6
    trend_strength = abs(up_days - down_days) / total
    
    # レジーム判定：より厳密に
    if trend_strength > 0.65 and vol_percentile < 0.55:
        regime = 'TRENDING'
    elif vol_percentile > 0.75:  # 高めに設定
        regime = 'CHOPPY'
    else:
        regime = 'MEAN_REVERSION'
    
    print(f"  [REGIME] {ticker}: {regime:15s} | Vol={vol_percentile:.2f} | Trend={trend_strength:.2f}")
    
    return regime, float(vol_percentile), float(trend_strength)


def get_ticker_specific_features(df, ticker):
    """ティッカー固有の特徴選択"""
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            booster = xgb.Booster()
            booster.load_model(model_path)

            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            return booster, scaler
        except Exception as e:
            print(f"  [WARNING] Failed to load existing model: {str(e)[:40]}")
            return None, None
    
    # 上位15-25特徴を選択（ティッカーごとに異なる）
    num_features = min(20, max(15, len(sorted_features)))
    selected = [f[0] for f in sorted_features[:num_features]]
    
    # KEEP_FEATURES との交差を優先
    keep_intersection = [f for f in selected if f in KEEP_FEATURES]
    other = [f for f in selected if f not in KEEP_FEATURES]
    
    final_features = keep_intersection + other
    final_features = final_features[:20]  # 最大20個
    
    print(f"  [FEATURES] {ticker}: {len(final_features)} selected")
    print(f"             Keep: {len(keep_intersection)} | Other: {len(other)}")
    
    return final_features


def get_ticker_dir(ticker):
    ticker_dir = MODELS_ROOT / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir


def train_ticker(ticker, features_df, use_online_learning=True, use_feature_reduction=True):
    """改善版：ティッカーごとに異なる特徴と予測を生成"""
    
    try:
        print(f"\n  [TRAIN] {ticker} - 改善版モード起動")
        
        if features_df is None or features_df.empty or len(features_df) < 40:
            print(f"  [ERROR] Insufficient data for {ticker}")
            return None
        
        df = features_df.copy()
        
        # Target生成
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        if len(df) < 40:
            return None
        
        # ===== ティッカー固有の特徴選択 =====
        feature_cols = get_ticker_specific_features(df, ticker)
        
        if not feature_cols:
            print(f"  [ERROR] No valid features for {ticker}")
            return None
        
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df['Target']
        
        # ===== レジーム検知 =====
        regime, vol, trend = detect_market_regime_fixed(df, ticker)
        
        # Remove NaN/Inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # ===== Time-Series Split (Fixing Data Leakage) =====
        # Split into 80% train, 20% test (strictly by time)
        split_idx = int(len(X) * 0.8)
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        print(f"  [SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")
        
        # ===== Online Learning Logic =====
        existing_model = None
        existing_scaler = None
        previous_accuracy = 0
        previous_train_count = 0
        
        if use_online_learning:
            existing_model, existing_scaler = load_existing_model(ticker)
            model_info = load_model_info(ticker)

            if model_info:
                previous_accuracy = model_info.get('accuracy', 0)
                previous_train_count = model_info.get('total_train_samples', 0)

            # Ensure feature compatibility before attempting an online update
            if existing_model is not None:
                try:
                    model_feature_count = existing_model.num_features()
                    data_feature_count = X.shape[1]
                    if model_feature_count != data_feature_count:
                        print(f"  [ONLINE] Feature mismatch (model={model_feature_count}, data={data_feature_count}); retraining from scratch")
                        existing_model = None
                        existing_scaler = None
                except Exception as e:
                    print(f"  [ONLINE] Failed to inspect existing model: {str(e)[:40]}")
                    existing_model = None
                    existing_scaler = None
        
        # Use existing scaler or create new one
        if existing_scaler is not None:
            scaler = existing_scaler
            print(f"  [ONLINE] Using existing scaler")
        else:
            w_simple, w_agg = 0.5, 0.5
            desc = "BALANCED"
        
        print(f"  [WEIGHTS] Simple={w_simple:.0%} | Aggressive={w_agg:.0%} ({desc})")
        
        # ===== シンプルモデル =====
        scaler_s = StandardScaler()
        X_train_s = scaler_s.fit_transform(X_train)
        X_test_s = scaler_s.transform(X_test)
        
        model_s = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        model_s.fit(X_train_s, y_train)
        acc_s = model_s.score(X_test_s, y_test)
        
        # ===== アグレッシブアンサンブル =====
        scaler_a = RobustScaler()
        X_train_a = scaler_a.fit_transform(X_train)
        X_test_a = scaler_a.transform(X_test)
        
        models_a = []
        accs_a = []
        
        for fold in range(5):
            m = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4 + fold % 2,
                learning_rate=0.08 * (0.9 + fold * 0.02),
                subsample=0.7 + fold * 0.04,
                colsample_bytree=0.75 + fold * 0.03,
                gamma=0.5 + fold * 0.1,
                random_state=42 + fold,
                eval_metric='logloss',
                verbosity=0
            )
            m.fit(X_train_a, y_train)
            models_a.append(m)
            accs_a.append(m.score(X_test_a, y_test))
        
        acc_a = np.mean(accs_a)
        
        # ===== ハイブリッド精度 =====
        hybrid_acc = acc_s * w_simple + acc_a * w_agg
        
        # ===== Train or Update Model =====
        model = None
        learning_type = "FRESH_TRAIN"

        if existing_model is not None and use_online_learning:
            # Online Learning: Update existing model
            print(f"  [ONLINE] Updating existing model...")

            try:
                # Convert to DMatrix
                dtrain_new = xgb.DMatrix(X_train, label=y_train)

                # Train on new data while keeping old knowledge
                model = xgb.train(
                    params={
                        'objective': 'binary:logistic',
                        'max_depth': 5,
                        'learning_rate': 0.05,  # Lower LR for gradual updates
                        'subsample': 0.8,
                        'colsample_bytree': 0.8
                    },
                    dtrain=dtrain_new,
                    num_boost_round=50,  # Add 50 new boosting rounds
                    xgb_model=existing_model  # Start from existing model
                )

                learning_type = "ONLINE_UPDATE"
            except Exception as e:
                print(f"  [ONLINE] Update failed: {str(e)[:60]} - retraining from scratch")
                model = None

        if model is None:
            # Fresh Training: Create new model
            print(f"  [ONLINE] Training new model...")

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )

            model.fit(X_train, y_train)
            model = model.get_booster()  # Convert to Booster for consistency

            learning_type = "FRESH_TRAIN"
        
        # Evaluate
        dtest = xgb.DMatrix(X_test, label=y_test)
        predictions = model.predict(dtest)
        pred_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_binary == y_test)
        
        # Calculate improvement
        accuracy_improvement = accuracy - previous_accuracy
        
        # Save model
        model_path = f"{MODELS_DIR}/{ticker}_model.json"
        model.save_model(model_path)
        print(f"  [ONLINE] Model saved: {model_path}")
        
        # Save scaler
        scaler_path = f"{MODELS_DIR}/{ticker}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save model info
        model_info = {
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
            "version": "2025-11-27-fixed",
            "saved_at": datetime.now().isoformat()
        }
        
        with open(ticker_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  [RESULT] {ticker} | Hybrid Acc: {hybrid_acc:.4f}")
        
        return {
            "accuracy": float(hybrid_acc),
            "regime": regime,
            "simple_weight": w_simple,
            "aggressive_weight": w_agg,
            "learning_type": "HYBRID_FIXED"
        }
    
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_ticker(ticker, features_df):
    """改善版：実際の精度に基づいた信頼度計算"""
    
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
        
        feature_cols = metadata['feature_names']
        
        # 最新データ準備
        latest_row = features_df.iloc[-1]
        current_close = float(latest_row.get('Close', 0))
        
        X_latest = features_df[feature_cols].iloc[-1:].copy()
        X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # ===== シンプルモデル予測 =====
        scaler_s = pickle.load(open(ticker_dir / "scaler_simple.pkl", "rb"))
        model_s = xgb.XGBClassifier()
        model_s.load_model(str(ticker_dir / "model_simple.json"))
        
        X_s = scaler_s.transform(X_latest)
        pred_s = float(model_s.predict_proba(X_s)[0][1])
        
        # ===== アグレッシブアンサンブル予測 =====
        scaler_a = pickle.load(open(ticker_dir / "scaler_agg.pkl", "rb"))
        X_a = scaler_a.transform(X_latest)
        
        pred_a_list = []
        for fold in range(5):
            m = xgb.XGBClassifier()
            m.load_model(str(ticker_dir / f"model_agg_{fold}.json"))
            pred = float(m.predict_proba(X_a)[0][1])
            pred_a_list.append(pred)
        
        pred_a = np.mean(pred_a_list)
        
        # ===== ハイブリッド予測 =====
        w_s = metadata['weights']['simple']
        w_a = metadata['weights']['aggressive']
        
        pred_hybrid = pred_s * w_s + pred_a * w_a
        
        if pred_class == 1:
            direction = "↑ Bullish"
        else:
            direction = "↓ Bearish"

        # predicted_price を計算 (信頼度に基づく微調整)
        predicted_price = current_close * (1 + (pred_proba - 0.5) * 0.02)

        # Get model info for additional context
        model_info = load_model_info(ticker)
        model_accuracy = model_info.get('accuracy', 0) if model_info else 0
        
        return {
            "ticker": ticker,
            "current_price": float(current_close),
            "predicted_price": float(predicted_price),
            "direction": direction,
            "confidence": float(final_confidence),
            "market_regime": regime,
            "model_accuracy": float(hybrid_accuracy),
            "simple_pred": float(pred_s),
            "aggressive_pred": float(pred_a),
            "hybrid_pred": float(pred_hybrid),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        print(f"  [PREDICT ERROR] {ticker}: {e}")
        return None
