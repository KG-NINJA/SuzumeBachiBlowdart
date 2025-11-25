"""
blowdart_ml_engine.py - XGBoost with Online Learning & Metadata Validation
毎日新しいデータで既存モデルを改善し、特徴量の不整合を自動検知して修復
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
from datetime import datetime

# Configuration
MODELS_ROOT = Path("models")
TRAINING_LOG_DIR = Path("analytics")
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_ticker_dir(ticker):
    """Get (and create) the directory for a specific ticker"""
    ticker_dir = MODELS_ROOT / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir


def save_checkpoint(ticker, model, scaler, feature_names, metrics=None):
    """
    モデル、スケーラー、メタデータを一括保存
    
    Args:
        ticker: 銘柄コード
        model: XGBoost Booster object
        scaler: Fitted StandardScaler
        feature_names: List of feature column names (order matters)
        metrics: Dictionary of performance metrics
    """
    ticker_dir = get_ticker_dir(ticker)
    
    try:
        # 1. Save XGBoost Model (JSON for compatibility)
        model_path = ticker_dir / "model.json"
        model.save_model(str(model_path))
        
        # 2. Save Scaler (Pickle)
        scaler_path = ticker_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        # 3. Save Metadata
        metadata = {
            "ticker": ticker,
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "saved_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "model_type": "XGBoost_Booster"
        }
        
        metadata_path = ticker_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"  [SAVE] Checkpoint saved for {ticker} at {ticker_dir}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to save checkpoint for {ticker}: {e}")
        return False


def load_checkpoint(ticker, current_feature_names=None):
    """
    モデルとスケーラーをロードし、特徴量の整合性を検証する
    
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
    
    return None, None


def train_ticker(ticker, features_df, use_online_learning=True):
    """
    Train or update XGBoost model for a ticker with validation
    """
    try:
        if features_df is None or features_df.empty or len(features_df) < 30:
            return None
        
        # Prepare data
        df = features_df.copy()
        
        # Drop non-numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create target: next day close > current close
        if 'Close' not in df.columns:
            return None
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        if len(df) < 30:
            return None
        
        # Separate features and target
        # 特徴量の名前リストを保存（重要）
        feature_cols = [c for c in numeric_cols if c != 'Close']
        X = df[feature_cols]
        y = df['Target']
        
        if X.empty or len(X) < 30:
            return None
        
        # Remove NaN/Inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # ===== Online Learning Preparation =====
        existing_model = None
        existing_scaler = None
        existing_metadata = None
        previous_accuracy = 0
        total_train_samples = 0
        learning_type = "FRESH_TRAIN"
        
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
        else:
            scaler = StandardScaler()
            scaler.fit(X) # Fresh fit if new scaler
            
        # Transform (Note: if online update, we generally should fit scaler on new data too? 
        # Ideally, online learning updates scaler, but StandardScaler is static. 
        # For simplicity in this script, we re-fit scaler on FRESH, use existing on UPDATE)
        if learning_type == "FRESH_TRAIN":
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
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
        predictions = model.predict(dtest)
        pred_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_binary == y_test)
        
        accuracy_improvement = accuracy - previous_accuracy
        
        # Metrics to save
        new_metrics = {
            "accuracy": float(accuracy),
            "previous_accuracy": float(previous_accuracy),
            "accuracy_improvement": float(accuracy_improvement),
            "train_samples": len(X_train),
            "total_train_samples": total_train_samples + len(X_train),
            "learning_type": learning_type
        }
        
        # ===== Save Checkpoint =====
        save_checkpoint(ticker, model, scaler, feature_cols, new_metrics)
        
        print(f"  [RESULT] {ticker} | Acc: {accuracy:.4f} ({accuracy_improvement:+.4f}) | {learning_type}")
        
        return new_metrics
    
    except Exception as e:
        print(f"  [TRAIN ERROR] {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_ticker(ticker, features_df):
    """
    Generate prediction for a ticker using trained model with validation
    """
    try:
        if features_df is None or features_df.empty or len(features_df) < 5:
            return None
        
        # Prepare features same as training
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != 'Close']
        
        if not feature_cols:
            return None
            
        # Load Checkpoint with Validation
        # ここで現在のデータフレームのカラム構造が保存時と一致するか確認
        model, scaler, metadata = load_checkpoint(ticker, feature_cols)
        
        if model is None:
            print(f"  [PREDICT] Model not found or incompatible for {ticker}")
            return None
            
        # Get latest data
        latest_row = features_df.iloc[-1]
        current_close = latest_row.get('Close')
        
        X_latest = features_df[feature_cols].iloc[-1:].copy()
        X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Predict
        X_latest_scaled = scaler.transform(X_latest)
        dmatrix = xgb.DMatrix(X_latest_scaled, feature_names=feature_cols)
        
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
            "direction": direction,
            "confidence": float(pred_proba),
            "model_accuracy": float(accuracy),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        print(f"  [PREDICT ERROR] {ticker}: {e}")
        return None
