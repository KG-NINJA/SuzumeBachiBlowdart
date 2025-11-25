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
    
    Args:
        ticker: 銘柄コード
        current_feature_names: 検証用現在の特徴量リスト（Noneの場合は検証スキップ）
        
    Returns:
        tuple: (booster, scaler, metadata) or (None, None, None)
    """
    ticker_dir = get_ticker_dir(ticker)
    model_path = ticker_dir / "model.json"
    scaler_path = ticker_dir / "scaler.pkl"
    metadata_path = ticker_dir / "metadata.json"
    
    # ファイル存在確認
    if not (model_path.exists() and scaler_path.exists() and metadata_path.exists()):
        return None, None, None
        
    try:
        # 1. Validate Metadata First
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        if current_feature_names is not None:
            saved_features = metadata.get('feature_names', [])
            
            # リストの完全一致を確認（XGBoostは順序に敏感なため、setではなくlist比較）
            if saved_features != current_feature_names:
                print(f"  [VALIDATION FAIL] Feature mismatch for {ticker}")
                print(f"    Expected: {len(saved_features)} features")
                print(f"    Got:      {len(current_feature_names)} features")
                # 差分があれば表示（デバッグ用）
                set_saved = set(saved_features)
                set_current = set(current_feature_names)
                if set_saved != set_current:
                    print(f"    Missing: {set_saved - set_current}")
                    print(f"    Extra:   {set_current - set_saved}")
                
                print("    -> Triggering fresh training.")
                return None, None, None

        # 2. Load Model & Scaler
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        return booster, scaler, metadata

    except Exception as e:
        print(f"  [LOAD ERROR] Failed to load {ticker}: {e}")
        return None, None, None


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
            # ここで特徴量名のリストを渡して整合性チェックを行う
            existing_model, existing_scaler, existing_metadata = load_checkpoint(ticker, feature_cols)
            
            if existing_model is not None:
                print(f"  [ONLINE] Valid existing model found for {ticker}")
                learning_type = "ONLINE_UPDATE"
                if existing_metadata:
                    metrics = existing_metadata.get('metrics', {})
                    previous_accuracy = metrics.get('accuracy', 0)
                    total_train_samples = metrics.get('total_train_samples', 0)

        # Scaler Logic
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
        
        # ===== Train or Update =====
        model = None
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

        params = {
            'objective': 'binary:logistic',
            'max_depth': 5,
            'learning_rate': 0.05 if learning_type == "ONLINE_UPDATE" else 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss'
        }

        if learning_type == "ONLINE_UPDATE":
            try:
                # Update existing model
                model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=50,
                    xgb_model=existing_model
                )
            except Exception as e:
                print(f"  [ONLINE FAIL] Update failed: {e}. Fallback to fresh train.")
                learning_type = "FRESH_TRAIN"

        if model is None or learning_type == "FRESH_TRAIN":
            # Fresh train
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=100
            )

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
        
        pred_proba = model.predict(dmatrix)[0]
        
        # Context info
        metrics = metadata.get('metrics', {})
        accuracy = metrics.get('accuracy', 0)
        
        # Estimate direction
        direction = "↑ Bullish" if pred_proba > 0.5 else "↓ Bearish"
        
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
