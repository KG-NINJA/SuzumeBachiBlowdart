"""
blowdart_ml_engine.py - XGBoost with Online Learning
毎日新しいデータで既存モデルを改善
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

# Configuration
MODELS_DIR = "models"
TRAINING_LOG_DIR = "analytics"
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(TRAINING_LOG_DIR).mkdir(parents=True, exist_ok=True)


def get_model_info_path(ticker):
    """Get path to model metadata"""
    return f"{MODELS_DIR}/{ticker}_info.json"


def load_model_info(ticker):
    """Load model training info"""
    info_path = get_model_info_path(ticker)
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def save_model_info(ticker, info):
    """Save model training info"""
    info_path = get_model_info_path(ticker)
    try:
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        return True
    except:
        return False


def load_existing_model(ticker):
    """Load existing model if available"""
    model_path = f"{MODELS_DIR}/{ticker}_model.json"
    scaler_path = f"{MODELS_DIR}/{ticker}_scaler.pkl"
    
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
    Train or update XGBoost model for a ticker
    
    Args:
        ticker: Stock symbol
        features_df: DataFrame with features and target
        use_online_learning: If True, update existing model; if False, train new
    
    Returns:
        dict: Training info (accuracy, samples, etc.) or None
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
        X = df[numeric_cols].drop('Close', axis=1) if 'Close' in numeric_cols else df[numeric_cols]
        y = df['Target']
        
        if X.empty or len(X) < 30:
            return None
        
        # Remove NaN/Inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
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
            scaler = StandardScaler()
            print(f"  [ONLINE] Creating new scaler")
        
        X_scaled = scaler.fit_transform(X)
        
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
            "accuracy": accuracy,
            "previous_accuracy": previous_accuracy,
            "accuracy_improvement": accuracy_improvement,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_train_samples": previous_train_count + len(X_train),
            "features": len(X.columns),
            "learning_type": learning_type,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        save_model_info(ticker, model_info)
        
        print(f"  [ONLINE] Accuracy: {accuracy:.4f} (Δ{accuracy_improvement:+.4f})")
        
        return model_info
    
    except Exception as e:
        print(f"  [TRAIN ERROR] {ticker}: {str(e)[:60]}")
        return None


def predict_ticker(ticker, features_df):
    """
    Generate prediction for a ticker using trained model
    
    Args:
        ticker: Stock symbol
        features_df: DataFrame with features
    
    Returns:
        dict: Prediction data or None
    """
    try:
        if features_df is None or features_df.empty or len(features_df) < 5:
            return None
        
        # Get latest row
        latest_row = features_df.iloc[-1]
        current_close = latest_row.get('Close')
        
        if current_close is None or current_close <= 0:
            return None
        
        # Load model
        model_path = f"{MODELS_DIR}/{ticker}_model.json"
        scaler_path = f"{MODELS_DIR}/{ticker}_scaler.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None
        
        # Load model and scaler
        booster = xgb.Booster()
        booster.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Prepare features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != 'Close']
        
        if not feature_cols:
            return None
        
        X_latest = features_df[feature_cols].iloc[-1:].copy()
        X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_latest_scaled = scaler.transform(X_latest)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X_latest_scaled)
        
        # Predict
        pred_proba = booster.predict(dmatrix)[0]
        pred_class = 1 if pred_proba > 0.5 else 0
        
        # Estimate price change (use historical volatility)
        returns = features_df['Close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0.02
        
        if pred_class == 1:
            predicted_price = current_close * (1 + abs(volatility) * 0.5)
            direction = "↑ Bullish"
        else:
            predicted_price = current_close * (1 - abs(volatility) * 0.5)
            direction = "↓ Bearish"
        
        # Get model info for additional context
        model_info = load_model_info(ticker)
        model_accuracy = model_info.get('accuracy', 0) if model_info else 0
        
        return {
            "ticker": ticker,
            "current_price": float(current_close),
            "predicted_price": float(predicted_price),
            "direction": direction,
            "confidence": float(pred_proba),
            "model_accuracy": float(model_accuracy),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        print(f"  [PREDICT ERROR] {ticker}: {str(e)[:60]}")
        return None


def get_training_history(ticker):
    """Get training history for a ticker"""
    info_path = get_model_info_path(ticker)
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None
