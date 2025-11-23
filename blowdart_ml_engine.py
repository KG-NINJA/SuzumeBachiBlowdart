"""
blowdart_ml_engine.py - XGBoost training and prediction engine
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuration
MODELS_DIR = "models"
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)


def train_ticker(ticker, features_df):
    """
    Train XGBoost model for a ticker
    
    Args:
        ticker: Stock symbol
        features_df: DataFrame with features and target
    
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
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Train XGBoost
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
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        # Save model
        model_path = f"{MODELS_DIR}/{ticker}_model.json"
        model.get_booster().save_model(model_path)
        
        # Save scaler
        scaler_path = f"{MODELS_DIR}/{ticker}_scaler.pkl"
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return {
            "ticker": ticker,
            "accuracy": accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(X.columns),
            "model_path": model_path,
            "scaler_path": scaler_path
        }
    
    except Exception as e:
        print(f"  [TRAIN ERROR] {ticker}: {str(e)[:60]}")
        return None


def predict_ticker(ticker, features_df):
    """
    Generate prediction for a ticker
    
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
        
        import pickle
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
        
        return {
            "ticker": ticker,
            "current_price": float(current_close),
            "predicted_price": float(predicted_price),
            "direction": direction,
            "confidence": float(pred_proba),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        print(f"  [PREDICT ERROR] {ticker}: {str(e)[:60]}")
        return None
