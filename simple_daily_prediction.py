"""
simple_daily_prediction.py - Main entry point for SuzumeBachiBlowdart
Runs: Data fetch → Feature engineering → Model training → Prediction → Output
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Import custom modules
from utils_data_fetch import safe_price_download, LOGS_DIR
from blowdart_features import build_feature_set
from blowdart_ml_engine import train_ticker, predict_ticker

# ===== Configuration =====
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]
PREDICTIONS_DIR = "daily_predictions"
ANALYTICS_DIR = "analytics"
MODELS_DIR = "models"
DOCS_DATA_DIR = "docs/data"

# Ensure all directories exist
for dir_path in [PREDICTIONS_DIR, ANALYTICS_DIR, MODELS_DIR, LOGS_DIR, DOCS_DATA_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# ===== Main Execution =====
def main():
    print("="*70)
    print(f"SuzumeBachiBlowdart - Daily Prediction Run")
    print(f"Start: {datetime.now().isoformat()}")
    print("="*70)
    
    all_predictions = []
    training_results = []
    
    # ===== Phase 1: Fetch & Train =====
    print("\n[PHASE 1] Data Fetch & Model Training")
    print("-" * 70)
    
    for ticker in TICKERS:
        try:
            # Step 1: Fetch price data
            print(f"\n>>> {ticker}")
            price_data = safe_price_download(ticker, days=180)
            
            if price_data is None or price_data.empty:
                print(f"  ✗ No data retrieved for {ticker}")
                training_results.append({
                    "ticker": ticker,
                    "status": "FAIL",
                    "reason": "No data"
                })
                continue
            
            print(f"  ✓ Data: {len(price_data)} rows")
            
            # Step 2: Build features
            features_df = build_feature_set(price_data, ticker)
            
            if features_df is None or features_df.empty:
                print(f"  ✗ Feature engineering failed")
                training_results.append({
                    "ticker": ticker,
                    "status": "FAIL",
                    "reason": "Feature engineering failed"
                })
                continue
            
            print(f"  ✓ Features: {len(features_df)} rows, {len(features_df.columns)} cols")
            
            # Step 3: Train model
            model_info = train_ticker(ticker, features_df)
            
            if model_info is None:
                print(f"  ✗ Model training failed")
                training_results.append({
                    "ticker": ticker,
                    "status": "FAIL",
                    "reason": "Training failed"
                })
                continue
            
            print(f"  ✓ Model trained: Accuracy={model_info.get('accuracy', 0):.4f}")
            training_results.append({
                "ticker": ticker,
                "status": "OK",
                "accuracy": model_info.get('accuracy'),
                "train_samples": model_info.get('train_samples')
            })
        
        except Exception as e:
            print(f"  ✗ Exception: {str(e)[:60]}")
            training_results.append({
                "ticker": ticker,
                "status": "ERROR",
                "error": str(e)[:60]
            })
    
    # ===== Phase 2: Predict =====
    print("\n" + "="*70)
    print("[PHASE 2] Generate Predictions")
    print("-" * 70)
    
    for ticker in TICKERS:
        try:
            # Fetch latest data
            price_data = safe_price_download(ticker, days=180)
            
            if price_data is None or price_data.empty:
                print(f"{ticker}: No data for prediction")
                continue
            
            # Build features
            features_df = build_feature_set(price_data, ticker)
            
            if features_df is None or features_df.empty:
                print(f"{ticker}: Feature engineering failed")
                continue
            
            # Predict
            prediction = predict_ticker(ticker, features_df)
            
            if prediction is not None:
                all_predictions.append(prediction)
                print(f"{ticker}: ✓ Prediction={prediction.get('predicted_price', 'N/A'):.2f}")
            else:
                print(f"{ticker}: ✗ Prediction failed")
        
        except Exception as e:
            print(f"{ticker}: Exception - {str(e)[:40]}")
    
    # ===== Phase 3: Output & Save =====
    print("\n" + "="*70)
    print("[PHASE 3] Save Results")
    print("-" * 70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save predictions
    predictions_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"✓ Predictions: {predictions_file} ({len(all_predictions)} tickers)")
    
    # Save to docs for GitHub Pages
    docs_file = f"{DOCS_DATA_DIR}/latest_predictions.json"
    with open(docs_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"✓ Dashboard data: {docs_file}")
    
    # Save training metrics
    metrics_file = f"{ANALYTICS_DIR}/training_metrics.json"
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "run_date": datetime.now().strftime('%Y-%m-%d'),
        "results": training_results,
        "total_trained": sum(1 for r in training_results if r['status'] == 'OK'),
        "total_failed": sum(1 for r in training_results if r['status'] != 'OK')
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics: {metrics_file}")
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Predictions generated: {len(all_predictions)}/{len(TICKERS)}")
    print(f"Models trained: {metrics['total_trained']}")
    print(f"Failed: {metrics['total_failed']}")
    print(f"Logs: {LOGS_DIR}/fetch_log_*.txt")
    print("="*70)
    print(f"End: {datetime.now().isoformat()}")
    print("="*70)
    
    # Return status
    if all_predictions:
        print("\n✓ SUCCESS: Predictions were generated")
        return 0
    else:
        print("\n✗ FAILURE: No predictions generated")
        return 1

if __name__ == "__main__":
    sys.exit(main())
