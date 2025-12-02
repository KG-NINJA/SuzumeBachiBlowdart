import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import subprocess

# Import custom modules
from utils_data_fetch import safe_price_download, LOGS_DIR
from blowdart_features import build_feature_set
from blowdart_ml_engine import train_ticker, predict_ticker
from confidence_filter import apply_confidence_filter, generate_confidence_report, generate_confidence_markdown

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
    print(f"SuzumeBachiBlowdart - Daily Prediction Run (Complete Pipeline)")
    run_timestamp = datetime.now().isoformat()
    print(f"Start: {run_timestamp}")
    print("="*70)
    
    all_predictions = []
    training_results = []
    
    # ===== Phase 0: Data Fetch & Model Training =====
    print("\n[PHASE 0] Data Fetch & Model Training (Online Learning)")
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
            
            print(f"  ✓ Model trained: Accuracy={model_info.get('hybrid_acc', 0):.4f}")
            training_results.append({
                "ticker": ticker,
                "status": "OK",
                "accuracy": model_info.get('hybrid_acc'),
                "simple_acc": model_info.get('simple_acc', 0),
                "aggressive_acc": model_info.get('aggressive_acc', 0),
                "train_samples": model_info.get('test_size'),
                "market_regime": model_info.get('market_regime', 'UNKNOWN')
            })
        
        except Exception as e:
            print(f"  ✗ Exception: {str(e)[:60]}")
            training_results.append({
                "ticker": ticker,
                "status": "ERROR",
                "error": str(e)[:60]
            })
    
    # ===== Phase 1: Generate Predictions =====
    print("\n" + "="*70)
    print("[PHASE 1] Generate Predictions")
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
    
    # ===== Phase 2: Apply Confidence Filter =====
    print("\n" + "="*70)
    print("[PHASE 2] Apply Confidence Filter")
    print("-" * 70)
    
    print("\n[2-1] Applying confidence-based filter...")
    # Threshold: 0.30 (30% score)
    filtered_predictions = apply_confidence_filter(all_predictions, min_confidence=0.30)
    
    execute_predictions = [p for p in filtered_predictions if p.get('action') == 'EXECUTE']
    skip_predictions = [p for p in filtered_predictions if p.get('action') == 'SKIP']
    
    print(f"  ✓ Execute (High Confidence): {len(execute_predictions)}")
    print(f"  ✓ Skip (Low Confidence): {len(skip_predictions)}")
    
    # ===== Phase 3: Save Results =====
    print("\n" + "="*70)
    print("[PHASE 3] Save Results")
    print("-" * 70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n[3-1] Saving predictions...")
    
    # Save all predictions (with confidence analysis)
    predictions_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(filtered_predictions, f, indent=2, default=str)
    print(f"✓ All predictions: {predictions_file}")
    
    # Save filtered predictions (separate file for easy access)
    filtered_file = f"{PREDICTIONS_DIR}/filtered_predictions.json"
    with open(filtered_file, 'w') as f:
        json.dump(execute_predictions, f, indent=2, default=str)
    print(f"✓ Filtered predictions (EXECUTE only): {filtered_file}")
    
    # Save to docs for GitHub Pages
    docs_file = f"{DOCS_DATA_DIR}/latest_predictions.json"
    with open(docs_file, 'w') as f:
        json.dump(filtered_predictions, f, indent=2, default=str)
    print(f"✓ Dashboard data: {docs_file}")
    
    # Generate confidence report
    print("\n[3-2] Generating confidence analysis...")
    try:
        confidence_report = generate_confidence_report(filtered_predictions)
        
        # Save JSON report
        report_file = f"{PREDICTIONS_DIR}/confidence_report.json"
        with open(report_file, 'w') as f:
            json.dump(confidence_report, f, indent=2, default=str)
        print(f"✓ Confidence report: {report_file}")
        
        # Generate Markdown Report
        try:
            md_report = generate_confidence_markdown(confidence_report, filtered_predictions)
            md_file = f"{PREDICTIONS_DIR}/confidence_report.md"
            
            # Delete stale file if exists
            if os.path.exists(md_file):
                os.remove(md_file)
            
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_report)
            print(f"✓ Confidence markdown: {md_file}")
        
        except Exception as md_error:
            print(f"⚠️ Markdown generation failed: {str(md_error)[:60]}")
    
    except Exception as report_error:
        print(f"⚠️ Confidence report generation failed: {str(report_error)[:60]}")
    
    # Save training metrics
    metrics_file = f"{ANALYTICS_DIR}/training_metrics.json"
    metrics = {
        "timestamp": run_timestamp,
        "run_date": datetime.now().strftime('%Y-%m-%d'),
        "results": training_results,
        "total_trained": sum(1 for r in training_results if r['status'] == 'OK'),
        "total_failed": sum(1 for r in training_results if r['status'] != 'OK'),
        "predictions": {
            "total": len(filtered_predictions),
            "execute": len(execute_predictions),
            "skip": len(skip_predictions)
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✓ Metrics: {metrics_file}")
    
    # ===== Phase 4: Market Environment Analysis =====
    print("\n" + "="*70)
    print("[PHASE 4] Market Environment Analysis (Phase 2)")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            ["python", "market_regime_analysis.py", "--timestamp", run_timestamp],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✓ Market regime analysis completed")
        else:
            print(f"⚠️ Market regime analysis failed")
    except subprocess.TimeoutExpired:
        print("⚠️ Market regime analysis timeout")
    except Exception as e:
        print(f"⚠️ Market regime analysis error: {str(e)[:60]}")
    
    # ===== Phase 5: Backtest Validation =====
    print("\n" + "="*70)
    print("[PHASE 5] Backtest Validation (Phase 3)")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            ["python", "backtest_engine.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✓ Backtest validation completed")
        else:
            print(f"⚠️ Backtest validation failed")
    except subprocess.TimeoutExpired:
        print("⚠️ Backtest validation timeout")
    except Exception as e:
        print(f"⚠️ Backtest validation error: {str(e)[:60]}")
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Predictions generated: {len(filtered_predictions)}/{len(TICKERS)}")
    print(f"Models trained: {metrics['total_trained']}")
    print(f"Failed: {metrics['total_failed']}")
    print(f"\nConfidence Filter Results:")
    print(f"  Execute (High Conf): {len(execute_predictions)}")
    print(f"  Skip (Low Conf): {len(skip_predictions)}")

    if len(filtered_predictions) > 0:
        execute_ratio = len(execute_predictions) / len(filtered_predictions) * 100
        print(f"  Execute Ratio: {execute_ratio:.1f}%")
    else:
        print(f"  Execute Ratio: N/A (no predictions)")
    
    if execute_predictions:
        avg_confidence_execute = np.mean([p['confidence_score'] for p in execute_predictions])
        print(f"  Avg Confidence (Execute): {avg_confidence_execute:.4f}")
    
    if skip_predictions:
        avg_confidence_skip = np.mean([p['confidence_score'] for p in skip_predictions])
        print(f"  Avg Confidence (Skip): {avg_confidence_skip:.4f}")
    
    print("\n" + "="*70)
    print(f"End: {datetime.now().isoformat()}")
    print("="*70)
    
    # Return status
    print("\n✅ SUCCESS: Complete pipeline executed")
    return 0


# ===========================================================
# FIXED: print_confidence_summary - 矛盾がない論理
# ===========================================================
def print_confidence_summary(execute_predictions, skip_predictions):
    """
    信頼度分析レポートを出力（矛盾のない論理）
    """
    
    print("\n" + "="*70)
    print("CONFIDENCE-BASED TRADING SUMMARY")
    print("="*70)
    
    # ===== HIGH CONFIDENCE セクション =====
    print("\n🟢 HIGH CONFIDENCE - EXECUTE THESE TRADES:")
    print("-" * 70)
    
    if len(execute_predictions) > 0:
        # HIGH CONFIDENCE がある場合
        for pred in sorted(execute_predictions, key=lambda x: x.get('confidence_score', 0), reverse=True):
            ticker = pred.get('ticker', '?')
            direction = pred.get('direction', '?')
            conf_score = pred.get('confidence_score', 0)
            model_acc = pred.get('model_accuracy', 0)
            curr_price = pred.get('current_price', 0)
            pred_price = pred.get('predicted_price', 0)
            
            direction_emoji = "📈" if "Bullish" in str(direction) else "📉"
            
            print(f"{ticker:6s} | {direction_emoji} {str(direction):12s} | "
                  f"Conf: {conf_score:.1%} | "
                  f"Model Acc: {model_acc:.1%} | "
                  f"${curr_price:.2f} → ${pred_price:.2f}")
    else:
        # HIGH CONFIDENCE がない場合
        print("No high-confidence signals at this time.")
        print("→ Market conditions uncertain. Waiting for clearer signals.")
    
    # ===== LOW CONFIDENCE セクション =====
    print("\n🔴 LOW CONFIDENCE - SKIP THESE (HOLD):")
    print("-" * 70)
    
    if len(skip_predictions) > 0:
        # LOW CONFIDENCE がある場合
        for pred in sorted(skip_predictions, key=lambda x: x.get('confidence_score', 0), reverse=True):
            ticker = pred.get('ticker', '?')
            direction = pred.get('direction', '?')
            conf_score = pred.get('confidence_score', 0)
            model_acc = pred.get('model_accuracy', 0)
            
            direction_emoji = "📈" if "Bullish" in str(direction) else "📉"
            
            print(f"{ticker:6s} | {direction_emoji} {str(direction):12s} | "
                  f"Conf: {conf_score:.1%} | "
                  f"Model Acc: {model_acc:.1%} | Reason: Low confidence")
    else:
        # LOW CONFIDENCE がない場合
        print("✅ All predictions have sufficient confidence!")
        print("→ Ready to execute all signals.")
    
    print("="*70)


if __name__ == "__main__":
    exit_code = main()
    
    # Print confidence summary（修正版）
    try:
        predictions_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
        if os.path.exists(predictions_file):
            with open(predictions_file, 'r') as f:
                filtered_predictions = json.load(f)
            
            execute_preds = [p for p in filtered_predictions if p.get('action') == 'EXECUTE']
            skip_preds = [p for p in filtered_predictions if p.get('action') == 'SKIP']
            
            # 修正版の関数を呼び出し
            print_confidence_summary(execute_preds, skip_preds)
        else:
            print("\n⚠️ Predictions file not found")
    
    except Exception as e:
        print(f"\n[WARNING] Could not print confidence summary: {str(e)[:60]}")
    
    sys.exit(exit_code)
