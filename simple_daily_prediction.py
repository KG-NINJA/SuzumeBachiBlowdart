# ===== ファイルの構造 =====

# インポート
import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# 設定
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]
PREDICTIONS_DIR = "daily_predictions"
# ... etc

# ===== メイン処理 =====

def main():
    """メイン実行関数"""
    
    print("="*70)
    print(f"SuzumeBachiBlowdart - Daily Prediction Run")
    print(f"Start: {datetime.now().isoformat()}")
    print("="*70)
    
    all_predictions = []
    training_results = []
    
    # ===== PHASE 1 =====
    print("\n[PHASE 1] Data Fetch & Model Training (Online Learning)")
    print("-" * 70)
    
    # ... Phase 1の処理 ...
    
    # ===== PHASE 2 =====
    print("\n" + "="*70)
    print("[PHASE 2] Generate Predictions")
    print("-" * 70)
    
    # ... Phase 2の処理 ...
    
    # ===== PHASE 3: Apply Confidence Filter & Save Results =====
    print("\n" + "="*70)
    print("[PHASE 3] Apply Confidence Filter & Save Results")
    print("-" * 70)
    
    from confidence_filter import apply_confidence_filter, generate_confidence_report
    
    # Apply confidence filter to predictions
    print("\n[3-1] Applying confidence-based filter...")
    filtered_predictions = apply_confidence_filter(all_predictions, min_confidence=0.15)
    
    # Count results
    execute_predictions = [p for p in filtered_predictions if p.get('action') == 'EXECUTE']
    skip_predictions = [p for p in filtered_predictions if p.get('action') == 'SKIP']
    
    print(f"  ✓ Execute (High Confidence): {len(execute_predictions)}")
    print(f"  ✓ Skip (Low Confidence): {len(skip_predictions)}")
    
    # ===== Save filtered predictions =====
    print("\n[3-2] Saving results...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
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
    docs_file = f"docs/data/latest_predictions.json"
    with open(docs_file, 'w') as f:
        json.dump(filtered_predictions, f, indent=2, default=str)
    print(f"✓ Dashboard data: {docs_file}")
    
    # Generate confidence report
    print("\n[3-3] Generating confidence analysis...")
    confidence_report = generate_confidence_report(filtered_predictions)
    
    # Save report
    report_file = f"{PREDICTIONS_DIR}/confidence_report.json"
    with open(report_file, 'w') as f:
        json.dump(confidence_report, f, indent=2, default=str)
    print(f"✓ Confidence report: {report_file}")
    
    # Save training metrics
    metrics_file = f"analytics/training_metrics.json"
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "run_date": datetime.now().strftime('%Y-%m-%d'),
        "results": training_results,
        "total_trained": sum(1 for r in training_results if r['status'] == 'OK'),
        "total_failed": sum(1 for r in training_results if r['status'] != 'OK'),
        "predictions": {
            "total": len(filtered_predictions),
            "execute": len(execute_predictions),
            "skip": len(skip_predictions)
        },
        "confidence": confidence_report
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✓ Metrics: {metrics_file}")
    
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
    print(f"  Execute Ratio: {len(execute_predictions) / len(filtered_predictions) * 100:.1f}%")
    
    if execute_predictions:
        avg_confidence_execute = np.mean([p['confidence'] for p in execute_predictions])
        print(f"  Avg Confidence (Execute): {avg_confidence_execute:.4f}")
    
    if skip_predictions:
        avg_confidence_skip = np.mean([p['confidence'] for p in skip_predictions])
        print(f"  Avg Confidence (Skip): {avg_confidence_skip:.4f}")
    
    # Calculate average improvement
    improvements = [r.get('improvement', 0) for r in training_results if r.get('status') == 'OK']
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"Average accuracy improvement: {avg_improvement:+.4f}")
    
    print("="*70)
    print(f"End: {datetime.now().isoformat()}")
    print("="*70)
    
    # ★ ここが重要：main()関数内で return する
    if execute_predictions:
        print("\n✓ SUCCESS: High-confidence predictions generated")
        return 0  # ← インデント重要！
    else:
        print("\n⚠️  WARNING: No high-confidence predictions (all filtered out)")
        return 1  # ← インデント重要！


# ===== 新しい関数を追加 =====

def print_confidence_summary(execute_predictions, skip_predictions):
    """Print detailed confidence analysis"""
    
    print("\n" + "="*70)
    print("CONFIDENCE-BASED TRADING SUMMARY")
    print("="*70)
    
    if execute_predictions:
        print("\n🟢 HIGH CONFIDENCE - EXECUTE THESE TRADES:")
        print("-" * 70)
        for pred in sorted(execute_predictions, key=lambda x: x['confidence'], reverse=True):
            direction_emoji = "📈" if "Bullish" in pred['direction'] else "📉"
            print(f"{pred['ticker']:6s} | {direction_emoji} {pred['direction']:12s} | "
                  f"Conf: {pred['confidence']:.2%} | "
                  f"Model Acc: {pred['model_accuracy']:.2%} | "
                  f"${pred['current_price']:.2f} → ${pred['predicted_price']:.2f}")
    else:
        print("\n🟢 HIGH CONFIDENCE - EXECUTE THESE TRADES:")
        print("-" * 70)
        print("No high-confidence signals at this time.")
    
    if skip_predictions:
        print("\n🔴 LOW CONFIDENCE - SKIP THESE (HOLD):")
        print("-" * 70)
        for pred in sorted(skip_predictions, key=lambda x: x['confidence'], reverse=True):
            direction_emoji = "📈" if "Bullish" in pred['direction'] else "📉"
            print(f"{pred['ticker']:6s} | {direction_emoji} {pred['direction']:12s} | "
                  f"Conf: {pred['confidence']:.2%} | "
                  f"Model Acc: {pred['model_accuracy']:.2%} | Reason: Low confidence")
    else:
        print("\n🔴 LOW CONFIDENCE - SKIP THESE (HOLD):")
        print("-" * 70)
        print("All predictions have sufficient confidence!")
    
    print("="*70)


# ===== エントリーポイント（ここが重要） =====

if __name__ == "__main__":
    # main()を実行
    exit_code = main()
    
    # 信頼度サマリーも実行
    try:
        predictions_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
        with open(predictions_file, 'r') as f:
            filtered_predictions = json.load(f)
        
        execute_preds = [p for p in filtered_predictions if p.get('action') == 'EXECUTE']
        skip_preds = [p for p in filtered_predictions if p.get('action') == 'SKIP']
        
        print_confidence_summary(execute_preds, skip_preds)
    except Exception as e:
        print(f"\n[WARNING] Could not print confidence summary: {str(e)}")
    
    # 最後に sys.exit() を呼び出す
    sys.exit(exit_code)
