"""
Integration guide: Add confidence filter to simple_daily_prediction.py

Replace the main() function's Phase 3 with the following code
"""

# ===== Phase 3: Apply Confidence Filter & Save Results =====
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
docs_file = f"{DOCS_DATA_DIR}/latest_predictions.json"
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
metrics_file = f"{ANALYTICS_DIR}/training_metrics.json"
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

print(f"Logs: {LOGS_DIR}/fetch_log_*.txt")
print("="*70)
print(f"End: {datetime.now().isoformat()}")
print("="*70)

# Return status
if execute_predictions:
    print("\n✓ SUCCESS: High-confidence predictions generated")
    return 0
else:
    print("\n⚠️  WARNING: No high-confidence predictions (all filtered out)")
    return 1


# ===== UPDATED SUMMARY SECTION TO ADD AFTER main() =====

"""
Additional code to add to simple_daily_prediction.py at the end:

def print_confidence_summary(execute_predictions, skip_predictions):
    '''Print detailed confidence analysis'''
    
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
                  f"{pred['current_price']:.2f} → {pred['predicted_price']:.2f}")
    
    if skip_predictions:
        print("\n🔴 LOW CONFIDENCE - SKIP THESE (HOLD):")
        print("-" * 70)
        for pred in sorted(skip_predictions, key=lambda x: x['confidence'], reverse=True):
            direction_emoji = "📈" if "Bullish" in pred['direction'] else "📉"
            print(f"{pred['ticker']:6s} | {direction_emoji} {pred['direction']:12s} | "
                  f"Conf: {pred['confidence']:.2%} | "
                  f"Model Acc: {pred['model_accuracy']:.2%} | Reason: Low confidence")
    
    print("="*70)
"""
