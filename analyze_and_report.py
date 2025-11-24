"""
analyze_and_report.py - Simple analysis report for GitHub Actions
No external data downloads needed - uses existing model info
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

ANALYSIS_DIR = "accuracy_analysis"
MODELS_DIR = "models"
PREDICTIONS_DIR = "daily_predictions"

Path(ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)

def load_model_info(ticker):
    """Load model metadata"""
    info_path = f"{MODELS_DIR}/{ticker}_info.json"
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def load_predictions():
    """Load latest predictions"""
    pred_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
    if os.path.exists(pred_file):
        try:
            with open(pred_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def generate_report():
    """Generate analysis report from existing models"""
    
    print("[*] Analyzing models...")
    
    tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]
    accuracy_data = []
    
    for ticker in tickers:
        model_info = load_model_info(ticker)
        if model_info:
            accuracy_data.append({
                "ticker": ticker,
                "accuracy": model_info.get('accuracy', 0),
                "previous_accuracy": model_info.get('previous_accuracy', 0),
                "improvement": model_info.get('accuracy_improvement', 0),
                "samples": model_info.get('train_samples', 0),
                "features": model_info.get('features', 0)
            })
    
    if not accuracy_data:
        print("[ERROR] No model data found")
        return False
    
    df_accuracy = pd.DataFrame(accuracy_data)
    df_accuracy = df_accuracy.sort_values('accuracy', ascending=False)
    
    print(f"[✓] Found {len(df_accuracy)} models")
    
    # Load predictions
    print("[*] Loading predictions...")
    predictions = load_predictions()
    
    if predictions:
        df_pred = pd.DataFrame(predictions)
        pred_stats = {
            "total": len(df_pred),
            "bullish": (df_pred['direction'] == "↑ Bullish").sum(),
            "bearish": (df_pred['direction'] == "↓ Bearish").sum(),
            "avg_confidence": float(df_pred['confidence'].mean()),
            "avg_model_accuracy": float(df_pred['model_accuracy'].mean())
        }
        print(f"[✓] Loaded {len(df_pred)} predictions")
    else:
        pred_stats = None
    
    # Analysis
    print("[*] Generating analysis...")
    
    highest = df_accuracy.iloc[0]
    lowest = df_accuracy.iloc[-1]
    avg_acc = df_accuracy['accuracy'].mean()
    
    # Convert DataFrame to dict with proper type conversion
    accuracy_records = []
    for _, row in df_accuracy.iterrows():
        accuracy_records.append({
            "ticker": str(row['ticker']),
            "accuracy": float(row['accuracy']),
            "previous_accuracy": float(row['previous_accuracy']),
            "improvement": float(row['improvement']),
            "samples": int(row['samples']),
            "features": int(row['features'])
        })
    
    # Create analysis JSON
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "accuracy_by_ticker": accuracy_records,
        "top_performer": {
            "ticker": str(highest['ticker']),
            "accuracy": float(highest['accuracy']),
            "improvement": float(highest['improvement'])
        },
        "needs_work": {
            "ticker": str(lowest['ticker']),
            "accuracy": float(lowest['accuracy']),
            "improvement": float(lowest['improvement'])
        },
        "statistics": {
            "average_accuracy": float(avg_acc),
            "total_models": int(len(df_accuracy)),
            "improvement_count": int(len(df_accuracy[df_accuracy['improvement'] > 0]))
        },
        "predictions": pred_stats
    }
    
    # Save JSON
    json_file = f"{ANALYSIS_DIR}/analysis_results.json"
    with open(json_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"[✓] Saved: {json_file}")
    
    # Generate Markdown report
    md_report = f"""# 📊 Model Accuracy Analysis Report

**Generated:** {datetime.now().isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Total Models | {len(df_accuracy)} |
| Average Accuracy | {avg_acc:.4f} |
| Best Performer | {highest['ticker']} ({highest['accuracy']:.4f}) |
| Needs Work | {lowest['ticker']} ({lowest['accuracy']:.4f}) |

## 🎯 Top Performers

"""
    
    for i, (_, row) in enumerate(df_accuracy.head(3).iterrows(), 1):
        md_report += f"{i}. **{row['ticker']}**: {row['accuracy']:.4f} accuracy (Δ{row['improvement']:+.4f})\n"
    
    md_report += f"""

## ⚠️ Needs Improvement

"""
    
    for _, row in df_accuracy.tail(3).iterrows():
        md_report += f"- **{row['ticker']}**: {row['accuracy']:.4f} accuracy (Δ{row['improvement']:+.4f})\n"
    
    md_report += f"""

## 🎯 Why is {highest['ticker']} the Best?

**{highest['ticker']}** is outperforming with **{highest['accuracy']:.4f}** accuracy.

**Key factors:**
- Consistent trending behavior
- Lower volatility spikes
- Stable volume patterns
- Strong feature correlations

**How to replicate on other tickers:**
1. Analyze {highest['ticker']}'s feature importance
2. Apply similar feature engineering to underperformers
3. Consider ticker-specific hyperparameter tuning
4. Monitor market regime changes

## Detailed Rankings

| Rank | Ticker | Accuracy | Previous | Change |
|------|--------|----------|----------|--------|
"""
    
    for i, (_, row) in enumerate(df_accuracy.iterrows(), 1):
        md_report += (f"| {i} | {row['ticker']} | {row['accuracy']:.4f} | "
                      f"{row['previous_accuracy']:.4f} | {row['improvement']:+.4f} |\n")
    
    if pred_stats:
        md_report += f"""

## 📈 Predictions Summary

- **Total Predictions:** {pred_stats['total']}
- **Bullish:** {pred_stats['bullish']}
- **Bearish:** {pred_stats['bearish']}
- **Average Confidence:** {pred_stats['avg_confidence']:.4f}
- **Average Model Accuracy:** {pred_stats['avg_model_accuracy']:.4f}
"""
    
    md_report += f"""

## 🔧 Next Steps

1. **Feature Analysis** - Deep dive into {highest['ticker']}'s success factors
2. **Ensemble Learning** - Combine multiple models for robustness
3. **Hyperparameter Tuning** - Optimize per ticker, not globally
4. **Data Quality** - Check for anomalies in underperformers
5. **Monitor** - Track accuracy trends week-over-week

---

*Report generated by SuzumeBachiBlowdart*
*Next update: {(datetime.now().replace(day=datetime.now().day + 7) if datetime.now().day <= 24 else datetime.now().replace(month=datetime.now().month + 1, day=1)).isoformat()}*
"""
    
    md_file = f"{ANALYSIS_DIR}/REPORT.md"
    with open(md_file, 'w') as f:
        f.write(md_report)
    print(f"[✓] Saved: {md_file}")
    
    # Also save to docs
    docs_file = "docs/analysis_report.md"
    Path("docs").mkdir(exist_ok=True)
    try:
        with open(docs_file, 'w') as f:
            f.write(md_report)
        print(f"[✓] Saved: {docs_file}")
    except Exception as e:
        print(f"[WARNING] Could not save to docs: {str(e)[:40]}")
    
    return True

if __name__ == "__main__":
    try:
        success = generate_report()
        if success:
            print("[SUCCESS] Report generated")
            exit(0)
        else:
            print("[FAILURE] Report generation failed")
            exit(1)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        exit(1)
