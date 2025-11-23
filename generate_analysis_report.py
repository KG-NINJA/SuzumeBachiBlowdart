"""
generate_analysis_report.py - Generate analysis report for GitHub Actions
Lightweight version that doesn't require all data downloads
Uses existing model info and predictions
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ANALYSIS_DIR = "accuracy_analysis"
MODELS_DIR = "models"
PREDICTIONS_DIR = "daily_predictions"
ANALYTICS_DIR = "analytics"

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


def analyze_existing_models():
    """Analyze already-trained models without re-downloading data"""
    
    tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]
    accuracy_data = []
    
    print("[1/4] Loading model metadata...")
    
    for ticker in tickers:
        model_info = load_model_info(ticker)
        if model_info:
            accuracy_data.append({
                "ticker": ticker,
                "accuracy": model_info.get('accuracy', 0),
                "previous_accuracy": model_info.get('previous_accuracy', 0),
                "improvement": model_info.get('accuracy_improvement', 0),
                "train_samples": model_info.get('train_samples', 0),
                "total_samples": model_info.get('total_train_samples', 0),
                "features": model_info.get('features', 0),
                "learning_type": model_info.get('learning_type', 'UNKNOWN'),
                "timestamp": model_info.get('timestamp', '')
            })
    
    df_accuracy = pd.DataFrame(accuracy_data)
    
    if df_accuracy.empty:
        print("  ⚠️  No model data found. Run training first.")
        return None
    
    df_accuracy = df_accuracy.sort_values('accuracy', ascending=False)
    
    print(f"  ✓ Loaded {len(df_accuracy)} models")
    
    return df_accuracy


def analyze_predictions():
    """Analyze prediction patterns"""
    
    print("[2/4] Analyzing predictions...")
    
    predictions_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
    
    if not os.path.exists(predictions_file):
        print("  ⚠️  No predictions found")
        return None
    
    try:
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        df_pred = pd.DataFrame(predictions)
        
        # Analyze confidence distribution
        confidence_data = {
            "mean_confidence": df_pred['confidence'].mean(),
            "std_confidence": df_pred['confidence'].std(),
            "bullish_count": (df_pred['direction'] == "↑ Bullish").sum(),
            "bearish_count": (df_pred['direction'] == "↓ Bearish").sum(),
            "avg_model_accuracy": df_pred['model_accuracy'].mean()
        }
        
        print(f"  ✓ Analyzed {len(df_pred)} predictions")
        
        return confidence_data
    
    except Exception as e:
        print(f"  ⚠️  Error analyzing predictions: {str(e)[:40]}")
        return None


def identify_googl_success_factors(df_accuracy):
    """Identify why GOOGL has high accuracy"""
    
    print("[3/4] Analyzing GOOGL success factors...")
    
    googl_data = df_accuracy[df_accuracy['ticker'] == 'GOOGL']
    
    if googl_data.empty:
        print("  ⚠️  GOOGL data not found")
        return None
    
    googl_accuracy = googl_data.iloc[0]['accuracy']
    average_accuracy = df_accuracy['accuracy'].mean()
    
    analysis = {
        "googl_accuracy": float(googl_accuracy),
        "average_accuracy": float(average_accuracy),
        "advantage": float(googl_accuracy - average_accuracy),
        "percentile": float((df_accuracy['accuracy'] >= googl_accuracy).sum() / len(df_accuracy) * 100),
        "rank": int((df_accuracy['accuracy'] >= googl_accuracy).sum()),
        "improvement_vs_previous": float(googl_data.iloc[0].get('improvement', 0))
    }
    
    print(f"  ✓ GOOGL: {googl_accuracy:.4f} (Rank: #{analysis['rank']}, "
          f"Advantage: {analysis['advantage']:+.4f})")
    
    return analysis


def identify_low_performers(df_accuracy):
    """Identify tickers that need improvement"""
    
    print("[4/4] Identifying improvement candidates...")
    
    # Get bottom 3
    low_performers = df_accuracy.nsmallest(3, 'accuracy')
    
    recommendations = []
    
    for _, row in low_performers.iterrows():
        ticker = row['ticker']
        accuracy = row['accuracy']
        improvement = row['improvement']
        
        rec = {
            "ticker": ticker,
            "current_accuracy": float(accuracy),
            "recent_improvement": float(improvement),
            "suggestions": get_suggestions_for_ticker(ticker, accuracy, improvement)
        }
        
        recommendations.append(rec)
        print(f"  ⚠️  {ticker}: {accuracy:.4f} accuracy")
    
    return recommendations


def get_suggestions_for_ticker(ticker, accuracy, improvement):
    """Generate improvement suggestions based on performance"""
    
    suggestions = []
    
    if accuracy < 0.52:
        suggestions.append("Consider adding volatility clustering indicators (GARCH, Vol of Vol)")
    
    if accuracy < 0.50:
        suggestions.append("Try ensemble learning (combine XGBoost + LightGBM)")
    
    if improvement < 0:
        suggestions.append("Recent performance degrading - check for market regime change")
    
    if improvement < -0.02:
        suggestions.append("Consider retraining model from scratch (online learning not helping)")
    
    if not suggestions:
        suggestions.append("Model performing steadily - focus on feature engineering improvements")
    
    return suggestions


def generate_markdown_report(df_accuracy, googl_analysis, low_performers, pred_analysis):
    """Generate markdown report"""
    
    print("\nGenerating markdown report...")
    
    # Build accuracy table
    acc_table = "| Rank | Ticker | Accuracy | Previous | Improvement |\n"
    acc_table += "|------|--------|----------|----------|-------------|\n"
    
    for i, (_, row) in enumerate(df_accuracy.iterrows(), 1):
        acc_table += (f"| {i} | {row['ticker']} | "
                      f"{row['accuracy']:.4f} | "
                      f"{row['previous_accuracy']:.4f} | "
                      f"{row['improvement']:+.4f} |\n")
    
    # Build recommendations table
    rec_table = "| Ticker | Current | Recommendations |\n"
    rec_table += "|--------|---------|------------------|\n"
    
    for rec in low_performers:
        suggestions_text = " | ".join(rec['suggestions'][:2])  # Top 2 suggestions
        rec_table += (f"| {rec['ticker']} | {rec['current_accuracy']:.4f} | "
                      f"{suggestions_text} |\n")
    
    # Generate markdown
    markdown = f"""# 📊 Model Accuracy Analysis Report

**Generated:** {datetime.now().isoformat()}

## Summary

- **Total Models:** {len(df_accuracy)}
- **Average Accuracy:** {df_accuracy['accuracy'].mean():.4f}
- **Best Performer:** {df_accuracy.iloc[0]['ticker']} ({df_accuracy.iloc[0]['accuracy']:.4f})
- **Needs Work:** {df_accuracy.iloc[-1]['ticker']} ({df_accuracy.iloc[-1]['accuracy']:.4f})

## 🎯 GOOGL Success Analysis

GOOGL is performing at the **top of the cohort**:

- **Current Accuracy:** {googl_analysis['googl_accuracy']:.4f}
- **System Average:** {googl_analysis['average_accuracy']:.4f}
- **Advantage:** {googl_analysis['advantage']:+.4f} ({googl_analysis['percentile']:.1f}th percentile)
- **Recent Improvement:** {googl_analysis['improvement_vs_previous']:+.4f}

**Why GOOGL Works Well:**
- Historically demonstrates strong trending behavior
- Lower volatility spikes compared to other tech stocks
- Consistent volume patterns
- Strong feature correlations with price movement

**Action Items to Replicate GOOGL Success:**
1. Analyze GOOGL's top feature importances
2. Ensure those same features are well-engineered for underperformers
3. Consider ticker-specific hyperparameter tuning (not one-size-fits-all)
4. Monitor market regime changes (trending vs range-bound)

## 📈 Accuracy Rankings

{acc_table}

## ⚠️ Improvement Priorities

{rec_table}

## 🔧 Detailed Recommendations

"""
    
    for rec in low_performers:
        markdown += f"\n### {rec['ticker']}\n\n"
        markdown += f"**Current Accuracy:** {rec['current_accuracy']:.4f}\n\n"
        markdown += "**Suggested Actions:**\n"
        for suggestion in rec['suggestions']:
            markdown += f"- {suggestion}\n"
    
    if pred_analysis:
        markdown += f"""

## 📊 Prediction Statistics

- **Mean Confidence:** {pred_analysis['mean_confidence']:.4f}
- **Bullish Predictions:** {pred_analysis['bullish_count']}
- **Bearish Predictions:** {pred_analysis['bearish_count']}
- **Average Model Accuracy:** {pred_analysis['avg_model_accuracy']:.4f}

"""
    
    markdown += """
---

## 📚 Next Steps

1. **Implement GOOGL features on other tickers** - Analyze what makes GOOGL work
2. **Ensemble learning** - Combine multiple models for robustness
3. **Feature engineering** - Add market regime indicators
4. **Hyperparameter tuning** - Use Optuna for automatic optimization
5. **Data validation** - Check for data quality issues in underperformers

---
*Report generated by SuzumeBachiBlowdart GitHub Actions*
"""
    
    return markdown


def main():
    print("="*70)
    print("SuzumeBachiBlowdart - Accuracy Analysis Report Generator")
    print("="*70 + "\n")
    
    # 1. Analyze existing models
    df_accuracy = analyze_existing_models()
    if df_accuracy is None:
        print("\n✗ Cannot proceed without model data")
        return 1
    
    # 2. Analyze predictions
    pred_analysis = analyze_predictions()
    
    # 3. Identify GOOGL success factors
    googl_analysis = identify_googl_success_factors(df_accuracy)
    if googl_analysis is None:
        return 1
    
    # 4. Identify low performers
    low_performers = identify_low_performers(df_accuracy)
    
    # 5. Generate markdown report
    markdown_report = generate_markdown_report(
        df_accuracy, googl_analysis, low_performers, pred_analysis
    )
    
    # 6. Save reports
    print("\nSaving reports...")
    
    # Save as JSON
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "accuracy_by_ticker": df_accuracy.to_dict(orient='records'),
        "googl_analysis": googl_analysis,
        "low_performers": low_performers,
        "prediction_stats": pred_analysis
    }
    
    json_path = f"{ANALYSIS_DIR}/analysis_results.json"
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"  ✓ JSON report: {json_path}")
    
    # Save as Markdown
    md_path = f"{ANALYSIS_DIR}/REPORT.md"
    with open(md_path, 'w') as f:
        f.write(markdown_report)
    print(f"  ✓ Markdown report: {md_path}")
    
    # Also save to docs for GitHub Pages
    docs_md = "docs/analysis_report.md"
    Path("docs").mkdir(exist_ok=True)
    with open(docs_md, 'w') as f:
        f.write(markdown_report)
    print(f"  ✓ GitHub Pages: {docs_md}")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())
