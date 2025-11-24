import json
import pandas as pd
from pathlib import Path
from datetime import datetime

PREDICTIONS_DIR = "daily_predictions"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)

# Configuration
MIN_CONFIDENCE = 0.35  # Predictions between 0.35-0.65 are unreliable
CONFIDENCE_THRESHOLDS = {
    'strong': 0.65,      # |confidence - 0.5| > 0.15 → Strong signal
    'medium': 0.55,      # |confidence - 0.5| > 0.05 → Medium signal
    'weak': 0.50         # |confidence - 0.5| ≤ 0.05 → Weak signal (HOLD)
}


def calculate_confidence_score(pred_proba):
    """
    Calculate confidence score from prediction probability
    
    A prediction is confident if it's far from 0.5 (coin flip)
    
    Args:
        pred_proba: Probability from model (0-1)
    
    Returns:
        confidence_score: Distance from 0.5 (0-0.5)
        confidence_level: 'STRONG', 'MEDIUM', 'WEAK'
    """
    confidence_score = abs(pred_proba - 0.5)
    
    if confidence_score > 0.15:
        confidence_level = 'STRONG'
    elif confidence_score > 0.05:
        confidence_level = 'MEDIUM'
    else:
        confidence_level = 'WEAK'
    
    return confidence_score, confidence_level


def apply_confidence_filter(predictions, min_confidence=0.15):
    """
    Filter predictions based on confidence level
    
    Args:
        predictions: List of prediction dicts
        min_confidence: Minimum confidence score to act on prediction
    
    Returns:
        List of filtered predictions with HOLD recommendations
    """
    
    filtered_predictions = []
    
    for pred in predictions:
        confidence = pred.get('confidence', 0.5)
        
        # Calculate confidence score
        conf_score, conf_level = calculate_confidence_score(confidence)
        
        # Add confidence analysis
        pred['confidence_score'] = float(conf_score)
        pred['confidence_level'] = conf_level
        
        # Apply filtering
        if conf_score < min_confidence:
            # Low confidence → HOLD
            pred['direction'] = "⏸ HOLD"
            pred['action'] = "SKIP"
            pred['reason'] = f"Low confidence ({confidence:.2%}) - Market noise"
            pred['recommendation'] = "Skip this trade - wait for clearer signal"
        else:
            # High confidence → BUY/SELL
            pred['action'] = "EXECUTE"
            pred['reason'] = f"High confidence ({confidence:.2%}) - {conf_level} signal"
            pred['recommendation'] = f"Execute {pred['direction']} trade"
        
        filtered_predictions.append(pred)
    
    return filtered_predictions


def generate_confidence_report(predictions):
    """
    Generate analysis report of confidence distribution
    
    Args:
        predictions: List of filtered predictions
    
    Returns:
        dict: Confidence analysis
    """
    
    df = pd.DataFrame(predictions)
    
    # Convert confidence to percentage
    df['confidence_pct'] = df['confidence'] * 100
    df['confidence_score'] = df['confidence_score'] * 100
    
    # Statistics
    strong_count = len(df[df['confidence_level'] == 'STRONG'])
    medium_count = len(df[df['confidence_level'] == 'MEDIUM'])
    weak_count = len(df[df['confidence_level'] == 'WEAK'])
    
    execute_count = len(df[df['action'] == 'EXECUTE'])
    skip_count = len(df[df['action'] == 'SKIP'])
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_predictions': len(df),
        'confidence_distribution': {
            'strong': int(strong_count),
            'medium': int(medium_count),
            'weak': int(weak_count)
        },
        'actions': {
            'execute': int(execute_count),
            'skip': int(skip_count)
        },
        'statistics': {
            'avg_confidence': float(df['confidence'].mean()),
            'min_confidence': float(df['confidence'].min()),
            'max_confidence': float(df['confidence'].max()),
            'std_confidence': float(df['confidence'].std()),
            'avg_confidence_score': float(df['confidence_score'].mean())
        },
        'by_ticker': {}
    }
    
    # Breakdown by ticker
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].iloc[0]
        report['by_ticker'][ticker] = {
            'ticker': ticker,
            'confidence': float(ticker_data['confidence']),
            'confidence_level': ticker_data['confidence_level'],
            'action': ticker_data['action'],
            'direction': ticker_data['direction'],
            'model_accuracy': float(ticker_data['model_accuracy'])
        }
    
    return report


def save_filtered_predictions(predictions, report):
    """
    Save filtered predictions and report
    
    Args:
        predictions: List of filtered predictions
        report: Confidence analysis report
    """
    
    # Save filtered predictions
    predictions_file = f"{PREDICTIONS_DIR}/filtered_predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"✓ Saved: {predictions_file}")
    
    # Save report
    report_file = f"{PREDICTIONS_DIR}/confidence_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Saved: {report_file}")


def generate_markdown_report(predictions, report):
    """
    Generate human-readable markdown report
    
    Args:
        predictions: List of filtered predictions
        report: Confidence analysis report
    
    Returns:
        str: Markdown report
    """
    
    df = pd.DataFrame(predictions)
    
    markdown = f"""# 📊 Confidence-Based Trading Report

**Generated:** {datetime.now().isoformat()}

## 🎯 Summary

- **Total Predictions:** {report['total_predictions']}
- **Execute (High Confidence):** {report['actions']['execute']} 🟢
- **Skip (Low Confidence):** {report['actions']['skip']} 🔴
- **Execute Ratio:** {report['actions']['execute'] / report['total_predictions'] * 100:.1f}%

## 📈 Confidence Distribution

| Level | Count | Percentage |
|-------|-------|-----------|
| 🟢 STRONG | {report['confidence_distribution']['strong']} | {report['confidence_distribution']['strong'] / report['total_predictions'] * 100:.1f}% |
| 🟡 MEDIUM | {report['confidence_distribution']['medium']} | {report['confidence_distribution']['medium'] / report['total_predictions'] * 100:.1f}% |
| 🔴 WEAK | {report['confidence_distribution']['weak']} | {report['confidence_distribution']['weak'] / report['total_predictions'] * 100:.1f}% |

## 📊 Confidence Statistics

| Metric | Value |
|--------|-------|
| Average Confidence | {report['statistics']['avg_confidence']:.4f} |
| Min Confidence | {report['statistics']['min_confidence']:.4f} |
| Max Confidence | {report['statistics']['max_confidence']:.4f} |
| Std Dev | {report['statistics']['std_confidence']:.4f} |
| Avg Confidence Score | {report['statistics']['avg_confidence_score']:.2f}% |

## 🎲 Detailed Predictions

"""
    
    # Executive trades (EXECUTE)
    execute_df = df[df['action'] == 'EXECUTE'].sort_values('confidence', ascending=False)
    
    markdown += "### 🟢 HIGH CONFIDENCE - EXECUTE TRADES\n\n"
    
    if len(execute_df) > 0:
        markdown += "| Ticker | Direction | Confidence | Score | Model Acc | Current | Target |\n"
        markdown += "|--------|-----------|-----------|-------|-----------|---------|--------|\n"
        
        for _, row in execute_df.iterrows():
            emoji = "📈" if "Bullish" in row['direction'] else "📉"
            markdown += (f"| {row['ticker']} | {emoji} {row['direction']} | "
                        f"{row['confidence']:.2%} | {row['confidence_score']:.1f}% | "
                        f"{row['model_accuracy']:.2%} | ${row['current_price']:.2f} | "
                        f"${row['predicted_price']:.2f} |\n")
        
        markdown += "\n**Interpretation:** These signals have strong confidence and should be executed.\n\n"
    else:
        markdown += "No high-confidence signals at this time.\n\n"
    
    # Skip trades (HOLD)
    skip_df = df[df['action'] == 'SKIP'].sort_values('confidence', ascending=False)
    
    markdown += "### 🔴 LOW CONFIDENCE - SKIP (HOLD)\n\n"
    
    if len(skip_df) > 0:
        markdown += "| Ticker | Direction | Confidence | Score | Model Acc | Reason |\n"
        markdown += "|--------|-----------|-----------|-------|-----------|--------|\n"
        
        for _, row in skip_df.iterrows():
            emoji = "📈" if "Bullish" in row['direction'] else "📉"
            markdown += (f"| {row['ticker']} | {emoji} {row['direction']} | "
                        f"{row['confidence']:.2%} | {row['confidence_score']:.1f}% | "
                        f"{row['model_accuracy']:.2%} | Low confidence |\n")
        
        markdown += "\n**Interpretation:** These signals are near coin-flip. Skip for now and wait for clearer signals.\n\n"
    else:
        markdown += "All predictions have sufficient confidence.\n\n"
    
    # Risk analysis
    markdown += """
## 📋 Ticker Analysis

"""
    
    for ticker, info in report['by_ticker'].items():
        status = "🟢" if info['action'] == 'EXECUTE' else "🟡"
        markdown += f"### {status} {ticker}\n\n"
        markdown += f"- **Prediction:** {info['direction']}\n"
        markdown += f"- **Confidence:** {info['confidence']:.2%} ({info['confidence_level']})\n"
        markdown += f"- **Model Accuracy:** {info['model_accuracy']:.2%}\n"
        markdown += f"- **Action:** {'✅ EXECUTE' if info['action'] == 'EXECUTE' else '⏸ HOLD'}\n\n"
    
    # Trading rules
    markdown += """## 🎯 Trading Rules

### Confidence Levels
- **STRONG** (Score > 15%): Execute the trade
  - Probability > 65% or < 35%
  - Model is very certain about direction
  
- **MEDIUM** (Score 5-15%): Consider with caution
  - Probability 55-65% or 35-45%
  - Model has some confidence
  
- **WEAK** (Score ≤ 5%): HOLD/SKIP
  - Probability 45-55%
  - Near coin-flip probability
  - Too much noise, skip this one

### Action Rules
1. **EXECUTE:** Only trade HIGH CONFIDENCE signals
2. **SKIP:** Wait for stronger signals on low confidence trades
3. **Combine:** Use Model Accuracy + Confidence Score together
   - High accuracy + High confidence = BEST
   - High accuracy + Low confidence = CAUTION
   - Low accuracy + Any confidence = AVOID

## 💡 Insights

- Filter out {report['actions']['skip']} noisy signals ({report['actions']['skip'] / report['total_predictions'] * 100:.1f}%)
- Focus on {report['actions']['execute']} high-confidence trades ({report['actions']['execute'] / report['total_predictions'] * 100:.1f}%)
- This reduces false signals while maintaining quality trades
- Especially useful when market is noisy (like today with 9/10 bearish)

---

**Next Steps:**
1. Execute only the 🟢 HIGH CONFIDENCE trades
2. Monitor HOLD predictions for signal strength increase
3. Track performance: which confidence levels predict best?
4. Adjust MIN_CONFIDENCE threshold based on results

*Report generated by SuzumeBachiBlowdart Confidence Filter*
"""
    
    return markdown


def main():
    """Main execution"""
    
    print("="*70)
    print("SuzumeBachiBlowdart - Confidence-Based Filtering")
    print("="*70)
    
    # Load latest predictions
    predictions_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
    
    if not Path(predictions_file).exists():
        print(f"[ERROR] {predictions_file} not found")
        return 1
    
    print(f"\n[1/4] Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    print(f"  ✓ Loaded {len(predictions)} predictions")
    
    # Apply confidence filter
    print(f"\n[2/4] Applying confidence filter (MIN_CONFIDENCE={MIN_CONFIDENCE})...")
    filtered_predictions = apply_confidence_filter(predictions, min_confidence=MIN_CONFIDENCE)
    print(f"  ✓ Filtered {len(filtered_predictions)} predictions")
    
    # Generate report
    print(f"\n[3/4] Generating confidence report...")
    report = generate_confidence_report(filtered_predictions)
    print(f"  ✓ Execute: {report['actions']['execute']}")
    print(f"  ✓ Skip: {report['actions']['skip']}")
    
    # Save results
    print(f"\n[4/4] Saving results...")
    save_filtered_predictions(filtered_predictions, report)
    
    # Generate markdown report
    md_report = generate_markdown_report(filtered_predictions, report)
    md_file = f"{PREDICTIONS_DIR}/confidence_report.md"
    with open(md_file, 'w') as f:
        f.write(md_report)
    print(f"✓ Saved: {md_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("CONFIDENCE FILTER SUMMARY")
    print("="*70)
    print(f"\nTotal Predictions: {report['total_predictions']}")
    print(f"Execute (High Conf): {report['actions']['execute']} 🟢")
    print(f"Skip (Low Conf): {report['actions']['skip']} 🔴")
    print(f"Execute Ratio: {report['actions']['execute'] / report['total_predictions'] * 100:.1f}%")
    
    print(f"\nConfidence Distribution:")
    print(f"  Strong: {report['confidence_distribution']['strong']}")
    print(f"  Medium: {report['confidence_distribution']['medium']}")
    print(f"  Weak: {report['confidence_distribution']['weak']}")
    
    print(f"\nAverage Confidence: {report['statistics']['avg_confidence']:.4f}")
    print(f"Avg Confidence Score: {report['statistics']['avg_confidence_score']:.2f}%")
    
    print("\n✅ Confidence filter applied successfully!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())
