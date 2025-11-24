"""
confidence_filter.py - Add confidence-based filtering to predictions
Converts low-confidence predictions to HOLD recommendations
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

PREDICTIONS_DIR = "daily_predictions"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)

MIN_CONFIDENCE = 0.15


def calculate_confidence_score(pred_proba):
    """Calculate confidence score from prediction probability"""
    confidence_score = abs(pred_proba - 0.5)
    
    if confidence_score > 0.15:
        confidence_level = 'STRONG'
    elif confidence_score > 0.05:
        confidence_level = 'MEDIUM'
    else:
        confidence_level = 'WEAK'
    
    return confidence_score, confidence_level


def apply_confidence_filter(predictions, min_confidence=0.15):
    """Filter predictions based on confidence level"""
    
    filtered_predictions = []
    
    for pred in predictions:
        confidence = pred.get('confidence', 0.5)
        conf_score, conf_level = calculate_confidence_score(confidence)
        
        pred['confidence_score'] = float(conf_score)
        pred['confidence_level'] = conf_level
        
        if conf_score < min_confidence:
            pred['direction'] = "⏸ HOLD"
            pred['action'] = "SKIP"
            pred['reason'] = f"Low confidence ({confidence:.2%}) - Market noise"
            pred['recommendation'] = "Skip this trade - wait for clearer signal"
        else:
            pred['action'] = "EXECUTE"
            pred['reason'] = f"High confidence ({confidence:.2%}) - {conf_level} signal"
            pred['recommendation'] = f"Execute {pred['direction']} trade"
        
        filtered_predictions.append(pred)
    
    return filtered_predictions


def generate_confidence_report(predictions):
    """Generate analysis report of confidence distribution"""
    
    df = pd.DataFrame(predictions)
    
    df['confidence_pct'] = df['confidence'] * 100
    df['confidence_score'] = df['confidence_score'] * 100
    
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


def main():
    """Main execution"""
    
    print("="*70)
    print("SuzumeBachiBlowdart - Confidence-Based Filtering")
    print("="*70)
    
    predictions_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
    
    if not Path(predictions_file).exists():
        print(f"[ERROR] {predictions_file} not found")
        return 1
    
    print(f"\n[1/4] Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    print(f"  ✓ Loaded {len(predictions)} predictions")
    
    print(f"\n[2/4] Applying confidence filter (MIN_CONFIDENCE={MIN_CONFIDENCE})...")
    filtered_predictions = apply_confidence_filter(predictions, min_confidence=MIN_CONFIDENCE)
    print(f"  ✓ Filtered {len(filtered_predictions)} predictions")
    
    print(f"\n[3/4] Generating confidence report...")
    report = generate_confidence_report(filtered_predictions)
    print(f"  ✓ Execute: {report['actions']['execute']}")
    print(f"  ✓ Skip: {report['actions']['skip']}")
    
    print(f"\n[4/4] Saving results...")
    
    predictions_file = f"{PREDICTIONS_DIR}/filtered_predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(filtered_predictions, f, indent=2, default=str)
    print(f"✓ Saved: {predictions_file}")
    
    report_file = f"{PREDICTIONS_DIR}/confidence_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Saved: {report_file}")
    
    print("\n" + "="*70)
    print("CONFIDENCE FILTER SUMMARY")
    print("="*70)
    print(f"\nTotal Predictions: {report['total_predictions']}")
    print(f"Execute (High Conf): {report['actions']['execute']} 🟢")
    print(f"Skip (Low Conf): {report['actions']['skip']} 🔴")
    
    if report['total_predictions'] > 0:
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
    import sys
    sys.exit(main())
