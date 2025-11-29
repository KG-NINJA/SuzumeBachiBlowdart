"""
confidence_filter.py - Confidence-based filtering with corrected logic
Converts predictions to EXECUTE/HOLD/SKIP based on actual probability values.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_DIR = "daily_predictions"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)


def calculate_confidence_score(pred_proba):
    """
    Calculate confidence score from prediction probability
    Returns a score between 0.0 (50/50) and 1.0 (Certain)
    """
    # Convert 0.5-1.0 probability to 0.0-1.0 score
    confidence_score = abs(pred_proba - 0.5) * 2

    # Thresholds: >30% Strong, >10% Medium, <=10% Weak
    if confidence_score > 0.30:
        confidence_level = 'STRONG'
    elif confidence_score > 0.10:
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
            pred['reason'] = f"Low confidence ({conf_score:.1%}) - Market noise"
            pred['recommendation'] = "Skip this trade - wait for clearer signal"
        else:
            pred['action'] = "EXECUTE"
            pred['reason'] = f"High confidence ({conf_score:.1%}) - {conf_level} signal"
            pred['recommendation'] = f"Execute {pred['direction']} trade"

        filtered_predictions.append(pred)

    return filtered_predictions


def generate_confidence_report(filtered_predictions):
    """Generate a confidence analysis report"""

    if not filtered_predictions or len(filtered_predictions) == 0:
        return {
            "total_predictions": 0,
            "average_confidence": 0,
            "execute_count": 0,
            "hold_count": 0,
            "skip_count": 0,
            "confidence_distribution": {"strong": 0, "medium": 0, "weak": 0},
            "execute_ratio": "0.0%",
            "note": "No predictions"
        }

    confidences = [p.get('confidence', 0) for p in filtered_predictions]

    return {
        "total_predictions": len(filtered_predictions),
        "average_confidence": float(np.mean(confidences)),
        "execute_count": sum(1 for p in filtered_predictions if p.get('action') == 'EXECUTE'),
        "hold_count": sum(1 for p in filtered_predictions if p.get('action') == 'HOLD'),
        "skip_count": sum(1 for p in filtered_predictions if p.get('action') == 'SKIP'),
        "confidence_distribution": {
            "strong": sum(1 for c in confidences if abs(c - 0.5) * 2 >= 0.30),
            "medium": sum(1 for c in confidences if 0.10 <= abs(c - 0.5) * 2 < 0.30),
            "weak": sum(1 for c in confidences if abs(c - 0.5) * 2 < 0.10)
        },
        "execute_ratio": f"{sum(1 for p in filtered_predictions if p.get('action') == 'EXECUTE') / len(filtered_predictions):.1%}"
    }



def generate_confidence_markdown(report, predictions):
    """Generate a markdown version of the confidence report"""
    
    md = f"""# 📊 Confidence Analysis Report
**Generated:** {datetime.now().isoformat()}

## 🎯 Summary
- **Average Confidence:** {report['average_confidence']:.1%} (Score 0-100%)
- **Execute Ratio:** {report['execute_ratio']}
- **Total Predictions:** {report['total_predictions']}

## 🚦 Action Breakdown
| Action | Count |
|--------|-------|
| **EXECUTE** | {report['execute_count']} |
| **HOLD** | {report['hold_count']} |
| **SKIP** | {report['skip_count']} |

## 📉 Confidence Distribution
- **STRONG (>30%):** {report['confidence_distribution']['strong']}
- **MEDIUM (10-30%):** {report['confidence_distribution']['medium']}
- **WEAK (<10%):** {report['confidence_distribution']['weak']}

## 📋 Detailed Predictions
| Ticker | Action | Conf Score | Direction | Reason |
|--------|--------|------------|-----------|--------|
"""
    
    # Sort by confidence score descending
    sorted_preds = sorted(predictions, key=lambda x: x.get('confidence_score', 0), reverse=True)
    
    for p in sorted_preds:
        score = p.get('confidence_score', 0)
        direction = "📈" if "Bullish" in p['direction'] else "📉"
        md += f"| **{p['ticker']}** | {p['action']} | {score:.1%} | {direction} | {p.get('reason', '')} |\n"
        
    return md


if __name__ == "__main__":
    # テスト用データ（修正版で期待される動作）
    test_predictions = [
        {"ticker": "A", "confidence": 0.75, "direction": "↑ Bullish"},    # Should be EXECUTE
        {"ticker": "B", "confidence": 0.55, "direction": "↓ Bearish"},    # Should be HOLD
        {"ticker": "C", "confidence": 0.45, "direction": "↑ Bullish"},    # Should be HOLD
        {"ticker": "D", "confidence": 0.30, "direction": "↓ Bearish"},    # Should be SKIP
        {"ticker": "E", "confidence": 0.65, "direction": "↑ Bullish"}     # Should be EXECUTE
    ]
    
    print("=== Testing Confidence Filter (Corrected Logic) ===\n")
    
    # Test 1: Basic filtering
    print("[Test 1] Basic Confidence Filter")
    filtered = apply_confidence_filter(test_predictions)
    
    for pred in filtered:
        print(f"{pred['ticker']}: conf={pred['confidence']:.2%} → "
              f"action={pred['action']:7s} level={pred['confidence_level']}")
    
    # Test 2: Report
    print("\n[Test 2] Confidence Report")
    report = generate_confidence_report(filtered)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Test 3: Regime-based limit
    print("\n[Test 3] Regime-based Position Limit (MIXED)")
    adjusted, regime_config = apply_regime_based_position_limit(
        filtered,
        market_regime="MIXED"
    )
    
    print(f"Market Regime: MIXED")
    print(f"Max Execute Ratio: {regime_config['max_execute_ratio']:.0%}")
    print(f"Max Execute Count: {regime_config['max_execute_count']}")
    print(f"Actual Execute Count: {regime_config['total_execute']}\n")
    
    for pred in adjusted:
        print(f"{pred['ticker']}: {pred['action']:7s} "
              f"(confidence: {pred['confidence']:.1%})")
