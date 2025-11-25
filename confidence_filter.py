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


def calculate_confidence_action(confidence):
    """
    Determine action based on confidence probability (修正版)
    
    Args:
        confidence: [0, 1] の確率値（0.5 = 50%, 1.0 = 100% など）
    
    Returns:
        tuple: (action, level, reason)
    """
    
    # ===== 修正: 確率そのものを判定基準に使用 =====
    if confidence >= 0.60:
        action = "EXECUTE"
        level = "STRONG"
        reason = f"高確率シグナル ({confidence:.1%})"
    
    elif confidence >= 0.50:
        action = "HOLD"
        level = "MEDIUM"
        reason = f"中程度確率 ({confidence:.1%}) - 様子見"
    
    elif confidence >= 0.40:
        action = "HOLD"
        level = "MEDIUM"
        reason = f"低中確率 ({confidence:.1%}) - 確信不足"
    
    else:  # < 0.40
        action = "SKIP"
        level = "WEAK"
        reason = f"低確度ノイズ ({confidence:.1%})"
    
    return action, level, reason


def apply_confidence_filter(predictions, min_confidence=0.40):
    """
    Filter predictions based on confidence level with corrected logic
    
    Args:
        predictions: List of prediction dicts with 'confidence' key
        min_confidence: Minimum confidence threshold for HOLD (default 0.40)
    
    Returns:
        List of predictions with action, confidence_level, reason added
    """
    
    filtered_predictions = []
    
    for pred in predictions:
        confidence = pred.get('confidence', 0.5)
        direction = pred.get('direction', '↓ Bearish')
        
        # ===== 修正: 確率ベースの判定 =====
        action, level, reason = calculate_confidence_action(confidence)
        
        # 計算結果をpredに追加
        pred['confidence_level'] = level
        pred['action'] = action
        pred['reason'] = reason
        
        # confidence_score は補足情報（0-0.5の距離ではなく、信頼度そのもの）
        pred['confidence_score'] = float(confidence)
        
        # recommendation を action に合わせる
        if action == "EXECUTE":
            pred['recommendation'] = f"Execute {direction} trade"
        elif action == "HOLD":
            pred['recommendation'] = "Hold - wait for clearer signal"
        else:  # SKIP
            pred['recommendation'] = "Skip this trade - low confidence"
        
        filtered_predictions.append(pred)
    
    return filtered_predictions


def apply_regime_based_position_limit(
    predictions,
    market_regime="MIXED",
    market_confidence=0.5
):
    """
    Apply market regime-based position size limits
    
    Args:
        predictions: List of filtered predictions
        market_regime: "BULLISH", "MIXED", or "BEARISH"
        market_confidence: 0-1 market confidence
    
    Returns:
        tuple: (adjusted_predictions, regime_config)
    """
    
    # Market regime별 실행 상한
    REGIME_LIMITS = {
        'BULLISH': {
            'max_execute_ratio': 0.80,
            'max_count': None
        },
        'MIXED': {
            'max_execute_ratio': 0.30,
            'max_count': 3
        },
        'BEARISH': {
            'max_execute_ratio': 0.10,
            'max_count': 1
        }
    }
    
    regime_config = REGIME_LIMITS.get(market_regime, REGIME_LIMITS['MIXED'])
    max_ratio = regime_config['max_execute_ratio']
    max_count = regime_config['max_count']
    
    # EXECUTE候補を信頼度でソート
    execute_candidates = [
        p for p in predictions
        if p.get('action') == 'EXECUTE'
    ]
    execute_candidates.sort(
        key=lambda x: x.get('confidence', 0),
        reverse=True
    )
    
    # 上限計算
    total_count = len(predictions)
    max_by_ratio = int(total_count * max_ratio)
    
    if max_count is not None:
        max_execute = min(max_by_ratio, max_count)
    else:
        max_execute = max_by_ratio
    
    # アクション調整
    for i, pred in enumerate(execute_candidates):
        if i < max_execute:
            pred['action'] = 'EXECUTE'
            pred['regime_adjustment'] = None
        else:
            pred['action'] = 'SKIP'
            pred['regime_adjustment'] = f"Regime {market_regime} limit exceeded"
    
    # 既にHOLD/SKIPのものは維持
    for pred in predictions:
        if pred.get('action') not in ['EXECUTE']:
            if 'regime_adjustment' not in pred:
                pred['regime_adjustment'] = None
    
    return predictions, {
        'market_regime': market_regime,
        'max_execute_ratio': max_ratio,
        'max_execute_count': max_execute,
        'total_execute': len(execute_candidates[:max_execute])
    }


def generate_confidence_report(filtered_predictions):
    """Generate a confidence analysis report"""
    
    if not filtered_predictions or len(filtered_predictions) == 0:
        return {
            "total_predictions": 0,
            "average_confidence": 0,
            "execute_count": 0,
            "hold_count": 0,
            "skip_count": 0,
            "note": "No predictions"
        }
    
    confidences = [p.get('confidence', 0) for p in filtered_predictions]
    
    return {
        "total_predictions": len(filtered_predictions),
        "average_confidence": float(np.mean(confidences)),
        "average_confidence_pct": f"{float(np.mean(confidences)):.1%}",
        "execute_count": sum(1 for p in filtered_predictions if p.get('action') == 'EXECUTE'),
        "hold_count": sum(1 for p in filtered_predictions if p.get('action') == 'HOLD'),
        "skip_count": sum(1 for p in filtered_predictions if p.get('action') == 'SKIP'),
        "confidence_distribution": {
            "strong": sum(1 for c in confidences if c >= 0.60),
            "medium": sum(1 for c in confidences if 0.40 <= c < 0.60),
            "weak": sum(1 for c in confidences if c < 0.40)
        },
        "execute_ratio": f"{sum(1 for p in filtered_predictions if p.get('action') == 'EXECUTE') / len(filtered_predictions):.1%}"
    }


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
