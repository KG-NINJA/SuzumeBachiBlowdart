"""
confidence_filter.py - Add confidence-based filtering to predictions
Converts low-confidence predictions to HOLD/SKIP recommendations based on probability thresholds.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_DIR = "daily_predictions"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)

# 注: 新ロジックでは関数内でハードコードされた閾値(0.6/0.5/0.4)を使用します
DEFAULT_MIN_CONFIDENCE = 0.60 


def calculate_confidence_level(confidence, direction):
    """
    再設計: 確率ベースの三値判定
    
    Args:
        confidence: [0, 1] の確率値
        direction: "Bullish" or "Bearish" (表示用に使用可能)
    
    Returns:
        tuple: (action, level, reason)
    """
    
    # ===== 新ロジック =====
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
        reason = f"中程度確率 ({confidence:.1%}) - 確信不足"
    
    else:  # < 0.40
        action = "SKIP"
        level = "WEAK"
        reason = f"低確度ノイズ ({confidence:.1%})"
    
    return action, level, reason


def apply_confidence_filter(predictions, min_confidence=DEFAULT_MIN_CONFIDENCE):
    """
    Filter predictions based on confidence level using the v2 logic.
    (修正版フィルタ: 矛盾を排除し、明確なアクションを割り当て)
    """
    filtered_predictions = []

    for pred in predictions:
        # データ取得（デフォルトは0）
        confidence = pred.get('confidence', 0)
        direction = pred.get('direction', '')
        
        # 新しい判定ロジックを適用
        action, level, reason = calculate_confidence_level(confidence, direction)
        
        # 結果を辞書に格納
        pred['confidence_level'] = level
        pred['action'] = action
        pred['reason'] = reason
        
        # EXECUTEの場合のみ推奨文を作成、それ以外はHold/Skipのメッセージ
        if action == "EXECUTE":
            pred['recommendation'] = f"Execute {direction} trade"
        elif action == "HOLD":
            pred['recommendation'] = "Hold position - Wait for clearer signal"
            # HOLDの場合は方向指示を一時停止アイコンに変更
            pred['direction'] = "⏸ HOLD"
        else: # SKIP
            pred['recommendation'] = "Skip trade - Market noise"
            pred['direction'] = "❌ SKIP"

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
            # 新しいロジックに合わせて分布の閾値を調整 (Strong >= 0.6, Medium >= 0.4)
            "high_strong": sum(1 for c in confidences if c >= 0.6),
            "medium": sum(1 for c in confidences if 0.4 <= c < 0.6),
            "low_weak": sum(1 for c in confidences if c < 0.4)
        }
    }

if __name__ == "__main__":
    # テスト用データ
    test_predictions = [
        {"confidence": 0.75, "direction": "Bullish"}, # Should be EXECUTE
        {"confidence": 0.55, "direction": "Bearish"}, # Should be HOLD
        {"confidence": 0.45, "direction": "Bullish"}, # Should be HOLD
        {"confidence": 0.30, "direction": "Bearish"}  # Should be SKIP
    ]

    print("--- Testing Confidence Filter ---")
    filtered = apply_confidence_filter(test_predictions)
    print(json.dumps(filtered, indent=2, ensure_ascii=False))
    
    print("\n--- Report ---")
    report = generate_confidence_report(filtered)
    print(json.dumps(report, indent=2))
