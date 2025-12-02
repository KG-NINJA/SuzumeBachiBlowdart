"""
confidence_filter.py - Confidence-based filtering with corrected logic
Converts predictions to EXECUTE/HOLD/SKIP based on actual probability values.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

PREDICTIONS_DIR = "daily_predictions"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)


# ===========================================================
# Core: Unified Confidence Score Calculation
# ===========================================================
def calculate_confidence_score(pred_proba: float) -> Tuple[float, str]:
    """
    唯一の信頼度スコア計算方法（全システムで統一）
    
    確率値から信頼度スコアを計算する
    - 50% 確率 → 0.0 スコア（確信なし）
    - 75% 確率 → 0.5 スコア（強い）
    - 100% 確率 → 1.0 スコア（完全確信）
    
    Args:
        pred_proba: 0.0-1.0 の予測確率
    
    Returns:
        (confidence_score, confidence_level)
            - confidence_score: 0.0-1.0 スケール
            - confidence_level: 'STRONG' | 'MEDIUM' | 'WEAK'
    
    Examples:
        >>> calculate_confidence_score(0.5)
        (0.0, 'WEAK')
        
        >>> calculate_confidence_score(0.75)
        (0.5, 'STRONG')
        
        >>> calculate_confidence_score(0.65)
        (0.3, 'STRONG')
    """
    # 確率を 0.5 からの距離に変換（0.0-1.0 スケール）
    confidence_score = abs(pred_proba - 0.5) * 2
    
    # 信頼度レベル分類
    if confidence_score >= 0.30:
        confidence_level = 'STRONG'
    elif confidence_score >= 0.10:
        confidence_level = 'MEDIUM'
    else:
        confidence_level = 'WEAK'
    
    return float(confidence_score), confidence_level


# ===========================================================
# Main: Apply Confidence Filter
# ===========================================================
def apply_confidence_filter(
    predictions: List[Dict[str, Any]], 
    min_confidence: float = 0.30
) -> List[Dict[str, Any]]:
    """
    信頼度ベースでフィルタリング
    
    Args:
        predictions: 予測結果のリスト
        min_confidence: EXECUTE の最小信頼度（デフォルト: 0.30）
    
    Returns:
        アクション情報を追加した予測リスト
    """
    
    filtered_predictions = []
    
    for pred in predictions:
        # confidence が既に存在するか確認
        if 'confidence' not in pred or pred['confidence'] is None:
            # なければ prob_up から計算
            prob_up = pred.get('prob_up', 0.5)
            confidence = prob_up
        else:
            confidence = pred['confidence']
        
        # 統一された信頼度スコア計算
        conf_score, conf_level = calculate_confidence_score(confidence)
        
        # アクション決定（単純明快なロジック）
        if conf_score >= min_confidence:
            action = 'EXECUTE'
            reason = f'High confidence ({conf_score:.1%}) - {conf_level} signal'
        else:
            action = 'SKIP'
            reason = f'Low confidence ({conf_score:.1%}) - Market noise'
        
        # 結果に追加
        pred['confidence_score'] = conf_score
        pred['confidence_level'] = conf_level
        pred['action'] = action
        pred['reason'] = reason
        pred['recommendation'] = (
            f"Execute {pred['direction']} trade" 
            if action == 'EXECUTE' 
            else "Skip this trade - wait for clearer signal"
        )
        
        filtered_predictions.append(pred)
    
    return filtered_predictions


# ===========================================================
# Report: Generate Confidence Report
# ===========================================================
def generate_confidence_report(
    filtered_predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    信頼度分析レポートを生成
    
    Args:
        filtered_predictions: フィルタリング済み予測リスト
    
    Returns:
        dict: 統計情報を含むレポート
    """
    
    if not filtered_predictions or len(filtered_predictions) == 0:
        return {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": 0,
            "average_confidence": 0.0,
            "average_confidence_pct": "0.0%",
            "execute_count": 0,
            "skip_count": 0,
            "confidence_distribution": {
                "strong": 0,
                "medium": 0,
                "weak": 0
            },
            "execute_ratio": "0.0%",
            "note": "No predictions"
        }
    
    # 統計計算
    confidence_scores = [p.get('confidence_score', 0) for p in filtered_predictions]
    
    execute_count = sum(1 for p in filtered_predictions if p.get('action') == 'EXECUTE')
    skip_count = sum(1 for p in filtered_predictions if p.get('action') == 'SKIP')
    
    strong_count = sum(1 for p in filtered_predictions if p.get('confidence_level') == 'STRONG')
    medium_count = sum(1 for p in filtered_predictions if p.get('confidence_level') == 'MEDIUM')
    weak_count = sum(1 for p in filtered_predictions if p.get('confidence_level') == 'WEAK')
    
    avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    execute_ratio = execute_count / len(filtered_predictions) if filtered_predictions else 0.0
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(filtered_predictions),
        "average_confidence": avg_confidence,
        "average_confidence_pct": f"{avg_confidence:.1%}",
        "execute_count": execute_count,
        "skip_count": skip_count,
        "confidence_distribution": {
            "strong": strong_count,
            "medium": medium_count,
            "weak": weak_count
        },
        "execute_ratio": f"{execute_ratio:.1%}"
    }


# ===========================================================
# Markdown: Generate Confidence Report in Markdown
# ===========================================================
def generate_confidence_markdown(
    report: Dict[str, Any], 
    predictions: List[Dict[str, Any]]
) -> str:
    """
    Markdown形式の信頼度レポートを生成
    
    Args:
        report: 統計レポート
        predictions: 予測リスト
    
    Returns:
        str: Markdown形式のレポート
    """
    
    md = f"""# 📊 Confidence-Based Trading Analysis

**Generated:** {datetime.now().isoformat()}

## 🎯 Summary

- **Total Predictions:** {report['total_predictions']}
- **Average Confidence:** {report['average_confidence_pct']}
- **Execute Ratio:** {report['execute_ratio']}

## 🚦 Action Breakdown

| Action | Count | Percentage |
|--------|-------|-----------|
| **EXECUTE** | {report['execute_count']} | {report['execute_count']/report['total_predictions']*100:.1f}% |
| **SKIP** | {report['skip_count']} | {report['skip_count']/report['total_predictions']*100:.1f}% |

## 📈 Confidence Distribution

| Level | Count | Percentage |
|-------|-------|-----------|
| **STRONG** (≥30%) | {report['confidence_distribution']['strong']} | {report['confidence_distribution']['strong']/report['total_predictions']*100:.1f}% |
| **MEDIUM** (10-30%) | {report['confidence_distribution']['medium']} | {report['confidence_distribution']['medium']/report['total_predictions']*100:.1f}% |
| **WEAK** (<10%) | {report['confidence_distribution']['weak']} | {report['confidence_distribution']['weak']/report['total_predictions']*100:.1f}% |

## 📋 Detailed Predictions

| Ticker | Action | Conf Score | Direction | Reason |
|--------|--------|------------|-----------|--------|
"""
    
    # 信頼度スコアで降順ソート
    sorted_preds = sorted(predictions, key=lambda x: x.get('confidence_score', 0), reverse=True)
    
    for p in sorted_preds:
        ticker = p.get('ticker', '?')
        action = p.get('action', '?')
        conf_score = p.get('confidence_score', 0)
        direction = p.get('direction', '?')
        reason = p.get('reason', '')
        
        # 絵文字を追加
        action_emoji = "🟢" if action == "EXECUTE" else "🔴"
        direction_emoji = "📈" if "Bullish" in direction else "📉"
        
        md += f"| **{ticker}** | {action_emoji} {action} | {conf_score:.1%} | {direction_emoji} | {reason} |\n"
    
    md += f"""

## 🎯 Trading Rules

### Confidence Thresholds

- **STRONG** (score ≥ 30%): Execute the trade
  - Probability > 65% or < 35%
  - Model is very certain about direction
  
- **MEDIUM** (score 10-30%): Consider with caution
  - Probability 55-65% or 35-45%
  - Model has some confidence
  
- **WEAK** (score < 10%): SKIP
  - Probability 45-55%
  - Near coin-flip, too much noise

### Action Rules

1. **EXECUTE:** Only trade HIGH CONFIDENCE (≥30%) signals
2. **SKIP:** Wait for stronger signals on low confidence trades

## 💡 Market Insights

- **High Confidence Trades:** {report['execute_count']} (ready to execute)
- **Low Confidence Trades:** {report['skip_count']} (too noisy)
- **Execution Rate:** {report['execute_ratio']} (realistic filtering)

---

**Next Steps:**
1. Execute only the 🟢 HIGH CONFIDENCE trades
2. Monitor SKIP predictions for signal strength increase
3. Track performance: which confidence levels predict best?
4. Adjust MIN_CONFIDENCE threshold based on results

*Report generated by SuzumeBachiBlowdart Confidence Filter*
"""
    
    return md


# ===========================================================
# Test / Debug: Confidence Score Examples
# ===========================================================
if __name__ == "__main__":
    print("="*70)
    print("Confidence Score Calculation Examples")
    print("="*70)
    
    test_cases = [
        (0.50, "Coin flip (50%)"),
        (0.55, "Slight bias (55%)"),
        (0.60, "Moderate (60%)"),
        (0.65, "Strong (65%)"),
        (0.75, "Very strong (75%)"),
        (1.00, "Certain (100%)"),
    ]
    
    print("\nConfidence Score Mapping:")
    print("-"*70)
    print(f"{'Probability':<15} {'Score':<10} {'Level':<12} {'Description':<25}")
    print("-"*70)
    
    for prob, desc in test_cases:
        score, level = calculate_confidence_score(prob)
        print(f"{prob:.0%}{'':<10} {score:.2f}{'':<5} {level:<12} {desc:<25}")
    
    print("\n" + "="*70)
    print("✅ Confidence Filter Module Loaded Successfully")
    print("="*70)
