"""
confidence_filter.py - Confidence-based filtering with corrected logic
Converts predictions to EXECUTE/HOLD/SKIP based on actual probability values.
Version: 2.0 (Stable with error handling)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import logging

# Setup logging
logger = logging.getLogger("confidence_filter")

PREDICTIONS_DIR = "daily_predictions"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)


# ===========================================================
# Utility: Safe Data Extraction
# ===========================================================
def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    辞書から安全に値を取得
    
    Args:
        obj: 辞書オブジェクト
        key: キー
        default: デフォルト値
    
    Returns:
        値またはデフォルト値
    """
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default
    except Exception:
        return default


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
    
    Formula:
        confidence_score = |pred_proba - 0.5| * 2
    
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
    
    Thresholds:
        - STRONG: score >= 0.30 (65%+ or 35%-の確率)
        - MEDIUM: 0.10 <= score < 0.30 (55-65% or 35-45%)
        - WEAK: score < 0.10 (45-55%の確率, コイントス)
    """
    try:
        # 確率を 0.5 からの距離に変換（0.0-1.0 スケール）
        confidence_score = abs(float(pred_proba) - 0.5) * 2
        
        # 信頼度レベル分類
        if confidence_score >= 0.30:
            confidence_level = 'STRONG'
        elif confidence_score >= 0.10:
            confidence_level = 'MEDIUM'
        else:
            confidence_level = 'WEAK'
        
        return float(confidence_score), confidence_level
    
    except Exception as e:
        logger.warning(f"[CONF_SCORE] Calculation failed: {e}")
        return 0.0, 'WEAK'


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
    
    if not predictions:
        logger.warning("[FILTER] Empty predictions list")
        return []
    
    filtered_predictions = []
    
    for pred in predictions:
        try:
            # confidence が既に存在するか確認
            confidence = safe_get(pred, 'confidence')
            
            if confidence is None:
                # なければ prob_up から計算
                prob_up = safe_get(pred, 'prob_up', 0.5)
                confidence = float(prob_up)
            else:
                confidence = float(confidence)
            
            # 確率の範囲チェック
            if not (0.0 <= confidence <= 1.0):
                logger.warning(f"[FILTER] Invalid confidence: {confidence}")
                confidence = 0.5
            
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
            pred_updated = pred.copy()
            pred_updated['confidence_score'] = float(conf_score)
            pred_updated['confidence_level'] = conf_level
            pred_updated['action'] = action
            pred_updated['reason'] = reason
            pred_updated['recommendation'] = (
                f"Execute {safe_get(pred, 'direction', '?')} trade" 
                if action == 'EXECUTE' 
                else "Skip this trade - wait for clearer signal"
            )
            
            filtered_predictions.append(pred_updated)
        
        except Exception as e:
            logger.error(f"[FILTER] Failed to process prediction: {e}")
            continue
    
    logger.info(f"[FILTER] Processed {len(filtered_predictions)}/{len(predictions)} predictions")
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
    
    if not filtered_predictions:
        logger.warning("[REPORT] No predictions for report")
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
    
    try:
        total_preds = len(filtered_predictions)
        
        # 統計計算
        confidence_scores = [
            safe_get(p, 'confidence_score', 0) 
            for p in filtered_predictions
        ]
        
        execute_count = sum(
            1 for p in filtered_predictions 
            if safe_get(p, 'action') == 'EXECUTE'
        )
        skip_count = sum(
            1 for p in filtered_predictions 
            if safe_get(p, 'action') == 'SKIP'
        )
        
        strong_count = sum(
            1 for p in filtered_predictions 
            if safe_get(p, 'confidence_level') == 'STRONG'
        )
        medium_count = sum(
            1 for p in filtered_predictions 
            if safe_get(p, 'confidence_level') == 'MEDIUM'
        )
        weak_count = sum(
            1 for p in filtered_predictions 
            if safe_get(p, 'confidence_level') == 'WEAK'
        )
        
        avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        execute_ratio = execute_count / total_preds if total_preds > 0 else 0.0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": total_preds,
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
        
        logger.info(f"[REPORT] Generated: {execute_count} execute, {skip_count} skip")
        return report
    
    except Exception as e:
        logger.error(f"[REPORT] Generation failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(filtered_predictions),
            "error": str(e),
            "execute_count": 0,
            "skip_count": len(filtered_predictions)
        }


# ===========================================================
# Markdown: Generate Confidence Report in Markdown
# ===========================================================
def generate_confidence_markdown(
    report: Dict[str, Any], 
    predictions: List[Dict[str, Any]]
) -> str:
    """
    Markdown形式の信頼度レポートを生成（完全エラー対応版）
    
    Args:
        report: 統計レポート
        predictions: 予測リスト
    
    Returns:
        str: Markdown形式のレポート
    """
    
    try:
        # 入力検証
        if not predictions:
            logger.warning("[MARKDOWN] No predictions for report")
            return f"""# 📊 Confidence-Based Trading Analysis

**Generated:** {datetime.now().isoformat()}

⚠️ No predictions available for analysis.
"""
        
        total_preds = len(predictions)
        
        # 統計抽出（安全版）
        execute_count = safe_get(report, 'execute_count', 0)
        skip_count = safe_get(report, 'skip_count', 0)
        avg_confidence = safe_get(report, 'average_confidence', 0.0)
        execute_ratio = safe_get(report, 'execute_ratio', '0.0%')
        
        strong_count = safe_get(report, 'confidence_distribution', {}).get('strong', 0)
        medium_count = safe_get(report, 'confidence_distribution', {}).get('medium', 0)
        weak_count = safe_get(report, 'confidence_distribution', {}).get('weak', 0)
        
        # Markdown 生成
        md = f"""# 📊 Confidence-Based Trading Analysis

**Generated:** {datetime.now().isoformat()}

## 🎯 Summary

- **Total Predictions:** {total_preds}
- **Average Confidence:** {avg_confidence:.1%}
- **Execute Ratio:** {execute_ratio}

## 🚦 Action Breakdown

| Action | Count | Percentage |
|--------|-------|-----------|
| **EXECUTE** | {execute_count} | {execute_count/total_preds*100:.1f}% |
| **SKIP** | {skip_count} | {skip_count/total_preds*100:.1f}% |

## 📈 Confidence Distribution

| Level | Count | Percentage |
|-------|-------|-----------|
| **STRONG** (≥30%) | {strong_count} | {strong_count/total_preds*100:.1f}% |
| **MEDIUM** (10-30%) | {medium_count} | {medium_count/total_preds*100:.1f}% |
| **WEAK** (<10%) | {weak_count} | {weak_count/total_preds*100:.1f}% |

## 📋 Detailed Predictions

| Ticker | Action | Confidence | Direction | Reason |
|--------|--------|------------|-----------|--------|
"""
        
        # 予測データをソート（安全版）
        try:
            sorted_preds = sorted(
                predictions, 
                key=lambda x: safe_get(x, 'confidence_score', 0),
                reverse=True
            )
        except Exception as sort_error:
            logger.warning(f"[MARKDOWN] Sorting failed: {sort_error}")
            sorted_preds = predictions
        
        # テーブル行を生成
        for pred in sorted_preds:
            try:
                ticker = safe_get(pred, 'ticker', '?')
                action = safe_get(pred, 'action', '?')
                conf_score = safe_get(pred, 'confidence_score', 0.0)
                direction = safe_get(pred, 'direction', '?')
                reason = safe_get(pred, 'reason', 'N/A')
                
                # 絵文字
                action_emoji = "🟢" if action == "EXECUTE" else "🔴"
                direction_emoji = "📈" if "Bullish" in str(direction) else "📉"
                
                # テーブル行
                md += (
                    f"| {ticker:6s} | {action_emoji} {action:7s} | "
                    f"{conf_score:6.1%} | {direction_emoji} {str(direction):12s} | "
                    f"{str(reason)[:30]:30s} |\n"
                )
            except Exception as row_error:
                logger.warning(f"[MARKDOWN] Row generation failed: {row_error}")
                continue
        
        # フッター
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

### Action Summary

- **High Confidence Trades:** {execute_count} (ready to execute)
- **Low Confidence Trades:** {skip_count} (too noisy, wait for clarity)
- **Execution Rate:** {execute_ratio}

## 💡 Next Steps

1. **EXECUTE:** Only trade HIGH CONFIDENCE (≥30%) signals
2. **SKIP:** Wait for stronger signals on low confidence trades
3. **Monitor:** Track which confidence levels predict best
4. **Adjust:** Modify MIN_CONFIDENCE threshold based on results

---

*Report generated by SuzumeBachiBlowdart Confidence Filter*
*Time: {datetime.now().isoformat()}*
"""
        
        logger.info("[MARKDOWN] Report generated successfully")
        return md
    
    except Exception as e:
        logger.error(f"[MARKDOWN] Generation failed: {e}", exc_info=True)
        
        # フォールバック: 最小限のレポート
        return f"""# 📊 Confidence Analysis Report

**Generated:** {datetime.now().isoformat()}

⚠️ Error generating detailed report

**Error Details:** {str(e)[:100]}

Please check the JSON report for raw data:
- `daily_predictions/confidence_report.json`
- `daily_predictions/latest_predictions.json`
"""


# ===========================================================
# Test / Debug: Confidence Score Examples
# ===========================================================
if __name__ == "__main__":
    print("="*70)
    print("Confidence Filter Module - Test Suite")
    print("="*70)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n[Test 1] Confidence Score Calculation")
    print("-"*70)
    
    test_cases = [
        (0.50, "Coin flip (50%)"),
        (0.55, "Slight bias (55%)"),
        (0.60, "Moderate (60%)"),
        (0.65, "Strong (65%)"),
        (0.75, "Very strong (75%)"),
        (1.00, "Certain (100%)"),
    ]
    
    print(f"{'Probability':<15} {'Score':<10} {'Level':<12} {'Description':<25}")
    print("-"*70)
    
    for prob, desc in test_cases:
        score, level = calculate_confidence_score(prob)
        print(f"{prob:.0%}{'':<10} {score:.2f}{'':<5} {level:<12} {desc:<25}")
    
    print("\n[Test 2] Filter Application")
    print("-"*70)
    
    test_predictions = [
        {
            "ticker": "NVDA",
            "prob_up": 0.75,
            "direction": "↑ Bullish",
            "current_price": 178.88,
            "predicted_price": 180.0
        },
        {
            "ticker": "AAPL",
            "prob_up": 0.52,
            "direction": "↑ Bullish",
            "current_price": 271.49,
            "predicted_price": 271.5
        }
    ]
    
    filtered = apply_confidence_filter(test_predictions)
    for pred in filtered:
        print(f"{pred['ticker']}: {pred['action']} (conf={pred['confidence_score']:.1%})")
    
    print("\n[Test 3] Report Generation")
    print("-"*70)
    
    report = generate_confidence_report(filtered)
    print(f"Total: {report['total_predictions']}")
    print(f"Execute: {report['execute_count']}")
    print(f"Skip: {report['skip_count']}")
    
    print("\n" + "="*70)
    print("✅ All tests completed successfully")
    print("="*70)
