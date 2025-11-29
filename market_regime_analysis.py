"""
market_regime_analysis.py - Phase 2: Market Environment Detection
Detects market regime (UPTREND, DOWNTREND, MIXED) and adjusts predictions accordingly
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PREDICTIONS_DIR = "daily_predictions"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)


def analyze_market_sentiment(predictions, timestamp=None):
    """
    Analyze overall market sentiment from predictions
    
    Args:
        predictions: List of prediction dicts
    
    Returns:
        dict: Market sentiment analysis
    """
    
    df = pd.DataFrame(predictions)
    
    # Count directions
    bullish_count = (df['direction'].str.contains('Bullish', na=False)).sum()
    bearish_count = (df['direction'].str.contains('Bearish', na=False)).sum()
    hold_count = (df['direction'].str.contains('HOLD', na=False)).sum()
    
    total = len(df)
    
    bullish_ratio = bullish_count / total if total > 0 else 0
    bearish_ratio = bearish_count / total if total > 0 else 0
    
    # Determine market regime
    if bullish_ratio > 0.7:
        regime = "STRONG_UPTREND"
        regime_signal = "🟢 BUY"
    elif bearish_ratio > 0.7:
        regime = "STRONG_DOWNTREND"
        regime_signal = "🔴 SELL"
    elif bullish_ratio > 0.55:
        regime = "UPTREND"
        regime_signal = "🟢 BUY"
    elif bearish_ratio > 0.55:
        regime = "DOWNTREND"
        regime_signal = "🔴 SELL"
    else:
        regime = "MIXED"
        regime_signal = "🟡 NEUTRAL"
    
    sentiment = {
        "timestamp": timestamp or datetime.now().isoformat(),
        "bullish_count": int(bullish_count),
        "bearish_count": int(bearish_count),
        "hold_count": int(hold_count),
        "bullish_ratio": float(bullish_ratio),
        "bearish_ratio": float(bearish_ratio),
        "market_regime": regime,
        "regime_signal": regime_signal,
        "interpretation": get_regime_interpretation(regime, bullish_ratio, bearish_ratio)
    }
    
    return sentiment


def get_regime_interpretation(regime, bullish_ratio, bearish_ratio):
    """Get interpretation text for market regime"""
    
    if regime == "STRONG_UPTREND":
        return f"Market is strongly bullish ({bullish_ratio*100:.1f}%). Focus on BUY signals. SELL signals should be avoided."
    elif regime == "STRONG_DOWNTREND":
        return f"Market is strongly bearish ({bearish_ratio*100:.1f}%). Focus on SELL signals. BUY signals should be avoided."
    elif regime == "UPTREND":
        return f"Market shows uptrend bias ({bullish_ratio*100:.1f}% bullish). Prefer BUY signals."
    elif regime == "DOWNTREND":
        return f"Market shows downtrend bias ({bearish_ratio*100:.1f}% bearish). Prefer SELL signals."
    else:
        return f"Market is mixed ({bullish_ratio*100:.1f}% bullish, {bearish_ratio*100:.1f}% bearish). Be selective."


def calculate_relative_strength(predictions, market_sentiment):
    """
    Calculate each ticker's relative strength vs market average
    
    Args:
        predictions: List of prediction dicts
        market_sentiment: Market sentiment analysis
    
    Returns:
        dict: Relative strength by ticker
    """
    
    df = pd.DataFrame(predictions)
    
    # Average metrics
    avg_accuracy = df['model_accuracy'].mean()
    avg_confidence = df['confidence'].mean()
    
    relative_strength = {}
    
    for _, row in df.iterrows():
        ticker = row['ticker']
        
        # Calculate relative metrics
        accuracy_vs_avg = row['model_accuracy'] - avg_accuracy
        confidence_vs_avg = row['confidence'] - avg_confidence
        
        # Relative strength score (-1 to +1)
        # Fix: Use standard deviation for normalization if available, else fallback
        std_accuracy = df['model_accuracy'].std() if len(df) > 1 else 0.05
        std_confidence = df['confidence'].std() if len(df) > 1 else 0.10
        
        z_accuracy = accuracy_vs_avg / (std_accuracy + 1e-6)
        z_confidence = confidence_vs_avg / (std_confidence + 1e-6)
        
        # Combine Z-scores (weighted)
        strength_score = (z_accuracy * 0.6) + (z_confidence * 0.4)
        
        # Normalize to roughly -1 to 1 range (assuming Z-scores mostly within -3 to 3)
        strength_score = np.clip(strength_score / 3.0, -1.0, 1.0)
        
        # Determine relative strength
        if strength_score > 0.2:
            strength_level = "STRONG"
        elif strength_score > -0.2:
            strength_level = "NEUTRAL"
        else:
            strength_level = "WEAK"
        
        relative_strength[ticker] = {
            "ticker": ticker,
            "accuracy": float(row['model_accuracy']),
            "accuracy_vs_avg": float(accuracy_vs_avg),
            "confidence": float(row['confidence']),
            "confidence_vs_avg": float(confidence_vs_avg),
            "strength_score": float(strength_score),
            "strength_level": strength_level
        }
    
    return relative_strength


def adjust_recommendations(predictions, market_sentiment, relative_strength):
    """
    Adjust trading recommendations based on market regime and relative strength
    
    Args:
        predictions: List of prediction dicts
        market_sentiment: Market sentiment analysis
        relative_strength: Relative strength by ticker
    
    Returns:
        List of adjusted predictions with phase2 recommendations
    """
    
    adjusted = []
    regime = market_sentiment['market_regime']
    
    for pred in predictions:
        ticker = pred['ticker']
        strength = relative_strength.get(ticker, {})
        
        # Original action
        original_action = pred.get('action', 'SKIP')
        
        # Adjust based on market regime
        if regime in ["STRONG_UPTREND", "UPTREND"]:
            # Prefer BUY signals
            if "Bullish" in pred.get('direction', ''):
                phase2_action = "BUY"
            else:
                phase2_action = "SELL_CAUTION"
        elif regime in ["STRONG_DOWNTREND", "DOWNTREND"]:
            # Prefer SELL signals
            if "Bearish" in pred.get('direction', ''):
                phase2_action = "SELL"
            else:
                phase2_action = "BUY_CAUTION"
        else:  # MIXED
            # Only execute high-accuracy trades
            if original_action == "EXECUTE" and strength.get('strength_level') == "STRONG":
                phase2_action = "EXECUTE"
            else:
                phase2_action = "SKIP"
        
        # Add phase 2 data
        pred['phase2'] = {
            "market_regime": regime,
            "market_signal": market_sentiment['regime_signal'],
            "phase2_action": phase2_action,
            "strength_level": strength.get('strength_level', 'UNKNOWN'),
            "strength_score": float(strength.get('strength_score', 0)),
            "recommendation": get_phase2_recommendation(
                original_action, phase2_action, strength, regime
            )
        }
        
        adjusted.append(pred)
    
    return adjusted


def get_phase2_recommendation(original_action, phase2_action, strength, regime):
    """Generate recommendation text"""
    
    if phase2_action == "EXECUTE" or phase2_action == "BUY" or phase2_action == "SELL":
        return f"✅ EXECUTE - {phase2_action} ({regime} environment, {strength.get('strength_level', '?')} strength)"
    elif phase2_action == "BUY_CAUTION":
        return f"⚠️ CAUTION - BUY signal but {regime} environment - wait for confirmation"
    elif phase2_action == "SELL_CAUTION":
        return f"⚠️ CAUTION - SELL signal but {regime} environment - wait for confirmation"
    else:
        return f"⏸️ SKIP - Low confidence or misaligned with {regime}"


def generate_phase2_report(predictions, market_sentiment, relative_strength, adjusted_predictions, timestamp=None):
    """Generate comprehensive Phase 2 report"""
    
    report = {
        "timestamp": timestamp or datetime.now().isoformat(),
        "market_sentiment": market_sentiment,
        "relative_strength": relative_strength,
        "adjusted_actions": {
            ticker: pred.get('phase2', {}) 
            for ticker, pred in [(p['ticker'], p) for p in adjusted_predictions]
        }
    }
    
    return report


def generate_markdown_report(market_sentiment, relative_strength, adjusted_predictions, timestamp=None):
    """Generate human-readable markdown report"""
    
    regime = market_sentiment['market_regime']
    regime_signal = market_sentiment['regime_signal']
    
    markdown = f"""# 📈 Phase 2: Market Environment Analysis

**Generated:** {timestamp or datetime.now().isoformat()}

## 🎯 Market Regime

| Metric | Value |
|--------|-------|
| **Regime** | {regime} {regime_signal} |
| **Bullish** | {market_sentiment['bullish_count']}/10 ({market_sentiment['bullish_ratio']*100:.1f}%) |
| **Bearish** | {market_sentiment['bearish_count']}/10 ({market_sentiment['bearish_ratio']*100:.1f}%) |
| **Hold** | {market_sentiment['hold_count']}/10 ({(1-market_sentiment['bullish_ratio']-market_sentiment['bearish_ratio'])*100:.1f}%) |

## 📊 Interpretation

{market_sentiment['interpretation']}

---

## 🏆 Relative Strength Analysis

| Rank | Ticker | Accuracy | Strength | Signal |
|------|--------|----------|----------|--------|
"""
    
    # Sort by strength score
    sorted_strength = sorted(
        relative_strength.items(),
        key=lambda x: x[1]['strength_score'],
        reverse=True
    )
    
    for rank, (ticker, data) in enumerate(sorted_strength[:5], 1):
        emoji = "🟢" if data['strength_level'] == "STRONG" else "🟡" if data['strength_level'] == "NEUTRAL" else "🔴"
        markdown += f"| {rank} | {ticker} | {data['accuracy']*100:.2f}% | {emoji} {data['strength_level']} | {data['strength_score']:+.2f} |\n"
    
    markdown += f"""

---

## 🎲 Adjusted Trading Recommendations

### 🟢 EXECUTE
"""
    
    execute_preds = [p for p in adjusted_predictions if p['phase2']['phase2_action'] in ['EXECUTE', 'BUY', 'SELL']]
    if execute_preds:
        for pred in execute_preds:
            direction = "📈 BUY" if "Bullish" in pred['direction'] else "📉 SELL"
            markdown += f"- **{pred['ticker']}** {direction} | Conf: {pred['confidence']*100:.2f}% | Strength: {pred['phase2']['strength_level']}\n"
            markdown += f"  → {pred['phase2']['recommendation']}\n"
    else:
        markdown += "No execute signals at this time.\n"
    
    markdown += f"""

### ⚠️ CAUTION
"""
    
    caution_preds = [p for p in adjusted_predictions if 'CAUTION' in p['phase2']['phase2_action']]
    if caution_preds:
        for pred in caution_preds:
            markdown += f"- **{pred['ticker']}** - {pred['phase2']['recommendation']}\n"
    else:
        markdown += "No cautionary signals.\n"
    
    markdown += f"""

### ⏸️ SKIP
"""
    
    skip_preds = [p for p in adjusted_predictions if p['phase2']['phase2_action'] == 'SKIP']
    if skip_preds:
        for pred in skip_preds:
            markdown += f"- **{pred['ticker']}** - {pred['phase2']['recommendation']}\n"
    else:
        markdown += "All predictions are actionable.\n"
    
    markdown += f"""

---

## 🎯 Trading Strategy by Market Regime

### Current: {regime}

**Strategy:**
{get_regime_strategy(regime)}

---

*Phase 2 Report - Market Environment Detection*
*Next: Phase 3 - Backtesting & Performance Tracking*
"""
    
    return markdown


def get_regime_strategy(regime):
    """Get trading strategy for current regime"""
    
    strategies = {
        "STRONG_UPTREND": "• Focus on BUY signals\n• Avoid SELL signals\n• Use tight stops\n• Scale into winners",
        "UPTREND": "• Prefer BUY over SELL\n• Look for pullback entries\n• Partial profit taking\n• Trailing stops",
        "MIXED": "• Be selective\n• High-accuracy trades only\n• Reduce position size\n• Wait for clarity",
        "DOWNTREND": "• Prefer SELL over BUY\n• Look for bounce exits\n• Scale out of losers\n• Tight risk management",
        "STRONG_DOWNTREND": "• Focus on SELL signals\n• Avoid BUY signals\n• Use tight stops\n• Scale into shorts"
    }
    
    return strategies.get(regime, "Market direction unclear - wait for signals")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str, help='Timestamp from main pipeline')
    args = parser.parse_args()
    
    run_timestamp = args.timestamp if args.timestamp else datetime.now().isoformat()
    
    print("="*70)
    print("Phase 2: Market Environment Detection")
    print("="*70)
    
    # Load predictions
    predictions_file = f"{PREDICTIONS_DIR}/latest_predictions.json"
    
    if not Path(predictions_file).exists():
        print(f"[ERROR] {predictions_file} not found")
        return 1
    
    print(f"\n[1/5] Loading predictions...")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    print(f"  ✓ Loaded {len(predictions)} predictions")
    
    # Analyze market sentiment
    print(f"\n[2/5] Analyzing market sentiment...")
    market_sentiment = analyze_market_sentiment(predictions, timestamp=run_timestamp)
    print(f"  ✓ Regime: {market_sentiment['market_regime']} {market_sentiment['regime_signal']}")
    print(f"  ✓ Bullish: {market_sentiment['bullish_ratio']*100:.1f}%")
    
    # Calculate relative strength
    print(f"\n[3/5] Calculating relative strength...")
    relative_strength = calculate_relative_strength(predictions, market_sentiment)
    print(f"  ✓ Analyzed {len(relative_strength)} tickers")
    
    # Adjust recommendations
    print(f"\n[4/5] Adjusting recommendations based on market regime...")
    adjusted_predictions = adjust_recommendations(predictions, market_sentiment, relative_strength)
    print(f"  ✓ Recommendations adjusted")
    
    # Generate reports
    print(f"\n[5/5] Generating reports...")
    
    # JSON report
    phase2_report = generate_phase2_report(predictions, market_sentiment, relative_strength, adjusted_predictions, timestamp=run_timestamp)
    report_file = f"{PREDICTIONS_DIR}/phase2_analysis.json"
    with open(report_file, 'w') as f:
        json.dump(phase2_report, f, indent=2, default=str)
    print(f"  ✓ Saved: {report_file}")
    
    # Markdown report
    md_report = generate_markdown_report(market_sentiment, relative_strength, adjusted_predictions, timestamp=run_timestamp)
    md_file = f"{PREDICTIONS_DIR}/phase2_analysis.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"  ✓ Saved: {md_file}")
    
    # Save adjusted predictions
    adjusted_file = f"{PREDICTIONS_DIR}/predictions_with_phase2.json"
    with open(adjusted_file, 'w') as f:
        json.dump(adjusted_predictions, f, indent=2, default=str)
    print(f"  ✓ Saved: {adjusted_file}")
    
    print("\n" + "="*70)
    print("Phase 2 Analysis Complete")
    print("="*70)
    print(f"\nMarket Regime: {market_sentiment['market_regime']}")
    print(f"Interpretation: {market_sentiment['interpretation']}")
    print("\nFiles generated:")
    print(f"  - phase2_analysis.json")
    print(f"  - phase2_analysis.md")
    print(f"  - predictions_with_phase2.json")
    
    return 0


if __name__ == "__main__":
    exit(main())
