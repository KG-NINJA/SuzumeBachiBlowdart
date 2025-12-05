import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simple_daily_prediction

def verify_outputs():
    print(">>> Verifying outputs...")
    
    pred_dir = "daily_predictions"
    files = [
        f"{pred_dir}/confidence_report.json",
        f"{pred_dir}/confidence_report.md",
        f"{pred_dir}/latest_predictions.json",
        f"{pred_dir}/phase2_analysis.json"
    ]
    
    # Check existence
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Missing file: {f}")
            return
    
    # Load JSONs
    with open(files[0], encoding='utf-8') as f: conf_report = json.load(f)
    with open(files[2], encoding='utf-8') as f: predictions = json.load(f)
    with open(files[3], encoding='utf-8') as f: phase2 = json.load(f)
    
    # 1. Check Timestamps
    t1 = phase2['timestamp']
    # confidence_report might not have a top-level timestamp in my implementation, let's check
    # predictions is a list, so no top-level timestamp
    
    print(f"Timestamp (Phase 2): {t1}")
    
    # 2. Check Confidence Scale
    conf_scores = [p['confidence'] for p in predictions]
    avg_conf = sum(conf_scores) / len(conf_scores)
    print(f"Average Confidence (JSON): {avg_conf:.4f}")
    
    if avg_conf > 0.6:
        print("⚠️ Average confidence > 0.6. Did you fix the scale? (Should be around 0.0-0.5 for score, or 0.5-1.0 for probability?)")
        # Wait, I changed it to "Score 0-1". 
        # calculate_confidence_score returns abs(prob-0.5)*2.
        # But `apply_confidence_filter` stores `pred['confidence']` as the original probability?
        # Let's check confidence_filter.py again.
        pass

    # 3. Check Accuracy
    accuracies = [p['model_accuracy'] for p in predictions]
    avg_acc = sum(accuracies) / len(accuracies)
    print(f"Average Accuracy: {avg_acc:.4f}")
    if avg_acc > 0.95:
        print("❌ Accuracy still > 95%. Leakage fix failed?")
    else:
        print("✅ Accuracy looks realistic.")

    # 4. Check Strength Levels
    strength_levels = [p['phase2']['strength_level'] for p in phase2['adjusted_actions'].values()]
    print(f"Strength Levels: {strength_levels}")
    if all(s == 'NEUTRAL' for s in strength_levels):
        print("❌ All strength levels are NEUTRAL.")
    else:
        print("✅ Strength levels are varied.")

if __name__ == "__main__":
    # Monkeypatch to run only for NVDA
    simple_daily_prediction.TICKERS = ["NVDA", "AMD"] # Run 2 to check relative strength
    
    print(">>> Running pipeline for NVDA, AMD...")
    simple_daily_prediction.main()
    
    verify_outputs()


