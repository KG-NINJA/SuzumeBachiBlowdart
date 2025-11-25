"""
confidence_filter.py - Add confidence-based filtering to predictions
Converts low-confidence predictions to HOLD recommendations
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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


def generate_confidence_report(filtered_predictions):
    """Generate a confidence analysis report"""

    if not filtered_predictions or len(filtered_predictions) == 0:
        return {
            "total_predictions": 0,
            "average_confidence": 0,
            "execute_count": 0,
            "skip_count": 0,
            "note": "No predictions"
        }

    confidences = [p.get('confidence', 0) for p in filtered_predictions]

    return {
        "total_predictions": len(filtered_predictions),
        "average_confidence": float(np.mean(confidences)),
        "execute_count": sum(1 for p in filtered_predictions if p.get('action') == 'EXECUTE'),
        "skip_count": sum(1 for p in filtered_predictions if p.get('action') == 'SKIP'),
        "confidence_distribution": {
            "high": sum(1 for c in confidences if c >= 0.7),
            "medium": sum(1 for c in confidences if 0.4 <= c < 0.7),
            "low": sum(1 for c in confidences if c < 0.4)
        }
    }
