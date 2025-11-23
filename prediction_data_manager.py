#!/usr/bin/env python3
# Prediction Data Manager #KGNINJA

from pathlib import Path
import json
from datetime import datetime

BASE = Path(__file__).resolve().parents[0]
OUT = BASE / "data" / "daily_predictions"
OUT.mkdir(parents=True, exist_ok=True)

class PredictionDataManager:

    @staticmethod
    def save_latest(payload):
        latest = OUT / "latest_predictions.json"
        latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # 履歴保存
        history_dir = OUT / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (history_dir / f"{ts}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
