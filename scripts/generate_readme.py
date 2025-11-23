#!/usr/bin/env python3
# NOROSHI README Auto Generator #KGNINJA

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PRED_FILE = BASE / "data" / "daily_predictions" / "latest_predictions.json"
README = BASE / "README.md"

def update_readme():
    payload = json.loads(PRED_FILE.read_text(encoding="utf-8"))
    us = payload["markets"]["US"]
    jp = payload["markets"]["JP"]

    md = []
    md.append("# NOROSHI Auto Stock Prediction #KGNINJA\n")
    md.append(f"Updated: **{payload['timestamp']}**\n")

    md.append("## US Market\n")
    for x in us:
        md.append(f"- **{x['ticker']}** → {x['predicted_change_pct']:.2f}% ({x['trend']})")

    md.append("\n## Japan Market\n")
    for x in jp:
        md.append(f"- **{x['ticker']}** → {x['predicted_change_pct']:.2f}% ({x['trend']})")

    README.write_text("\n".join(md), encoding="utf-8")
    print("README Updated #KGNINJA")

if __name__ == "__main__":
    update_readme()
