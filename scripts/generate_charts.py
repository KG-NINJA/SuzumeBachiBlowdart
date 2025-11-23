#!/usr/bin/env python3
# NOROSHI Chart Generator #KGNINJA

import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
PRED_FILE = BASE / "data" / "daily_predictions" / "latest_predictions.json"
OUT = BASE / "analytics"
OUT.mkdir(exist_ok=True)

def generate_chart():
    payload = json.loads(PRED_FILE.read_text(encoding="utf-8"))

    us = payload["markets"]["US"]
    jp = payload["markets"]["JP"]

    # US チャート
    tickers = [x["ticker"] for x in us]
    pct = [x["predicted_change_pct"] for x in us]

    plt.figure(figsize=(10,5))
    plt.bar(tickers, pct)
    plt.title("US Market Prediction (%) #KGNINJA")
    plt.savefig(OUT / "us_predictions.png")
    plt.close()

    # Japan チャート
    tickers = [x["ticker"] for x in jp]
    pct = [x["predicted_change_pct"] for x in jp]

    plt.figure(figsize=(10,5))
    plt.bar(tickers, pct, color="orange")
    plt.title("Japan Market Prediction (%) #KGNINJA")
    plt.savefig(OUT / "jp_predictions.png")
    plt.close()

    print("Charts generated #KGNINJA")

if __name__ == "__main__":
    generate_chart()
