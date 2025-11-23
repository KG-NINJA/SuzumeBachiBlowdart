import json, os, matplotlib.pyplot as plt
from datetime import datetime

LOG_DIR = "logs"
OUT_DIR = "docs/images"
os.makedirs(OUT_DIR, exist_ok=True)

logs = []

for file in sorted(os.listdir(LOG_DIR)):
    if file.endswith(".json"):
        with open(f"{LOG_DIR}/{file}", encoding="utf-8") as f:
            logs.append(json.load(f))

timestamps = [l["timestamp"] for l in logs]
confidence = [l["confidence"] for l in logs]
signal_strength = [l["signal_strength"] for l in logs]

# 信頼スコア
plt.figure(figsize=(12,4))
plt.plot(timestamps, confidence, marker="o")
plt.xticks(rotation=45)
plt.title("NVDA Confidence Trend")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confidence_trend.png")
plt.close()

# シグナル強度
plt.figure(figsize=(12,4))
plt.plot(timestamps, signal_strength, marker="o", color="orange")
plt.xticks(rotation=45)
plt.title("Signal Strength Trend")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/signal_trend.png")
plt.close()
