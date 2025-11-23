#!/usr/bin/env python3
"""NVDA向けCONFIG評価ランナー（GitHub Actions対応版）"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# === CONFIG ===
CONFIG: Dict[str, Any] = {
    "target_ticker": "NVDA",

    # GitHub Actions では continuous_*** のフォルダが存在しないため
    # すべて logs/ に統一して "cv_run_*.json" を取得する形へ変更
    "cv_sources": [
        {
            "name": "cv_runs",
            "type": "validation",
            "directory": "logs",
            "weight": 1.0,
            "max_files": 10,
        },
    ],

    "confidence_weights": {
        "cv": 0.5,
        "mape": 0.2,
        "signal": 0.3,
    },

    "signal_rule": {
        "predictions_file": "data/daily_predictions/latest_predictions.json",
        "change_scale_pct": 5.0,
        "buy_threshold_pct": 1.0,
        "sell_threshold_pct": -1.0,
    },

    "required": {
        "cv_average": 60.0,
        "high_confidence": 65.0,
    },
}


# === データ構造 ===
@dataclass
class CVComponent:
    name: str
    score: float
    weight: float


# === ユーティリティ ===
def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def gather_validation_component(project_root: Path, source: Dict[str, Any], ticker: str) -> Tuple[Optional[CVComponent], Optional[float]]:
    directory = project_root / source.get("directory", "logs")
    if not directory.exists():
        return (None, None)

    pattern = f"cv_run_"
    files = sorted(
        [p for p in directory.glob("*.json") if p.name.startswith(pattern)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    selected = files[: int(source.get("max_files", 3))]
    if not selected:
        return (None, None)

    cv_scores, mapes = [], []

    for path in selected:
        payload = load_json(path)
        if not payload:
            continue

        # cv_average そのものを読む
        score = payload.get("cv_average")
        if score is None:
            continue

        cv_scores.append(float(score))

        # mape はない場合がある → デフォルト扱い
        mape = 100 - float(score)
        mapes.append(mape)

    if not cv_scores:
        return (None, None)

    avg_score = mean(cv_scores)
    avg_mape = mean(mapes)
    component = CVComponent(source["name"], avg_score, float(source["weight"]))

    return (component, avg_mape)


def normalize_weights(components: List[CVComponent]) -> List[CVComponent]:
    total = sum(c.weight for c in components)
    if total == 0:
        return components
    return [CVComponent(c.name, c.score, c.weight / total) for c in components]


def calculate_signal(project_root: Path, cfg: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    predictions_file = project_root / cfg["predictions_file"]
    payload = load_json(predictions_file) or {}

    markets = payload.get("markets", {})
    entries = []
    for market in markets.values():
        entries.extend(market)

    entry = next((x for x in entries if x.get("ticker") == ticker), None)
    if not entry:
        return {"status": "missing"}

    change_pct = float(entry.get("predicted_change_pct", 0.0))
    trend = entry.get("trend", "N/A")
    method = entry.get("prediction_method", "unknown")
    scale = max(0.1, float(cfg.get("change_scale_pct", 5.0)))
    normalized = max(-1.0, min(1.0, change_pct / scale))
    strength_pct = (normalized + 1.0) / 2.0 * 100.0

    if change_pct >= cfg["buy_threshold_pct"]:
        signal = "BUY"
    elif change_pct <= cfg["sell_threshold_pct"]:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "status": "ok",
        "predicted_change_pct": change_pct,
        "trend": trend,
        "prediction_method": method,
        "signal": signal,
        "signal_strength": strength_pct,
    }


def print_section(title: str):
    print(f"\n=== {title} ===")


# === メイン ===
def main() -> None:
    # GitHub Actions ではこれが正しい
    project_root = Path(__file__).resolve().parent

    ticker = CONFIG["target_ticker"]

    cv_components: List[CVComponent] = []
    validation_mape: Optional[float] = None

    for source in CONFIG["cv_sources"]:
        c, m = gather_validation_component(project_root, source, ticker)
        if c:
            cv_components.append(c)
        if m is not None:
            validation_mape = m if validation_mape is None else (validation_mape + m) / 2

    if not cv_components:
        print("CV情報が見つかりません。（logs フォルダを確認してください）")
        return

    cv_components = normalize_weights(cv_components)
    cv_average = sum(c.score * c.weight for c in cv_components)

    print_section("CV Components")
    for c in cv_components:
        print(f"- {c.name}: {c.score:.2f}% (weight {c.weight:.2f})")
    print(f"\n=> 加重CV平均: {cv_average:.2f}%")

    signal_info = calculate_signal(project_root, CONFIG["signal_rule"], ticker)
    signal_strength = signal_info.get("signal_strength", 50.0)
    mape_component = 100.0 - validation_mape if validation_mape is not None else 50.0

    conf_cfg = CONFIG["confidence_weights"]
    conf_total = sum(conf_cfg.values())
    confidence = (
        cv_average * conf_cfg["cv"]
        + mape_component * conf_cfg["mape"]
        + signal_strength * conf_cfg["signal"]
    ) / conf_total

    print_section("Confidence & Signal")
    print(f"- 最新MAPEスコア: {mape_component:.2f}%")
    print(f"- シグナル: {signal_info.get('signal')} (強度 {signal_strength:.1f}%)")
    print(f"\n=> 信頼スコア: {confidence:.2f}%")

    req = CONFIG["required"]
    print_section("Goal Check")
    print(f"- CV平均 >= {req['cv_average']}% : {'PASS' if cv_average >= req['cv_average'] else 'FAIL'}")
    print(f"- 高確信 >= {req['high_confidence']}% : {'PASS' if confidence >= req['high_confidence'] else 'FAIL'}")

    save_run_log(cv_average, confidence, signal_info, signal_strength)


# === JSONログ保存機能 ===
def save_run_log(cv_average=None, confidence=None, signal_info=None, signal_strength=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "cv_average": round(cv_average or 0, 2),
        "confidence": round(confidence or 0, 2),
        "signal": signal_info.get("signal") if signal_info else None,
        "signal_strength": round(signal_strength or 0, 2),
    }

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"cv_run_{datetime.now():%Y%m%d_%H%M%S}.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    print(f"[ログ保存完了] {log_path}")


if __name__ == "__main__":
    main()
