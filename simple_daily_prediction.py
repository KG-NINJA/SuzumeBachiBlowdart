"""
Blowdart daily pipeline entrypoint.
- trains XGBoost models for the supported tickers
- generates daily predictions saved as JSON
- integrates CV runner confidence and signal strength as features
"""
from __future__ import annotations

import argparse
import json
from typing import List

from blowdart_ml_engine import BlowdartMLEngine, TICKERS


def run_train(engine: BlowdartMLEngine) -> List[dict]:
    summary = engine.train_all_tickers()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_predict(engine: BlowdartMLEngine) -> List[dict]:
    predictions = engine.predict_all_tickers()
    print(json.dumps(predictions, indent=2, ensure_ascii=False))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Blowdart ML daily runner")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "all"],
        default="all",
        help="Run training, prediction, or both",
    )
    args = parser.parse_args()

    engine = BlowdartMLEngine()

    if args.mode in {"train", "all"}:
        run_train(engine)
    if args.mode in {"predict", "all"}:
        run_predict(engine)


if __name__ == "__main__":
    main()
