"""
simple_daily_prediction.py - Dual Engine Support
フラグ一つで ML エンジン v1 or v2 を切り替え可能
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from blowdart_features import build_feature_set
from pipeline_safety import (
    apply_confidence_decisions,
    classify_run_status,
    normalize_prediction_probabilities,
    should_fail_run,
)

# ===== エンジン選択フラグ =====
USE_V2_ENGINE = True  # ← True で v2, False で v1 を使用

# エンジン インポート
try:
    from ml_engine_v2 import predict_ticker_v2, train_model_v2

    v2_available = True
except ImportError:
    logging.warning("ml_engine_v2 not available, falling back to v1")
    v2_available = False

from blowdart_ml_engine import predict_ticker, train_model
from confidence_filter import generate_confidence_markdown, generate_confidence_report

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_daily_prediction")

# ディレクトリ設定
PREDICTIONS_DIR = Path("daily_predictions")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================
# 共通ヘルパー
# ===========================================================

def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def _make_error_record(ticker: str, stage: str, error: BaseException) -> dict:
    message = str(error).strip() or error.__class__.__name__
    return {
        "symbol": ticker,
        "stage": stage,
        "error_type": error.__class__.__name__,
        "message": message[:1000],
        "timestamp": datetime.now().isoformat(),
    }


# ===========================================================
# エンジンマネージャー
# ===========================================================

class DualEngineManager:
    """V1 と V2 のエンジンを統一インターフェースで管理"""

    def __init__(self, use_v2: bool = True):
        self.use_v2 = use_v2 and v2_available
        self.version = "v2" if self.use_v2 else "v1"
        logger.info(f"[ENGINE] Using {self.version.upper()} engine")

    def train(self, df: pd.DataFrame, ticker: str, **kwargs) -> dict:
        """Train one ticker and return a normalized result dictionary."""

        if self.use_v2:
            optimize_hyperparams = kwargs.get('optimize_hyperparams', True)
            n_optuna_trials = kwargs.get('n_optuna_trials', 50)

            result = train_model_v2(
                df,
                ticker,
                optimize_hyperparams=optimize_hyperparams,
                n_optuna_trials=n_optuna_trials,
            )
        else:
            # The legacy engine historically returned a numeric accuracy rather
            # than a result dictionary. Normalize both possible return shapes.
            legacy_result = train_model(df, ticker, use_existing=True)
            if isinstance(legacy_result, dict):
                result = legacy_result
            else:
                try:
                    accuracy = float(legacy_result)
                except (TypeError, ValueError):
                    accuracy = 0.0
                result = {
                    "ok": accuracy > 0.0,
                    "ticker": ticker,
                    "hybrid_acc": accuracy,
                    "error": None if accuracy > 0.0 else "legacy_training_failed",
                }

        if not isinstance(result, dict):
            result = {
                "ok": False,
                "ticker": ticker,
                "error": "training_result_is_not_a_dictionary",
            }

        result['engine_version'] = self.version
        return result

    def predict(self, ticker: str, df: pd.DataFrame) -> dict | None:
        """Predict one ticker and normalize fields shared by both engines."""

        if self.use_v2:
            result = predict_ticker_v2(ticker, df)
        else:
            result = predict_ticker(ticker, df)

        if not result:
            return None
        if not isinstance(result, dict):
            raise TypeError("prediction result is not a dictionary")

        result = dict(result)
        result['engine_version'] = self.version

        # V1 calls the target price "forecast"; downstream code uses one name.
        if result.get('predicted_price') is None and result.get('forecast') is not None:
            result['predicted_price'] = result['forecast']

        # V2's raw positive-class probability is ensemble_pred. Normalizing here
        # fixes the historical bearish prob_up/prob_down reversal and prevents a
        # confidence score from being transformed a second time.
        return normalize_prediction_probabilities(result)


# ===========================================================
# メイン: 日次予測パイプライン
# ===========================================================

def run_daily_prediction(
    tickers: list,
    price_fetcher,
    use_v2: bool = USE_V2_ENGINE,
    optimize_hyperparams: bool = True,
    min_confidence: float = 0.30,
) -> dict:
    """Run the daily prediction pipeline and return a diagnostic summary."""

    start_time = datetime.now()
    run_id = start_time.strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("[PIPELINE] Starting daily prediction run")
    logger.info(f"[PIPELINE] Engine Version: {'V2 (Advanced)' if use_v2 else 'V1 (Classic)'}")
    logger.info(f"[PIPELINE] Tickers: {len(tickers)} symbols")
    logger.info("=" * 70)

    engine = DualEngineManager(use_v2=use_v2)

    all_predictions: list[dict] = []
    filtered_predictions: list[dict] = []
    training_results: list[dict] = []
    failed_tickers: list[str] = []
    errors: list[dict] = []

    def record_failure(ticker: str, stage: str, error: BaseException) -> None:
        if ticker not in failed_tickers:
            failed_tickers.append(ticker)
        error_record = _make_error_record(ticker, stage, error)
        errors.append(error_record)
        logger.error(
            f"[{ticker}] Failed at {stage}: "
            f"{error_record['error_type']}: {error_record['message']}"
        )

    for ticker in tickers:
        stage = "initialization"
        try:
            logger.info(f"\n[{ticker}] Processing...")

            stage = "price_fetch"
            price_data = price_fetcher(ticker)
            if price_data is None or price_data.empty:
                raise ValueError("price_fetcher returned no price rows")
            logger.info(f"[{ticker}] Fetched {len(price_data)} price records")

            stage = "feature_engineering"
            features = build_feature_set(
                price_data,
                ticker,
                raise_on_error=True,
            )
            if features is None or features.empty:
                raise ValueError("feature engineering returned no usable rows")
            logger.info(f"[{ticker}] Built {features.shape[1]} features")

            stage = "training"
            kwargs = {}
            if use_v2:
                kwargs['optimize_hyperparams'] = optimize_hyperparams
                kwargs['n_optuna_trials'] = 50

            train_result = engine.train(features, ticker, **kwargs)
            training_results.append(train_result)
            if not train_result.get('ok', False):
                raise RuntimeError(
                    f"model training failed: {train_result.get('error', 'unknown error')}"
                )

            logger.info(
                f"[{ticker}] Training complete: "
                f"Accuracy={train_result.get('ensemble_accuracy', train_result.get('hybrid_acc', 'N/A'))}"
            )

            stage = "prediction"
            prediction = engine.predict(ticker, features)
            if prediction is None:
                raise RuntimeError("prediction engine returned no result")
            if prediction.get('predicted_price') is None:
                raise ValueError("prediction has no predicted_price/forecast")

            stage = "confidence_decision"
            decision = apply_confidence_decisions(
                [prediction],
                min_confidence=min_confidence,
            )[0]

            predicted_price = float(prediction['predicted_price'])
            confidence_score = float(prediction['confidence_score'])

            logger.info(
                f"[{ticker}] Prediction: "
                f"{prediction.get('direction', '?')} @ "
                f"{predicted_price:.2f} "
                f"(confidence_score={confidence_score:.2%}, "
                f"action={decision['action']})"
            )

            all_predictions.append(prediction)
            filtered_predictions.append(decision)

        except Exception as error:
            logger.debug(f"[{ticker}] Exception details", exc_info=True)
            record_failure(ticker, stage, error)
            continue

    logger.info(
        f"\n[RESULTS] Successful inferences: "
        f"{len(all_predictions)}/{len(tickers)}"
    )

    report = generate_confidence_report(filtered_predictions)
    execute_count = int(report.get('execute_count', 0))
    skip_count = int(report.get('skip_count', 0))

    status = classify_run_status(
        attempted_symbols=len(tickers),
        successful_inferences=len(all_predictions),
        executable_signals=execute_count,
        failed_symbols=len(failed_tickers),
    )
    workflow_should_fail = should_fail_run(status)

    report.update(
        {
            "pipeline_status": status,
            "workflow_should_fail": workflow_should_fail,
            "attempted_symbols": len(tickers),
            "successful_inferences": len(all_predictions),
            "failed_symbols": len(failed_tickers),
            "failed_tickers": failed_tickers,
        }
    )

    markdown = generate_confidence_markdown(report, filtered_predictions)
    markdown = (
        f"> Pipeline status: **{status}** | "
        f"inferences: {len(all_predictions)}/{len(tickers)} | "
        f"execute: {execute_count} | skip: {skip_count} | "
        f"failed: {len(failed_tickers)}\n\n"
        + markdown
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    symbol_outcomes = [
        {
            "symbol": prediction.get('ticker', '?'),
            "result": "PREDICTED" if prediction.get('action') == 'EXECUTE' else "NO_SIGNAL",
            "action": prediction.get('action'),
            "direction": prediction.get('direction'),
            "prob_up": prediction.get('prob_up'),
            "confidence_score": prediction.get('confidence_score'),
            "confidence_level": prediction.get('confidence_level'),
        }
        for prediction in filtered_predictions
    ]
    symbol_outcomes.extend(
        {
            "symbol": error['symbol'],
            "result": "FAILED",
            "stage": error['stage'],
            "error_type": error['error_type'],
            "message": error['message'],
        }
        for error in errors
    )

    summary = {
        "schema_version": "2.0",
        "run_id": run_id,
        "timestamp": end_time.isoformat(),
        "started_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
        "status": status,
        "workflow_should_fail": workflow_should_fail,
        "engine_version": engine.version.upper(),
        "model_version": engine.version,
        "min_confidence": min_confidence,
        "attempted_symbols": len(tickers),
        "successful_inferences": len(all_predictions),
        "executable_signals": execute_count,
        "abstained_low_confidence": skip_count,
        "failed_symbols": len(failed_tickers),
        "duration_seconds": duration,
        "failed_tickers": failed_tickers,
        "errors": errors,
        "symbol_outcomes": symbol_outcomes,
        # Backward-compatible aliases used by existing dashboards/scripts.
        "total_tickers": len(tickers),
        "successful": len(all_predictions),
        "failed": len(failed_tickers),
        "execute_count": execute_count,
        "skip_count": skip_count,
        "average_confidence": report.get('average_confidence', 0),
    }

    # Save prediction artifacts even when the run is going to fail. The workflow
    # commits these diagnostics before marking the job red.
    predictions_file = PREDICTIONS_DIR / f"predictions_{run_id}.json"
    _write_json(predictions_file, all_predictions)
    _write_json(PREDICTIONS_DIR / "latest_predictions.json", all_predictions)
    logger.info(f"[SAVE] Predictions → {predictions_file}")

    filtered_file = PREDICTIONS_DIR / f"filtered_predictions_{run_id}.json"
    _write_json(filtered_file, filtered_predictions)
    logger.info(f"[SAVE] Filtered predictions → {filtered_file}")

    report_file = PREDICTIONS_DIR / f"confidence_report_{run_id}.json"
    _write_json(report_file, report)
    _write_json(PREDICTIONS_DIR / "confidence_report.json", report)
    logger.info(f"[SAVE] Report (JSON) → {report_file}")

    markdown_file = PREDICTIONS_DIR / f"confidence_report_{run_id}.md"
    markdown_file.write_text(markdown, encoding="utf-8")
    (PREDICTIONS_DIR / "confidence_report.md").write_text(markdown, encoding="utf-8")
    logger.info(f"[SAVE] Report (MD) → {markdown_file}")

    training_log_file = LOGS_DIR / f"training_log_{run_id}.json"
    _write_json(training_log_file, training_results)
    logger.info(f"[SAVE] Training log → {training_log_file}")

    diagnostics_file = PREDICTIONS_DIR / f"run_diagnostics_{run_id}.json"
    _write_json(diagnostics_file, summary)
    _write_json(PREDICTIONS_DIR / "run_diagnostics.json", summary)
    logger.info(f"[SAVE] Run diagnostics → {diagnostics_file}")

    logger.info("=" * 70)
    logger.info(f"[SUMMARY] Status: {status}")
    logger.info(
        f"[SUMMARY] Inferences: {summary['successful_inferences']}/"
        f"{summary['attempted_symbols']}"
    )
    logger.info(
        f"[SUMMARY] Execute: {execute_count} | Skip: {skip_count} | "
        f"Failed: {summary['failed_symbols']}"
    )
    logger.info(f"[SUMMARY] Avg Confidence: {summary['average_confidence']:.1%}")
    logger.info(f"[SUMMARY] Duration: {duration:.1f}s")
    logger.info(f"[SUMMARY] Engine: {engine.version.upper()}")
    logger.info("=" * 70)

    return summary


# ===========================================================
# 使用例
# ===========================================================

if __name__ == "__main__":
    import os

    import yfinance as yf

    use_v2_env = os.environ.get('USE_V2', '1').lower() == '1'
    tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'TSLA']

    def fetch_price_data(ticker, period='1y'):
        try:
            return yf.download(ticker, period=period, progress=False)
        except Exception as error:
            logger.error(f"Failed to fetch {ticker}: {error}")
            raise

    summary = run_daily_prediction(
        tickers=tickers,
        price_fetcher=fetch_price_data,
        use_v2=use_v2_env,
        optimize_hyperparams=False,
        min_confidence=0.30,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary['workflow_should_fail']:
        sys.exit(1)
