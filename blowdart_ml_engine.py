"""
Blowdart machine learning engine powered by XGBoost.
Handles training and prediction for multiple tickers with automatic
feature engineering and JSON outputs.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from blowdart_features import build_feature_set, ensure_directories

TICKERS = [
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "QQQ",
]


class BlowdartMLEngine:
    def __init__(
        self,
        data_dir: Path | str = "data",
        model_dir: Path | str = "models",
        analytics_dir: Path | str = "analytics",
        predictions_dir: Path | str = "daily_predictions",
        tickers: Optional[List[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.analytics_dir = Path(analytics_dir)
        self.predictions_dir = Path(predictions_dir)
        self.docs_data_dir = Path("docs/data")
        self.log_dir = Path("logs")
        self.tickers = tickers or TICKERS

        for path in [self.data_dir, self.model_dir, self.analytics_dir, self.predictions_dir, self.docs_data_dir, self.log_dir]:
            path.mkdir(exist_ok=True)

        ensure_directories()

    def _model_path(self, ticker: str) -> Path:
        return self.model_dir / f"{ticker}_xgb.json"

    def _model_meta_path(self, ticker: str) -> Path:
        return self.model_dir / f"{ticker}_xgb_meta.json"

    def _load_model(self, ticker: str) -> Optional[XGBClassifier]:
        path = self._model_path(ticker)
        if not path.exists():
            return None
        model = XGBClassifier()
        model.load_model(path)
        return model

    def _save_model(self, ticker: str, model: XGBClassifier) -> None:
        model.save_model(self._model_path(ticker))

    def _save_meta(
        self,
        ticker: str,
        feature_cols: List[str],
        accuracy: float,
        params: Dict,
        best_iteration: Optional[int] = None,
    ) -> None:
        meta = {
            "ticker": ticker,
            "features": feature_cols,
            "trained_at": datetime.utcnow().isoformat(),
            "accuracy": accuracy,
            "params": params,
        }
        if best_iteration is not None:
            meta["best_iteration"] = best_iteration
        with self._model_meta_path(ticker).open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

    def _load_meta(self, ticker: str) -> Dict:
        if not self._model_meta_path(ticker).exists():
            return {}
        with self._model_meta_path(ticker).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _log_fetch_error(self, ticker: str, stage: str, error: str) -> None:
        self.log_dir.mkdir(exist_ok=True)
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        target = self.log_dir / f"fetch_errors_{date_str}.json"
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "stage": stage,
            "error": error,
        }

        existing: List[Dict] = []
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append(entry)
        target.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    def _xgb_params(self, ticker: str, train_df: "pd.DataFrame", feature_cols: List[str]) -> Dict:
        """Provide ticker-aware defaults to improve weaker performers while avoiding overfit."""
        base = {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
        }
        overrides = {
            "AAPL": {"max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.95},
            "NFLX": {"n_estimators": 500, "learning_rate": 0.04},
            "GOOGL": {"n_estimators": 450, "max_depth": 4},
        }
        params = {**base, **overrides.get(ticker, {})}

        pos = float(train_df["TARGET"].sum())
        neg = float(len(train_df) - pos)
        if pos > 0 and neg > 0:
            params["scale_pos_weight"] = max(0.5, neg / pos)

        if len(feature_cols) > 60:
            params["max_depth"] = min(params.get("max_depth", 5), 5)
            params["min_child_weight"] = 2
        return params

    def _train_single(self, ticker: str) -> Optional[Dict]:
        try:
            dataset, feature_cols = build_feature_set(
                ticker, use_feature_reduction=True, return_feature_cols=True
            )
        except Exception as exc:
            self._log_fetch_error(ticker, "train", str(exc))
            return None

        if dataset.empty or not feature_cols or len(dataset) < 60:
            self._log_fetch_error(ticker, "train", "empty dataset or missing features")
            return None

        # chronological split: last 20% for validation
        split_idx = int(len(dataset) * 0.8)
        train_df = dataset.iloc[:split_idx]
        test_df = dataset.iloc[split_idx:]

        X_train = train_df[feature_cols]
        y_train = train_df["TARGET"]
        X_test = test_df[feature_cols]
        y_test = test_df["TARGET"]

        params = self._xgb_params(ticker, train_df, feature_cols)
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=40,
            verbose=False,
        )

        best_iter = getattr(model, "best_iteration", None)
        preds = model.predict(X_test)
        accuracy = float(np.mean(preds == y_test)) if len(y_test) else 0.0
        self._save_model(ticker, model)
        self._save_meta(ticker, feature_cols, accuracy, params, best_iter)

        importance = model.get_booster().get_score(importance_type="gain")
        importance_sorted = sorted(
            [{"feature": k, "importance": float(v)} for k, v in importance.items()],
            key=lambda x: x["importance"],
            reverse=True,
        )

        return {
            "ticker": ticker,
            "accuracy": accuracy,
            "feature_cols": feature_cols,
            "feature_importance": importance_sorted,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "scale_pos_weight": params.get("scale_pos_weight"),
            "best_iteration": best_iter,
            "latest_date": dataset["DATE"].max().isoformat(),
        }

    def train_all_tickers(self) -> List[Dict]:
        summary: List[Dict] = []
        for ticker in self.tickers:
            result = self._train_single(ticker)
            if result:
                summary.append(result)
        if summary:
            self._append_training_metrics(summary)
            self._write_feature_importance(summary)
        return summary

    def _append_training_metrics(self, metrics: List[Dict]) -> None:
        path = self.analytics_dir / "training_metrics.json"
        existing: List[Dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        timestamp = datetime.utcnow().isoformat()
        for entry in metrics:
            entry["timestamp"] = timestamp
        path.write_text(json.dumps(existing + metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        docs_path = self.docs_data_dir / "training_metrics.json"
        docs_path.write_text(json.dumps(existing + metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_feature_importance(self, metrics: List[Dict]) -> None:
        importance_map = {
            m["ticker"]: m.get("feature_importance", []) for m in metrics if "ticker" in m
        }
        target_path = self.analytics_dir / "feature_importance.json"
        target_path.write_text(json.dumps(importance_map, ensure_ascii=False, indent=2), encoding="utf-8")
        docs_path = self.docs_data_dir / "feature_importance.json"
        docs_path.write_text(json.dumps(importance_map, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_prediction_entry(
        self, ticker: str, prob_up: float, latest_row: Dict[str, float], method: str
    ) -> Dict:
        direction = "UP" if prob_up >= 0.5 else "DOWN"
        predicted_change_pct = float(latest_row.get("RETURN_1D", 0) * 100)
        return {
            "ticker": ticker,
            "prob_up": prob_up,
            "prob_down": 1 - prob_up,
            "predicted_direction": direction,
            "predicted_change_pct": predicted_change_pct,
            "prediction_method": method,
            "cv_confidence": float(latest_row.get("CV_CONFIDENCE", 0.5)),
            "signal_strength": float(latest_row.get("CV_SIGNAL_STRENGTH", 0.5)),
        }

    def predict_all_tickers(self) -> List[Dict]:
        predictions: List[Dict] = []
        for ticker in self.tickers:
            entry = self._predict_single(ticker)
            if entry:
                predictions.append(entry)
        if predictions:
            self._write_predictions(predictions)
        return predictions

    def _predict_single(self, ticker: str) -> Optional[Dict]:
        meta = self._load_meta(ticker)
        feature_cols = meta.get("features") or []
        try:
            dataset, built_feature_cols = build_feature_set(
                ticker, use_feature_reduction=True, return_feature_cols=True
            )
        except Exception as exc:
            self._log_fetch_error(ticker, "predict", str(exc))
            return None

        if not feature_cols:
            feature_cols = built_feature_cols
        available_cols = [col for col in feature_cols if col in dataset.columns]
        if dataset.empty or not available_cols:
            self._log_fetch_error(ticker, "predict", "empty dataset or missing features")
            return None

        model = self._load_model(ticker)
        if model is None:
            train_result = self._train_single(ticker)
            if not train_result:
                return None
            model = self._load_model(ticker)
            available_cols = [col for col in train_result.get("feature_cols", []) if col in dataset.columns]

        latest_row = dataset.iloc[-1]
        X_latest = latest_row[available_cols].values.reshape(1, -1)
        prob_up = float(model.predict_proba(X_latest)[0][1])

        return self._build_prediction_entry(ticker, prob_up, latest_row.to_dict(), "xgboost")

    def _write_predictions(self, predictions: List[Dict]) -> None:
        timestamp = datetime.utcnow().isoformat()
        payload = {
            "generated_at_utc": timestamp,
            "tickers": predictions,
        }
        dated_path = self.predictions_dir / f"predictions_{timestamp.split('T')[0]}.json"
        self.predictions_dir.mkdir(exist_ok=True)
        dated_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_path = self.predictions_dir / "latest_predictions.json"
        latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        docs_predictions = self.docs_data_dir / "latest_predictions.json"
        docs_predictions.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        history_path = self.analytics_dir / "prediction_history.json"
        existing: List[Dict] = []
        if history_path.exists():
            try:
                existing = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []

        for entry in predictions:
            entry["timestamp"] = timestamp
        history_path.write_text(json.dumps(existing + predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        docs_history = self.docs_data_dir / "prediction_history.json"
        docs_history.write_text(json.dumps(existing + predictions, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["BlowdartMLEngine", "TICKERS"]
