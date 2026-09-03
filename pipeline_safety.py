"""Safety helpers for the daily prediction pipeline.

The functions in this module are intentionally independent from the model code so
that data-shape failures, probability semantics, and run-status decisions can be
unit tested cheaply.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

_REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")
_COLUMN_ALIASES = {
    "date": "Date",
    "datetime": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adjusted close": "Adj Close",
    "volume": "Volume",
}


def _normalise_token(value: object) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _column_parts(column: object) -> Iterable[object]:
    if isinstance(column, tuple):
        return column
    return (column,)


def _fallback_column_name(column: object) -> str:
    parts = [
        str(part).strip()
        for part in _column_parts(column)
        if part is not None and str(part).strip() and str(part).lower() != "nan"
    ]
    return "_".join(parts) if parts else "unnamed_column"


def normalize_ohlcv_columns(
    price_data: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Return a copy whose OHLCV columns use canonical names.

    Recent yfinance versions may return either flat columns or a two-level
    ``MultiIndex`` for a single ticker. This function recognises OHLCV labels in
    any tuple level and fails explicitly when multiple tickers make a required
    field ambiguous.
    """

    label = ticker or "price data"
    if not isinstance(price_data, pd.DataFrame):
        raise TypeError(f"{label}: expected pandas.DataFrame")
    if price_data.empty:
        raise ValueError(f"{label}: price data is empty")

    frame = price_data.copy()
    normalized_names: list[str] = []

    for column in frame.columns:
        canonical_name = None
        for part in _column_parts(column):
            canonical_name = _COLUMN_ALIASES.get(_normalise_token(part))
            if canonical_name is not None:
                break

        normalized_names.append(
            canonical_name if canonical_name is not None else _fallback_column_name(column)
        )

    frame.columns = normalized_names

    duplicate_required = [
        column for column in _REQUIRED_OHLCV if normalized_names.count(column) > 1
    ]
    if duplicate_required:
        duplicate_text = ", ".join(duplicate_required)
        raise ValueError(
            f"{label}: ambiguous OHLCV columns ({duplicate_text}); "
            "provide data for one ticker at a time"
        )

    if "Close" not in frame.columns and "Adj Close" in frame.columns:
        frame["Close"] = frame["Adj Close"]

    missing = [column for column in _REQUIRED_OHLCV if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required OHLCV columns: {', '.join(missing)}"
        )

    return frame


def _as_probability(value: Any, field_name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc

    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return probability


def normalize_prediction_probabilities(
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize probability fields without confusing probability and confidence.

    ``ml_engine_v2`` exposes the raw probability of the positive class as
    ``ensemble_pred`` while its historical ``confidence`` field is already a
    distance-from-50% score. The v1 engine exposes ``prob_up`` directly. This
    function produces a common representation for both engines.
    """

    if not isinstance(prediction, Mapping):
        raise TypeError("prediction must be a mapping")

    normalized = dict(prediction)

    if normalized.get("ensemble_pred") is not None:
        probability_up = _as_probability(normalized["ensemble_pred"], "ensemble_pred")
        probability_source = "ensemble_pred"
    elif normalized.get("prob_up") is not None:
        probability_up = _as_probability(normalized["prob_up"], "prob_up")
        probability_source = "prob_up"
    elif normalized.get("confidence") is not None:
        # Legacy v1 records used confidence as a directional probability.
        probability_up = _as_probability(normalized["confidence"], "confidence")
        probability_source = "legacy_confidence"
    else:
        raise ValueError("prediction has no probability field")

    confidence_score = abs(probability_up - 0.5) * 2.0
    normalized["prob_up"] = float(probability_up)
    normalized["prob_down"] = float(1.0 - probability_up)
    normalized["confidence_score"] = float(confidence_score)
    normalized["probability_source"] = probability_source

    return normalized


def _confidence_level(score: float) -> str:
    if score >= 0.30:
        return "STRONG"
    if score >= 0.10:
        return "MEDIUM"
    return "WEAK"


def apply_confidence_decisions(
    predictions: Sequence[Mapping[str, Any]],
    min_confidence: float = 0.30,
) -> list[dict[str, Any]]:
    """Attach EXECUTE/SKIP decisions using normalized probability semantics."""

    threshold = _as_probability(min_confidence, "min_confidence")
    decisions: list[dict[str, Any]] = []

    for prediction in predictions:
        normalized = normalize_prediction_probabilities(prediction)
        score = float(normalized["confidence_score"])
        level = _confidence_level(score)

        if score >= threshold:
            action = "EXECUTE"
            reason = f"High confidence ({score:.1%}) - {level} signal"
            recommendation = f"Execute {normalized.get('direction', '?')} trade"
        else:
            action = "SKIP"
            reason = f"Low confidence ({score:.1%}) - Market noise"
            recommendation = "Skip this trade - wait for clearer signal"

        normalized.update(
            {
                "confidence_level": level,
                "action": action,
                "reason": reason,
                "recommendation": recommendation,
            }
        )
        decisions.append(normalized)

    return decisions


def classify_run_status(
    *,
    attempted_symbols: int,
    successful_inferences: int,
    executable_signals: int,
    failed_symbols: int,
) -> str:
    """Classify a run without confusing abstention with execution failure."""

    counts = {
        "attempted_symbols": attempted_symbols,
        "successful_inferences": successful_inferences,
        "executable_signals": executable_signals,
        "failed_symbols": failed_symbols,
    }
    if any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("run counters must be non-negative integers")
    if successful_inferences + failed_symbols > attempted_symbols:
        raise ValueError("successful and failed symbol counts exceed attempted symbols")
    if executable_signals > successful_inferences:
        raise ValueError("executable signals cannot exceed successful inferences")

    # Fail closed when the pipeline produced no usable inference. This catches
    # empty ticker lists, total data failures, and total model failures.
    if attempted_symbols == 0 or successful_inferences == 0:
        return "FAILED"

    if failed_symbols > 0:
        return "PARTIAL_SUCCESS" if executable_signals > 0 else "PARTIAL_NO_SIGNAL"

    return "PREDICTED" if executable_signals > 0 else "NO_SIGNAL"


def should_fail_run(status: str) -> bool:
    """Return whether GitHub Actions should mark the prediction run as failed."""

    return status == "FAILED"
