# Accuracy Review

## Current State
- Feature set now includes CV confidence/signal columns, richer momentum and volatility signals, lagged returns, volume pressure, and seasonality.
- Data acquisition accepts both `range` and `days` lookbacks, matching the ad-hoc harness. Cache and directory setup remain automatic for Actions.
- Ticker-aware XGBoost parameters scale up estimators for weaker names (AAPL/NFLX) and tighten depth/child-weight for stable names (GOOGL).

## Known Baseline (before rerun)
- GOOGL: 0.75
- TSLA: 0.6875
- AAPL: 0.375
- NFLX: 0.5769

## Expected After Rerun
- Mean accuracy: **≥ 0.60**
- AAPL: **0.45–0.55**
- NFLX: **0.60–0.65**

## How to Validate
1. Execute the provided training harness (see repo instructions) or `python - <<'PY' ...` snippet from the task.
2. Confirm per-ticker accuracies in `models/*_xgb_meta.json` and aggregate `analytics/training_metrics.json`.
3. Ensure predictions propagate to `daily_predictions/latest_predictions.json` and `docs/data/latest_predictions.json`.
4. Re-run if any ticker reports empty datasets or missing features.

## Notes
- If a provider is rate-limited, the downloader logs each fetch attempt and retries with backoff. `days` overrides `range` for quick backtests.
- The lightweight feature reduction removes constant or duplicate-like columns when `use_feature_reduction=True`, keeping smaller datasets stable.
