# Stock Prediction Performance

Generated from `retraining_results.csv` containing 10 tickers.

## Current retraining snapshot

- Mean accuracy: 0.5242
- Median accuracy: 0.5404
- Status breakdown: {'SUCCESS': 10}
- Best ticker: MSFT (0.6000)
- Lowest ticker: NFLX (0.3846)

### Regime breakdown
Regime | Tickers | Mean | Min | Max
--- | --- | --- | --- | ---
CHOPPY | 10 | 0.5242 | 0.3846 | 0.6000

### Top performers
- MSFT: 0.6000 (status=SUCCESS)
- NVDA: 0.5923 (status=SUCCESS)
- AMD: 0.5769 (status=SUCCESS)

### Bottom performers
- NFLX: 0.3846 (status=SUCCESS)
- META: 0.3885 (status=SUCCESS)
- AMZN: 0.5269 (status=SUCCESS)

## Historical context
- Previous average accuracy (from `accuracy_analysis/analysis_results.json`): 0.5750
- Change vs current mean: -0.0508

## Notes
- All figures are derived from existing offline evaluation artifacts; no live trading signals were generated.
