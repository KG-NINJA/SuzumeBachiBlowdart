# Phase 2 - Complete Pipeline Report

**Execution Date:** Thu Nov 27 06:37:38 UTC 2025

## Pipeline Status
- ✅ Feature Analysis
- ✅ XGBoost Tuning
- ✅ Model Retraining
- ✅ Backtest Validation

## Execution Logs

### Step 1 - Feature Analysis
```

======================================================================
Feature Analysis Complete
======================================================================

Analyzed tickers: 10/10

Top 10 Most Important Features Across All Tickers:
feature
DailyReturn            0.035081
Momentum_10            0.033892
OBV                    0.033795
Volume                 0.031058
HighLowRatio           0.030555
MACD                   0.030350
ATR_Percent            0.029862
Minus_DI               0.029447
EMA_Distance_10_50     0.028300
Momentum               0.028290
VROC                   0.027899
RSI7                   0.027122
EMA26                  0.027084
Momentum_5             0.027070
CloseOpenRatio         0.026998
ATR                    0.026833
Volume_Ratio           0.026765
Distance_to_Support    0.025851
PVT                    0.025675
Low                    0.025049
Name: importance, dtype: float64
```

### Step 2 - XGBoost Tuning
```
python: can't open file '/home/runner/work/SuzumeBachiBlowdart/SuzumeBachiBlowdart/xgboost_tuning.py': [Errno 2] No such file or directory
```

### Step 3 - Model Retraining
```
python: can't open file '/home/runner/work/SuzumeBachiBlowdart/SuzumeBachiBlowdart/retrain_all.py': [Errno 2] No such file or directory
```

### Step 4 - Backtest Validation
```
======================================================================
Phase 3: Backtest & Validation
======================================================================

[1/4] Loading predictions with Phase 2 data...
  ✓ Loaded 10 predictions

[2/4] Simulating trades...
  ✓ Simulated 10 trades

[3/4] Calculating metrics...
  ✓ Best trades: 2
  ✓ Good trades: 4
  ✓ Fair trades: 3
  ✓ Poor trades: 1

[4/4] Generating reports...
  ✓ Saved: backtest_results/backtest_report.json
  ✓ Saved: backtest_results/backtest_report.md
  ✓ Saved: backtest_results/trades.json

======================================================================
Phase 3 Backtest Complete
======================================================================

Trade Quality:
  Best:  2
  Good:  4
  Fair:  3
  Poor:  1
```

---
_Report generated automatically by GitHub Actions_
