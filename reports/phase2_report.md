# Phase 2 - Complete Pipeline Report

**Execution Date:** Thu Nov 27 11:38:47 UTC 2025

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
ATR                    0.069317
Minus_DI               0.068229
OBV                    0.066006
DailyReturn            0.065358
CloseOpenRatio         0.062408
Momentum_10            0.059757
HighLowRatio           0.059716
Volume                 0.059620
Momentum               0.058993
ATR_Percent            0.058859
VROC                   0.057896
MACD                   0.057875
Volume_Ratio           0.056772
Distance_to_Support    0.055342
Low                    0.054414
RSI7                   0.054308
PVT                    0.053562
Momentum_5             0.052770
EMA26                  0.051631
EMA_Distance_10_50     0.050162
Name: importance, dtype: float64
```

### Step 2 - XGBoost Tuning
```
    Training set: (100, 21)
    Starting grid search...
    ✓ Best params: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}
    ✓ CV Score: 1.0000
    ✓ Test Accuracy: 1.0000
    ✓ Improvement: +0.4000

======================================================================
TUNING RESULTS SUMMARY
======================================================================
  ticker                                                                         best_params  best_cv_score  train_accuracy  test_accuracy  improvement
0   NVDA  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
1   AAPL  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
2   MSFT  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
3  GOOGL  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
4   AMZN  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
5   META  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
6   TSLA  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
7    AMD  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
8   NFLX  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4
9    QQQ  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8}            1.0             1.0            1.0          0.4

Average Improvement: +0.4000
Expected Accuracy: 1.0000 (target: 70%)

✅ Results saved to tuning_results/

======================================================================
XGBoost Tuning Complete
======================================================================
```

### Step 3 - Model Retraining
```
  [REDUCTION] ✓ Reduced from 74 → 21 features
  [REDUCTION] Final features: 21
  [FEATURES] Final dataset: 126 rows × 21 columns

  [HYBRID] Starting hybrid training for QQQ
  [TRAIN ERROR] QQQ: 'numpy.float64' object has no attribute 'quantile'
  ✗ Training failed

======================================================================
RETRAINING SUMMARY
======================================================================
  ticker  status           reason
0   NVDA  FAILED  Training failed
1   AAPL  FAILED  Training failed
2   MSFT  FAILED  Training failed
3  GOOGL  FAILED  Training failed
4   AMZN  FAILED  Training failed
5   META  FAILED  Training failed
6   TSLA  FAILED  Training failed
7    AMD  FAILED  Training failed
8   NFLX  FAILED  Training failed
9    QQQ  FAILED  Training failed

✅ Results saved to:
   - retraining_results.csv
   - retraining_results.json

======================================================================
Retraining Complete
======================================================================
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
  ✓ Best trades: 5
  ✓ Good trades: 1
  ✓ Fair trades: 4
  ✓ Poor trades: 0

[4/4] Generating reports...
  ✓ Saved: backtest_results/backtest_report.json
  ✓ Saved: backtest_results/backtest_report.md
  ✓ Saved: backtest_results/trades.json

======================================================================
Phase 3 Backtest Complete
======================================================================

Trade Quality:
  Best:  5
  Good:  1
  Fair:  4
  Poor:  0
```

---
_Report generated automatically by GitHub Actions_
