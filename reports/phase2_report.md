# Phase 2 - Complete Pipeline Report

**Execution Date:** Mon Dec  1 04:40:57 UTC 2025

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
    Training set: (100, 72)
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
  [Features] Total features: 74
  [FEATURES] Advanced features added successfully
  [FEATURES] Total features: 74
  [FEATURES] Final dataset: 126 rows × 74 columns

  [TRAIN] QQQ - 改善版モード起動
  [FEATURES] QQQ: 20 selected
  [REGIME] QQQ: CHOPPY          | Vol=1.00 | Trend=0.05
  [SPLIT] Train: 100 | Test: 26
  [WEIGHTS] Simple=70% | Aggressive=30% (CONSERVATIVE)
  [RESULT] QQQ | Hybrid Acc: 0.5400 | Improvement: +0.0562
  成功 | Regime: CHOPPY | Hybrid Acc: 0.0000

======================================================================
宗叡最終版 訓練結果
======================================================================
ticker  status regime  accuracy
  NVDA SUCCESS CHOPPY         0
  AAPL SUCCESS CHOPPY         0
  MSFT SUCCESS CHOPPY         0
 GOOGL SUCCESS CHOPPY         0
  AMZN SUCCESS CHOPPY         0
  META SUCCESS CHOPPY         0
  TSLA SUCCESS CHOPPY         0
   AMD SUCCESS CHOPPY         0
  NFLX SUCCESS CHOPPY         0
   QQQ SUCCESS CHOPPY         0

結果保存完了 → retraining_results.csv / json
次は本物の性能が見える。
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
  ✓ Best trades: 0
  ✓ Good trades: 4
  ✓ Fair trades: 6
  ✓ Poor trades: 0

[4/4] Generating reports...
  ✓ Saved: backtest_results/backtest_report.json
  ✓ Saved: backtest_results/backtest_report.md
  ✓ Saved: backtest_results/trades.json

======================================================================
Phase 3 Backtest Complete
======================================================================

Trade Quality:
  Best:  0
  Good:  4
  Fair:  6
  Poor:  0
```

---
_Report generated automatically by GitHub Actions_
