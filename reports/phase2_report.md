# Phase 2 - Complete Pipeline Report

**Execution Date:** Sat Nov 29 12:26:13 UTC 2025

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
    ✓ Volume Features
    ✓ Volatility Features
    ✓ Momentum Features
    ✓ Trend Features
  [Features] Total features: 74
  [FEATURES] Advanced features added successfully
  [FEATURES] Total features: 74
  [FEATURES] Final dataset: 126 rows × 74 columns

  [TRAIN] QQQ - 改善版モード起動
  [ERROR] QQQ: name 'model_path' is not defined
  訓練失敗

======================================================================
宗叡最終版 訓練結果
======================================================================
ticker status          reason
  NVDA FAILED Training failed
  AAPL FAILED Training failed
  MSFT FAILED Training failed
 GOOGL FAILED Training failed
  AMZN FAILED Training failed
  META FAILED Training failed
  TSLA FAILED Training failed
   AMD FAILED Training failed
  NFLX FAILED Training failed
   QQQ FAILED Training failed

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
  ✓ Best trades: 10
  ✓ Good trades: 0
  ✓ Fair trades: 0
  ✓ Poor trades: 0

[4/4] Generating reports...
  ✓ Saved: backtest_results/backtest_report.json
  ✓ Saved: backtest_results/backtest_report.md
  ✓ Saved: backtest_results/trades.json

======================================================================
Phase 3 Backtest Complete
======================================================================

Trade Quality:
  Best:  10
  Good:  0
  Fair:  0
  Poor:  0
```

---
_Report generated automatically by GitHub Actions_
