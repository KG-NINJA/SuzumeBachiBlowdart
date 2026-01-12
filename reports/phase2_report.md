# Phase 2 - Complete Pipeline Report

**Execution Date:** Mon Jan 12 04:41:25 UTC 2026

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
DailyReturn_lag1       0.034781
Volume                 0.034349
CloseOpenRatio_lag1    0.032811
OBV                    0.032683
ROC_5                  0.032166
MACD                   0.030203
Plus_DI                0.029803
VROC                   0.029404
Stoch_K                0.029161
ATR_Percent            0.028806
ATR                    0.028349
Momentum               0.028328
EMA_Distance_10_50     0.027194
HighLowRatio_lag1      0.027149
Low                    0.026214
ROC10                  0.026119
Open                   0.025947
High                   0.025895
Momentum_10            0.025676
ADX                    0.025616
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
    ✓ ADX
    ✓ RSI & Divergence
    ✓ MACD
    ✓ Volume Features
    ✓ Volatility Features
    ✓ Momentum Features
    ✓ Trend Features
  [Features] Total features: 74
  [FEATURES] Advanced features added successfully
  [FEATURES] Total features: 74
  [FEATURES] Final dataset: 126 rows × 74 columns
  成功 | Regime: CHOPPY | Hybrid Acc: 0.5269

======================================================================
宗叡最終版 訓練結果
======================================================================
ticker  status regime  accuracy
  NVDA SUCCESS CHOPPY    0.5923
  AAPL SUCCESS CHOPPY    0.5500
  MSFT SUCCESS CHOPPY    0.6000
 GOOGL SUCCESS CHOPPY    0.5654
  AMZN SUCCESS CHOPPY    0.5269
  META SUCCESS CHOPPY    0.3885
  TSLA SUCCESS CHOPPY    0.5308
   AMD SUCCESS CHOPPY    0.5769
  NFLX SUCCESS CHOPPY    0.3846
   QQQ SUCCESS CHOPPY    0.5269

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
