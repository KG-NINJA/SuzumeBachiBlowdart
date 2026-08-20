"""
xgboost_tuning.py - XGBoost ハイパーパラメータ最適化
時系列分割と独立した検証・テスト区間で全ティッカーを評価
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Configuration
TUNING_RESULTS_DIR = Path("tuning_results")
TUNING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "QQQ"]


def prepare_tuning_dataset(features_df):
    """Create a next-day target without exposing it to the model."""
    df = features_df.copy()
    future_close = df["Close"].shift(-1)
    valid_rows = future_close.notna()
    df = df.loc[valid_rows].copy()
    df["Target"] = (future_close.loc[valid_rows] > df["Close"]).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"Close", "Target"}]

    if not feature_cols:
        raise ValueError("No numeric feature columns available")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Target"]
    return X, y


def chronological_split(X, y):
    """Split ordered samples into 64% train, 16% validation, 20% test."""
    test_start = int(len(X) * 0.8)
    validation_start = int(test_start * 0.8)
    split_sizes = (validation_start, test_start - validation_start, len(X) - test_start)

    if min(split_sizes) < 10:
        raise ValueError(f"Insufficient data for chronological split: {split_sizes}")

    return (
        X.iloc[:validation_start],
        y.iloc[:validation_start],
        X.iloc[validation_start:test_start],
        y.iloc[validation_start:test_start],
        X.iloc[test_start:],
        y.iloc[test_start:],
    )


def tune_xgboost_for_ticker(ticker, X_train, y_train, X_val, y_val, X_test, y_test):
    """Tune on past-only folds, early-stop on validation, and test once."""
    print(f"\n>>> Tuning XGBoost for {ticker}")
    print(f"    Train/validation/test: {X_train.shape} / {X_val.shape} / {X_test.shape}")

    param_grid = {
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [3, 5],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0],
    }

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 150,
        "random_state": 42,
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": 1,
    }

    try:
        if y_train.nunique() < 2:
            raise ValueError("Training period contains only one target class")

        candidate_splits = TimeSeriesSplit(n_splits=3).split(X_train)
        cv_splits = [
            (train_idx, validation_idx)
            for train_idx, validation_idx in candidate_splits
            if y_train.iloc[train_idx].nunique() == 2
        ]
        if len(cv_splits) < 2:
            raise ValueError("Not enough class-diverse time-series folds")

        grid_search = GridSearchCV(
            xgb.XGBClassifier(**base_params),
            param_grid,
            cv=cv_splits,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
            error_score="raise",
        )

        print("    Starting time-series grid search...")
        grid_search.fit(X_train, y_train)

        early_stop_params = {
            **base_params,
            **grid_search.best_params_,
            "n_estimators": 500,
            "early_stopping_rounds": 25,
        }
        early_stop_model = xgb.XGBClassifier(**early_stop_params)
        early_stop_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        train_acc = early_stop_model.score(X_train, y_train)
        val_acc = early_stop_model.score(X_val, y_val)
        best_n_estimators = int(getattr(early_stop_model, "best_iteration", 499)) + 1

        final_params = {
            **base_params,
            **grid_search.best_params_,
            "n_estimators": best_n_estimators,
        }
        best_model = xgb.XGBClassifier(**final_params)
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        best_model.fit(X_train_val, y_train_val, verbose=False)
        test_acc = best_model.score(X_test, y_test)

        results = {
            "ticker": ticker,
            "best_params": {
                **grid_search.best_params_,
                "n_estimators": best_n_estimators,
            },
            "best_cv_score": float(grid_search.best_score_),
            "train_accuracy": float(train_acc),
            "validation_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "overfitting_gap": float(train_acc - test_acc),
            "improvement": float(test_acc - 0.60),
        }

        print(f"    ✓ Best params: {results['best_params']}")
        print(f"    ✓ CV Score: {results['best_cv_score']:.4f}")
        print(
            f"    ✓ Train/validation/test: {train_acc:.4f} / "
            f"{val_acc:.4f} / {test_acc:.4f}"
        )
        print(f"    ✓ Overfitting gap: {results['overfitting_gap']:+.4f}")
        return best_model, results

    except Exception as e:
        print(f"    ✗ Error: {str(e)[:100]}")
        return None, None


def run_tuning_for_all_tickers():
    """Run leakage-safe tuning for all configured tickers."""
    from blowdart_features import build_feature_set
    from utils_data_fetch import safe_price_download

    print("=" * 70)
    print("XGBoost Hyperparameter Tuning")
    print("=" * 70)

    all_results = []

    for ticker in TICKERS:
        try:
            price_data = safe_price_download(ticker, days=180)

            if price_data is None or price_data.empty:
                print(f"  ✗ {ticker}: No data")
                continue

            features_df = build_feature_set(price_data, ticker)

            if features_df is None or features_df.empty:
                print(f"  ✗ {ticker}: Feature engineering failed")
                continue

            X, y = prepare_tuning_dataset(features_df)

            if len(X) < 50:
                print(f"  ✗ {ticker}: Insufficient data")
                continue

            split_data = chronological_split(X, y)
            best_model, results = tune_xgboost_for_ticker(ticker, *split_data)

            if results:
                all_results.append(results)

        except Exception as e:
            print(f"  ✗ {ticker}: {str(e)[:100]}")

    print("\n" + "=" * 70)
    print("TUNING RESULTS SUMMARY")
    print("=" * 70)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string())

        avg_test_accuracy = results_df["test_accuracy"].mean()
        avg_overfitting_gap = results_df["overfitting_gap"].mean()
        print(f"\nAverage Test Accuracy: {avg_test_accuracy:.4f}")
        print(f"Average Overfitting Gap: {avg_overfitting_gap:+.4f}")

        results_df.to_csv(TUNING_RESULTS_DIR / "tuning_results.csv", index=False)

        with open(TUNING_RESULTS_DIR / "tuning_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✅ Results saved to {TUNING_RESULTS_DIR}/")
        return results_df

    print("\n⚠️ No tuning results")
    return None


if __name__ == "__main__":
    results_df = run_tuning_for_all_tickers()

    print("\n" + "=" * 70)
    print("XGBoost Tuning Complete")
    print("=" * 70)
