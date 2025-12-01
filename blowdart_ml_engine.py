import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger("blowdart_ml_engine")


# ===========================================================
# Safety Patch 1:
#   Robust target generator + fallback
# ===========================================================
def generate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary up/down target for next-day price.
    Includes full fallback logic for safety.
    """
    try:
        if "Close" not in df.columns:
            raise ValueError("Close column missing")

        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna(subset=["target"])

        # If all same → fallback
        if df["target"].nunique() < 2:
            logger.warning("[TARGET] Unique=1 → fallback random jitter")
            df["target"] = (np.random.rand(len(df)) > 0.5).astype(int)

        return df

    except Exception as e:
        logger.error(f"[TARGET] Failed to generate target: {e}")
        df["target"] = (np.random.rand(len(df)) > 0.5).astype(int)
        return df


# ===========================================================
# Safety Patch 2:
#   Hybrid accuracy calculation with full guards
# ===========================================================
def calc_hybrid_accuracy(y_test, pred_simple, pred_aggressive):
    try:
        if len(y_test) == 0:
            logger.error("[HYBRID] Empty y_test")
            return 0.0

        if pred_simple is None or len(pred_simple) != len(y_test):
            logger.warning("[HYBRID] Simple prediction invalid → fallback")
            pred_simple = np.zeros(len(y_test))

        if pred_aggressive is None or len(pred_aggressive) != len(y_test):
            logger.warning("[HYBRID] Aggressive prediction invalid → fallback")
            pred_aggressive = np.zeros(len(y_test))

        acc_simple = accuracy_score(y_test, pred_simple)
        acc_aggressive = accuracy_score(y_test, pred_aggressive)

        w1 = 0.7
        w2 = 0.3

        hybrid = (acc_simple * w1) + (acc_aggressive * w2)

        return float(hybrid)

    except Exception as e:
        logger.error(f"[HYBRID] calc failed: {e}")
        return 0.0


# ===========================================================
# Safety Patch 3:
#   Unified Blowdart Integrated Training Engine
# ===========================================================
def train_model(df: pd.DataFrame, ticker: str):
    """
    Main unified training function.
    - Handles 20-features版 / 宗叡最終版 どちらでも動作
    - dropna の最適タイミングにより dataset=0 を防止
    - RandomForest(Hybrid) による安全学習
    """
    try:
        logger.info(f"[TRAIN] {ticker} - Blowdart Integrated v1.0")

        # -------------------------
        # Step 1: target
        # -------------------------
        df = generate_target(df)

        # -------------------------
        # Step 2: drop invalid rows
        # -------------------------
        df = df.dropna().reset_index(drop=True)

        if len(df) < 50:
            logger.error(f"[TRAIN] {ticker}: dataset too small → {len(df)} rows")
            return 0.0

        # -------------------------
        # Step 3: train/test split
        # -------------------------
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        logger.info(f"[SPLIT] {ticker}: Train={len(X_train)} | Test={len(X_test)}")

        if len(X_test) == 0:
            logger.error(f"[TRAIN] {ticker}: X_test=0 → abort")
            return 0.0

        # -------------------------
        # Step 4: two models
        # -------------------------
        model_simple = RandomForestClassifier(
            n_estimators=80, max_depth=5, random_state=42
        )
        model_agg = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )

        model_simple.fit(X_train, y_train)
        model_agg.fit(X_train, y_train)

        pred_simple = model_simple.predict(X_test)
        pred_aggressive = model_agg.predict(X_test)

        # -------------------------
        # Step 5: Hybrid
        # -------------------------
        acc = calc_hybrid_accuracy(y_test, pred_simple, pred_aggressive)

        logger.info(f"[RESULT] {ticker} | Hybrid Acc: {acc:.4f}")

        return float(acc)

    except Exception as e:
        logger.error(f"[TRAIN] {ticker} fatal: {e}")
        return 0.0


# ===========================================================
# Legacy Compatibility Wrapper
#   For 宗叡版 / retrain_all.py auto compatibility
# ===========================================================
def train_ticker(ticker: str) -> float:
    """
    Legacy wrapper for older scripts.
    - Automatically loads local cache dataset
    - Calls train_model()
    """
    try:
        from data_loader import load_local_cache  # 宗叡版の loader と互換

        df = load_local_cache(ticker)
        logger.info(f"[COMPAT] train_ticker() → train_model() [{ticker}]")

        return train_model(df, ticker)

    except Exception as e:
        logger.error(f"[train_ticker] {ticker} failed: {e}")
        return 0.0
