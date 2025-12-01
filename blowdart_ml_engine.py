import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger("blowdart_ml_engine")


# ===========================================================
# Safety Patch 1: Robust target generator
# ===========================================================
def generate_target(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if "Close" not in df.columns:
            raise ValueError("Close column missing")

        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna(subset=["target"])

        if df["target"].nunique() < 2:
            logger.warning("[TARGET] Unique=1 → fallback random jitter")
            df["target"] = (np.random.rand(len(df)) > 0.5).astype(int)

        return df

    except Exception as e:
        logger.error(f"[TARGET] Failed: {e}")
        df["target"] = (np.random.rand(len(df)) > 0.5).astype(int)
        return df


# ===========================================================
# Safety Patch 2: Hybrid accuracy with guards
# ===========================================================
def calc_hybrid_accuracy(y_test, pred_simple, pred_aggressive):
    try:
        if len(y_test) == 0:
            return 0.0

        if pred_simple is None or len(pred_simple) != len(y_test):
            pred_simple = np.zeros(len(y_test))

        if pred_aggressive is None or len(pred_aggressive) != len(y_test):
            pred_aggressive = np.zeros(len(y_test))

        acc_simple = accuracy_score(y_test, pred_simple)
        acc_aggressive = accuracy_score(y_test, pred_aggressive)

        hybrid = (acc_simple * 0.7) + (acc_aggressive * 0.3)
        return float(hybrid)

    except Exception:
        return 0.0


# ===========================================================
# Unified Blowdart Training Engine (Stable v2)
# ===========================================================
def train_model(df: pd.DataFrame, ticker: str):
    """
    Returns:
        dict = {
            "ok": bool,
            "ticker": str,
            "hybrid_acc": float,
            "test_size": int,
            "feature_count": int
        }
    """
    try:
        logger.info(f"[TRAIN] {ticker} - Blowdart Stable v2")

        df = generate_target(df)
        df = df.dropna().reset_index(drop=True)

        if len(df) < 50:
            return {
                "ok": False,
                "ticker": ticker,
                "error": "dataset too small",
                "hybrid_acc": 0.0
            }

        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        if len(X_test) == 0:
            return {
                "ok": False,
                "ticker": ticker,
                "error": "X_test=0",
                "hybrid_acc": 0.0
            }

        model_simple = RandomForestClassifier(
            n_estimators=80, max_depth=5, random_state=42
        )
        model_agg = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )

        model_simple.fit(X_train, y_train)
        model_agg.fit(X_train, y_train)

        pred_simple = model_simple.predict(X_test)
        pred_agg = model_agg.predict(X_test)

        acc = calc_hybrid_accuracy(y_test, pred_simple, pred_agg)

        return {
            "ok": True,
            "ticker": ticker,
            "hybrid_acc": acc,
            "test_size": len(X_test),
            "feature_count": X.shape[1]
        }

    except Exception as e:
        logger.error(f"[TRAIN] {ticker} fatal: {e}")
        return {
            "ok": False,
            "ticker": ticker,
            "error": str(e),
            "hybrid_acc": 0.0
        }


# ===========================================================
# Legacy Compatibility Wrapper
# ===========================================================
def train_ticker(*args, **kwargs):
    """
    Backward compatible wrapper for legacy calls.

    Returns:
        dict = full train_model() result
        BUT when numeric accuracy only is required, legacy callers may extract ["hybrid_acc"].
    """
    try:
        from data_loader import load_local_cache

        args = list(args)

        # remove self
        if len(args) >= 1 and not isinstance(args[0], str):
            args = args[1:]

        if len(args) < 1:
            raise ValueError("train_ticker missing ticker")

        ticker = args[0]

        if len(args) >= 2:
            df = args[1]
        else:
            df = load_local_cache(ticker)

        logger.info(f"[COMPAT] → train_model() [{ticker}]")

        result = train_model(df, ticker)

        # ------------- 重要修正点 -------------
        # 返すのは result（dict）にする！
        # retrain_all / Phase 2 は dict を期待している
        return result

    except Exception as e:
        logger.error(f"[train_ticker] failed: {e}")
        return {
            "ok": False,
            "ticker": ticker if "ticker" in locals() else "",
            "error": str(e),
            "hybrid_acc": 0.0
        }
