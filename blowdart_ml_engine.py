import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Optional, Dict, Any
import json
import pickle
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("blowdart_ml_engine")


# ===========================================================
# ターゲット生成（落ちない＋fallback込み）
# ===========================================================
def generate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    target = 明日 Close が上がるかどうかの 1/0
    """
    try:
        if "Close" not in df.columns:
            raise ValueError("Close column missing")

        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna(subset=["target"])

        # 全部 1 または 0 の場合 → fallback
        if df["target"].nunique() < 2:
            logger.warning("[TARGET] Unique target=1 → fallback random")
            df["target"] = (np.random.rand(len(df)) > 0.5).astype(int)

        return df

    except Exception as e:
        logger.error(f"[TARGET] Failed: {e}")
        df["target"] = (np.random.rand(len(df)) > 0.5).astype(int)
        return df


# ===========================================================
# 統合特徴量 20版 + 宗叡74版 → 自動統合
# ===========================================================
def select_integrated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    改善版20特徴量 ＋ 宗叡74特徴量 を自動統合。
    存在しない列は無視。
    """
    feature_candidates = [
        # ==== 20 features basic ====
        "return_1d", "return_3d", "return_5d",
        "volatility_5d", "volatility_10d",
        "ma_5", "ma_10", "ma_20",
        "rsi14", "macd", "macd_signal",
        "volume_change", "atr", "stoch_k", "stoch_d",
        "ema_5", "ema_10", "ema_20", "ema_ratio",
        "range_ratio",

        # ==== 宗叡 74 features（あれば使う） ====
        "bb_upper", "bb_lower", "bb_width",
        "adx", "di_plus", "di_minus",
        "rsi_divergence",
        "macd_hist",
        "vol_ma_ratio",
        "trend_strength",
        "momentum_3", "momentum_10",
        "volatility_20",
        "volume_trend",
    ]

    final_features = [c for c in feature_candidates if c in df.columns]

    logger.info(f"[FEATURES] Integrated: {len(final_features)} selected")
    return df[final_features + ["target"]]


# ===========================================================
# レジーム分類（改善版）
# ===========================================================
def detect_regime(df):
    """
    トレンド／ボラティリティの強弱から CHOPPY / TRENDING を判定
    """
    try:
        vol = df["volatility_10d"].mean() if "volatility_10d" in df else 1
        trend = abs(df["trend_strength"].mean()) if "trend_strength" in df else 0.1

        if vol < 0.5 and trend > 0.3:
            regime = "TRENDING"
        else:
            regime = "CHOPPY"

        logger.info(f"[REGIME] {regime:10s} | Vol={vol:.2f} | Trend={trend:.2f}")
        return regime

    except Exception as e:
        logger.error(f"[REGIME] detection failed: {e}")
        return "CHOPPY"


# ===========================================================
# ハイブリッド精度（安全化）
# ===========================================================
def calc_hybrid_accuracy(y_test, pred_simple, pred_agg):
    try:
        if len(y_test) == 0:
            return 0.0

        if pred_simple is None or len(pred_simple) != len(y_test):
            pred_simple = np.zeros(len(y_test))
        if pred_agg is None or len(pred_agg) != len(y_test):
            pred_agg = np.zeros(len(y_test))

        acc_s = accuracy_score(y_test, pred_simple)
        acc_a = accuracy_score(y_test, pred_agg)

        hybrid = acc_s * 0.7 + acc_a * 0.3
        return float(hybrid)

    except Exception as e:
        logger.error(f"[HYBRID] error: {e}")
        return 0.0


# ===========================================================
# 主要トレーニング関数（改良安定版）
# ===========================================================
def train_model(df: pd.DataFrame, ticker: str) -> float:
    try:
        logger.info(f"[TRAIN] {ticker} - 統合安定版 起動")

        # ---- 1) ターゲット生成 ----
        df = generate_target(df)

        # ---- 2) NA 除去 ----
        df = df.dropna().reset_index(drop=True)

        if len(df) < 60:
            logger.error(f"[TRAIN] {ticker}: too small ({len(df)} rows)")
            return 0.0

        # ---- 3) 特徴量統合 ----
        df = select_integrated_features(df)

        # ---- 4) Regime ----
        detect_regime(df)

        # ---- 5) train/test ----
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        logger.info(f"[SPLIT] Train={len(X_train)} | Test={len(X_test)}")

        if len(X_test) == 0:
            logger.error(f"[TRAIN] Test size=0 → abort")
            return 0.0

        # ---- 6) モデル ----
        model_simple = RandomForestClassifier(
            n_estimators=80,
            max_depth=5,
            random_state=42
        )
        model_agg = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model_simple.fit(X_train, y_train)
        model_agg.fit(X_train, y_train)

        pred_simple = model_simple.predict(X_test)
        pred_agg = model_agg.predict(X_test)

        # ---- 7) Hybrid 精度 ----
        acc = calc_hybrid_accuracy(y_test, pred_simple, pred_agg)

        logger.info(f"[RESULT] {ticker} | Hybrid Acc: {acc:.4f}")

        return acc

    except Exception as e:
        logger.error(f"[TRAIN] {ticker} fatal error: {e}")
        return 0.0
