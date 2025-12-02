"""
ml_engine_v2.py - Next Generation ML Engine
Features:
  - LightGBM + XGBoost + CatBoost アンサンブル
  - Optuna による自動ハイパーパラメータ最適化
  - 5-Fold Cross Validation
  - 既存の blowdart_ml_engine.py と互換性あり
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
import logging
from typing import Optional, Dict, Any, Tuple
import json
import pickle
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import optuna
from optuna.pruners import MedianPruner

logger = logging.getLogger("blowdart_ml_engine_v2")

# ディレクトリ設定
MODELS_DIR = Path("models_v2")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ANALYTICS_DIR = Path("analytics")
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

OPTUNA_DIR = Path("optuna_studies")
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================
# 1. ハイパーパラメータ最適化 (Optuna)
# ===========================================================

def create_study_name(ticker: str, model_type: str) -> str:
    """Optuna study 名を生成"""
    return f"{ticker}_{model_type}_{datetime.now().strftime('%Y%m%d')}"


def objective_lightgbm(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """LightGBM ハイパーパラメータ最適化目的関数"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1, log=True),
    }
    
    model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    
    # 5-Fold CV で評価
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    return scores.mean()


def objective_xgboost(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """XGBoost ハイパーパラメータ最適化目的関数"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, log=True),
    }
    
    model = XGBClassifier(**params, random_state=42, n_jobs=-1)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    return scores.mean()


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ticker: str,
    model_type: str = 'lightgbm',
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Optuna で最適なハイパーパラメータを探索
    
    Args:
        X_train: 訓練用特徴データ
        y_train: 訓練用ターゲット
        ticker: 株式シンボル
        model_type: 'lightgbm' or 'xgboost'
        n_trials: 探索試行回数
    
    Returns:
        最適パラメータとスコア
    """
    try:
        logger.info(f"[OPTUNA] Starting optimization for {ticker} ({model_type})")
        
        study_name = create_study_name(ticker, model_type)
        
        # 目的関数選択
        if model_type == 'lightgbm':
            objective = lambda trial: objective_lightgbm(trial, X_train, y_train)
        else:
            objective = lambda trial: objective_xgboost(trial, X_train, y_train)
        
        # Study 作成
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_warmup_steps=10),
            study_name=study_name
        )
        
        # 最適化実行
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_score = study.best_value
        best_params = study.best_params
        
        logger.info(
            f"[OPTUNA] {ticker} ({model_type}): "
            f"Best Score={best_score:.4f} | "
            f"Trials={len(study.trials)}"
        )
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials)
        }
    
    except Exception as e:
        logger.error(f"[OPTUNA] {ticker} failed: {e}")
        return None


# ===========================================================
# 2. 複数モデル評価 & 選択
# ===========================================================

def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Optional[Dict] = None
) -> Tuple[lgb.LGBMClassifier, float]:
    """LightGBM モデルを訓練"""
    
    try:
        if best_params is None:
            # デフォルトパラメータ
            best_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 7,
                'min_child_samples': 10,
            }
        
        model = lgb.LGBMClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train, eval_metric='binary_logloss')
        
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        logger.info(f"[LGB] Accuracy: {accuracy:.4f}")
        return model, accuracy
    
    except Exception as e:
        logger.error(f"[LGB] Training failed: {e}")
        return None, 0.0


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Optional[Dict] = None
) -> Tuple[XGBClassifier, float]:
    """XGBoost モデルを訓練"""
    
    try:
        if best_params is None:
            best_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        
        model = XGBClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        logger.info(f"[XGB] Accuracy: {accuracy:.4f}")
        return model, accuracy
    
    except Exception as e:
        logger.error(f"[XGB] Training failed: {e}")
        return None, 0.0


def train_catboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[CatBoostClassifier, float]:
    """CatBoost モデルを訓練"""
    
    try:
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        logger.info(f"[CAT] Accuracy: {accuracy:.4f}")
        return model, accuracy
    
    except Exception as e:
        logger.error(f"[CAT] Training failed: {e}")
        return None, 0.0


# ===========================================================
# 3. アンサンブルモデル構築
# ===========================================================

def create_super_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    lgb_params: Optional[Dict] = None,
    xgb_params: Optional[Dict] = None
) -> Tuple[VotingClassifier, Dict[str, Any]]:
    """
    LightGBM + XGBoost + CatBoost の投票アンサンブルを構築
    
    Returns:
        (アンサンブルモデル, 個別モデルの精度辞書)
    """
    
    logger.info("[ENSEMBLE] Building super ensemble...")
    
    models_info = {}
    
    # 1. LightGBM (Conservative)
    lgb_model, lgb_acc = train_lightgbm_model(
        X_train, y_train, X_test, y_test, lgb_params
    )
    if lgb_model:
        models_info['lgb'] = lgb_acc
    
    # 2. XGBoost
    xgb_model, xgb_acc = train_xgboost_model(
        X_train, y_train, X_test, y_test, xgb_params
    )
    if xgb_model:
        models_info['xgb'] = xgb_acc
    
    # 3. CatBoost
    cat_model, cat_acc = train_catboost_model(
        X_train, y_train, X_test, y_test
    )
    if cat_model:
        models_info['cat'] = cat_acc
    
    # アンサンブル構築
    estimators = []
    if lgb_model:
        estimators.append(('lgb', lgb_model))
    if xgb_model:
        estimators.append(('xgb', xgb_model))
    if cat_model:
        estimators.append(('cat', cat_model))
    
    if not estimators:
        logger.error("[ENSEMBLE] No models available")
        return None, {}
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    voting_clf.fit(X_train, y_train)
    
    ensemble_pred = voting_clf.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    logger.info(f"[ENSEMBLE] Ensemble Accuracy: {ensemble_acc:.4f}")
    logger.info(f"[ENSEMBLE] Individual accuracies: {models_info}")
    
    return voting_clf, {
        'individual': models_info,
        'ensemble': ensemble_acc
    }


# ===========================================================
# 4. モデル保存・読み込み
# ===========================================================

def save_model_v2(ticker: str, model: Any, metadata: Dict[str, Any]) -> bool:
    """V2 モデルを保存"""
    
    try:
        ticker_dir = MODELS_DIR / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル保存
        model_path = ticker_dir / "model_v2.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # メタデータ保存
        metadata["saved_at"] = datetime.now().isoformat()
        metadata["model_version"] = "v2"
        metadata_path = ticker_dir / "metadata_v2.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"[SAVE] {ticker}: V2 Model saved successfully")
        return True
    
    except Exception as e:
        logger.error(f"[SAVE] {ticker}: Failed to save model: {e}")
        return False


def load_model_v2(ticker: str) -> Optional[Any]:
    """V2 モデルを読み込み"""
    
    try:
        ticker_dir = MODELS_DIR / ticker
        model_path = ticker_dir / "model_v2.pkl"
        
        if not model_path.exists():
            logger.debug(f"[LOAD] {ticker}: No V2 model found")
            return None
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        logger.info(f"[LOAD] {ticker}: V2 Model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"[LOAD] {ticker}: Failed to load model: {e}")
        return None


def load_metadata_v2(ticker: str) -> Optional[Dict[str, Any]]:
    """V2 メタデータを読み込み"""
    
    try:
        ticker_dir = MODELS_DIR / ticker
        metadata_path = ticker_dir / "metadata_v2.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    
    except Exception as e:
        logger.error(f"[METADATA] {ticker}: Failed to load metadata: {e}")
        return None


# ===========================================================
# 5. 主関数: 統合訓練エンジン (V2)
# ===========================================================

def train_model_v2(
    df: pd.DataFrame,
    ticker: str,
    use_existing: bool = True,
    optimize_hyperparams: bool = True,
    n_optuna_trials: int = 50
) -> Dict[str, Any]:
    """
    V2 統合訓練エンジン
    
    Args:
        df: 特徴データフレーム
        ticker: 株式シンボル
        use_existing: 既存モデルを使用するか
        optimize_hyperparams: Optuna でハイパラ最適化するか
        n_optuna_trials: Optuna 試行回数
    
    Returns:
        訓練結果辞書
    """
    
    try:
        logger.info(
            f"[TRAIN_V2] {ticker} - Starting (hyperopt={optimize_hyperparams})"
        )
        
        # ターゲット生成
        if "Close" not in df.columns:
            logger.error(f"[TRAIN_V2] {ticker}: Close column missing")
            return {"ok": False, "error": "Close missing"}
        
        df = df.copy()
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        
        if len(df) < 100:
            logger.warning(f"[TRAIN_V2] {ticker}: Insufficient data ({len(df)} rows)")
            return {
                "ok": False,
                "ticker": ticker,
                "error": "insufficient_data"
            }
        
        # 特徴とターゲット分離
        X = df.drop(columns=["target", "Close"], errors="ignore")
        y = df["target"]
        
        # 時系列順での 80/20 分割
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(
            f"[TRAIN_V2] {ticker}: Train={len(X_train)}, Test={len(X_test)}"
        )
        
        # ハイパーパラメータ最適化
        lgb_params = None
        xgb_params = None
        
        if optimize_hyperparams:
            logger.info(f"[TRAIN_V2] {ticker}: Running Optuna optimization...")
            
            lgb_result = optimize_hyperparameters(
                X_train, y_train, ticker, 'lightgbm', n_optuna_trials
            )
            if lgb_result:
                lgb_params = lgb_result['best_params']
                logger.info(f"[TRAIN_V2] LightGBM best score: {lgb_result['best_score']:.4f}")
            
            xgb_result = optimize_hyperparameters(
                X_train, y_train, ticker, 'xgboost', n_optuna_trials
            )
            if xgb_result:
                xgb_params = xgb_result['best_params']
                logger.info(f"[TRAIN_V2] XGBoost best score: {xgb_result['best_score']:.4f}")
        
        # アンサンブル構築
        ensemble_model, accuracies = create_super_ensemble(
            X_train, y_train, X_test, y_test, lgb_params, xgb_params
        )
        
        if ensemble_model is None:
            return {
                "ok": False,
                "ticker": ticker,
                "error": "ensemble_creation_failed"
            }
        
        # メタデータ生成
        metadata = {
            "ticker": ticker,
            "ensemble_accuracy": accuracies.get('ensemble', 0.0),
            "individual_accuracies": accuracies.get('individual', {}),
            "test_size": len(X_test),
            "feature_count": X.shape[1],
            "hyperparams_optimized": optimize_hyperparams,
            "lgb_params": lgb_params,
            "xgb_params": xgb_params,
        }
        
        # モデル保存
        save_model_v2(ticker, ensemble_model, metadata)
        
        # 精度履歴記録
        track_accuracy_history_v2(ticker, accuracies.get('ensemble', 0.0))
        
        logger.info(
            f"[TRAIN_V2] {ticker} complete: "
            f"Ensemble={accuracies.get('ensemble', 0.0):.4f}"
        )
        
        return {
            "ok": True,
            "ticker": ticker,
            "ensemble_accuracy": accuracies.get('ensemble', 0.0),
            "individual_accuracies": accuracies.get('individual', {}),
            "test_size": len(X_test),
            "feature_count": X.shape[1],
            "hyperparams_optimized": optimize_hyperparams,
            "model_saved": True
        }
    
    except Exception as e:
        logger.error(f"[TRAIN_V2] {ticker} fatal: {e}", exc_info=True)
        return {
            "ok": False,
            "ticker": ticker,
            "error": str(e)
        }


# ===========================================================
# 6. 予測関数 (V2)
# ===========================================================

def predict_ticker_v2(ticker: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    V2 アンサンブルで予測
    
    Args:
        ticker: 株式シンボル
        df: 特徴データフレーム
    
    Returns:
        予測結果辞書
    """
    
    try:
        logger.info(f"[PREDICT_V2] {ticker}: Starting prediction")
        
        if df is None or df.empty or "Close" not in df.columns:
            logger.error(f"[PREDICT_V2] {ticker}: Invalid dataframe")
            return None
        
        current_close = float(df["Close"].iloc[-1])
        
        # モデル読み込み
        model = load_model_v2(ticker)
        
        if model is None:
            logger.info(f"[PREDICT_V2] {ticker}: No saved model, training...")
            train_result = train_model_v2(df, ticker, optimize_hyperparams=False)
            if not train_result.get('ok'):
                return None
            model = load_model_v2(ticker)
        
        # 特徴抽出（最新行）
        X_latest = df.drop(columns=["Close"], errors="ignore").iloc[-1:].copy()
        
        # 予測確率
        pred_proba = model.predict_proba(X_latest)[0][1]
        pred_class = model.predict(X_latest)[0]
        
        # 信頼度計算
        confidence_score = abs(pred_proba - 0.5) * 2
        
        # 価格変動幅計算
        price_change_pct = confidence_score * 0.04
        
        if pred_class == 1:
            predicted_price = current_close * (1 + price_change_pct)
            direction = "↑ Bullish"
            prob_up = pred_proba
        else:
            predicted_price = current_close * (1 - price_change_pct)
            direction = "↓ Bearish"
            prob_up = 1.0 - pred_proba
        
        result = {
            "ticker": ticker,
            "current_price": float(current_close),
            "predicted_price": float(predicted_price),
            "predicted_change_pct": float(
                (predicted_price - current_close) / current_close * 100
            ),
            "direction": direction,
            "prob_up": float(prob_up),
            "prob_down": float(1.0 - prob_up),
            "confidence": float(confidence_score),
            "model_version": "v2_ensemble",
            "ensemble_pred": float(pred_proba),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(
            f"[PREDICT_V2] {ticker} complete: "
            f"{direction} @ {predicted_price:.2f}"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"[PREDICT_V2] {ticker} failed: {e}", exc_info=True)
        return None


# ===========================================================
# 7. 精度追跡
# ===========================================================

def track_accuracy_history_v2(ticker: str, accuracy: float) -> bool:
    """V2 精度履歴を記録"""
    
    try:
        history_file = ANALYTICS_DIR / "accuracy_history_v2.json"
        
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
        else:
            history = {}
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in history:
            history[today] = {}
        
        history[today][ticker] = float(accuracy)
        
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"[HISTORY_V2] {ticker}: {accuracy:.4f} recorded")
        return True
    
    except Exception as e:
        logger.error(f"[HISTORY_V2] Failed: {e}")
        return False
