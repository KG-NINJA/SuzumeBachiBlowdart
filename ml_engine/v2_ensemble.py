"""
ml_engine/v2_ensemble.py - アンサンブル予測エンジン

LightGBM + XGBoost + CatBoost のアンサンブル
旧 ml_engine_v2.py からの移行版
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from typing import Optional, Dict, Any, Tuple
import json
import pickle
from pathlib import Path
from datetime import datetime

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from sklearn.ensemble import VotingClassifier
except ImportError:
    VotingClassifier = None

try:
    import optuna
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from .base import BasePredictor
from .config import get_logger, MODELS_V2_DIR, ANALYTICS_DIR

logger = get_logger("ml_engine.v2")


# ===========================================================
# ハイパーパラメータ最適化 (Optuna)
# ===========================================================
def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ticker: str,
    model_type: str = 'lightgbm',
    n_trials: int = 50
) -> Optional[Dict[str, Any]]:
    """
    Optuna でハイパーパラメータを最適化
    """
    if not HAS_OPTUNA:
        logger.warning("[OPTUNA] optuna not installed, skipping optimization")
        return None
    
    try:
        logger.info(f"[OPTUNA] {ticker} ({model_type}): Starting {n_trials} trials...")
        
        def objective(trial):
            if model_type == 'lightgbm' and HAS_LGB:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                }
                model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
            elif model_type == 'xgboost' and HAS_XGB:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                }
                model = XGBClassifier(**params, random_state=42, n_jobs=-1)
            else:
                return 0.5
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', pruner=MedianPruner())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"[OPTUNA] {ticker} ({model_type}): Best={study.best_value:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
        }
    
    except Exception as e:
        logger.error(f"[OPTUNA] {ticker}: {e}")
        return None


# ===========================================================
# モデル訓練関数
# ===========================================================
def train_lightgbm_model(X_train, y_train, X_test, y_test, params=None):
    """LightGBMモデルを訓練"""
    if not HAS_LGB:
        return None, 0.0
    
    try:
        if params is None:
            params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7}
        
        model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        
        logger.info(f"[LGB] Accuracy: {acc:.4f}")
        return model, acc
    except Exception as e:
        logger.error(f"[LGB] Failed: {e}")
        return None, 0.0


def train_xgboost_model(X_train, y_train, X_test, y_test, params=None):
    """XGBoostモデルを訓練"""
    if not HAS_XGB:
        return None, 0.0
    
    try:
        if params is None:
            params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5}
        
        model = XGBClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        
        logger.info(f"[XGB] Accuracy: {acc:.4f}")
        return model, acc
    except Exception as e:
        logger.error(f"[XGB] Failed: {e}")
        return None, 0.0


def train_catboost_model(X_train, y_train, X_test, y_test):
    """CatBoostモデルを訓練"""
    if not HAS_CAT:
        return None, 0.0
    
    try:
        model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        
        logger.info(f"[CAT] Accuracy: {acc:.4f}")
        return model, acc
    except Exception as e:
        logger.error(f"[CAT] Failed: {e}")
        return None, 0.0


# ===========================================================
# アンサンブル構築
# ===========================================================
def create_super_ensemble(X_train, y_train, X_test, y_test, lgb_params=None, xgb_params=None):
    """
    LightGBM + XGBoost + CatBoost のスタッキングアンサンブルを構築
    
    Architecture:
        Level 0: Base Models (LGBM, XGB, CatBoost)
        Level 1: Meta Learner (Logistic Regression) - ベースモデルの予測値を入力として最終判断を行う
    """
    
    logger.info("[ENSEMBLE] Building Stacking Ensemble (Professional Grade)...")
    
    estimators = []
    
    # --- Level 0: Base Models ---
    
    # 1. LightGBM
    if HAS_LGB:
        if lgb_params is None:
            lgb_params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7}
        lgb_clf = lgb.LGBMClassifier(**lgb_params, random_state=42, n_jobs=-1, verbose=-1)
        estimators.append(('lgb', lgb_clf))
    
    # 2. XGBoost
    if HAS_XGB:
        if xgb_params is None:
            xgb_params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5}
        xgb_clf = XGBClassifier(**xgb_params, random_state=42, n_jobs=-1)
        estimators.append(('xgb', xgb_clf))
        
    # 3. CatBoost
    if HAS_CAT:
        cat_clf = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
        estimators.append(('cat', cat_clf))

    if not estimators:
        logger.error("[ENSEMBLE] No models available")
        return None, {}

    try:
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # --- Level 1: Meta Learner ---
        # 予測確率を入力として、最終的な 0/1 を判定する "監督役" のモデル
        # LogisticRegression を使用することで、各モデルの信頼度を係数として学習する
        final_estimator = LogisticRegression()
        
        # StackingClassifier 構築
        # cv=5 で内部的にクロスバリデーションを行い、リークを防ぎながらメタモデルを学習する
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1,
            passthrough=False # 元の特徴量は使わず、純粋にモデルの予測値のみで判断する（純粋なメタ学習）
        )
        
        logger.info("[ENSEMBLE] Training Stacking Meta-Learner...")
        ensemble.fit(X_train, y_train)
        
        # 評価
        ensemble_pred = ensemble.predict(X_test)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        # 個別モデルの精度も確認（参考用）
        individual_accuracies = {}
        for name, model in estimators:
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                individual_accuracies[name] = acc
            except Exception as e:
                logger.warning(f"[ENSEMBLE] Individual calc failed per {name}: {e}")
                individual_accuracies[name] = 0.0

        logger.info(f"[ENSEMBLE] Stacking Accuracy: {ensemble_acc:.4f}")
        logger.info(f"[ENSEMBLE] Meta-Learner Coefficients: {ensemble.final_estimator_.coef_}")

        return ensemble, {'individual': individual_accuracies, 'ensemble': ensemble_acc}

    except ImportError:
        logger.error("[ENSEMBLE] StackingClassifier not found (scikit-learn update required?). Fallback to Voting.")
        # Fallback to Voting if Stacking is not available (old sklearn)
        if VotingClassifier is None:
            return None, {}
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        return ensemble, {'individual': {}, 'ensemble': ensemble_acc}


# ===========================================================
# V2 統合訓練関数
# ===========================================================
def train_model_v2(
    df: pd.DataFrame,
    ticker: str,
    optimize_hyperparams: bool = True,
    n_optuna_trials: int = 50
) -> Dict[str, Any]:
    """
    V2 アンサンブルモデルを訓練
    """
    try:
        logger.info(f"[TRAIN_V2] {ticker}: Starting (hyperopt={optimize_hyperparams})")
        
        if "Close" not in df.columns:
            return {"ok": False, "error": "Close missing"}
        
        df = df.copy()
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        
        if len(df) < 100:
            return {"ok": False, "error": "insufficient_data"}
        
        # 特徴量とターゲット分離
        X = df.drop(columns=["target", "Close"], errors="ignore")
        y = df["target"]
        
        # --- 時系列分割（Walk-Forward Validation）---
        # TimeSeriesSplit を使用して、未来のデータが過去の訓練に漏れ込むのを防止
        # 金融データでは必須のベストプラクティス
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        # Walk-Forward で複数回評価し、最終的には最後の分割で本番モデルを作成
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
            y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]
            logger.debug(f"[TRAIN_V2] {ticker}: Fold {fold+1}/5 - Train={len(X_fold_train)}, Test={len(X_fold_test)}")
        
        # 最終分割をメイン訓練/テストセットとして使用
        final_train_idx, final_test_idx = list(tscv.split(X))[-1]
        X_train, X_test = X.iloc[final_train_idx], X.iloc[final_test_idx]
        y_train, y_test = y.iloc[final_train_idx], y.iloc[final_test_idx]
        
        logger.info(f"[TRAIN_V2] {ticker}: Final Split - Train={len(X_train)}, Test={len(X_test)}")
        
        # ハイパーパラメータ最適化
        lgb_params = xgb_params = None
        if optimize_hyperparams and HAS_OPTUNA:
            lgb_result = optimize_hyperparameters(X_train, y_train, ticker, 'lightgbm', n_optuna_trials)
            if lgb_result:
                lgb_params = lgb_result['best_params']
            
            xgb_result = optimize_hyperparameters(X_train, y_train, ticker, 'xgboost', n_optuna_trials)
            if xgb_result:
                xgb_params = xgb_result['best_params']
        
        # アンサンブル構築
        ensemble, accuracies = create_super_ensemble(X_train, y_train, X_test, y_test, lgb_params, xgb_params)
        
        if ensemble is None:
            return {"ok": False, "error": "ensemble_failed"}
        
        # モデル保存
        save_model_v2(ticker, ensemble, {
            "ticker": ticker,
            "ensemble_accuracy": accuracies.get('ensemble', 0.0),
            "individual": accuracies.get('individual', {}),
        })
        
        return {
            "ok": True,
            "ticker": ticker,
            "ensemble_accuracy": accuracies.get('ensemble', 0.0),
            "individual_accuracies": accuracies.get('individual', {}),
        }
    
    except Exception as e:
        logger.error(f"[TRAIN_V2] {ticker} failed: {e}")
        return {"ok": False, "error": str(e)}


# ===========================================================
# V2 予測関数
# ===========================================================
def predict_ticker_v2(ticker: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    V2 アンサンブルで予測（SHAP説明可能性付き）
    
    Returns:
        予測結果と上位影響特徴量の辞書
    """
    try:
        logger.info(f"[PREDICT_V2] {ticker}: Starting")
        
        if df is None or df.empty or "Close" not in df.columns:
            return None
        
        current_close = float(df["Close"].iloc[-1])
        
        # モデル読み込み
        model = load_model_v2(ticker)
        
        if model is None:
            logger.info(f"[PREDICT_V2] {ticker}: No model, training...")
            result = train_model_v2(df, ticker, optimize_hyperparams=False)
            if not result.get('ok'):
                return None
            model = load_model_v2(ticker)
        
        # 予測
        X_latest = df.drop(columns=["Close"], errors="ignore").iloc[-1:].copy()
        pred_proba = model.predict_proba(X_latest)[0][1]
        pred_class = model.predict(X_latest)[0]
        
        confidence = abs(pred_proba - 0.5) * 2
        price_change = confidence * 0.04
        
        if pred_class == 1:
            predicted_price = current_close * (1 + price_change)
            direction = "↑ Bullish"
        else:
            predicted_price = current_close * (1 - price_change)
            direction = "↓ Bearish"
        
        # --- SHAP 説明可能性分析 ---
        # なぜこの予測をしたのかを上位特徴量で説明
        top_features = []
        try:
            import shap
            
            # StackingClassifier の場合、最初のベースモデルを使って SHAP 値を計算
            # (VotingClassifier の場合も同様)
            base_model = None
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                base_model = model.estimators_[0]  # 最初のベースモデル (通常 LightGBM)
            
            if base_model is not None and hasattr(base_model, 'feature_importances_'):
                # Background data for SHAP (直近100サンプル)
                X_background = df.drop(columns=["Close"], errors="ignore").iloc[-100:].copy()
                
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_latest)
                
                # SHAP値の絶対値でソートして上位5特徴量を取得
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]  # クラス1の SHAP 値
                else:
                    shap_vals = shap_values[0]
                
                feature_importance = list(zip(X_latest.columns, shap_vals))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                top_features = [
                    {"feature": name, "impact": float(val), "direction": "↑" if val > 0 else "↓"}
                    for name, val in feature_importance[:5]
                ]
                
                logger.info(f"[PREDICT_V2] {ticker}: Top SHAP features: {[f['feature'] for f in top_features]}")
                
        except ImportError:
            logger.debug("[PREDICT_V2] SHAP not installed, skipping explainability")
        except Exception as e:
            logger.debug(f"[PREDICT_V2] SHAP analysis failed (non-critical): {e}")
        
        return {
            "ticker": ticker,
            "current_price": float(current_close),
            "predicted_price": float(predicted_price),
            "predicted_change_pct": float((predicted_price - current_close) / current_close * 100),
            "direction": direction,
            "confidence": float(confidence),
            "prob_up": float(pred_proba),
            "model_version": "v2_stacking_ensemble",
            "timestamp": datetime.now().isoformat(),
            "explainability": {
                "top_features": top_features,
                "method": "SHAP TreeExplainer" if top_features else "N/A"
            }
        }
    
    except Exception as e:
        logger.error(f"[PREDICT_V2] {ticker} failed: {e}")
        return None


# ===========================================================
# モデル保存・読み込み
# ===========================================================
def save_model_v2(ticker: str, model, metadata: Dict) -> bool:
    """V2モデルを保存"""
    try:
        ticker_dir = MODELS_V2_DIR / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        with open(ticker_dir / "model_v2.pkl", "wb") as f:
            pickle.dump(model, f)
        
        metadata["saved_at"] = datetime.now().isoformat()
        with open(ticker_dir / "metadata_v2.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"[SAVE_V2] {ticker}: Saved")
        return True
    except Exception as e:
        logger.error(f"[SAVE_V2] {ticker}: {e}")
        return False


def load_model_v2(ticker: str):
    """V2モデルを読み込み"""
    try:
        model_path = MODELS_V2_DIR / ticker / "model_v2.pkl"
        if not model_path.exists():
            return None
        
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"[LOAD_V2] {ticker}: {e}")
        return None


# ===========================================================
# EnsemblePredictor クラス
# ===========================================================
class EnsemblePredictor(BasePredictor):
    """
    アンサンブル予測エンジン (V2)
    
    LightGBM + XGBoost + CatBoost
    """
    
    def __init__(self):
        super().__init__()
        self.version = "v2_ensemble"
    
    def train(self, df: pd.DataFrame, ticker: str, **kwargs) -> Dict[str, Any]:
        """モデルを訓練"""
        optimize = kwargs.get('optimize_hyperparams', True)
        n_trials = kwargs.get('n_optuna_trials', 50)
        result = train_model_v2(df, ticker, optimize_hyperparams=optimize, n_optuna_trials=n_trials)
        self.metadata = result
        return result
    
    def predict(self, ticker: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """予測を実行"""
        return predict_ticker_v2(ticker, df)
    
    def save_model(self, ticker: str) -> bool:
        """モデルを保存"""
        return self.model is not None and save_model_v2(ticker, self.model, self.metadata)
    
    def load_model(self, ticker: str) -> bool:
        """モデルを読み込み"""
        self.model = load_model_v2(ticker)
        return self.model is not None
