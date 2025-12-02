"""
simple_daily_prediction.py - Dual Engine Support
フラグ一つで ML エンジン v1 or v2 を切り替え可能
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from blowdart_features import build_feature_set

# ===== エンジン選択フラグ =====
USE_V2_ENGINE = True  # ← True で v2, False で v1 を使用

# エンジン インポート
try:
    from ml_engine_v2 import train_model_v2, predict_ticker_v2
    v2_available = True
except ImportError:
    logging.warning("ml_engine_v2 not available, falling back to v1")
    v2_available = False

from blowdart_ml_engine import train_model, predict_ticker, train_ticker
from confidence_filter import apply_confidence_filter, generate_confidence_report, generate_confidence_markdown

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_daily_prediction")

# ディレクトリ設定
PREDICTIONS_DIR = Path("daily_predictions")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================
# エンジンマネージャー
# ===========================================================

class DualEngineManager:
    """V1 と V2 のエンジンを統一インターフェースで管理"""
    
    def __init__(self, use_v2: bool = True):
        self.use_v2 = use_v2 and v2_available
        self.version = "v2" if self.use_v2 else "v1"
        logger.info(f"[ENGINE] Using {self.version.upper()} engine")
    
    def train(self, df: pd.DataFrame, ticker: str, **kwargs) -> dict:
        """
        訓練処理
        
        Args:
            df: 特徴データフレーム
            ticker: 株式シンボル
            **kwargs: エンジン固有のパラメータ
        
        Returns:
            訓練結果辞書
        """
        
        if self.use_v2:
            # V2: ハイパラ最適化オプション
            optimize_hyperparams = kwargs.get('optimize_hyperparams', True)
            n_optuna_trials = kwargs.get('n_optuna_trials', 50)
            
            result = train_model_v2(
                df, ticker,
                optimize_hyperparams=optimize_hyperparams,
                n_optuna_trials=n_optuna_trials
            )
        else:
            # V1: 従来の訓練
            result = train_model(df, ticker, use_existing=True)
        
        # 結果に version タグを追加
        result['engine_version'] = self.version
        
        return result
    
    def predict(self, ticker: str, df: pd.DataFrame) -> dict:
        """
        予測処理
        
        Args:
            ticker: 株式シンボル
            df: 特徴データフレーム
        
        Returns:
            予測結果辞書
        """
        
        if self.use_v2:
            result = predict_ticker_v2(ticker, df)
        else:
            result = predict_ticker(ticker, df)
        
        if result:
            result['engine_version'] = self.version
        
        return result


# ===========================================================
# メイン: 日次予測パイプライン
# ===========================================================

def run_daily_prediction(
    tickers: list,
    price_fetcher,
    use_v2: bool = USE_V2_ENGINE,
    optimize_hyperparams: bool = True,
    min_confidence: float = 0.30
) -> dict:
    """
    日次予測を実行
    
    Args:
        tickers: 予測対象の株式シンボル列
        price_fetcher: 価格データ取得関数
        use_v2: V2 エンジンを使用するか
        optimize_hyperparams: ハイパーパラメータ最適化を実行するか
        min_confidence: 信頼度フィルタの最小値
    
    Returns:
        日次予測結果サマリー
    """
    
    start_time = datetime.now()
    logger.info("="*70)
    logger.info(f"[PIPELINE] Starting daily prediction run")
    logger.info(f"[PIPELINE] Engine Version: {'V2 (Advanced)' if use_v2 else 'V1 (Classic)'}")
    logger.info(f"[PIPELINE] Tickers: {len(tickers)} symbols")
    logger.info("="*70)
    
    # エンジン初期化
    engine = DualEngineManager(use_v2=use_v2)
    
    # 結果格納
    all_predictions = []
    training_results = []
    failed_tickers = []
    
    # 各銘柄を処理
    for ticker in tickers:
        try:
            logger.info(f"\n[{ticker}] Processing...")
            
            # 1. 価格データ取得
            price_data = price_fetcher(ticker)
            
            if price_data is None or price_data.empty:
                logger.warning(f"[{ticker}] Failed to fetch price data")
                failed_tickers.append(ticker)
                continue
            
            logger.info(f"[{ticker}] Fetched {len(price_data)} price records")
            
            # 2. 特徴エンジニアリング
            features = build_feature_set(price_data, ticker)
            
            if features is None or features.empty:
                logger.warning(f"[{ticker}] Feature engineering failed")
                failed_tickers.append(ticker)
                continue
            
            logger.info(f"[{ticker}] Built {features.shape[1]} features")
            
            # 3. モデル訓練
            kwargs = {}
            if use_v2:
                kwargs['optimize_hyperparams'] = optimize_hyperparams
                kwargs['n_optuna_trials'] = 50  # ← 調整可能
            
            train_result = engine.train(features, ticker, **kwargs)
            training_results.append(train_result)
            
            if not train_result.get('ok', False):
                logger.warning(f"[{ticker}] Training failed: {train_result.get('error')}")
                failed_tickers.append(ticker)
                continue
            
            logger.info(
                f"[{ticker}] Training complete: "
                f"Accuracy={train_result.get('ensemble_accuracy', train_result.get('hybrid_acc', 'N/A'))}"
            )
            
            # 4. 予測
            prediction = engine.predict(ticker, features)
            
            if prediction is None:
                logger.warning(f"[{ticker}] Prediction failed")
                failed_tickers.append(ticker)
                continue
            
            all_predictions.append(prediction)
            
            logger.info(
                f"[{ticker}] Prediction: "
                f"{prediction['direction']} @ {prediction['predicted_price']:.2f} "
                f"(confidence={prediction['confidence']:.2%})"
            )
        
        except Exception as e:
            logger.error(f"[{ticker}] Fatal error: {e}", exc_info=True)
            failed_tickers.append(ticker)
            continue
    
    # 結果処理
    logger.info(f"\n[RESULTS] Total predictions: {len(all_predictions)}/{len(tickers)}")
    
    # 信頼度フィルタリング
    filtered_predictions = apply_confidence_filter(all_predictions, min_confidence)
    
    # レポート生成
    report = generate_confidence_report(filtered_predictions)
    markdown = generate_confidence_markdown(report, filtered_predictions)
    
    # ファイル保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 予測結果 JSON
    predictions_file = PREDICTIONS_DIR / f"predictions_{timestamp}.json"
    with open(predictions_file, "w") as f:
        json.dump(all_predictions, f, indent=2, default=str)
    logger.info(f"[SAVE] Predictions → {predictions_file}")
    
    # 最新予測リンク
    latest_predictions = PREDICTIONS_DIR / "latest_predictions.json"
    with open(latest_predictions, "w") as f:
        json.dump(all_predictions, f, indent=2, default=str)
    
    # フィルタ済み予測 JSON
    filtered_file = PREDICTIONS_DIR / f"filtered_predictions_{timestamp}.json"
    with open(filtered_file, "w") as f:
        json.dump(filtered_predictions, f, indent=2, default=str)
    logger.info(f"[SAVE] Filtered predictions → {filtered_file}")
    
    # レポート JSON
    report_file = PREDICTIONS_DIR / f"confidence_report_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"[SAVE] Report (JSON) → {report_file}")
    
    # Latest report link
    latest_report = PREDICTIONS_DIR / "confidence_report.json"
    with open(latest_report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # レポート Markdown
    markdown_file = PREDICTIONS_DIR / f"confidence_report_{timestamp}.md"
    with open(markdown_file, "w") as f:
        f.write(markdown)
    logger.info(f"[SAVE] Report (MD) → {markdown_file}")
    
    # Latest markdown link
    latest_markdown = PREDICTIONS_DIR / "confidence_report.md"
    with open(latest_markdown, "w") as f:
        f.write(markdown)
    
    # 訓練結果ログ
    training_log_file = LOGS_DIR / f"training_log_{timestamp}.json"
    with open(training_log_file, "w") as f:
        json.dump(training_results, f, indent=2, default=str)
    logger.info(f"[SAVE] Training log → {training_log_file}")
    
    # 実行時間
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 最終サマリー
    summary = {
        "timestamp": datetime.now().isoformat(),
        "engine_version": engine.version.upper(),
        "total_tickers": len(tickers),
        "successful": len(all_predictions),
        "failed": len(failed_tickers),
        "execute_count": report.get('execute_count', 0),
        "skip_count": report.get('skip_count', 0),
        "average_confidence": report.get('average_confidence', 0),
        "duration_seconds": duration,
        "failed_tickers": failed_tickers,
        "model_version": engine.version
    }
    
    logger.info("="*70)
    logger.info(f"[SUMMARY] Successful: {summary['successful']}/{summary['total_tickers']}")
    logger.info(f"[SUMMARY] Execute: {summary['execute_count']} | Skip: {summary['skip_count']}")
    logger.info(f"[SUMMARY] Avg Confidence: {summary['average_confidence']:.1%}")
    logger.info(f"[SUMMARY] Duration: {duration:.1f}s")
    logger.info(f"[SUMMARY] Engine: {engine.version.upper()}")
    logger.info("="*70)
    
    return summary


# ===========================================================
# 使用例
# ===========================================================

if __name__ == "__main__":
    """
    実行例:
        python simple_daily_prediction.py
    
    環境変数:
        USE_V2=1  → V2 エンジンを使用
        USE_V2=0  → V1 エンジンを使用
    """
    
    import os
    import yfinance as yf
    
    # 環境変数からエンジン選択
    use_v2_env = os.environ.get('USE_V2', '1').lower() == '1'
    
    # テスト用銘柄
    tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    # 価格取得関数
    def fetch_price_data(ticker, period='1y'):
        try:
            df = yf.download(ticker, period=period, progress=False)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return None
    
    # 日次予測実行
    summary = run_daily_prediction(
        tickers=tickers,
        price_fetcher=fetch_price_data,
        use_v2=use_v2_env,
        optimize_hyperparams=False,  # 速度優先の場合は False に
        min_confidence=0.30
    )
    
    print(json.dumps(summary, indent=2))
