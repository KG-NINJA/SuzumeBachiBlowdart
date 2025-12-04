"""
main.py - 予測エンジンのメインエントリーポイント

使用例:
    python -m prediction_engine.main --ticker NVDA --period 6mo

Author: SuzumeBachiBlowdart Team
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from prediction_engine.models.multitimescale import MultiTimeScaleForecast
from prediction_engine.models.confidence_scoring import ConfidenceScoring
from prediction_engine.analysis.cross_asset import CrossAssetCorrelation
from prediction_engine.analysis.scenario import ScenarioAnalysis
from prediction_engine.analysis.risk_framework import RiskFramework
from prediction_engine.output.formatters import OutputFormatter


def setup_logging(level: str = "INFO") -> None:
    """ロギング設定"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_full_analysis(
    ticker: str,
    period: str = "6mo",
    output_formats: list = ["json", "markdown"],
    use_lstm: bool = True
) -> Dict[str, Any]:
    """
    フル分析パイプラインを実行
    
    Args:
        ticker: 銘柄コード
        period: データ期間
        output_formats: 出力形式
        use_lstm: LSTMを使用するか
    
    Returns:
        分析結果の辞書
    """
    logger = logging.getLogger("prediction_engine.main")
    logger.info(f"=== {ticker} のフル分析開始 ===")
    
    # データ取得
    if not HAS_YFINANCE:
        raise ImportError("yfinanceがインストールされていません: pip install yfinance")
    
    logger.info(f"[DATA] {ticker} のデータを取得中 (期間: {period})...")
    data = yf.download(ticker, period=period, progress=False)
    
    if len(data) < 50:
        raise ValueError(f"データが不足しています: {len(data)} 行")
    
    current_price = float(data['Close'].iloc[-1])
    logger.info(f"[DATA] 取得完了: {len(data)} 行, 現在価格: ${current_price:.2f}")
    
    result = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "current_price": current_price,
        "data_points": len(data)
    }
    
    # 1. マルチタイムスケール予測
    logger.info("[PHASE 1] マルチタイムスケール予測...")
    mts = MultiTimeScaleForecast(use_lstm=use_lstm)
    train_result = mts.train(data, ticker, epochs=30, verbose=0)
    prediction = mts.predict(ticker, data, current_price)
    
    result["predictions"] = prediction["predictions"]
    result["model_info"] = prediction["model_info"]
    result["training_accuracies"] = train_result["accuracies"]
    
    # 2. 信頼度スコアリング
    logger.info("[PHASE 2] 信頼度スコアリング...")
    scorer = ConfidenceScoring()
    confidence = scorer.score(prediction, data, train_result["accuracies"])
    result["confidence"] = confidence
    
    # 3. クロスアセット相関分析
    logger.info("[PHASE 3] クロスアセット相関分析...")
    correlator = CrossAssetCorrelation()
    correlations = correlator.analyze(ticker, prediction, data)
    result["correlations"] = correlations["correlations"]
    result["correlation_summary"] = correlations["summary"]
    
    # 4. シナリオ分析
    logger.info("[PHASE 4] シナリオ分析...")
    scenario_analyzer = ScenarioAnalysis()
    scenarios = scenario_analyzer.generate_scenarios(ticker, prediction, data)
    result["scenarios"] = scenarios["scenarios"]
    result["weighted_forecast"] = scenarios["weighted_forecast"]
    result["risk_reward"] = scenarios["risk_reward"]
    result["recommendation"] = scenarios["recommendation"]
    
    # 5. リスク評価
    logger.info("[PHASE 5] リスク評価...")
    risk_framework = RiskFramework()
    risk = risk_framework.evaluate_risk(prediction, data, train_result["accuracies"])
    result["risk_evaluation"] = {
        "prediction_risk": risk["prediction_risk"],
        "tail_risk": risk["tail_risk"],
        "overall_risk_score": risk["overall_risk_score"],
        "risk_level": risk["risk_level"],
        "risk_recommendation": risk["recommendation"]
    }
    
    # 6. 出力
    logger.info("[PHASE 6] 結果出力...")
    formatter = OutputFormatter(output_dir="prediction_output")
    output_files = formatter.export(result, formats=output_formats)
    result["output_files"] = output_files
    
    logger.info(f"=== {ticker} の分析完了 ===")
    
    return result


def print_summary(result: Dict[str, Any]) -> None:
    """結果サマリーを表示"""
    print("\n" + "=" * 60)
    print(f"📊 {result['ticker']} 予測サマリー")
    print("=" * 60)
    
    print(f"\n現在価格: ${result['current_price']:,.2f}")
    print(f"データポイント: {result['data_points']} 件")
    
    print("\n【マルチタイムスケール予測】")
    for term, pred in result['predictions'].items():
        direction = pred.get('direction', 'N/A')
        emoji = "📈" if direction == "UP" else "📉" if direction == "DOWN" else "➡️"
        confidence = pred.get('confidence', 0)
        
        if 'forecast_3d' in pred:
            print(f"  短期: {emoji} ${pred['forecast_3d']:,.2f} (信頼度: {confidence:.1%})")
        elif 'forecast_4w' in pred:
            print(f"  中期: {emoji} ${pred['forecast_4w']:,.2f} (信頼度: {confidence:.1%})")
        elif 'forecast_3m' in pred:
            print(f"  長期: {emoji} ${pred['forecast_3m']:,.2f} (信頼度: {confidence:.1%})")
    
    print("\n【リスク評価】")
    risk = result['risk_evaluation']
    print(f"  総合リスクスコア: {risk['overall_risk_score']:.2f} ({risk['risk_level']})")
    
    print("\n【推奨アクション】")
    rec = result['recommendation']
    print(f"  {rec['action']}: {rec['reasoning']}")
    
    print("\n【出力ファイル】")
    for fmt, path in result.get('output_files', {}).items():
        print(f"  {fmt}: {path}")
    
    print("\n" + "=" * 60)


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="SuzumeBachiBlowdart 予測エンジン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python -m prediction_engine.main --ticker NVDA
  python -m prediction_engine.main --ticker AAPL --period 1y --no-lstm
  python -m prediction_engine.main --ticker MSFT --output json markdown
        """
    )
    
    parser.add_argument(
        "--ticker", "-t",
        required=True,
        help="銘柄コード (例: NVDA, AAPL, MSFT)"
    )
    
    parser.add_argument(
        "--period", "-p",
        default="6mo",
        help="データ期間 (例: 1mo, 3mo, 6mo, 1y). デフォルト: 6mo"
    )
    
    parser.add_argument(
        "--output", "-o",
        nargs="+",
        default=["json", "markdown"],
        choices=["json", "csv", "markdown"],
        help="出力形式. デフォルト: json markdown"
    )
    
    parser.add_argument(
        "--no-lstm",
        action="store_true",
        help="LSTMを使用しない（高速化）"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル. デフォルト: INFO"
    )
    
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="結果をJSON形式のみで出力（サマリー表示なし）"
    )
    
    args = parser.parse_args()
    
    # ロギング設定
    setup_logging(args.log_level)
    
    try:
        # フル分析実行
        result = run_full_analysis(
            ticker=args.ticker.upper(),
            period=args.period,
            output_formats=args.output,
            use_lstm=not args.no_lstm
        )
        
        if args.json_only:
            # JSON出力のみ
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            # サマリー表示
            print_summary(result)
        
        return 0
        
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
