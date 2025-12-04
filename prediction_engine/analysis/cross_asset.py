"""
cross_asset.py - クロスアセット相関分析モジュール

株価予測から他の資産クラスへの応用可能性を分析:
- 為替レート（USD/JPY, EUR/USD など）
- セクターETF（XLK, SMH など）
- マクロ指標（金利, VIX, 商品）
- サプライチェーン連動

Author: SuzumeBachiBlowdart Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import warnings

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    warnings.warn("yfinanceがインストールされていません")

logger = logging.getLogger("prediction_engine.cross_asset")


class CrossAssetCorrelation:
    """
    株価予測から他の資産クラスへの応用可能性を分析
    
    分析対象:
    1. 為替レート: USD/JPY, EUR/USD など
    2. セクターETF: XLK, XLC, SMH など
    3. マクロ指標: TLT（金利）, GLD（金）, USO（原油）, VIX
    4. サプライチェーン: 関連銘柄
    """
    
    # 資産クラス定義
    ASSET_CLASSES = {
        'currency': {
            'symbols': ['JPY=X', 'EURUSD=X', 'GBPUSD=X'],
            'names': ['USD/JPY', 'EUR/USD', 'GBP/USD'],
            'expected_accuracy': (0.55, 0.65)
        },
        'sector': {
            'symbols': ['XLK', 'XLC', 'SMH', 'SOXX'],
            'names': ['Technology ETF', 'Communication ETF', 'Semiconductor ETF', 'Semiconductor Index'],
            'expected_accuracy': (0.60, 0.70)
        },
        'macro': {
            'symbols': ['TLT', 'GLD', 'USO', '^VIX'],
            'names': ['20Y Treasury', 'Gold', 'Oil', 'VIX'],
            'expected_accuracy': (0.50, 0.60)
        },
        'tech_leaders': {
            'symbols': ['MSFT', 'GOOGL', 'META', 'AMZN'],
            'names': ['Microsoft', 'Alphabet', 'Meta', 'Amazon'],
            'expected_accuracy': (0.55, 0.65)
        }
    }
    
    # セクター-銘柄マッピング
    SECTOR_MAPPING = {
        'NVDA': ['SMH', 'SOXX', 'XLK', 'AMD', 'INTC', 'TSM'],
        'AAPL': ['XLK', 'QQQ', 'MSFT', 'GOOGL'],
        'MSFT': ['XLK', 'QQQ', 'AAPL', 'GOOGL', 'CRM'],
        'GOOGL': ['XLC', 'META', 'MSFT', 'AMZN'],
        'META': ['XLC', 'GOOGL', 'SNAP', 'PINS'],
        'AMZN': ['XLY', 'WMT', 'TGT', 'SHOP'],
        'TSLA': ['XLY', 'RIVN', 'LCID', 'F', 'GM']
    }
    
    def __init__(self, lookback_days: int = 60):
        """
        初期化
        
        Args:
            lookback_days: 相関計算に使用する過去日数
        """
        self.lookback_days = lookback_days
        self.cache = {}  # データキャッシュ
        logger.info(f"[INIT] CrossAssetCorrelation 初期化完了 (lookback: {lookback_days}日)")
    
    def analyze(
        self, 
        ticker: str,
        stock_prediction: Dict[str, Any],
        stock_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        株価予測から他資産への相関信号を分析
        
        Args:
            ticker: 銘柄コード
            stock_prediction: MultiTimeScaleForecastの予測結果
            stock_data: 株価データ（Noneの場合はAPIから取得）
        
        Returns:
            相関分析結果
        """
        logger.info(f"[ANALYZE] {ticker} のクロスアセット分析開始")
        
        # データ準備
        stock_df = self._get_stock_data(ticker, stock_data)
        if stock_df is None or len(stock_df) < 30:
            logger.warning(f"[ANALYZE] {ticker} のデータが不十分です")
            return self._generate_fallback_result(ticker)
        
        result = {
            "stock": ticker,
            "timestamp": datetime.now().isoformat(),
            "correlations": {}
        }
        
        # 予測方向を取得
        pred_direction = self._get_prediction_direction(stock_prediction)
        
        # 各資産クラスの相関分析
        for asset_class, config in self.ASSET_CLASSES.items():
            result['correlations'][asset_class] = {}
            
            for symbol, name in zip(config['symbols'], config['names']):
                try:
                    correlation_result = self._analyze_single_asset(
                        stock_df, symbol, name, pred_direction
                    )
                    if correlation_result:
                        result['correlations'][asset_class][symbol] = correlation_result
                except Exception as e:
                    logger.warning(f"[ANALYZE] {symbol} の分析に失敗: {e}")
        
        # セクター相関（銘柄固有）
        if ticker in self.SECTOR_MAPPING:
            result['correlations']['related'] = {}
            for related_symbol in self.SECTOR_MAPPING[ticker][:3]:  # 上位3つ
                try:
                    correlation_result = self._analyze_single_asset(
                        stock_df, related_symbol, related_symbol, pred_direction
                    )
                    if correlation_result:
                        result['correlations']['related'][related_symbol] = correlation_result
                except Exception as e:
                    logger.warning(f"[ANALYZE] {related_symbol} の分析に失敗: {e}")
        
        # サマリー生成
        result['summary'] = self._generate_summary(result['correlations'], pred_direction)
        
        logger.info(f"[ANALYZE] {ticker} のクロスアセット分析完了")
        
        return result
    
    def _get_stock_data(
        self, 
        ticker: str, 
        provided_data: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """株価データを取得"""
        if provided_data is not None:
            return provided_data
        
        if not HAS_YFINANCE:
            return None
        
        # キャッシュチェック
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            period = f"{self.lookback_days}d"
            data = yf.download(ticker, period=period, progress=False)
            self.cache[cache_key] = data
            return data
        except Exception as e:
            logger.error(f"[DATA] {ticker} のデータ取得失敗: {e}")
            return None
    
    def _get_asset_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """資産データを取得"""
        if not HAS_YFINANCE:
            return None
        
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            period = f"{self.lookback_days}d"
            data = yf.download(symbol, period=period, progress=False)
            self.cache[cache_key] = data
            return data
        except Exception as e:
            logger.warning(f"[DATA] {symbol} のデータ取得失敗: {e}")
            return None
    
    def _get_prediction_direction(self, prediction: Dict[str, Any]) -> str:
        """予測から方向を抽出"""
        if 'predictions' in prediction:
            short_term = prediction['predictions'].get('short_term', {})
            return short_term.get('direction', 'NEUTRAL')
        return 'NEUTRAL'
    
    def _analyze_single_asset(
        self, 
        stock_df: pd.DataFrame,
        asset_symbol: str,
        asset_name: str,
        pred_direction: str
    ) -> Optional[Dict[str, Any]]:
        """単一資産との相関を分析"""
        # 資産データ取得
        asset_df = self._get_asset_data(asset_symbol)
        if asset_df is None or len(asset_df) < 20:
            return None
        
        # リターン計算
        stock_returns = stock_df['Close'].pct_change().dropna()
        asset_returns = asset_df['Close'].pct_change().dropna()
        
        # 共通期間で揃える
        common_idx = stock_returns.index.intersection(asset_returns.index)
        if len(common_idx) < 20:
            return None
        
        stock_ret = stock_returns.loc[common_idx]
        asset_ret = asset_returns.loc[common_idx]
        
        # 相関係数計算
        correlation = float(np.corrcoef(stock_ret, asset_ret)[0, 1])
        
        # ラグ相関（1-5日）
        lag_correlations = []
        for lag in range(1, 6):
            if len(stock_ret) > lag:
                lagged_corr = float(np.corrcoef(
                    stock_ret.iloc[:-lag], 
                    asset_ret.iloc[lag:]
                )[0, 1])
                lag_correlations.append((lag, lagged_corr))
        
        # 最適ラグを見つける
        best_lag = max(lag_correlations, key=lambda x: abs(x[1])) if lag_correlations else (0, correlation)
        
        # 予測信号
        if correlation > 0:
            predicted_direction = pred_direction
        else:
            predicted_direction = "DOWN" if pred_direction == "UP" else "UP"
        
        # 信頼度計算
        confidence = abs(correlation) * 0.8  # 相関の強さを信頼度に変換
        
        return {
            "name": asset_name,
            "correlation": round(correlation, 4),
            "correlation_strength": self._get_correlation_strength(correlation),
            "predicted_direction": predicted_direction,
            "inverse_correlation": correlation < 0,
            "lag_days": best_lag[0],
            "lag_correlation": round(best_lag[1], 4),
            "confidence": round(confidence, 4),
            "impact_level": self._get_impact_level(correlation)
        }
    
    def _get_correlation_strength(self, corr: float) -> str:
        """相関の強さを判定"""
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "STRONG"
        elif abs_corr >= 0.4:
            return "MODERATE"
        elif abs_corr >= 0.2:
            return "WEAK"
        else:
            return "NEGLIGIBLE"
    
    def _get_impact_level(self, corr: float) -> str:
        """影響度を判定"""
        abs_corr = abs(corr)
        if abs_corr >= 0.6:
            return "HIGH"
        elif abs_corr >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_summary(
        self, 
        correlations: Dict[str, Any],
        pred_direction: str
    ) -> Dict[str, Any]:
        """分析サマリーを生成"""
        total_assets = 0
        strong_correlations = []
        
        for asset_class, assets in correlations.items():
            for symbol, data in assets.items():
                total_assets += 1
                if data.get('correlation_strength') == 'STRONG':
                    strong_correlations.append({
                        'symbol': symbol,
                        'name': data['name'],
                        'correlation': data['correlation']
                    })
        
        return {
            "prediction_direction": pred_direction,
            "total_assets_analyzed": total_assets,
            "strong_correlations_count": len(strong_correlations),
            "top_correlated_assets": strong_correlations[:3],
            "recommendation": self._generate_recommendation(strong_correlations, pred_direction)
        }
    
    def _generate_recommendation(
        self, 
        strong_corrs: List[Dict],
        direction: str
    ) -> str:
        """推奨テキストを生成"""
        if not strong_corrs:
            return "強い相関を持つ資産は見つかりませんでした。"
        
        top = strong_corrs[0]
        if top['correlation'] > 0:
            return f"{top['name']}との正の相関が強く、同方向の動きが期待されます。"
        else:
            return f"{top['name']}との負の相関が強く、逆方向の動きが期待されます。"
    
    def _generate_fallback_result(self, ticker: str) -> Dict[str, Any]:
        """フォールバック結果を生成"""
        return {
            "stock": ticker,
            "timestamp": datetime.now().isoformat(),
            "correlations": {},
            "summary": {
                "prediction_direction": "NEUTRAL",
                "total_assets_analyzed": 0,
                "strong_correlations_count": 0,
                "top_correlated_assets": [],
                "recommendation": "データ不足のため分析できませんでした。"
            },
            "error": "Insufficient data for correlation analysis"
        }


# ========== テスト用コード ==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== CrossAssetCorrelation テスト ===\n")
    
    # サンプル予測データ
    sample_prediction = {
        "ticker": "NVDA",
        "predictions": {
            "short_term": {
                "direction": "UP",
                "probability": 0.72,
                "confidence": 0.44
            }
        }
    }
    
    # 分析実行
    analyzer = CrossAssetCorrelation(lookback_days=60)
    result = analyzer.analyze("NVDA", sample_prediction)
    
    print("\n[RESULT] クロスアセット相関分析結果:")
    print(f"  銘柄: {result['stock']}")
    
    for asset_class, assets in result['correlations'].items():
        print(f"\n  【{asset_class}】")
        for symbol, data in assets.items():
            print(f"    {symbol} ({data['name']})")
            print(f"      相関: {data['correlation']:.4f} ({data['correlation_strength']})")
            print(f"      予測方向: {data['predicted_direction']}")
            print(f"      信頼度: {data['confidence']:.4f}")
    
    print(f"\n  [サマリー]")
    summary = result['summary']
    print(f"    分析資産数: {summary['total_assets_analyzed']}")
    print(f"    強相関数: {summary['strong_correlations_count']}")
    print(f"    推奨: {summary['recommendation']}")
