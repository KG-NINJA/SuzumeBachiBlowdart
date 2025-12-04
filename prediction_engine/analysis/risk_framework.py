"""
risk_framework.py - 多次元リスク評価フレームワーク

予測に伴うリスクを多角的に評価:
- 予測リスク: モデル/データ/レジーム変化
- テールリスク: VaR/CVaR
- 相関崩壊リスク
- システマティックリスク

Author: SuzumeBachiBlowdart Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger("prediction_engine.risk_framework")


class RiskLevel(Enum):
    """リスクレベル"""
    LOW = "LOW"           # 0.0-0.3
    MODERATE = "MODERATE" # 0.3-0.6
    HIGH = "HIGH"         # 0.6-0.8
    EXTREME = "EXTREME"   # 0.8-1.0


@dataclass
class RiskBreakdown:
    """リスク内訳"""
    model_risk: float       # モデル不確実性リスク
    data_risk: float        # データ品質リスク
    regime_risk: float      # レジーム変化リスク
    black_swan_risk: float  # ブラックスワンリスク
    systematic_risk: float  # システマティックリスク
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "model_risk": round(self.model_risk, 4),
            "data_risk": round(self.data_risk, 4),
            "regime_risk": round(self.regime_risk, 4),
            "black_swan_risk": round(self.black_swan_risk, 4),
            "systematic_risk": round(self.systematic_risk, 4)
        }


class RiskFramework:
    """
    予測に伴うリスクを多角的に評価
    
    評価項目:
    1. 予測リスク
       - モデルリスク: モデル不確実性
       - データリスク: データ品質問題
       - レジームリスク: 市場環境変化
       - ブラックスワンリスク: 予期しないイベント
       - システマティックリスク: 市場全体のリスク
    
    2. テールリスク
       - Value at Risk (VaR): 95%, 99%
       - Conditional VaR (CVaR): 95%, 99%
    
    3. 相関崩壊リスク
       - 最大ドローダウン相関
       - ボラティリティスパイク確率
    """
    
    # リスクレベル閾値
    THRESHOLDS = {
        RiskLevel.LOW: 0.3,
        RiskLevel.MODERATE: 0.6,
        RiskLevel.HIGH: 0.8
    }
    
    # VaR/CVaR信頼水準
    VAR_CONFIDENCE = [0.95, 0.99]
    
    def __init__(self):
        """初期化"""
        logger.info("[INIT] RiskFramework 初期化完了")
    
    def evaluate_risk(
        self, 
        prediction: Dict[str, Any],
        historical_data: pd.DataFrame,
        model_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        予測に対する総合リスク評価を実行
        
        Args:
            prediction: 予測結果辞書
            historical_data: 過去の価格データ
            model_metrics: モデルの評価指標（オプション）
        
        Returns:
            リスク評価結果
        """
        logger.info(f"[RISK] リスク評価開始: {prediction.get('ticker', 'UNKNOWN')}")
        
        # 1. 予測リスク評価
        prediction_risk = self._evaluate_prediction_risk(
            prediction, historical_data, model_metrics
        )
        
        # 2. テールリスク評価
        tail_risk = self._evaluate_tail_risk(historical_data)
        
        # 3. 相関崩壊リスク評価
        correlation_breakdown = self._evaluate_correlation_breakdown(historical_data)
        
        # 総合リスクスコア計算
        overall_score = self._calculate_overall_risk_score(
            prediction_risk, tail_risk, correlation_breakdown
        )
        
        # リスクレベル判定
        risk_level = self._determine_risk_level(overall_score)
        
        result = {
            "ticker": prediction.get('ticker', 'UNKNOWN'),
            "timestamp": datetime.now().isoformat(),
            "prediction_risk": prediction_risk.to_dict(),
            "tail_risk": tail_risk,
            "correlation_breakdown": correlation_breakdown,
            "overall_risk_score": round(overall_score, 4),
            "risk_level": risk_level.value,
            "recommendation": self._generate_risk_recommendation(risk_level, prediction_risk)
        }
        
        logger.info(f"[RISK] 評価完了: {risk_level.value} ({overall_score:.4f})")
        
        return result
    
    def _evaluate_prediction_risk(
        self,
        prediction: Dict[str, Any],
        data: pd.DataFrame,
        model_metrics: Optional[Dict[str, float]] = None
    ) -> RiskBreakdown:
        """予測リスクを評価"""
        
        # 1. モデルリスク
        model_risk = self._calculate_model_risk(prediction, model_metrics)
        
        # 2. データリスク
        data_risk = self._calculate_data_risk(data)
        
        # 3. レジームリスク
        regime_risk = self._calculate_regime_risk(data)
        
        # 4. ブラックスワンリスク
        black_swan_risk = self._calculate_black_swan_risk(data)
        
        # 5. システマティックリスク
        systematic_risk = self._calculate_systematic_risk(data)
        
        return RiskBreakdown(
            model_risk=model_risk,
            data_risk=data_risk,
            regime_risk=regime_risk,
            black_swan_risk=black_swan_risk,
            systematic_risk=systematic_risk
        )
    
    def _calculate_model_risk(
        self,
        prediction: Dict[str, Any],
        model_metrics: Optional[Dict[str, float]] = None
    ) -> float:
        """モデル不確実性リスクを計算"""
        risks = []
        
        # 予測確率から不確実性を推定
        for term in ['short_term', 'medium_term', 'long_term']:
            if term in prediction.get('predictions', {}):
                prob = prediction['predictions'][term].get('probability', 0.5)
                # 0.5に近いほど不確実
                uncertainty = 1 - abs(prob - 0.5) * 2
                risks.append(uncertainty)
        
        # モデル精度からのリスク
        if model_metrics:
            accuracies = [v for v in model_metrics.values() if isinstance(v, (int, float))]
            if accuracies:
                # 精度が低いほどリスク高
                avg_accuracy = np.mean(accuracies)
                accuracy_risk = 1 - avg_accuracy
                risks.append(accuracy_risk)
        
        return float(np.mean(risks)) if risks else 0.5
    
    def _calculate_data_risk(self, data: pd.DataFrame) -> float:
        """データ品質リスクを計算"""
        risks = []
        
        # 欠損値リスク
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        risks.append(min(missing_ratio * 10, 1))
        
        # 外れ値リスク
        if 'Close' in data.columns:
            close = data['Close'].dropna()
            if len(close) > 0 and close.std() > 0:
                z_scores = np.abs((close - close.mean()) / close.std())
                outlier_ratio = (z_scores > 3).sum() / len(z_scores)
                risks.append(min(outlier_ratio * 10, 1))
        
        # データ量リスク（少ないほどリスク高）
        data_length_risk = max(0, 1 - len(data) / 200)
        risks.append(data_length_risk)
        
        return float(np.mean(risks))
    
    def _calculate_regime_risk(self, data: pd.DataFrame) -> float:
        """レジーム変化リスクを計算"""
        if len(data) < 60:
            return 0.5
        
        close = data['Close']
        
        # 直近30日と過去30日のボラティリティを比較
        recent_vol = close.tail(30).pct_change().std()
        past_vol = close.iloc[-60:-30].pct_change().std()
        
        if past_vol > 0:
            vol_ratio = recent_vol / past_vol
            # ボラティリティが大きく変化しているとリスク高
            regime_risk = min(abs(vol_ratio - 1) * 2, 1)
        else:
            regime_risk = 0.5
        
        # トレンド変化も考慮
        recent_trend = close.tail(30).pct_change().mean()
        past_trend = close.iloc[-60:-30].pct_change().mean()
        
        if recent_trend * past_trend < 0:
            # トレンド反転
            regime_risk = min(regime_risk + 0.2, 1)
        
        return float(regime_risk)
    
    def _calculate_black_swan_risk(self, data: pd.DataFrame) -> float:
        """ブラックスワンリスクを計算"""
        if len(data) < 30:
            return 0.1
        
        returns = data['Close'].pct_change().dropna()
        
        # 尖度（kurtosis）を計算
        # 正規分布は3、それより大きいと裾が重い（ブラックスワンリスク高）
        kurtosis = float(stats.kurtosis(returns))
        
        # 尖度を0-1のリスクスコアに変換
        # 尖度3（正規分布）を基準に、6以上で高リスク
        excess_kurtosis = max(0, kurtosis - 3)
        risk = min(excess_kurtosis / 10, 1)
        
        # 過去の極端な動きも考慮
        extreme_moves = (abs(returns) > returns.std() * 3).sum() / len(returns)
        risk = max(risk, min(extreme_moves * 20, 1))
        
        return float(risk)
    
    def _calculate_systematic_risk(self, data: pd.DataFrame) -> float:
        """システマティックリスクを計算"""
        if len(data) < 30:
            return 0.4
        
        # ボラティリティレベルで評価
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # 年率換算
        annual_vol = volatility * np.sqrt(252)
        
        # ボラティリティを0-1のリスクスコアに変換
        # 年率30%を基準に
        risk = min(annual_vol / 0.5, 1)
        
        return float(risk)
    
    def _evaluate_tail_risk(self, data: pd.DataFrame) -> Dict[str, float]:
        """テールリスク（VaR/CVaR）を評価"""
        result = {}
        
        if len(data) < 30:
            return {
                "value_at_risk_95": -0.05,
                "value_at_risk_99": -0.10,
                "conditional_var_95": -0.08,
                "conditional_var_99": -0.15
            }
        
        returns = data['Close'].pct_change().dropna()
        
        # VaR計算（ヒストリカル法）
        for conf in self.VAR_CONFIDENCE:
            var = float(np.percentile(returns, (1 - conf) * 100))
            result[f"value_at_risk_{int(conf*100)}"] = round(var * 100, 2)  # パーセント表示
            
            # CVaR (Expected Shortfall)
            cvar = float(returns[returns <= var].mean())
            result[f"conditional_var_{int(conf*100)}"] = round(cvar * 100, 2)
        
        return result
    
    def _evaluate_correlation_breakdown(self, data: pd.DataFrame) -> Dict[str, Any]:
        """相関崩壊リスクを評価"""
        if len(data) < 60:
            return {
                "max_drawdown_risk": 0.3,
                "volatility_spike_probability": 0.2,
                "correlation_stability": 0.7
            }
        
        close = data['Close']
        
        # 最大ドローダウン計算
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax
        max_drawdown = float(drawdown.min())
        
        # ドローダウンリスク
        dd_risk = min(abs(max_drawdown) * 2, 1)
        
        # ボラティリティスパイク確率
        returns = close.pct_change().dropna()
        vol_20 = returns.rolling(20).std()
        vol_5 = returns.rolling(5).std()
        
        if len(vol_20.dropna()) > 0 and vol_20.mean() > 0:
            vol_ratio = vol_5 / vol_20
            spike_prob = float((vol_ratio > 1.5).sum() / len(vol_ratio.dropna()))
        else:
            spike_prob = 0.2
        
        return {
            "max_drawdown": round(max_drawdown * 100, 2),  # パーセント
            "max_drawdown_risk": round(dd_risk, 4),
            "volatility_spike_probability": round(spike_prob, 4),
            "correlation_stability": round(1 - spike_prob, 4)
        }
    
    def _calculate_overall_risk_score(
        self,
        prediction_risk: RiskBreakdown,
        tail_risk: Dict[str, float],
        correlation_breakdown: Dict[str, Any]
    ) -> float:
        """総合リスクスコアを計算"""
        # 予測リスクの平均（重み40%）
        pred_avg = np.mean([
            prediction_risk.model_risk,
            prediction_risk.data_risk,
            prediction_risk.regime_risk,
            prediction_risk.black_swan_risk,
            prediction_risk.systematic_risk
        ])
        
        # テールリスク（重み30%）
        # VaR 99%を基準に（-10%以上で高リスク）
        var_99 = abs(tail_risk.get('value_at_risk_99', -5))
        tail_score = min(var_99 / 15, 1)  # -15%で満点
        
        # 相関崩壊リスク（重み30%）
        corr_score = correlation_breakdown.get('max_drawdown_risk', 0.3)
        
        # 加重平均
        overall = pred_avg * 0.4 + tail_score * 0.3 + corr_score * 0.3
        
        return float(np.clip(overall, 0, 1))
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """リスクレベルを判定"""
        if score < self.THRESHOLDS[RiskLevel.LOW]:
            return RiskLevel.LOW
        elif score < self.THRESHOLDS[RiskLevel.MODERATE]:
            return RiskLevel.MODERATE
        elif score < self.THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def _generate_risk_recommendation(
        self,
        level: RiskLevel,
        breakdown: RiskBreakdown
    ) -> Dict[str, Any]:
        """リスクに基づく推奨を生成"""
        # 最大リスク要因を特定
        risk_factors = breakdown.to_dict()
        max_factor = max(risk_factors.items(), key=lambda x: x[1])
        
        if level == RiskLevel.LOW:
            action = "PROCEED"
            position_size = 1.0
            message = "リスクは低レベルです。通常通り進めてください。"
        elif level == RiskLevel.MODERATE:
            action = "CAUTION"
            position_size = 0.7
            message = f"中程度のリスクがあります。{max_factor[0]}に注意してください。"
        elif level == RiskLevel.HIGH:
            action = "REDUCE"
            position_size = 0.4
            message = f"高リスク状態です。ポジションサイズを縮小してください。{max_factor[0]}が特に高いです。"
        else:
            action = "AVOID"
            position_size = 0.0
            message = f"極めて高いリスクです。取引を避けることを推奨します。主因: {max_factor[0]}"
        
        return {
            "action": action,
            "recommended_position_size": position_size,
            "message": message,
            "primary_risk_factor": max_factor[0],
            "primary_risk_value": round(max_factor[1], 4)
        }


# ========== テスト用コード ==========
if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== RiskFramework テスト ===\n")
    
    # サンプルデータ取得
    ticker = "NVDA"
    print(f"[TEST] {ticker} のデータを取得中...")
    data = yf.download(ticker, period="6mo", progress=False)
    
    # サンプル予測データ
    sample_prediction = {
        "ticker": ticker,
        "predictions": {
            "short_term": {"probability": 0.65, "confidence": 0.30},
            "medium_term": {"probability": 0.58, "confidence": 0.16},
            "long_term": {"probability": 0.52, "confidence": 0.04}
        }
    }
    
    # モデル評価指標（オプション）
    model_metrics = {
        "short_term_accuracy": 0.68,
        "medium_term_accuracy": 0.55,
        "long_term_accuracy": 0.52
    }
    
    # リスク評価実行
    framework = RiskFramework()
    result = framework.evaluate_risk(sample_prediction, data, model_metrics)
    
    print(f"\n[RESULT] リスク評価結果:")
    print(f"  銘柄: {result['ticker']}")
    print(f"  総合リスクスコア: {result['overall_risk_score']:.4f}")
    print(f"  リスクレベル: {result['risk_level']}")
    
    print(f"\n  [予測リスク内訳]")
    for factor, value in result['prediction_risk'].items():
        bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
        print(f"    {factor:20s}: {bar} {value:.4f}")
    
    print(f"\n  [テールリスク]")
    for metric, value in result['tail_risk'].items():
        print(f"    {metric}: {value:.2f}%")
    
    print(f"\n  [相関崩壊リスク]")
    for metric, value in result['correlation_breakdown'].items():
        if isinstance(value, float):
            print(f"    {metric}: {value:.4f}")
    
    print(f"\n  [推奨]")
    rec = result['recommendation']
    print(f"    アクション: {rec['action']}")
    print(f"    推奨ポジションサイズ: {rec['recommended_position_size']:.0%}")
    print(f"    メッセージ: {rec['message']}")
