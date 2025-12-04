"""
confidence_scoring.py - 多角的信頼度評価システム

予測の信頼性を5つの要因から総合評価:
1. モデル不確実性 (20%)
2. データ品質 (15%)
3. マーケット状況 (30%)
4. 予測安定性 (20%)
5. 外部ファクター (15%)

Author: SuzumeBachiBlowdart Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("prediction_engine.confidence_scoring")


class ConfidenceLevel(Enum):
    """信頼度レベル"""
    HIGH = "HIGH"           # 70%以上 - 高信頼度
    MODERATE = "MODERATE"   # 50-70% - 中信頼度
    LOW = "LOW"             # 50%未満 - 低信頼度（警告）


@dataclass
class ConfidenceBreakdown:
    """信頼度スコアの内訳"""
    model_uncertainty: float      # モデル不確実性 (0-1)
    data_quality: float           # データ品質 (0-1)
    market_condition: float       # 市場状況 (0-1)
    prediction_stability: float   # 予測安定性 (0-1)
    external_factors: float       # 外部要因 (0-1)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "model_uncertainty": self.model_uncertainty,
            "data_quality": self.data_quality,
            "market_condition": self.market_condition,
            "prediction_stability": self.prediction_stability,
            "external_factors": self.external_factors
        }


class ConfidenceScoring:
    """
    予測の信頼性を多角的に評価
    
    各要因の重み付け:
    - モデル不確実性: 20%
    - データ品質: 15%
    - マーケット状況: 30%
    - 予測安定性: 20%
    - 外部ファクター: 15%
    """
    
    # 各要因の重み
    WEIGHTS = {
        'model_uncertainty': 0.20,
        'data_quality': 0.15,
        'market_condition': 0.30,
        'prediction_stability': 0.20,
        'external_factors': 0.15
    }
    
    # 信頼度レベル閾値
    HIGH_THRESHOLD = 0.70
    MODERATE_THRESHOLD = 0.50
    
    def __init__(self):
        """初期化"""
        self.historical_predictions = []  # 過去予測履歴（安定性評価用）
        logger.info("[INIT] ConfidenceScoring 初期化完了")
    
    def score(
        self, 
        prediction: Dict[str, Any],
        historical_data: pd.DataFrame,
        model_accuracies: Optional[Dict[str, float]] = None,
        external_events: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        予測に対する総合信頼度スコアを計算
        
        Args:
            prediction: 予測結果辞書（MultiTimeScaleForecastの出力）
            historical_data: 過去の価格データ
            model_accuracies: モデルの過去精度（オプション）
            external_events: 外部イベント情報（オプション）
        
        Returns:
            信頼度評価結果の辞書
        """
        # 各要因のスコアを計算
        model_score = self._evaluate_model_uncertainty(prediction, model_accuracies)
        data_score = self._evaluate_data_quality(historical_data)
        market_score = self._evaluate_market_condition(historical_data)
        stability_score = self._evaluate_prediction_stability(prediction, historical_data)
        external_score = self._evaluate_external_factors(external_events)
        
        # 内訳オブジェクト作成
        breakdown = ConfidenceBreakdown(
            model_uncertainty=model_score,
            data_quality=data_score,
            market_condition=market_score,
            prediction_stability=stability_score,
            external_factors=external_score
        )
        
        # 総合スコア計算（重み付き平均）
        total_score = (
            model_score * self.WEIGHTS['model_uncertainty'] +
            data_score * self.WEIGHTS['data_quality'] +
            market_score * self.WEIGHTS['market_condition'] +
            stability_score * self.WEIGHTS['prediction_stability'] +
            external_score * self.WEIGHTS['external_factors']
        )
        
        # 信頼度レベル判定
        if total_score >= self.HIGH_THRESHOLD:
            level = ConfidenceLevel.HIGH
            recommendation = "HIGH_CONFIDENCE"
        elif total_score >= self.MODERATE_THRESHOLD:
            level = ConfidenceLevel.MODERATE
            recommendation = "MODERATE_CONFIDENCE"
        else:
            level = ConfidenceLevel.LOW
            recommendation = "LOW_CONFIDENCE_WARNING"
        
        # 信頼区間（標準誤差ベース）
        confidence_range = self._calculate_confidence_interval(total_score, breakdown)
        
        result = {
            "confidence": round(total_score, 4),
            "confidence_range": [round(confidence_range[0], 4), round(confidence_range[1], 4)],
            "confidence_level": level.value,
            "breakdown": breakdown.to_dict(),
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
        
        # 履歴に追加（安定性評価用）
        self._record_prediction(prediction, total_score)
        
        logger.info(f"[SCORE] 信頼度スコア: {total_score:.4f} ({level.value})")
        
        return result
    
    def _evaluate_model_uncertainty(
        self, 
        prediction: Dict[str, Any],
        model_accuracies: Optional[Dict[str, float]] = None
    ) -> float:
        """
        モデル不確実性を評価
        
        評価基準:
        - アンサンブル内の合意度
        - 過去の予測誤差
        - 予測確率の確信度
        """
        scores = []
        
        # 予測確率の確信度（0.5からの距離）
        for term in ['short_term', 'medium_term', 'long_term']:
            if term in prediction.get('predictions', {}):
                prob = prediction['predictions'][term].get('probability', 0.5)
                # 確率が0.5から離れているほど確信度が高い
                certainty = abs(prob - 0.5) * 2  # 0-1にスケール
                scores.append(certainty)
        
        # 過去精度が提供されている場合
        if model_accuracies:
            for acc in model_accuracies.values():
                # 精度50%を基準に評価
                scores.append(max(0, (acc - 0.5) * 2))
        
        if not scores:
            return 0.5  # デフォルト値
        
        return float(np.clip(np.mean(scores), 0, 1))
    
    def _evaluate_data_quality(self, data: pd.DataFrame) -> float:
        """
        データ品質を評価
        
        評価基準:
        - 外れ値の有無
        - 欠損値比率
        - データ量
        """
        if data is None or len(data) == 0:
            return 0.0
        
        scores = []
        
        # 欠損値評価
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        missing_score = 1 - min(missing_ratio * 10, 1)  # 10%以上で0
        scores.append(missing_score)
        
        # 外れ値評価（Close列）
        if 'Close' in data.columns:
            close = data['Close'].dropna()
            if len(close) > 0:
                mean = close.mean()
                std = close.std()
                if std > 0:
                    z_scores = np.abs((close - mean) / std)
                    outlier_ratio = (z_scores > 3).sum() / len(z_scores)
                    outlier_score = 1 - min(outlier_ratio * 20, 1)  # 5%以上で0
                    scores.append(outlier_score)
        
        # データ量評価
        data_length_score = min(len(data) / 200, 1)  # 200日以上で満点
        scores.append(data_length_score)
        
        return float(np.clip(np.mean(scores), 0, 1))
    
    def _evaluate_market_condition(self, data: pd.DataFrame) -> float:
        """
        マーケット状況を評価
        
        評価基準:
        - 現在のレジーム（トレンド/レンジ）
        - ボラティリティ環境
        - 流動性
        """
        if data is None or len(data) < 20:
            return 0.5
        
        scores = []
        
        close = data['Close'].tail(60)
        
        # トレンド明確度
        if len(close) >= 20:
            returns = close.pct_change()
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return > 0:
                # シャープレシオ的な指標
                trend_clarity = abs(mean_return) / std_return
                trend_score = min(trend_clarity * 5, 1)  # 0.2以上で満点
                scores.append(trend_score)
        
        # ボラティリティ環境
        if len(close) >= 20:
            volatility = close.pct_change().std()
            # 低ボラティリティほど予測しやすい
            vol_score = 1 - min(volatility * 20, 1)  # 日次5%以上で0
            scores.append(vol_score)
        
        # 流動性（出来高の安定性）
        if 'Volume' in data.columns and len(data) >= 20:
            volume = data['Volume'].tail(60)
            if volume.mean() > 0:
                vol_cv = volume.std() / volume.mean()  # 変動係数
                liquidity_score = 1 - min(vol_cv, 1)
                scores.append(liquidity_score)
        
        return float(np.clip(np.mean(scores) if scores else 0.5, 0, 1))
    
    def _evaluate_prediction_stability(
        self, 
        prediction: Dict[str, Any],
        data: pd.DataFrame
    ) -> float:
        """
        予測安定性を評価
        
        評価基準:
        - 時系列での一貫性
        - 入力データの微小変化への感応度
        """
        # 過去の予測履歴がある場合
        if len(self.historical_predictions) >= 3:
            recent_scores = [p['score'] for p in self.historical_predictions[-5:]]
            # スコアの標準偏差が小さいほど安定
            std_scores = np.std(recent_scores)
            stability_score = 1 - min(std_scores * 5, 1)
            return float(stability_score)
        
        # 履歴がない場合は中程度の評価
        return 0.6
    
    def _evaluate_external_factors(
        self, 
        external_events: Optional[List[Dict]] = None
    ) -> float:
        """
        外部ファクターを評価
        
        評価基準:
        - 経済カレンダーイベントの有無
        - マクロ環境の安定性
        """
        if external_events is None or len(external_events) == 0:
            return 0.7  # イベントなしは比較的安定
        
        # イベント数に応じてスコアを下げる
        high_impact_events = sum(1 for e in external_events if e.get('impact', 'low') == 'high')
        
        if high_impact_events >= 3:
            return 0.3
        elif high_impact_events >= 1:
            return 0.5
        else:
            return 0.7
    
    def _calculate_confidence_interval(
        self, 
        score: float, 
        breakdown: ConfidenceBreakdown
    ) -> Tuple[float, float]:
        """
        信頼区間を計算
        
        各要因のばらつきに基づいて区間を算出
        """
        values = [
            breakdown.model_uncertainty,
            breakdown.data_quality,
            breakdown.market_condition,
            breakdown.prediction_stability,
            breakdown.external_factors
        ]
        
        std = np.std(values)
        margin = std * 0.5  # 標準誤差の50%をマージンとして使用
        
        lower = max(0, score - margin)
        upper = min(1, score + margin)
        
        return (lower, upper)
    
    def _record_prediction(self, prediction: Dict, score: float):
        """予測履歴を記録"""
        self.historical_predictions.append({
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'ticker': prediction.get('ticker', 'UNKNOWN')
        })
        
        # 直近100件のみ保持
        if len(self.historical_predictions) > 100:
            self.historical_predictions = self.historical_predictions[-100:]
    
    def get_recommendation_text(self, result: Dict[str, Any]) -> str:
        """
        信頼度結果から推奨テキストを生成
        
        Args:
            result: score()の出力
        
        Returns:
            人間可読な推奨テキスト
        """
        level = result['confidence_level']
        score = result['confidence']
        breakdown = result['breakdown']
        
        # 最も低い要因を特定
        lowest_factor = min(breakdown.items(), key=lambda x: x[1])
        
        if level == 'HIGH':
            text = f"🟢 高信頼度 ({score:.1%}): 予測を信頼できます。"
        elif level == 'MODERATE':
            text = f"🟡 中信頼度 ({score:.1%}): 注意して使用してください。"
            text += f" ({lowest_factor[0]}が低い: {lowest_factor[1]:.1%})"
        else:
            text = f"🔴 低信頼度 ({score:.1%}): 予測の使用は推奨しません。"
            text += f" ({lowest_factor[0]}が特に低い: {lowest_factor[1]:.1%})"
        
        return text


# ========== テスト用コード ==========
if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== ConfidenceScoring テスト ===\n")
    
    # サンプルデータ取得
    ticker = "AAPL"
    print(f"[TEST] {ticker} のデータを取得中...")
    data = yf.download(ticker, period="6mo", progress=False)
    
    # サンプル予測データ
    sample_prediction = {
        "ticker": ticker,
        "predictions": {
            "short_term": {
                "direction": "UP",
                "probability": 0.72,
                "confidence": 0.44
            },
            "medium_term": {
                "direction": "UP",
                "probability": 0.58,
                "confidence": 0.16
            },
            "long_term": {
                "direction": "UP",
                "probability": 0.55,
                "confidence": 0.10
            }
        }
    }
    
    # スコアリング実行
    scorer = ConfidenceScoring()
    
    # モデル精度情報（オプション）
    model_accuracies = {
        "short_term_xgb": 0.68,
        "medium_term": 0.58,
        "long_term": 0.52
    }
    
    result = scorer.score(
        sample_prediction,
        data,
        model_accuracies=model_accuracies
    )
    
    print("\n[RESULT] 信頼度評価結果:")
    print(f"  総合スコア: {result['confidence']:.4f}")
    print(f"  信頼区間: [{result['confidence_range'][0]:.4f}, {result['confidence_range'][1]:.4f}]")
    print(f"  レベル: {result['confidence_level']}")
    print(f"  推奨: {result['recommendation']}")
    
    print("\n  [内訳]")
    for factor, score in result['breakdown'].items():
        print(f"    - {factor}: {score:.4f}")
    
    print(f"\n  {scorer.get_recommendation_text(result)}")
