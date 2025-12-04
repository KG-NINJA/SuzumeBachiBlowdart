"""
scenario.py - シナリオベース分析モジュール

5つの市場シナリオを同時シミュレーション:
1. ベースケース (50%): 現状継続
2. ブル・シナリオ (25%): トレンド強化
3. ベア・シナリオ (15%): 反転トレンド
4. ボラティリティ・ショック (7%): 急変動
5. マクロショック (3%): 経済的混乱

Author: SuzumeBachiBlowdart Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("prediction_engine.scenario")


class ScenarioType(Enum):
    """シナリオタイプ"""
    BASE_CASE = "base_case"
    BULL = "bull"
    BEAR = "bear"
    VOLATILITY_SHOCK = "volatility_shock"
    MACRO_SHOCK = "macro_shock"


@dataclass
class ScenarioConfig:
    """シナリオ設定"""
    name: str
    probability: float
    description: str
    return_multiplier: float  # ベースリターンへの乗数
    volatility_multiplier: float  # ボラティリティへの乗数


class ScenarioAnalysis:
    """
    複数の市場シナリオ下での予測を同時生成
    
    各シナリオの確率と特性:
    - ベースケース (50%): 現在の市場条件が継続
    - ブル (25%): トレンド強化、ボラティリティ低下
    - ベア (15%): 反転トレンド、ボラティリティ上昇
    - ボラティリティ・ショック (7%): 予期しない急変動
    - マクロショック (3%): 金利急上昇、為替急変など
    """
    
    # シナリオ定義
    SCENARIOS = {
        ScenarioType.BASE_CASE: ScenarioConfig(
            name="ベースケース",
            probability=0.50,
            description="現在の市場条件が継続",
            return_multiplier=1.0,
            volatility_multiplier=1.0
        ),
        ScenarioType.BULL: ScenarioConfig(
            name="ブル・シナリオ",
            probability=0.25,
            description="トレンド強化、好材料出現",
            return_multiplier=1.8,
            volatility_multiplier=0.7
        ),
        ScenarioType.BEAR: ScenarioConfig(
            name="ベア・シナリオ",
            probability=0.15,
            description="反転トレンド、悪材料出現",
            return_multiplier=-0.8,
            volatility_multiplier=1.5
        ),
        ScenarioType.VOLATILITY_SHOCK: ScenarioConfig(
            name="ボラティリティ・ショック",
            probability=0.07,
            description="予期しない急変動",
            return_multiplier=0.0,  # 方向不定
            volatility_multiplier=3.0
        ),
        ScenarioType.MACRO_SHOCK: ScenarioConfig(
            name="マクロショック",
            probability=0.03,
            description="金利急上昇、経済統計悪化",
            return_multiplier=-2.0,
            volatility_multiplier=2.5
        )
    }
    
    # トリガーイベント例
    TRIGGER_EVENTS = {
        ScenarioType.BULL: [
            "決算サプライズ（上方修正）",
            "新製品発表",
            "アナリスト格上げ",
            "大型契約獲得",
            "自社株買い発表"
        ],
        ScenarioType.BEAR: [
            "決算ミス",
            "サプライチェーン混乱",
            "競合の台頭",
            "規制強化",
            "CFO辞任"
        ],
        ScenarioType.VOLATILITY_SHOCK: [
            "オプション満期集中",
            "ミームストック化",
            "ショートスクイーズ",
            "大口売却報道"
        ],
        ScenarioType.MACRO_SHOCK: [
            "FRB利上げ",
            "インフレ悪化",
            "地政学リスク",
            "金融危機懸念",
            "景気後退入り"
        ]
    }
    
    def __init__(self):
        """初期化"""
        logger.info("[INIT] ScenarioAnalysis 初期化完了")
    
    def generate_scenarios(
        self, 
        ticker: str,
        prediction: Dict[str, Any],
        historical_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        5つのシナリオを同時生成
        
        Args:
            ticker: 銘柄コード
            prediction: MultiTimeScaleForecastの予測結果
            historical_data: 過去の価格データ
        
        Returns:
            シナリオ分析結果
        """
        logger.info(f"[SCENARIO] {ticker} のシナリオ分析開始")
        
        # 基準値を取得
        current_price = prediction.get('current_price', 
                                       float(historical_data['Close'].iloc[-1]))
        
        # 過去データからボラティリティを計算
        returns = historical_data['Close'].pct_change().dropna()
        base_volatility = float(returns.std())
        base_return = float(returns.mean())
        
        # 予測の方向と確率を取得
        short_term = prediction.get('predictions', {}).get('short_term', {})
        base_prob = short_term.get('probability', 0.5)
        base_direction = 1 if base_prob > 0.5 else -1
        base_confidence = abs(base_prob - 0.5) * 2
        
        result = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "scenarios": {}
        }
        
        # 各シナリオを生成
        for scenario_type, config in self.SCENARIOS.items():
            scenario_result = self._generate_single_scenario(
                scenario_type=scenario_type,
                config=config,
                current_price=current_price,
                base_return=base_return,
                base_volatility=base_volatility,
                base_direction=base_direction,
                base_confidence=base_confidence
            )
            result['scenarios'][scenario_type.value] = scenario_result
        
        # 加重平均予測を計算
        weighted_forecast = self._calculate_weighted_forecast(result['scenarios'])
        result['weighted_forecast'] = weighted_forecast
        
        # リスク/リワード分析
        result['risk_reward'] = self._analyze_risk_reward(result['scenarios'], current_price)
        
        # 推奨アクション
        result['recommendation'] = self._generate_recommendation(result)
        
        logger.info(f"[SCENARIO] {ticker} の分析完了")
        
        return result
    
    def _generate_single_scenario(
        self,
        scenario_type: ScenarioType,
        config: ScenarioConfig,
        current_price: float,
        base_return: float,
        base_volatility: float,
        base_direction: int,
        base_confidence: float
    ) -> Dict[str, Any]:
        """単一シナリオを生成"""
        
        # 期間別の予測を計算
        if scenario_type == ScenarioType.BASE_CASE:
            # ベースケース：通常の予測を使用
            forecast_3d = current_price * (1 + base_return * 3 * base_direction * base_confidence)
            forecast_1m = current_price * (1 + base_return * 20 * base_direction * base_confidence)
            volatility = base_volatility
            trend = "UP" if base_direction > 0 else "DOWN"
            
        elif scenario_type == ScenarioType.BULL:
            # ブル：上昇方向に強化
            adj_return = abs(base_return) * config.return_multiplier
            forecast_3d = current_price * (1 + adj_return * 3)
            forecast_1m = current_price * (1 + adj_return * 20)
            volatility = base_volatility * config.volatility_multiplier
            trend = "UP"
            
        elif scenario_type == ScenarioType.BEAR:
            # ベア：下落方向
            adj_return = abs(base_return) * abs(config.return_multiplier)
            forecast_3d = current_price * (1 - adj_return * 3)
            forecast_1m = current_price * (1 - adj_return * 20)
            volatility = base_volatility * config.volatility_multiplier
            trend = "DOWN"
            
        elif scenario_type == ScenarioType.VOLATILITY_SHOCK:
            # ボラティリティショック：方向不定、大きな振れ幅
            max_swing = current_price * base_volatility * config.volatility_multiplier * 5
            forecast_3d = current_price  # 中立
            forecast_1m = current_price
            volatility = base_volatility * config.volatility_multiplier
            trend = "VOLATILE"
            
        else:  # MACRO_SHOCK
            # マクロショック：急落
            adj_return = abs(base_return) * abs(config.return_multiplier)
            forecast_3d = current_price * (1 - adj_return * 3)
            forecast_1m = current_price * (1 - adj_return * 15)  # 回復を見込む
            volatility = base_volatility * config.volatility_multiplier
            trend = "DOWN"
        
        # トリガーイベント
        triggers = self.TRIGGER_EVENTS.get(scenario_type, [])
        trigger = np.random.choice(triggers) if triggers else None
        
        scenario_result = {
            "name": config.name,
            "probability": config.probability,
            "description": config.description,
            "forecast_3d": round(forecast_3d, 2),
            "forecast_1m": round(forecast_1m, 2),
            "expected_return_3d": round((forecast_3d / current_price - 1) * 100, 2),
            "expected_return_1m": round((forecast_1m / current_price - 1) * 100, 2),
            "volatility": round(volatility * 100, 2),  # パーセント表示
            "trend": trend
        }
        
        # シナリオ固有の情報
        if scenario_type == ScenarioType.VOLATILITY_SHOCK:
            max_swing_pct = base_volatility * config.volatility_multiplier * 5 * 100
            scenario_result['max_swing'] = f"±{max_swing_pct:.1f}%"
            scenario_result['duration_days'] = 3
            
        elif scenario_type == ScenarioType.MACRO_SHOCK:
            scenario_result['impact'] = "SEVERE"
            scenario_result['recovery_time'] = "2-4 weeks"
        
        if trigger:
            scenario_result['trigger'] = trigger
        
        return scenario_result
    
    def _calculate_weighted_forecast(
        self, 
        scenarios: Dict[str, Dict]
    ) -> Dict[str, float]:
        """確率加重平均予測を計算"""
        weighted_3d = 0.0
        weighted_1m = 0.0
        
        for scenario_data in scenarios.values():
            prob = scenario_data['probability']
            weighted_3d += prob * scenario_data['forecast_3d']
            weighted_1m += prob * scenario_data['forecast_1m']
        
        return {
            "forecast_3d": round(weighted_3d, 2),
            "forecast_1m": round(weighted_1m, 2)
        }
    
    def _analyze_risk_reward(
        self, 
        scenarios: Dict[str, Dict],
        current_price: float
    ) -> Dict[str, Any]:
        """リスク/リワード分析"""
        # 最良・最悪ケースを抽出
        forecasts_1m = [(s['name'], s['forecast_1m'], s['probability']) 
                        for s in scenarios.values()]
        
        best_case = max(forecasts_1m, key=lambda x: x[1])
        worst_case = min(forecasts_1m, key=lambda x: x[1])
        
        upside = (best_case[1] / current_price - 1) * 100
        downside = (worst_case[1] / current_price - 1) * 100
        
        # リスク/リワード比率
        if abs(downside) > 0:
            ratio = upside / abs(downside)
        else:
            ratio = float('inf')
        
        return {
            "best_case": {
                "scenario": best_case[0],
                "price": best_case[1],
                "return_pct": round(upside, 2),
                "probability": best_case[2]
            },
            "worst_case": {
                "scenario": worst_case[0],
                "price": worst_case[1],
                "return_pct": round(downside, 2),
                "probability": worst_case[2]
            },
            "risk_reward_ratio": round(ratio, 2)
        }
    
    def _generate_recommendation(self, result: Dict) -> Dict[str, Any]:
        """推奨アクションを生成"""
        weighted = result['weighted_forecast']
        current = result['current_price']
        risk_reward = result['risk_reward']
        
        expected_return = (weighted['forecast_1m'] / current - 1) * 100
        
        if expected_return > 5 and risk_reward['risk_reward_ratio'] > 1.5:
            action = "STRONG_BUY"
            reasoning = "期待リターンが高く、リスク/リワード比も良好"
        elif expected_return > 2 and risk_reward['risk_reward_ratio'] > 1.0:
            action = "BUY"
            reasoning = "緩やかな上昇が期待される"
        elif expected_return < -5 and risk_reward['risk_reward_ratio'] < 0.5:
            action = "SELL"
            reasoning = "下落リスクが高い"
        elif expected_return < -2:
            action = "REDUCE"
            reasoning = "ポジションの縮小を推奨"
        else:
            action = "HOLD"
            reasoning = "現状維持が適切"
        
        return {
            "action": action,
            "reasoning": reasoning,
            "expected_return_1m": round(expected_return, 2),
            "risk_reward_ratio": risk_reward['risk_reward_ratio']
        }
    
    def adjust_probabilities(
        self, 
        market_regime: str,
        volatility_level: str = "NORMAL"
    ) -> Dict[str, float]:
        """
        市場環境に応じてシナリオ確率を調整
        
        Args:
            market_regime: 'TRENDING' | 'CHOPPY' | 'MEAN_REVERSION'
            volatility_level: 'LOW' | 'NORMAL' | 'HIGH'
        
        Returns:
            調整後の確率辞書
        """
        adjusted = {}
        
        for scenario_type, config in self.SCENARIOS.items():
            prob = config.probability
            
            # 市場レジームによる調整
            if market_regime == 'TRENDING':
                if scenario_type in [ScenarioType.BULL, ScenarioType.BEAR]:
                    prob *= 1.2  # トレンド継続の確率UP
                elif scenario_type == ScenarioType.BASE_CASE:
                    prob *= 0.9
                    
            elif market_regime == 'CHOPPY':
                if scenario_type == ScenarioType.VOLATILITY_SHOCK:
                    prob *= 1.5
                elif scenario_type == ScenarioType.BASE_CASE:
                    prob *= 1.1
            
            # ボラティリティ環境による調整
            if volatility_level == 'HIGH':
                if scenario_type in [ScenarioType.VOLATILITY_SHOCK, ScenarioType.MACRO_SHOCK]:
                    prob *= 1.3
            elif volatility_level == 'LOW':
                if scenario_type == ScenarioType.BASE_CASE:
                    prob *= 1.2
            
            adjusted[scenario_type.value] = prob
        
        # 正規化（合計を1にする）
        total = sum(adjusted.values())
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}
        
        return adjusted


# ========== テスト用コード ==========
if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== ScenarioAnalysis テスト ===\n")
    
    # サンプルデータ取得
    ticker = "NVDA"
    print(f"[TEST] {ticker} のデータを取得中...")
    data = yf.download(ticker, period="6mo", progress=False)
    
    # サンプル予測データ
    current_price = float(data['Close'].iloc[-1])
    sample_prediction = {
        "ticker": ticker,
        "current_price": current_price,
        "predictions": {
            "short_term": {
                "direction": "UP",
                "probability": 0.68,
                "confidence": 0.36
            }
        }
    }
    
    # シナリオ分析実行
    analyzer = ScenarioAnalysis()
    result = analyzer.generate_scenarios(ticker, sample_prediction, data)
    
    print(f"\n[RESULT] シナリオ分析結果:")
    print(f"  銘柄: {result['ticker']}")
    print(f"  現在価格: ${result['current_price']:.2f}")
    
    print("\n  [各シナリオ]")
    for scenario_name, scenario_data in result['scenarios'].items():
        print(f"\n  【{scenario_data['name']}】確率: {scenario_data['probability']:.0%}")
        print(f"    3日後予測: ${scenario_data['forecast_3d']:.2f} ({scenario_data['expected_return_3d']:+.2f}%)")
        print(f"    1ヶ月後予測: ${scenario_data['forecast_1m']:.2f} ({scenario_data['expected_return_1m']:+.2f}%)")
        print(f"    トレンド: {scenario_data['trend']}")
        if 'trigger' in scenario_data:
            print(f"    トリガー: {scenario_data['trigger']}")
    
    print(f"\n  [加重平均予測]")
    wf = result['weighted_forecast']
    print(f"    3日後: ${wf['forecast_3d']:.2f}")
    print(f"    1ヶ月後: ${wf['forecast_1m']:.2f}")
    
    print(f"\n  [リスク/リワード分析]")
    rr = result['risk_reward']
    print(f"    最良ケース: {rr['best_case']['scenario']} (+{rr['best_case']['return_pct']:.2f}%)")
    print(f"    最悪ケース: {rr['worst_case']['scenario']} ({rr['worst_case']['return_pct']:.2f}%)")
    print(f"    リスク/リワード比: {rr['risk_reward_ratio']:.2f}")
    
    print(f"\n  [推奨]")
    rec = result['recommendation']
    print(f"    アクション: {rec['action']}")
    print(f"    理由: {rec['reasoning']}")
