"""
tests/test_confidence_score.py - pytest ユニットテスト
対象: confidence_filter.py の calculate_confidence_score()
"""

import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from confidence_filter import calculate_confidence_score


class TestConfidenceScore:
    """calculate_confidence_score() の基本動作テスト"""
    
    def test_fifty_percent_probability(self):
        """pred_proba=0.5 → score=0.0, level='WEAK'"""
        score, level = calculate_confidence_score(0.5)
        assert score == 0.0
        assert level == 'WEAK'
    
    def test_sixty_five_percent_probability(self):
        """pred_proba=0.65 → score=0.3, level='STRONG'"""
        score, level = calculate_confidence_score(0.65)
        assert abs(score - 0.30) < 0.001  # 浮動小数点誤差を許容
        assert level == 'STRONG'
    
    def test_seventy_five_percent_probability(self):
        """pred_proba=0.75 → score=0.5, level='STRONG'"""
        score, level = calculate_confidence_score(0.75)
        assert abs(score - 0.50) < 0.001
        assert level == 'STRONG'
    
    def test_one_hundred_percent_probability(self):
        """pred_proba=1.0 → score=1.0, level='STRONG'"""
        score, level = calculate_confidence_score(1.0)
        assert abs(score - 1.0) < 0.001
        assert level == 'STRONG'
    
    def test_zero_percent_probability(self):
        """pred_proba=0.0 → score=1.0, level='STRONG' (完全下落確信)"""
        score, level = calculate_confidence_score(0.0)
        assert abs(score - 1.0) < 0.001
        assert level == 'STRONG'
    
    def test_fifty_five_percent_probability(self):
        """pred_proba=0.55 → score=0.10, level='MEDIUM' (境界値)"""
        score, level = calculate_confidence_score(0.55)
        assert abs(score - 0.10) < 0.001
        assert level == 'MEDIUM'
    
    def test_eighty_percent_probability(self):
        """pred_proba=0.8 → score=0.6, level='STRONG'"""
        score, level = calculate_confidence_score(0.8)
        assert abs(score - 0.60) < 0.001
        assert level == 'STRONG'


class TestConfidenceScoreWithAccuracy:
    """model_accuracy パラメータのテスト"""
    
    def test_with_60_percent_accuracy(self):
        """
        pred_proba=0.6, model_accuracy=0.6
        base_score = |0.6-0.5|*2 = 0.2
        adjusted = 0.2 * (0.5 + 0.6) = 0.2 * 1.1 = 0.22
        """
        score, level = calculate_confidence_score(0.6, model_accuracy=0.6)
        assert abs(score - 0.22) < 0.001
        assert level == 'MEDIUM'
    
    def test_with_50_percent_accuracy(self):
        """
        pred_proba=0.7, model_accuracy=0.5 (baseline)
        base_score = 0.4
        adjusted = 0.4 * (0.5 + 0.5) = 0.4 * 1.0 = 0.4 (変化なし)
        """
        score, level = calculate_confidence_score(0.7, model_accuracy=0.5)
        assert abs(score - 0.40) < 0.001
        assert level == 'STRONG'
    
    def test_with_70_percent_accuracy(self):
        """
        pred_proba=0.6, model_accuracy=0.7 (高精度)
        base_score = 0.2
        adjusted = 0.2 * (0.5 + 0.7) = 0.2 * 1.2 = 0.24
        """
        score, level = calculate_confidence_score(0.6, model_accuracy=0.7)
        assert abs(score - 0.24) < 0.001
        assert level == 'MEDIUM'


class TestConfidenceScoreWithRegimeFactor:
    """market_regime_factor パラメータのテスト"""
    
    def test_with_regime_factor_0_8(self):
        """
        pred_proba=0.7, market_regime_factor=0.8 (不安定な市場)
        base_score = 0.4
        adjusted = 0.4 * 0.8 = 0.32
        """
        score, level = calculate_confidence_score(0.7, market_regime_factor=0.8)
        assert abs(score - 0.32) < 0.001
        assert level == 'STRONG'
    
    def test_with_regime_factor_1_2(self):
        """
        pred_proba=0.6, market_regime_factor=1.2 (安定した市場)
        base_score = 0.2
        adjusted = 0.2 * 1.2 = 0.24
        """
        score, level = calculate_confidence_score(0.6, market_regime_factor=1.2)
        assert abs(score - 0.24) < 0.001
        assert level == 'MEDIUM'


class TestConfidenceScoreCombined:
    """model_accuracy と market_regime_factor を両方使用"""
    
    def test_combined_parameters(self):
        """
        pred_proba=0.65, model_accuracy=0.6, market_regime_factor=0.9
        base_score = 0.3
        with_accuracy = 0.3 * 1.1 = 0.33
        with_regime = 0.33 * 0.9 = 0.297
        """
        score, level = calculate_confidence_score(
            0.65, 
            model_accuracy=0.6, 
            market_regime_factor=0.9
        )
        assert abs(score - 0.297) < 0.001
        assert level == 'MEDIUM'  # 0.297 < 0.30 なので MEDIUM


class TestConfidenceLevelThresholds:
    """レベル閾値のテスト"""
    
    def test_weak_threshold_upper_bound(self):
        """score=0.09 → WEAK"""
        # pred_proba = 0.545 → score = 0.09
        score, level = calculate_confidence_score(0.545)
        assert score < 0.10
        assert level == 'WEAK'
    
    def test_medium_threshold_lower_bound(self):
        """score=0.10 → MEDIUM"""
        score, level = calculate_confidence_score(0.55)
        assert abs(score - 0.10) < 0.001
        assert level == 'MEDIUM'
    
    def test_medium_threshold_upper_bound(self):
        """score=0.29 → MEDIUM"""
        # pred_proba = 0.645 → score = 0.29
        score, level = calculate_confidence_score(0.645)
        assert 0.10 <= score < 0.30
        assert level == 'MEDIUM'
    
    def test_strong_threshold_lower_bound(self):
        """score=0.30 → STRONG"""
        score, level = calculate_confidence_score(0.65)
        assert abs(score - 0.30) < 0.001
        assert level == 'STRONG'


class TestConfidenceScoreEdgeCases:
    """エッジケースとクリップ動作のテスト"""
    
    def test_score_clipping_to_one(self):
        """
        スコアが1.0を超える場合、1.0にクリップされる
        pred_proba=0.8, model_accuracy=0.8, market_regime_factor=1.3
        base = 0.6, with_acc = 0.78, with_regime = 1.014 → clip to 1.0
        """
        score, level = calculate_confidence_score(
            0.8,
            model_accuracy=0.8,
            market_regime_factor=1.3
        )
        assert score <= 1.0
        assert level == 'STRONG'
    
    def test_symmetric_probability(self):
        """0.3と0.7は同じスコアを出力 (対称性)"""
        score_low, level_low = calculate_confidence_score(0.3)
        score_high, level_high = calculate_confidence_score(0.7)
        assert abs(score_low - score_high) < 0.001
        assert level_low == level_high


# pytest 実行時のエントリーポイント
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
