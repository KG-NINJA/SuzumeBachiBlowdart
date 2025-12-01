"""
config/hyperparameters.py - システム全体の設定とハイパーパラメータ
すべてのハードコードされた定数を集約し、調整を容易にする
"""

from typing import Dict, Any, Tuple


class XGBoostConfig:
    """
    XGBoost モデルのハイパーパラメータ
    
    シンプルモデル: 保守的な設定で過学習を防ぐ
    アグレッシブモデル: より複雑なパターンを学習（アンサンブル用）
    """
    
    # === シンプルモデル用パラメータ ===
    SIMPLE_N_ESTIMATORS = 120  # ツリー数: 適度な複雑さを保ちつつ過学習を防ぐ
    SIMPLE_MAX_DEPTH = 4  # 木の深さ: 浅めで過学習を抑制
    SIMPLE_LEARNING_RATE = 0.05  # 学習率: 低めでゆっくり学習
    SIMPLE_SUBSAMPLE = 0.8  # サンプル比率: 80%のデータでツリー構築
    SIMPLE_COLSAMPLE_BYTREE = 0.8  # 特徴比率: 80%の特徴をランダム選択
    
    # === アグレッシブモデル用パラメータ ===
    AGGRESSIVE_N_ESTIMATORS = 200  # ツリー数: より多く
    AGGRESSIVE_BASE_MAX_DEPTH = 4  # ベース深さ
    AGGRESSIVE_MAX_DEPTH_VARIATION = 2  # 深さの変動 (4 or 5)
    AGGRESSIVE_BASE_LEARNING_RATE = 0.08  # ベース学習率
    AGGRESSIVE_LEARNING_RATE_MULTIPLIER_START = 0.9  # 倍率開始値
    AGGRESSIVE_LEARNING_RATE_MULTIPLIER_STEP = 0.02  # 倍率増加ステップ
    AGGRESSIVE_SUBSAMPLE_START = 0.7  # サンプル比率開始値
    AGGRESSIVE_SUBSAMPLE_STEP = 0.04  # サンプル比率増加ステップ
    AGGRESSIVE_COLSAMPLE_START = 0.75  # 特徴比率開始値
    AGGRESSIVE_COLSAMPLE_STEP = 0.03  # 特徴比率増加ステップ
    AGGRESSIVE_GAMMA_START = 0.5  # 正則化パラメータ開始値
    AGGRESSIVE_GAMMA_STEP = 0.1  # 正則化パラメータ増加ステップ
    AGGRESSIVE_N_FOLDS = 5  # アンサンブルのフォールド数
    
    # === 共通パラメータ ===
    RANDOM_STATE = 42  # 再現性のためのシード値
    EVAL_METRIC = 'logloss'  # 評価指標
    VERBOSITY = 0  # ログ出力レベル (0=サイレント)
    
    @classmethod
    def get_simple_params(cls) -> Dict[str, Any]:
        """シンプルモデル用パラメータ辞書を取得"""
        return {
            'n_estimators': cls.SIMPLE_N_ESTIMATORS,
            'max_depth': cls.SIMPLE_MAX_DEPTH,
            'learning_rate': cls.SIMPLE_LEARNING_RATE,
            'subsample': cls.SIMPLE_SUBSAMPLE,
            'colsample_bytree': cls.SIMPLE_COLSAMPLE_BYTREE,
            'random_state': cls.RANDOM_STATE,
            'eval_metric': cls.EVAL_METRIC,
            'verbosity': cls.VERBOSITY
        }
    
    @classmethod
    def get_aggressive_params(cls, fold: int) -> Dict[str, Any]:
        """
        アグレッシブモデル用パラメータ辞書を取得
        
        Args:
            fold: フォールド番号 (0-4)
        
        Returns:
            ハイパーパラメータ辞書
        """
        return {
            'n_estimators': cls.AGGRESSIVE_N_ESTIMATORS,
            'max_depth': cls.AGGRESSIVE_BASE_MAX_DEPTH + fold % cls.AGGRESSIVE_MAX_DEPTH_VARIATION,
            'learning_rate': cls.AGGRESSIVE_BASE_LEARNING_RATE * (
                cls.AGGRESSIVE_LEARNING_RATE_MULTIPLIER_START + fold * cls.AGGRESSIVE_LEARNING_RATE_MULTIPLIER_STEP
            ),
            'subsample': cls.AGGRESSIVE_SUBSAMPLE_START + fold * cls.AGGRESSIVE_SUBSAMPLE_STEP,
            'colsample_bytree': cls.AGGRESSIVE_COLSAMPLE_START + fold * cls.AGGRESSIVE_COLSAMPLE_STEP,
            'gamma': cls.AGGRESSIVE_GAMMA_START + fold * cls.AGGRESSIVE_GAMMA_STEP,
            'random_state': cls.RANDOM_STATE + fold,
            'eval_metric': cls.EVAL_METRIC,
            'verbosity': cls.VERBOSITY
        }


class MarketRegimeConfig:
    """
    市場レジーム検知の設定
    
    3つのレジーム:
    - TRENDING: 明確なトレンド（上昇または下降）
    - CHOPPY: 高ボラティリティで方向性なし
    - MEAN_REVERSION: 平均回帰的な動き
    """
    
    LOOKBACK_PERIOD = 20  # レジーム判定に使用する直近日数
    
    # === レジーム判定閾値 ===
    TREND_STRENGTH_THRESHOLD = 0.65  # これ以上でトレンド判定
    VOLATILITY_THRESHOLD_LOW = 0.55  # これ以下で低ボラティリティ
    VOLATILITY_THRESHOLD_HIGH = 0.75  # これ以上で高ボラティリティ(CHOPPY)
    
    # === レジーム別の重み付け ===
    # (simple_weight, aggressive_weight)
    TRENDING_WEIGHTS = (0.3, 0.7)  # トレンド市場ではアグレッシブ重視
    CHOPPY_WEIGHTS = (0.7, 0.3)  # チョッピー市場では保守的に
    MEAN_REVERSION_WEIGHTS = (0.5, 0.5)  # 平均回帰市場ではバランス


class FeatureSelectionConfig:
    """特徴選択の設定"""
    
    MAX_FEATURES = 20  # モデルに使用する最大特徴数
    MIN_FEATURES = 5  # 最低限必要な特徴数（これ以下で警告）
    
    # === リーク特徴（学習から除外） ===
    # 理由: 翌日予測において未来情報のリークとなる、または非定常性の原因となる
    LEAK_FEATURES = {
        'CloseOpenRatio',  # 当日のClose/Open比率
        'DailyReturn',     # 当日の収益率
        'HighLowRatio',    # 当日のHigh/Low比率
        'Close',           # 絶対価格（非定常性の要因）
        'Volume',          # 出来高の絶対値（非定常性の要因）
    }
    
    # === 保持する特徴（優先的に使用） ===
    KEEP_FEATURES = {
        'ATR', 'OBV', 'MACD', 'RSI7', 'RSI14',
        'Momentum', 'Momentum_5', 'Momentum_10',
        'Volume_Ratio',
        'EMA12', 'EMA26', 'MA5', 'MA10', 'MA20', 'MA50',
        'Plus_DI', 'Minus_DI', 'ADX',
        'VROC', 'PVT',
        'Low', 'High', 'Open'
    }


class ConfidenceConfig:
    """
    信頼度スコアの設定
    
    信頼度レベル:
    - STRONG: 高確信（実行推奨）
    - MEDIUM: 中程度（監視対象）
    - WEAK: 低確信（スキップ推奨）
    """
    
    STRONG_THRESHOLD = 0.30  # これ以上でSTRONG判定（約65%確率に相当）
    MEDIUM_THRESHOLD = 0.10  # これ以上でMEDIUM判定（約55%確率に相当）
    
    # === 信頼度フィルタリング ===
    MIN_CONFIDENCE_TO_EXECUTE = 0.15  # 実行に必要な最低信頼度（15%）


class TrainingConfig:
    """モデル訓練の設定"""
    
    TRAIN_TEST_SPLIT_RATIO = 0.8  # 訓練データの割合（80% train, 20% test）
    MIN_TRAINING_SAMPLES = 40  # 訓練に必要な最低サンプル数
    MIN_TEST_SAMPLES = 10  # テストに必要な最低サンプル数
    
    # === オンライン学習 ===
    ENABLE_ONLINE_LEARNING = True  # 既存モデルを読み込んで改善するか
    ENABLE_FEATURE_REDUCTION = True  # 特徴削減を有効にするか


class PredictionConfig:
    """予測の設定"""
    
    # === 予測価格計算 ===
    MAX_PRICE_CHANGE_PCT = 0.04  # 最大変動率（±2%まで）
    # 計算式: price_change = confidence_delta * MAX_PRICE_CHANGE_PCT
    
    # === 信頼度計算 ===
    # final_confidence は calculate_confidence_score() で統一計算


class PathConfig:
    """ファイルパスの設定"""
    
    MODELS_ROOT = "models"  # モデル保存ディレクトリ
    REGIME_LOG = "regime_detection"  # レジーム検出ログディレクトリ
    PREDICTIONS_DIR = "daily_predictions"  # 予測結果ディレクトリ


class DevelopmentConfig:
    """開発環境用の設定（デバッグ用）"""
    
    # XGBoost（高速化のため小さい値）
    SIMPLE_N_ESTIMATORS = 50
    AGGRESSIVE_N_ESTIMATORS = 100
    AGGRESSIVE_N_FOLDS = 3
    
    # マーケットレジーム（短期間）
    LOOKBACK_PERIOD = 10
    
    # 訓練（少ないサンプルで実験）
    MIN_TRAINING_SAMPLES = 20
    MIN_TEST_SAMPLES = 5


class ProductionConfig:
    """本番環境用の設定"""
    
    # 本番では XGBoostConfig のデフォルト値を使用
    pass


# === グローバル設定の選択 ===
ACTIVE_CONFIG = "production"  # "development" or "production"


def get_config(env: str = ACTIVE_CONFIG) -> Dict[str, Any]:
    """
    環境に応じた設定を取得
    
    Args:
        env: 環境名 ("development" or "production")
    
    Returns:
        設定辞書
    """
    if env == "development":
        return {
            "xgboost": DevelopmentConfig,
            "regime": MarketRegimeConfig,
            "features": FeatureSelectionConfig,
            "confidence": ConfidenceConfig,
            "training": TrainingConfig,
            "prediction": PredictionConfig,
            "paths": PathConfig
        }
    else:  # production
        return {
            "xgboost": XGBoostConfig,
            "regime": MarketRegimeConfig,
            "features": FeatureSelectionConfig,
            "confidence": ConfidenceConfig,
            "training": TrainingConfig,
            "prediction": PredictionConfig,
            "paths": PathConfig
        }


if __name__ == "__main__":
    # 設定のテスト出力
    print("=== Production Config ===")
    print("Simple XGBoost params:", XGBoostConfig.get_simple_params())
    print("Aggressive XGBoost params (fold 0):", XGBoostConfig.get_aggressive_params(0))
    print(f"\nMarket Regime - Lookback: {MarketRegimeConfig.LOOKBACK_PERIOD}")
    print(f"Feature Selection - Max: {FeatureSelectionConfig.MAX_FEATURES}")
    print(f"Confidence - Strong Threshold: {ConfidenceConfig.STRONG_THRESHOLD}")
