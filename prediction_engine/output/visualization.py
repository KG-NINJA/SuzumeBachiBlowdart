"""
visualization.py - 予測結果可視化モジュール

Matplotlib/Plotlyを使用した予測チャート生成:
- 価格予測チャート
- 信頼度ヒートマップ
- シナリオ比較グラフ
- リスク/リワードプロット

Author: SuzumeBachiBlowdart Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlibがインストールされていません")

logger = logging.getLogger("prediction_engine.visualization")


class Visualization:
    """
    予測結果の可視化
    
    対応チャート:
    - 価格予測チャート（時系列 + 予測）
    - 信頼度ヒートマップ
    - シナリオ比較チャート
    - リスク/リワードプロット
    """
    
    # カラーパレット（ダークテーマ）
    COLORS = {
        'background': '#1a1a2e',
        'text': '#eaeaea',
        'grid': '#2d2d44',
        'up': '#00ff88',
        'down': '#ff4444',
        'neutral': '#888888',
        'confidence_high': '#00ff88',
        'confidence_medium': '#ffcc00',
        'confidence_low': '#ff4444',
        'scenarios': ['#4ecdc4', '#ff6b6b', '#95e1d3', '#f38181', '#aa96da']
    }
    
    def __init__(self, output_dir: str = "prediction_charts"):
        """
        初期化
        
        Args:
            output_dir: チャート出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.style.use('dark_background')
        
        logger.info(f"[INIT] Visualization 初期化完了 (output: {output_dir})")
    
    def create_prediction_chart(
        self,
        historical_data: pd.DataFrame,
        prediction: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        価格予測チャートを作成
        
        過去データ + 各時間軸の予測を表示
        
        Args:
            historical_data: 過去の価格データ
            prediction: 予測結果
            filename: 保存ファイル名
        
        Returns:
            ファイルパス or None
        """
        if not HAS_MATPLOTLIB:
            logger.warning("[CHART] matplotlibが利用できません")
            return None
        
        ticker = prediction.get('ticker', 'UNKNOWN')
        current_price = prediction.get('current_price', 0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(self.COLORS['background'])
        ax.set_facecolor(self.COLORS['background'])
        
        # 過去データをプロット
        dates = historical_data.index if isinstance(historical_data.index, pd.DatetimeIndex) else pd.date_range(end=datetime.now(), periods=len(historical_data))
        prices = historical_data['Close'].values
        
        ax.plot(dates[-60:], prices[-60:], color='white', linewidth=2, label='過去価格')
        
        # 予測をプロット
        last_date = dates[-1]
        predictions = prediction.get('predictions', {})
        
        forecast_points = []
        
        # 短期予測 (3日後)
        if 'short_term' in predictions:
            short = predictions['short_term']
            forecast_3d = short.get('forecast_3d', current_price)
            confidence = short.get('confidence', 0)
            
            date_3d = last_date + timedelta(days=3)
            forecast_points.append((date_3d, forecast_3d, confidence, '3日後'))
            
            color = self.COLORS['up'] if short.get('direction') == 'UP' else self.COLORS['down']
            ax.scatter([date_3d], [forecast_3d], color=color, s=100, zorder=5)
            ax.annotate(f"${forecast_3d:.2f}", (date_3d, forecast_3d), 
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', color=color, fontsize=10)
        
        # 中期予測 (4週後)
        if 'medium_term' in predictions:
            medium = predictions['medium_term']
            forecast_4w = medium.get('forecast_4w', current_price)
            confidence = medium.get('confidence', 0)
            
            date_4w = last_date + timedelta(weeks=4)
            forecast_points.append((date_4w, forecast_4w, confidence, '4週後'))
            
            color = self.COLORS['up'] if medium.get('direction') == 'UP' else self.COLORS['down']
            ax.scatter([date_4w], [forecast_4w], color=color, s=100, zorder=5)
            ax.annotate(f"${forecast_4w:.2f}", (date_4w, forecast_4w),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', color=color, fontsize=10)
        
        # 長期予測 (3ヶ月後)
        if 'long_term' in predictions:
            long_pred = predictions['long_term']
            forecast_3m = long_pred.get('forecast_3m', current_price)
            
            date_3m = last_date + timedelta(days=90)
            forecast_points.append((date_3m, forecast_3m, 0, '3ヶ月後'))
            
            color = self.COLORS['up'] if long_pred.get('direction') == 'UP' else self.COLORS['down']
            ax.scatter([date_3m], [forecast_3m], color=color, s=100, zorder=5)
            ax.annotate(f"${forecast_3m:.2f}", (date_3m, forecast_3m),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', color=color, fontsize=10)
        
        # 予測線（点線）
        if forecast_points:
            all_dates = [last_date] + [p[0] for p in forecast_points]
            all_prices = [current_price] + [p[1] for p in forecast_points]
            ax.plot(all_dates, all_prices, '--', color='#888888', linewidth=1, alpha=0.7)
        
        # 現在価格ライン
        ax.axhline(y=current_price, color='yellow', linestyle='--', alpha=0.5, label=f'現在価格: ${current_price:.2f}')
        
        # 装飾
        ax.set_title(f'{ticker} 価格予測チャート', fontsize=16, color=self.COLORS['text'], pad=20)
        ax.set_xlabel('日付', color=self.COLORS['text'])
        ax.set_ylabel('価格 ($)', color=self.COLORS['text'])
        ax.legend(loc='upper left', facecolor=self.COLORS['background'], edgecolor=self.COLORS['grid'])
        ax.grid(True, color=self.COLORS['grid'], alpha=0.3)
        ax.tick_params(colors=self.COLORS['text'])
        
        # 日付フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存
        if filename is None:
            filename = f"{ticker}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, facecolor=self.COLORS['background'])
        plt.close()
        
        logger.info(f"[CHART] 保存完了: {filepath}")
        return str(filepath)
    
    def create_confidence_heatmap(
        self,
        predictions_list: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        複数銘柄の信頼度ヒートマップを作成
        
        Args:
            predictions_list: 予測結果のリスト
            filename: 保存ファイル名
        
        Returns:
            ファイルパス or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not predictions_list:
            return None
        
        # データ準備
        tickers = []
        terms = ['short_term', 'medium_term', 'long_term']
        term_labels = ['短期', '中期', '長期']
        confidence_matrix = []
        
        for pred in predictions_list:
            ticker = pred.get('ticker', 'N/A')
            tickers.append(ticker)
            
            row = []
            for term in terms:
                conf = pred.get('predictions', {}).get(term, {}).get('confidence', 0)
                row.append(conf)
            confidence_matrix.append(row)
        
        confidence_matrix = np.array(confidence_matrix)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(tickers) * 0.5)))
        fig.patch.set_facecolor(self.COLORS['background'])
        ax.set_facecolor(self.COLORS['background'])
        
        # ヒートマップ
        im = ax.imshow(confidence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # ラベル
        ax.set_xticks(np.arange(len(terms)))
        ax.set_yticks(np.arange(len(tickers)))
        ax.set_xticklabels(term_labels, color=self.COLORS['text'])
        ax.set_yticklabels(tickers, color=self.COLORS['text'])
        
        # 値表示
        for i in range(len(tickers)):
            for j in range(len(terms)):
                value = confidence_matrix[i, j]
                text_color = 'black' if value > 0.5 else 'white'
                ax.text(j, i, f'{value:.0%}', ha='center', va='center', 
                       color=text_color, fontsize=10, fontweight='bold')
        
        # カラーバー
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('信頼度', rotation=-90, va='bottom', color=self.COLORS['text'])
        cbar.ax.tick_params(colors=self.COLORS['text'])
        
        ax.set_title('銘柄別 信頼度ヒートマップ', fontsize=14, color=self.COLORS['text'], pad=15)
        
        plt.tight_layout()
        
        # 保存
        if filename is None:
            filename = f"confidence_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, facecolor=self.COLORS['background'])
        plt.close()
        
        logger.info(f"[HEATMAP] 保存完了: {filepath}")
        return str(filepath)
    
    def create_scenario_comparison(
        self,
        scenario_result: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        シナリオ比較チャートを作成
        
        Args:
            scenario_result: ScenarioAnalysisの結果
            filename: 保存ファイル名
        
        Returns:
            ファイルパス or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        ticker = scenario_result.get('ticker', 'UNKNOWN')
        current_price = scenario_result.get('current_price', 100)
        scenarios = scenario_result.get('scenarios', {})
        
        if not scenarios:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(self.COLORS['background'])
        
        for ax in [ax1, ax2]:
            ax.set_facecolor(self.COLORS['background'])
        
        # 左図：シナリオ別予測価格
        scenario_names = []
        forecast_3d = []
        forecast_1m = []
        probs = []
        
        for name, data in scenarios.items():
            scenario_names.append(data.get('name', name)[:8])
            forecast_3d.append(data.get('forecast_3d', current_price))
            forecast_1m.append(data.get('forecast_1m', current_price))
            probs.append(data.get('probability', 0))
        
        x = np.arange(len(scenario_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, forecast_3d, width, label='3日後', color=self.COLORS['scenarios'][0])
        bars2 = ax1.bar(x + width/2, forecast_1m, width, label='1ヶ月後', color=self.COLORS['scenarios'][1])
        
        ax1.axhline(y=current_price, color='yellow', linestyle='--', alpha=0.7, label='現在価格')
        ax1.set_ylabel('価格 ($)', color=self.COLORS['text'])
        ax1.set_title('シナリオ別予測価格', color=self.COLORS['text'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenario_names, color=self.COLORS['text'], rotation=45, ha='right')
        ax1.legend(loc='upper right', facecolor=self.COLORS['background'])
        ax1.tick_params(colors=self.COLORS['text'])
        ax1.grid(True, color=self.COLORS['grid'], alpha=0.3)
        
        # 右図：シナリオ確率（ドーナツチャート）
        colors = self.COLORS['scenarios'][:len(probs)]
        wedges, texts, autotexts = ax2.pie(
            probs, 
            labels=scenario_names,
            autopct='%1.0f%%',
            colors=colors,
            wedgeprops=dict(width=0.5)
        )
        
        for text in texts:
            text.set_color(self.COLORS['text'])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax2.set_title('シナリオ確率', color=self.COLORS['text'])
        
        plt.suptitle(f'{ticker} シナリオ分析', fontsize=16, color=self.COLORS['text'], y=1.02)
        plt.tight_layout()
        
        # 保存
        if filename is None:
            filename = f"{ticker}_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, facecolor=self.COLORS['background'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"[SCENARIO] 保存完了: {filepath}")
        return str(filepath)
    
    def create_risk_reward_plot(
        self,
        risk_result: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        リスク/リワードプロットを作成
        
        Args:
            risk_result: RiskFrameworkの結果
            filename: 保存ファイル名
        
        Returns:
            ファイルパス or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        ticker = risk_result.get('ticker', 'UNKNOWN')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(self.COLORS['background'])
        
        for ax in [ax1, ax2]:
            ax.set_facecolor(self.COLORS['background'])
        
        # 左図：予測リスク内訳（レーダーチャート風）
        prediction_risk = risk_result.get('prediction_risk', {})
        if prediction_risk:
            factors = list(prediction_risk.keys())
            values = list(prediction_risk.values())
            
            # 水平棒グラフ
            y_pos = np.arange(len(factors))
            colors = [self.COLORS['confidence_high'] if v < 0.3 else 
                     self.COLORS['confidence_medium'] if v < 0.6 else 
                     self.COLORS['confidence_low'] for v in values]
            
            ax1.barh(y_pos, values, color=colors, alpha=0.8)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([f.replace('_', ' ').title() for f in factors], color=self.COLORS['text'])
            ax1.set_xlim(0, 1)
            ax1.set_xlabel('リスクスコア', color=self.COLORS['text'])
            ax1.set_title('リスク内訳', color=self.COLORS['text'])
            ax1.tick_params(colors=self.COLORS['text'])
            ax1.grid(True, color=self.COLORS['grid'], alpha=0.3, axis='x')
            
            # 値ラベル
            for i, v in enumerate(values):
                ax1.text(v + 0.02, i, f'{v:.2f}', va='center', color=self.COLORS['text'])
        
        # 右図：VaRゲージ
        tail_risk = risk_result.get('tail_risk', {})
        var_95 = tail_risk.get('value_at_risk_95', -5)
        var_99 = tail_risk.get('value_at_risk_99', -10)
        
        risk_labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
        risk_values = [
            abs(var_95),
            abs(var_99),
            abs(tail_risk.get('conditional_var_95', var_95 * 1.5)),
            abs(tail_risk.get('conditional_var_99', var_99 * 1.5))
        ]
        
        y_pos = np.arange(len(risk_labels))
        ax2.barh(y_pos, risk_values, color=self.COLORS['scenarios'][3], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(risk_labels, color=self.COLORS['text'])
        ax2.set_xlabel('損失 (%)', color=self.COLORS['text'])
        ax2.set_title('テールリスク指標', color=self.COLORS['text'])
        ax2.tick_params(colors=self.COLORS['text'])
        ax2.grid(True, color=self.COLORS['grid'], alpha=0.3, axis='x')
        
        for i, v in enumerate(risk_values):
            ax2.text(v + 0.1, i, f'{v:.1f}%', va='center', color=self.COLORS['text'])
        
        # 総合リスクスコア表示
        overall = risk_result.get('overall_risk_score', 0)
        level = risk_result.get('risk_level', 'UNKNOWN')
        
        fig.text(0.5, 0.02, f'総合リスクスコア: {overall:.2f} ({level})', 
                ha='center', fontsize=14, color=self.COLORS['text'])
        
        plt.suptitle(f'{ticker} リスク評価', fontsize=16, color=self.COLORS['text'], y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 保存
        if filename is None:
            filename = f"{ticker}_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, facecolor=self.COLORS['background'])
        plt.close()
        
        logger.info(f"[RISK PLOT] 保存完了: {filepath}")
        return str(filepath)


# ========== テスト用コード ==========
if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Visualization テスト ===\n")
    
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlibがインストールされていません")
        exit(1)
    
    # サンプルデータ取得
    ticker = "NVDA"
    print(f"[TEST] {ticker} のデータを取得中...")
    data = yf.download(ticker, period="6mo", progress=False)
    
    current_price = float(data['Close'].iloc[-1])
    
    # サンプル予測データ
    sample_prediction = {
        "ticker": ticker,
        "current_price": current_price,
        "predictions": {
            "short_term": {
                "forecast_3d": current_price * 1.02,
                "direction": "UP",
                "confidence": 0.44
            },
            "medium_term": {
                "forecast_4w": current_price * 1.05,
                "direction": "UP",
                "confidence": 0.28
            },
            "long_term": {
                "forecast_3m": current_price * 1.10,
                "direction": "UP",
                "confidence": 0.15
            }
        }
    }
    
    # 可視化
    viz = Visualization(output_dir="prediction_charts")
    
    # 予測チャート
    print("\n[TEST] 予測チャート生成中...")
    chart_path = viz.create_prediction_chart(data, sample_prediction)
    print(f"  完了: {chart_path}")
    
    print("\n[RESULT] 全チャート生成完了")
