"""
formatters.py - マルチフォーマット出力モジュール

予測結果を様々な形式で出力:
- JSON: プログラム連携用
- CSV: ダッシュボード用
- Markdown: レポート用

Author: SuzumeBachiBlowdart Team
"""

import json
import csv
import io
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger("prediction_engine.formatters")


class OutputFormatter:
    """
    予測結果を様々な形式で出力
    
    対応形式:
    - JSON: 構造化データ、すべての予測詳細を含む
    - CSV: 時系列データ、比較分析用
    - Markdown: 人間可読なレポート形式
    """
    
    def __init__(self, output_dir: str = "prediction_output"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[INIT] OutputFormatter 初期化完了 (output: {output_dir})")
    
    def export(
        self,
        predictions: Dict[str, Any],
        formats: List[str] = ["json", "csv", "markdown"],
        filename_prefix: Optional[str] = None
    ) -> Dict[str, str]:
        """
        予測結果を指定形式でエクスポート
        
        Args:
            predictions: 予測結果辞書
            formats: 出力形式リスト
            filename_prefix: ファイル名プレフィックス
        
        Returns:
            各形式のファイルパス or 内容
        """
        # ファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = predictions.get('ticker', 'UNKNOWN')
        prefix = filename_prefix or f"{ticker}_{timestamp}"
        
        result = {}
        
        if "json" in formats:
            result['json'] = self.export_json(predictions, f"{prefix}.json")
        
        if "csv" in formats:
            result['csv'] = self.export_csv(predictions, f"{prefix}.csv")
        
        if "markdown" in formats:
            result['markdown'] = self.export_markdown(predictions, f"{prefix}.md")
        
        logger.info(f"[EXPORT] エクスポート完了: {list(result.keys())}")
        
        return result
    
    def export_json(
        self, 
        predictions: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        JSON形式でエクスポート
        
        Args:
            predictions: 予測結果
            filename: ファイル名（Noneの場合は文字列を返す）
        
        Returns:
            JSON文字列 or ファイルパス
        """
        # datetimeオブジェクトを文字列に変換
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        json_str = json.dumps(predictions, indent=2, ensure_ascii=False, default=json_serializer)
        
        if filename:
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"[JSON] 保存完了: {filepath}")
            return str(filepath)
        
        return json_str
    
    def export_csv(
        self,
        predictions: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        CSV形式でエクスポート
        
        時系列予測をフラット化してCSVに変換
        
        Args:
            predictions: 予測結果
            filename: ファイル名
        
        Returns:
            CSV文字列 or ファイルパス
        """
        rows = []
        
        # 基本情報
        base_row = {
            'ticker': predictions.get('ticker', 'UNKNOWN'),
            'timestamp': predictions.get('timestamp', ''),
            'current_price': predictions.get('current_price', 0)
        }
        
        # 予測データをフラット化
        if 'predictions' in predictions:
            for term, pred in predictions['predictions'].items():
                row = base_row.copy()
                row['time_horizon'] = term
                
                for key, value in pred.items():
                    if isinstance(value, (int, float, str, bool)):
                        row[f'{term}_{key}'] = value
                
                rows.append(row)
        
        # シナリオデータ
        if 'scenarios' in predictions:
            for scenario_name, scenario_data in predictions['scenarios'].items():
                row = base_row.copy()
                row['data_type'] = 'scenario'
                row['scenario_name'] = scenario_name
                
                for key, value in scenario_data.items():
                    if isinstance(value, (int, float, str, bool)):
                        row[f'scenario_{key}'] = value
                
                rows.append(row)
        
        # CSV生成
        if not rows:
            rows = [base_row]
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        csv_str = output.getvalue()
        
        if filename:
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_str)
            logger.info(f"[CSV] 保存完了: {filepath}")
            return str(filepath)
        
        return csv_str
    
    def export_markdown(
        self,
        predictions: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Markdown形式でエクスポート
        
        人間可読なレポート形式で出力
        
        Args:
            predictions: 予測結果
            filename: ファイル名
        
        Returns:
            Markdown文字列 or ファイルパス
        """
        ticker = predictions.get('ticker', 'UNKNOWN')
        timestamp = predictions.get('timestamp', datetime.now().isoformat())
        current_price = predictions.get('current_price', 0)
        
        md = f"""# 📊 {ticker} 予測レポート

**生成日時**: {timestamp}  
**現在価格**: ${current_price:,.2f}

---

"""
        
        # 予測セクション
        if 'predictions' in predictions:
            md += "## 🎯 マルチタイムスケール予測\n\n"
            md += "| 期間 | 方向 | 予測価格 | 信頼度 |\n"
            md += "|------|------|----------|--------|\n"
            
            term_names = {
                'short_term': '短期 (1-3日)',
                'medium_term': '中期 (1-4週)',
                'long_term': '長期 (1-3ヶ月)'
            }
            
            for term, pred in predictions['predictions'].items():
                name = term_names.get(term, term)
                direction = pred.get('direction', 'N/A')
                direction_emoji = "📈" if direction == "UP" else "📉" if direction == "DOWN" else "➡️"
                
                # 予測価格を探す
                forecast_price = 0
                for key in ['forecast_3d', 'forecast_4w', 'forecast_3m']:
                    if key in pred:
                        forecast_price = pred[key]
                        break
                
                confidence = pred.get('confidence', 0)
                
                md += f"| {name} | {direction_emoji} {direction} | ${forecast_price:,.2f} | {confidence:.1%} |\n"
            
            md += "\n"
        
        # シナリオ分析セクション
        if 'scenarios' in predictions:
            md += "## 🎲 シナリオ分析\n\n"
            
            for scenario_name, scenario_data in predictions['scenarios'].items():
                prob = scenario_data.get('probability', 0)
                name = scenario_data.get('name', scenario_name)
                
                md += f"### {name} (確率: {prob:.0%})\n\n"
                
                if 'forecast_3d' in scenario_data:
                    md += f"- 3日後予測: ${scenario_data['forecast_3d']:,.2f}\n"
                if 'forecast_1m' in scenario_data:
                    md += f"- 1ヶ月後予測: ${scenario_data['forecast_1m']:,.2f}\n"
                if 'trigger' in scenario_data:
                    md += f"- トリガー: {scenario_data['trigger']}\n"
                
                md += "\n"
        
        # 加重平均予測
        if 'weighted_forecast' in predictions:
            wf = predictions['weighted_forecast']
            md += "## 📌 加重平均予測\n\n"
            md += f"- **3日後**: ${wf.get('forecast_3d', 0):,.2f}\n"
            md += f"- **1ヶ月後**: ${wf.get('forecast_1m', 0):,.2f}\n\n"
        
        # リスク評価セクション
        if 'risk_reward' in predictions:
            rr = predictions['risk_reward']
            md += "## ⚠️ リスク/リワード分析\n\n"
            
            if 'best_case' in rr:
                bc = rr['best_case']
                md += f"- **最良ケース**: {bc.get('scenario', 'N/A')} (+{bc.get('return_pct', 0):.2f}%)\n"
            
            if 'worst_case' in rr:
                wc = rr['worst_case']
                md += f"- **最悪ケース**: {wc.get('scenario', 'N/A')} ({wc.get('return_pct', 0):.2f}%)\n"
            
            md += f"- **リスク/リワード比**: {rr.get('risk_reward_ratio', 0):.2f}\n\n"
        
        # 推奨アクション
        if 'recommendation' in predictions:
            rec = predictions['recommendation']
            action = rec.get('action', 'HOLD')
            reasoning = rec.get('reasoning', '')
            
            action_emoji = {
                'STRONG_BUY': '🟢',
                'BUY': '🟢',
                'HOLD': '🟡',
                'REDUCE': '🟠',
                'SELL': '🔴',
                'AVOID': '🔴'
            }.get(action, '⚪')
            
            md += f"## 💡 推奨\n\n"
            md += f"{action_emoji} **{action}**: {reasoning}\n\n"
        
        # 信頼度評価
        if 'confidence' in predictions:
            conf = predictions.get('confidence', {})
            if isinstance(conf, dict):
                md += "## 🔍 信頼度評価\n\n"
                md += f"- 総合スコア: {conf.get('confidence', 0):.1%}\n"
                md += f"- レベル: {conf.get('confidence_level', 'N/A')}\n"
                
                if 'breakdown' in conf:
                    md += "\n**内訳**:\n"
                    for factor, score in conf['breakdown'].items():
                        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                        md += f"- {factor}: {bar} {score:.1%}\n"
        
        md += "\n---\n*このレポートは SuzumeBachiBlowdart 予測エンジンにより自動生成されました*\n"
        
        if filename:
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md)
            logger.info(f"[MARKDOWN] 保存完了: {filepath}")
            return str(filepath)
        
        return md
    
    def generate_summary_table(
        self,
        predictions_list: List[Dict[str, Any]]
    ) -> str:
        """
        複数銘柄の予測サマリーテーブルを生成
        
        Args:
            predictions_list: 予測結果のリスト
        
        Returns:
            Markdownテーブル
        """
        if not predictions_list:
            return "データがありません"
        
        md = "## 📋 予測サマリー\n\n"
        md += "| 銘柄 | 現在価格 | 3日後予測 | 1ヶ月後予測 | 信頼度 | 推奨 |\n"
        md += "|------|----------|-----------|-------------|--------|------|\n"
        
        for pred in predictions_list:
            ticker = pred.get('ticker', 'N/A')
            current = pred.get('current_price', 0)
            
            # 短期予測
            short = pred.get('predictions', {}).get('short_term', {})
            forecast_3d = short.get('forecast_3d', current)
            confidence = short.get('confidence', 0)
            
            # 中期予測
            medium = pred.get('predictions', {}).get('medium_term', {})
            forecast_1m = medium.get('forecast_4w', current)
            
            # 推奨
            rec = pred.get('recommendation', {}).get('action', 'HOLD')
            
            change_3d = (forecast_3d / current - 1) * 100 if current > 0 else 0
            change_1m = (forecast_1m / current - 1) * 100 if current > 0 else 0
            
            md += f"| {ticker} | ${current:,.2f} | ${forecast_3d:,.2f} ({change_3d:+.1f}%) | ${forecast_1m:,.2f} ({change_1m:+.1f}%) | {confidence:.1%} | {rec} |\n"
        
        return md


# ========== テスト用コード ==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== OutputFormatter テスト ===\n")
    
    # サンプル予測データ
    sample_prediction = {
        "ticker": "NVDA",
        "timestamp": datetime.now().isoformat(),
        "current_price": 145.50,
        "predictions": {
            "short_term": {
                "horizon": "1-3 days",
                "forecast_3d": 148.20,
                "direction": "UP",
                "probability": 0.72,
                "confidence": 0.44
            },
            "medium_term": {
                "horizon": "1-4 weeks",
                "forecast_4w": 152.00,
                "direction": "UP",
                "probability": 0.62,
                "confidence": 0.24
            },
            "long_term": {
                "horizon": "1-3 months",
                "forecast_3m": 160.00,
                "direction": "UP",
                "probability": 0.58,
                "confidence": 0.13
            }
        },
        "scenarios": {
            "base_case": {
                "name": "ベースケース",
                "probability": 0.50,
                "forecast_3d": 148.20,
                "forecast_1m": 152.00,
                "trigger": None
            },
            "bull": {
                "name": "ブルシナリオ",
                "probability": 0.25,
                "forecast_3d": 155.00,
                "forecast_1m": 170.00,
                "trigger": "決算サプライズ"
            }
        },
        "weighted_forecast": {
            "forecast_3d": 149.50,
            "forecast_1m": 155.00
        },
        "risk_reward": {
            "best_case": {"scenario": "ブルシナリオ", "return_pct": 16.8},
            "worst_case": {"scenario": "ベアシナリオ", "return_pct": -8.5},
            "risk_reward_ratio": 1.98
        },
        "recommendation": {
            "action": "BUY",
            "reasoning": "期待リターンが高く、リスク/リワード比も良好"
        }
    }
    
    # フォーマッター作成
    formatter = OutputFormatter(output_dir="prediction_output")
    
    # 各形式でエクスポート
    result = formatter.export(sample_prediction)
    
    print("\n[RESULT] エクスポート結果:")
    for fmt, path in result.items():
        print(f"  {fmt}: {path}")
    
    # Markdown出力を表示
    print("\n[MARKDOWN 出力プレビュー]")
    print("-" * 50)
    md_content = formatter.export_markdown(sample_prediction)
    print(md_content[:1000] + "..." if len(md_content) > 1000 else md_content)
