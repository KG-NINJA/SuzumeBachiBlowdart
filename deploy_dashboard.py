import os
import json
import shutil
from pathlib import Path
from datetime import datetime

DOCS_DIR = "docs"
DATA_DIR = f"{DOCS_DIR}/data"
PREDICTIONS_DIR = "daily_predictions"
BACKTEST_DIR = "backtest_results"

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def copy_prediction_data():
    """Copy latest prediction data to docs/data/"""
    
    print("="*70)
    print("Deploying Dashboard Data to GitHub Pages")
    print("="*70)
    
    files_copied = 0
    
    # Copy prediction files
    prediction_files = [
        "latest_predictions.json",
        "filtered_predictions.json",
        "confidence_report.json"
    ]
    
    for filename in prediction_files:
        src = f"{PREDICTIONS_DIR}/{filename}"
        dst = f"{DATA_DIR}/{filename}"
        
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
                print(f"✓ Copied: {filename}")
                files_copied += 1
            except Exception as e:
                print(f"✗ Error copying {filename}: {str(e)[:40]}")
        else:
            print(f"⚠️  {filename} not found")
    
    # Copy backtest results if available
    backtest_files = [
        "backtest_report.json",
        "backtest_report.md"
    ]
    
    for filename in backtest_files:
        src = f"{BACKTEST_DIR}/{filename}"
        dst = f"{DATA_DIR}/{filename}"
        
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
                print(f"✓ Copied: {filename}")
                files_copied += 1
            except Exception as e:
                print(f"⚠️  Error copying {filename}: {str(e)[:40]}")
    
    print(f"\n✓ Total files copied: {files_copied}")
    return files_copied > 0


def generate_github_pages_index():
    """Generate GitHub Pages index page"""
    
    html = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuzumeBachiBlowdart</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 12px;
            padding: 40px;
            max-width: 600px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
            font-size: 36px;
        }
        p {
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
            line-height: 1.6;
        }
        .button {
            display: inline-block;
            padding: 12px 30px;
            margin: 10px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .button:hover {
            transform: translateY(-2px);
        }
        .primary {
            background: #667eea;
            color: white;
        }
        .secondary {
            background: #e5e7eb;
            color: #333;
        }
        .footer {
            margin-top: 40px;
            color: #999;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 SuzumeBachiBlowdart</h1>
        <p>自動株価予測・取引推奨システム</p>
        
        <p>毎日10銘柄の価格を予測し、高信頼度のシグナルのみを自動抽出します。</p>
        
        <div>
            <a href="dashboard.html" class="button primary">📊 ダッシュボード</a>
            <a href="https://github.com/KG-NINJA/SuzumeBachiBlowdart" class="button secondary">📖 GitHub</a>
        </div>
        
        <div class="footer">
            <p>Phase 1: 信頼度フィルタ | Phase 2: 市場環境分析 | Phase 3: バックテスト</p>
            <p>毎日自動実行 | GitHub Actions統合</p>
        </div>
    </div>
</body>
</html>
"""
    
    index_path = f"{DOCS_DIR}/index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ Generated: {index_path}")


def generate_github_pages_config():
    """Generate _config.yml for GitHub Pages"""
    
    config = """theme: jekyll-theme-minimal
title: SuzumeBachiBlowdart
description: Automated Stock Price Prediction System
url: https://kg-ninja.github.io/SuzumeBachiBlowdart

# GitHub Pages settings
plugins:
  - jekyll-feed
  - jekyll-sitemap

# Build settings
markdown: kramdown
kramdown:
  input: GFM
  auto_ids: true
  hard_wrap: false

# Exclude files
exclude:
  - .gitignore
  - .github
  - models
  - backtest_results
  - logs
  
include:
  - data
  - analysis_report.md
"""
    
    config_path = f"{DOCS_DIR}/_config.yml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config)
    print(f"✓ Generated: {config_path}")


def main():
    print("[1/3] Copying prediction data...")
    copy_prediction_data()
    
    print("\n[2/3] Generating GitHub Pages index...")
    generate_github_pages_index()
    
    print("\n[3/3] Generating GitHub Pages config...")
    generate_github_pages_config()
    
    print("\n" + "="*70)
    print("Dashboard Deployment Complete!")
    print("="*70)
    print("\n📊 Access dashboard at:")
    print("  https://kg-ninja.github.io/SuzumeBachiBlowdart/dashboard.html")
    print("\n📁 Files generated in docs/:")
    print("  - index.html (ホームページ)")
    print("  - dashboard.html (ダッシュボード)")
    print("  - data/ (予測データ)")
    print("  - _config.yml (GitHub Pages設定)")
    print("\n✅ GitHub Pagesを有効にして自動デプロイが開始されます")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
