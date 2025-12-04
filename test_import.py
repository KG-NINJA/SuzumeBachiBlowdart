#!/usr/bin/env python3
"""
test_import.py - モジュールインポートテスト
"""

import sys
import traceback

def test_imports():
    """全モジュールのインポートをテスト"""
    
    tests = [
        ("MultiTimeScaleForecast", "from prediction_engine.models.multitimescale import MultiTimeScaleForecast"),
        ("ConfidenceScoring", "from prediction_engine.models.confidence_scoring import ConfidenceScoring"),
        ("CrossAssetCorrelation", "from prediction_engine.analysis.cross_asset import CrossAssetCorrelation"),
        ("ScenarioAnalysis", "from prediction_engine.analysis.scenario import ScenarioAnalysis"),
        ("RiskFramework", "from prediction_engine.analysis.risk_framework import RiskFramework"),
        ("OutputFormatter", "from prediction_engine.output.formatters import OutputFormatter"),
        ("Visualization", "from prediction_engine.output.visualization import Visualization"),
    ]
    
    results = []
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            results.append(f"✓ {name}")
            print(f"✓ {name}")
        except Exception as e:
            results.append(f"✗ {name}: {str(e)}")
            print(f"✗ {name}: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("インポートテスト結果:")
    for r in results:
        print(f"  {r}")
    
    success_count = sum(1 for r in results if r.startswith("✓"))
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    
    return success_count == total_count

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
