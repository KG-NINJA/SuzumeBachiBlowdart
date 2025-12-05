# Codex Autofix 運用ポリシー（SuzumeBachiBlowdart）

このプロジェクトでは、OpenAI Codex Autofix 機能を利用し、
CI の自動修正を行います。

## 目的
- CI の落ちを迅速に解消し、開発を止めない
- 文法エラーや依存関係、軽度の技術的エラーを自動修復する
- モデル性能や予測ロジックを損なう危険を排除する

## Allowed（自動修正可能）
- SyntaxError、ImportError、TypeError, KeyError の修正
- requirements.txt の補正
- CI/CD yml の整合性チェック
- utils や support scripts の微修正
- コメント、ドキュメント

## Forbidden（絶対禁止）
- trainer.py、predictor.py の編集
- feature engineering / backtesting
- data/, models/, datasets/ の変更
- モデル精度改善やアルゴリズム変更
- 80行超の修正、大規模改変

## PR 対応ルール
- Codex PR は自動マージしない
- レビュー必須
- 内容が forbidden に触れてないか確認
- CI が緑なら merge

## 緊急停止手順
1. PR を revert
2. `codex-autofix` workflow を停止
3. 手作業修復
4. 安定後に再有効化
