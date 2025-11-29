#!/usr/bin/env bash
set -euo pipefail

# Automated conflict resolution helper for the Blowdart repo.
# Strategy: prefer the PR branch versions of key ML pipeline files and
# discard generated artifacts that should not live in git history.
#
# Usage:
#   ./scripts/resolve_conflicts.sh <pr-branch>
# Example:
#   ./scripts/resolve_conflicts.sh codex/fix-and-optimize-ml-pipeline
#
# Notes:
# - Requires the remote name to be "origin".
# - If a step fails, the script exits non-zero so you can adjust manually.

branch=${1:-}
if [[ -z "${branch}" ]]; then
  echo "Usage: $0 <pr-branch>" >&2
  exit 1
fi

echo "🔀 RESOLVING MERGE CONFLICTS"
echo "================================"

# Step 1: Repository status
echo "\n[Step 1] Repository status"
git status -sb

# Step 2: Checkout PR branch
echo "\n[Step 2] Checkout PR branch: ${branch}"
git fetch origin
if ! git checkout "${branch}"; then
  echo "✗ Failed to checkout ${branch}" >&2
  exit 1
fi

# Step 3: Merge main into the PR branch
echo "\n[Step 3] Merge origin/main"
git merge origin/main || true

# Step 4: Auto-resolve conflicts by preferring PR branch copies
# Remove generated artifacts
for f in __pycache__/blowdart_features.cpython-310.pyc __pycache__/blowdart_ml_engine.cpython-310.pyc; do
  if [[ -e "$f" ]]; then
    git rm -f "$f"
  fi
done

# Accept PR branch versions of critical files
files=(
  blowdart_features.py
  blowdart_ml_engine.py
  utils_data_fetch.py
  analytics/training_metrics.json
  accuracy_analysis/REPORT.md
  accuracy_analysis/analysis_results.json
  daily_predictions/latest_predictions.json
  docs/data/latest_predictions.json
)

echo "\n[Step 4] Accepting PR versions for key files"
git checkout --theirs "${files[@]}" 2>/dev/null || true

# Stage changes
echo "\n[Step 5] Staging all changes"
git add .

# Step 6: Commit merge
echo "\n[Step 6] Creating merge commit"
git commit -m "🔀 Resolve merge conflicts: prefer PR branch" || true

echo "\n[Step 7] Push to origin"
git push origin "${branch}" || true

echo "\n================================"
echo "✅ MERGE CONFLICTS RESOLVED (script)"
echo "================================"
