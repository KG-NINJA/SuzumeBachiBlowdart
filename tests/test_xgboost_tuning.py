from pathlib import Path

import numpy as np
import pandas as pd

from xgboost_tuning import chronological_split, prepare_tuning_dataset


def test_target_is_not_exposed_as_feature():
    features = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 99.0, 102.0],
            "Momentum": [0.1, 0.2, -0.1, 0.3],
        }
    )

    X, y = prepare_tuning_dataset(features)

    assert list(X.columns) == ["Momentum"]
    assert len(X) == 3
    assert y.tolist() == [1, 0, 1]


def test_chronological_split_preserves_order():
    X = pd.DataFrame({"position": np.arange(100)})
    y = pd.Series(np.arange(100) % 2)

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    assert (len(X_train), len(X_val), len(X_test)) == (64, 16, 20)
    assert (len(y_train), len(y_val), len(y_test)) == (64, 16, 20)
    assert X_train.index.max() < X_val.index.min() < X_test.index.min()


def test_tuning_workflow_autostashes_unstaged_changes_before_rebase():
    workflow = Path(".github/workflows/xgboost_tuning.yml").read_text(encoding="utf-8")

    assert "git rebase --autostash origin/main" in workflow
    assert "git rebase origin/main" not in workflow
