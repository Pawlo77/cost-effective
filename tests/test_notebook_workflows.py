import pandas as pd

from cost_effective.notebook_workflows import oof_confusion_matrix


def test_oof_confusion_matrix_shape() -> None:
    y = pd.Series([0, 0, 1, 1, 0, 1])
    proba = [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]
    table = oof_confusion_matrix(y, proba, k=2)
    assert table.shape == (2, 2)
    assert int(table.loc["Actual 1", "Predicted 1"]) == 2
