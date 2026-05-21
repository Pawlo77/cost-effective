import pandas as pd

from cost_effective.models import attach_best_hyperparams


def test_attach_best_hyperparams_avoids_dict_alignment_nan() -> None:
    comparison = pd.DataFrame({
        "model_name": ["logistic_regression", "logistic_regression", "lightgbm"],
        "cv_score_mean": [1.0, 2.0, 3.0],
    })
    params = {"logisticregression__C": 0.001, "logisticregression__penalty": "l2"}

    result = attach_best_hyperparams(
        comparison,
        model_name="logistic_regression",
        best_params=params,
    )

    logistic = result.loc[result["model_name"] == "logistic_regression", "best_hyperparams"]
    assert logistic.notna().all()
    assert logistic.str.len().gt(0).all()
    assert '"logisticregression__C": 0.001' in logistic.iloc[0]

    other = result.loc[result["model_name"] == "lightgbm", "best_hyperparams"].iloc[0]
    assert other == ""
