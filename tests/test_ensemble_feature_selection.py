import numpy as np
import pandas as pd

from cost_effective.dataset.ensemble_feature_selection import (
    build_prescreen_ensembles,
    candidate_pools_from_prescreen_scores,
    collect_prescreen_rankings,
)


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(123)
    x = pd.DataFrame(rng.normal(size=(90, 6)), columns=[f"var_{idx}" for idx in range(6)])
    y = pd.Series((x["var_0"] + 0.7 * x["var_2"] > 0).astype(int), name="target")
    return x, y


def test_collect_prescreen_rankings_and_ensembles() -> None:
    x, y = _toy_data()
    methods = ("mutual_info", "abs_corr", "logistic_l1")

    rankings, failures = collect_prescreen_rankings(x, y, methods=methods, random_state=7)
    ensembles = build_prescreen_ensembles(
        rankings,
        methods=methods,
        combination_sizes=(1, 2, 3),
        include_all=True,
    )

    assert failures.empty
    assert set(rankings["method"]) == set(methods)
    assert rankings.groupby("method").size().eq(x.shape[1]).all()
    assert len(ensembles) == 7
    assert "single__mutual_info" in ensembles
    assert "pair__mutual_info+abs_corr" in ensembles
    assert "triple__mutual_info+abs_corr+logistic_l1" in ensembles
    assert ensembles["single__mutual_info"]["order"].is_monotonic_increasing


def test_candidate_pools_follow_best_prescreen_scores() -> None:
    x, y = _toy_data()
    methods = ("mutual_info", "abs_corr")
    rankings, _failures = collect_prescreen_rankings(x, y, methods=methods, random_state=7)
    ensembles = build_prescreen_ensembles(
        rankings,
        methods=methods,
        combination_sizes=(1,),
        include_all=False,
    )
    scores = pd.DataFrame({
        "prescreen_name": ["single__abs_corr", "single__mutual_info"],
        "oof_business_score": [12.0, 8.0],
        "source_feature_count": [3, 2],
    })

    pools = candidate_pools_from_prescreen_scores(
        ensembles,
        scores,
        top_n=4,
        max_pools=1,
        include_names=("single__mutual_info",),
    )

    assert list(pools) == ["single__mutual_info", "single__abs_corr"]
    assert len(pools["single__mutual_info"]) == 4
    assert len(pools["single__abs_corr"]) == 4
